import numpy as np

from gymnasium.spaces import Sequence, Tuple, Discrete, MultiDiscrete, MultiBinary, Box, Text, Graph, Dict
from pettingzoo.utils.env import ParallelEnv

import juliacall
from juliacall import Main as jl

class MagesimParallelEnv(ParallelEnv):

    def __init__(self, config_name='default'):

        print("Preparing Julia instance...")

        # super(MagesimParallelEnv, self).__init__()
        # jl = Julia()

        # Modified include as wrapper does not currently support rendering via Julia
        # And some extra Julia utils are needed to interface with PettingZoo

        jl.seval('include("src/utils/pz_include.jl")')
        jl.seval('import .Types: WorldState, AgentState, Logger, DummyNode')
        jl.seval('import .ConfigLoader: load_config')
        jl.seval('import .LogWriter: log')
        jl.seval('import .World: create_world, world_step')
        jl.seval('import .AgentHandler: spawn_agents, step_agents_')
        jl.seval('import .InterfaceUtils: generate_action_space, unwrap_node_values, unwrap_world, parse_from_py')

        print("Loading config...")

        _, world_fpath, obstacle_map, n_agents, agent_starts, _, _, _ = jl.load_config(config_name)

        print("Creating world...")

        world = jl.create_world(world_fpath)
        jl.agents = jl.spawn_agents(n_agents, agent_starts, world)

        self.world = world

        print("Generating spaces...")

        self.n_nodes = world.n_nodes

        self.agents = [i+1 for i in range(n_agents)]
        # Arbitrarily chosen 16 as max n_agents
        self.possible_agents = [i+1 for i in range(16)]
        self.observation_spaces = []

        # Only StepTowardsAction is supported
        self.action_spaces = dict(enumerate([Discrete(self.n_nodes) for _ in range(n_agents)]))

        # Assemble observation (world belief) space

        agent_position_space = Box(low=-np.inf, high=np.inf, shape=(1, 2))
        map_space = Graph(node_space=Box(low=-np.inf, high=np.inf, shape=(2,)), edge_space=None)
        # other_agents_position_space = 

        node_contents, node_contents_labels = jl.unwrap_node_values()

        # Take the tuples of default values and labels from unwrap_node_values() and 
        # use them to construct appropriate spaces that can represent custom NodeValues
        # composite type in observations

        individual_spaces = []

        for content in node_contents:

            if isinstance(content, juliacall.ArrayValue):
                content = np.asarray(content)

            if isinstance(content, bool):
                individual_spaces.append(MultiBinary(1))
            elif isinstance(content, int):
                individual_spaces.append(Discrete(content))
            elif isinstance(content, float):
                individual_spaces.append(Box(low=-np.inf, high=np.inf, shape=(1,)))
            elif isinstance(content, str):
                individual_spaces.append(Text(int(content)))
            elif isinstance(content, np.ndarray):
                if isinstance(content[0], np.integer):
                    individual_spaces.append(MultiDiscrete(np.array(content)))
                elif isinstance(content[0], np.floating):
                    individual_spaces.append(Box(low=-np.inf, high=np.inf, shape=np.shape(content)))
                elif isinstance(content[0], np.bool_):
                    individual_spaces.append(MultiBinary(shape=np.shape(content)))
                elif isinstance(content[0], str):
                    individual_spaces.append(Tuple([Text(int(c)) for c in content]))
                else:
                    raise TypeError("Invalid type in NodeValues: may only be int, float, bool, str, or 1-d array thereof")
            else:
                raise TypeError("Invalid type in NodeValues: may only be int, float, bool, str, or 1-d array thereof")

        node_space = Dict(dict(zip(node_contents_labels, individual_spaces)))
        all_node_spaces = Sequence(node_space)

        individual_observation = Dict(dict(zip(["agent_position", "map", "node_values"], 
                                               [agent_position_space, map_space, all_node_spaces])))

        self.observation_spaces = dict(enumerate([individual_observation for _ in range(n_agents)]))

        print("Environment ready!")

    def get_observations(self):

        observations = {}

        for i in self.agents:
            nodes, edges, node_values = jl.unwrap_world(jl.agents[i-1].world_state_belief)
            agent_position = [jl.agents[i-1].position.x, jl.agents[i-1].position.y]
            map = {"nodes": nodes, "edge_links": edges}

            observations[i] = dict(zip(["agent_position", "map", "node_values"], 
                                       [agent_position, map, node_values]))
        return observations
    
    def step(self, actions):
        jl.step_agents_(jl.agents, self.world, False, jl.parse_from_py(actions))
        world_running, self.world, rewards_arr = jl.world_step(self.world, jl.agents)
        observations = self.get_observations()
        rewards = dict(enumerate(rewards_arr))
        terminated = dict(enumerate([False for _ in range(self.num_agents)]))
        truncated = dict(enumerate([False for _ in range(self.num_agents)]))
        info = dict(enumerate([None for _ in range(self.num_agents)]))

        return observations, rewards, terminated, truncated, info

    def reset(self, seed=0):

        jl.world = jl.seval('create_world(world_fpath)')
        jl.agents = jl.seval('spawn_agents(n_agents, agent_starts, world)')

        self.world = jl.world

        return self.get_observations()

    def render():
        pass

    def close():
        pass

    def state():
        return jl.seval('unwrap_world(world)')

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
