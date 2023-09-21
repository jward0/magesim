import numpy as np

from gymnasium.spaces import Sequence, Tuple, Discrete, MultiDiscrete, MultiBinary, Box, Text, Graph, Dict
from pettingzoo.utils.env import ParallelEnv

from julia.api import Julia
Julia(runtime='/home/james/julia/julia-1.9.2/bin/julia', compiled_modules=False)
from julia import Main


class MagesimParallelEnv(ParallelEnv):

    def __init__(self, config_name='default'):

        print("Preparing Julia instance...")

        # super(MagesimParallelEnv, self).__init__()
        jl = Julia()
        # Modified include as wrapper does not currently support rendering via Julia
        # And some extra Julia utils are needed to interface with PettingZoo

        jl.eval('''
            include("src/utils/pz_include.jl")
            import .Types: WorldState, AgentState, Logger, DummyNode
            import .ConfigLoader: load_config
            import .LogWriter: log
            import .World: create_world, world_step
            import .AgentHandler: spawn_agents, step_agents!
            import .SpaceHandler: generate_action_space, unwrap_node_values, unwrap_world
        ''')

        Main.config_name = config_name

        print("Loading config...")

        _, world_fpath, obstacle_map, n_agents, agent_starts, _, _, _ = Main.eval('load_config([config_name])')

        Main.world_fpath = world_fpath
        Main.n_agents = n_agents
        Main.agent_starts = agent_starts

        print("Creating world...")

        Main.world = Main.eval('create_world(world_fpath)')
        Main.agents = Main.eval('spawn_agents(n_agents, agent_starts, world)')

        self.world = Main.world

        print("Generating spaces...")

        self.n_nodes = Main.eval('world.n_nodes')

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

        node_contents, node_contents_labels = Main.eval('unwrap_node_values()')

        # Take the tuples of default values and labels from unwrap_node_values() and 
        # use them to construct appropriate spaces that can represent custom NodeValues
        # composite type in observations

        individual_spaces = []

        for content in node_contents:

            if isinstance(content, bool):
                individual_spaces.append(MultiBinary(1))
            elif isinstance(content, int):
                individual_spaces.append(Discrete(content))
            elif isinstance(content, float):
                individual_spaces.append(Box(low=-np.inf, high=np.inf, shape=(1,)))
            elif isinstance(content, str):
                individual_spaces.append(Text(int(content)))
            elif isinstance(content, np.ndarray) or isinstance(content, list):
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

    def step(self, actions):
        Main.world_running, Main.world = Main.eval('world_step(world, agents)')
        Main.eval('step_agents!(agents, world, false)')

        observations = {}

        for agent in self.agents:
            Main.i = agent
            nodes, edges, node_values = Main.eval('unwrap_world(agents[i].world_state_belief)')
            agent_position = Main.eval('[agents[i].position.x, agents[i].position.y]')

            map = {"nodes": nodes, "edge_links": edges}

            observations[agent] = dict(zip(["agent_position", "map", "node_values"], 
                                           [agent_position, map, node_values]))
        reward = {}
        terminated = {}
        truncated = {}
        info = {}

        return observations, reward, terminated, truncated, info

    def reset():
        pass

    def render():
        pass

    def close():
        pass

    def state():
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
