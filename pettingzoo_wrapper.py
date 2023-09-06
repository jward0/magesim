from julia.api import Julia
from julia import Main
from pettingzoo.utils.env import ParallelEnv


class MagesimParallelEnv(ParallelEnv):

    def __init__(self, world_fpath):
        super(MagesimParallelEnv, self).__init__(world_fpath)
        Julia(runtime='/home/james/julia/julia-1.9.2/bin/julia', compiled_modules=False)
        jl = Julia()
        jl.eval('include("src/utils/include.jl")')
        jl.eval('import .World: create_world')
        Main.world_fpath = world_fpath
        world = Main.eval('create_world(world_fpath)')
        agents = []
        num_agents = []
        possible_agents = []
        max_num_agents = []
        observation_spaces = []
        action_spaces = []

    def step():
        pass

    def reset():
        pass

    def render():
        pass

    def close():
        pass

    def state():
        pass

    def observation_space(agent):
        pass

    def action_space(agent):
        pass
