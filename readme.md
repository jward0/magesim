## MAGESim: Multi-Robot Patrolling

This branch is used for testing of the multi-robot patrolling problem, with implementations of several literature algorithms and some of my own unpublished work.

Also included is the ability to test dynamic environments by loading a "temporal profile" that alters the speed at which an agent can traverse an edge in real time.

# Instructions

The simulator requires Julia 1.8.1. To run the simulator, use the command "julia magesim.jl $CONFIGNAME $CONFIGSWEEPNAME" (eg. "julia magesim.jl default_dynamic comm_range_sweep") where the config sweep parameter can be omitted if no parameter sweeps are desired. Example config files can be found in the "configs" directory, and any new configs should be placed there. 

To load environment dynamics via config or a config sweep, the "custom_config" field in the config file should be set as follows:
For no dynamics:
{
  "source": "none",
  "dyn_mode: "nothing"
}
To load dynamic profiles with suffix e.g. "moving_blockages" but not have the agents react to them:
{
  "source": "load",
  "name": "moving_blockages",
  "dyn_mode": "none"
}
"dyn_mode" governs how the agent responds to the environment dynamics. Options are "none" (no response), "simple" (agents update their beliefs according to last observed weight), "active" (simple plus decay), and "perfect" (agents have perfect knowledge).

Dynamic profiles should be placed in directory "temporal_profiles" and named as $MAPNAME_$PROFILENAME.jld, using the .jld format. The structure of a profile is Dict{String, Any} with a single key ("data") associated with a t-element Vector{Matrix{Float64}} for profile of time length t. Each n\*n matrix (for n nodes in the map) consists of the scale factors to apply to agent speeds between map nodes.

Results are saved in a "logs" directory. Each scenario run will be logged in a timestamped subdirectory, and will consist of the following files:
"agent_positions.csv": (x, y) positions of all agents at all timesteps
"config.txt": The config values used
"idleness.csv": True idlenesses of all nodes at all timesteps
