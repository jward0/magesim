module ConfigLoader

using JSON, Images

"""
    load_config(args::Vector{String})

Pulls config information from json file in configs directory, specified as command line argument
"""
function load_config(arg::String)

    config = JSON.parsefile(string("configs/", string(arg), ".json"))

    world_info = JSON.parsefile(string("maps/", string(config["world"]), ".info"))

    headless::Bool = config["headless"]
    world_fpath::String = string("maps/", world_info["graph"])
    scale_factor::Float64 = 1.0
    if "image" in keys(world_info)
        obstacle_map = load(string("maps/", world_info["image"]))[end:-1:1, 1:1:end]
        scale_factor = world_info["scale_factor"]
    else
        obstacle_map = nothing
    end
    n_agents::Int64 = config["n_agents"]
    agent_starts::Array{Int64, 1} = convert(Array{Int64, 1}, config["agent_starts"])
    speedup::Float64 = convert(Float64, config["speedup"])
    timeout::Int64 = config["timeout"]
    multithreaded::Bool = config["multithreaded"]

    # Custom config loading
    custom_config::Bool = config["custom_config"]

    return headless, world_fpath, obstacle_map, scale_factor, n_agents, agent_starts, speedup, timeout, multithreaded, custom_config
end

end