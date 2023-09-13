module ConfigLoader

using JSON, Images

"""
    load_config(args::Vector{String})

Pulls config information from json file in configs directory, specified as command line argument
"""
function load_config(args::Vector{String})

    if length(args) != 1
        throw(ArgumentError("Invalid number of arguments: $(length(args)). Please supply config name as only argument."))
    end

    config = JSON.parsefile(string("configs/", string(args[1]), ".json"))

    headless::Bool = config["headless"]
    world_fpath::String = string("maps/", config["world_file"])
    if "obstacle_map" in keys(config)
        obstacle_map = load(string("maps/", config["obstacle_map"]))[end:-1:1, 1:1:end]
    else
        obstacle_map = nothing
    end
    n_agents::Int64 = config["n_agents"]
    agent_starts::Array{Int64, 1} = convert(Array{Int64, 1}, config["agent_starts"])
    speedup::Float64 = convert(Float64, config["speedup"])
    timeout::Int64 = config["timeout"]
    multithreaded::Bool = config["multithreaded"]

    return headless, world_fpath, obstacle_map, n_agents, agent_starts, speedup, timeout, multithreaded
end

end