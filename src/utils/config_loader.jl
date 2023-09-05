module ConfigLoader

using JSON

"""
    load_config(args::Vector{String})

Pulls config information from json file in configs directory, specified as command line argument
"""
function load_config(args::Vector{String})

    config = JSON.parsefile(string("configs/", string(args[1]), ".json"))

    headless::Bool = config["headless"]
    world_fpath::String = string("maps/", config["world_file"])
    n_agents::Int64 = config["n_agents"]
    agent_starts::Array{Int64, 1} = convert(Array{Int64, 1}, config["agent_starts"])
    speedup::Float64 = convert(Float64, config["speedup"])
    timeout::Int64 = config["timeout"]

    return headless, world_fpath, n_agents, agent_starts, speedup, timeout
end

end