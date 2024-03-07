module ConfigLoader

import ..Types: Config, UserConfig

using JSON, Images, IterTools

"""
    load_config(args::Vector{String})

Pulls config information from json file in configs directory, specified as command line argument
"""
function load_configs(arg::String)

    config_dict = JSON.parsefile(string("configs/", string(arg), ".json"))

    config = process_config_dict(config_dict)

    return [config]
end

function load_configs(conf_arg::String, sweep_arg::String)

    conf_dict = JSON.parsefile(string("configs/", string(conf_arg), ".json"))
    sweep_config = JSON.parsefile(string("configs/", string(sweep_arg), ".json"))

    configs = Vector{Config}()

    ks = [k for k in keys(sweep_config)]

    for prod in product([sweep_config[k] for k in ks]...)
        for (k, v) in zip(ks, [p for p in prod])
            conf_dict[k] = v
            if k == "n_agents"
                if v == 1
                    # conf_dict["agent_starts"] = [31]
                    conf_dict["agent_starts"] = [17]
                elseif v == 2
                    # conf_dict["agent_starts"] = [9, 50]
                    conf_dict["agent_starts"] = [11, 22]
                elseif v == 4
                    # conf_dict["agent_starts"] = [9, 22, 38, 59]
                    conf_dict["agent_starts"] = [5, 16, 18, 22]
                elseif v == 6
                    # conf_dict["agent_starts"] = [7, 12, 25, 38, 46, 55]
                    conf_dict["agent_starts"] = [5, 11, 16, 22, 24, 28]
                elseif v == 8
                    # conf_dict["agent_starts"] = [7, 9, 16, 28, 34, 41, 50, 55]
                    conf_dict["agent_starts"] = [5, 11, 9, 16, 18, 22, 24, 28]
                elseif v == 12
                    # conf_dict["agent_starts"] = [7, 6, 9, 16, 24, 28, 34, 37, 41, 50, 54, 55]
                    conf_dict["agent_starts"] = [1, 5, 8, 11, 12, 14, 15, 17, 20, 22, 24, 26]
                else
                    conf_dict["agent_starts"] = [range(1, v)...]
                end
            end
        end
        push!(configs, process_config_dict(conf_dict))
    end

    return configs
end

function process_config_dict(config_dict::Dict{String, Any})

    world_info = JSON.parsefile(string("maps/", string(config_dict["world"]), ".info"))

    headless::Bool = config_dict["headless"]
    world_fpath::String = string("maps/", world_info["graph"])
    scale_factor::Float64 = 1.0
    if "image" in keys(world_info)
        obstacle_map = load(string("maps/", world_info["image"]))[end:-1:1, 1:1:end]
        scale_factor = world_info["scale_factor"]
    else
        obstacle_map = nothing
    end
    n_agents::Int64 = config_dict["n_agents"]
    agent_starts::Array{Int64, 1} = convert(Array{Int64, 1}, config_dict["agent_starts"])
    speedup::Float64 = convert(Float64, config_dict["speedup"])
    timeout::Int64 = config_dict["timeout"]
    multithreaded::Bool = config_dict["multithreaded"]
    do_log::Bool = config_dict["do_log"]
    comm_range::Float64 = config_dict["comm_range"]
    check_los::Bool = config_dict["check_los"]

    # Custom config loading
    custom_config::UserConfig = UserConfig(config_dict["custom_config"])

    config = Config(
        world_fpath, 
        obstacle_map,
        scale_factor,
        n_agents,
        agent_starts,
        comm_range,
        check_los,
        headless,
        speedup,
        timeout,
        multithreaded,
        do_log,
        custom_config
    )

    return config
end

end