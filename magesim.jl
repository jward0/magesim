include("src/utils/include.jl")

import .Types: WorldState, AgentState, Logger, DummyNode, Config
import .World: create_world, world_step, stop_world
import .LogWriter: log
import .WorldRenderer: create_window, update_window!, close_window
import .AgentHandler: spawn_agents, step_agents!
import .ConfigLoader: load_config

function main(args)

    if length(args) != 1
        throw(ArgumentError("Invalid number of arguments: $(length(args)). Please supply config name as only argument."))
    end

    # headless, world_fpath, obstacle_map, scale_factor, n_agents, agent_starts, speedup, timeout, multithreaded, do_log, custom_config = load_config(args[1])

    cf = load_config(args[1])

    if !cf.headless
        builder = create_window()
        speedup = min(cf.speedup, 10.0)
    else
        speedup = cf.speedup
    end

    world = create_world(cf)
    agents = spawn_agents(world, cf)
    ts = 1/speedup
    actual_speedup = speedup
    gtk_running = true
    if cf.do_log
        logger = Logger()
        log_frequency = 1
    end

    for step in 1:cf.timeout
        t = @elapsed begin

            step_agents!(agents, world, cf.multithreaded)
            world_running, world, _ = world_step(world, agents)
            
            if !cf.headless
                gtk_running = update_window!(world, agents, actual_speedup, builder)
            end

            if cf.do_log && step % log_frequency == 0 
                log(world, logger, step)
                log(agents, logger, step)
            end
        end

        if !cf.headless && (world_running && gtk_running)
            sleep(max(ts-t, 0))
            actual_speedup = 1/max(t, ts)
        elseif !world_running
            break
        end
    end

    stop_world()
    if !cf.headless
        close_window(builder)
    end

end

main(ARGS)