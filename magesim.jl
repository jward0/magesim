include("src/utils/include.jl")

import .Types: WorldState, AgentState, Logger, DummyNode, Config
import .World: create_world, world_step, stop_world
import .LogWriter: log
import .WorldRenderer: create_window, update_window!, close_window
import .AgentHandler: spawn_agents, step_agents!
import .ConfigLoader: load_configs

function main(args)

    if length(args) in [1, 2]
        configs = load_configs(args...)
    else
        throw(ArgumentError("Invalid number of arguments: $(length(args)). Please supply config name and optionally sweep config name as only arguments."))
    end

    # Due to constraints, headless-ness cannot vary across parameter sweep
    # println(configs[1])
    headless = configs[1].headless

    if !headless
        builder = create_window()
    end

    # headless, world_fpath, obstacle_map, scale_factor, n_agents, agent_starts, speedup, timeout, multithreaded, do_log, custom_config = load_config(args[1])

    for cf in configs

        if !headless
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
            logger = Logger(cf)
            log_frequency = 1
        end

        full_t = @elapsed begin

            for step in 1:cf.timeout
                t = @elapsed begin

                    step_agents!(agents, world, cf.multithreaded)
                    world_running, world, _ = world_step(world, agents)
                    
                    if !headless
                        gtk_running = update_window!(world, agents, actual_speedup, builder)
                    end

                    if cf.do_log && step % log_frequency == 0 
                        log(world, logger, step)
                        log(agents, logger, step)
                    end
                end

                if !headless && (world_running && gtk_running)
                    sleep(max(ts-t, 0))
                    actual_speedup = 1/max(t, ts)
                elseif !world_running
                    break
                end
            end

        end

        sleep(max(1.1 - full_t, 0))
        println(full_t)

        stop_world()
    end

    if !headless
        close_window(builder)
    end

end

main(ARGS)