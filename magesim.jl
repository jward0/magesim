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
    headless = configs[1].headless

    if !headless
        builder = create_window()
    end

    for cf in configs

        if !headless
            speedup = min(cf.speedup, 10.0)
        else
            speedup = cf.speedup
        end

        for run_n in 1:10 # HOW MANY RUNS
            println("")
            print("Starting run: ")
            println(run_n)
            world = create_world(cf)
            agents = spawn_agents(world, cf)
            ts = 1/speedup
            actual_speedup = speedup
            gtk_running = true
            if cf.do_log
                logger = Logger(cf, run_n)
                log_frequency = 1
            end

            for step in 1:cf.timeout
                t = @elapsed begin
                    # if step % 10 == 0
                    #     println(step)
                    # end
                    if step == cf.timeout*1/4
                        #print(step)
                        #println(cf.timeout*1/4)
                        flag = 1
                    elseif step == cf.timeout*2/4
                        flag = 2
                    elseif step == cf.timeout*3/4
                        flag = 3
                    else
                        flag = 0
                    end
                    # println(flag)
                    # println(step)
                    step_agents!(agents, world, cf.multithreaded)
                    world_running, world, _ = world_step(world, agents, flag)
                    


                    if !headless
                        gtk_running = update_window!(world, agents, actual_speedup, builder)
                    end

                    if cf.do_log && step % log_frequency == 0 
                        for agent in agents
                            log(agent, logger, step)
                        end
                    end
                end

                if !headless && (world_running && gtk_running)
                    sleep(max(ts-t, 0))
                    actual_speedup = 1/max(t, ts)
                elseif !world_running
                    break
                end
            end

            stop_world()
        end
    end

    if !headless
        close_window(builder)
    end

end

main(ARGS)