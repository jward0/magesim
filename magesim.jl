include("src/utils/include.jl")

import .Types: WorldState, AgentState, Logger
import .World: create_world, world_step, stop_world
import .LogWriter: log, new_logger
import .WorldRenderer: create_window, update_window!, close_window
import .AgentHandler: spawn_agents, step_agents!
import .ConfigLoader: load_config

# TODO Python wrapper, run from config, LOS checker with image layer
# Then commit as v1.0, write some docs, and start looking at what's needed to make eg. patrolling work
function main(args)

    headless, world_fpath, n_agents, agent_starts, speedup, timeout = load_config(args)

    if !headless
        builder = create_window()
    end

    world = create_world(world_fpath)
    agents = spawn_agents(n_agents, agent_starts, world)
    ts = 1/speedup
    actual_speedup = speedup
    gtk_running = true
    logger = new_logger()

    for step in 1:timeout
        t = @elapsed begin
            world_running, world = world_step(world, agents)
            step_agents!(agents, world)
            if !headless
                gtk_running = update_window!(world, agents, actual_speedup, builder)
            end
            log(world, logger, step)
            for node in world.nodes
                log(node, logger, step)
            end
            for agent in agents
                log(agent, logger, step)
            end
        end

        if world_running && gtk_running
            sleep(max(ts-t, 0))
            actual_speedup = 1/max(t, ts)
        else
            break
        end
    end

    stop_world()
    if !headless
        close_window(builder)
    end

end

main(ARGS)