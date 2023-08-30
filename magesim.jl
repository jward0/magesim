include("src/utils/include.jl")

import .Types: WorldState, AgentState
import .World: create_world, world_step, stop_world
import .Agent: create_agent, agent_step!
import .WorldRenderer: create_window, update_window!, close_window

# TODO agent handler/parallelisation, agent logging, world logging, headless mode, message passer, Python wrapper
# Then commit as v1.0, write some docs, and start looking at what's needed to make eg. patrolling work
function main()

    headless = false
    if !headless
        builder = create_window()
    end

    world = create_world("maps/test.json")
    agents = [create_agent(world, 1)]
    speedup = 2.0
    ts = 1/speedup
    actual_speedup = speedup
    gtk_running = true

    for _ in 1:10000
        t = @elapsed begin
            world_running, world = world_step(world, agents)
            for agent in agents
                agent_step!(agent, world)
            end
            if !headless
                gtk_running = update_window!(world, agents, actual_speedup, builder)
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

main()