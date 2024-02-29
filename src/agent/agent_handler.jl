module AgentHandler

import ..Types: AgentState, WorldState, Position, StepTowardsAction
import ..Agent: agent_step!, make_decisions!, observe_world!
import ..MessagePasser: pass_messages!

using DataStructures

"""
    spawn_agents(agent_count::Int64, start_nodes::Array{Int64, 1}, world::WorldState)

Create agents at specified nodes in the world and return them in an array
"""
function spawn_agents(custom_config::Array{Float64, 1}, agent_count::Int64, start_nodes::Array{Int64, 1}, world::WorldState)

    agents = Array{AgentState, 1}(undef, agent_count)

    for i = 1:agent_count
        agents[i] = AgentState(i, start_nodes[i], world.nodes[start_nodes[i]].position, agent_count, world.n_nodes, custom_config)
        observe_world!(agents[i], world)
    end

    return agents
end

"""
    step_agents!(agents::Array{AgentState, 1}, world::WorldState)

Modify action queues, step once through movement dynamics, generate observations, and pass 
messages between agents   
"""
function step_agents!(agents::Array{AgentState, 1}, 
                      world::WorldState, 
                      multithreaded::Bool=true, 
                      force_actions::Union{Bool, Array{Int64, 1}}=false)

    # Note that WorldState MUST be immutable for this to guarantee thread-safety for custom user code
    # These are intentionally in sequential loops (multithreaded performance would be improved by having
    # a single loop instead) as users may wish to insert message-passing steps between steps, and the
    # seperate loops give an easy way to achieve synchronicity

    if multithreaded

        Threads.@threads for agent in agents
            if force_actions != false
                empty!(agent.action_queue)
                enqueue!(agent.action_queue, StepTowardsAction(force_actions[agent.id]))
            else
                make_decisions!(agent)
            end
        end
    
        # This step has to happen sequentially for collision avoidance purposes (but is also quite fast so w/e)
        # I suppose technically it could be broken out into 2 parts to keep it multithreaded, but would need
        # synchronisation halfway through so might not be worth it for the time saved
        for agent in agents
            agent_step!(agent, world, [agent.position for agent in agents[1:agent.id-1]])
        end
    
        Threads.@threads for agent in agents
            observe_world!(agent, world)
        end

    else

        for agent in agents
            if force_actions != false
                empty!(agent.action_queue)
                enqueue!(agent.action_queue, StepTowardsAction(force_actions[agent.id-1]))
            else
                make_decisions!(agent)
            end
        end
    
        for agent in agents
            agent_step!(agent, world, [agent.position for agent in agents[1:agent.id]])
        end
    
        for agent in agents
            observe_world!(agent, world)
        end   
    end

    pass_messages!(agents, world)

end

"""
    step_agents_(agents::Array{AgentState, 1}, world::WorldState)

Identical to above but with Python compatible name
"""
function step_agents_(agents::Array{AgentState, 1}, 
                      world::WorldState, 
                      multithreaded::Bool=true, 
                      force_actions::Union{Bool, Array{Int64, 1}}=false)

    # Note that WorldState MUST be immutable for this to guarantee thread-safety for custom user code
    # These are intentionally in sequential loops (multithreaded performance would be improved by having
    # a single loop instead) as users may wish to insert message-passing steps between steps, and the
    # seperate loops give an easy way to achieve synchronicity

    if multithreaded

        Threads.@threads for agent in agents
            if force_actions != false
                empty!(agent.action_queue)
                enqueue!(agent.action_queue, StepTowardsAction(force_actions[agent.id]))
            else
                make_decisions!(agent)
            end
        end
    
        Threads.@threads for agent in agents
            agent_step!(agent, world)
        end
    
        Threads.@threads for agent in agents
            observe_world!(agent, world)
        end

    else

        for agent in agents
            if force_actions != false
                empty!(agent.action_queue)
                enqueue!(agent.action_queue, StepTowardsAction(force_actions[agent.id]))
            else
                make_decisions!(agent)
            end
        end
    
        for agent in agents
            agent_step!(agent, world)
        end
    
        for agent in agents
            observe_world!(agent, world)
        end   
    end

    pass_messages!(agents, world)

end

end