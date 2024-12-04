module AgentHandler

import ..Types: AgentState, WorldState, Position, StepTowardsAction, Config
import ..Agent: agent_step!, make_decisions!, observe_world!
import ..MessagePasser: pass_messages!

using DataStructures

"""
    spawn_agents(agent_count::Int64, start_nodes::Array{Int64, 1}, world::WorldState)

Create agents at specified nodes in the world and return them in an array
"""
function spawn_agents(world::WorldState, config::Config)

    agent_count = config.n_agents
    start_nodes = config.agent_starts
    
    agents = Array{AgentState, 1}(undef, agent_count)

    for i = 1:agent_count
        agents[i] = AgentState(
            i, 
            start_nodes[i], 
            world.nodes[start_nodes[i]].position, 
            agent_count, 
            world.n_nodes, 
            config.comm_range,
            config.check_los, 
            config.custom_config)

        agents[i].world_state_belief = world
        agents[i].values.effective_adj = world.adj
        agents[i].values.original_adj_belief = world.adj
        agents[i].values.last_visited = start_nodes[i]
        agents[i].values.original_dr = 3.0  # copy(maximum(world.adj) / minimum(world.adj[findall(!iszero, world.adj)]))
        agents[i].values.comm_failure = config.comm_failure
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

        pass_messages!(agents, world)

        Threads.@threads for agent in agents
            observe_world!(agent, world)

            if force_actions != false
                empty!(agent.action_queue)
                enqueue!(agent.action_queue, StepTowardsAction(force_actions[agent.id]))
            else
                make_decisions!(agent)
            end

            agent_step!(agent, world, [agent.position for agent in agents[1:agent.id-1]])
        end
    
    else

        pass_messages!(agents, world)

        for agent in agents
            observe_world!(agent, world)
        end   

        for agent in agents
            if force_actions != false
                empty!(agent.action_queue)
                enqueue!(agent.action_queue, StepTowardsAction(force_actions[agent.id]))
            else
                make_decisions!(agent)
            end
        end
    
        for agent in agents
            agent_step!(agent, world, [agent.position for agent in agents[1:agent.id-1]])
        end
    
    end

    

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
                enqueue!(agent.action_queue, StepTowardsAction(force_actions[agent.id]))
            else
                make_decisions!(agent)
            end
        end
    
        for agent in agents
            agent_step!(agent, world, [agent.position for agent in agents[1:agent.id-1]])
        end
    
        for agent in agents
            observe_world!(agent, world)
        end   
    end

    pass_messages!(agents, world)


end

end