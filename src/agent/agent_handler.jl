module AgentHandler

import ..Types: AgentState, WorldState, Position
import ..Agent: agent_step!, make_decisions!, observe_world!
import ..MessagePasser: pass_messages!

"""
    spawn_agents(agent_count::Int64, start_nodes::Array{Int64, 1}, world::WorldState)

Create agents at specified nodes in the world and return them in an array
"""
function spawn_agents(agent_count::Int64, start_nodes::Array{Int64, 1}, world::WorldState)

    agents = Array{AgentState, 1}(undef, agent_count)

    for i = 1:agent_count
        agents[i] = AgentState(i, start_nodes[i], world.nodes[i].position, nothing)
    end

    return agents
end

"""
    step_agents!(agents::Array{AgentState, 1}, world::WorldState)

Modify action queues, step once through movement dynamics, generate observations, and pass 
messages between agents   
"""
function step_agents!(agents::Array{AgentState, 1}, world::WorldState, multithreaded::Bool = true)

    # Note that WorldState MUST be immutable for this to guarantee thread-safety for custom user code
    # These are intentionally in sequential loops (multithreaded performance would be improved by having
    # a single loop instead) as users may wish to insert message-passing steps between steps, and the
    # seperate loops give an easy way to achieve synchronicity

    if multithreaded
        Threads.@threads for agent in agents
            make_decisions!(agent)
        end
    
        Threads.@threads for agent in agents
            agent_step!(agent, world)
        end
    
        Threads.@threads for agent in agents
            observe_world!(agent, world)
        end
    else
        for agent in agents
            make_decisions!(agent)
        end
    
        for agent in agents
            agent_step!(agent, world)
        end
    
        for agent in agents
            observe_world!(agent, world)
        end   
    end

    pass_messages!(agents)

end

end