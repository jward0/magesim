module AgentHandler

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction
import ..Agent: agent_step!, create_agent

function spawn_agents(agent_count::Int64, start_nodes::Array{Int64, 1}, world::WorldState)

    agents = Array{AgentState, 1}(undef, agent_count)

    for i = 1:agent_count
        agents[i] = create_agent(i, world.nodes[i].position, start_nodes[i])
    end

    return agents
end


function step_agents!(agents::Array{AgentState, 1}, world::WorldState)

    # Note that WorldState MUST be immutable for this to guarantee thread-safety for custom user code

    Threads.@threads for agent in agents
        agent_step!(agent, world)
    end
end


end