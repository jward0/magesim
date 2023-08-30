module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction
import ..AgentDynamics: calculate_next_position

using DataStructures

"""
    create_agent(world::WorldState, start_node::Int64)

+++Temporary function until I get agent handler written+++
"""
function create_agent(world::WorldState, start_node::Int64)
    
    # TODO: move id handling to agent handler which creates agents from a spawn_agent function
    # Checking that a valid start pos is handed to the state constructor is also a handler job
    id = 1
    start_position = world.nodes[start_node].position
    values = nothing

    agent = AgentState(id, start_node, start_position, values)
    enqueue!(agent.action_queue, MoveToAction(5))

    return agent
end

"""
    agent_step!(agent::AgentState, world::WorldState)

Select and perform an action and update agent position and stat accordingly
"""
function agent_step!(agent::AgentState, world::WorldState)

    # Modify action queue
    if isempty(agent.action_queue)
        enqueue!(agent.action_queue, MoveToAction(rand(1:5)))
    end

    # Do action from queue

    action = first(agent.action_queue)

    if action isa WaitAction
        # Do nothing for one timestep
        new_pos = agent.position
        new_graph_pos = agent.graph_position
        done = true
    elseif action isa MoveToAction
        # Move towards target and do not pop action from queue until target reached
        new_pos, new_graph_pos, done = calculate_next_position(agent, action.target, world)
    elseif action isa StepTowardsAction
        # Take one step towards target 
        new_pos, new_graph_pos, _ = calculate_next_position(agent, action.target, world)
        done = true
    else
        nothing
    end

    agent.position = new_pos
    agent.graph_position = new_graph_pos

    if done
        dequeue!(agent.action_queue)
    end

end

end