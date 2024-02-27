module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, StringMessage, ArrivedAtNodeMessage
import ..AgentDynamics: calculate_next_position
import ..Utils: get_neighbours

using DataStructures
using Graphs, SimpleWeightedGraphs, LinearAlgebra

"""
    agent_step!(agent::AgentState, world::WorldState)

Select and perform an action and update agent position and stat accordingly
"""
function agent_step!(agent::AgentState, world::WorldState, blocked_pos::Array{Position, 1})

    # Wait if no other action found
    if isempty(agent.action_queue)
        enqueue!(agent.action_queue, WaitAction())
    end

    # Do action from queue

    action = first(agent.action_queue)

    if action isa WaitAction
        # Do nothing for one timestep
        new_pos = agent.position
        new_graph_pos = agent.graph_position
        action_done = true
    elseif action isa MoveToAction
        # Move towards target and do not pop action from queue until target reached
        new_pos, new_graph_pos, action_done = calculate_next_position(agent, action.target, world, blocked_pos)
    elseif action isa StepTowardsAction
        # Take one step towards target 
        new_pos, new_graph_pos, _ = calculate_next_position(agent, action.target, world, blocked_pos)
        action_done = true
    else
        error("Error: no behaviour found for action of type $(nameof(typeof))")
    end

    agent.position = new_pos
    agent.graph_position = new_graph_pos

    if action_done
        dequeue!(agent.action_queue)
    end

end

"""
    observe_world!(agent::AgentState, world::WorldState)

Extract an agent's observation from the true world state and update the agent's belief of the
world state, and generate messages to send to other agents
"""
function observe_world!(agent::AgentState, world::WorldState)
    agent.world_state_belief = world
    agent.values.idleness_log = [i + 1.0 for i in agent.values.idleness_log]
    if agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes
        agent.values.idleness_log[agent.graph_position] = 0.0
    end
end

"""
    make_decisions!(agent::AgentState)

Read messages and modify agent's action queue based on received messages, world state belief, and 
internal values
"""
function make_decisions!(agent::AgentState)

    # Read ArrivedAtNodeMessages to update idleness and intention logs
    while !isempty(agent.inbox)
        message = dequeue!(agent.inbox)
        if message isa ArrivedAtNodeMessage
            agent.values.idleness_log[message.message[1]] = 0.0
            agent.values.intention_log[message.source] = message.message[2]
        end
    end

    # If no action in progress, select node to move towards
    if isempty(agent.action_queue)

        neighbours = get_neighbours(agent.graph_position, agent.world_state_belief, true)

        if length(neighbours) == 1
            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
        elseif !isa(agent.graph_position, Int64)
            # Catch the potential problem of an agent needing a new action
            # while midway between two nodes (not covered by algo) - 
            # solution to this is just to pick one
            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
        else
            # Do SEBS
            gains = map(n -> calculate_gain(n, agent), neighbours)
            posteriors = map(g -> calculate_posterior(g, agent), gains)
            n_intentions::Array{Int64, 1} = zeros(agent.world_state_belief.n_nodes)
            for i in agent.values.intention_log
                if i != 0 n_intentions[i] += 1 end
            end
            intention_weights = map(n -> calculate_intention_weight(n, agent), n_intentions)
            final_posteriors = [posteriors[i] * intention_weights[neighbours[i]] for i in 1:length(posteriors)]
            target = neighbours[argmax(final_posteriors)]
            enqueue!(agent.action_queue, MoveToAction(target))
            enqueue!(agent.outbox, ArrivedAtNodeMessage(agent, nothing, (agent.graph_position, target)))
        end
    end
    
end

function calculate_gain(node::Int64, agent::AgentState)
    distance = agent.world_state_belief.paths.dists[agent.graph_position, node]
    return agent.values.idleness_log[node] / distance
end

function calculate_posterior(gain::Float64, agent::AgentState)
    g1 = agent.values.sebs_gains[1]
    g2 = agent.values.sebs_gains[2]

    if gain >= g2
        return 1.0
    else
        return g1 * 2.7183^((gain/g2) * log(1/g1))
    end
end

function calculate_intention_weight(n_intentions::Int64, agent::AgentState)

    n_agents = agent.values.n_agents_belief
    return 2^(n_agents - n_intentions)/(2^n_agents - 1)
end

end