module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, IdlenessLogMessage, PriorityMessage, PosMessage
import ..AgentDynamics: calculate_next_position
import ..Utils: get_neighbours, pos_distance, get_distances

using DataStructures
using Flux
using Graphs, SimpleWeightedGraphs, LinearAlgebra
using LinearAlgebra
using Statistics

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

    enqueue!(agent.outbox, PosMessage(agent, nothing, agent.position))
    enqueue!(agent.outbox, IdlenessLogMessage(agent, nothing, agent.values.idleness_log))
end

"""
    make_decisions!(agent::AgentState)

Read messages and modify agent's action queue based on received messages, world state belief, and 
internal values
"""
function make_decisions!(agent::AgentState)

    message_received = false

    # input[0] is data (shape=n_nodesx2 (distance, idleness))
    # input[1] is normalised weighted world adjacency matrix (shape=n_nodesxn_nodes)

    # Check messages every timestep (necessary to avoid idleness info going stale)
    while !isempty(agent.inbox)
        message = dequeue!(agent.inbox)
        if message isa IdlenessLogMessage
            # Min pool observed idleness with idleness from message
            agent.values.idleness_log = min.(agent.values.idleness_log, message.message)
            message_received = true
        elseif message isa PriorityMessage
            # Update priority log
            agent.values.priority_log[message.source, :] = message.message
            message_received = true
        elseif message isa PosMessage
            agent.values.agent_dists_log[message.source] = pos_distance(message.message, agent.position)
        end
    end

    if message_received
        empty!(agent.action_queue)
    end

    # Currently only calculates at target reached and doesn't recalculate at any point
    # May need to tweak this
    if isempty(agent.action_queue)

        c = mean(agent.world_state_belief.adj[agent.world_state_belief.adj .!= 0])
        adjacency_matrix = agent.world_state_belief.adj / c

        # distances = [agent.world_state_belief.paths.dists[agent.graph_position, node.id] for node in agent.world_state_belief.nodes[1:agent.world_state_belief.n_nodes]]
        distances = get_distances(agent.graph_position, agent.position, agent.world_state_belief)
        # idlenesses = [node.values.idleness for node in agent.world_state_belief.nodes[1:agent.world_state_belief.n_nodes]]
        idlenesses = agent.values.idleness_log

        node_values = hcat(idlenesses/(c*agent.world_state_belief.n_nodes), distances/c)

        model_in = [node_values, adjacency_matrix]

        """
        model_out = custom_regularise(10.0, vec(forward_nn(model_in)))

        # Comm layer
        final_priorities = softmax(custom_regularise(10.0, dropdims(do_psm(agent, model_out, adjacency_matrix), dims=2)), dims=1)
        enqueue!(agent.outbox, PriorityMessage(agent, nothing, final_priorities))
        """

        # model_out = softmax(vec(forward_nn(model_in)), dims=1)
        model_out = vec(forward_nn(model_in))

        # priorities = distance_filter(distances, model_out)
        priorities = model_out

        enqueue!(agent.outbox, PriorityMessage(agent, nothing, priorities))

        # println("))))))))))))))))))))))))))))))))))))))))))))))")
        # println(model_out)
        # println(argmax(model_out))
        final_priorities = do_priority_greedy(agent, priorities)
        # println(final_priorities)
        # println(argmax(final_priorities))

        # Prevents sitting still at node
        if agent.values.last_visited != 0
            final_priorities[agent.values.last_visited] = 0.0
        end

        if agent.graph_position isa Int && agent.graph_position <= agent.world_state_belief.n_nodes
            final_priorities[agent.graph_position] = 0.0
        end

        target = argmax(final_priorities)

        # enqueue!(agent.outbox, PosMessage(agent, nothing, agent.position))
        # enqueue!(agent.outbox, IdlenessLogMessage(agent, nothing, agent.values.idleness_log))
        enqueue!(agent.action_queue, MoveToAction(target))
    end
end

function distance_filter(taps, values)

    taps = 1 .- (taps.-minimum(taps))/maximum(taps)

    return taps .* values
end

function forward_nn(input)

    data = input[1]
    adj = input[2]

    unweighted_adj = copy(adj)
    unweighted_adj[adj .!= 0.] .= 1.
    repeated_edge = reshape(copy(adj), (size(adj)...,1))

    # REMEMBER COLUMN MAJOR ORDERING WHEN DEBUGGING

    # repeated_data has shape (n_nodes, n_nodes, 2)
    repeated_data = repeat(
        reshape(data, (size(data)[1], 1, size(data)[2])), 
        outer=[1, size(data)[1], 1])

    # combined_data has shape (n_nodes, n_nodes, 3)
    combined_data = cat(dims=3, repeated_data, repeated_edge)

    sc = sd_out(sd_1(data))

    nc = sum(unweighted_adj .* transpose(nd_out(nd_1(combined_data))), dims=2)

    output = leakyrelu(c0(sc) + c1(nc), 0.3)

    return output

end

function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] = -0.02560344 * d[1] +  0.47448905 * d[2]
        out[i, 2] =  0.37850753 * d[1] + -0.44805954 * d[2]
        out[i, 3] =  0.72529513 * d[1] + -1.18498965 * d[2]
        out[i, 4] =  0.73166021 * d[1] +  0.71390661 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = -2.15976341 * d[1] + 1.89191872 * d[2] + 1.3302514 * d[3] + 1.58470547 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] = -0.63064267 * d[1] + -0.66394034 * d[2] +  0.70292034 * d[3]
            out[i, j, 2] =  0.57047792 * d[1] + -0.06849411 * d[2] + -0.43628767 * d[3]
            out[i, j, 3] =  0.79983717 * d[1] + -0.77574031 * d[2] +  0.17639368 * d[3]
            out[i, j, 4] =  0.31979277 * d[1] +  0.23128230 * d[2] +  0.74832512 * d[3]
            out[i, j, 5] = -0.01553424 * d[1] +  0.78119776 * d[2] +  0.32078049 * d[3]
            out[i, j, 6] =  0.05096390 * d[1] + -0.74479295 * d[2] + -0.62604261 * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = -0.16932522 * d[1] + -0.08513338 * d[2] + -2.21655511 * d[3] + 0.74814952 * d[4] + -1.32369776 * d[5] + -0.30468585 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return 1.47268558 * input
end

function c1(input)
    return -0.8183988 * input
end

function do_psm(agent, self_priorities, adj)

    # Currently NOT set up to handle non-infinite communication ranges
    priority_mask = [float(agent.id > i) for i in 1:agent.values.n_agents_belief]

    # Agent adjacency not an issue handling one agent at a time
    unweighted_adj = copy(adj)
    unweighted_adj[adj .!= 0.] .= 1.

    # Division by 0 for self index is hidden by min
    # Blanket division by n_agents is only valid for infinite comm range

    normalised_agent_adjacency = priority_mask .* (min.(1 ./ agent.values.agent_dists_log, 10) ./ max(sum(priority_mask), 1))

    self_contribution = self_priorities

    next_contribution = softmax(agent.values.priority_log .* 10, dims=2) .* normalised_agent_adjacency
    next_contribution = sum(next_contribution, dims=1)
    convolved_next = leakyrelu(next_contribution' + unweighted_adj*next_contribution', 0.3)

    # hardcoded k=3

    convolved_next = leakyrelu(convolved_next + unweighted_adj*convolved_next, 0.3)
    convolved_next = leakyrelu(convolved_next + unweighted_adj*convolved_next, 0.3)

    return leakyrelu(self_contribution - convolved_next, 0.3)
end

function do_priority_greedy(agent::AgentState, self_priorities::Array{Float64, 1})

    # Note that this can only work for homogeneous agent policies
    # No guarantee of performance of behaviour otherwise

    flags::Array{Float64, 1} = ones(size(self_priorities))

    for i in 1:size(agent.values.priority_log)[1]
        if i != agent.id
            flags .*= (self_priorities .> agent.values.priority_log[i, :])
        end
    end

    if max(flags...) == 0.0
        return self_priorities
    end

    return self_priorities .* flags
end

function custom_regularise(factor::Float64, data::Array{Float64, 1})
    out = data .- minimum(data) .+ eps(Float64)
    out = out ./ (maximum(out)/factor)
    return out
end

end