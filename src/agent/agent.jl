module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, IdlenessLogMessage, PriorityMessage, PosMessage, ArrivedAtNodeMessage, GoingToMessage
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
    agent.values.idleness_log .+= 1.0

    # Upon arrival at a node:
    if isempty(agent.action_queue) && agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes
        agent.values.idleness_log[agent.graph_position] = 0.0
        agent.values.last_last_visited = copy(agent.values.last_visited)
        agent.values.last_visited = agent.graph_position
    end
    agent.values.other_targets_freshness .+= 1.0
    for (ndx, f) in enumerate(agent.values.other_targets_freshness)
        if f > 60
            agent.values.other_targets[ndx] = 0
            agent.values.priority_log[ndx, :] .*= 0
        end
    end
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
        agent.values.n_messages += 1
        message_received = true
        if message isa IdlenessLogMessage
            # Min pool observed idleness with idleness from message
            agent.values.idleness_log = min.(agent.values.idleness_log, message.message)
        elseif message isa ArrivedAtNodeMessage
            agent.values.idleness_log[message.message] = 0.0
        elseif message isa PriorityMessage
            agent.values.priority_log[message.source, :] = message.message
            agent.values.other_targets_freshness[message.source] = 0.0
        elseif message isa PosMessage
            agent.values.agent_dists_log[message.source] = pos_distance(message.message, agent.position)
        elseif message isa GoingToMessage
            agent.values.other_targets[message.source] = message.message
            agent.values.other_targets_freshness[message.source] = 0.0
        end
    end

    # if agent.graph_position isa Int64
    #     empty!(agent.action_queue)
    # end

    if isempty(agent.action_queue)

        c = mean(agent.world_state_belief.adj[agent.world_state_belief.adj .!= 0])
        adjacency_matrix = agent.world_state_belief.adj / c

        distances = get_distances(agent.graph_position, agent.position, agent.world_state_belief)
        idlenesses = agent.values.idleness_log

        node_values = hcat(idlenesses/maximum(idlenesses), distances/maximum(distances))

        model_in = [node_values, adjacency_matrix]

        model_out = vec(forward_nn(model_in))

        priorities = model_out
        final_priorities = priorities

        # Prevents sitting still at node
        if agent.values.last_visited != 0
            final_priorities[agent.values.last_visited] -= 10000
        end

        if agent.graph_position isa Int && agent.graph_position <= agent.world_state_belief.n_nodes
            final_priorities[agent.graph_position] -= 10000
        end

        if agent.values.last_last_visited != 0
            final_priorities[agent.values.last_last_visited] -= 100
        end

        if maximum(agent.values.other_targets) == 0
            long_range_target = argmax(final_priorities)
            neighbours = get_neighbours(agent.graph_position, agent.world_state_belief, true)
            distances_via_neighbours = map(n -> agent.world_state_belief.paths.dists[n, long_range_target] + pos_distance(agent.position, agent.world_state_belief.nodes[n].position), neighbours)
            target = neighbours[argmin(distances_via_neighbours)]
        else
            target = do_sebs_style(agent, final_priorities)
        end


        enqueue!(agent.action_queue, MoveToAction(target))

        enqueue!(agent.outbox, IdlenessLogMessage(agent, nothing, agent.values.idleness_log))
        enqueue!(agent.outbox, GoingToMessage(agent, nothing, target))

        # if agent.id == 1
        #     println(target)
        # end
        # enqueue!(agent.outbox, IdlenessLogMessage(agent, nothing, agent.values.idleness_log))

        agent.values.current_target = target

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

# Candidate g
function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] =  0.39542921 * d[1] + -0.65675830 * d[2]
        out[i, 2] =  1.04956550 * d[1] + -2.41169670 * d[2]
        out[i, 3] = -1.76400600 * d[1] + -1.35503092 * d[2]
        out[i, 4] =  0.35702324 * d[1] + -0.21459921 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = 1.23378153 * d[1] + -1.74570142 * d[2] + 1.67926374 * d[3] + 1.03526079 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] =  0.01888764 * d[1] + -0.45915342 * d[2] + -0.47627992 * d[3]
            out[i, j, 2] =  0.37894088 * d[1] +  0.84592537 * d[2] +  0.03730955 * d[3]
            out[i, j, 3] = -0.47623729 * d[1] + -0.48666733 * d[2] +  0.58645635 * d[3]
            out[i, j, 4] = -0.26577757 * d[1] + -0.49801225 * d[2] + -0.00811519 * d[3]
            out[i, j, 5] =  0.57938169 * d[1] +  1.04516341 * d[2] + -0.19110703 * d[3]
            out[i, j, 6] = -0.13873973 * d[1] + -0.50178453 * d[2] + -0.94114514 * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = -1.12949772 * d[1] + 1.03677392 * d[2] + -1.21275977 * d[3] + -0.05812893 * d[4] + -1.04156908 * d[5] + 0.77585069 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return -0.62272069 * input
end

function c1(input)
    return -0.04584406 * input
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

    flags::Array{Float64, 1} = zeros(size(self_priorities))

    for i in 1:size(agent.values.priority_log)[1]
        if i != agent.id
            flags .-= (self_priorities .< agent.values.priority_log[i, :]) * 999
        end
    end

    if max(flags...) == -999
        return self_priorities
    end

    return self_priorities .+ flags
end

function do_sebs_style(agent::AgentState, self_priorities::Array{Float64, 1})

    new_prio = copy(self_priorities)

    for ndx in [a for a in agent.values.other_targets if a > 0]
        new_prio[ndx] -= 999
    end

    ns = get_neighbours(agent.graph_position, agent.world_state_belief, true)

    modified_prio = ones(size(new_prio)) * -99999.0

    for i = 1:size(new_prio)[1]
        if i in ns
            modified_prio[i] = new_prio[i]
        end
    end

    target = argmax(modified_prio)

    return target
end

function custom_regularise(factor::Float64, data::Array{Float64, 1})
    out = data .- minimum(data) .+ eps(Float64)
    out = out ./ (maximum(out)/factor)
    return out
end

end