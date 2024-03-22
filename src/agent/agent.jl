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
    agent.values.idleness_log = [i + 1.0 for i in agent.values.idleness_log]
    if agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes
        agent.values.idleness_log[agent.graph_position] = 0.0
        agent.values.last_visited = agent.graph_position
        enqueue!(agent.outbox, ArrivedAtNodeMessage(agent, nothing, agent.graph_position))
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
        elseif message isa PosMessage
            agent.values.agent_dists_log[message.source] = pos_distance(message.message, agent.position)
        elseif message isa GoingToMessage
            agent.values.other_targets[message.source] = message.message
        end
    end

    if message_received || agent.graph_position isa Int64
        empty!(agent.action_queue)
    end

    if isempty(agent.action_queue)

        c = mean(agent.world_state_belief.adj[agent.world_state_belief.adj .!= 0])
        adjacency_matrix = agent.world_state_belief.adj / c

        distances = get_distances(agent.graph_position, agent.position, agent.world_state_belief)
        idlenesses = agent.values.idleness_log

        node_values = hcat(idlenesses/maximum(idlenesses), distances/maximum(distances))

        model_in = [node_values, adjacency_matrix]

        model_out = vec(forward_nn(model_in))

        priorities = model_out
        final_priorities = do_priority_greedy(agent, priorities)
        # final_priorities = priorities

        # Prevents sitting still at node
        if agent.values.last_visited != 0
            final_priorities[agent.values.last_visited] -= 10000
        end

        if agent.graph_position isa Int && agent.graph_position <= agent.world_state_belief.n_nodes
            final_priorities[agent.graph_position] -= 10000
        end

        target = argmax(final_priorities)
        # target = do_sebs_style(agent, final_priorities)
        
        enqueue!(agent.action_queue, MoveToAction(target))
        # enqueue!(agent.outbox, GoingToMessage(agent, nothing, target))

        if agent.graph_position isa Int64
            enqueue!(agent.outbox, PriorityMessage(agent, nothing, final_priorities))
        end

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

# SA baseline
#=
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
=#
#=
# Candidate 5
function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] =  1.98241903 * d[1] + -1.26181915 * d[2]
        out[i, 2] = -1.16416914 * d[1] + -0.03950104 * d[2]
        out[i, 3] = -1.53572437 * d[1] + -0.80523494 * d[2]
        out[i, 4] = -0.67152170 * d[1] + -0.80642394 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = -2.12636974 * d[1] + 1.16137201 * d[2] + 1.91836535 * d[3] + 0.29847511 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] =  0.93456114 * d[1] +  0.32234672 * d[2] + -0.14471714 * d[3]
            out[i, j, 2] =  0.41134737 * d[1] +  0.22697582 * d[2] + -0.07067039 * d[3]
            out[i, j, 3] =  0.13262621 * d[1] + -0.11402553 * d[2] +  0.00722151 * d[3]
            out[i, j, 4] =  0.11607106 * d[1] + -0.01143001 * d[2] + -0.29395003 * d[3]
            out[i, j, 5] =  0.02858441 * d[1] + -1.02008190 * d[2] + -0.31749845 * d[3]
            out[i, j, 6] =  1.83966085 * d[1] + -0.16705342 * d[2] +  1.29978674 * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = -1.48834262 * d[1] + -0.64857298 * d[2] + 0.3531105 * d[3] +  0.10315508 * d[4] + -2.46000117 * d[5] + -1.12933257 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return -1.38606064 * input
end

function c1(input)
    return 0.51913897 * input
end
=#
#=
# Candidate 0
function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] =  0.21145167 * d[1] + -0.27925428 * d[2]
        out[i, 2] = -1.44247192 * d[1] + -1.33494336 * d[2]
        out[i, 3] =  0.57172370 * d[1] + -0.85223489 * d[2]
        out[i, 4] =  0.40209655 * d[1] + -0.50196633 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = -1.55681413 * d[1] + 1.27792288 * d[2] + -0.51313773 * d[3] + -2.19374748 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] =  0.02569938 * d[1] + -0.29096973 * d[2] + -0.21568785 * d[3]
            out[i, j, 2] =  0.41319706 * d[1] +  0.75811129 * d[2] +  0.03202006 * d[3]
            out[i, j, 3] = -0.77203510 * d[1] +  0.47926454 * d[2] + -0.07546965 * d[3]
            out[i, j, 4] = -0.00010602 * d[1] + -0.36923795 * d[2] + -0.56227554 * d[3]
            out[i, j, 5] = -0.44041089 * d[1] +  0.84039208 * d[2] +  0.28826833 * d[3]
            out[i, j, 6] =  0.21577007 * d[1] +  0.58216880 * d[2] + -0.85432579 * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = -1.96664497 * d[1] + -1.85750399 * d[2] + -1.12159551 * d[3] + -1.70362948 * d[4] + 1.17730152 * d[5] + 1.97568741 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return -1.64993142 * input
end

function c1(input)
    return -0.08408668 * input
end
=#
#=
# Candidate a
function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] =  0.38340402 * d[1] +  0.48314985 * d[2]
        out[i, 2] =  0.04715937 * d[1] +  0.46571011 * d[2]
        out[i, 3] =  0.03571544 * d[1] +  0.18513540 * d[2]
        out[i, 4] =  1.16349590 * d[1] + -0.87012585 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = -2.5669888 * d[1] + 1.0625178 * d[2] +  1.34443389 * d[3] + -2.29184781 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] = -0.10481829 * d[1] +  0.30997706 * d[2] + -0.28924909 * d[3]
            out[i, j, 2] = -0.15479256 * d[1] + -0.99204339 * d[2] + -0.66706460 * d[3]
            out[i, j, 3] = -0.82617384 * d[1] + -0.71983851 * d[2] + -0.35385266 * d[3]
            out[i, j, 4] =  0.03228171 * d[1] + -0.00199264 * d[2] +  0.57788682 * d[3]
            out[i, j, 5] = -0.31177772 * d[1] + -0.72719435 * d[2] + -0.51248340 * d[3]
            out[i, j, 6] = -0.48989367 * d[1] +  0.07865114 * d[2] + -0.27926827 * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = -1.38059483 * d[1] + 0.8793949 * d[2] + -0.90870255 * d[3] + -0.94483813 * d[4] + -1.98277356 * d[5] + 0.98023151 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return -1.41337745 * input
end

function c1(input)
    return -1.10066678 * input
end
=#
#=
# Candidate c
function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] = -0.58406186 * d[1] + -1.13754903 * d[2]
        out[i, 2] = -0.37576691 * d[1] + -0.12235257 * d[2]
        out[i, 3] = -1.73644509 * d[1] +  0.13809729 * d[2]
        out[i, 4] = -1.00878043 * d[1] + -0.60134646 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = 1.55832648 * d[1] + -0.95695622 * d[2] +  -1.70631365 * d[3] + -1.97175723 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] = -0.88329016 * d[1] +  0.18759748 * d[2] +  0.34942367 * d[3]
            out[i, j, 2] = -0.62774281 * d[1] +  0.23151634 * d[2] + -0.14543803 * d[3]
            out[i, j, 3] =  0.31766794 * d[1] +  0.19683674 * d[2] + -0.01112670 * d[3]
            out[i, j, 4] =  0.90822474 * d[1] +  0.00652581 * d[2] + -0.50223844 * d[3]
            out[i, j, 5] =  0.46804047 * d[1] +  0.29701673 * d[2] +  0.52062282 * d[3]
            out[i, j, 6] =  0.59898807 * d[1] + -0.45112344 * d[2] +  0.15419817 * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = 0.30416439 * d[1] + -0.16476198 * d[2] + -2.29214487 * d[3] + -2.27658257 * d[4] + 0.53050808 * d[5] + 0.46814707 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return 2.00546969 * input
end

function c1(input)
    return 0.29135257 * input
end
=#
#=
# Candidate e
function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] =  1.55286694 * d[1] + -0.44110086 * d[2]
        out[i, 2] = -0.67343548 * d[1] +  0.17652840 * d[2]
        out[i, 3] = -1.67024631 * d[1] +  1.16802774 * d[2]
        out[i, 4] = -0.40900121 * d[1] + -0.20207343 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = -3.09391917 * d[1] + 1.86547219 * d[2] +  -0.70012984 * d[3] + 1.98574134 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] =  0.29622339 * d[1] + -1.13492925 * d[2] +  0.31488647 * d[3]
            out[i, j, 2] = -0.42093213 * d[1] +  0.23324459 * d[2] + -0.75787474 * d[3]
            out[i, j, 3] = -0.70319289 * d[1] + -0.83720823 * d[2] + -0.51324397 * d[3]
            out[i, j, 4] =  0.27840781 * d[1] +  0.39243370 * d[2] + -0.51859291 * d[3]
            out[i, j, 5] =  0.22624910 * d[1] + -0.42383448 * d[2] + -1.00603054 * d[3]
            out[i, j, 6] =  1.06420315 * d[1] +  0.19624948 * d[2] +  1.09041112 * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = -0.75642422 * d[1] + 0.7263272 * d[2] + 1.45487278 * d[3] + -1.63886386 * d[4] + 1.08887729 * d[5] + 0.9865759 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return -2.38748449 * input
end

function c1(input)
    return 0.54197564 * input
end
=#
# Candidate f
function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] = -0.66290329 * d[1] + -1.31846606 * d[2]
        out[i, 2] =  0.65034865 * d[1] + -1.48956611 * d[2]
        out[i, 3] =  0.28390010 * d[1] + -0.48346823 * d[2]
        out[i, 4] = -1.90242116 * d[1] + -0.83423582 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = 1.61741099 * d[1] + -2.49534479 * d[2] + 2.38210474 * d[3] + 0.19223465 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] = -0.96888649 * d[1] + -0.57236659 * d[2] + -0.57240128 * d[3]
            out[i, j, 2] = -0.59767775 * d[1] + -0.10536716 * d[2] + -0.55625470 * d[3]
            out[i, j, 3] = -0.77850319 * d[1] +  0.52268095 * d[2] +  0.15994723 * d[3]
            out[i, j, 4] = -0.11633870 * d[1] +  0.11544369 * d[2] +  0.37772713 * d[3]
            out[i, j, 5] =  0.17265815 * d[1] +  1.19896348 * d[2] + -0.30094113 * d[3]
            out[i, j, 6] = -0.12250413 * d[1] + -0.72660719 * d[2] + -0.81008904 * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = 2.30023962 * d[1] + -0.04810535 * d[2] + -1.53099819 * d[3] + 1.86170104 * d[4] + 0.38627405 * d[5] + -0.31040836 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return -3.21830443 * input
end

function c1(input)
    return 0.46392383 * input
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
            flags .-= (self_priorities .< agent.values.priority_log[i, :]) * 9999
        end
    end

    if max(flags...) == -9999
        return self_priorities
    end

    return self_priorities .+ flags
end

function do_sebs_style(agent::AgentState, self_priorities::Array{Float64, 1})

    new_prio = copy(self_priorities)

    for ndx in [a for a in agent.values.other_targets if a > 0]
        new_prio[ndx] -= 9999
    end

    ns = get_neighbours(agent.graph_position, agent.world_state_belief, true)

    modified_prio = zeros(size(new_prio))

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