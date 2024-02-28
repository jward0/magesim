module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, StringMessage, ArrivedAtNodeMessage
import ..AgentDynamics: calculate_next_position
import ..Utils: get_neighbours

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
end

"""
    make_decisions!(agent::AgentState)

Read messages and modify agent's action queue based on received messages, world state belief, and 
internal values
"""
function make_decisions!(agent::AgentState)

    # Currently single-agent only: no message handling

    # input[0] is data (shape=n_nodesx2 (distance, idleness))
    # input[1] is normalised weighted world adjacency matrix (shape=n_nodesxn_nodes)

    if isempty(agent.action_queue) || agent.graph_position isa Int == first(agent.action_queue)

        c = mean(agent.world_state_belief.adj[agent.world_state_belief.adj .!= 0])
        adjacency_matrix = agent.world_state_belief.adj / c

        distances = [agent.world_state_belief.paths.dists[agent.graph_position, node.id] for node in agent.world_state_belief.nodes[1:agent.world_state_belief.n_nodes]]
        idlenesses = [node.values.idleness for node in agent.world_state_belief.nodes[1:agent.world_state_belief.n_nodes]]
        node_values = hcat(idlenesses/(c*agent.world_state_belief.n_nodes), distances/c)

        model_in = [node_values, adjacency_matrix]

        model_out = vec(forward_nn(model_in))
        target = argmax(model_out)

        # Prevents sitting still at node
        if target == agent.graph_position
            model_out[target] = 0.0
            target = argmax(model_out)
        end

        enqueue!(agent.action_queue, MoveToAction(target))
    end
end

function forward_nn(input)

    data = input[1]
    adj = input[2]

    """
    data = [[0.56559504, 3.49180782],
    [0.5652032 , 2.97984821],
    [0.56532588, 2.3140171 ],
    [0.5653524 , 2.40376639],
    [0.56537381, 3.49788887],
    [0.04541397, 1.42510632],
    [0.5652027 , 2.25137478],
    [0.56560128, 2.80796181],
    [0.        , 0.        ],
    [0.56536157, 2.30310981],
    [0.56560602, 1.1178051 ],
    [0.01513799, 0.43735239],
    [0.56564141, 0.77945993],
    [0.56528495, 1.26236185],
    [0.56565227, 1.73240754],
    [0.56542418, 3.09993423],
    [0.56539575, 2.4633817 ],
    [0.56555229, 1.6226571 ],
    [0.56533367, 2.44766656],
    [0.56533547, 6.05058681],
    [0.56527313, 4.63644034],
    [0.56550354, 3.861566  ],
    [0.56542233, 3.07136286],
    [0.56530527, 4.93712028],
    [0.56556039, 4.58591888],
    [0.56525165, 4.8932201 ],
    [0.56537269, 4.73047564],
    [0.56530177, 5.31027176],
    [0.56555699, 4.52006862]]

    adj = [[0.,0.51195961,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.51195961,0.,0.,0.,0.51804066,1.5547419,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.88891078,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.14140454,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.51804066,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,1.5547419,0.88891078,0.,0.,0.,0.82626846,0.,0.,0.,0.,0.98775393,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.82626846,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.50485201,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.43735239,0.77945993,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.50485201,0.,0.,1.18530471,0.,0.,0.,0.,1.09750436,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.18530471,0.,0.6804527,0.,0.,0.61460244,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.98775393,0.,0.,0.43735239,0.,0.6804527,0.,0.,0.,0.,0.,0.,1.18530471,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.77945993,0.,0.,0.,0.,0.48290192,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,1.14140454,0.,0.,0.,0.,0.,0.,0.,0.,0.48290192,0.,0.,0.,0.,0.,1.18530471,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.61460244,0.,0.,0.,0.,0.,0.73097416,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.09750436,0.,0.,0.,0.,0.,0.,0.63655253,0.,0.,0.,1.53650611,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.73097416,0.63655253,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.18530471,0.,0.,0.,0.,0.,0.,1.66820663,0.,0.,0.,1.44870576,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.18530471,0.,0.,0.,1.66820663,0.,0.,0.,0.,0.,0.,0.,0.,2.28280907,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.32011117,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.53650611,0.,0.,0.,0.,0.,1.07555428,0.,0.,0.,0.,0.,1.80431854,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.07555428,0.,0.79020314,0.,0.72435288,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.44870576,0.,0.,0.,0.79020314,0.,0.,0.,0.,0.,0.,1.44870576],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.3512014,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.72435288,0.,0.3512014,0.,0.30730122,0.,0.72435288,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.30730122,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,2.28280907,1.32011117,0.,0.,0.,0.,0.,0.,0.,0.,1.77782157],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.80431854,0.,0.,0.,0.72435288,0.,0.,0.,0.79020314],
    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.44870576,0.,0.,0.,1.77782157,0.79020314,0.]]
    
    data = transpose(hcat(data...))
    adj = hcat(adj...)
    """
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

end