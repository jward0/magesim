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
function agent_step!(agent::AgentState, world::WorldState)

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
        new_pos, new_graph_pos, action_done = calculate_next_position(agent, action.target, world)
    elseif action isa StepTowardsAction
        # Take one step towards tfalsearget 
        new_pos, new_graph_pos, _ = calculate_next_position(agent, action.target, world)
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

    distances = [agent.world_state_belief.paths.dists[agent.graph_position, node.id] for node in agent.world_state_belief.nodes[1:agent.world_state_belief.n_nodes]]
    idlenesses = [node.values.idleness for node in agent.world_state_belief.nodes[1:agent.world_state_belief.n_nodes]]
    node_values = hcat(distances, idlenesses)

    if agent.world_state_belief.map isa SimpleWeightedDiGraph
        a = agent.world_state_belief.map.weights
        c = mean(a[a .!= 0])
        adjacency_matrix = a / c
    else
        adjacency_matrix = adjacency_matrix(agent.world_state_belief.map)
    end

    # TODO: implement a way to extract adjacency matrix of the map without running into
    # snags with dummy nodes (current problem)
    println(adjacency_matrix)

    model_in = [node_values, adjacency_matrix]

    model_out = forward_nn(model_in)
    println(model_out)
    target = argmax(output)
    enqueue!(agent.action_queue, StepTowardsAction(target))

end

function forward_nn(input)

    data = input[1]
    adj = input[2]
    unweighted_adj = copy(adj)
    unweighted_adj[adj .!= 0] .= 1
    repeated_edge = reshape(Matrix{Float64}(I, size(adj)), (size(adj)...,1))

    # repeated_data has shape (n_nodes, n_nodes, 2)
    repeated_data = repeat(
        reshape(data, (size(data)[1], 1, size(data)[2])), 
        outer=[1, size(data)[1], 1])

    # combined_data has shape (n_nodes, n_nodes, 3)
    combined_data = cat(dims=3, repeated_data, repeated_edge)
    println("+++++++++++++++++++++++++++++")
    println(data)
    println(adj)
    println(combined_data)

    sc = Matrix(sd_out(sd_1(repeated_data)))
    nc = Matrix(unweighted_adj) * Matrix(nd_out(nd_1(combined_data)))

    output = leakyrelu(c0(sc) + c1(nc), 0.3)

    println(output)

    return output

end

function sd_1(input)
    out = [[[-0.02560344 * d[1] +  0.47448905 * d[2], 
              0.37850753 * d[1] + -0.44805954 * d[2], 
              0.72529513 * d[1] + -1.18498965 * d[2], 
              0.73166021 * d[1] +  0.71390661 * d[2]] 
            for d in r] 
           for r in input]
    
    return leakyrelu(out, 0.3)
end

function sd_out(input)
    out = [[-2.15976341 * d[1] + 1.89191872 * d[2] + 1.3302514 * d[3] + 1.58470547 * d[4] for d in r] for r in input]
    return leakyrelu(out, 0.3)
end

function nd_1(input)
    out = [[[-0.63064267 * d[1] + -0.66394034 * d[2] +  0.70292034 * d[3], 
              0.57047792 * d[1] + -0.06849411 * d[2] + -0.43628767 * d[3], 
              0.79983717 * d[1] + -0.77574031 * d[2] +  0.17639368 * d[3], 
              0.31979277 * d[1] +  0.23128230 * d[2] +  0.74832512 * d[3],
             -0.01553424 * d[1] +  0.78119776 * d[2] +  0.32078049 * d[3],
              0.05096390 * d[1] + -0.74479295 * d[2] + -0.62604261 * d[3]] 
            for d in r] 
           for r in input]
    return leakyrelu(out, 0.3)
end

function nd_out(input)
    out = [[-0.16932522 * d[1] + -0.08513338 * d[2] + -2.21655511 * d[3] + 0.74814952 * d[4] + -1.32369776 * d[5] + -0.30468585 * d[6] for d in r] for r in input]
    return leakyrelu(out, 0.3)
end

function c0(input)
    return 1.47268558 * input
end

function c1(input)
    return -0.8183988 * input
end

end