module World

import ..Types: WorldState, AgentState, Node, DummyNode, AbstractNode
import ..Utils: pos_distance
using Graphs, SimpleWeightedGraphs
using JSON

"""
    create_world(fpath::String)

Load world info from JSON file, construct node and map representations, and return world state
"""
function create_world(fpath::String, obstacle_map::Union{Nothing, Array{}})
    nodes_dict = JSON.parsefile(fpath)
    nodes = Array{AbstractNode, 1}(undef, length(nodes_dict))

    n_nodes::Int = 0

    for (strid, node) in nodes_dict
        id = parse(Int, strid)
        if id < 0
            nodes[length(nodes_dict) + id + 1] = DummyNode(string(length(nodes_dict) + id + 1), node)
        else
            n_nodes +=1
            nodes[id] = Node(strid, node)
        end
    end
    
    sources = Vector{Int64}()
    destinations = Vector{Int64}()
    weights = Vector{Float64}()

    for node in nodes
        for n in node.neighbours
            if n < 0
                neighbour = nodes[length(nodes_dict) + n + 1]
            else
                neighbour = nodes[n]
            end
            push!(sources, node.id)
            push!(destinations, neighbour.id)
            push!(weights, pos_distance(node.position, neighbour.position))
        end
    end

    graph_map = SimpleWeightedDiGraph(sources, destinations, weights)

    world_state = WorldState(nodes, n_nodes, graph_map, obstacle_map)
    return world_state
end

"""
    world_step(world_state::WorldState, agents::Array{AgentState, 1})

Return updated world state and reward allocated to agents
"""
function world_step(world_state::WorldState, agents::Array{AgentState, 1})
    updated_world_state = WorldState(world_state.nodes, world_state.n_nodes, world_state.map, world_state.obstacle_map, world_state.paths, world_state.time + 1, world_state.done)
    
    rewards = zeros(Float64, length(agents))

    return true, updated_world_state, rewards
end

"""
    stop_world()

Safely stop the simulation and close the GUI
"""
function stop_world()
    nothing
end

end