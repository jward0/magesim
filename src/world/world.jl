module World

import ..Types: WorldState, AgentState, Node, DummyNode, AbstractNode
import ..Utils: pos_distance, get_real_adj
using Graphs, SimpleWeightedGraphs
using JSON

"""
    create_world(fpath::String)

Load world info from JSON file, construct node and map representations, and return world state
"""
function create_world(fpath::String, obstacle_map::Union{Nothing, Array{}}, scale_factor::Float64)
    nodes_dict = JSON.parsefile(fpath)
    nodes = Array{AbstractNode, 1}(undef, length(nodes_dict))

    n_nodes::Int = 0

    for (strid, node) in nodes_dict
        id = parse(Int, strid)
        if id < 0
            nodes[length(nodes_dict) + id + 1] = DummyNode(string(length(nodes_dict) + id + 1), node, scale_factor)
        else
            n_nodes +=1
            nodes[id] = Node(strid, node, scale_factor)
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

    # TODO: This is getting messy with adj needing paths to generate. Sticky circular dependency
    world_state = WorldState(nodes, n_nodes, graph_map, obstacle_map, scale_factor)
    adj = get_real_adj(world_state)
    world_state = WorldState(nodes, n_nodes, graph_map, obstacle_map, scale_factor, adj)
    return world_state
end

"""
    world_step(world_state::WorldState, agents::Array{AgentState, 1})

Return updated world state and reward allocated to agents
"""
function world_step(world_state::WorldState, agents::Array{AgentState, 1})

    nodes = copy(world_state.nodes)
    for node in nodes
        if node isa Node
            node.values.idleness += 1.0
            for agent in agents
                if agent.graph_position isa Int64 && agent.graph_position == node.id
                    node.values.idleness = 0.0
                end
            end
        end
    end

    updated_world_state = WorldState(nodes, world_state.n_nodes, world_state.map, world_state.obstacle_map, world_state.scale_factor, world_state.adj, world_state.paths, world_state.time + 1, world_state.done)
    
    rewards = zeros(Float64, length(agents))

    return true, updated_world_state, rewards
end

"""
    stop_world()

Safely stop the simulation
"""
function stop_world()
    nothing
end

end