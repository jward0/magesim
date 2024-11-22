module World

import ..Types: WorldState, AgentState, Node, DummyNode, AbstractNode, Config
import ..Utils: pos_distance, get_real_adj
using Graphs, SimpleWeightedGraphs
using JSON
using Accessors

"""
    create_world(fpath::String)

Load world info from JSON file, construct node and map representations, and return world state
"""
function create_world(config::Config)
    
    fpath = config.world_fpath
    obstacle_map = config.obstacle_map
    scale_factor = config.scale_factor

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

    world_state = WorldState(nodes, n_nodes, graph_map, obstacle_map, scale_factor)
    adj = get_real_adj(world_state)
    weight_limited_paths = generate_weight_limited_paths(adj, 20.0)

    @reset world_state.adj=adj
    @reset world_state.weight_limited_paths=weight_limited_paths

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
   
    @reset world_state.nodes=nodes    
    @reset world_state.time=world_state.time+1

    rewards = zeros(Float64, length(agents))

    return true, world_state, rewards
end

function generate_weight_limited_paths(adj::Matrix{Float64}, max_w::Float64)

    n_nodes = size(adj)[1]
    neighbours = [findall(!iszero, adj[i, :]) for i in 1:n_nodes]

    # First axis is start node
    # Second axis is paths from that start node
    # Then each path is a vector of {Node ID, distance from start node}
    paths_from_nodes::Vector{Vector{Vector{Tuple{Int64, Float64}}}} = [[] for _ in 1:n_nodes]

    for node in 1:n_nodes
        get_valid_paths!(node, adj, neighbours, [(node, 0.0)], max_w, paths_from_nodes[node])
    end

    return paths_from_nodes
end

function get_valid_paths!(node::Int64, adj::Matrix{Float64}, neighbours::Vector{Vector{Int64}}, path_so_far::Vector{Tuple{Int64, Float64}}, w::Float64, paths::Vector{Vector{Tuple{Int64, Float64}}})
    
    # Check if path must terminate here
    step_weights = adj[node, neighbours[node]]
    valid_next_steps = findall(n->(n<=w)&&(!iszero(n)), adj[node, :])
    if length(valid_next_steps) == 0
        push!(paths, path_so_far)
    else
        # Iterate on valid next steps
        for s in valid_next_steps
            step_w = ceil(adj[node, s])
            remaining_w = w - step_w
            get_valid_paths!(s, adj, neighbours, vcat(path_so_far, (s, path_so_far[end][2]+step_w)), remaining_w, paths)
        end
    end
end

"""
    stop_world()

Safely stop the simulation
"""
function stop_world()
    nothing
end

end