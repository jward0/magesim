module Utils

import ..Types: WorldState, AgentState, Node, Position
using Graphs, SimpleWeightedGraphs, LinearAlgebra

function pos_distance(p1::Position, p2::Position)
    return ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)^0.5
end

function pos_norm(p::Position)
    return (p.x^2 + p.y^2)^0.5
end

function operate_pos(p1::Position, p2::Position, f)
    return Position(f(p1.x, p2.x), f(p1.y, p2.y))
end

function operate_pos(p::Position, n::Number, f)
    return Position(f(p.x, n), f(p.y, n))
end

function get_neighbours(agent_pos::Union{AbstractEdge, Int64}, world::WorldState, no_dummy_nodes::Bool)
    """
    Returns array of node ids (integers)
    """
    if agent_pos isa Int64
        neighbours = copy(neighbors(world.map, agent_pos))
    elseif has_edge(world.map, dst(agent_pos), src(agent_pos))
        neighbours = [src(agent_pos), dst(agent_pos)]
    else
        return [dst(agent_pos)]
    end

    if no_dummy_nodes
        # Will currently break if agent_pos is an edge
        log::Array{Int64, 1} = []

        function iterate(node, previous_node)
            if node <= world.n_nodes
                push!(log, node)
            else
                next_neighbours = copy(neighbors(world.map, node))
                filter!(e -> e != previous_node, next_neighbours)
                iterate(next_neighbours[1], node)
            end
        end

        for n in neighbours
            iterate(n, agent_pos)
        end

        neighbours = unique(log)
    end

    return neighbours
end

function get_real_adj(world::WorldState)
    """
    Gets adjacency matrix, skipping dummy nodes
    """

    adj = Matrix(zeros(Float64, (world.n_nodes, world.n_nodes)))

    for node in world.nodes[1:world.n_nodes]
        for n in get_neighbours(node.id, world, true)
            adj[node.id, n] = world.paths.dists[node.id, n]
        end
    end

    return adj
end

end