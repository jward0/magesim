module Utils

import ..Types: WorldState, AgentState, Node, Position
using Graphs

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

end