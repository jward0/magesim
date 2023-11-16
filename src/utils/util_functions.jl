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

function get_neighbours(agent_pos::Union{AbstractEdge, Int64}, map::AbstractGraph)
    if agent_pos isa Int64
        neighbours = neighbors(map, agent_pos)
    elseif has_edge(map, dst(agent_pos), src(agent_pos))
        neighbours = [src(agent_pos), dst(agent_pos)]
    else
        neighbours = [dst(agent_pos)]
    end
end

