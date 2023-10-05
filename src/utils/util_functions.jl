module Utils

import ..Types: WorldState, AgentState, Node, Position

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


end