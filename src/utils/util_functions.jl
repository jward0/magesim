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
    if agent_pos isa Int64
        neighbours = neighbors(world.map, agent_pos)
    elseif has_edge(world.map, dst(agent_pos), src(agent_pos))
        neighbours = [src(agent_pos), dst(agent_pos)]
    else
        return [dst(agent_pos)]
    end
    # Dummy node check - ensures that only real nodes will be returned
    if no_dummy_nodes
        for i in 1:length(neighbours)
            if neighbours[i] > world.n_nodes
                next_neighbours  = neighbors(world.map, neighbours[i])
                for j in 1:length(next_neighbours)
                    if !(next_neighbours[j] in neighbours)
                        neighbours[i] = next_neighbours[j]
                    end
                end
            end
        end
    end
    
    return neighbours
end

end
