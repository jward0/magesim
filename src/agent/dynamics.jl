module AgentDynamics

import ..Types: Position, WorldState, AgentState
import ..Utils: pos_distance, operate_pos, pos_norm, get_neighbours
using Graphs, SimpleWeightedGraphs, LinearAlgebra

"""
    calculate_next_position(agent::AgentState, target::Int64, world::WorldState)

Calculate the updated cartesian and graph position of an agent following one timestep of movement
towards a target node.

Path chosen to step along is the shortest path towards the target. Additionally returns flag 
to indicate whether target has been reached.

returns: 
new_pos::Position, updated cartesian position of agent
new_graph_pos::Union{AbstractEdge, Int64}, edge or node ID associated with agent's graph position
at_target::Bool, target reached flag
"""

function calculate_next_position(agent::AgentState, target::Int64, world::WorldState, blocked_pos::Array{Position, 1})

    # Get list of valid immediate targets

    if agent.graph_position isa Int64 && target == agent.graph_position
        # Break early if already at target
        return agent.position, agent.graph_position, true
    end

    neighbours = get_neighbours(agent.graph_position, world, false)

    # Select immediate node target

    distances_via_neighbours = map(n -> world.paths.dists[n, target] + pos_distance(agent.position, world.nodes[n].position), neighbours)
    t = world.nodes[neighbours[argmin(distances_via_neighbours)]]

    # Step along current edge towards immediate target

    diff = operate_pos(t.position, agent.position, -)

    step_size = min(agent.step_size, pos_norm(diff))

    step = operate_pos(diff, step_size/pos_norm(diff), *)

    new_pos = operate_pos(agent.position, step, +)

    # Check if it would be colliding with any block points (usually used to handle agent collision)
    if new_pos in blocked_pos
        return agent.position, agent.graph_position, false
    end

    at_target = false

    if pos_norm(diff) == step_size
        # Stepping along edge, arrived at a (potentially intermediate) target
        new_graph_pos = t.id
        if t.id == target 
            at_target = true
        end       
    elseif agent.graph_position isa Int64 && pos_norm(step) > 0
        # Stepping away from node 
        new_graph_pos = Graphs.SimpleEdge(agent.graph_position, t.id)
    else
        # Stepping along edge, not arrived at target
        new_graph_pos = agent.graph_position
    end

    return new_pos, new_graph_pos, at_target

end

end