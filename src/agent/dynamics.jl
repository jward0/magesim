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

    # Break early if already at target

    if agent.graph_position isa Int64 && target == agent.graph_position
        return agent.position, agent.graph_position, true
    end

    # Get list of valid immediate targets

    neighbours = get_neighbours(agent.graph_position, world, false)

    # Select immediate node target

    # TODO: modify this to account for dynamic env (move dists into agent?)
    distances_via_neighbours = map(n -> world.paths.dists[n, target] + pos_distance(agent.position, world.nodes[n].position), neighbours)
    t = world.nodes[neighbours[argmin(distances_via_neighbours)]]

    # Get agent speed (based on environment dynamics)

    # If at node:
    if agent.graph_position isa Int64
        src = agent.graph_position
        dst = t.id
    else # If on edge: 
        src = agent.graph_position.src
        dst = agent.graph_position.dst
    end

    # Correct for dummy nodes
    if src > world.n_nodes
        src = agent.values.last_visited
    end
    if dst > world.n_nodes
        dst = target
    end

    slowdown = world.temporal_profiles[floor(Integer, world.time)+1][src, dst]

    # if agent.id == 1
    #     println(slowdown)
    # end

    speed = agent.step_size * slowdown

    # Step along current edge towards immediate target

    diff = operate_pos(t.position, agent.position, -)

    step_size = min(speed, pos_norm(diff))

    step = operate_pos(diff, step_size/pos_norm(diff), *)

    if pos_norm(diff) == 0
        step = Position(0, 0)
    end

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