module SpaceHandler
include("types.jl")
import .Types: WaitAction, MoveToAction, StepTowardsAction, AgentState, WorldState, NodeValues

function generate_action_space(agent::AgentState)

end

function parse_action_from_python()

end

function unwrap_node_values()

    node_values = NodeValues()

    default_arr = []
    labels_arr = []

    for n in fieldnames(NodeValues)
        push!(default_arr, getfield(node_values, n))
        push!(labels_arr, n)
    end

    default_tuple = tuple(default_arr...)
    labels_tuple = tuple(labels_arr...)

    return default_tuple, labels_tuple
end

function parse_world_from_python()

end

end