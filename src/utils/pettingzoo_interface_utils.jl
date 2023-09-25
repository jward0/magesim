module InterfaceUtils
# include("types.jl")
# Doesn't include here as this module should only be used with PZ wrapper,
# which include it
import ..Types: WaitAction, MoveToAction, StepTowardsAction, AgentState, WorldState, NodeValues, DummyNode
using Graphs, SimpleWeightedGraphs
using PythonCall

function generate_action_space(agent::AgentState)

end

function parse_from_py(input::PyArray)
    return pyconvert(Array{Int64}, input)
end

function parse_from_py(input::PyList)
    return pyconvert(Array{Int64}, input)
end

function unwrap_node_values(node_values::NodeValues = NodeValues())

    values_arr = []
    labels_arr = []

    for n in fieldnames(NodeValues)
        push!(values_arr, getfield(node_values, n))
        push!(labels_arr, n)
    end

    values_tuple = tuple(values_arr...)
    labels_tuple = tuple(labels_arr...)

    return values_tuple, labels_tuple
end

function unwrap_world(world::WorldState)

    map_node_positions::Vector{Vector{Float64}} = []
    for node in world.nodes
        append!(map_node_positions, [[node.position.x, node.position.y]])
    end

    map_edges = collect(edges(world.map))
    map_edge_links::Vector{Vector{Int64}} = []
    for edge in map_edges
        append!(map_edge_links, [[edge.src, edge.dst]])
    end

    node_dicts::Vector{Dict{String, Any}} = []
    for node in world.nodes
        if !(node isa DummyNode)
            values, labels =  unwrap_node_values(node.values)
            l = tuple([String(label) for label in labels]...)
            push!(node_dicts, Dict(zip(l, values)))
        end
    end

    return map_node_positions, map_edge_links, node_dicts
end

function parse_world_from_python()

end

end