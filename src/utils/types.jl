module Types

using Graphs, DataStructures

# --- General utility types ---

struct Position
    x::Float64
    y::Float64
end

# --- Agent action types ---

abstract type AbstractAction end

struct WaitAction <: AbstractAction
    field::Nothing
    
    function WaitAction()
        new(nothing)
    end
end

struct MoveToAction <: AbstractAction
    target::Int64
end

struct StepTowardsAction <: AbstractAction
    target::Int64
end

# --- Node + map types

struct NodeValues
    example_value::String
end

struct Node
    id::Integer
    label::String
    position::Position
    neighbours::Array{Integer, 1}
    values::NodeValues

    function Node(strid::String, node_dict::Dict{String, Any})

        id = parse(Int64, strid)
        label = node_dict["label"]
        position = Position(node_dict["position"]["x"], 
                            node_dict["position"]["y"])
        neighbours = node_dict["neighbours"]
        values = NodeValues(node_dict["values"])

        new(id, label, position, neighbours, values)
    end
end

struct WorldState
    nodes::Array{Node, 1}
    map::AbstractGraph
    paths::Graphs.AbstractPathState  # Has fields dists, parents (for back-to-front navigation)
    time::Real
    done::Bool
    
    function WorldState(nodes::Array{Node, 1}, 
                        map::AbstractGraph, 
                        paths::Union{Graphs.AbstractPathState, Nothing}=nothing, 
                        time::Float64=0.0, 
                        done::Bool=false)

        if paths === nothing
            generated_paths = floyd_warshall_shortest_paths(map)
            new(nodes, map, generated_paths, time, done)
        else
            new(nodes, map, paths, time, done)
        end
    end
end

# --- Agent types ---

struct AgentValues
    example_value::Nothing
end

mutable struct AgentState
    id::Integer
    position::Position
    values::AgentValues
    action_queue::Queue{AbstractAction}
    graph_position::Union{AbstractEdge, Int64}
    step_size::Float64

    function AgentState(id::Int64, start_node_idx::Int64, start_node_pos::Position, values::Nothing)

        new(id, start_node_pos, AgentValues(values), Queue{AbstractAction}(), start_node_idx, 1.0)    
    end
end

export WorldState
export AgentState

end