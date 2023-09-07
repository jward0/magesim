module Types

using Graphs, DataStructures, Infinity

# --- Abstract types ---

abstract type AbstractMessage end

abstract type AbstractAction end

abstract type AbstractNode end

# --- General utility types ---

struct Position
    x::Float64
    y::Float64
end

struct Logger
    log_directory::String
end

# --- Agent action types ---

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
    value_string::String
end

struct DummyNode <: AbstractNode
    id::Integer
    position::Position
    neighbours::Array{Integer, 1}

    function DummyNode(strid::String, node_dict::Dict{String, Any})

        id = parse(Int64, strid)
        position = Position(node_dict["position"]["x"], 
                            node_dict["position"]["y"])
        neighbours = node_dict["neighbours"]

        new(id, position, neighbours)
    end
end

struct Node <: AbstractNode
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


"""
Make WorldState mutable at your peril! This simulator is only guaranteed thread-safe if WorldState is immutable,
thus allowing multiple agents in multiple threads to safely read it simultaneously
"""
struct WorldState
    nodes::Array{AbstractNode, 1}
    n_nodes::Int
    map::AbstractGraph
    paths::Graphs.AbstractPathState  # Has fields dists, parents (for back-to-front navigation)
    time::Real
    done::Bool
    
    function WorldState(nodes::Array{AbstractNode, 1},
                        n_nodes::Int,
                        map::AbstractGraph, 
                        paths::Union{Graphs.AbstractPathState, Nothing}=nothing, 
                        time::Float64=0.0, 
                        done::Bool=false)

        if paths === nothing
            generated_paths = floyd_warshall_shortest_paths(map)
            new(nodes, n_nodes, map, generated_paths, time, done)
        else
            new(nodes, n_nodes, map, paths, time, done)
        end
    end
end

# --- Agent types ---

struct AgentValues
    example_value::Nothing
end

struct AgentObservation
    observation::WorldState
end

mutable struct AgentState
    id::Integer
    position::Position
    values::AgentValues
    action_queue::Queue{AbstractAction}
    graph_position::Union{AbstractEdge, Int64}
    step_size::Float64
    comm_range::Float64
    inbox::Queue{AbstractMessage}
    outbox::Queue{AbstractMessage}
    world_state_belief::Union{WorldState, Nothing}
    observation::Union{AgentObservation, Nothing}

    function AgentState(id::Int64, start_node_idx::Int64, start_node_pos::Position, values::Nothing)

        new(id, start_node_pos, AgentValues(values), Queue{AbstractAction}(), start_node_idx, 1.0, ∞, Queue{AbstractMessage}(), Queue{AbstractMessage}(), nothing)    
    end
end

# --- Message types ---

struct StringMessage <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    message::String

    function StringMessage(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::String)

        new(agent.id, targets, message)
    end
end

export WorldState
export AgentState

end