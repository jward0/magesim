module Types

using Graphs, DataStructures, Infinity, Dates

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

    function Logger()

        log_directory = string("logs/", Dates.format(now(), "yyyymmdd_HH:MM:SS/"))

        if !isdir(log_directory)
            Base.Filesystem.mkpath(log_directory)
        end

        new(log_directory)
    end
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

Base.@kwdef mutable struct NodeValues
    """
    For python wrapping purposes, the only types supported in NodeValues are String, Int, Float, Bool, 
    and 1-d Array of these types

    Similarly for these purposes, avoid using 'Int' fields unless they can only a finite number of positive
    integral values, in which case the default value should be set to the largest possible. It is recommended
    that this is only used for eg. enums, and actual numbers should be represented as floats here. 
        
    Likewise, default string values should be a string of the integer value representing the longest
    possible length of the string (in characters)

    If you do not wish to use the PettingZoo wrapper, you may disregard the above comments.
    """
    value_string::String = "10"
    idleness::Float64 = 0.0

end

struct DummyNode <: AbstractNode
    id::Integer
    position::Position
    neighbours::Array{Integer, 1}

    function DummyNode(strid::String, node_dict::Dict{String, Any}, scale_factor::Float64)

        id = parse(Int64, strid)
        position = Position(node_dict["position"]["x"] * scale_factor, 
                            node_dict["position"]["y"] * scale_factor)
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

    function Node(strid::String, node_dict::Dict{String, Any}, scale_factor::Float64)

        id = parse(Int64, strid)
        label = node_dict["label"]
        position = Position(node_dict["position"]["x"] * scale_factor, 
                            node_dict["position"]["y"] * scale_factor)
        neighbours = node_dict["neighbours"]
        values = NodeValues()

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
    obstacle_map::Union{Nothing, Array{}}
    scale_factor::Float64
    paths::Graphs.FloydWarshallState  # Has fields dists, parents (for back-to-front navigation)
    time::Real
    done::Bool
    
    function WorldState(nodes::Array{AbstractNode, 1},
                        n_nodes::Int,
                        map::AbstractGraph,
                        obstacle_map::Union{Nothing, Array{}},
                        scale_factor::Float64,
                        paths::Union{Graphs.AbstractPathState, Nothing}=nothing, 
                        time::Float64=0.0, 
                        done::Bool=false)

        if paths === nothing
            generated_paths = floyd_warshall_shortest_paths(map)
            new(nodes, n_nodes, map, obstacle_map, scale_factor, generated_paths, time, done)
        else
            new(nodes, n_nodes, map, obstacle_map, scale_factor, paths, time, done)
        end
    end
end

# --- Agent types ---

mutable struct AgentValues
    intention_log::Array{Int64, 1}
    idleness_log::Array{Float64, 1}
    sebs_gains::Tuple{Float64, Float64}
    n_agents_belief::Int64

    function AgentValues(g1::Float64, g2::Float64, n_agents::Int64, n_nodes::Int64)
        new(zeros(Int64, n_agents), zeros(Float64, n_nodes), (g1, g2), n_agents)
    end
end

mutable struct AgentState
    id::Integer
    position::Position
    values::AgentValues
    action_queue::Queue{AbstractAction}
    graph_position::Union{AbstractEdge, Int64}
    step_size::Float64
    comm_range::Float64
    sight_range::Float64
    inbox::Queue{AbstractMessage}
    outbox::Queue{AbstractMessage}
    world_state_belief::Union{WorldState, Nothing}

    function AgentState(id::Int64, start_node_idx::Int64, start_node_pos::Position, n_agents::Int64, n_nodes::Int64, custom_config::Array{Float64, 1})

        values = AgentValues(custom_config[1], custom_config[2], n_agents, n_nodes)
        new(id, start_node_pos, values, Queue{AbstractAction}(), start_node_idx, 1.0, ∞, 10.0, Queue{AbstractMessage}(), Queue{AbstractMessage}(), nothing)    
    end
end

# --- Message types ---

struct ArrivedAtNodeMessage <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    # Node arrived at, next target
    message::Tuple{Int64, Int64}

    function ArrivedAtNodeMessage(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::Tuple{Int64, Int64})

        new(agent.id, targets, message)
    end
end

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