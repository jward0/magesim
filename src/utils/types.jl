module Types

using Graphs, SimpleWeightedGraphs, DataStructures, Infinity, Dates, Flux, SparseArrays

# --- Abstract types ---

abstract type AbstractMessage end

abstract type AbstractAction end

abstract type AbstractNode end

# --- General utility types ---

struct Position
    x::Float64
    y::Float64
end

struct UserConfig
    data::Dict{String, Union{String, Array{Any, 1}}}

    function UserConfig(d)
        new(d)
    end
end

mutable struct Config
    # This is mutable to allow for easy parameter sweeping
    # (Each sweep just re-runs with modified config)

    # World configs
    world_fpath::String
    obstacle_map::Union{Nothing, Array{}}
    scale_factor::Float64
    # Agent configs
    n_agents::Int64
    agent_starts::Array{Int64, 1}
    comm_range::Float64
    check_los::Bool
    # Run configs
    headless::Bool
    speedup::Float64
    timeout::Int64
    multithreaded::Bool
    do_log::Bool
    # User-defined config
    custom_config::UserConfig
end

struct Logger
    log_directory::String

    function Logger(config::Config)

        log_directory = string("logs/", Dates.format(now(), "yyyymmdd_HH:MM:SS/"))

        if !isdir(log_directory)
            Base.Filesystem.mkpath(log_directory)
        end

        fpath = string(log_directory, "config.txt") 
        open(fpath, "a") do f
            println(f, "world: $(config.world_fpath)")
            println(f, "n_agents: $(config.n_agents)")
            println(f, "agent_starts: $(config.agent_starts)")
            println(f, "comm_range: $(config.comm_range)")
            println(f, "check_los: $(config.check_los)")
            println(f, "timeout: $(config.timeout)")
            println(f, "custom_config: $(config.custom_config)")
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
    adj::Matrix{Float64} # Adjacency matrix of only real nodes
    temporal_profiles::Union{Vector{Matrix{Float64}}, Nothing}
    paths::Graphs.FloydWarshallState  # Has fields dists, parents (for back-to-front navigation)
    # temporal_profiles::Union{Vector{SparseMatrixCSC{Float64, Int64}}, Nothing}
    time::Real
    done::Bool
    
    function WorldState(nodes::Array{AbstractNode, 1},
                        n_nodes::Int,
                        map::AbstractGraph,
                        obstacle_map::Union{Nothing, Array{}},
                        scale_factor::Float64,
                        adj::Union{Matrix{Float64}, Nothing}=nothing,
                        temporal_profiles::Union{Vector{Matrix{Float64}}, Nothing}=nothing,
                        paths::Union{Graphs.AbstractPathState, Nothing}=nothing,
                        # temporal_profiles::Union{Vector{SparseMatrixCSC{Float64, Int64}}, Nothing}=nothing,
                        time::Float64=0.0, 
                        done::Bool=false)

        # TODO: sort all this out properly

        if adj === nothing
            new_adj = Matrix(zeros(Float64, (n_nodes, n_nodes)))
        else
            new_adj = adj
        end

        if paths === nothing
            generated_paths = floyd_warshall_shortest_paths(map)
            new(nodes, n_nodes, map, obstacle_map, scale_factor, new_adj, temporal_profiles, generated_paths, time, done)
        else
            new(nodes, n_nodes, map, obstacle_map, scale_factor, new_adj, temporal_profiles, paths, time, done)
        end
    end

    # function WorldState(original::WorldState, new_adj::Matrix{Float64})

    #     new(original.nodes, original.n_nodes, original.map, original.obstacle_map, original.scale_factor, new_adj, original.temporal_profiles, original.paths, original.time, original.done)
    # end
end

# --- Agent types ---

mutable struct AgentValues
    priority_log::Array{Float64, 2}
    priority_weights::Array{Float64, 1}
    other_targets::Array{Int64, 1}
    idleness_log::Array{Float64, 1}
    agent_dists_log::Array{Float64, 1}
    n_agents_belief::Int64
    last_visited::Int64
    last_last_visited::Int64
    n_messages::Int64
    current_target::Int64
    other_targets_freshness::Array{Float64, 1}
    # Keys are (src, dst), values are (time, observed weight)
    observed_weights_log::Dict{Tuple{Int64, Int64}, Array{Tuple{Real, Float64}, 1}}
    effective_adjacency_matrix::Matrix{Float64}
    departed_time::Float64
    strategy::String
    # SEBS bits
    intention_log::Array{Int64, 1}
    sebs_gains::Tuple{Float64, Float64}

    function AgentValues(n_agents::Int64, n_nodes::Int64, custom_config::UserConfig)
        new(ones(Float64, (n_agents, n_nodes)) .* -9999, 
            zeros(Float64, n_nodes), 
            zeros(Int64, n_agents),
            zeros(Float64, n_nodes),
            zeros(Float64, n_agents),
            n_agents,
            0,
            0,
            0,
            0,
            zeros(Float64, n_agents),
            Dict(),
            zeros(Float64, (n_nodes, n_nodes)),
            0,
            "SEBS",
            zeros(Int64, n_agents),
            (0.1, 100.0))

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
    check_los_flag::Bool
    sight_range::Float64
    inbox::Queue{AbstractMessage}
    outbox::Queue{AbstractMessage}
    world_state_belief::Union{WorldState, Nothing}

    function AgentState(id::Int64, start_node_idx::Int64, start_node_pos::Position, n_agents::Int64, n_nodes::Int64, comm_range::Float64, check_los::Bool, custom_config::UserConfig)

        values = AgentValues(n_agents, n_nodes, custom_config)
        new(id, start_node_pos, values, Queue{AbstractAction}(), start_node_idx, 1.0, comm_range, check_los, 10.0, Queue{AbstractMessage}(), Queue{AbstractMessage}(), nothing)    
    end

    function AgentState(
        id::Int64,
        position::Position,
        values::AgentValues,
        action_queue::Queue{AbstractAction},
        graph_position::Union{AbstractEdge, Int64},
        step_size::Float64,
        comm_range::Float64,
        check_los_flag::Bool,
        sight_range::Float64,
        inbox::Queue{AbstractMessage},
        outbox::Queue{AbstractMessage},
        world_state_belief::Union{WorldState, Nothing})

        new(id, position, values, action_queue, graph_position, step_size, comm_range, check_los_flag, sight_range, inbox, outbox, world_state_belief)

    end
end

# --- Message types ---

struct ObservedWeightMessage <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    # ((dst, src), (timestamp, weight))
    message::Tuple{Tuple{Int64, Int64}, Tuple{Real, Float64}}

    function ObservedWeightMessage(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::Tuple{Tuple{Int64, Int64}, Tuple{Real, Float64}})
        new(agent.id, targets, message)
    end
end

struct PriorityMessage <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    message::Array{Float64, 1}

    function PriorityMessage(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::Array{Float64, 1})

        new(agent.id, targets, message)
    end
end

struct IdlenessLogMessage <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    message::Array{Float64, 1}

    function IdlenessLogMessage(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::Array{Float64, 1})

        new(agent.id, targets, message)
    end
end

struct PosMessage <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    message::Position

    function PosMessage(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::Position)

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

struct ArrivedAtNodeMessageSPNS <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    message::Int64

    function ArrivedAtNodeMessageSPNS(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::Int64)

        new(agent.id, targets, message)
    end
end

struct ArrivedAtNodeMessageSEBS <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    message::Tuple{Int64, Int64}

    function ArrivedAtNodeMessageSEBS(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::Tuple{Int64, Int64})

        new(agent.id, targets, message)
    end
end

struct GoingToMessage <: AbstractMessage
    source::Int64
    targets::Union{Array{Int64, 1}, Nothing}
    message::Int64

    function GoingToMessage(agent::AgentState, targets::Union{Array{Int64, 1}, Nothing}, message::Int64)

        new(agent.id, targets, message)
    end
end

export WorldState
export AgentState

end