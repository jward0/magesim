module World

import ..Types: WorldState, AgentState, Node, DummyNode, AbstractNode, Config
import ..Utils: pos_distance, get_real_adj
import ..WorldDynamics: generate_temporal_profiles, save_profile, load_profile
using Graphs, SimpleWeightedGraphs
using JSON
using JLD
using Accessors

"""
    create_world(fpath::String)

Load world info from JSON file, construct node and map representations, and return world state
"""
function create_world(config::Config)
    
    fpath = config.world_fpath
    obstacle_map = config.obstacle_map
    scale_factor = config.scale_factor

    nodes_dict = JSON.parsefile(fpath)
    nodes = Array{AbstractNode, 1}(undef, length(nodes_dict))

    n_nodes::Int = 0

    for (strid, node) in nodes_dict
        id = parse(Int, strid)
        if id < 0
            nodes[length(nodes_dict) + id + 1] = DummyNode(string(length(nodes_dict) + id + 1), node, scale_factor)
        else
            n_nodes +=1
            nodes[id] = Node(strid, node, scale_factor)
        end
    end
    
    sources = Vector{Int64}()
    destinations = Vector{Int64}()
    weights = Vector{Float64}()

    for node in nodes
        for n in node.neighbours
            if n < 0
                neighbour = nodes[length(nodes_dict) + n + 1]
            else
                neighbour = nodes[n]
            end
            push!(sources, node.id)
            push!(destinations, neighbour.id)
            push!(weights, pos_distance(node.position, neighbour.position))
        end
    end

    graph_map = SimpleWeightedDiGraph(sources, destinations, weights)

    # TODO: This is getting messy with adj needing paths to generate. Sticky circular dependency
    world_state = WorldState(nodes, n_nodes, graph_map, obstacle_map, scale_factor)
    adj = get_real_adj(world_state)
    
    # Generate temporal profiles
    if config.custom_config.data["source"] == "generate"
        temporal_profiles = generate_temporal_profiles(adj, config.timeout, config.custom_config.data["args"]...)
        save_profile(config.custom_config.data["name"], temporal_profiles)
    elseif config.custom_config.data["source"] == "load"
        temporal_profiles = load_profile("$(config.world_name)_$(config.custom_config.data["name"])")
    elseif config.custom_config.data["source"] == "none"
        temporal_profiles = [ones(Float64, size(adj)) for _ in 1:config.timeout]
    else
        throw("Unrecognised temporal profile generation type (must be \"generate\", \"load\", or \"none\")")
    end

    @reset world_state.adj=adj
    @reset world_state.temporal_profiles=temporal_profiles

    return world_state
end

"""
    world_step(world_state::WorldState, agents::Array{AgentState, 1})

Return updated world state and reward allocated to agents
"""

function world_step(world_state::WorldState, agents::Array{AgentState, 1})

    nodes = copy(world_state.nodes)
    for node in nodes
        if node isa Node
            node.values.idleness += 1.0
            for agent in agents
                if agent.graph_position isa Int64 && agent.graph_position == node.id
                    node.values.idleness = 0.0
                end
            end
        end
    end
   
    @reset world_state.nodes=nodes
    @reset world_state.time=world_state.time+1

    rewards = zeros(Float64, length(agents))

    return true, world_state, rewards
end

"""
    stop_world()

Safely stop the simulation
"""
function stop_world()
    nothing
end

end