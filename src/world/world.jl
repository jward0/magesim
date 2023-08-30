module World

import ..Types: WorldState, AgentState, Node
import ..WorldRenderer: create_window, update_window!, close_window, gtk_is_running
import ..Utils: pos_distance
using Gtk
using Graphs, SimpleWeightedGraphs
using JSON

"""
    create_world(fpath::String)

Load world info from JSON file, construct node and map representations, and return world state
"""
function create_world(fpath::String)
    nodes_dict = JSON.parsefile(fpath)
    nodes = Array{Node, 1}(undef, length(nodes_dict))

    for (strid, node) in nodes_dict
        nodes[parse(Int, strid)] = Node(strid, node)
    end
    
    sources = Vector{Int64}()
    destinations = Vector{Int64}()
    weights = Vector{Float64}()

    for node in nodes
        for n in node.neighbours
            neighbour = nodes[n]
            push!(sources, node.id)
            push!(destinations, neighbour.id)
            push!(weights, pos_distance(node.position, neighbour.position))
        end
    end

    graph_map = SimpleWeightedDiGraph(sources, destinations, weights)

    world_state = WorldState(nodes, graph_map)
    return world_state
end


"""
    world_step(world_state::WorldState, agents::Array{AgentState, 1}, builder::Gtk.GtkBuilderLeaf)

Update world state
"""
function world_step(world_state::WorldState, agents::Array{AgentState, 1}, builder::Gtk.GtkBuilderLeaf, headless::Bool=false)
    updated_world_state = WorldState(world_state.nodes, world_state.map, world_state.paths, world_state.time + 1, world_state.done)
    return true, updated_world_state
end

"""
    stop_world(builder::Gtk.GtkBuilderLeaf)

Safely stop the simulation and close the GUI
"""
function stop_world()
    nothing
end

end