module WorldRenderer

import ..Types: WorldState, AgentState
using Gtk, Cairo, Plots, GraphRecipes, Graphs, Images
gr()

const canvas = GtkCanvas()
const io = PipeBuffer()
running::Bool = true


"""
    create_window()

Create and return the GtkBuilder for a GTK window containing all GUI elements. 
"""
function create_window()
    b = GtkBuilder(filename="src/world/window.glade")
    win = b["window1"]
    push!(b["box2"], canvas)
    set_gtk_property!(b["box2"], :expand, canvas, true)

    # Hijack window close button to raise safe close flag
    signal_connect(win, "delete-event") do widget, event
        global running = false
    end

    showall(win)
    return b
end


"""
    update_window!(world_state::WorldState, agents::Array{AgentState, 1}, builder::Gtk.GtkBuilderLeaf)

Set GUI elements and draw canvas based on current world and agents state
"""
function update_window!(world_state::WorldState, agents::Array{AgentState, 1}, actual_speedup::Float64, builder::Gtk.GtkBuilderLeaf)
    timer = builder["timelabel"]
    GAccessor.text(timer, string(world_state.time))
    speedup = builder["speeduplabel"]
    GAccessor.text(speedup, string(round(actual_speedup, digits=1)))

    sf = world_state.scale_factor
    
    node_xs = [node.position.x / sf for node in world_state.nodes]
    node_ys = [node.position.y / sf for node in world_state.nodes]
    agent_xs = [agent.position.x / sf for agent in agents]
    agent_ys = [agent.position.y / sf for agent in agents]

    x_size=600

    @guarded draw(canvas) do widget
        ctx = getgc(canvas)

        if !isnothing(world_state.obstacle_map)
            lower_lims = (0, 0)
            upper_lims = size(world_state.obstacle_map)
        else
            lower_lims = (minimum(node_ys)-5, minimum(node_xs)-5)
            upper_lims = (maximum(node_ys)+5, maximum(node_xs)+5)
        end

        aspect_ratio = (upper_lims[1] - lower_lims[1]) / (upper_lims[2] - lower_lims[2])

        println(ne(world_state.map))

        # Draw graph
        graphplot(world_state.map, 
        curves=false, 
        nodesize=0,
        x=node_xs, 
        y=node_ys, 
        xlims=(lower_lims[2], upper_lims[2]), 
        ylims=(lower_lims[1], upper_lims[1]), 
        aspect_ratio = :equal, 
        framestyle=:box, 
        ticks=:none, 
        legend=false,
        size=(x_size, trunc(Int64, x_size*aspect_ratio)))            
        resize!(builder["window1"], x_size, trunc(Int64, x_size*aspect_ratio)+20)

        # Render obstacle layer (if present)
        if !isnothing(world_state.obstacle_map)
            plot!(world_state.obstacle_map, yflip=false, z_order=1)
            # resize!(builder["window1"], 600, 400)
        end

        # Draw agents and nodes
        scatter!(node_xs[1:world_state.n_nodes], node_ys[1:world_state.n_nodes], markercolor=:red)
        show(io, MIME("image/png"), scatter!(agent_xs, agent_ys, markercolor=:blue))
        img = read_from_png(io)
        set_source_surface(ctx, img, 0, 0)
        paint(ctx)
    end

    draw(canvas)
    show(canvas)

    return running
end


"""
    gtk_is_running()

Return running::Bool to indicate whether GTK is running or not
"""
function gtk_is_running()
    return running
end


"""
    close_window(builder::Gtk.GtkBuilderLeaf)

Safely close IO buffer and GTK builder
"""
function close_window(builder::Gtk.GtkBuilderLeaf)
    close(io)
    Gtk.destroy(builder["window1"])
end

export create_window, update_window!, close_window, gtk_is_running

end