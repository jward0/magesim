module WorldRenderer

import ..Types: WorldState, AgentState
using Gtk, Cairo, Plots, GraphRecipes, Graphs
gr()

const canvas = GtkCanvas()
const io = PipeBuffer()
running::Bool = true


"""
    create_window()

Create and return the GtkBuilder for a GTK window containing all GUI elements. 
"""
function create_window()
    b = GtkBuilder(filename="src/world/test.glade")
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
    
    node_xs = [node.position.x for node in world_state.nodes]
    node_ys = [node.position.y for node in world_state.nodes]
    agent_xs = [agent.position.x for agent in agents]
    agent_ys = [agent.position.y for agent in agents]
    
    @guarded draw(canvas) do widget
        ctx = getgc(canvas)
        # Draw graph
        graphplot(world_state.map, 
            curves=false, 
            nodesize=1, 
            x=node_xs, 
            y=node_ys, 
            xlims=(-5, maximum(node_xs)+5), 
            ylims=(-5, maximum(node_ys)+5), 
            aspect_ratio = :equal, 
            framestyle=:box, 
            ticks=:none, 
            legend=false)
        # Draw agents and nodes
        show(io, MIME("image/png"), scatter!(agent_xs, agent_ys))
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