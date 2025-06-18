module LogWriter

import ..Types: afLogger, Logger, WorldState, AgentState, Node
import ..Utils: pos_distance, get_distances
using Graphs, SimpleWeightedGraphs
using Dates

function af_log(agents::Array{AgentState, 1}, world::WorldState, logger::afLogger, timestep::Int)

    n_agents = length(agents)
    n_nodes = world.n_nodes

    # each element is the distance of each agent to every node
    dists = [get_distances(agent, world) for agent in agents]
    # each row is the distance of every agent to each node
    transpose_dists = reduce(hcat, dists)

    # Log agent-node distances
    for node in 1:n_nodes
        dist_fpath = "$(logger.log_directory)/distances/distances_$(node).csv"
        contents = join(string.(transpose_dists[node, :]), ",")
        write_csv_line(contents, dist_fpath)
    end

    # Log node idlenesses
    idlenesses = [node.values.idleness for node in world.nodes if node isa Node]
    id_fpath = "$(logger.log_directory)/idleness.csv" 
    contents = join(string.(vcat([timestep], idlenesses)), ",")
    write_csv_line(contents, id_fpath)

    # Log agent positions
    positions = vcat([[agent.position.x, agent.position.y] for agent in agents]...)
    pos_fpath = "$(logger.log_directory)/position_log.csv"
    contents = join(string.(vcat([timestep], positions)), ",")
    write_csv_line(contents, pos_fpath)
end
"""
    log(target::AgentState, logger::Logger, timestep::Int)

Log individual AgentState data
"""
function log(target::AgentState, logger::Logger, timestep::Int)
    fpath = string(logger.log_directory, "agent_", string(target.id), ".csv") 

    if !isfile(fpath)
        header = "timestep, x, y, graph location type, edge source/node, edge destination/node"
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    if target.graph_position isa Int64
        graph_pos_str = string("node,", string(target.graph_position), ",", string(target.graph_position))
    else
        graph_pos_str = string("edge,", string(src(target.graph_position)), ",", string(dst(target.graph_position)))
    end

    csv_line = string(string(timestep), ",", string(target.position.x), ",", string(target.position.y), ",", graph_pos_str)

    open(fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    log(target::Array{AgentState, 1}, logger::Logger, timestep::Int)

Log multiple instances of AgentState data
"""
function log(target::Array{AgentState, 1}, logger::Logger, timestep::Int)
   
    header_contents = ["x$n,y$n" for n in [1:1:length(target)...]]
    comm_header_contents = ["agent_$n" for n in [1:1:length(target)...]]
    positions = vcat([[agent.position.x, agent.position.y] for agent in target]...)

    pos_fpath = string(logger.log_directory, "agent_positions.csv") 

    if !isfile(pos_fpath)
        header = make_line("timestep", header_contents)
        open(pos_fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = make_line(timestep, string.(positions))
    open(pos_fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end

    interference_fpath = string(logger.log_directory, "interferences.csv")
    ts_interferences = 0
    for i in 1:length(target)
        for j in 1:i-1
            ts_interferences +=  pos_distance(target[i].position, target[j].position) < 1.01
        end
    end
    if !isfile(interference_fpath)
        header = make_line("timestep", ["interferences"])
        open(interference_fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end
    csv_line = make_line(timestep, string.([ts_interferences]))
    open(interference_fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    log(target::WorldState, logger::Logger, timestep::Int)

Log WorldState data
"""
function log(target::WorldState, logger::Logger, timestep::Int)

    header_contents = ["node_$n" for n in [1:1:target.n_nodes...]]
    idlenesses = [node.values.idleness for node in target.nodes if node isa Node]

    fpath = string(logger.log_directory, "idleness.csv") 

    if !isfile(fpath)
        header = make_line("timestep", string.(header_contents))
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = make_line(timestep, string.(idlenesses))

    open(fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    Utility functions for formatting
"""
function make_line(timestep::Int, contents::Array{String, 1})
    return join(vcat(string(timestep), contents), ',')
end
function make_line(timestep::String, contents::Array{String, 1})
    return join(vcat(timestep, contents), ',')
end

function write_csv_line(contents::String, fpath::String)
    Base.Filesystem.touch(fpath)
    open(fpath, "a") do file
        write(file, contents)
        write(file,"\n")
    end
end

end