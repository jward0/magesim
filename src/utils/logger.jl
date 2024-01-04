module LogWriter

import ..Types: Logger, WorldState, AgentState, Node
using Graphs, SimpleWeightedGraphs
using Dates

"""
    log(target::AgentState, logger::Logger, timestep::Int)

Log AgentState data
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
    positions = vcat([[agent.position.x, agent.position.y] for agent in target]...)


    fpath = string(logger.log_directory, "agent_positions.csv") 

    if !isfile(fpath)
        header = make_line("timestep", header_contents)
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = make_line(timestep, string.(positions))
    open(fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    log(target::Node, logger::Logger, timestep::Int)

Log Node data
"""
function log(target::Node, logger::Logger, timestep:: Int)
    fpath = string(logger.log_directory, "node_", string(target.id), ".csv")

    if !isfile(fpath)
        header = "timestep, value"
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = string(string(timestep, target.values.value_string))

    open(fpath, "a") do file
        write(file, csv_line)
        write(file,"\n")
    end
end

"""
    log(target::WorldState, logger::Logger, timestep::Int)

Log WorldState data
"""
function log(target::WorldState, logger::Logger, timestep::Int)
    fpath = string(logger.log_directory, "world.csv") 

    if !isfile(fpath)
        header = "timestep"
        open(fpath, "w") do file
            write(file, header)
            write(file,"\n")
        end
    end

    csv_line = string(timestep)

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

end