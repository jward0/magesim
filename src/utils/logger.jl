module LogWriter

import ..Types: Logger, WorldState, AgentState, Node
using Graphs, SimpleWeightedGraphs
using Dates

"""
    new_logger()

Create required log directory and return new logger
"""
function new_logger()

    logger = Logger(string("logs/", Dates.format(now(), "yyyymmdd_HH:MM:SS/")))

    if !isdir(logger.log_directory)
        Base.Filesystem.mkpath(logger.log_directory)
    end

    return logger
end

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

end