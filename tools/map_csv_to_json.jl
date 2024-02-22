using CSV, DataFrames, DataStructures
import JSON

function main(args)
    if length(args) != 3 
        throw(ArgumentError("Arguments must be: map name, path to node positions CSV, and path to adjacency matrix CSV"))
    end

    pos = Matrix(CSV.read(args[2], DataFrame, header=false))
    adj = Matrix(CSV.read(args[3], DataFrame, header=false))

    if size(pos, 2) != 2 || size(pos, 1) != size(adj, 1) || size(adj, 1) != size(adj, 2)
        throw(DimensionMismatch("Sizes of inputs do not match"))
    end

    map_dict = OrderedDict{String, Any}()

    for node in 1:size(pos, 1)

        neighbours::Vector{Int64} = []

        for n in 1:size(adj, 2)
            if adj[node, n] != 0
                append!(neighbours, n)
            end
        end

        map_dict[string(node)] = Dict{String, Any}()
        map_dict[string(node)]["position"] = Dict{String, Any}()

        map_dict[string(node)]["label"] = ""
        map_dict[string(node)]["position"]["x"] = pos[node, 1]
        map_dict[string(node)]["position"]["y"] = pos[node, 2]
        map_dict[string(node)]["neighbours"] = neighbours
        map_dict[string(node)]["values"] = ""

    end

    json_dict = JSON.json(map_dict, 4)
    print(json_dict)

    f = open(string(args[1], ".json"), "w")
    open(string("maps/", args[1], ".json"), "w") do f
        write(f, json_dict)
    end

end

main(ARGS)