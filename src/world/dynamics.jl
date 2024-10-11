module WorldDynamics

using Graphs, SimpleWeightedGraphs, SparseArrays, LinearAlgebra

function generate_temporal_profiles(adj::Matrix{Float64}, timeout::Int64)

    edge_locs = findnz(sparse(triu(adj)))
    n_edges = length(edge_locs[1])
    n_nodes = length(adj)

    profiles::Vector{Vector{Float64}} = [profile(timeout) for _ in 1:n_edges]

    # profiled_adj = Vector{SparseMatrixCSC{Float64, Int64}}(undef, timeout)
    profiled_adj = Vector{Matrix{Float64}}(undef, timeout)

    ts_adj = Matrix(zeros(n_nodes, n_nodes))

    for t in 1:timeout

        for i in 1:n_edges
            ts_adj[edge_locs[1][i], edge_locs[2][i]] = profiles[i][t]
            ts_adj[edge_locs[2][i], edge_locs[1][i]] = profiles[i][t]
        end

        # ts_adj = Matrix(Symmetric(ts_adj))
        # push!(profiled_adj, sparse(ts_adj))
        # profiled_adj[t] = sparse(ts_adj)
        profiled_adj[t] = ts_adj

    end

    return profiled_adj

end

function profile(timeout::Int64)

    # walk = rand(Float64)
    noise = rand(Float64)
    # blocks = rand(Float64)

    # walk_profile::Vector{Float64} = vec![0 for _ in 1:timeout]
    noise_profile::Vector{Float64} = [1 for _ in 1:timeout] - abs.(randn(timeout) .* 0.1 .* noise)
    # blocks_profile::Vector{Float64} = vec![0 for _ in 1:timeout]

    return noise_profile
end

end