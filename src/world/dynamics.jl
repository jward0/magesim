module WorldDynamics

using Graphs, SimpleWeightedGraphs, SparseArrays, LinearAlgebra

function generate_temporal_profiles(adj::Matrix{Float64}, timeout::Int64, noise_scale::Float64=0.0, walk_scale::Float64=0.0)

    edge_locs::Tuple{Vector{Int64}, Vector{Int64}, Vector{Float64}} = findnz(sparse(triu(adj)))
    n_edges = size(edge_locs[1])[1]
    n_nodes = size(adj)[1]

    profiles::Vector{Vector{Float64}} = [profile(timeout, noise_scale, walk_scale) for _ in 1:n_edges]

    # profiled_adj = Vector{SparseMatrixCSC{Float64, Int64}}(undef, timeout)
    profiled_adj = Vector{Matrix{Float64}}(undef, timeout)

    ts_adj = Matrix(zeros(n_nodes, n_nodes))

    for t in 1:timeout

        for i in 1:n_edges
            ts_adj[edge_locs[1][i], edge_locs[2][i]] = profiles[i][t]
            ts_adj[edge_locs[2][i], edge_locs[1][i]] = profiles[i][t]
        end

        profiled_adj[t] = ts_adj

    end

    return profiled_adj

end

function profile(timeout::Int64, noise_scale::Float64, walk_scale::Float64)

    # walk = rand(Float64)
    # blocks = rand(Float64)

    # walk_profile::Vector{Float64} = vec![0 for _ in 1:timeout]
    noise_profile::Vector{Float64} = [1 for _ in 1:timeout] - abs.(randn(timeout) .* noise_scale)
    walk_profile::Vector{Float64} = [1 for _ in 1:timeout]
    for i in 2:timeout
        walk_profile[i] = clamp(walk_profile[i-1] + randn(1)[1] * walk_scale, 0.0, 1.0)
    end

    clamp!(noise_profile, 0.0, 1.0)
    
    # blocks_profile::Vector{Float64} = vec![0 for _ in 1:timeout]

    return noise_profile .* walk_profile
end

end