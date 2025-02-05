function visit_maximisation!(agent::AgentState)

    if isempty(agent.action_queue)

        neighbours = get_neighbours(agent.graph_position, agent.world_state_belief, true)

        if length(neighbours) == 1
            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
        elseif !isa(agent.graph_position, Int64)
            # Catch the potential problem of an agent needing a new action
            # while midway between two nodes (not covered by algo) - 
            # solution to this is just to pick one
            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
        else
            distances = ceil.([agent.world_state_belief.adj[agent.graph_position, n] for n in neighbours])

            min_dist_indices = findall(distances .== minimum(distances))

            target = neighbours[rand(min_dist_indices)]

            enqueue!(agent.action_queue, MoveToAction(target))
        end

        if agent.graph_position isa Int64
            agent.values.departed_time = agent.world_state_belief.time
        end
    end
end

"""
    make_decisions_SPNS!(agent::AgentState)

Does SPNS
"""
function make_decisions_SPNS!(agent::AgentState)

    message_received = false

    if isempty(agent.action_queue)

        c = mean(agent.world_state_belief.adj[agent.world_state_belief.adj .!= 0])

        adjacency_matrix = agent.world_state_belief.adj / c
        idlenesses = agent.values.idleness_log

        # Original
        
        distances = dijkstra_shortest_paths(SimpleWeightedDiGraph(adjacency_matrix), agent.graph_position).dists

        node_values = hcat(idlenesses/maximum(idlenesses), distances/maximum(distances))

        model_in = [node_values, adjacency_matrix]

        # model_out = vec(forward_nn(model_in))
        model_out = vec(minimal_nn(node_values))

        priorities = model_out
        final_priorities = priorities
        
        target = do_sebs_style(agent, final_priorities)


        enqueue!(agent.action_queue, MoveToAction(target))

        enqueue!(agent.outbox, IdlenessLogMessage(agent, nothing, agent.values.idleness_log))
        enqueue!(agent.outbox, GoingToMessage(agent, nothing, target))

        agent.values.current_target = target

        if agent.graph_position isa Int64
            agent.values.departed_time = agent.world_state_belief.time
        end

    end
end

function minimal_nn(data::Matrix{Float64})

    id = data[:, 1]
    dis = data[:, 2]

    n0 = leakyrelu.((0.41935197 .* id) .- (1.036573603 .* dis), -0.21384633)
    n1 = leakyrelu.((1.02416510 .* id) .- (0.262816527 .* dis), 2.57125805)
    n2 = leakyrelu.((-0.43140551 .* id) .- (0.027082452 .* dis), 0.48526923)

    out = n0 .+ n1 .+ n2

    return out
end


function forward_nn(input)

    data = input[1]
    adj = input[2]

    unweighted_adj = copy(adj)
    unweighted_adj[adj .!= 0.] .= 1.
    repeated_edge = reshape(copy(adj), (size(adj)...,1))

    # REMEMBER COLUMN MAJOR ORDERING WHEN DEBUGGING

    # repeated_data has shape (n_nodes, n_nodes, 2)
    repeated_data = repeat(
        reshape(data, (size(data)[1], 1, size(data)[2])), 
        outer=[1, size(data)[1], 1])

    # combined_data has shape (n_nodes, n_nodes, 3)
    combined_data = cat(dims=3, repeated_data, repeated_edge)

    sc = sd_out(sd_1(data))

    nc = sum(unweighted_adj .* transpose(nd_out(nd_1(combined_data))), dims=2)

    output = leakyrelu(c0(sc) + c1(nc), 0.3)

    return output

end

# Candidate p
function sd_1(input)

    out = zeros(Float64, (size(input)[1], 4))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i, 1] =  0.32799476 * d[1] +  0.42836051 * d[2]
        out[i, 2] = -0.27840284 * d[1] + -1.03966187 * d[2]
        out[i, 3] = -2.14337451 * d[1] +  0.11616698 * d[2]
        out[i, 4] = -0.99945436 * d[1] + -0.90525171 * d[2]
    end

    return leakyrelu(out, 0.3)
end

function sd_out(input)

    out = zeros(Float64, (size(input)[1]))

    for i in 1:size(input)[1]
        d = input[i, :]
        out[i] = 1.2784934 * d[1] + -1.32684055 * d[2] + 2.28593493 * d[3] + 0.77020878 * d[4]
    end

    return leakyrelu(out, 0.3)
end

function nd_1(input)

    out = zeros(Float64, (size(input)[1], size(input)[2], 6))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j, 1] = -0.79343972 * d[1] +  1.31276063 * d[2] +  0.44181687 * d[3]
            out[i, j, 2] = -0.24341169 * d[1] +  2.32866916 * d[2] +  0.11335441 * d[3]
            out[i, j, 3] = -0.67571924 * d[1] + -1.08829292 * d[2] +  0.66480484 * d[3]
            out[i, j, 4] = -0.04343144 * d[1] + -1.24084405 * d[2] +  0.69653756 * d[3]
            out[i, j, 5] = -0.16365868 * d[1] +  0.32452302 * d[2] +  0.2104867  * d[3]
            out[i, j, 6] =  0.65356036 * d[1] +  0.2377102  * d[2] +  0.4955258  * d[3]
        end
    end

    return leakyrelu(out, 0.3)
end

function nd_out(input)

    out = zeros(Float64, (size(input)[1], size(input)[2]))

    for i in 1:size(input)[1]
        for j in 1:size(input)[2]
            d = input[i, j, :]
            out[i, j] = -0.66478846 * d[1] + -2.07131517 * d[2] + 1.27891665 * d[3] + 1.37051785 * d[4] + -0.68602119 * d[5] + -1.08835212 * d[6]
        end
    end

    return leakyrelu(out, 0.3)
end

function c0(input)
    return -2.43062395 * input
end

function c1(input)
    return 0.15660883 * input
end

function do_sebs_style(agent::AgentState, self_priorities::Array{Float64, 1})

    new_prio = copy(self_priorities)

    for ndx in [a for a in agent.values.other_targets if a > 0]
        new_prio[ndx] -= 9999.0
    end

    ns = get_neighbours(agent.graph_position, agent.world_state_belief, true)

    modified_prio = ones(size(new_prio)) * -99999.0

    for i = 1:size(new_prio)[1]
        if i in ns
            modified_prio[i] = new_prio[i]
        end
    end

    target = argmax(modified_prio)

    return target
end

"""
    make_decisions_SEBS!(agent::AgentState)

does SEBS
"""
function make_decisions_SEBS!(agent::AgentState)

    # If no action in progress, select node to move towards
    if isempty(agent.action_queue)

        neighbours = get_neighbours(agent.graph_position, agent.world_state_belief, true)

        if length(neighbours) == 1
            enqueue!(agent.outbox, ArrivedAtNodeMessageSEBS(agent, nothing, (agent.graph_position, neighbours[1])))
            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
        elseif !isa(agent.graph_position, Int64)
            # Catch the potential problem of an agent needing a new action
            # while midway between two nodes (not covered by algo) - 
            # solution to this is just to pick one
            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
        else
            # Do SEBS
            gains = map(n -> calculate_gain(n, agent), neighbours)
            posteriors = map(g -> calculate_posterior(g, agent), gains)
            n_intentions::Array{Int64, 1} = zeros(agent.world_state_belief.n_nodes)
            for i in agent.values.intention_log
                if i != 0 n_intentions[i] += 1 end
            end
            intention_weights = map(n -> calculate_intention_weight(n, agent), n_intentions)
            final_posteriors = [posteriors[i] * intention_weights[neighbours[i]] for i in 1:length(posteriors)]
            target = neighbours[argmax(final_posteriors)]
            enqueue!(agent.action_queue, MoveToAction(target))
            enqueue!(agent.outbox, ArrivedAtNodeMessageSEBS(agent, nothing, (agent.graph_position, target)))
        end

        if agent.graph_position isa Int64
            agent.values.departed_time = agent.world_state_belief.time
        end
    end
    
end

function calculate_gain(node::Int64, agent::AgentState)
    # Only valid for 1-hop
    distance = agent.world_state_belief.adj[agent.graph_position, node]
    # distance = ceil(agent.world_state_belief.adj[agent.graph_position, node])
    return agent.values.idleness_log[node] / distance
end

function calculate_posterior(gain::Float64, agent::AgentState)
    g1 = agent.values.sebs_gains[1]
    g2 = agent.values.sebs_gains[2]

    if gain >= g2
        return 1.0
    else
        return g1 * 2.7183^((gain/g2) * log(1/g1))
    end
end

function calculate_intention_weight(n_intentions::Int64, agent::AgentState)

    n_agents = agent.values.n_agents_belief
    return 2^(n_agents - n_intentions)/(2^n_agents - 1)
end

"""
    make_decisions_ER!(agent::AgentState)

does ER
"""
function make_decisions_ER!(agent::AgentState)

    # If no action in progress, select node to move towards
    if isempty(agent.action_queue)

        t = agent.world_state_belief.time

        # Trim visit intention queues to current time

        for q in agent.values.projected_node_visit_times
            while !isempty(q)
                if peek(q)[1] < t
                    dequeue!(q)
                else
                    break
                end
            end
        end

        neighbours = get_neighbours(agent.graph_position, agent.world_state_belief, true)

        if length(neighbours) == 1
            enqueue!(agent.outbox, ArrivedAtNodeMessageSPNS(agent, nothing, neighbours[1]))
            enqueue!(agent.outbox, 
                GoingToMessageER(agent, 
                                 nothing, 
                                 (neighbours[1], t + ceil(agent.world_state_belief.adj[agent.graph_position, neighbours[1]]))
                                )
            )
            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
        elseif !isa(agent.graph_position, Int64)
            # Catch the potential problem of an agent needing a new action
            # while midway between two nodes (not covered by algo) - 
            # solution to this is just to pick one
            enqueue!(agent.action_queue, MoveToAction(neighbours[1]))
        else
            # Do ER
            utilities = map(n -> er_utility(n, agent), neighbours)
            target = neighbours[argmax(utilities)]

            enqueue!(agent.action_queue, MoveToAction(target))
            enqueue!(agent.outbox, ArrivedAtNodeMessageSPNS(agent, nothing, target))
            enqueue!(agent.outbox, 
                GoingToMessageER(agent, 
                                nothing, 
                                (target, t + ceil(agent.world_state_belief.adj[agent.graph_position, target]))
                                )
            )
        end

        if agent.graph_position isa Int64
            agent.values.departed_time = agent.world_state_belief.time
        end
    end
    
end

function er_utility(node::Int64, agent::AgentState)

    idleness = agent.values.idleness_log[node]
    distance = agent.world_state_belief.adj[agent.graph_position, node]
    expected_visits = agent.values.projected_node_visit_times[node]
    time = agent.world_state_belief.time

    t_next = time + distance
    t_expected = time - idleness

    vt = [v[1] for v in expected_visits]

    for t in vt
        if t >= time
            t_expected = t
            break
        end
    end

    return abs(t_next - t_expected) / distance
end

function make_decisions_RHAUM!(agent::AgentState)
    pass
end