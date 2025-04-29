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

    # Assemble projected visit times from announced paths
    projected_node_visit_times::Vector{Vector{Float64}} = [[] for _ in 1:agent.world_state_belief.n_nodes]

    for path in agent.values.other_agent_announced_paths
        for n in path
            push!(projected_node_visit_times[n[1]], n[2]) 
        end
    end

    sort!.(projected_node_visit_times)

    # Trim projected visits to current timestep

    # TODO: this can be done better
    for q in projected_node_visit_times # agent.values.projected_node_visit_times
        while !isempty(q)
            if q[1] < agent.world_state_belief.time
                popfirst!(q)
            else
                break
            end
        end
    end

    if isempty(agent.action_queue)

        best_path, reward = best_path_astar(astar_heuristic, agent, projected_node_visit_times)
        target = best_path[2][1]
        enqueue!(agent.action_queue, MoveToAction(target))

        enqueue!(agent.outbox, ArrivedAtNodeMessageSPNS(agent, nothing, agent.graph_position))
        enqueue!(agent.outbox, IntendedPathMessageRHAUM(agent, nothing, best_path))

        if agent.graph_position isa Int64
            agent.values.departed_time = agent.world_state_belief.time
        end
    end
end

function best_path_astar(h::Function, agent::AgentState, projected_node_visit_times::Vector{Vector{Float64}})

    start_time = agent.world_state_belief.time
    horizon_length = agent.values.utility_horizon
    idlenesses = agent.values.idleness_log

    adj = ceil.(copy(agent.world_state_belief.adj))
    n_nodes = size(adj)[1]
    end_time = start_time + horizon_length

    open_set = PriorityQueue()

    best_path = []
    best_path_reward = 0.0

    # each entry in open_set has form {"path": Vector{Tuple{Int64, Float64}}, "r" : Float64}
    # priority is then -(r + h(t, horizon, idlenesses, adj))
    enqueue!(open_set, Dict([("path", [(agent.graph_position, start_time)]), ("r", 0.0)]), 0.0)

    while !isempty(open_set)

        current = dequeue!(open_set)
        at_node = current["path"][end][1]
        t = current["path"][end][2]
        r = current["r"]

        # this is faster than utils.get_neighbours for no dummy nodes
        # also checks for timelimit breaking
        neighbours = [i for i in 1:n_nodes if adj[at_node, i] != 0 && t+adj[at_node, i] <= start_time+horizon_length]

        # Checks termination - goal is reached when no more steps can be taken in time limit
        if isempty(neighbours)
            return current["path"], current["r"]
        end

        for target in neighbours

            w = adj[at_node, target]
            visit_t = t + w

            # Get self visits
            self_visits = [n[2] for n in current["path"] if n[1] == target]

            # Calculate step reward
            real_reward = r + step_reward(start_time, end_time, t, idlenesses[target], w, self_visits, projected_node_visit_times[target])
            heuristic_reward = h(target, start_time, end_time, visit_t, idlenesses, adj, agent.world_state_belief.paths.dists)
            enqueue!(open_set, 
                Dict([("path", [current["path"]; [(target, visit_t)]]), 
                      ("r", real_reward)
                      ]), 
                -(real_reward + heuristic_reward))
        end
    end
end

function step_reward(start_time::Float64, end_time::Float64, current_time::Float64, idleness::Float64, weight::Float64, self_visits::Vector{Float64}, other_visits::Vector{Float64})

    # remaining_horizon = end_time - (current_time + weight)
    horizon = end_time - start_time

    alpha = current_time - idleness
    beta = end_time
    arrival_time = current_time + weight

    visits = sort([self_visits; other_visits])

    for visit in visits
        if visit <= arrival_time && visit > alpha
            alpha = visit
        # elseif visit >= arrival_time && visit < beta
        #     beta = visit
        end
    end

    raw_reward = (arrival_time - alpha) * (beta - arrival_time)
    discount_factor = astar_discount(start_time, arrival_time, end_time)

    return raw_reward * discount_factor
end

function astar_heuristic(start_node::Int64, start_time::Float64, end_time::Float64, current_time::Float64, idlenesses::Vector{Float64}, adj::Matrix{Float64}, fw_dists::Matrix{Float64})

    remaining_horizon = end_time - current_time
    horizon = end_time - start_time

    discount_window = sum([astar_discount(start_time, ts, end_time) for ts in current_time:end_time])

    n = size(adj)[1]

    edge_rewards_per_second = [0.0]

    # Limits search to arcs between nodes that are reachable in the remaining time
    reachable_nodes = [i for i in 1:n if fw_dists[start_node, i] <= horizon]
    push!(reachable_nodes, start_node)
    
    # It's this or a horrible one-liner (or I have to think a bit harder, and it's Friday afternoon)
    for a in 1:length(reachable_nodes)
        for b in 1:a
            i = reachable_nodes[a]
            j = reachable_nodes[b]
            if adj[i, j] != 0
                reward_per_second = (max(idlenesses[i], idlenesses[j]) + adj[i, j]) * (remaining_horizon - adj[i, j]) / adj[i, j]
                push!(edge_rewards_per_second, reward_per_second)
            end
        end
    end

    return discount_window * maximum(edge_rewards_per_second)

end

function astar_discount(start_time::Float64, arrival_time::Float64, end_time::Float64)
    return 0.95 ^ (arrival_time - start_time) # x^n
end

function DTAP_utility(agent::AgentState, target::Int64)

    idleness_term = agent.values.idleness_log[target]
    travel_term = get_distances(agent.graph_position, agent.position, agent.world_state_belief)[target]
    distance_from_start_term = agent.world_state_belief.paths.dists[agent.values.dtap_start, target]

    return sum(agent.values.dtap_utility_gains .* [idleness_term, travel_term, distance_from_start_term])
end

function DTAP_calculate_bid(agent::AgentState, target::Int64)

    central_node = 0
    central_node_cost = ∞

    for task in agent.values.dtap_agent_tasks
        cost = 0
        for task_ in agent.values.dtap_agent_tasks
            cost += agent.world_state_belief.paths.dists[task, task_]
        end
        if cost < central_node_cost
            central_node = task
            central_node_cost = cost
        end
    end

    # Important error catch for when agents have no tasks allocated
    # if central_node == 0
    #     return 0
    # end

    travel_cost = agent.world_state_belief.paths.dists[central_node, target]

    return travel_cost * length(agent.values.dtap_agent_tasks)
end

function make_decisions_DTAP!(agent::AgentState, agents::Vector{AgentState})

    timeout = 1

    if isempty(agent.action_queue)

        # node_utilities = [DTAP_utility(agent, target) for target in agent.values.dtap_available_tasks]

        # Do not attempt to move to own node
        deleteat!(agent.values.dtap_available_tasks, findall(x->x==agent.graph_position, agent.values.dtap_available_tasks))

        while true

            node_utilities = [DTAP_utility(agent, target) for target in agent.values.dtap_available_tasks]
            target = agent.values.dtap_available_tasks[argmax(node_utilities)]
            deleteat!(agent.values.dtap_available_tasks, findall(x->x==target, agent.values.dtap_available_tasks))

            # If fewer than 2 tasks allocated to agent, force
            if length(agent.values.dtap_agent_tasks) < 2
                agent.values.dtap_available_tasks = [i for i in 1:agent.world_state_belief.n_nodes]
                push!(agent.values.dtap_agent_tasks, target)
                enqueue!(agent.action_queue, MoveToAction(target))
                break               
            end

            # Spoof the bid collecting process by exposing all agents
            bids = [DTAP_calculate_bid(a, target) for a in agents]

            message_failed = false

            for i in 1:length(agents)
                if rand() < 1 - (1 - agent.values.comm_failure)^2 # One or both messages failed
                    bids[i] =  ∞   
                    message_failed = true                
                end
            end
            # Provisional - have to think about how appropriate this is
            if message_failed
                enqueue!(agent.action_queue, WaitAction(timeout))
            end

            # push!(agents[argmin(bids)].values.dtap_agent_tasks, target)

            deleteat!(agent.values.dtap_agent_tasks, findall(x->x==target, agent.values.dtap_agent_tasks))

            if argmin(bids) == agent.id
                agent.values.dtap_available_tasks = [i for i in 1:agent.world_state_belief.n_nodes]
                push!(agent.values.dtap_agent_tasks, target)
                enqueue!(agent.action_queue, MoveToAction(target))
                break
            end

            if isempty(agent.values.dtap_available_tasks)
                agent.values.dtap_available_tasks = [i for i in 1:agent.world_state_belief.n_nodes]
            end
        end

        enqueue!(agent.outbox, IdlenessLogMessage(agent, nothing, agent.values.idleness_log))
    end
end