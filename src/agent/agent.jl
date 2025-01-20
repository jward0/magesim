module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, IntendedPathMessage, ArrivedAtNodeMessage
import ..AgentDynamics: calculate_next_position
import ..Utils: get_neighbours

using DataStructures

"""
    agent_step!(agent::AgentState, world::WorldState)

Select and perform an action and update agent position and stat accordingly
"""
function agent_step!(agent::AgentState, world::WorldState, blocked_pos::Array{Position, 1})

    # Wait if no other action found
    if isempty(agent.action_queue)
        enqueue!(agent.action_queue, WaitAction())
    end

    # Do action from queue

    action = first(agent.action_queue)

    if action isa WaitAction
        # Do nothing for one timestep and then decrement wait duration
        new_pos = agent.position
        new_graph_pos = agent.graph_position
        action.duration -= 1
        if action.duration <= 0
            action_done = true
        else
            action_done = false
        end
    elseif action isa MoveToAction
        # Move towards target and do not pop action from queue until target reached
        new_pos, new_graph_pos, action_done = calculate_next_position(agent, action.target, world, blocked_pos)
    elseif action isa StepTowardsAction
        # Take one step towards target 
        new_pos, new_graph_pos, _ = calculate_next_position(agent, action.target, world, blocked_pos)
        action_done = true
    else
        error("Error: no behaviour found for action of type $(nameof(typeof(action)))")
    end

    agent.position = new_pos
    agent.graph_position = new_graph_pos

    if action_done
        dequeue!(agent.action_queue)
    end

end

"""
    observe_world!(agent::AgentState, world::WorldState)

Extract an agent's observation from the true world state and update the agent's belief of the
world state, and generate messages to send to other agents
"""
function observe_world!(agent::AgentState, world::WorldState)
    agent.world_state_belief = world
    agent.values.node_idleness_log .+= 1.0

    # Read messages
    while !isempty(agent.inbox)
        message = dequeue!(agent.inbox)

        if message isa IntendedPathMessage
            n = message.source
            # Log intended paths
            agent.values.other_agent_announced_paths[message.source] = message.message
            # for v in message.message
                # Insert projected visit time into relevant priority queue, with priority equal to projected visit time
                # agent.values.projected_node_visit_times[v[1]][v[2]] = v[2]                
            # end
        elseif message isa ArrivedAtNodeMessage
            agent.values.node_idleness_log[message.message] = 1.0
        end
    end

    # Upon arrival at (real) node
    if agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes
        agent.values.node_idleness_log[agent.graph_position] = 0.0
    end    

    # enqueue!(agent.outbox, StringMessage(agent, nothing, string(agent.id)))
end

"""
    make_decisions!(agent::AgentState)

Read messages and modify agent's action queue based on received messages, world state belief, and 
internal values
"""
function make_decisions!(agent::AgentState)

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
            # if peek(q)[1] < agent.world_state_belief.time
                # dequeue!(q)
            else
                break
            end
        end
    end

    if isempty(agent.action_queue)
        possible_paths = agent.world_state_belief.weight_limited_paths[agent.graph_position]
        # # path_utilities = [calculate_path_utility(agent.world_state_belief.time, agent.values.utility_horizon, agent.values.node_idleness_log, p, agent.values.projected_node_visit_times) for p in possible_paths]
        path_utilities = [calculate_path_utility(agent, agent.world_state_belief.time, agent.values.utility_horizon, agent.values.node_idleness_log, p, projected_node_visit_times) for p in possible_paths]
        selected_path = possible_paths[argmax(path_utilities)]
        adjusted_path = deepcopy(selected_path)
        # Override to make receding horizon
        # for i in 1:length(adjusted_path)
        # for i in 2:2
            # adjusted_path[i] = adjusted_path[i] .+ (0.0, agent.world_state_belief.time)
            # enqueue!(agent.action_queue, MoveToAction(adjusted_path[i][1]))
        # end

        best_path, reward = best_path_astar(astar_heuristic, agent, projected_node_visit_times)
        # println([b[1] for b in best_path])
        # First entry is current position
        target = best_path[2][1]
        enqueue!(agent.action_queue, MoveToAction(target))

        enqueue!(agent.outbox, ArrivedAtNodeMessage(agent, nothing, agent.graph_position))
        # enqueue!(agent.outbox, IntendedPathMessage(agent, nothing, [adjusted_path[2]]))
        # enqueue!(agent.outbox, IntendedPathMessage(agent, nothing, adjusted_path))
        enqueue!(agent.outbox, IntendedPathMessage(agent, nothing, best_path))

        println("+++++++++++++++++++++++")
        println(possible_paths)
        println(path_utilities)
        println(adjusted_path)
        println(maximum(path_utilities))
        println("===")
        println(best_path)
        println(reward)
    end
end

function best_path_astar(h::Function, agent::AgentState, projected_node_visit_times::Vector{Vector{Float64}})

    start_time = agent.world_state_belief.time
    horizon_length = agent.values.utility_horizon
    idlenesses = agent.values.node_idleness_log

    adj = ceil.(copy(agent.world_state_belief.adj))
    n_nodes = size(adj)[1]
    end_time = start_time + horizon_length

    open_set = PriorityQueue()

    best_path = []
    best_path_reward = 0.0

    # Remove nodes set for visit by other agents from consideration - best way to handle this
    for i in 1:n_nodes
        if length(projected_node_visit_times[i]) > 0
            adj[:, i] .= 0
        end
    end
    # If no route from start exists due to removals, ignore the most immediate ones
    if sum(adj[agent.graph_position, :]) == 0
        # Don't need to copy here due to taking a view on original adj
        adj[agent.graph_position, :] = agent.world_state_belief.adj[agent.graph_position, :]
    end

    # each entry in open_set has form {"path": Vector{Tuple{Int64, Float64}}, "r" : Float64}
    # priority is then -(r + h(t, horizon, idlenesses, adj))
    
    enqueue!(open_set, Dict([("path", [(agent.graph_position, start_time)]), ("r", 0.0)]), 0.0)

    while !isempty(open_set)

        current = dequeue!(open_set)
        at_node = current["path"][end][1]
        t = current["path"][end][2]
        r = current["r"]
        # t = current["t"] 

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
            real_reward = r + step_reward(start_time, end_time, t, idlenesses[target], w, self_visits)
            heuristic_reward = h(start_time, end_time, t, idlenesses, adj)

            enqueue!(open_set, 
                Dict([("path", [current["path"]; [(target, visit_t)]]), 
                      ("r", real_reward), 
                      ("interfered", false)]), 
                -(real_reward + heuristic_reward))
        end
    end
end

function step_reward(start_time::Float64, end_time::Float64, current_time::Float64, idleness::Float64, weight::Float64, self_visits::Vector{Float64})

    remaining_horizon = end_time - (current_time + weight)
    horizon = end_time - start_time

    alpha = current_time - idleness
    arrival_time = current_time + weight

    for visit in self_visits
        if visit <= arrival_time && visit > alpha
            alpha = visit
        end
    end

    raw_reward = (arrival_time - alpha) * remaining_horizon
    # discount_factor = remaining_horizon / horizon
    discount_factor = astar_discount(start_time, arrival_time, end_time)

    return raw_reward * discount_factor
end

function astar_heuristic(start_time::Float64, end_time::Float64, current_time::Float64, idlenesses::Vector{Float64}, adj::Matrix{Float64})

    remaining_horizon = end_time - current_time
    horizon = end_time - start_time

    # discount_window = (1/horizon) * sum([end_time-t for t in current_time:end_time])
    discount_window = sum([astar_discount(start_time, ts, end_time) for ts in current_time:end_time])

    n = size(adj)[1]

    edge_rewards_per_second = []

    # It's this or a horrible one-liner (or I have to think a bit harder, and it's Friday afternoon)
    for i in 1:n
        for j in 1:i
            if adj[i, j] != 0
                reward_per_second = (max(idlenesses[i], idlenesses[j]) + adj[i, j]) * (remaining_horizon - adj[i, j]) / adj[i, j]
                push!(edge_rewards_per_second, reward_per_second)
            end
        end
    end

    return discount_window * maximum(edge_rewards_per_second)

end

function astar_discount(start_time::Float64, arrival_time::Float64, end_time::Float64)
    # return 2.71828 ^ -(arrival_time - start_time)
    return (end_time - arrival_time) / (end_time - start_time)
    # return 1.0
end

# TODO: messy that this takes agent and also a load of stuff that gets pulled from agent
function calculate_path_utility(agent::AgentState, current_time::Float64, horizon::Float64, node_idleness_log::Vector{Float64}, path::Vector{Tuple{Int64, Float64}}, projected_node_visit_times::Vector{Vector{Float64}})
    
    # temp_visit_times = deepcopy(projected_node_visit_times)
    path_utility = 0.0

    # Todo: expensive
    self_visit_times = [[] for _ in 1:length(projected_node_visit_times)]
    residual_time = horizon

    for v in path[2:end]

        n = v[1]
        t = v[2] + current_time

        if v[2] > horizon
            break
        end

        alpha = current_time - node_idleness_log[n]
        beta = current_time + horizon
        
        # Get projected visit time to node from other agents
        # A bit expensive
        # visits = [vt[1] for vt in projected_node_visit_times[n]]
        visits = [vt for vt in projected_node_visit_times[n]]

        # Append to prior visit times from self on path
        # A bit expensive
        full_visits = sort!([visits; self_visit_times[n]])

        interference = false

        for visit in visits
            if visit <= t && visit > alpha
                alpha = visit
                interference = true
            elseif visit >= t && visit < horizon
                beta = visit
                interference = true
                break
            end
        end

        for visit in self_visit_times[n]
            if visit <= t && visit > alpha
                alpha = visit
            elseif visit >= t && visit < horizon && visit < beta
                beta = visit
                break
            end
        end
        # for visit in full_visits
        #     if visit <= t && visit > alpha
        #         alpha = visit
        #     elseif visit >= t && visit < horizon
        #         beta = visit
        #         break
        #     end
        # end

        # temp_visit_times[n][t] = t
        # Log self visit
        # Todo: expensive

        if interference == true
            break
        end

        push!(self_visit_times[n], t)
        residual_time = horizon - v[2]

        node_utility = (t - alpha) * (beta - t)
        # Preferentially weighting sooner nodes 
        # HORIZON SCALING APPLIED HERE
        path_utility += node_utility  * (residual_time/horizon)  

    end


    path_gps = path_utility / (horizon - residual_time)

    if horizon == residual_time
        return 0.0
    else
        # RESIDUAL SCALING APPLIED HERE
        return path_utility  # / (horizon - residual_time)
    end

end

end