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
            else
                break
            end
        end
    end

    if isempty(agent.action_queue)

        best_path, reward = best_path_astar(astar_heuristic, agent, projected_node_visit_times)
        target = best_path[2][1]
        enqueue!(agent.action_queue, MoveToAction(target))

        enqueue!(agent.outbox, ArrivedAtNodeMessage(agent, nothing, agent.graph_position))
        enqueue!(agent.outbox, IntendedPathMessage(agent, nothing, best_path))
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
            heuristic_reward = h(target, start_time, end_time, visit_t, idlenesses, adj, agent.world_state_belief.paths.dists)
            enqueue!(open_set, 
                Dict([("path", [current["path"]; [(target, visit_t)]]), 
                      ("r", real_reward)
                      ]), 
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
    return (end_time - arrival_time) / (end_time - start_time)
end

end