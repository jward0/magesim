module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, IntendedPathMessage, ArrivedAtNodeMessage
import ..AgentDynamics: calculate_next_position

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
        # path_utilities = [calculate_path_utility(agent.world_state_belief.time, agent.values.utility_horizon, agent.values.node_idleness_log, p, agent.values.projected_node_visit_times) for p in possible_paths]
        path_utilities = [calculate_path_utility(agent, agent.world_state_belief.time, agent.values.utility_horizon, agent.values.node_idleness_log, p, projected_node_visit_times) for p in possible_paths]
        
        selected_path = possible_paths[argmax(path_utilities)]
        adjusted_path = deepcopy(selected_path)
        # Override to make receding horizon
        # for i in 1:length(adjusted_path)
        for i in 2:2
            adjusted_path[i] = adjusted_path[i] .+ (0.0, agent.world_state_belief.time)
            enqueue!(agent.action_queue, MoveToAction(adjusted_path[i][1]))
        end
        enqueue!(agent.outbox, ArrivedAtNodeMessage(agent, nothing, agent.graph_position))
        # enqueue!(agent.outbox, IntendedPathMessage(agent, nothing, [adjusted_path[2]]))
        enqueue!(agent.outbox, IntendedPathMessage(agent, nothing, adjusted_path))
    end
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

        for visit in full_visits
            if visit <= t && visit > alpha
                alpha = visit
            elseif visit >= t && visit < horizon
                beta = visit
                break
            end
        end

        # temp_visit_times[n][t] = t
        # Log self visit
        # Todo: expensive
        push!(self_visit_times[n], t)

        node_utility = (t - alpha) * (beta - t)
        path_utility += node_utility

        residual_time = horizon - v[2]
    end


    path_gps = path_utility / (horizon - residual_time)

    return path_utility / (horizon - residual_time)

end

end