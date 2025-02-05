module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, ArrivedAtNodeMessageSEBS, ArrivedAtNodeMessageSPNS, GoingToMessageER, ObservedWeightMessage, IdlenessLogMessage, PriorityMessage, PosMessage, GoingToMessage
import ..AgentDynamics: calculate_next_position
import ..Utils: get_neighbours, pos_distance, get_distances
# import ..Strategies: make_decisions_SPNS!, make_decisions_SEBS!

using Accessors
using DataStructures
using Graphs, SimpleWeightedGraphs, SparseArrays, LinearAlgebra
using LinearAlgebra
using Optim
using Statistics
using Flux

include("strategies.jl")

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
        # Do nothing for one timestep
        new_pos = agent.position
        new_graph_pos = agent.graph_position
        action_done = true
    elseif action isa MoveToAction
        # Move towards target and do not pop action from queue until target reached
        new_pos, new_graph_pos, action_done = calculate_next_position(agent, action.target, world, blocked_pos)
    elseif action isa StepTowardsAction
        # Take one step towards target 
        new_pos, new_graph_pos, _ = calculate_next_position(agent, action.target, world, blocked_pos)
        action_done = true
    else
        error("Error: no behaviour found for action of type $(nameof(typeof))")
    end

    agent.position = new_pos
    agent.graph_position = new_graph_pos

    if action_done
        dequeue!(agent.action_queue)
    end

end

function update_communicated_weights_decay!(agent::AgentState, edge::Tuple{Int64, Int64}, w::Float64)

    decay = true

    # delta rule
    if !decay
        delta = 0.2
        old_w = agent.values.effective_adj[edge...]
        new_w = old_w + delta*(w - old_w)
        agent.values.effective_adj[edge...] = new_w
        agent.values.effective_adj[reverse(edge)...] = new_w
    end

    # decay rule
    if decay
        agent.values.effective_adj[edge...] = w
        agent.values.effective_adj[reverse(edge)...] = w
    end

end

function update_effective_adj_decay!(agent::AgentState, visited_edge::Tuple{Int64, Int64}, observed_w::Float64)

    decay = true

    # ~~~ delta rule
    if !decay
        update_communicated_weights_decay!(agent, visited_edge, observed_w)    
    end

    # ~~~ decay rule
    if decay

        # SIMPLE MONITORING (A=0) HAS DELTA = 1.0
        # NORMAL DECAY RULE HAS 0.975

        # decay_constant = 1.0
        decay_constant = 0.975
        mask = findall(iszero, agent.values.effective_adj)

        mean_w = mean(agent.values.effective_adj[findall(!iszero, agent.values.effective_adj)])
        agent.values.effective_adj = (decay_constant .* (agent.values.effective_adj .- mean_w)) .+ mean_w

        agent.values.effective_adj[mask] .= 0.0

        agent.values.effective_adj[visited_edge...] = observed_w
        agent.values.effective_adj[reverse(visited_edge)...] = observed_w

    end

end

"""
    observe_world!(agent::AgentState, world::WorldState)

Extract an agent's observation from the true world state and update the agent's belief of the
world state, and generate messages to send to other agents
"""
function observe_world!(agent::AgentState, world::WorldState)
    agent.world_state_belief = world
    agent.values.idleness_log .+= 1.0

    # Read ArrivedAtNodeMessages to update idleness and intention logs
    while !isempty(agent.inbox)
        message = dequeue!(agent.inbox)
        agent.values.n_messages += 1
        if message isa ArrivedAtNodeMessageSEBS
            n = message.message[1]
            # +/-1 here to offset messages being sent on the other side of the idleness increment
            agent.values.last_terminal_idlenesses[n] = copy(agent.values.idleness_log[n] - 1.0)
            agent.values.idleness_log[n] = 1.0
            agent.values.intention_log[message.source] = message.message[2]
        elseif message isa IdlenessLogMessage
            # Min pool observed idleness with idleness from message
            agent.values.idleness_log = min.(agent.values.idleness_log, message.message)
        elseif message isa ArrivedAtNodeMessageSPNS
            agent.values.idleness_log[message.message] = 0.0
        elseif message isa GoingToMessage
            agent.values.other_targets[message.source] = message.message
        elseif message isa GoingToMessageER
            target = message.message[1]
            arrival_time = message.message[2]
            agent.values.intention_log[message.source] = target
            agent.values.projected_node_visit_times[target][arrival_time] = arrival_time
        elseif message isa ObservedWeightMessage
            ((src, dst), (ts, w)) = message.message
            update_communicated_weights_decay!(agent, (src, dst), w)
        end
    end

    # Upon arrival at a node:
    if isempty(agent.action_queue) && agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes
        agent.values.last_terminal_idlenesses[agent.graph_position] = copy(agent.values.idleness_log[agent.graph_position])
        agent.values.idleness_log[agent.graph_position] = 0.0
        agent.values.last_last_visited = copy(agent.values.last_visited)
        agent.values.last_visited = agent.graph_position

        # Update observed weights log
        t = convert(Float64, agent.world_state_belief.time)
        src = agent.values.last_last_visited
        dst = agent.values.last_visited
        enqueue!(agent.outbox, ObservedWeightMessage(agent, nothing, ((dst, src), (t, t - agent.values.departed_time))))
        if src != dst
            update_effective_adj_decay!(agent, (src, dst), t - agent.values.departed_time)
            
        end
    end

end

function make_decisions!(agent::AgentState)

    if agent.values.dyn_mode == "perfect"
        tp = agent.world_state_belief.temporal_profiles[floor(Integer, agent.world_state_belief.time)+1]
        new_effective_adj = ceil.(agent.world_state_belief.adj ./ tp)
        new_effective_adj[isnan.(new_effective_adj)] .= 0.0
    elseif agent.values.dyn_mode == "active"
        new_effective_adj = agent.values.effective_adj
    end

    if agent.values.dyn_mode != "nothing"
        wsb = agent.world_state_belief
        @reset wsb.adj=new_effective_adj
        agent.world_state_belief = wsb
    end

    # If perfect
    # tp = agent.world_state_belief.temporal_profiles[floor(Integer, agent.world_state_belief.time)+1]
    # new_effective_adj = ceil.(agent.world_state_belief.adj ./ tp)
    # new_effective_adj[isnan.(new_effective_adj)] .= 0.0

    # If nothing

    # otherwise
    # new_effective_adj = agent.values.effective_adj

    if agent.values.strategy == "SEBS"
        make_decisions_SEBS!(agent)
    elseif agent.values.strategy == "SPNS"
        make_decisions_SPNS!(agent)
    elseif agent.values.strategy == "ER"
        make_decisions_ER!(agent)
    elseif agent.values.strategy == "visitmaxing"
        visit_maximisation!(agent)
    end
end

end