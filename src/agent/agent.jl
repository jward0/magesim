module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, ArrivedAtNodeMessageSEBS, ArrivedAtNodeMessageSPNS, GoingToMessageER, ObservedWeightMessage, IdlenessLogMessage, PriorityMessage, PosMessage, GoingToMessage, IntendedPathMessageRHAUM
import ..AgentDynamics: calculate_next_position
import ..Utils: get_neighbours, pos_distance, get_distances
# import ..Strategies: make_decisions_SPNS!, make_decisions_SEBS!

using Accessors
using DataStructures
using Graphs, SimpleWeightedGraphs, SparseArrays, LinearAlgebra
using LinearAlgebra
using Infinity
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

    assume_symmetry = true

    t = agent.world_state_belief.time

    # delta rule
    if !decay
        delta = 0.2
        old_w = agent.values.effective_adj[edge...]
        new_w = old_w + delta*(w - old_w)
        agent.values.effective_adj[edge...] = new_w
        if assume_symmetry
            agent.values.effective_adj[reverse(edge)...] = new_w
        end
    end

    # decay rule
    if decay
        agent.values.effective_adj[edge...] = w
        agent.values.last_edge_visits[edge...] = t
        if assume_symmetry
            agent.values.effective_adj[reverse(edge)...] = w
            agent.values.last_edge_visits[reverse(edge)...] = t
        end
    end

end

function update_effective_adj_decay!(agent::AgentState, visited_edge::Tuple{Int64, Int64}, observed_w::Float64)

    decay = true
    assume_symmetry = true

    # ~~~ delta rule
    if !decay
        update_communicated_weights_decay!(agent, visited_edge, observed_w)    
    end

    # ~~~ decay rule
    if decay

        # SIMPLE MONITORING (A=0) HAS DELTA = 1.0
        # NORMAL DECAY RULE HAS 0.975

        # ORIGINAL DECAY
        # if agent.values.dyn_mode == "simple"
        #     decay = 1.0
        # else
        #     decay = 0.975
        # end

        # mask = findall(iszero, agent.values.effective_adj)

        # mean_w = mean(agent.values.effective_adj[findall(!iszero, agent.values.effective_adj)])
        # agent.values.effective_adj = (decay .* (agent.values.effective_adj .- mean_w)) .+ mean_w

        # agent.values.effective_adj[mask] .= 0.0

        # # TIME-DEPENDENT DECAY
        if agent.values.dyn_mode == "simple"
            factor = 1.0
        else
            # factor = 0.99925
            factor = 0.9975
        
            time_deltas = agent.world_state_belief.time .- agent.values.last_edge_visits
            # required for proper iterative decay
            effective_time_deltas = min.(time_deltas, observed_w)
            decays = factor .^ effective_time_deltas

            mask = findall(iszero, agent.values.effective_adj)

            mean_w = mean(agent.values.effective_adj[findall(!iszero, agent.values.effective_adj)])
            agent.values.effective_adj = (decays .* (agent.values.effective_adj .- mean_w))  .+ mean_w
            # agent.values.effective_adj = (decay .* (agent.values.effective_adj .- mean_w)) .+ mean_w
            
            agent.values.effective_adj[mask] .= 0.0
        end

        agent.values.effective_adj[visited_edge...] = observed_w
        agent.values.last_edge_visits[visited_edge...] = agent.world_state_belief.time
        if assume_symmetry
            agent.values.effective_adj[reverse(visited_edge)...] = observed_w
            agent.values.last_edge_visits[reverse(visited_edge)...] = agent.world_state_belief.time
        end

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

        if message isa ArrivedAtNodeMessageSEBS
            n = message.message[1]
            # +/-1 here to offset messages being sent on the other side of the idleness increment
            agent.values.idleness_log[n] = 1.0
            agent.values.intention_log[message.source] = message.message[2]
        elseif message isa IdlenessLogMessage
            # Min pool observed idleness with idleness from message
            agent.values.idleness_log = min.(agent.values.idleness_log, message.message)
        elseif message isa ArrivedAtNodeMessageSPNS
            agent.values.idleness_log[message.message] = 1.0
        elseif message isa GoingToMessage
            agent.values.other_targets[message.source] = message.message
        elseif message isa GoingToMessageER
            target = message.message[1]
            arrival_time = message.message[2]
            agent.values.intention_log[message.source] = target
            agent.values.projected_node_visit_times[target][arrival_time] = arrival_time
        elseif message isa IntendedPathMessageRHAUM
            agent.values.other_agent_announced_paths[message.source] = message.message
        elseif message isa ObservedWeightMessage
            ((src, dst), (ts, w)) = message.message
            update_communicated_weights_decay!(agent, (src, dst), w)
        end
    end

    # Upon arrival at a node:
    if agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes
        agent.values.idleness_log[agent.graph_position] = 0.0

        agent.values.last_last_visited = copy(agent.values.last_visited)
        agent.values.last_visited = agent.graph_position

        # # Update observed weights log
        t = convert(Float64, agent.world_state_belief.time)
        if t > 0.0
            src = agent.values.last_last_visited
            dst = agent.values.last_visited
            enqueue!(agent.outbox, ObservedWeightMessage(agent, nothing, ((dst, src), (t, t - agent.values.departed_time))))
            if src != dst
                update_effective_adj_decay!(agent, (src, dst), t - agent.values.departed_time)
            end
        end
    end

end

function make_decisions!(agent::AgentState, agents::Vector{AgentState})

    if agent.values.dyn_mode == "perfect"
        tp = agent.world_state_belief.temporal_profiles[floor(Integer, agent.world_state_belief.time)+1]
        new_effective_adj = ceil.(agent.world_state_belief.adj ./ tp)
        
        new_effective_adj[isnan.(new_effective_adj)] .= 0.0

        # BRISTOL ONLY
        # new_effective_adj = agent.values.secret_knowledge[floor(Integer, agent.world_state_belief.time)+1]
        
    elseif agent.values.dyn_mode == "active" || agent.values.dyn_mode == "simple" || agent.values.dyn_mode in ["1", "2", "3", "4", "5"] 
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

    # println("+++++++++++++++++++++++++++++++++++++++")
    # println(agent.world_state_belief.adj)
    # println(agent.values.secret_knowledge[floor(Integer, agent.world_state_belief.time)+1])
    do_flag = isempty(agent.action_queue)
    t = @elapsed begin
        if agent.values.strategy == "SEBS"
            make_decisions_SEBS!(agent)
        elseif agent.values.strategy == "SPNS"
            make_decisions_SPNS!(agent, "full")
        elseif agent.values.strategy == "MNS"
            make_decisions_SPNS!(agent, "minimal")
        elseif agent.values.strategy == "ER"
            make_decisions_ER!(agent)
        elseif agent.values.strategy == "RHAUM"
            make_decisions_RHAUM!(agent)
        elseif agent.values.strategy == "DTAP"
            make_decisions_DTAP!(agent, agents)
        elseif agent.values.strategy == "CRA"
            make_decisions_CRA!(agent)
        elseif agent.values.strategy == "RAND"
            make_decisions_RAND!(agent)
        elseif agent.values.strategy == "visitmaxing"
            visit_maximisation!(agent)
        end
    end
    # if agent.id == 1 && do_flag
    #     println(t)
    # end
end

end