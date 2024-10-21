module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, IdlenessLogMessage, PriorityMessage, PosMessage, GoingToMessage
import ..AgentDynamics: calculate_next_position
import ..Utils: get_neighbours, pos_distance, get_distances
import ..Strategies: make_decisions_SPNS!, make_decisions_SEBS!

using Accessors
using DataStructures
using Graphs, SimpleWeightedGraphs, LinearAlgebra
using LinearAlgebra
using Statistics

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

function update_weight_logs!(agent::AgentState, src::Int64, dst::Int64, ts::Real, w::Float64)
    if !haskey(agent.values.observed_weights_log, (src, dst))
        agent.values.observed_weights_log[(src, dst)] = []
    end
    if !haskey(agent.values.observed_weights_log, (dst, src))
        agent.values.observed_weights_log[(dst, src)] = []
    end

    push!(agent.values.observed_weights_log[(src, dst)], (ts, w))
    push!(agent.values.observed_weights_log[(dst, src)], (ts, w))
    agent.world_state_belief.adj[src, dst] = w
    agent.world_state_belief.adj[dst, src] = w
end

"""
    observe_world!(agent::AgentState, world::WorldState)

Extract an agent's observation from the true world state and update the agent's belief of the
world state, and generate messages to send to other agents
"""
function observe_world!(agent::AgentState, world::WorldState)
    agent.world_state_belief = world
    agent.values.idleness_log .+= 1.0

    # Upon arrival at a node:
    if isempty(agent.action_queue) && agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes
        agent.values.idleness_log[agent.graph_position] = 0.0
        agent.values.last_last_visited = copy(agent.values.last_visited)
        agent.values.last_visited = agent.graph_position

        # Update observed weights log
        t = convert(Float64, agent.world_state_belief.time)
        src = agent.values.last_last_visited
        dst = agent.values.last_visited

        # update_weight_logs!(agent, src, dst, t, t - agent.values.departed_time)
    end
    agent.values.other_targets_freshness .+= 1.0
    for (ndx, f) in enumerate(agent.values.other_targets_freshness)
        if f > 60
            agent.values.other_targets[ndx] = 0
            agent.values.priority_log[ndx, :] .*= 0
        end
    end

    adj_belief = world.adj ./ world.temporal_profiles[floor(Integer, world.time)+1]
    adj_belief[isnan.(adj_belief)] .= 0.0
    adj_belief = ceil.(adj_belief)
    wsb = agent.world_state_belief
    @reset wsb.adj=adj_belief
    agent.world_state_belief = wsb

    # println("___________________________________________________________-_")
    # println(agent.world_state_belief.adj)

    # @reset agent.world_state_belief.adj=adj_belief

    # if world.time > 1 && world.time % 1000000 == 0
    #     agent.world_state_belief.adj = deepcopy(world.adj ./ world.temporal_profiles[floor(Integer, world.time)+1])
    #     agent.world_state_belief.adj[isnan.(agent.world_state_belief.adj)] .= 0.0
    #     # println("__________________________________")
    #     # println(world.adj)
    #     # println(world.temporal_profiles[floor(Integer, world.time)+1])
    #     # println(agent.world_state_belief.adj)
    # end

    # agent.world_state_belief.adj = agent.values.effective_adjacency_matrix
    
    # println(world.temporal_profiles[floor(Integer, world.time)+1])
    
end

function make_decisions!(agent::AgentState)
    if agent.values.strategy == "SEBS"
        make_decisions_SEBS!(agent)
    elseif agent.values.strategy == "SPNS"
        make_decisions_SPNS!(agent)
    end
end

end