module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, ArrivedAtNodeMessageSEBS, ArrivedAtNodeMessageSPNS, ObservedWeightMessage, IdlenessLogMessage, PriorityMessage, PosMessage, GoingToMessage
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

function update_weight_logs!(agent::AgentState, src::Int64, dst::Int64, ts::Real, w::Float64)
    if !haskey(agent.values.observed_weights_log, (src, dst))
        agent.values.observed_weights_log[(src, dst)] = []
    end
    if !haskey(agent.values.observed_weights_log, (dst, src))
        agent.values.observed_weights_log[(dst, src)] = []
    end

    push!(agent.values.observed_weights_log[(src, dst)], [ts, w])

    # If this is not the most recent observation, sort
    if length(agent.values.observed_weights_log[(src, dst)]) > 1 && ts < agent.values.observed_weights_log[(src, dst)][end-1][1]
        sort!(agent.values.observed_weights_log[(src, dst)], by = x -> x[1])
    end
    # agent.world_state_belief.adj[src, dst] = w
    # agent.world_state_belief.adj[dst, src] = w

    agent.values.observed_weights_log[(dst, src)] = agent.values.observed_weights_log[(src, dst)]

end

function update_communicated_weights_decay!(agent::AgentState, edge::Tuple{Int64, Int64}, w::Float64)

    # decay stuff

    agent.values.effective_adj[edge...] = w
    agent.values.effective_adj[reverse(edge)...] = w

end

"""
    log_likelihood(params::Tuple{Float64, Float64}, x::Vector{Float64}, delta_t::Vector{Float64})

    Log-likelihood fitting observed edge weights at certain times to constant value c plus geometric brownian motion
    process parameterised by process stddev sigma
"""
function log_likelihood(params::Vector{Float64}, x::Vector{Float64}, delta_t::Vector{Float64})

    c, sigma = params
    sigma = max(sigma, 0)

    last_x = x[1:end-1]
    x = x[2:end]

    # println("________________________")
    # println(x.-c)
    # println(delta_t)
    # println(sigma)

    first_part = sum(log.(1/((x.-c) .* sqrt.(delta_t) .* sigma .* sqrt(2*3.14159))))
    second_part = sum(((log.((x.-c) ./ (last_x.-c)) .+ (0.5 .* sigma^2 .* delta_t)).^2) ./ (2 .* sigma^2 .* delta_t))

    l = first_part .- second_part

    return -l * sigma

end

"""
    estimate_process_parameters(weights_log::Dict{Tuple{Int64, Int64}, Array{Tuple{Real, Float64}, 1}})

    Estimates parameters of Geometric Brownian Motion process used as a model of edge weight. Outputs "c", estimate
    of minimum edge weight (if recoverable), and "sigma", process standard deviation
"""
function estimate_process_parameters(edge_weights_log::Vector{Vector{Float64}})

    # return 0.0, 0.0

    observed_x::Vector{Float64} = [e[2] for e in edge_weights_log]
    observed_delta_t::Vector{Float64} = [edge_weights_log[i+1][1] - edge_weights_log[i][1] for i in 1:length(observed_x)-1]
    initial_guess = [minimum(observed_x) - 0.1, 0.1]
    lb = [0, 1e-6]
    ub = [minimum(observed_x) - 1e-3, Inf]
    result = optimize(p -> log_likelihood(p, observed_x, observed_delta_t), lb, ub, initial_guess)
    c, sigma = Optim.minimizer(result)

    return c, sigma
end

"""
    calculate_effective_weight(c::Float64, sigma::Float64, last_value::Float64, delta_t::Float64, alpha::Float64)

    Given estimates of process parameters, last observed value, time since last value, and explore/exploit parameter
    calculates an effective value to use - more exploration (ie. negative alpha) makes uncertain edges look more 
    attractive, more exploitation (ie. positive alpha) makes uncertain edges look less attractive. "Uncertain" here
    is based on the variance of the expected value of an edge weight based on estimates of process params. 
"""
function calculate_effective_weight(c::Float64, sigma::Float64, last_value::Float64, delta_t::Float64, alpha::Float64)

    # println("$c, $sigma")

    edge_stddev = (last_value - c)*sqrt(exp(sigma^2 * delta_t) - 1)

    if isnan(edge_stddev) || isinf(edge_stddev)
        edge_stddev = 0
    end

    w = max(c, last_value + alpha*edge_stddev)
    w = min(w, 2*last_value)

    if isnan(w)
        println("AAAAAAAAAAAAAAAAAA")
        println(last_value)
        println(c)
        println(sigma)
        println(delta_t)
        println(edge_stddev)
    end

    # println("$last_value, $w")
    return w
end

"""
    update_effective_adj!(agent::AgentState)

    Re-calculates agent.values.effective_adj for new timestep, based on process parameter estimates
"""
function update_effective_adj!(agent::AgentState, dt::Float64)

    edge_locs::Tuple{Vector{Int64}, Vector{Int64}, Vector{Float64}} = findnz(sparse(triu(agent.world_state_belief.adj)))

    # Override for testing
    agent.values.alpha = -0.5
    
    # Probably inefficient, can probably just go through all keys in process_parameter_estimates
    for i in 1:size(edge_locs[1])[1]

        src = edge_locs[1][i]
        dst = edge_locs[2][i]

        if haskey(agent.values.process_parameter_estimates, (src, dst))

            c, sigma, last_t = agent.values.process_parameter_estimates[(src, dst)]
            delta_t = convert(Float64, agent.world_state_belief.time) - last_t
            last_value = agent.values.observed_weights_log[(src, dst)][end][2]

            # Ceil is as we discretise by timestep
            w = calculate_effective_weight(c, sigma, last_value, delta_t, agent.values.alpha)

            agent.values.effective_adj[src, dst] = ceil(w^1.0)
            agent.values.effective_adj[dst, src] = ceil(w^1.0)
        end
    end
end


function update_effective_adj_decay!(agent::AgentState, visited_edge::Tuple{Int64, Int64}, observed_w::Float64)

    # ~~~ decay stuff
    # decay_constant = lr
    decay_constant = 0.975
    mask = findall(iszero, agent.values.effective_adj)

    # relative_obstruction = (agent.values.effective_adj .- agent.values.original_adj_belief)
    # relative_obstruction[findall(isnan, relative_obstruction)] .= 0.0


    # regressed_obstruction = (decay_constant .* (relative_obstruction .- mean(relative_obstruction))) .+ mean(relative_obstruction)

    # agent.values.effective_adj = regressed_obstruction .+ agent.values.original_adj_belief

    mean_w = mean(agent.values.effective_adj[findall(!iszero, agent.values.effective_adj)])
    agent.values.effective_adj = (decay_constant .* (agent.values.effective_adj .- mean_w)) .+ mean_w

    agent.values.effective_adj[mask] .= 0.0

    agent.values.effective_adj[visited_edge...] = observed_w
    agent.values.effective_adj[reverse(visited_edge)...] = observed_w

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
            agent.values.other_targets_freshness[message.source] = 0.0
        elseif message isa ObservedWeightMessage
            ((src, dst), (ts, w)) = message.message
            # update_weight_logs!(agent, src, dst, ts, w)
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
            # Comment these next two lines in for static last-obs tracking (and remove thing from make_decisions)
            # agent.values.effective_adj[src, dst] = t - agent.values.departed_time
            # agent.values.effective_adj[dst, src] = t - agent.values.departed_time

            # Bits start here
            
            # update_weight_logs!(agent, src, dst, t, t - agent.values.departed_time)
            """
            # Estimate process params based on new info (wants moving somewhere, probably)
            # println(agent.values.observed_weights_log)
            
            if length(agent.values.observed_weights_log[(src, dst)]) == 1
                c, sigma = (0.0, 0.0)
            else
                c, sigma = estimate_process_parameters(agent.values.observed_weights_log[(src, dst)])
            end
            # Assume symmetric
            agent.values.process_parameter_estimates[(src, dst)] = (c, sigma, t)
            agent.values.process_parameter_estimates[(dst, src)] = (c, sigma, t)
            """
            update_effective_adj_decay!(agent, (src, dst), t - agent.values.departed_time)
            
        end
        # If non-static
        # update_effective_adj!(agent, t - agent.values.departed_time)
    end

    agent.values.other_targets_freshness .+= 1.0

end

function make_decisions!(agent::AgentState)

    # If perfect
    # tp = agent.world_state_belief.temporal_profiles[floor(Integer, agent.world_state_belief.time)+1]
    # new_effective_adj = agent.world_state_belief.adj ./ tp
    # new_effective_adj[isnan.(new_effective_adj)] .= 0.0

    
    # otherwise
    new_effective_adj = agent.values.effective_adj

    wsb = agent.world_state_belief
    @reset wsb.adj=new_effective_adj
    agent.world_state_belief = wsb

    if agent.values.strategy == "SEBS"
        make_decisions_SEBS!(agent)
    elseif agent.values.strategy == "SPNS"
        make_decisions_SPNS!(agent)
    end
end

end