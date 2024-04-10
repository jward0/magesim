module Agent

import ..Types: AgentState, WorldState, Position, AbstractAction, WaitAction, MoveToAction, StepTowardsAction, StringMessage, TagMessage, ArrivedAtNodeMessage
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
    agent.values.idleness_log = [i + 1.0 for i in agent.values.idleness_log]
    if agent.graph_position isa Int64 && agent.graph_position <= world.n_nodes
        agent.values.idleness_log[agent.graph_position] = 0.0
    end
    # enqueue!(agent.outbox, StringMessage(agent, nothing, string(agent.id)))
end


"""
    make_decisions!(agent::AgentState)

Read messages and modify agent's action queue based on received messages, world state belief, and 
internal values
"""
function make_decisions!(agent::AgentState)
    patrol_method::String = "SEBS"

    while !isempty(agent.inbox)
        message = dequeue!(agent.inbox)

        # Message handling
        if message isa TagMessage #tag stuff
            # if agent.id == 1
            #     println(message.reward_message)
            #     println(message.unreward_message)
            # end

            for tag in message.reward_message # first row, rewarding tags
                if !(tag in agent.values.belief_nodes_rewarding)
                    push!(agent.values.belief_nodes_rewarding)
                end
                if tag in agent.values.belief_nodes_unrewarding
                    agent.values.belief_nodes_unrewarding = filter(x -> x!=tag, agent.values.belief_nodes_unrewarding)
                end
            end

            for tag in message.unreward_message # second row, unrewarding tags
                if !(tag in agent.values.belief_nodes_unrewarding)
                    push!(agent.values.belief_nodes_unrewarding)
                end
                if tag in agent.values.belief_nodes_rewarding
                    agent.values.belief_nodes_rewarding = filter(x -> x!=tag, agent.values.belief_nodes_rewarding)
                end
            end
        end

        if message isa ArrivedAtNodeMessage #SEBS
            agent.values.idleness_log[message.message[1]] = 0.0
            agent.values.intention_log[message.source] = message.message[2]
        end
    end

    # Do stuff



    if !isnothing(agent.world_state_belief) #Give next action
        if isempty(agent.action_queue) #if action queue is empty 
            if agent.graph_position isa Int64 # if at a node 
                # if agent.id == 1 
                #     print("At node ")
                #     print(agent.graph_position)
                #     println(" -----")
                # end
                tag_values = agent.world_state_belief.nodes[agent.graph_position].values.is_reward
                for (index,tag_value) in enumerate(tag_values)
                    tag = ((agent.graph_position - 1) * 4) + index
                    if tag in agent.values.belief_nodes_rewarding || tag in agent.values.belief_nodes_unrewarding #if seen before
                        if tag in agent.values.belief_nodes_rewarding
                            # if agent.id == 1 
                            #     println("Scanning tag believed rewarding")
                            # end
                            scan_tag!(agent, tag, tag_value)
                        else
                            chance_to_scan = 1 - agent.values.li
                            if rand() <= chance_to_scan #scan
                                # if agent.id == 1 
                                #     println("Scanning tag believed unrewarding")
                                # end
                                scan_tag!(agent, tag, tag_value)
                            # else
                            #     if agent.id == 1 
                            #         println("Not scanning tag believed unrewarding")
                            #     end
                            end
                        end
                    else #if not seen before
                        # if agent.id == 1 
                        #     println("Scanning tag not seen before")
                        # end
                        scan_tag!(agent, tag, tag_value)
                    end
                end

                # println("========")
                # println(agent.id)
                # println(agent.values.belief_nodes_rewarding)
                # println(agent.values.belief_nodes_unrewarding)
                # println("========")

                if !isempty(agent.values.recent_scans_rewarding) || !isempty(agent.values.recent_scans_unrewarding) #if theres data to share 
                    enqueue!(agent.outbox, TagMessage(agent, nothing, agent.values.recent_scans_rewarding, agent.values.recent_scans_unrewarding))
                    agent.values.recent_scans_rewarding = [] #sent message now can clear recent scans and continue onto next node
                    agent.values.recent_scans_unrewarding = []
                end

                #  GO TO NEXT NODE -------------------------
                if patrol_method == "CGG"
                    if agent.graph_position == 25
                        next_node = 1
                        #println("ended route going back to 1")
                        enqueue!(agent.action_queue, MoveToAction(next_node))
                    else 
                        next_node = agent.graph_position + 1
                        enqueue!(agent.action_queue, MoveToAction(next_node))
                    end


                elseif patrol_method == "SEBS"
                    
                    neighbours = get_neighbours(agent.graph_position, agent.world_state_belief, true)

                    if length(neighbours) == 1
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
                        enqueue!(agent.outbox, ArrivedAtNodeMessage(agent, nothing, (agent.graph_position, target)))
                    end

                end
            end
        end

        #rand(1:agent.world_state_belief.n_nodes)
    end
end


function scan_tag!(agent::AgentState, tag::Int64, tag_value::Bool)
    if tag_value #rewarding
        if tag in agent.values.belief_nodes_unrewarding
            agent.values.belief_nodes_unrewarding = filter(x -> x!=tag, agent.values.belief_nodes_unrewarding)
        end
        if !(tag in agent.values.belief_nodes_rewarding)
            push!(agent.values.belief_nodes_rewarding,tag)
        end
        agent.values.cumulative_reward += 1
        push!(agent.values.recent_scans_rewarding,tag)
    else #not rewarding
        if tag in agent.values.belief_nodes_rewarding
            agent.values.belief_nodes_rewarding = filter(x -> x!=tag, agent.values.belief_nodes_rewarding)
        end
        if !(tag in agent.values.belief_nodes_unrewarding)
            push!(agent.values.belief_nodes_unrewarding,tag) 
        end
        push!(agent.values.recent_scans_unrewarding,tag)
    end
    enqueue!(agent.action_queue, WaitAction(10)) #scanned then wait to simulate time taken
end


function calculate_gain(node::Int64, agent::AgentState)
    distance = agent.world_state_belief.paths.dists[agent.graph_position, node]
    return agent.values.idleness_log[node] / distance
end

function calculate_posterior(gain::Float64, agent::AgentState)
    g1 = 0.1
    g2 = 100

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

end
