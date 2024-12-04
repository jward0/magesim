module MessagePasser

import ..Types: AgentState, WorldState
import ..Utils: pos_distance
import ..LOSChecker: check_los
using DataStructures

"""
    pass_messages!(agents::Array{AgentState, 1}, world::WorldState, check_los_flag::Bool=false)

Take messages from agent outboxes and place them in the inboxes of their intended targets, within
communication range. If check_los_flag is true, checks line-of-sight against obstacle map instead
of just checking range.
"""
function pass_messages!(agents::Array{AgentState, 1}, world::WorldState)
    for agent in agents
        while !isempty(agent.outbox)
            message = dequeue!(agent.outbox)
            if rand() > agent.values.comm_failure
                targets::Array{Int64, 1} = isnothing(message.targets) ? [x for x in collect(1:length(agents)) if x != agent.id] : message.targets
                for id in targets
                    if agent.check_los_flag
                        if check_los(world.obstacle_map, 
                                    world.scale_factor, 
                                    agent.position, 
                                    agents[id].position, 
                                    agent.comm_range)
                            
                            enqueue!(agents[id].inbox, message)
                        end
                    else
                        if pos_distance(agent.position, agents[id].position) <= agent.comm_range && id != agent.id
                            enqueue!(agents[id].inbox, message)               
                        end
                    end
                end
            end
        end
    end
end

end