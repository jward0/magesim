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
function pass_messages!(agents::Array{AgentState, 1}, world::WorldState, check_los_flag::Bool=false)
    for agent in agents
        while !isempty(agent.outbox)
            message = dequeue!(agent.outbox)
            targets::Array{Int64, 1} = isnothing(message.targets) ? collect(1:length(agents)) : message.targets
            for id in targets
                if check_los_flag
                    if check_los(world.obstacle_map, 
                                 world.scale_factor, 
                                 agent.position, 
                                 agents[id].position, 
                                 agent.comm_range)
                        
                        enqueue!(agent[id].inbox, message)
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