module MessagePasser

import ..Types: AgentState
import ..Utils: pos_distance
using DataStructures

"""
    pass_messages!(agents::Array{AgentState, 1})

Take messages from agent outboxes and place them in the inboxes of their intended targets, within
communication range
"""
function pass_messages!(agents::Array{AgentState, 1})
    for agent in agents
        while !isempty(agent.outbox)
            message = dequeue!(agent.outbox)
            targets::Array{Int64, 1} = isnothing(message.targets) ? collect(1:length(agents)) : message.targets
            for id in targets
                if pos_distance(agent.position, agents[id].position) <= agent.comm_range && id != agent.id
                    enqueue!(agents[id].inbox, message)               
                end
            end
        end
    end
end

end