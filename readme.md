## MAGESim

## Modification guide
### Agents
#### Internal state
The `AgentValues` type, in `src/utils/types.jl`, is a catch-all composite type for user-defined agent values beyond what exists in the `AgentState` type. This includes any internal states, or additional observations as discussed under the Observation heading.
#### Observation
`observe_world!()` in `src/agent/agent.jl` governs how agents generate a believed world state from the true world state and the state of the agent. This world state belief exists as `AgentState.world_state_belief::Union{WorldState, Nothing}`. Any observations or beliefs that an agent may make beyond what exists in `WorldState` can be included in the `AgentValues` type. This function should typically result in the agent updating its `world_state_belief` field, and its `AgentValues` field if needed. It may also result in enqueuing a message to its outbox, as discussed under the Message passing heading. If line-of-sight checking is desired during observation, `check_los()` in `src/utils/los_checker.jl` provides this.
#### Decision making
`make_decisions!()` in `src/agent/agent.jl` governs how agents select actions from their own state, observations, and beliefs, and any messages received from other agents. This function should always result in an action being enqueued to the agent's action queue. It may also result in enqueuing a message to its outbox, as discussed under the Message passing heading.
#### Action types 
The action types in `src/utils/types.jl` are used to determine agent behaviour during calls to the `agent_step!()` function. The three default actions are `WaitAction`, which does nothing for one timestep and is subsequently dequeued, `MoveToAction`, which takes one step towards the target and is not dequeued until the target is reached, and `StepTowardsAction`, which takes one step towards the target and is subsequently dequeued. Note that actions as described here are what governs the movement of the agents in the world - decision making and message passing are not actions and occur independently.
#### Message passing
`pass_messages!()` in `src/agent/message_passer.jl` governs distribution of messages from agent outboxes to agent inboxes. By default, this is called once per timestep (in `step_agents!()` in `src/agent/agent_handler.jl`) after all agents have called `make_decision!()`, `agent_step!()` and `observe_world!()`. If desired, it can also be called inbetween these functions. Agents can add messages to their outboxes at any point, but they will only be passed to their targets the next time `pass_messages!()` is called. If desired, agents can be given limited broadcast ranges, and line-of-sight checking can be enabled with an optional argument.
#### Message types
Message types in `src/utils/types.jl` govern what messages are available to be passed between agents. The default provided `StringMessage` provides a useful template, as custom message types are expected to contain `source` and `targets` fields identical to those in `StringMessage`, but the `message` field can be any type or composite type desired by the user.
### Environments
#### Node states
#### Mutating the map
#### Reward allocation
### Logging
### Map creation

## Interface with PettingZoo ParallelEnv API
### Limitations
When using the provided wrapper to create a PettingZoo ParallelEnv object that wraps the simulator code, there are several restrictions that are not otherwise present that must be observed.
* `NodeValues` can only have fields of types `Int`, `Float`, `Bool`, `String`, or 1-d `Array` of these types. Default values of `Int` or `Array{Int}` must refer to the maximum possible value of the field, and no negative values are allowed. As such, it is recommended to only use these types to represent enums, and to use `Float` or `Array{Float}` to represent values that may be negative or may have unknown upper bounds. Default values of `String` or `Array{String}` must be strings of the integer values of the maximum allowable length (in characters) of the strings.
* Currently, all agents must have the same observation space (but may make different observations within that space).
* The only action permitted is `StepTowardsAction`, as this is the most straightforwardly compatible with the standard RL flow of observe -> select action -> step environment -> observe.
* Manual control of speedup is disabled, the simulator will run as fast as possible.
* Multi-threading of agents is currently not supported.
### Action spaces
Currently, the only action compatible with the interface is the `StepTowardsAction`. The corresponding action space is a `Discrete` space with a size equal to the number of nodes in the map graph.
### Observation spaces
Currently, the observation space of all agents is that agent's belief of the world state, existing in `Agent.world_state_belief`. This consists of the map, the agent's location, and all node values. The map is represented as a `Graph` space observing node positions, edges, and optional edge weights (custom edge weights (ie. other than Cartesian distance) not currently implemented). The agent's location is represented as a `Box` in R2. The representation of node values varies depending on the fields of the `NodeValues` struct, but will be a composite space of n instances of the space generated in the wrapper code to represent an individual instance of `NodeValues`.