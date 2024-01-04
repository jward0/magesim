## MAGESim: Multi-Robot Patrolling

An example implementation of the multi-robot patrolling problem, using the SEBS algorithm (Distributed multi-robot patrol: A scalable and fault-tolerant framework, Portugal and Rocha, 2013)

Note that the purpose of this implementation is to demonstrate how MAGESim can be modified to suit multiple purposes, and as such there is no guarantee that I have implemented SEBS correctly, as I have not carried out the validation that would be required to present this in good conscience as an implementation for others to use as-is.

### Type modifications
`NodeValues`: given fields `idleness::Float64`
`AgentValues`: given fields `intention_log::Array{Integer, 1}`, `idleness_log::Array{Float, 1}`, `sebs_gains::Tuple{Float64, Float64}` and `n_agents_belief::Int64`. The values of `sebs_gains` are set in the config file and propagated via the `custom_config` variable.
New type `ArrivedAtNodeMessage` for use with SEBS algorithm

### World
Every timestep, the `NodeValues.idleness` fields of every node are incremented by 1, or set to 0 if an agent has arrived at the node.

### Agents
`make_decisions!` function now performs the SEBS algorithm. See comments in the function and the original paper for more details.

### Config
New `custom_config` field added to config, containing values propagated to `sebs_gains` field of `AgentValues`. These values are used in the SEBS algorithm.
`observe_world` now causes agents to update their belief of node idlenesses every timestep

### Message-passing
Upon arriving at a node, agents publish an `ArrivedAtNodeMessage` containing the node arrived at and the next target node. This is passed to all other agents.

### Logging
Agent xy positions are logged to `agent_positions.csv` every timestep. Node idlenesses are logged to `idleness.csv` every timestep.