## MAGESim: Multi-Robot Patrolling

An example implementation of the multi-robot patrolling problem, using the SEBS algorithm (Distributed multi-robot patrol: A scalable and fault-tolerant framework, Portugal and Rocha, 2013)

### Type modifications
`NodeValues`: given fields `idleness::Float64`
`AgentValues`: given fields `intention_log::Array{Integer, 1}` and `idleness_log::Array{Float, 1}`

### Message-passing
Upon arriving at a node, agents publish an `ArrivedAtNodeMessage` containing the node arrived at and the next target node. This is passed to all other agents.

### Logging