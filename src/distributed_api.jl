"""
    distributed_quantum_trajectories(model, rho0, times, n;
        workers=Distributed.workers(), seed=0, dt, trajectory_keywords...)

Generate an ordered ensemble of independent PI quantum-jump trajectories on
multiple Julia worker processes. Loading the `Distributed` standard library
activates the implementation. The master creates the same trajectory-indexed
random streams as [`quantum_trajectories`](@ref), partitions them into
deterministic contiguous chunks, and restores global trajectory order after
the workers finish.

Each worker lowers the supplied `PIModel` once and reuses one
[`TrajectoryWorkspace`](@ref) for its chunk. The model, initial state, time
grid, parameters, and callable rates must therefore be serializable and every
worker must be started with an environment containing a compatible package
version. Full state histories cross the process boundary; use the threaded or
adaptive ensemble APIs when only online statistics are required.
"""
function distributed_quantum_trajectories end

"""
    distributed_diffusive_trajectories(model, rho0, times, monitors, n;
        workers=Distributed.workers(), seed=0, dt, parameters=nothing,
        observables=nothing, save_states=true)

Generate an ordered ensemble of diffusive PI trajectories on multiple Julia
worker processes. The optional implementation is activated by loading the
`Distributed` standard library. Random streams are derived from the global
trajectory index, so results agree with [`diffusive_trajectories`](@ref) for
the same model, controls, seed, and numerical environment.

Every worker constructs one [`DiffusiveBatchPlan`](@ref) and reuses one
[`DiffusiveWorkspace`](@ref). The model, monitors, observables, initial state,
parameters, and time grid must be serializable. Saved states, measurement
records, and requested observable histories are transferred back to the
master; set `save_states=false` when only records or online observable values
are needed.
"""
function distributed_diffusive_trajectories end
