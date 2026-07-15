# Quantum trajectories and trajectory statistics

Source: [`quantum_trajectories.jl`](quantum_trajectories.jl)

## Model

Six initially excited qubits decay through independent local channels. The
same Lindblad equation is solved in two ways: deterministic density-matrix
evolution and a Monte Carlo ensemble of quantum trajectories.

## Solution

The script runs 500 trajectories with `dt = 0.005` and a fixed seed. The
trajectory API returns ensemble-averaged states and supports online statistics
of observables, here the excitation number. The result is compared with
deterministic PI density-operator evolution.

The deterministic model is prepared once with
`compile(model; backend=:matrixfree)`. The typed high-level command

```julia
deterministic = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times, steps_per_interval=20,
)
```

returns a `DynamicsResult` whose entries are `PIState` objects at the same 11
times used by the trajectory ensemble. The script compares every averaged
state with the corresponding deterministic state. A prepared
`CollectiveObservablePlan` evaluates the final excited fraction for both
solutions, and `diagnostics` validates the final ensemble state.

Local jumps may move a pure trajectory between total-spin sectors; the PI
trajectory implementation retains the sector information without constructing
full many-body wave functions. Use the threaded option for independent paths
when reproducible scheduling is not required.

## Run and convergence

```sh
julia --project=. examples/quantum_trajectories.jl
```

Statistical error decreases only as the inverse square root of the trajectory
count. Converge both `dt` (bias) and the number of paths (sampling error), and
report the supplied standard errors with trajectory estimates. The script
prints the total jump count, largest density-state discrepancy, final traces
and excited fractions, mean and Fano factor of the jump count, and the final
observable standard error. It does not assert a fixed Monte Carlo error because
that error is statistical even with a reproducible seed.
