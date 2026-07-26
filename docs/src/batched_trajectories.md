# Matrix-RHS conditional trajectories

`quantum_trajectories` is the safe high-level ensemble interface. It assigns a
random stream to each trajectory index, dynamically schedules independent
paths, and lets each path shorten its own fixed step when its jump intensity
would exceed `max_jump_probability`.

The conditional density matrices of many paths can nevertheless be propagated
more efficiently when they currently share the same physical time and step.
`BatchedConditionalPlan` and `BatchedConditionalWorkspace` expose that
matrix-RHS kernel without changing the stochastic algorithm.

```julia
using PermutationalInvariantDynamics

basis = PIBasis(10, 2)
spin = spin_matrices()
model = PIModel(basis, (
    LocalHamiltonian(spin.jx; rate=0.2),
    LocalJump(spin.jm; rate=0.7),
))
rho0 = iid_pure_state(basis, ComplexF64[0, 1])

trajectory_plan = TrajectoryPlan(model)
plan = BatchedConditionalPlan(trajectory_plan)
work = BatchedConditionalWorkspace(plan, rho0, 16)

# Each column is one normalized PI density operator.
states = repeat(reshape(rho0.data, :, 1), 1, 16)
batched_conditional_rk4!(states, plan, 0.0, 0.002, nothing, work)
```

The workspace has immutable capacity. Passing more than 16 columns raises
instead of growing its matrix buffers. Construction is protected by the same
512 MiB default memory budget as the high-level solvers.

## Intensities and jumps

Channel intensities are evaluated for the whole cohort:

```julia
rates = zeros(length(trajectory_plan.jumps), size(states, 2))
batched_channel_intensities!(
    rates, plan, states, 0.002, nothing, work)
```

Keep one RNG per trajectory index. Regrouping paths must move the RNG together
with its state:

```julia
rngs = batched_trajectory_rngs(1234, size(states, 2))
channels = zeros(Int, size(states, 2))
batched_sample_jumps!(channels, rates, 0.002, rngs)
batched_apply_jumps!(
    states, channels, plan, 0.002, nothing, work)
```

`channels[j] == 0` means that path did not jump. Columns selecting the same
nonzero channel are gathered and passed to one matrix-RHS gain kernel.

## Why this is not a replacement for `quantum_trajectories`

After jumps, path intensities differ. The scalar fixed-step algorithm caps
each next step independently:

```math
h_j = \min\left(\Delta t,\,
\frac{p_{\max}}{\lambda_j}\right).
```

Forcing all paths to use the smallest `h_j` changes their RK grid and random
draw times. The package therefore does not silently do this. Batch only paths
whose existing scheduler selected the same step, or use
`quantum_trajectories` when exact high-level scheduling and output collection
are wanted.

The matrix-RHS action and RK4 implementation are checked against repeated
scalar trajectory workspaces in `test/test_batched_trajectories.jl`. Run the
focused performance comparison with:

```sh
julia --project=benchmark benchmark/batched_trajectories.jl
```

## API

```@docs
BatchedConditionalPlan
BatchedConditionalWorkspace
batched_conditional_action!
batched_conditional_rk4!
batched_channel_intensities!
batched_apply_jumps!
batched_trajectory_rngs
batched_sample_jumps!
```
