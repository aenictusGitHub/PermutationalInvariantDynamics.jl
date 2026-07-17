# Architecture and efficient workflows

## Data flow

The package separates declarative physics, immutable prepared data, and
mutable numerical scratch:

```text
PIBasis → PIModel → compile → CompiledPIModel
                              ├─ LiouvillianPlan
                              ├─ sparse or matrix-free backend
                              └─ LiouvillianWorkspace per task

(N, d, terms) → MeanFieldPlan → one-site nonlinear RHS
                                └─ MeanFieldWorkspace per task

invariant PIModel → PopulationPlan → reduced sparse action
                                   └─ PopulationWorkspace per task

(PIBasis..., finite factors...) → CompositePIBasis
                               → CompositeSuperoperator
                                 └─ CompositeSuperoperatorWorkspace per task
```

`PIBasis` stores only polynomial-size Schur data. `PIModel` is an immutable
tuple of physical terms. `LiouvillianPlan` lowers fixed operators to prepared
sector blocks and local gain maps. It never contains mutable application
scratch. A `LiouvillianWorkspace` contains the multiplication matrices needed
for each Schur sector and, for an in-place operator schedule, the evaluated
operator and dynamic Schur/gain buffers. It can be reused by dynamics and
Krylov algorithms.

Composite coordinates keep each PI factor in the same equation-(7)
normalization and each finite factor in column-major matrix-unit coordinates.
The first factor varies fastest. Tensor-mode kernels apply a sum of factorized
maps with two reusable composite buffers and one small fibre buffer per active
factor; they do not construct the corresponding global Kronecker matrix.

## Recommended commands

```julia
prepared = compile(model)  # backend=:auto

sol = solve_dynamics(prepared, rho0, (0.0, 10.0); saveat=0.1)
rho_ss = stationary_state(prepared)
values = liouvillian_spectrum(prepared; target=:largest_real, nev=6)

diagnostics(rho_ss)
recommend_solver(model; task=:steady_state)
```

When only observables are needed, replace the first call by

```julia
series = solve_dynamics(
    prepared, rho0, (0.0, 10.0);
    saveat=0.1,
    observables=(Jz=sz/2,),
    save_states=false,
)
```

This retains one evolving PI vector and the requested scalar series. The same
keywords on `quantum_trajectories` produce online Welford statistics without
sampled `PIState` histories. See [Streaming output](streaming_output.md).

Autonomous dynamical correlations have their own prepared flow:

```text
compiled Liouvillian + A,B[,R] → CorrelationPlan
                               → CorrelationWorkspace
                               ├─ sequential RK4 QRT samples
                               └─ shifted-GMRES connected spectra
```

The readout convention is explicitly `tr(A*exp(L*tau)*(B*rho*R))`; it differs
from the adjoint implicit in a Hilbert--Schmidt `expectation` call. See
[Quantum regression and optical correlations](correlations.md).

These are high-level commands. Lower-level research control remains available
through `liouvillian`, `apply!`, `steady_state`,
`krylov_liouvillian_spectrum`, `harmonic_arnoldi_spectrum`,
`implicitly_restarted_arnoldi_spectrum`, and
`jacobi_davidson_spectrum`. The latter combines hard invariant-subspace
locking with optionally preconditioned matrix-free correction solves.
See [API tiers and prepared analysis](api_tiers.md) for the supported
high-level, advanced, and experimental surfaces.

## Certified population-only workflow

When a model preserves the GT-basis diagonal subspace, construct one
`PopulationPlan` and inspect its stored report:

```julia
population_plan = PopulationPlan(model)
@assert population_plan.invariance.invariant === true
p0 = diagonal_populations(rho0)
solution = solve_populations(
    population_plan, p0, (0.0, 10.0); saveat=0.1)
```

The default certificate is exact at the working scalar precision. Explicit
nonzero tolerances opt into approximate projection; they are never selected
automatically from a generator scale. Compile-time reverse lookup has one
entry per population and is discarded after construction. The retained plan
contains only the basis and reduced sparse kernels, so it does not retain a
full-PI-coordinate lookup. Reuse `PopulationWorkspace` for in-place scans and
one workspace per concurrent task.

At present, plan construction forms the standalone `sqrt(f^nu)` population
coordinate conversion and the parent PI trace functional in the chosen real
type. It therefore raises if that factor is outside the finite scalar range;
use wider coefficients for such a plan. This bound is stricter than
`diagonal_populations` and `state_from_populations`, which use prepared scaled
products and can still handle finite population values without materializing
the standalone factor.

## Concurrency

For parallel applications, construct one workspace per task or thread:

```julia
prepared = compile(model; backend=:matrixfree)
work = [LiouvillianWorkspace(prepared) for _ in 1:Threads.nthreads()]

Threads.@threads for column in axes(X,2)
    apply!(view(Y,:,column), prepared, view(X,:,column), 0.0, nothing,
           work[Threads.threadid()])
end
```

The compatibility `mul!` and `MatrixFreeLiouvillian.action!` routes are
synchronized and safe, but concurrent calls serialize. Explicit workspaces
avoid that lock.

The same ownership rule applies to `SymmetryProjectorWorkspace` and
`ArnoldiWorkspace`, and to `MeanFieldWorkspace`: workspaces are mutable and
must not be shared by concurrent calls.

## Mean-field prediction beside exact PI dynamics

`MeanFieldPlan(model)` reuses the physical term conventions but does not
compile Schur geometry. It evaluates the product-state closure
`Tr_{2:N} L[sigma^otimes N]` with storage set by the local dimension and the
largest supported body order. The direct `MeanFieldPlan(N, d, terms)`
constructor therefore remains practical even when constructing `PIBasis(N,d)`
would be unnecessary.

Use `limit=:finite` when comparing the derivative with exact finite-`N` PI
dynamics. Use `limit=:thermodynamic` only for explicitly size-scaled rates; it
keeps leading combinatorial contributions and drops subleading collective
self terms. The package does not infer or insert Kac normalization. See
[Mean-field predictions](meanfield.md) for equations and validation examples.

## Static and driven generators

`isautonomous(model)` reports whether every operator is fixed and every rate
is constant. Stationary-state, spectrum, gap, adjoint, and ordinary `mul!`
operations reject driven generators. To analyse an instantaneous generator,
make the choice explicit:

```julia
Lt = freeze(model; time=1.25, parameters=params)
rho_t_stationary = stationary_state(Lt; basis=model.basis)
```

Use `solve_dynamics`, `dynamics_problem`, or Floquet routines for actual
time-dependent evolution.

### In-place operator schedules

When the operator itself changes with time, wrap a fixed-shape prototype and
an in-place callback in `InPlaceTimeOperator`:

```julia
sm = ComplexF64[0 1; 0 0]
sx = ComplexF64[0 1; 1 0]

drive = InPlaceTimeOperator(sm, (destination, t, parameters) -> begin
    @. destination = cos(t) * sm + parameters.mix * sin(t) * sx
    nothing
end)

model = PIModel(basis, [LocalJump(drive; rate=(t, p) -> p.gamma)])
prepared = compile(model; backend=:matrixfree)
work = LiouvillianWorkspace(prepared)
apply!(du, prepared, u, t, parameters, work)
```

The destination is reset to the prototype before every callback. The callback
must mutate it without changing its shape, basis, or scalar type, and return
the destination or `nothing`. Callback allocations remain the callback
author's responsibility. Prepared schedules are implemented for every built-in
term family, including direct PI and Appendix-D Hamiltonians and jumps.
Dynamic collective blocks, quadratic jump products, and all local-gain cross
terms are rebuilt in the task-local workspace without assembling a PI
Liouvillian. A dynamic local p-body jump retains one contraction buffer per
compatible pair of Appendix-D removal paths and applies each gain block as
`C * X * C'` with one reusable matrix-multiplication buffer. It does not store
the quadratic-size PI-coordinate `I/J/value` representation. Inspect the
factorized workspace memory for large qudit irreps just as for the static
p-body geometry. Fixed-precision scheduled one-body blocks are rejected at
compile time once large-`N` branch cancellation requires wider geometry; widen
the `InPlaceTimeOperator` prototype. Scheduled direct-PI blocks use prepared
fused inverse-multiplicity scales and keep the ordinary direct-divisor loop for
small representable multiplicities.

A batched `apply!` or `apply_adjoint!` evaluates every scheduled operator once
and reuses the result for all columns. Separate workspaces are required for
concurrent calls. A plain operator-valued `Function` remains available as an
allocating compatibility path and constructs an instantaneous sparse
generator. Dissipative rates must evaluate to real numbers; negative real
rates remain accepted for deterministic time-local evolution.

Krylov and Floquet storage is precision-derived rather than unconditionally
`ComplexF64`. Fully `Float32` inputs retain half-width complex buffers; wider
seeds, targets, periods, or time origins promote compatible materialized and
custom operators. A compiled matrix-free plan owns fixed-precision block
scratch, so wider inputs are rejected and must be handled by compiling the
model at the wider precision. Incompatible explicit workspaces also raise.
This saves RAM without silently narrowing mixed-precision research inputs.

## Prepared observables and reductions

Repeated observable evaluation should prepare collective blocks once:

```julia
geometry = OneBodyGeometry(basis)
Jx = CollectiveObservablePlan(basis, sx/2; cache=geometry)
Jz = CollectiveObservablePlan(basis, sz/2; cache=geometry)
means = collective_expectation.(states, Ref(Jz))
F = qfim(last(states), [sx/2, sz/2]; plans=(Jx, Jz))
```

The geometry is shared only during construction; each observable plan retains
one matrix per Schur sector. For a fixed bipartition,
`ReductionPlan(basis,k)` caches qubit recouplers or qudit LR/subduction
intertwiners:

```julia
reduction = ReductionPlan(basis, k)
work = ReductionWorkspace(reduction, rho)
rhoA = reduced_state(rho, k; plan=reduction, workspace=work)
purityA = reduced_purity(rho, k; plan=reduction, workspace=work)
neg = negativity(rho, k; plan=reduction, workspace=work)
```

Both plan types require the exact basis object used at construction. Observable
plans are compact and usually worthwhile for state scans. Qudit reduction
plans may retain many dense intertwiners and their construction performs
sparse weight-restricted nullspaces, so they are intended for repeated
fixed-`k` analysis rather than every one-off bipartition. Plans contain no mutable scratch and may be
shared read-only. A `ReductionWorkspace` belongs to one exact plan and reuses
the product-block application buffers; it must not be shared concurrently. It
also converts the compact real qubit recouplers once to its complex working
type so repeated multiplication stays allocation-light on every supported
Julia release. Already type-matched qudit intertwiners are shared read-only.
This conversion increases retained qubit workspace memory in exchange for a
homogeneous hot loop. Use `reduced_state!` when the output `PIState` should
also be reused.

For repeated one-body marginals, `one_body_rdm(rho; cache=geometry)` contracts
all matrix units in one pass instead of preparing `d^2` independent collective
operators.

## Schur-block diagnostics

`schur_block_structure(rho)` measures the retained state/operator block in
each partition. For a PI superoperator it returns a matrix whose row is the
output sector and whose column is the input sector. Sparse matrices are
scanned directly. Matrix-free generators are probed one PI coordinate at a
time with reusable scratch, so the calculation avoids materializing the
global PI Liouvillian but still costs one application per input coordinate.

Rendering is separate: `visualize_schur_blocks(structure)` creates a compact
object with text and SVG displays, and `save_schur_block_visualization` writes
that SVG without adding a plotting dependency. Reuse the numerical structure
when experimenting with titles, normalization, or colour scales. See
[Schur-block visualization](schur_visualization.md) for representation and
norm conventions.

## Spectral diagnostics

Density and complex spectral data use the same analysis/presentation split.
`pi_density_spectrum` diagonalizes physical Schur blocks and retains one value
per irrep eigenvector with its exact symmetric-group degeneracy;
`visualize_density_spectrum` renders that compressed result without repeating
the diagonalization or constructing a `d^N` list.
`liouvillian_spectrum_data` normalizes complete or selected results into a
`ComplexSpectrum` without changing their order or repeated roots.
`floquet_spectrum_data` accepts an already converged one-period map or a
precomputed multiplier/exponent vector. `visualize_spectrum` then renders the
stored values repeatedly without invoking a solver.

Density plots show raw eigenvalue versus compressed mode rank and colour points
by Schur sector. Liouvillian plots show `real(lambda)=0`; multiplier plots show the unit circle;
principal-branch exponent plots show `real(xi)=0` and `imag(xi)=±π/T`.
Classifications are tolerance-based presentation metadata and never clip or
move a point. Complete dense spectra and selected ordinary/harmonic/IRAM/JD windows
remain explicitly distinguished. See
[Spectral visualization](spectral_visualization.md) for conventions and
efficient reuse.

## Backend selection and memory

`compile(...; backend=:auto, memory_budget=...)` keeps driven models
matrix-free and uses a conservative sparse-storage bound for autonomous
models. Inspect `prepared.estimates`; the choice is never hidden.

Additional estimates are available through:

- `pi_dimension`
- `estimate_state_bytes`
- `estimate_basis_bytes`
- `estimate_liouvillian_bytes`
- `estimate_geometry_bytes`
- `estimate_solver_bytes`
- `recommend_solver`

`basis_summary` and `estimate_state_bytes` use the retained coordinate
dimension, including for a sector-restricted basis; the legacy `(N,d)`
`estimate_memory` function intentionally describes a complete basis.
`estimate_geometry_bytes` separately reports conservative retained and peak
setup bounds for the sparse one-body transition geometry. Qudit estimates
count every content-compatible parent in multiplicity-bearing GT weight
spaces before bounding contraction tuples; they do not use a
multiplicity-free approximation. Krylov estimates
are accumulated as exact `BigInt` byte counts so the estimator itself cannot
overflow on a large problem. Fixed-size isbits scalar arrays retain exact
inline `sizeof(T)` accounting. Heap-backed `BigFloat` and
`Complex{BigFloat}` arrays instead use a conservative retained-storage bound
at the explicit `bigfloat_precision` keyword (the active precision by
default); this is an upper estimate, not an ABI promise. GMRES also accounts
for its real residual/history array using the real component type of `T`.
Other heap-backed scalar types use a padded zero-value sample and report
`scalar_storage_estimate=:sample_based_retained_estimate`; their arbitrary
value payload is not claimed to have a type-level worst-case bound.

For a `PIModel` or `CompiledPIModel`, `recommend_solver` includes one-body
geometry only when its terms require that lowering. Bare bases, states,
operators, and lowered plans lack term provenance, so they retain the
conservative allowance and report
`geometry_requirement=:conservative_unknown`. The result includes these
geometry and solver-vector contributions in `estimated_peak_bytes` and
reports whether that bound fits the requested memory budget.

The estimates distinguish retained storage from temporary setup allocations.
Always benchmark representative values of `N`, `d`, body order, and sector
content before a large scan.

## Validation contract

`validate_state` rejects trace, Hermiticity, or positivity violations without
normalizing, clipping, or symmetrizing the input. `state_diagnostics` returns
the corresponding errors and tolerances without throwing. Analysis routines
only remove roundoff-level skew-Hermitian components after validation. A
validated spectral routine may also project accepted roundoff eigenvalues to
numerical rank zero where a square root, logarithm, or support projector
requires it. This is temporary internal arithmetic: it does not repair the
input, change returned eigenvalues, or authorize clipping a genuinely invalid
state. Endpoint correction in fidelity is similarly bounded and throws when
the discrepancy is larger than documented roundoff.
