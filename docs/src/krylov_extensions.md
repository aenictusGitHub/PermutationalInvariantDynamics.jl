# Block, shifted, recycled, and exponential Krylov methods

These advanced solvers reuse matrix-free applications of a prepared operator
when a research calculation has more structure than one linear system. They
accept an object implementing `mul!(y, A, x)`. Their low-level mutating paths
also accept either `A(y, x)` or an allocating `A(x)` callable. None of the
methods materializes the PI Liouvillian.

Every solver returns convergence flags, residuals, iteration counts, and the
number of vector operator applications. Nonconvergence raises by default. Use
`require_convergence=false` only when the returned diagnostics will be checked
explicitly.

The workspaces reuse the dominant full-coordinate bases and residual arrays;
they do not promise a zero-allocation solve. Small projected QR/eigenvalue or
matrix-exponential problems are formed as ordinary dense arrays. Size and
reuse the workspace to bound the large memory term, and benchmark projected
work separately when the block or Krylov dimension is itself large.

## Clustered modes: thick-restarted block Arnoldi

`block_arnoldi_spectrum` advances several search directions in one operator
call. This is useful for degenerate or tightly clustered slow modes and for
sized operators with a genuine matrix--matrix action. A bare callable must be
wrapped in `MatrixFreeLiouvillian` so the solver has an explicit dimension:

```julia
workspace = BlockArnoldiWorkspace(ComplexF64, size(A, 1), 48, 4)
result = block_arnoldi_spectrum(A;
    nev=6,
    block_size=4,
    krylovdim=48,
    retained_dimension=12,
    workspace,
    vectors=true,
)
```

The total search space, rather than the number of block steps, is bounded by
`krylovdim`. Dependent directions are deflated, selected Ritz vectors are
retained across a thick restart, and every reported mode is reapplied to the
full operator before its residual is accepted. This is deliberately named a
block/thick-restarted Arnoldi method; it is not exact-shift block IRAM.
The default memory guard includes the reusable block arrays, projected dense
eigensolve, predictable restart diagnostics, returned modes, source-action
scratch, and the actual capacity of supplied workspaces. Set
`memory_budget=Inf` only as an explicit opt-out.

For a matrix-free Floquet map, pair it with a fixed-capacity forward batch
workspace so the four RK stages evaluate a drive once for the entire block:

```julia
period_work = FloquetBatchWorkspace(period_map, 4; mode=:forward)
result = block_arnoldi_spectrum(period_map;
    nev=4,
    block_size=4,
    operator_workspace=period_work,
)
```

`selected_floquet_multipliers(period_map; method=:block_arnoldi,
block_size=4)` prepares this forward workspace automatically. Use
`mode=:full` only when the same batch workspace must also apply the exact
discrete adjoint. Its capacity is fixed and an oversized block raises instead
of allocating hidden stage matrices.

## Several right-hand sides: block GMRES

Block GMRES expands all residual directions together. This can reuse common
spectral information when computing several response columns, parameter
derivatives, or trace-fixed linear corrections:

```julia
B = hcat(source_1, source_2, source_3)
workspace = BlockGMRESWorkspace(ComplexF64, length(source_1), 3, 12)
X = zeros(ComplexF64, size(B))

result = block_gmres!(X, A, B, workspace;
    maxiter=60,
    atol=1e-11,
    rtol=1e-9,
)
```

Here `block_krylovdim=12` counts block Arnoldi steps, not scalar basis
vectors. The largest stored basis has at most
`3 * (12 + 1)` columns. Dependent initial or generated directions are
deflated deterministically. A fixed left preconditioner may be passed with
`preconditioner=P`; the implementation checks both preconditioned and raw
residuals for every column.

Use the allocating wrapper for one-off work:

```julia
result = block_gmres(A, B; block_krylovdim=12)
X = result.solution
```

## Many shifts from one Arnoldi factorization

Resolvents on a frequency grid involve systems

```math
(A-\sigma_j I)x_j=b.
```

The Krylov spaces of these shifted operators coincide when the initial guess
is zero. `multishift_gmres` builds one Arnoldi factorization and solves every
small shifted least-squares problem:

```julia
shifts = im .* frequencies
result = multishift_gmres(A, source, shifts;
    krylovdim=60,
    atol=1e-11,
    rtol=1e-9,
)
responses = result.solutions
```

This implementation deliberately uses one unrestarted shared space. Generic
preconditioners and nonzero initial guesses destroy shift invariance and are
not accepted. If some shifts have not converged, increase `krylovdim` or solve
those systems separately with an appropriate preconditioner. The diagnostic
`projected_residuals` comes from the shared Arnoldi relation, while
`residuals` is recomputed in the full space.

Only an exactly zero Arnoldi remainder is treated as a happy breakdown. A
small representable remainder is retained because it can still be larger than
the requested residual tolerance. Projected solves use rank-revealing QR; a
singular or rank-deficient projected system therefore returns an honest full
residual/nonconvergence result rather than manufacturing a solution.

## Continuation with recycled GMRES

`RecycledGMRESWorkspace` retains approximate invariant directions between
solves. Before each new solve it applies the current operator to the retained
space, so the operator may change along a parameter continuation:

```julia
workspace = RecycledGMRESWorkspace(A0, 40, 8)

x = zeros(ComplexF64, size(A0, 1))
first = recycled_gmres!(x, A0, b0, workspace)

# A1 can be the matrix-free operator at the next parameter point.
fill!(x, 0)
second = recycled_gmres!(x, A1, b1, workspace)
```

The GCRO projection maintains a pair of spaces $U$ and $C$ satisfying
$M^{-1} A U=C$ with orthonormal columns of $C$. Arnoldi then acts on the
complement of $C$. Near-`target` Ritz directions from the final correction
space replace `U` for the next call. The default `target=0` is appropriate for
trace-fixed steady-state and low-frequency response systems.

Recycling is mutable algorithmic state. Use one workspace per continuation
chain and never share it concurrently. The `recycle_reset` diagnostic is true
if a changed operator made the retained image rank deficient and the solver
safely discarded it.

[`ParameterScanPlan`](@ref) uses this mechanism when configured with
`RecycledGMRESAlgorithm`; ordinary `GMRESAlgorithm` remains a state-only warm
start. Use `RecycledGMRESWorkspace` directly when the sequence consists of
custom linear systems.

## Adaptive exponential action

`krylov_expv` computes $\exp(tA)b$ directly. It restarts in time and adapts
the slice length from an augmented-Hessenberg defect estimate:

```julia
workspace = KrylovExpvWorkspace(A, 30)
result = krylov_expv(A, rho0, dt;
    workspace=workspace,
    atol=1e-11,
    rtol=1e-9,
)
rho_dt = result.value
```

This is useful for autonomous propagation and exponential-integrator
building blocks. It does not enforce trace, Hermiticity, or positivity after
the action; these properties must follow from the supplied generator and the
requested numerical tolerance. The returned `estimated_error` is the sum of
accepted local defect estimates. `reached_time` makes a deliberately returned
partial result unambiguous when `require_convergence=false`.

When a proposed slice is rejected, its initial state has not changed. The
implementation therefore retains that Arnoldi factorization and evaluates
only the small projected exponential at the shorter trial step. The returned
`arnoldi_factorizations` and `trial_evaluations` diagnostics expose this reuse;
`operator_applications` counts only full-space actions.

For repeated calls, construct each workspace at the intended scalar type.
Compiled matrix-free PI operators have fixed-precision scratch, so a wider
workspace, time, initial/minimum step, or step-safety control must be rejected
rather than silently narrowed. Accepted slices always advance by a
representable amount in that precision.
Shifts, recycling targets, preconditioners, and right-hand sides obey the same
non-narrowing rule.

## API

```@docs
BlockArnoldiWorkspace
block_arnoldi_spectrum
BlockGMRESWorkspace
block_gmres!
block_gmres
MultiShiftGMRESWorkspace
multishift_gmres!
multishift_gmres
RecycledGMRESWorkspace
recycled_gmres!
recycled_gmres
KrylovExpvWorkspace
krylov_expv!
krylov_expv
```
