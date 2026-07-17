# Matrix-free Krylov solvers

The Krylov routines compute stationary states and selected Liouvillian modes
using only products of the form `mul!(y, L, x)`. They do not assemble the
Liouvillian matrix, compute a dense spectrum, or perform a sparse LU
factorization. This is the preferred path when the PI Liouville dimension is
large enough that storing the superoperator is the dominant cost.

This page covers stationary and selected-eigenvalue solvers. For several
right-hand sides, a common shift family, continuation with a recycled
subspace, or direct exponential action, see [Block, shifted, recycled, and
exponential Krylov methods](krylov_extensions.md).

## Constructing the matrix-free Liouvillian

From a static `PIModel`, request the matrix-free representation explicitly:

```julia
L = liouvillian(model; representation=:matrixfree)
```

Passing `model` directly to a public routine with `method=:krylov`,
`:harmonic`, `:iram`, or `:jd` also selects this representation automatically:

```julia
rho_ss = steady_state(model; method=:krylov)
```

Fixed operators and their Schur geometry are cached in `L`. Scalar
time-dependent rates are supported by matrix-free application, but stationary
and spectral calculations require a time-independent Liouvillian.

## Stationary states: restarted GMRES

`steady_state(...; method=:krylov)` and `method=:gmres` select the same solver.
The stationary equation is singular, so the implementation solves the
trace-fixed rank-one system

```math
(\mathcal L + v t^\dagger)\rho=v,
\qquad t^\dagger v=1,
```

where `t` is the physical PI trace functional. For a trace-preserving
Liouvillian with a unique stationary state, this is equivalent to
``\mathcal L\rho=0`` and ``\mathrm{tr}\rho=1``.

```julia
info = steady_state(model;
    method=:krylov,
    krylovdim=40,
    maxiter=500,
    atol=1e-11,
    rtol=1e-9,
    return_info=true,
)

rho_ss = PIState(model.basis, info.state)
```

The principal options are:

| Option | Meaning |
|---|---|
| `krylovdim` | Maximum GMRES basis size before a restart; default `30` |
| `maxiter` | Maximum total Krylov iterations across all restarts |
| `atol`, `rtol` | Absolute and relative residual tolerances |
| `initial_state` | Initial vector or `PIState`; useful in parameter scans |
| `workspace` | Reusable `KrylovWorkspace` |
| `preconditioner` | `nothing`, `:schur`, or a reusable preconditioner object |
| `return_info` | Return residuals and iteration diagnostics |

The returned information contains `state`, `residual`, `normalized_residual`,
`operator_scale`, `trace_error`, `linear_residual`, `iterations`, `restarts`,
`krylov_dimension`, and `converged`. The trace-fixed system uses
`L/operator_scale`, making convergence invariant under a global rate
rescaling. `residual` remains the physical ``\|\mathcal L\rho\|`` value;
`normalized_residual` is the scale-independent accuracy check.

### Reusing storage

GMRES stores ``O(nm)`` numbers for Liouville dimension `n` and Krylov dimension
`m`. Reuse that storage for repeated solves:

```julia
L = liouvillian(model; representation=:matrixfree)
workspace = KrylovWorkspace(L, 40)

first = steady_state(L;
    method=:krylov,
    workspace=workspace,
    return_info=true,
)

second = steady_state(L;
    method=:krylov,
    workspace=workspace,
    initial_state=first.state,
    return_info=true,
)
```

One workspace must not be used concurrently. For threaded parameter sweeps,
allocate one workspace per thread.

Workspace scalar types follow the actual problem precision. A fully
`ComplexF32` Liouvillian with `Float32` seeds, shifts, and explicit scales
keeps its dominant Krylov matrices in `ComplexF32`, roughly halving their
storage relative to `ComplexF64`. Materialized and compatible custom operators
promote for a wider initial vector, non-integer target, preconditioner,
projector, regularization, or operator scale. A compiled matrix-free PI plan
instead owns scratch in its compiled scalar type, so a wider solver input is
rejected before allocating the Krylov basis; compile the model at that wider
precision when it is required. A caller-supplied workspace that cannot
represent all solver inputs is likewise rejected instead of silently narrowing
data. Integer zero remains a precision-neutral default target.

### Schur-sector preconditioning

The Schur preconditioner keeps the diagonal sector blocks of the trace-fixed
operator and LU-factorizes them independently. Local processes may couple
different Schur sectors; those off-diagonal couplings remain in the exact
matrix-free GMRES operator but are omitted from the preconditioner.

For a one-off solve, construction can be requested automatically:

```julia
info = steady_state(model;
    method=:krylov,
    preconditioner=:schur,
    krylovdim=40,
    return_info=true,
)
```

For repeated solves, construct and retain the factors:

```julia
L = liouvillian(model; representation=:matrixfree)
P = schur_sector_preconditioner(L, model.basis)
workspace = KrylovWorkspace(L, 40)

info = steady_state(L;
    basis=model.basis,
    method=:krylov,
    preconditioner=P,
    workspace=workspace,
    return_info=true,
)
```

The setup uses matrix-free Liouvillian applications and stores
``\sum_s n_s^2`` block coefficients rather than the full ``n^2`` matrix. Its
cost is worthwhile when GMRES is difficult or the factors are reused. For a
small, rapidly convergent problem, unpreconditioned GMRES may be faster because
the block construction and triangular solves have overhead.

The reusable object's `metadata` reports setup Liouvillian applications,
factorizations, stored coefficients/bytes, per-apply triangular solves, and a
conservative suggested reuse count. Construction warns when
`expected_reuses` is too small to amortize setup.

If a diagonal sector block is singular, an error is raised. A small explicit
regularization can be supplied:

```julia
P = schur_sector_preconditioner(
    L, model.basis; regularization=1e-12)
```

The same shortcut is available as
`preconditioner_regularization=1e-12` with `preconditioner=:schur`. Increase it
only enough to make the approximate block solves stable, and always validate
the final unpreconditioned `residual`. The report fields `preconditioned` and
`preconditioner` record whether and which preconditioner was used.

### Supplying a trace functional

A `MatrixFreeLiouvillian` already stores the correct PI trace vector. For a
custom linear operator, supply either its PI basis or its trace vector:

```julia
rho = krylov_steady_state(custom_L; basis=basis)
# or
rho = krylov_steady_state(custom_L; trace_vector=t)
```

The routine rejects a custom operator when the physical trace is ambiguous.

## Partial spectra: Arnoldi

Use Arnoldi when only a few modes are needed:

```julia
modes = pi_liouvillian_spectrum(model;
    method=:krylov,       # :arnoldi is an alias
    nev=6,
    krylovdim=40,
    sortby=:real,
    vectors=true,
    atol=1e-11,
    rtol=1e-9,
)
```

With `vectors=true`, the result contains:

- `values`: selected Ritz eigenvalues;
- `vectors`: corresponding right Ritz vectors;
- `residuals`: estimates of ``\|\mathcal Lv-\lambda v\|``;
- `converged`: convergence flag for every requested pair;
- `iterations`, `krylov_dimension`, and full PI `dimension`.

With `vectors=false`, `pi_liouvillian_spectrum` returns only the selected
eigenvalues. The lower-level `krylov_liouvillian_spectrum` always returns the
diagnostic named tuple and additionally accepts `which`:

| `which` | Selected Ritz values |
|---|---|
| `:LR` | Largest real part; appropriate for slow Liouvillian modes |
| `:LM` | Largest magnitude |
| `:SM` | Smallest magnitude |

The public wrapper maps `sortby=:real, rev=true` to `:LR`, and magnitude
sorting to `:LM` or `:SM`. Ascending real-part selection is not supported.

The starting vector can be controlled with `initial_vector`. High-level
wrappers use a deterministic local RNG by default; pass an explicit
`rng=MersenneTwister(seed)` when a scan should own and record its random
stream.

For repeated spectra, reuse the large basis and pencil arrays:

```julia
workspace = ArnoldiWorkspace(L, 40)
modes = krylov_liouvillian_spectrum(L;
    nev=6, krylovdim=40, workspace=workspace)
```

The same workspace can be reused sequentially by harmonic Arnoldi, but one
workspace must not be shared concurrently.

By default, failure of any requested Ritz pair raises an error. Set
`require_convergence=false` only when inspecting the returned residuals
manually.

### Thick-restarted harmonic Arnoldi near zero

For stationary and metastable modes clustered around zero, use:

```julia
modes = pi_liouvillian_spectrum(model;
    method=:harmonic, nev=6, krylovdim=48,
    thickdim=12, maxrestarts=30, vectors=true)
```

Harmonic Ritz extraction targets interior eigenvalues without factorizing a
shifted Liouvillian. At each restart, converged and best unconverged Ritz
vectors are retained as a thick subspace. The result additionally reports
`restarts`, `retained_dimension`, `target`, `harmonic_shift`,
`ritz_extraction`, and `search_space_exhausted`. While the search space is
partial, an exact zero target receives a tolerance-scale negative internal
shift because the stationary vector otherwise makes the harmonic pencil
singular. If the full ambient space or the exact range of a
`MatrixFreeSymmetryProjector` has been spanned, the solver instead uses
ordinary Rayleigh--Ritz extraction in that complete invariant space. This
removes the singular-pencil sensitivity without changing the requested
residual tolerance or materializing the projected Liouvillian.

Increase `thickdim` for clustered modes while keeping it below `krylovdim`.
Increase `maxrestarts` when Ritz residuals decrease steadily but have not
reached tolerance. The lower-level interface is
`harmonic_arnoldi_spectrum`.

### Implicit-QR restarted Arnoldi

For spectral-edge modes when an unrestarted Arnoldi basis is too large, the
exact-shift route applies unwanted Ritz values as implicit QR
shifts of the Hessenberg factorization:

```julia
modes = pi_liouvillian_spectrum(model;
    method=:iram,
    nev=6,
    sortby=:real,
    krylovdim=40,
    retained_dimension=12,
    maxrestarts=30,
    vectors=true,
)
```

The direct interface is `implicitly_restarted_arnoldi_spectrum`, where
`which=:LR`, `:LM`, or `:SM` controls selection.

The transformed Arnoldi relation is truncated and expanded using only
`mul!(y,L,x)`. `retained_dimension` must be smaller than `krylovdim`; retaining
more directions helps clusters but leaves fewer new directions per cycle. A
non-`nothing` `target` changes selection to distance from that point, although
polynomial implicit restarting remains primarily an edge solver. Use harmonic
Arnoldi or Jacobi--Davidson for a difficult interior cluster.

A scalar Arnoldi start contains only one direction in an exactly degenerate
eigenspace. Requesting every vector in a repeated eigenspace therefore
requires a block method or the hard-locking Jacobi--Davidson route below.

### Preconditioned Jacobi--Davidson and hard locking

`jacobi_davidson_spectrum` targets the eigenvalues nearest a chosen value and
solves an inexact projected correction equation by restarted GMRES:

```julia
P = schur_sector_preconditioner(
    L, model.basis; expected_reuses=20)
workspace = JacobiDavidsonWorkspace(L, 40, 20)

modes = pi_liouvillian_spectrum(model;
    method=:jd,
    nev=6,
    target=0,
    krylovdim=40,
    correction_krylovdim=20,
    correction_maxiter=60,
    preconditioner=P,
    workspace=workspace,
    vectors=true,
)
```

The direct interface is `jacobi_davidson_spectrum`. The `:jd` wrapper always
orders by distance from the numeric `target`; `sortby` does not replace that
near-target selection.

The correction projector removes both the current Ritz direction and the
already converged invariant subspace. Converged Schur directions are hard
locked, so subsequent searches can resolve repeated or tightly clustered
modes. The supplied preconditioner is also projected and need only implement
`size` and `ldiv!(destination,P,source)`; it is an approximation to the
shifted correction operator, so an imperfect Schur preconditioner affects
speed but not the final residual test.

Important result fields are `hard_locked`, `correction_iterations`,
`correction_failures`, `operator_applications`, `restarts`, and
`restart_history`. Returned right eigenvectors are reconstructed from the
locked Ritz block and checked explicitly with
``\|L v-\lambda v\|``. A nonzero `correction_failures` is not by itself a
failure: inexact corrections are intentional, and a projected Davidson
direction is used when GMRES stagnates. Convergence is decided only by the
outer eigenpair residuals.

### Matrix-free weak-symmetry projection

For a unitary weak symmetry, a Liouville charge satisfies

```math
U\rho U^\dagger=q\rho,\qquad |q|=1.
```

The implementation diagonalizes only sector-sized Schur representations of
`U`. Projection is block-local; neither the full symmetry superoperator nor a
reduced Liouvillian matrix is constructed.

```julia
sz = ComplexF64[1 0; 0 -1]
modes = pi_liouvillian_spectrum(model;
    method=:harmonic, symmetry=sz, charge=1,
    nev=4, krylovdim=32, vectors=true)
```

The trivial charge also accepts `charge=:trivial`. `symmetry=:auto` searches
common local unitaries and accepts a projector only when deterministic
matrix-free commutator probes pass. Assembled matrices use the full
commutator instead of probes. For reuse, construct the projector explicitly:

```julia
Pq = matrixfree_symmetry_projector(model.basis, sz; charge=-1)
Pwork = SymmetryProjectorWorkspace(Pq)
modes = harmonic_arnoldi_spectrum(L;
    projector=Pq, projector_workspace=Pwork, nev=4)
```

The ordinary `mul!` path uses a locked compatibility workspace and is
allocation-free, so sharing one projector is safe but serialized. Parallel
scans should use one explicit `SymmetryProjectorWorkspace` per task.

Antiunitary symmetries do not define complex-linear charge sectors and cannot
be used for this projection.

## Matrix-free Liouvillian gap

The gap is obtained from Ritz values with largest real part:

```julia
gapinfo = pi_liouvillian_gap(model;
    method=:iram, # or :krylov for one unrestarted Arnoldi factorization
    nev=6,
    krylovdim=40,
    return_info=true,
)
```

Implicit QR remains a largest-real Ritz search in the gap command. The gap
routine rejects `method=:jd`: distance from zero can miss a slowly decaying
mode with large imaginary part, so near-target Jacobi--Davidson cannot certify
the global gap. Use it to study a known near-zero cluster, not to replace a
largest-real search.

Important fields are:

- `gap` and `decay_eigenvalue`;
- `oscillation_frequency`;
- `ritz_residuals`;
- `stationary_multiplicity`;
- `stationary_multiplicity_certified`;
- `stable` and `spectral_abscissa`.

Choose `nev` large enough to include the stationary cluster and at least one
decaying mode. A partial spectrum cannot certify the stationary multiplicity
when stationary modes fill the entire requested window. The
`stationary_multiplicity_certified` field records this distinction.

Harmonic Ritz extraction must not be used as a certified global-gap method:
it orders modes by distance to zero rather than by real part, and can therefore
miss a slowly decaying mode with a large oscillation frequency. A request for
`method=:harmonic` without a symmetry consequently raises an error.

With `method=:harmonic`, `symmetry=U`, `charge=q`, and `return_info=true`, the
routine reports a near-zero charge-sector decay estimate. Nontrivial charge
sectors are not required to contain a stationary mode. The result records
`scope=:charge_sector`, `selection=:near_zero`, `sector_dimension`, and
`gap_certified`. The last field is true only when the complete selected sector
was extracted; otherwise inspect the Ritz residuals directly. Use the
largest-real Krylov route or a complete dense symmetry-block spectrum when a
global gap is required.

## Choosing tolerances and Krylov dimensions

Start with `krylovdim` between 30 and 60. Increase it when:

- GMRES reaches `maxiter` without satisfying the residual tolerance;
- requested Ritz residuals remain above tolerance;
- the Liouvillian is strongly nonnormal;
- several eigenvalues cluster near zero.

A larger Krylov basis usually improves convergence but uses more memory and
orthogonalization time. For a fixed memory budget, increase `maxiter` to allow
more GMRES restarts. Basic `method=:krylov` Arnoldi uses one factorization;
`method=:harmonic` supports thick restarting through `maxrestarts`.
The lower-level implicit-QR and Jacobi--Davidson routines add bounded-memory
restarts and, for Jacobi--Davidson, hard locking plus preconditioned correction
solves.

When unpreconditioned GMRES stagnates, try the Schur-sector preconditioner
before making the Krylov basis very large. Compare both the number of
iterations and total setup-plus-solve time: preconditioning is not guaranteed
to reduce iterations for every model.

Always check the physical residual, trace error, and (when required) positivity
of the resulting state. Krylov convergence alone does not certify that an
approximate stationary vector is a physical density operator.

## Choosing between methods

| Task | Recommended method |
|---|---|
| Small or moderate sparse steady state | `method=:direct` |
| Large matrix-free steady state | `method=:krylov` |
| Sparse eigenvalue nearest a chosen shift | `method=:shiftinvert` |
| Complete PI spectrum | `method=:dense` |
| Modes with largest real part | `method=:krylov` |
| Interior modes nearest zero | `method=:harmonic` |
| Restarted spectral-edge extraction | `method=:iram` / `implicitly_restarted_arnoldi_spectrum` |
| Hard nonnormal/degenerate interior cluster | `method=:jd` / `jacobi_davidson_spectrum` |
| Near-zero charge-sector decay estimate | `method=:harmonic, symmetry=..., charge=..., return_info=true` |
| Diagnose a degenerate nullspace | `method=:svd` for manageable dimensions |

The Krylov steady-state formulation assumes the trace constraint selects a
unique solution. If the Liouvillian has multiple stationary states, use a
nullspace-aware method or impose additional conserved-charge constraints.
