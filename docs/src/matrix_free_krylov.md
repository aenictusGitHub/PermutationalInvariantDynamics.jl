# Matrix-free Krylov solvers

The Krylov routines compute stationary states and selected Liouvillian modes
using only products of the form `mul!(y, L, x)`. They do not assemble the
Liouvillian matrix, compute a dense spectrum, or perform a sparse LU
factorization. This is the preferred path when the PI Liouville dimension is
large enough that storing the superoperator is the dominant cost.

This page covers stationary states, selected eigenvalues, matrix-free Floquet
maps, and response calculations. For several right-hand sides, a common shift
family, continuation with a recycled subspace, or direct exponential action,
see [Block, shifted, recycled, and exponential Krylov
methods](krylov_extensions.md).

For autonomous GKSL models whose no-jump generator is inexpensive to invert
sector by sector, the [no-jump-resolvent iterative solvers](no_jump_iterative_solvers.md)
provide a different matrix-free route: an exact no-jump right preconditioner,
a CPTP fixed-point stationary map, nested shift-invert slow modes, and
implicit Euler.

Public high-level commands report canonical solver names even when a retained
compatibility alias is supplied:

| Task | Preferred spelling | Compatibility aliases |
|---|---|---|
| Trace-fixed GMRES | `:gmres` | `:krylov` |
| Shift-invert stationary solve | `:shiftinvert` | `:shift_invert`, `:inverse_iteration` |
| Ordinary Arnoldi spectrum | `:arnoldi` | `:krylov`, `:ordinary_arnoldi` |
| Block Arnoldi spectrum | `:block_arnoldi` | `:block` |
| Implicitly restarted Arnoldi | `:iram` | `:implicit_qr` |
| Jacobi--Davidson | `:jd` | `:jacobi_davidson` |

The low-level `steady_state(...; method=:krylov)` spelling remains supported
because it predates the high-level `GMRESAlgorithm` interface.

## Constructing the matrix-free Liouvillian

From a static `PIModel`, request the matrix-free representation explicitly:

```julia
L = liouvillian(model; representation=:matrixfree)
```

Passing `model` directly to a public routine with `method=:krylov` for a
steady state, or `method=:arnoldi`, `:harmonic`, `:iram`, or `:jd` for a
spectrum (with `:block_arnoldi` for a block start), also selects this
representation automatically:

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
$\mathcal L\rho=0$ and $\mathrm{tr}\rho=1$.

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
rescaling. `residual` remains the physical $\|\mathcal L\rho\|$ value;
`normalized_residual` is the scale-independent accuracy check.

### Reusing storage

GMRES stores $O(nm)$ numbers for Liouville dimension `n` and Krylov dimension
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

For a prepared `LiouvillianPlan`, `CompiledPIModel`, or compiled
`MatrixFreeLiouvillian`, setup lowers each diagonal sector block directly from
the immutable physical term kernels. The only complete Liouvillian
applications are the three reproducible operator-scale probes (or none when
`operator_scale` is supplied). An arbitrary matrix-free operator retains the
generic compatibility route, which probes one basis vector per PI coordinate.
`SpecializedPIModel` uses the same direct route while evaluating every
prepared family-rate schedule with that specialization's bound rates. Pass the
specialization itself with `schur_sector_preconditioner(specialized)`; its
exact basis is inferred. A detached, plan-less matrix-free callback
deliberately retains the generic fallback because its bound parameters cannot
be inferred.
Both routes store $\sum_s n_s^2$ block coefficients rather than the full
$n^2$ matrix and produce the same trace-bordered block operator. Setup is
worthwhile when GMRES is difficult or the factors are reused. For a small,
rapidly convergent problem, unpreconditioned GMRES may be faster because the
block construction and triangular solves have overhead.

The reusable object's `metadata` reports setup Liouvillian applications,
their split between scale and block probes, the block-construction route,
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
    method=:arnoldi,      # :krylov remains a compatibility alias
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
- `residuals`: estimates of $\|\mathcal Lv-\lambda v\|$;
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
workspace = ArnoldiWorkspace(L, 40; mode=:ordinary)
modes = krylov_liouvillian_spectrum(L;
    nev=6, krylovdim=40, workspace=workspace)
```

`mode=:ordinary` retains only the Arnoldi basis, Hessenberg matrix, and two
full-coordinate vectors. The ordinary solver selects this lean mode
automatically when it constructs its own workspace. Use the default
`mode=:full` when one workspace must also be reused sequentially by harmonic
Arnoldi, implicit-QR Arnoldi, or Jacobi--Davidson. Those advanced solvers
reject an ordinary-only workspace rather than allocating omitted arrays.
No workspace may be shared concurrently.

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
$\|L v-\lambda v\|$. A nonzero `correction_failures` is not by itself a
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

### Compressed strong-symmetry coordinates

A strong diagonal local symmetry can reduce the stored Krylov vectors to one
ket/bra charge block rather than repeatedly projecting a full PI vector. For
example, a parity-preserving model may be restricted with

```julia
Z = Diagonal(ComplexF64[1, -1])
restriction = diagonal_symmetry_restriction(
    model.basis, Z; ket_charge=1, bra_charge=1)

L = compile(model; backend=:matrixfree)
restricted = RestrictedLiouvillian(L, restriction)
work = RestrictedLiouvillianWorkspace(restricted)
rho_ss = restricted_steady_state(restricted)
```

Construction exhaustively checks leakage before accepting the restriction.
Explicit matrices are scanned columnwise (stored nonzeros only for sparse
matrices); a matrix-free source is probed once per retained coordinate.
`restriction_full_residual` subsequently embeds a computed mode and evaluates
the original generator, independently checking both the retained and omitted
coordinates. Nothing is silently projected when leakage is nonzero.

With a sparse or dense source, `RestrictedLiouvillian(...; backend=:auto)`
stores the actual submatrix. With a prepared matrix-free source containing
fixed Hamiltonian, collective-dissipator, or local-gain kernels, the default
`:lowered` backend restricts the ket and bra Schur blocks and filters gain
coordinates once. Its applications, adjoints, and Krylov vectors never touch
ambient PI vectors. `RestrictedLiouvillianWorkspace(restricted)` returns the
corresponding task-owned reduced scratch. `ResponseWorkspace(restricted)`
keeps the same reduced application path for resolvents, adjoint evolution,
and sensitivities.

An explicit mask that is not a Cartesian ket-pattern by bra-pattern block, or
an unsupported operator-valued prepared kernel, retains the certified
`:embedded` fallback. That fallback still compresses Krylov storage but uses
two ambient application vectors. The selected route is recorded in
`restricted.backend`; `backend=:compressed` requires an explicit matrix and
`backend=:lowered` can be requested to reject rather than fall back.

Separate `ket_charge` and `bra_charge` values expose off-diagonal operator
blocks. Such a block normally has a zero physical trace and is useful for
decay modes, but `restricted_steady_state` rejects it because it cannot contain
a normalized density operator. Plans and workspaces are safe to share only
according to the same immutable-plan/task-owned-workspace rules as the rest of
the matrix-free API.

`restriction_full_residual` is deliberately different: it re-evaluates the
original ambient generator to certify omitted-coordinate leakage. Repeated
calls therefore need an ambient `RestrictedLiouvillianWorkspace` constructed
from `restricted.source` and `restricted.restriction`, not the reduced
workspace returned for ordinary lowered applications.

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

## Matrix-free Floquet maps

Periodic problems have the same plan/workspace split. `floquet_map` prepares
the RK grid and the underlying Liouvillian action, but does not construct the
$n_{\mathrm{PI}}\times n_{\mathrm{PI}}$ one-period matrix:

```julia
period_map = floquet_map(prepared, period; steps=320)
map_work = FloquetWorkspace(period_map)

image = similar(rho0.data)
apply!(image, period_map, rho0.data, map_work)

multipliers = selected_floquet_multipliers(
    period_map;
    method=:block_arnoldi,
    block_size=4,
    which=:LM,
    nev=min(6, size(period_map, 1)),
    krylovdim=40,
    vectors=true,
)

periodic = floquet_steady_state(
    period_map;
    method=:krylov,
    krylovdim=40,
    return_info=true,
)
```

Ordinary Arnoldi (`method=:arnoldi`), thick-restarted block Arnoldi
(`method=:block_arnoldi`), and IRAM select multipliers by a spectral
criterion; `which=:LM` is the usual slow-decay choice. Block Arnoldi uses a
fixed-capacity matrix workspace and sends each search block through one
batched period-map application. Jacobi--Davidson
(`method=:jd`) targets a specified multiplier, one by default, and is not a
global spectral-radius calculation. Every selected result retains Ritz
residuals, convergence flags, the full operator dimension, and whether its
scope is partial. `floquet_steady_state` solves the trace-fixed equation
$(F-I)\rho=0$ by restarted GMRES and reports the physical period residual.

`floquet_gap(period_map; return_info=true)` reports residual certification for
the selected multipliers. A partial largest-modulus window can identify a
well-resolved candidate decay rate, but `global_gap_certified` becomes true
only when the complete map dimension has been resolved and every required
residual passes. Never promote a partial multiplier window to a global gap or
stability certificate.

The single-vector forward and adjoint actions reuse the same
`FloquetWorkspace`. For blocks, construct `FloquetBatchWorkspace(map, p;
mode=:forward)` to retain only three stage matrices beyond the destination,
or use `mode=:full` when exact batched adjoint actions are required. Capacity
is explicit and never grows during application. The destination is the
low-storage RK4 accumulator and must therefore have exactly the map's scalar
type; a representable narrower input is copied into map-precision scratch.
The adjoint reverses the
actual finite-step RK4 computation, so
`apply_adjoint!` is the numerical adjoint of the discretized period map. This
is important for response calculations and differs from independently
discretizing a continuous adjoint equation.

For a certified strong diagonal symmetry, first build a
`SymmetryCoordinateRestriction`, then call
`restricted_floquet_map(period_map, restriction)`. Selected multiplier and
fixed-point Krylov vectors are stored in reduced coordinates. Period
applications still use caller-owned ambient scratch, and the returned
fixed-point report includes the full-space residual and leakage. The
restriction is accepted only after exhaustive invariance checks.

Converge two numerical layers separately: increase the period-map `steps`
until the physical results are stable, then increase Krylov dimensions and
iteration limits until the reported linear or Ritz residuals pass. The dense
`floquet_propagator` remains useful as a small-problem reference or when the
complete explicit channel is genuinely required.

## Matrix-free response and adjoint analysis

The response tools accept the same prepared matrix-free source. A
`ResponseWorkspace` can retain restarted-GMRES storage, adaptive exponential-
action storage, or both:

```julia
work = ResponseWorkspace(
    prepared; krylovdim=40, expv_krylovdim=40, mode=:both)

resolvent = resolvent_norm(
    prepared, 0.2im;
    method=:krylov,
    workspace=work,
    return_info=true,
)

A_t = adjoint_evolve(
    prepared, A, 2.0;
    method=:krylov,
    workspace=work,
)

tau = integrated_correlation_time(
    prepared, rho_ss, A;
    method=:krylov,
    workspace=work,
    return_info=true,
)

chi = steady_state_susceptibility(
    prepared, rho_ss, dL;
    observable=A,
    method=:krylov,
    workspace=work,
    return_info=true,
)
```

Here `A` is a `PIOperator`, `rho_ss` is a stationary `PIState`, and `dL` is
the autonomous derivative of the Liouvillian with respect to the parameter of
interest. `integrated_correlation_time` and
`steady_state_susceptibility` solve trace-fixed Poisson/tangent equations and
return the raw physical residual and trace error. The tangent state is not
normalized or positivity-projected.

For a matrix-free source, `resolvent_norm` uses power iteration on the forward
and adjoint resolvents, with two shifted GMRES solves per outer iteration. It
therefore requires an explicit adjoint action. Nonconverged shifted solves or
power iterations raise instead of returning a finite value, but a converged
result is still a numerical estimate rather than a rigorous upper bound.
`pseudospectral_abscissa` repeats this calculation on a supplied complex grid;
reuse one `mode=:linear` workspace to avoid rebuilding the dominant Krylov
arrays at every point.

`adjoint_evolve` applies adaptive `krylov_expv!` to the exact prepared adjoint.
Use `mode=:evolution` when this is the only requested operation. Use
`mode=:linear` for resolvents and trace-fixed response solves, or `mode=:both`
when one task will call both kinds sequentially. As with every mutable solver
workspace, do not share a `ResponseWorkspace` between concurrent tasks.

## Choosing tolerances and Krylov dimensions

Start with `krylovdim` between 30 and 60. Increase it when:

- GMRES reaches `maxiter` without satisfying the residual tolerance;
- requested Ritz residuals remain above tolerance;
- the Liouvillian is strongly nonnormal;
- several eigenvalues cluster near zero.

A larger Krylov basis usually improves convergence but uses more memory and
orthogonalization time. For a fixed memory budget, increase `maxiter` to allow
more GMRES restarts. Basic `method=:arnoldi` uses one factorization;
`:krylov` remains a compatibility alias on the spectral API.
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
| Modes with largest real part | `method=:arnoldi` |
| Interior modes nearest zero | `method=:harmonic` |
| Restarted spectral-edge extraction | `method=:iram` / `implicitly_restarted_arnoldi_spectrum` |
| Hard nonnormal/degenerate interior cluster | `method=:jd` / `jacobi_davidson_spectrum` |
| Near-zero charge-sector decay estimate | `method=:harmonic, symmetry=..., charge=..., return_info=true` |
| Selected Floquet multipliers | `floquet_map`, then `selected_floquet_multipliers` |
| Matrix-free periodic state | `floquet_steady_state(floquet_map(...); method=:krylov)` |
| Resolvent or pseudospectral estimate | `ResponseWorkspace(...; mode=:linear)`, then shifted-GMRES response tools |
| Adjoint observable evolution | `ResponseWorkspace(...; mode=:evolution)`, then `adjoint_evolve` |
| Diagnose a degenerate nullspace | `method=:svd` for manageable dimensions |

The Krylov steady-state formulation assumes the trace constraint selects a
unique solution. If the Liouvillian has multiple stationary states, use a
nullspace-aware method or impose additional conserved-charge constraints.
