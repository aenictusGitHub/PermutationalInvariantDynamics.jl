# Performance benchmarks and Julia ecosystem comparisons

The benchmark suite is intended to make performance investigations
reproducible, not to provide a universal package ranking. Results depend on
the physical model, retained representation, scalar type, Julia and package
versions, solver choices, thread settings, and hardware. A result is useful
only together with that context and the numerical checks performed on the
same case.

There are four complementary internal suites and one cross-package track:

1. `performance_regression.jl` supplies deterministic CI correctness,
   equivalence, retained-storage, and hot-allocation gates without
   machine-dependent timing thresholds;
2. `performance_audit.jl` is a broad warmed, human-readable survey;
3. `benchmarks.jl` is the longer `BenchmarkTools` collection for targeted
   optimization;
4. `internal_scaling.jl` retains a stable four-family schema for core PI
   preparation and application scaling; and
5. the cross-package runner compares a deliberately small common operation
   with general-purpose Julia quantum-dynamics packages where the underlying
   representations can be matched or clearly identified.

The command-line details and current case grids are canonical in
[`benchmark/README.md`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/benchmark/README.md)
and
[`benchmark/comparison/README.md`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/benchmark/comparison/README.md).
Generated timing tables and local manifests are intentionally ignored rather
than committed. This avoids presenting one developer machine's measurements
as package-wide results.

## Internal workflow coverage

The regression, audit, and detailed suites exercise current prepared APIs over
a broader surface than the fixed internal-scaling runner:

| Workflow | Regression gate | Human-readable audit | Detailed `BenchmarkTools` suite | Internal scaling |
|---|---|---|---|---|
| Prepared PI core | sparse/matrix-free equivalence, forward/adjoint batch action, threading, allocation | setup, retained storage, scalar and batch action | separate setup, sparse, matrix-free, driven, threaded, and observable workloads | four fixed qubit/qutrit families |
| Time evolution and Floquet | low-storage RK4 and adaptive-expv storage checks | RK4, matrix-free Floquet-map setup, period/batch action, and stroboscopic evolution | prepared RK4 and Floquet setup/period/batch workloads | driven one-period action is not included |
| Appendix-D p-body and reductions | packed-support and allocation checks | p-body geometry/application, qudit reduction, local-factor trace | cached/uncached p-body setup and prepared reduction hot paths | not included |
| Identical local pseudomodes | supersite compile/application and spin-factor trace checks | prepared supersite construction, action, and reduction | reusable local-pseudomode setup and hot action | not included |
| Shared/global pseudomodes | matrix-free composite action and subsystem reduction checks | global-mode setup, action, and reduction | prepared global-pseudomode setup and hot action | not included |
| PI--HEOM | packed hierarchy forward/adjoint/batch checks | hierarchy setup and forward/adjoint/batch action | setup, prepared forward/adjoint/batch action, and preconditioner workloads | not included |
| PI--HOPS | conditioned hierarchy action and reused-ensemble checks | plan/workspace, conditioned action, and ensemble survey | plan, conditioned action, and reused-ensemble workloads | not included |
| Composite and stochastic systems | composite scalar/batch/adjoint action plus density, weak-PI, and composite trajectory gates | deterministic composite, batched, diffusive, density, weak-PI, and composite trajectory survey | reusable composite/batch and trajectory workloads | not included |
| Stationary and spectral solvers | adaptive-expv storage and symmetry-restricted workspace checks | GMRES, Arnoldi, harmonic Arnoldi, symmetry restriction, and preconditioner survey | reusable Krylov, preconditioned, recycled, and multi-shift workloads | not included |

This is a code-path inventory, not a claim that every suite uses the same
problem size or accuracy target. Regression uses deliberately small
deterministic stochastic and non-Markovian cases; the audit and detailed suite
are where their cutoffs, ensemble sizes, or Krylov dimensions should be
increased.

## Internal PI scaling suite

From the repository root, prepare the isolated benchmark environment once:

```sh
julia --startup-file=no --project=benchmark -e \
  'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

Run either the short development grid or the larger research grid:

```sh
julia --startup-file=no --project=benchmark \
  benchmark/internal_scaling.jl --mode quick
julia --startup-file=no --project=benchmark \
  benchmark/internal_scaling.jl --mode full
```

The runner also accepts explicit sample, timing, warm-up, validation-size, and
output controls. Consult `benchmark/README.md` rather than copying defaults
into long-lived automation.

The runner remains the internal-scaling schema-version-2 suite with exactly
four families:

- all-Schur-sector qubits with local and collective terms;
- all-Schur-sector qutrits with local and collective terms;
- fully symmetric qubits with collective terms only; and
- fully symmetric qutrits with collective terms only.

The current grids are explicit:

| Mode | All-sector qubits | All-sector qutrits | Symmetric qubits | Symmetric qutrits |
|---|---|---|---|---|
| `quick` | `N = 2, 4, 8` | `N = 2, 3, 4` | `N = 4, 8, 16` | `N = 2, 4` |
| `full` | `N = 2, 4, 8, 12, 16, 20, 24` | `N = 2, 3, 4, 5, 6` | `N = 4, 8, 16, 32, 48, 64, 96` | `N = 2, 4, 8, 12` |

All current cases use `ComplexF64`. The source script is authoritative for the
term coefficients and any future grid changes. Pseudomodes, HEOM, HOPS,
composite dynamics, trajectories, p-body reductions, and solver
time-to-solution are covered by the broader suites above rather than added to
this scaling schema.

Every case separates the following phases:

| Phase | Included work | Interpretation |
|---|---|---|
| Basis setup | construction of the retained PI basis | representation setup cost for that `N` and `d` |
| Matrix-free compilation | lowering the already specified model into a prepared plan | reusable model-preparation cost |
| Prepared application | one explicit-workspace generator action after warm-up | hot operator-action cost; it excludes basis and model preparation |
| Symmetric sparse compilation/application | explicit sparse setup and warmed `mul!` | sparse-first construction and retained-support action for the occupation route |
| Symmetric driven compilation/application | `InPlaceTimeOperator` collective Hamiltonian and jump with reusable workspace | preallocated time-dependent occupation-block cost |
| Automatic backend probe | `backend=:auto` with a fixed 512 MiB budget | whether structured support estimates select sparse or matrix-free |
| Validation | sparse-oracle comparison when bounded, trace preservation, and adjoint duality | numerical evidence, timed separately from the performance phases |

Setup and hot application report minimum and median elapsed nanoseconds. Their
allocated bytes and allocation counts belong to the fastest measured sample.
Retained-memory columns report the standalone `summarysize` of the basis,
model, compiled object, plan, workspace, and vectors, together with a non-
double-counted retained total. `summarysize` is a Julia object-graph estimate:
it is not peak resident memory and does not include every native-library or
operating-system allocation.

The schema also records the geometry route, conservative geometry storage,
exact-support contribution and retained-nonzero bounds, predicted sparse
operator/assembly/peak bytes, actual sparse nonzeros and retained bytes, and
the automatic backend. Sparse, driven, and automatic-probe fields are `NA` for
all-sector rows because these additional phases target the symmetric
occupation-space collective path; the common matrix-free phases still cover
every row.

This is internal-scaling result schema version 2. The sibling metadata table
records the same version and the sparse, driven, and automatic-probe policies.

The runner uses `BenchmarkTools`, performs warm-up passes, sets one evaluation
per sample, and forces BLAS to one thread. Its metadata sidecar records the
mode and controls, Julia/package and Git information, thread settings, machine
details, active project and selected manifest SHA-256 hashes, and whether the
startup file was disabled. The default output is
`benchmark/results/internal_scaling_<mode>.tsv` with a sibling
`.metadata.tsv`; neither is a source artifact.

The audit, regression, and detailed entry points serve different purposes:

```sh
julia --startup-file=no --project=benchmark \
  benchmark/performance_audit.jl
JULIA_NUM_THREADS=4 julia --startup-file=no --project=benchmark \
  benchmark/performance_regression.jl
julia --startup-file=no --project=benchmark -e \
  'include("benchmark/benchmarks.jl"); display(run(SUITE; verbose=true))'
```

`performance_audit.jl` is a human-readable warmed timing/allocation survey.
`performance_regression.jl` primarily enforces allocation and numerical-
equivalence gates and deliberately avoids fragile wall-clock thresholds.
`benchmark/benchmarks.jl` remains the detailed `BenchmarkTools` workload
collection.

## Cross-package comparison

The comparison environments are isolated from both the package root and one
another. This is necessary because the compared packages can resolve
different SciML dependency ranges; it also prevents one backend from changing
another backend's dependency graph. Package loading and precompilation are not
part of the timed operation.

Prepare all comparison environments and then run every backend with:

```sh
julia --startup-file=no benchmark/comparison/setup.jl
julia --startup-file=no benchmark/comparison/run_all.jl
```

For example, a controlled run can specify
`--samples 50 --seconds 2 --threads 1 --output /tmp/pid-comparison` on
`run_all.jl`. The default ignored output is under
`benchmark/comparison/results/`. The runner writes one TSV per backend and a
joined comparison table. It also records package versions, representation and
dimensions, sparse nonzero counts, retained `summarysize`, minimum and median
application times, allocations, execution backend and action kind, validation
results, requested and actual sample counts, timestamp, OS/architecture,
BenchmarkTools version, benchmark Git revision/dirty state and worktree hash,
full BLAS configuration, and a SHA-256 of the active manifest.
The current cross-package row schema is version 3 and is identical for every
adapter.

Each backend is loaded and warmed before measurement. The **setup** phase
constructs the basis, initial state, and sparse generator. The **apply** phase
measures a warmed in-place sparse matrix-vector product. These phases must not
be mixed: a package with more reusable preparation can have a different setup
tradeoff from its repeated-action tradeoff.

### Matched collective dynamics

The first track uses collective emission
`gamma * D[J_-]`, with `gamma = 0.37`, from the fully excited state at
`N = 4, 8, 16, 32, 64`. It is an exact comparison in the fully symmetric
spin-`N/2` sector:

- `PermutationalInvariantDynamics` explicitly retains only partition
  `(N,0)` for this track;
- `QuantumOptics` uses its spin-`N/2` basis; and
- `QuantumToolbox` uses its spin-`N/2` representation.

The state, generator convention, and retained Hilbert dimension are therefore
matched. Every row validates the vanishing trace derivative and the analytical
Frobenius norm `sqrt(2) * gamma * N`, Hermiticity, and the complete analytical
derivative eigenspectrum before its timing is interpreted. The equation-(7)
PI coordinates are orthonormal, so their Euclidean norm is the same physical
Hilbert-space Frobenius norm used by the full-space adapters.

Adapters reuse an existing `SparseMatrixCSC` without copying it and convert
another matrix representation exactly once. PID setup therefore measures the
symmetric occupation-number lowering and sparse-first materialization; the
`backend` and `action_kind` columns distinguish that package-specific setup
from the common warmed `SparseArrays.mul!` action.

This track measures a collective symmetric-sector operation. It does **not**
measure the defining advantage of an all-Schur-sector PI representation under
local noise. Moreover, the materialized sparse generators have the same CSC
dimensions and nonzero count, so every warmed action invokes the same Julia
`SparseArrays.mul!` implementation. Treat tiny hot-action differences as
measurement noise; package-specific differences in this track are setup costs.

### Independent local emission

The second track uses `gamma * sum_i D[sigma_-^(i)]`, again with
`gamma = 0.37`, from the fully excited product state. Independent local jumps
populate several total-spin sectors, so a single spin-`N/2` density matrix is
not an exact representation of this dynamics.

For this reason:

- `PermutationalInvariantDynamics` uses its complete PI operator basis at
  `N = 2, 4, 6, 8, 16`; while
- `QuantumOptics` and `QuantumToolbox` use the full tensor-product Hilbert
  space only at `N = 2, 4, 6`, where that construction is deliberately
  bounded.

Every row validates the analytical derivative norm
`gamma * sqrt(N * (N + 1))`, Hermiticity, trace preservation, and the full
analytical eigenvalue/multiplicity signature. The result should be read as two
representation-scaling curves with a small common correctness overlap, not as
like-for-like backend speed. A fixed-spin result is not reported as a speed
ratio for local dissipation, because it would solve a different problem.

The table separates `physical_hilbert_dimension=2^N` from the vector length
actually used by the sparse action, `retained_operator_dimension`. The latter
is `(N+1)^2` for a fixed spin irrep, the complete PI coefficient count for the
all-sector backend, or `4^N` in the full-Hilbert baselines.

## What counts as a direct competitor?

Our review did not identify another maintained Julia package whose public
numerical representation covers exact mixed PI states spanning all retained
Schur sectors with both independent local and collective open-system terms.
That is a statement about the packages and interfaces reviewed for this
benchmark, not a proof that no such code exists. Contributions extending the
comparison are welcome when they provide an equivalent physical problem and a
reproducible environment.

The packages discussed here occupy useful but different scopes:

| Package | Relevant scope | Role in this comparison |
|---|---|---|
| [QuantumOptics.jl documentation](https://docs.qojulia.org/) / [repository](https://github.com/qojulia/QuantumOptics.jl) | General quantum dynamics with spin and composite bases | Direct for the matched symmetric collective track; full-Hilbert small-`N` reference for local emission |
| [QuantumToolbox.jl documentation](https://qutip.org/QuantumToolbox.jl/) / [repository](https://github.com/qutip/QuantumToolbox.jl) | General quantum dynamics with dense/sparse operators and SciML solvers | Same bounded comparison roles as QuantumOptics |
| [QuantumCumulants.jl documentation](https://qojulia.github.io/QuantumCumulants.jl/stable/) / [repository](https://github.com/qojulia/QuantumCumulants.jl) | Symbolic equations closed by a selected-order cumulant approximation | Contextual alternative for large systems, not a direct benchmark of exact PI mixed-state dynamics |
| [CollectiveSpins.jl documentation](https://qojulia.github.io/CollectiveSpins.jl/) / [repository](https://github.com/qojulia/CollectiveSpins.jl) | Spatially distributed, interacting collective-spin models and reduced descriptions | Contextual ecosystem package, not an exact all-Schur-sector PI backend in this benchmark |

The registry search also considered
[PermutationSymmetricTensors.jl](https://github.com/IlianPihlajamaa/PermutationSymmetricTensors.jl),
which is a general permutation-symmetric multidimensional-array container, and
[DickeModel.jl](https://github.com/saulpila/DickeModel.jl), which targets the
specific quantum and classical Dicke model. Neither exposes the equivalent
general mixed-state all-Schur PI dynamics workload, so neither is assigned a
timing ratio here.

QuantumCumulants or a mean-field/cluster method may be the scientifically
appropriate choice when controlled approximate observables are sufficient at
sizes beyond exact PI calculations. Its runtime should be accompanied by a
closure-order and accuracy study, not divided directly by an exact density-
operator runtime. Similarly, general-purpose full-Hilbert packages provide
features and model classes outside this package's PI scope; one microbenchmark
does not rank those broader capabilities.

## Reading results responsibly

When reporting a benchmark:

1. retain the generated metadata and resolved package versions;
2. state the physical equation, initial state, representation, and retained
   dimensions;
3. report validation errors beside timing and memory metrics;
4. distinguish cold setup, reusable preparation, and warmed application;
5. use the same thread and BLAS settings, scalar precision, and hardware; and
6. repeat the run after changing Julia, a dependency, or the machine.

Minimum time is useful only when it is comfortably above measurement overhead,
while the median is a more robust typical sample. A near-zero corrected
minimum, especially `0.001 ns`, is unresolved and must not be used in a speed
ratio; time a larger batch instead. Julia allocation bytes do not equal peak
RAM, and sparse `nnz` does not include every retained object.
Neither metric establishes time-to-solution for steady states, trajectories,
adaptive dynamics, or spectra. Those tasks require their own matched accuracy
criterion and convergence study.

Accordingly, the repository publishes the harness and validation rules, not a
timeless claim that one package is universally faster.
