# Cross-package PI benchmark

This directory provides a reproducible, validation-first comparison with
[QuantumOptics.jl](https://docs.qojulia.org/) and
[QuantumToolbox.jl](https://qutip.org/QuantumToolbox.jl/stable/). Each package
runs in a separate Julia project and process. This isolation is intentional:
it prevents their SciML dependency trees and preferences from changing the
environment used to benchmark `PermutationalInvariantDynamics.jl`.

The scripts record raw measurements, not a headline speed ratio. The output
includes package, Julia, and BenchmarkTools versions; timestamp, OS, CPU,
architecture, Julia-thread, and BLAS metadata; the benchmark Git revision and
dirty state plus a worktree SHA-256; an active-manifest SHA-256; requested and
actual sample counts;
representation labels; physical-Hilbert and retained-operator dimensions;
sparse nonzeros; retained bytes; setup and warmed-application median/minimum
times; allocations; execution backend and action kind; and validation errors.
Every adapter pins BLAS to one thread. Generated manifests and result tables
are ignored and must not be committed.

The current comparison-row schema is version 3; every adapter must return the
same ordered fields so `run_all.jl` can combine the tables without coercion.

## Workloads and interpretation

The `matched_collective` track computes

```math
\mathcal{L}\rho=\gamma\mathcal{D}[J_-]\rho,
\qquad \rho=|N/2,N/2\rangle\langle N/2,N/2|,
```

with `gamma=0.37` and `N=4,8,16,32,64`. PID explicitly uses
`PIBasis(N,2; sectors=[(N,0)])`; QuantumOptics uses `SpinBasis(N/2)` and
QuantumToolbox uses `jmat(N/2)`. These are the same spin-
`N/2` irrep and the same standard dissipator. Every adapter verifies

```math
|\mathrm{tr}(\mathcal{L}\rho)|\simeq0,
\qquad \|\mathcal{L}\rho\|_F\simeq\sqrt{2}\,\gamma N
```

before writing a result. Hermiticity and the complete analytical
eigenvalue/multiplicity signature of the initial derivative are also checked.
This distinguishes the intended derivative from another trace-zero matrix
with the same norm. PID's equation-(7) basis is orthonormal, so the Euclidean
norm of its coefficient vector is the physical Hilbert-space Frobenius norm
used by the full-space adapters.

The `local_emission_scaling` track computes independent local emission,

```math
\mathcal{L}\rho=\gamma\sum_i\mathcal{D}[\sigma_-^{(i)}]\rho,
```

from the fully excited product state. This process transfers weight between
Schur sectors. PID therefore uses its complete all-sector PI operator space at
`N=2,4,6,8,16`. The other packages are generic quantum-system toolkits, not
all-sector PI backends, so their adapters construct the full tensor-product
Hilbert space and stop at the deliberately conservative `N<=6`. Each adapter
checks the exact initial derivative norm
`gamma*sqrt(N*(N+1))`. The representation column says `all_schur_sectors_pi`
or `full_hilbert`; do not interpret this track as a fixed-spin speed ratio.

`physical_hilbert_dimension` is the dimension `2^N` of the underlying qubit
system in both tracks. `retained_operator_dimension` is the length of the
actual vector acted on: `(N+1)^2` in a fixed spin irrep, the PI coefficient
count in the all-sector backend, or `4^N` in full Hilbert space. Keeping both
columns prevents a restricted representation from being mistaken for the
full operator space.

`retained_bytes` is `Base.summarysize((L,x,y))` for the sparse generator and
the dense input/output vectors used by the hot action, with shared objects
counted once. Setup timing constructs the package basis/state/operators and
sparse generator after compilation warm-up. Apply timing is a warmed
`mul!(y,L,x)` with one evaluation per sample, so it excludes Julia compilation
and solver integration choices.

All adapters reuse an existing `SparseMatrixCSC` without copying it and
convert another matrix representation exactly once. The `backend` and
`action_kind` columns make this explicit. PID setup therefore includes its
symmetric occupation-number lowering and sparse-first materialization, while
the timed action is the same `SparseArrays.mul!` operation used by the other
adapters.

In the matched collective track all three packages ultimately produce
equally sized `SparseMatrixCSC` generators with the same nonzero count. The
hot action therefore invokes the same Julia `SparseArrays.mul!` implementation;
sub-microsecond differences are measurement noise, not a package-specific
dynamics kernel. Only setup is package-specific in that track. Conversely,
local emission compares PI and full-Hilbert representation scaling, not
like-for-like backend speed at fixed retained dimension.

## Running the comparison

From the repository root, prepare all three isolated environments once:

```sh
julia --startup-file=no benchmark/comparison/setup.jl
```

Then run the single-threaded quick comparison:

```sh
julia --startup-file=no benchmark/comparison/run_all.jl
```

Results are written under `benchmark/comparison/results/`, including the
combined `comparison.tsv`. Useful controls are:

```sh
julia --startup-file=no benchmark/comparison/run_all.jl \
  --samples 50 --seconds 2 --threads 1 --output /tmp/pid-comparison
```

Keep CPU power mode, Julia executable, thread count, and background load fixed
when comparing runs. The recorded minimum is useful for low-noise latency;
the median is less sensitive to occasional scheduling interruptions. A
minimum near the BenchmarkTools overhead-correction floor, especially
`0.001 ns`, is unresolved rather than a physical latency and must not enter a
ratio; benchmark a larger batch instead. Since no
manifest is committed, the version, Git, and manifest-hash columns are
essential provenance. Prefer a clean Git checkout. For a dirty checkout, the
worktree hash covers the tracked binary diff and non-ignored untracked files;
archive that diff beside the result. Archive the generated manifests too when
a publication requires exact dependency reconstruction, but keep those
artifacts outside this repository.
