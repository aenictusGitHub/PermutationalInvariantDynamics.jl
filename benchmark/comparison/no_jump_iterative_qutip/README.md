# No-jump-resolvent iterative PI steady-state comparison with QuTiP PIQS

This optional cross-language benchmark compares the prepared No-jump-resolvent iterative solver in
`PermutationalInvariantDynamics.jl` with a sparse direct solve built from the
public QuTiP 5.2 PIQS Liouvillian. It records raw setup and warmed-solve
samples, versions, machine information, dimensions, tolerances, true
stationary residuals, positivity/Hermiticity checks, and a matched observable.
It is a reproducibility harness, not a universal package ranking.

The primary metric is `time_to_solution`: fresh model and solver setup followed
by the first stationary-state solve in the same timed region. Setup and
prepared solve-only samples are retained as secondary diagnostics.

## Matched model

Both adapters solve the complete all-Schur-sector PI master equation

```math
\dot\rho=-i[\Omega J_x+\Delta J_z,\rho]
+\gamma_\downarrow\sum_i\mathcal D[\sigma_-^{(i)}]\rho
+\gamma_\uparrow\sum_i\mathcal D[\sigma_+^{(i)}]\rho,
```

with `Omega=0.70`, `Delta=0.23`, `gamma_down=0.31`, and
`gamma_up=0.09`. Independent emission and pumping transfer population between
total-spin sectors, so a single spin-`N/2` calculation would not solve this
problem. The two retained physical operator spaces have the same dimension,

```math
\sum_j(2j+1)^2=\binom{N+3}{3}.
```

The QuTiP PIQS API returns a superoperator embedded in the square of the
concatenated Dicke Hilbert space. That embedding also contains cross-sector
coherences, which are outside the PI density-operator space and can introduce
irrelevant null directions in a generic steady-state solve. The adapter checks
that the block-diagonal subspace is exactly invariant, restricts the public
PIQS generator to those physical coordinates, and applies the exact diagonal
similarity transformation from PIQS multiplicity-weighted blocks to PID's
orthonormal equation-(7) coefficients. It then adds one trace row and calls
`scipy.sparse.linalg.spsolve`. The one-time conversion is included in setup,
not warmed solve. This deliberately gives QuTiP a stronger
baseline than solving its larger embedding. The adapter is labelled
`QuTiP PIQS + SciPy direct`; it is not presented as an unmodified call to
`qutip.steadystate`.

The direct comparison explicitly calls SciPy `spsolve(...,
use_umfpack=False)`, so its recorded backend is SuperLU and it refactors the
trace-augmented system on every solve-only sample. A separate
`QuTiP-prepared` baseline constructs `splu` during setup and times only its
reusable triangular solve. This makes the repeated-solve tradeoff visible
instead of conflating sparse factorization with prepared application. The
active native thread pools and their observed thread counts are recorded with
every Python row through `threadpoolctl`.

The harness also probes the official high-level
`qutip.steadystate(..., method="direct", solver="spsolve", use_rcm=True)`
route on the untrimmed PIQS embedding. Successful, fully validated sizes are
recorded as separate `QuTiP-public` solve rows. If the padded embedding is
singular or the result fails a physical check, an `unavailable` row records
the exception and no timing ratio is formed. The exact block-restricted
baseline remains the primary comparison because it gives both packages the
same physical equation-(7) coordinates.
`retained_operator_dimension` is therefore the common physical coordinate
count in primary PID/QuTiP rows, but the full `nds^2` ambient count in
`QuTiP-public` rows.

The PID adapter prepares `NoJumpIterativePlan` and one fixed-capacity
`NoJumpIterativeWorkspace`, then defaults to the restarted-Arnoldi fixed-point route.
Use `--pid-method gmres` to benchmark trace-deflated right-preconditioned
GMRES instead. The fixed-point task workspace uses only its required scalar
GMRES-scratch capacity (`krylovdim=1`, `recycle_dim=0`), while its distinct
outer Arnoldi subspace has dimension `60`. The GMRES route uses its separately
declared `60`/`8` workspace capacities. Setup includes basis/lowering,
sectorwise no-jump Schur factorizations, and workspace construction. Warmed
solve samples reuse all of that preparation. QuTiP generator setup includes
PIQS construction, exact restriction, coordinate conversion, and trace-row
construction.

Every solution is checked against the true undeflated Liouvillian. Residuals
are reported as the largest entry of each physical per-copy Schur block. The
observable `real(<Jz>)/N` is checked both across packages and against the
analytic one-qubit optical-Bloch stationary value. The default `5e-7`
validation threshold is recorded in every raw row; QuTiP 5.2 PIQS generator
coefficients, rather than its direct linear solve, set the observed residual
floor for some sizes.

The `time_to_solution` and warmed-solve clocks follow the public calls being
compared. In
particular, `no_jump_iterative_steady_state(...; return_info=true)` computes its true
physical residual and state diagnostics before returning, so that work is
inside the PID time. The primary SciPy clock stops when `spsolve` returns; the
same residual, trace, positivity, Hermiticity, observable, and generator
fingerprint checks are then performed outside its clock. Consequently the
reported solve-time ratio is conservative for PID rather than an estimate
that silently removes PID's built-in certification cost. This timing-scope
asymmetry is also repeated in the raw-row notes.

As a separate model-identity guard, both adapters apply their equation-(7)
generator to the same deterministic complex probe and record its norm and a
second complex checksum. The combined runner rejects the results before
forming timing ratios if these fingerprints differ beyond their documented
floating tolerance. This checks much more of the generator than the single
stationary observable alone.

## Run

Create an optional Python virtual environment without changing the Julia
package environment:

```sh
python3 -m venv benchmark/comparison/no_jump_iterative_qutip/.venv
benchmark/comparison/no_jump_iterative_qutip/.venv/bin/pip install -r \
  benchmark/comparison/no_jump_iterative_qutip/requirements.txt
```

Run the conservative matrix on one numerical thread:

```sh
PYTHON=benchmark/comparison/no_jump_iterative_qutip/.venv/bin/python \
  julia --startup-file=no benchmark/comparison/no_jump_iterative_qutip/run_all.jl \
  --mode quick
```

The checked-in requirements reproduce the initially validated QuTiP 5.2.0,
NumPy 1.25.2, SciPy 1.11.2, and threadpoolctl 3.2.0 environment. A different
compatible environment may be selected with `--python PATH`; the actual
versions are always written to the output.

Useful controls are:

```sh
julia --startup-file=no benchmark/comparison/no_jump_iterative_qutip/run_all.jl \
  --mode full --samples 7 --warmups 2 \
  --sizes 8,16,24,32,40,48 \
  --pid-method fixed_point --python python3 --output /tmp/no-jump-iterative-qutip
```

Generated files are ignored under
`benchmark/comparison/results/no_jump_iterative_qutip/`:

- `pid_raw.tsv` and `qutip_raw.tsv` retain every timing sample;
- `raw.tsv` combines the identically shaped records; and
- `summary.tsv` reports the primary time-to-solution median, the distinct
  setup and solve-only medians, and descriptive ratios `QuTiP time / PID
  time`. It includes the official public-solver median only at sizes where
  that independent probe succeeded.

A ratio above one means PID was faster for that phase on that run. Quote the
fresh-setup `time_to_solution` ratio first. Prepared no-jump-resolvent iterative,
refactor-each-call `spsolve`, and prepared-`splu` solve-only ratios answer
different repeated-workload questions and must retain those labels. Never
quote a ratio without the raw table: small problems are dominated by fixed
overhead, sparse-direct and iterative scaling cross at workload-dependent
sizes, and results from different hardware, versions, tolerances, or power
states are not interchangeable.

API references: [QuTiP 5.2 PIQS guide](https://qutip.readthedocs.io/en/v5.2.0/guide/dynamics/dynamics-piqs.html),
[`qutip.steadystate` 5.2](https://qutip.readthedocs.io/en/v5.2.0/apidoc/solver.html#qutip.steadystate),
and [SciPy `spsolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html).
