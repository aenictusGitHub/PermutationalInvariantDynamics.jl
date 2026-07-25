# Quantum regression and optical correlations

The quantum regression backend evolves operator-inserted PI states with the
same compiled Liouvillian used for density-matrix dynamics. It remains in the
PI coefficient space and does not construct a $d^N$ state, a
$d^{2N}$ superoperator, or a dense Liouvillian.

## Convention

For a stationary or specified initial PI state $\rho$, the two-operator
function is

```math
C_{AB}(\tau)=\mathrm{tr}\!\left[
A e^{\mathcal L\tau}(B\rho R)\right],
```

where the optional right insertion $R$ is omitted by default. The first
operator is not implicitly adjointed. For example, the optical first-order
correlation $\langle c^\dagger(\tau)c(0)\rangle$ uses
`A=adjoint(c)` and `B=c`.

This explicit convention matters because the general package operation
`expectation(rho, A)` is $\mathrm{tr}(A^\dagger\rho)$. Internally the
correlation readout is arranged so that it instead evaluates exactly
$\mathrm{tr}(A\rho_{\rm conditional})$.

## Prepared workflow

```julia
prepared = compile(model; backend=:matrixfree)
plan = CorrelationPlan(prepared, A, B)
workspace = CorrelationWorkspace(plan; krylovdim=40)

values = two_time_correlation(
    plan, rho, delays;
    steps_per_interval=64,
    workspace=workspace,
)
```

`CorrelationPlan` is immutable and owns copied, read-only physical Schur
blocks for the insertions. `CorrelationWorkspace` owns the conditional state,
block-product scratch, one `EvolutionWorkspace`, and shifted-GMRES storage.
A plan may be shared between tasks; a workspace must be owned by one task and
reused only sequentially.

For time-domain work without resolvent spectra, omit the shifted-GMRES basis:

```julia
time_workspace = CorrelationWorkspace(plan; mode=:time)
values = two_time_correlation(
    plan, rho, delays; workspace=time_workspace)
```

The default `mode=:both` remains appropriate when the same workspace will
also be passed to `stationary_correlation_spectrum`. A time-only workspace is
also accepted by the sampled `correlation_spectrum_fft` route; it is rejected
explicitly by the shifted-GMRES spectrum rather than growing behind the
caller's back.

The representation and insertion are exact within the retained PI basis. The
default time-domain integrator is numerical fixed-step RK4. Increase
`steps_per_interval` and verify convergence when quantitative integration
error matters.

Time and frequency arithmetic follows the prepared plan's real precision. A
matching delay vector is reused directly; a narrower vector is converted once
to the plan precision. Wider floating delays, frequencies, and explicit
tolerances are rejected instead of being silently narrowed, while integer
inputs must be exactly representable.

## Delayed intensity correlations

For a PI lowering or output operator `c`,
`delayed_second_order_correlation` evaluates

```math
G^{(2)}(\tau)=\mathrm{tr}\!\left[
c^\dagger c\,e^{\mathcal L\tau}(c\rho c^\dagger)\right].
```

With `normalized=true`, it returns
$g^{(2)}(\tau)=G^{(2)}(\tau)/I^2$, where
$I=\mathrm{tr}(c^\dagger c\rho)$. This normalization is stationary: the
routine validates unit trace and stationarity, and raises when $I=0$.
Choose `normalized=false` for a nonstationary or unnormalized result.

## Infinite-time and sampled spectra

`stationary_correlation_spectrum` evaluates

```math
S_{AB}(\omega)=\int_0^\infty e^{-i\omega\tau}
C^{\rm conn}_{AB}(\tau)\,d\tau
```

through matrix-free shifted GMRES. The exact stationary product is subtracted
from the conditional seed. A rank-one trace constraint regularizes the zero
mode without changing the trace-zero connected solution. The function returns
the complex one-sided transform: it does not apply a factor two or discard the
imaginary part. The disconnected stationary spectrum contains a Dirac delta,
so requesting it as an ordinary resolvent function is rejected.

For a frequency grid, the default `solver=:auto` groups shifts into bounded
batches and solves them from one shared Arnoldi factorization. Columns that do
not meet the requested full-space residual are retried with restarted
single-shift GMRES. Set `solver=:sequential` for the historical path or
`solver=:multishift` to require shared-Arnoldi convergence without fallback.
`shared_memory_budget` and `multishift_batchsize` bound live solution storage;
the result reports `shared_batches` and `fallback_frequencies`. Repeated calls
may pass a compatible `MultiShiftGMRESWorkspace`.
The supplied workspace's fixed `nshifts` is the batch size. Automatic mode
handles a final short remainder sequentially; forced multi-shift mode requires
an exact number of full batches.

`optical_spectrum(L, rho, c, frequencies)` is the convenience form with
$A=c^\dagger$ and $B=c$.

For a finite uniformly sampled record, `correlation_spectrum_fft` provides a
dependency-free radix-two FFT. It uses the same
$e^{-i\omega\tau}$ sign, trapezoidal endpoint weights, and optional zero
padding. Its result is a finite-window integral. When called with a plan and a
stationary state, it subtracts the exact stationary offset instead of guessing
it from a tail sample.

See the runnable [quantum-regression example](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/quantum_regression.md)
for a pumped emitter with analytical time-domain, antibunching, and spectral
checks.

## API

```@docs
CorrelationPlan
CorrelationWorkspace
two_time_correlation
two_time_correlation!
delayed_second_order_correlation
second_order_correlation
stationary_correlation_spectrum
optical_spectrum
correlation_spectrum_fft
```
