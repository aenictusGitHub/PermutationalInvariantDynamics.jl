# Bath fitting and hierarchy preparation

PI--HEOM and PI--HOPS consume a finite exponential correlation,

```math
C(t)=\sum_{k=1}^{K} c_k e^{-\nu_k t},
\qquad \mathrm{Re}\,\nu_k>0.
```

The bath-preparation workflow turns either correlation samples or a sampled
spectral density into this representation. It deliberately keeps three
convergence questions separate:

1. convergence of the frequency quadrature and spectral cutoff;
2. residual and identifiability of the finite exponential fit;
3. convergence of the HEOM/HOPS hierarchy, time step, and trajectory count.

A small fitting residual proves only item 2.

## Fit correlation samples

If physically motivated poles are known, fitting only their coefficients is
the most reliable route:

```julia
times = range(0.0, 12.0; length=401)
correlation = measured_or_analytic_correlation.(times)
poles = ComplexF64[
    0.35,
    0.8 + 1.2im,
    0.8 - 1.2im,
]

fit = fit_bath_correlation(
    times,
    correlation;
    poles,
    weights=ones(length(times)),
    rtol=1e-7,
)

fit.report.converged
fit.report.relative_residual
fit.report.condition_number
fit.report.identifiable
```

The fit is linear in the coefficients. No nonlinear optimizer, random
initialization, or unbounded search is hidden in this path.

If the poles are not known, provide a finite candidate set and a maximum term
count:

```julia
candidates = ComplexF64[
    decay + im * frequency
    for decay in exp10.(range(-2, 1; length=20))
    for frequency in range(-3, 3; length=25)
]

fit = fit_bath_correlation(
    times,
    correlation;
    candidate_poles=candidates,
    nterms=8,
    rtol=1e-5,
    memory_budget=512 * 1024^2,
)
```

This uses deterministic residual-minimizing forward selection followed by a
final weighted least-squares solve. Candidate count and retained dense storage
are checked against `memory_budget`. Supplying neither pole keyword constructs
a bounded positive-real grid; that convenience is intended for
nonoscillatory correlations.

The following fields deserve inspection before hierarchy construction:

- `converged`: the requested weighted residual was reached;
- `identifiable`: the selected exponential columns have full numerical rank;
- `condition_number`: local coefficient sensitivity;
- `stable`: every pole has positive real decay;
- `hops_stationary_ou_compatible`: coefficients work with the built-in
  stationary Ornstein--Uhlenbeck HOPS path.

An explicit `ridge` adds Tikhonov regularization. Its value is never selected
automatically, and the report records it.

The fitting precision is promoted from the samples, poles, weights, and
explicit floating-point `ridge`, `atol`, `rtol`, and `rank_rtol` values.
Thus a `Float64` fitting parameter is never silently narrowed into a
`Float32` fit. Leaving `rtol=nothing` selects `1e-6` directly in the inferred
fit precision, so the default does not widen otherwise-`Float32` data.

## Start from a spectral density

For the convention

```math
C(t)=\frac{1}{\pi}\int_0^\infty J(\omega)
\left[
\coth\left(\frac{\beta\omega}{2}\right)\cos(\omega t)
-i\sin(\omega t)
\right]d\omega ,
```

prepare finite trapezoidal samples first:

```julia
omega = range(1e-4, 20.0; length=20_001)
J = @. 2lambda * gamma * omega / (gamma^2 + omega^2)
times = range(0.0, 12.0; length=401)

samples = correlation_from_spectral_density(
    omega,
    J,
    times;
    inverse_temperature=beta,
)

fit = fit_bath_correlation(
    samples;
    candidate_poles=candidates,
    nterms=8,
    rtol=1e-5,
)
```

The frequency grid must be strictly positive. At zero frequency the limit of
$J(\omega)\coth(\beta\omega/2)$ is model dependent, so the routine refuses to
invent it. Refine the spacing, upper cutoff, and time window independently.
Every thermal factor, integrand, weighted contribution, accumulator, and final
quadrature value is checked for finite arithmetic. Overflow or nonzero
underflow raises with rescaling/wider-precision guidance.

`fit_bath_from_spectral_density` combines these two calls while keeping their
options in separate named tuples:

```julia
fit = fit_bath_from_spectral_density(
    omega,
    J,
    times;
    correlation_options=(inverse_temperature=beta,),
    fit_options=(
        candidate_poles=candidates,
        nterms=8,
        rtol=1e-5,
    ),
)
```

## Construct HEOM or HOPS baths

For a Hermitian PI coupling:

```julia
basis = PIBasis(N, 2)
coupling = collective_operator(basis, spin_matrices().jz)

heom_bath = prepare_heom_bath(coupling, fit)
heom_plan = HEOMPlan(model, heom_bath; max_depth=4)
```

`prepare_heom_bath` rejects an unconverged or rank-deficient fit by default.
The explicit `accept_unconverged=true` and
`accept_rank_deficient=true` switches are intended for controlled convergence
studies, not as automatic repairs.

HOPS preparation follows the same contract:

```julia
hops_bath = prepare_hops_bath(
    coupling,
    fit;
    require_stationary_ou=true,
)
hops_plan = HOPSPlan(hamiltonian, hops_bath; max_depth=4)
```

`require_stationary_ou=true` is appropriate when using the built-in
stationary noise generator. General signed or complex decompositions need an
explicit covariance-correct noise provider.

The bath fit stores no hierarchy workspace, does not construct a full
$d^N$ state, and does not cache mutable process-global data.
Known-length input iterators are memory-preflighted before they are consumed.
Unknown-length iterators are collected under an incremental bound. Raw input
copies, converted arrays, dense fitting work, and returned fit arrays are
included conservatively in `estimated_peak_bytes`.

## API

```@docs
BathCorrelationSamples
BathFitResult
correlation_from_spectral_density
fit_bath_correlation
fit_bath_from_spectral_density
prepare_heom_bath
prepare_hops_bath
```
