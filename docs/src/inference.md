# Parameter inference

The inference layer fits model predictions to real observations while keeping
the derivative route and local identifiability explicit. Its objective is

```math
\chi^2(\theta)/2
=\frac{1}{2}\sum_j
\left[
\frac{f_j(\theta)-y_j}{\sigma_j}
\right]^2 ,
```

where $\sigma_j$ are supplied observation standard deviations.

The core implementation is dependency free. It uses a finite,
memory-preflighted Levenberg--Marquardt search and never changes the model,
normalizes a failed state, or accepts an unconverged stationary solve.

## Fit a generic predictor

```julia
design = [
    1.0  2.0
    3.0 -1.0
    1.0  0.0
]
observations = [0.2, 1.1, 0.4]

problem = LeastSquaresInferenceProblem(
    parameters -> design * parameters,
    observations,
    [0.1, 0.1];
    jacobian=parameters -> design,
    standard_deviations=[0.03, 0.05, 0.02],
    parameter_names=(:drive, :detuning),
    lower_bounds=[0.0, -2.0],
    upper_bounds=[2.0, 2.0],
)

result = fit_parameters(
    problem;
    derivative_method=:auto,
    maxiter=50,
    memory_budget=512 * 1024^2,
)
```

Inspect convergence instead of using the fitted vector alone:

```julia
result.converged
result.termination
result.objective
result.derivative_method
result.identifiability.identifiable
result.identifiability.condition_number
result.identifiability.covariance
```

When `jacobian` is absent, `:auto` uses finite differences and records every
central or one-sided scheme and step. Bounds determine whether central,
forward, or backward differences are possible. No complex-step or automatic
differentiation method is claimed implicitly.

## Steady-state observable fitting

For PI models, an analytic generator derivative can be propagated through the
stationary equation,

```math
\mathcal L\,\partial_\mu\rho_{\mathrm{ss}}
=-(\partial_\mu\mathcal L)\rho_{\mathrm{ss}},
\qquad
\mathrm{tr}(\partial_\mu\rho_{\mathrm{ss}})=0.
```

Prepare a model and one derivative generator per parameter:

```julia
basis = PIBasis(N, 2)
spin = spin_matrices()
lowering = ComplexF64[0 1; 0 0]
excited = collective_operator(
    basis,
    ComplexF64[0 0; 0 1],
)

model_builder = parameters -> PIModel(basis, (
    CollectiveJump(lowering; rate=gamma),
    LocalHamiltonian(spin.jx; rate=parameters[1]),
))

derivative_builder = (parameters, model) -> (
    PIModel(basis, (
        LocalHamiltonian(spin.jx; rate=1.0),
    )),
)

problem = steady_state_inference_problem(
    model_builder,
    (excited,),
    measured_populations,
    [0.2];
    derivative_builder,
    parameter_names=(:drive,),
    lower_bounds=[0.0],
    upper_bounds=[2.0],
    solver_options=(
        algorithm=GMRESAlgorithm(
            krylovdim=30,
            maxiter=500,
        ),
    ),
)

result = fit_parameters(problem)
```

With `derivative_builder`, `:auto` uses
`implicit_steady_state_gradient`. The already computed stationary state is
reused for its tangent solve, and its unscaled residual and trace diagnostics
remain in the inference metadata. Without it, the same workflow uses and
reports finite differences of complete stationary solves.

`model_builder` may return a `PIModel`, `CompiledPIModel`, or
`SpecializedPIModel`. Returning specializations of a shared
`CompiledPIModelFamily` avoids rebuilding fixed Schur geometry during a fit.

## Fisher and identifiability diagnostics

At a chosen parameter point:

```julia
report = parameter_identifiability(problem, result.parameters)
```

For the weighted Jacobian
$\widetilde J_{j\mu}=J_{j\mu}/\sigma_j$, the reported local Fisher matrix is

```math
F=\widetilde J^\mathsf{T}\widetilde J .
```

The report includes:

- singular values and the explicit numerical-rank threshold;
- local numerical rank and a full-column-rank `identifiable` flag;
- condition number;
- covariance and correlation matrices when the Jacobian is full rank;
- a truncated-SVD pseudocovariance for diagnostics when it is not.

The implementation requests a thin SVD, so a tall data set does not retain an
unused observations-by-observations left-singular matrix. Weighting,
standardized residuals, objectives, Fisher products, covariance
normalizations, and accepted parameter updates are checked for nonfinite
overflow and nonzero-to-zero underflow. A failure asks for wider inference
data instead of continuing with a silently corrupted diagnostic.

The pseudocovariance of a rank-deficient fit is not an uncertainty
certificate. Likewise, this local Fisher analysis does not establish global
identifiability, resolve multimodality, or include model discrepancy.

## Resource and convergence boundaries

The inference memory budget covers its retained Jacobian, Fisher,
factorization, and optimization arrays. User builders, compiled models,
stationary solvers, and external data have their own resource contracts and
are listed as exclusions in the result.

`fit_parameters` has finite `maxiter` and `max_trials` bounds. It returns
`converged=false` with an explicit termination reason when those conditions
are not met. Use `require_convergence=true` when a nonconverged result should
raise.

## API

```@docs
LeastSquaresInferenceProblem
ParameterIdentifiabilityReport
ParameterInferenceResult
steady_state_inference_problem
parameter_identifiability
fit_parameters
```
