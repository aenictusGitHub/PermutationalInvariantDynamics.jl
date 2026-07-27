# 90-second quickstart

This page gets from a physical model to checked results with the guided
interface. It keeps the complete PI basis, lets the package choose a
memory-bounded backend, and records the numerical route that was used.

## 1. Install and load

```julia
using Pkg
Pkg.add(url="https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl")

using PermutationalInvariantDynamics
```

The package supports Julia 1.10 and later. See [Installation and environment
checks](installing.md) for a reproducible project environment and the
registered-package command once the first release reaches General.

## 2. Choose a tested model

```julia
model = Models.driven_qubits(8)
rho0 = computational_product_state(model.basis, 2)
```

Local qubit order is $(|g\rangle,|e\rangle)$. The model uses the complete PI
operator basis, so identical local noise may populate every Schur sector.
Browse `Models.find()` or the [example gallery](example_gallery.md) for other
starting points.

## 3. Explain, then solve

```julia
study = PIStudy(
    model;
    task=:dynamics,
    initial_state=rho0,
    tspan=(0.0, 4.0),
    saveat=0.1,
)

plan = explain(study)
result = solve(study)
```

`explain` performs validation and resource planning without running the
solver. `solve` delegates to the same tested implementation as
`solve_dynamics`; it does not weaken tolerances or bypass the default
512 MiB memory guard.

Inspect the outcome through the common result interface:

```julia
summary = summarize(result)
times = result_times(result)
states = result_states(result)
final = result_final_state(result)
```

Unavailable output is reported explicitly; the accessors never recompute it.

## 4. Compute a stationary state

```julia
steady_study = PIStudy(model; task=:steady_state)
steady = solve(steady_study)

@assert result_converged(steady) === true
rho_ss = result_final_state(steady)
@assert diagnostics(rho_ss).valid
```

The convergence flag describes the numerical solver. `diagnostics` separately
checks the returned density operator; neither call repairs invalid data.

## 5. If something fails

```julia
doctor()
check(study)

try
    solve(study)
catch error
    show(explain_failure(error))
    rethrow()
end
```

These commands return structured issue codes and actionable suggestions. Keep
the original exception and stack trace when reporting a problem.

## Next

- [Choose the right workflow](choosing_workflow.md) for composite ensembles,
  local or shared pseudomodes, PI--HEOM, and PI--HOPS.
- [Build a custom model](getting_started.md) to translate Hamiltonians and
  jump channels term by term.
- [Generate a model from supported LaTeX](model_code_generator.md).
- [Results, tables, plots, and exports](result_outputs.md).
- [Architecture and efficient workflows](architecture.md) before scaling up.
