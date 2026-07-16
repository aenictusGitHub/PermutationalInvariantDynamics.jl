# Reusable literature-model constructors

Source: [`paper_models.jl`](paper_models.jl)

This file is a helper module used to keep paper-specific normalizations and
spin conventions in one place. It provides constructors for the Damanet (2016),
Pausch (2024), Kitagawa-Ueda, Huelga, Shammah, Morrison, Meiser, Iemini,
Piccitto, Nakanishi--Sasamoto, and Zhang--Mølmer models, together with the
package's public `spin_matrices` convention. The Nakanishi--Sasamoto helper
also returns the exact balanced-gain/loss Liouvillian spectrum used as a
finite-size regression oracle.

## Use

```julia
using PermutationalInvariantDynamics
include("examples/paper_models.jl")
using .PaperModels

model = PaperModels.kitagawa1993_oat_model(8; chi=0.2)
prepared = compile(model)
```

Each constructor returns a `PIModel` containing its basis and immutable term
tuple. Compile it once, then pass the resulting `CompiledPIModel` to
`solve_dynamics`, `stationary_state`, or `liouvillian_spectrum`. Use
`CollectiveObservablePlan` when the same collective one-particle observable is
evaluated repeatedly. The individual example scripts show complete workflows
for each family and retain explicit matrix exponentiation only where that is
the independent small-system validation method.

## Why use the constructors?

Literature comparisons are sensitive to whether spin operators are Pauli
matrices or Pauli matrices divided by two, and to the definition of
`D[L]`. These constructors encode the convention used by the accompanying
benchmarks. When changing parameters, preserve the documented `N` scaling of
collective rates.

`zhang2018_superradiance_model` returns the decay-only specialization of
Zhang, Zhang, and Mølmer's collective-plus-local master equation.
`zhang2018_radiation_operators` constructs its cavity and free-space fluxes,
``\Gamma_cJ_+J_-`` and ``\gamma_l\sum_i\sigma_+^{(i)}\sigma_-^{(i)}``.
These helpers are used by the trajectory/master-equation comparison without
introducing a plotting or cavity-mode dependency.

The time-crystal constructors make two easily missed convention conversions
explicit. `nakanishi2023_pt_model` converts the paper's dissipator
`D_paper = 2D_std` to the package convention. The normalized magnetizations
in `piccitto2021_interacting_btc_model` give collective jump rates `4Gamma/N`,
and its `Jz^2` interaction is lowered to a two-body term after removing the
irrelevant identity component.

`pausch2024_model` likewise retains the microscopic provenance of its LMG
Hamiltonian: the self part of `Jx^2 - Jy^2` is a `LocalHamiltonian`, and the
cross part is a symmetric `PBodyHamiltonian` with its required factor of two.
This is exactly equivalent to the collective quadratic operator in the paper,
but also allows the returned model to be passed directly to `MeanFieldPlan`
for finite-product or thermodynamic predictions.
