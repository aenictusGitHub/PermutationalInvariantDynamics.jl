# Mean-field dynamics and predictions

The mean-field layer derives a closed equation for the identical one-particle
density matrix \(\sigma\) directly from the physical terms used by a
`PIModel`. It is a product-BBGKY closure,

\[
\rho^{(p)}\approx\sigma^{\otimes p},
\]

and works for a general local dimension \(d\). It is not restricted to qubit
Bloch vectors and never constructs a \(d^N\) state, a PI basis vector, or a
Liouvillian matrix. The evolved state contains only \(d^2\) complex entries;
for a supported \(p\)-body term, the largest scratch matrices are
\(d^p\times d^p\), requiring \(O(d^{2p})\) storage but remaining independent
of \(N\) and the full many-body Hilbert space.

## Quick start

```julia
using LinearAlgebra
using PermutationalInvariantDynamics

N = 100
sx = ComplexF64[0 1; 1 0]
sm = ComplexF64[0 1; 0 0]
basis = PIBasis(N, 2; sectors=[(N, 0)])
model = PIModel(basis, [
    CollectiveHamiltonian(sx / 2; rate=1.3),
    CollectiveJump(sm; rate=0.8 / N),
])

plan = MeanFieldPlan(model; limit=:finite)
sigma0 = ComplexF64[1 0; 0 0]
times = range(0.0, 10.0; length=101)
solution = solve_meanfield(
    plan, sigma0, (first(times), last(times));
    saveat=times, steps_per_interval=16)

population = [real(meanfield_expectation(sigma, sm' * sm))
              for sigma in solution]
```

When no PI calculation is needed, construct the plan directly from the
particle number, local dimension, and physical terms:

```julia
plan = MeanFieldPlan(N, 2, (
    CollectiveHamiltonian(sx / 2; rate=1.3),
    CollectiveJump(sm; rate=0.8 / N),
); limit=:thermodynamic)
```

This form is the efficient route for a very large formal \(N\): it does not
construct Schur geometry merely to obtain a mean-field equation.

## Product-closure equations

The library uses

\[
\mathcal D[L]\sigma
=L\sigma L^\dagger-\tfrac12\{L^\dagger L,\sigma\}.
\]

For a one-particle Hamiltonian term with rate \(r\),

\[
\dot\sigma=-\frac{ir}{\hbar}[h,\sigma].
\]

For a permutation-symmetric \(p\)-particle Hamiltonian summed over unordered
subsets, define

\[
h_{\rm eff}(\sigma)=\mathrm{tr}_{2\ldots p}
\left[h\left(I\otimes\sigma^{\otimes(p-1)}\right)\right].
\]

The finite product closure is

\[
\dot\sigma=-\frac{ir}{\hbar}
{N-1\choose p-1}[h_{\rm eff}(\sigma),\sigma].
\]

A local one-particle jump contributes \(r\mathcal D[L]\sigma\). For
`LocalPBodyJump(L,p)`, let

\[
G(\sigma)=\mathrm{tr}_{2\ldots p}
\left[L\sigma^{\otimes p}L^\dagger\right],\qquad
Q_{\rm eff}(\sigma)=\mathrm{tr}_{2\ldots p}
\left[L^\dagger L
\left(I\otimes\sigma^{\otimes(p-1)}\right)\right].
\]

Its contribution is

\[
r{N-1\choose p-1}
\left(G(\sigma)-\tfrac12\{Q_{\rm eff}(\sigma),\sigma\}\right).
\]

Finally, for a collective one-particle jump
\(J=\sum_iL_i\), write \(\ell=\mathrm{tr}(L\sigma)\). The exact
finite-\(N\) product closure is

\[
\dot\sigma\big|_J=r\left[
\mathcal D[L]\sigma+
\frac{N-1}{2}
\left(\ell^*[L,\sigma]-\ell[L^\dagger,\sigma]\right)
\right].
\]

These equations use the same unordered-subset convention as
`PBodyHamiltonian` and `LocalPBodyJump`.

For the collective \(p\)-particle jump

\[
J=\sum_{|S|=p}L_S,
\]

the dissipator contains ordered pairs of subsets. Define the contracted
operators

\[
E_m(\sigma)=\mathrm{tr}_{m+1\ldots p}
\left[L\left(I^{\otimes m}\otimes
\sigma^{\otimes(p-m)}\right)\right],
\qquad 0\leq m\leq p,
\]

so that \(E_0=\ell=\mathrm{tr}(L\sigma^{\otimes p})\) and
\(E_p=L\), and introduce the cross dissipator

\[
\Phi_{A,B}(R)=ARB^\dagger-
\frac12\left(B^\dagger A R+R B^\dagger A\right).
\]

Only ordered subset pairs whose union contains the retained particle
contribute. If both subsets contain it and have total overlap \(r\), their
number is

\[
a_r={N-1\choose p-1}{p-1\choose r-1}{N-p\choose p-r},
\qquad 1\leq r\leq p.
\]

If exactly the first subset contains it and the remaining overlap is \(r\),
the number is

\[
b_r={N-1\choose p-1}{p-1\choose r}{N-p\choose p-r},
\qquad 0\leq r\leq p-1.
\]

A binomial coefficient is understood to vanish when its lower argument lies
outside its allowed range; this automatically removes overlap classes that do
not fit at small \(N\).

Writing \(\widehat E_r=I_1\otimes E_r\), the exact finite product closure is

\[
\begin{aligned}
\dot\sigma\big|_J=r_J\Bigg[&
\sum_{r=1}^{p}a_r\,
\mathrm{tr}_{2\ldots r}
\Phi_{E_r,E_r}(\sigma^{\otimes r})\\
&+\sum_{r=0}^{p-1}b_r\,
\mathrm{tr}_{2\ldots r+1}
\left(\Phi_{E_{r+1},\widehat E_r}
+\Phi_{\widehat E_r,E_{r+1}}\right)
(\sigma^{\otimes(r+1)})\Bigg].
\end{aligned}
\]

The implementation evaluates every \(E_m\) directly and never constructs an
operator on the union of two subsets. Its largest matrices therefore remain
\(d^p\times d^p\), independent of \(N\). For \(p=1\), the two sums reduce to
the one-particle dissipator and the collective field shown above.

## Supported terms and explicit errors

| PI term | Product mean-field lowering |
|---|---|
| `LocalHamiltonian`, `CollectiveHamiltonian` | Supported |
| `PBodyHamiltonian` | Supported for permutation-symmetric input operators |
| `LocalJump` | Supported |
| `CollectiveJump` | Supported, including the finite self term |
| `CorrelatedLocalJumps`, `CorrelatedCollectiveJumps` | Rejected by the current product-closure compiler; factor the fixed Kossakowski matrix into ordinary jumps first |
| `LocalPBodyJump` | Supported for permutation-symmetric input operators |
| `CollectivePBodyJump(..., 1)` | Equivalent to `CollectiveJump` |
| `CollectivePBodyJump(..., p)` with `p > 1` | Supported through exact finite overlap classes and their leading thermodynamic class |
| `DirectPIHamiltonian`, `DirectPIJump` | Rejected; Schur blocks do not retain a unique microscopic body-order decomposition |
| Operator-valued time functions | Rejected by the prepared path; use fixed operators and time-dependent scalar rates |

An unsupported term raises an error. It is never silently omitted or replaced
by a lower-body process. In particular, a `PIOperator` stored in a direct term
may represent many inequivalent microscopic operators, and a mixed product
state may occupy sectors absent from a restricted basis. Supply an equivalent
local or `PBody` decomposition when a mean-field prediction is required.

## Finite product closure versus thermodynamic closure

`limit=:finite` keeps exact finite-size subset counts. At the initial time it
reproduces the exact one-body derivative of an initially factorized PI state
for every supported term. Subsequent propagation closes newly generated
correlations as products, so it is generally not exact finite-system
dynamics.

`limit=:thermodynamic` retains the leading subset count

\[
{N-1\choose p-1}\longrightarrow
\frac{N^{p-1}}{(p-1)!}
\]

for Hamiltonian and local-jump `p`-body terms. For collective one-particle
jumps it drops the subleading local dissipator and replaces \(N-1\) by
\(N\). Thus a rate written explicitly as \(r=\kappa/N\)
has the finite closure

\[
\frac{\kappa}{N}\mathcal D[L]\sigma
+\kappa\frac{N-1}{N}\,C_L(\sigma),
\]

and thermodynamic closure \(\kappa C_L(\sigma)\), where
\(C_L=(\ell^*[L,\sigma]-\ell[L^\dagger,\sigma])/2\).

The plan cannot infer whether an arbitrary numerical rate was intended to
scale as \(1/N\), \(1/N^{p-1}\), or remain constant. The user must encode the
physical Kac scaling in the term rate. In both modes rates are used exactly as
supplied; `limit=:thermodynamic` changes combinatorial factors but never
inserts a compensating power of \(N\). It is therefore not a symbolic limit of
arbitrary numeric terms.

For a collective \(p\)-body jump, the unique leading class consists of two
disjoint subsets with exactly one containing the retained particle. Its count
is replaced by

\[
b_0={N-1\choose p-1}{N-p\choose p}
\longrightarrow
\frac{N^{2p-1}}{(p-1)!p!}.
\]

All shared-particle and self terms are subleading and are dropped in this
mode. With \(E_1\) and \(\ell=E_0\), the retained contribution is

\[
r_J\frac{N^{2p-1}}{2(p-1)!p!}
\left(\ell^*[E_1,\sigma]-\ell[E_1^\dagger,\sigma]\right).
\]

Consequently, an extensive collective-\(p\)-body model commonly supplies an
explicit rate proportional to \(N^{-(2p-1)}\). The package does not insert
that factor automatically.

## Plans, workspaces, and time dependence

A `MeanFieldPlan` owns copied, read-only operator data and prepared tensor
contractions. Share it across parameter scans or tasks. A
`MeanFieldWorkspace` owns mutable stage and contraction scratch and must be
used by only one task at a time:

```julia
workspace = MeanFieldWorkspace(plan, sigma0)
du = zeros(eltype(workspace), size(sigma0))

meanfield_rhs!(du, plan, sigma0, 0.0, parameters, workspace)
```

After warm-up, the explicit-workspace RHS is the allocation-conscious hot
path. For threaded scans, create one workspace per thread. The convenience
form allocates its result:

```julia
du = meanfield_rhs(plan, sigma0; time=0.0, parameters=parameters)
```

The workspace type is inferred from the state, fixed operators, constant
rates, and Hamiltonian prefactors. Reusing it with a state or destination that
would lose precision raises an error. A scalar rate function must likewise
return a number representable by that workspace type; for example, return a
`Float32` rate in an otherwise `ComplexF32` calculation.

Fixed operators with scalar rates of the form `(t, p) -> rate` remain on the
prepared path. Negative time-dependent rates are accepted, consistently with
the PI dynamics API; the resulting time-local equation need not generate a
completely positive map.

The RHS preserves trace and Hermiticity algebraically. Numerical propagation
does not normalize, symmetrize, or positivity-clip a density matrix. Converge
the time step and inspect trace, Hermiticity, and eigenvalues when these
properties matter.

## Fixed-step and adaptive evolution

`solve_meanfield` is the compact fixed-step interface. It uses a reusable RK4
workspace and returns a `MeanFieldResult` with `times`, `states`, `limit`, and
`algorithm` fields; the result also supports indexing and iteration:

```julia
solution = solve_meanfield(
    plan, sigma0, (0.0, 20.0);
    saveat=0.1, steps_per_interval=32)

sigma_final = last(solution)
```

For repeated propagation into caller-owned storage, use
`meanfield_evolve!`. Its classical RK4 path uses three full one-site
integration matrices rather than retaining four derivatives simultaneously:

```julia
workspace = MeanFieldWorkspace(plan, sigma0)
destination = zeros(eltype(workspace), size(sigma0))
meanfield_evolve!(destination, plan, sigma0, (0.0, 1.0);
                  steps=256, workspace=workspace)
```

For adaptive or stiff integration, use the SciML adapter and add a solver in
the active environment:

```julia
using OrdinaryDiffEq

problem = meanfield_problem(plan, sigma0, (0.0, 20.0);
                            parameters=parameters)
adaptive = solve(problem, Rodas5P(); saveat=0.1)
```

The `ODEProblem` captures one workspace. Construct a separate problem for
each concurrently running solve.

## Product-state observable predictions

For a local observable \(X\),

```julia
local_mean = meanfield_expectation(sigma, X)
moments = meanfield_collective_moments(plan, sigma, X)
```

The returned collective moments use

\[
\langle X_c\rangle=N\langle X\rangle_\sigma,
\qquad
\langle X_c^2\rangle=
N\langle X^2\rangle_\sigma+N(N-1)\langle X\rangle_\sigma^2,
\]

for \(X_c=\sum_iX_i\). A symmetric \(p\)-body sum is predicted with

```julia
value = meanfield_pbody_expectation(plan, sigma, Xp, p)
```

which evaluates

\[
{N\choose p}\mathrm{tr}
\left(X_p\sigma^{\otimes p}\right).
\]

These are product-closure predictions. They omit connected correlations and
must not be presented as exact moments of an evolved finite PI state.

## Fixed points and linear stability

An autonomous nonlinear closure can have several stable fixed points,
unstable solutions, limit cycles, or no attracting stationary point. The
stationary helper relaxes from a supplied seed and therefore selects a basin:

```julia
fixed = meanfield_stationary_state(
    plan, sigma_seed; return_info=true)
sigma_fixed = fixed.state
```

Inspect its convergence and RHS residual before using the state. Failure to
converge is not evidence that no fixed point exists; change the seed and
integration controls, or use a dedicated nonlinear solver.

Linearization is performed in the real traceless-Hermitian tangent space of
dimension \(d^2-1\):

```julia
J = meanfield_jacobian(plan, sigma_fixed)
stability = meanfield_stability(plan, sigma_fixed)

stability.stable
stability.eigenvalues
```

Negative real parts identify a linearly attracting fixed point. Eigenvalues
near the imaginary axis require tolerance and time-step convergence checks;
the linear test does not determine the nonlinear basin of attraction. Fixed
points and autonomous stability analysis are not substitutes for Floquet
analysis of a periodically driven mean-field equation.

See `examples/meanfield_time_crystal.jl` for finite and thermodynamic closures
compared with an exact matrix-free PI calculation.
