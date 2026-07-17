# Permutationally invariant HEOM

The hierarchy equations of motion (HEOM) backend adds finite-memory bosonic
environments to an otherwise time-local PI system model. Every auxiliary
density operator (ADO) is stored in the same PI Schur coordinates as the
physical density operator. Consequently, the method enlarges the calculation
by the hierarchy size, but it never reconstructs a matrix of dimension
`d^N`.

The current backend is intended for bath correlations represented by finite
sums of exponentials. It provides a precise low-level convention rather than
silently choosing a Drude, Matsubara, Padé, temperature, or spectral-density
decomposition. Those decompositions determine the exponential coefficients
supplied by the user.

## Convention

For bath `b`, the coefficients supplied to [`HEOMBath`](@ref) define the
left correlation

```math
C_b^L(t)=\sum_{k\in b}\ell_k e^{-\nu_k t},\qquad t\geq0.
```

The physical conjugate correlation is

```math
C_b^R(t)=C_b^L(t)^*.
```

Both are stored on one prepared pole list,
`C_b^R(t)=sum_k r_k exp(-nu_k*t)`. Real poles receive
`r_k=conj(ell_k)`. Exact complex-conjugate pole pairs are cross-paired. If a
complex pole has no exact conjugate partner, the constructor appends its
conjugate with zero left coefficient and the required right coefficient.
Thus a single input complex exponential normally becomes two prepared HEOM
poles while leaving `C_b^L` unchanged. This completion is exact rather than
tolerance based.

The advanced `right_coefficients` keyword supplies `r_k` explicitly on the
original pole list and disables completion. It is useful for already prepared
left/right decompositions, but the library then does not certify
`C_b^R=C_b^{L*}`; an inconsistent choice can destroy Hermiticity of the root
ADO and is never repaired.

The coupling `Q_b` is a fixed Hermitian PI operator. All coefficients and
`nu_k` may be complex, but `Re(nu_k)` must be strictly positive. Flatten all
prepared exponentials from all baths into one index `k`, and let `b(k)`
identify their bath. The unscaled hierarchy implemented by [`HEOMPlan`](@ref)
is

```math
\begin{aligned}
\dot\rho_{\boldsymbol n}={}&
\mathcal L_S\rho_{\boldsymbol n}
-\sum_k n_k\nu_k\rho_{\boldsymbol n}\\
&-i\sum_k[Q_{b(k)},\rho_{\boldsymbol n+\boldsymbol e_k}]\\
&-i\sum_k n_k\left(
 \ell_k Q_{b(k)}\rho_{\boldsymbol n-\boldsymbol e_k}
 -r_k\rho_{\boldsymbol n-\boldsymbol e_k}Q_{b(k)}
\right).
\end{aligned}
```

`L_S` can itself contain Hamiltonian and Markovian dissipative terms. The root
ADO `rho_0` is the physical reduced state. Auxiliary ADOs generally have
neither unit trace, Hermiticity, nor positivity and must not be interpreted as
density matrices.

The hierarchy retains `sum(n_k) <= D`, where `D=max_depth`. Upward ADOs beyond
that boundary are set to zero. This is the `terminator=:none` hard truncation;
no Markovian or time-local terminator is currently inferred. Results must be
converged in `max_depth`.

For `K` exponential terms, the number of retained ADOs is exactly

```math
M=\binom{K+D}{D},
```

and the coordinate dimension is `M * length(basis)`. Setup checks this count
with exact integer arithmetic before allocation.

The hierarchy state therefore stores `M * n_PI` complex coordinates. The plan
also stores `O(M*K)` upward/downward topology and one dense physical coupling
block per selected PI sector and bath. PI compression removes the system's
`d^N` scaling; it does not remove the combinatorial hierarchy growth or the
dense cost of a large retained Schur block.

## Basic workflow

The following is a one-qubit exponentially correlated dephasing bath. The
same code works for a larger PI ensemble; only the PI basis and collective
coupling operator change.

```julia
using PermutationalInvariantDynamics

basis = PIBasis(1, 2)
spin = spin_matrices()
Q = collective_operator(basis, spin.jz)

# C(t) = 0.30 exp(-1.20t)
bath = HEOMBath(Q, 0.30, 1.20)
system = PIModel(basis, ())
plan = HEOMPlan(system, bath; max_depth=4)

rho0 = iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2))
hierarchy0 = heom_initial_state(plan, rho0)
hierarchy_t = heom_evolve(plan, hierarchy0, (0.0, 0.7); steps=200)
rho_t = heom_reduced_state(hierarchy_t)
```

[`heom_initial_state`](@ref) constructs the conventional factorized initial
condition: `rho_0` is supplied by the user and every auxiliary ADO is zero.
Use [`heom_ado`](@ref) to inspect a detached auxiliary operator by position or
occupation vector.

```julia
first_auxiliary = heom_ado(hierarchy_t, [1])
indices = heom_multiindices(plan)
```

For several baths, pass a tuple. Each bath may contain several exponentials:

```julia
bath_z = HEOMBath(Qz, [c1, c2], [nu1, nu2])
bath_x = HEOMBath(Qx, [c3], [nu3])
plan = HEOMPlan(system, (bath_z, bath_x); max_depth=4)
```

The occupation-vector entries follow the prepared flattened order: all terms
of `bath_z`, including any automatically appended conjugate poles, followed
by all prepared terms of `bath_x`.

If a bath decomposition already provides both actions on one common pole
list, pass them explicitly:

```julia
bath = HEOMBath(Q, left, poles; right_coefficients=right)
```

Here `left`, `right`, and `poles` must have the same length. This route does
not append poles or impose a conjugacy relation.

## Repeated and matrix-free calculations

[`HEOMPlan`](@ref) is immutable prepared data. [`HEOMWorkspace`](@ref) owns
only application scratch: a system-Liouvillian workspace and two PI-sized
vectors for left/right coupling actions. Reuse one workspace per task:

```julia
work = HEOMWorkspace(plan)
y = similar(hierarchy0.data)
apply!(y, plan, hierarchy0.data, work)  # autonomous system
```

[`HEOMEvolutionWorkspace`](@ref) additionally owns the five hierarchy-sized
vectors required by fixed-step RK4. Passing it to [`heom_evolve!`](@ref)
avoids repeated allocations across a parameter or convergence scan.

```julia
evolution_work = HEOMEvolutionWorkspace(plan)
destination = copy(hierarchy0)
heom_evolve!(destination, plan, hierarchy0, (0.0, 0.5);
             steps=200, workspace=evolution_work)
```

The plan scalar type is promoted from the prepared system, couplings, both
sets of bath coefficients, and frequencies. Sources, destinations, times, and step counts
must be representable without narrowing; compiled matrix-free system scratch
cannot be widened after preparation. Build the system and hierarchy at the
wider precision instead. Plans may be shared read-only, but each concurrent
application or evolution requires its own workspace.

Integer bath coefficients and frequencies are accepted only when their exact
values survive conversion to that prepared scalar type. Pass a suitably
precise `BigFloat` value (and prepare the system at matching precision) when a
large integer is not exactly representable; it is never silently rounded.

The bath owns a detached copy of its coupling, and a raw matrix supplied as
the system generator is copied into the plan. Subsequent mutation of either
caller object therefore cannot change prepared HEOM dynamics.

Hierarchy truncation is independent of the RK4 error. Use
[`heom_depth_convergence`](@ref) to compare successive depths in the
Hilbert--Schmidt norm of the final reduced state:

```julia
report = heom_depth_convergence(
    system, bath, rho0, (0.0, 0.7);
    depths=(2, 4, 6), steps=200,
    atol=1e-9, rtol=1e-7, consecutive=2)

collect(skipmissing(report.pairwise_errors))
report.pairwise_converged
report.converged
```

The system model is compiled once, and every truncated plan shares the same
prepared system and coupling blocks. Intermediate results retain only the PI
root states; `last(report.results).hierarchy` is the sole complete saved
hierarchy, while each raw result exposes its ADO count, dimension, trace drift,
and elapsed time. The standard [`ConvergenceStudyResult`](@ref) decision uses
the final `consecutive` requested depth pairs and never normalizes or repairs
either state. Repeat the report with a finer time step before attributing its
entire difference to hierarchy depth. The reported trace error is drift from
the supplied initial trace, not an assumption that the input was normalized.

[`heom_liouvillian`](@ref) adapts the plan to the package's synchronized
`MatrixFreeLiouvillian` interface. Its physical trace vector is nonzero only
on the root ADO:

```math
\mathrm{tr}_{\mathrm{HEOM}}(\boldsymbol\rho)
=\mathrm{tr}(\rho_{\boldsymbol0}).
```

This enables the existing matrix-free Krylov machinery without materializing
the full hierarchy generator. [`heom_steady_state`](@ref) is the direct
trace-fixed GMRES convenience:

```julia
stationary_hierarchy = heom_steady_state(
    plan; krylovdim=40, maxiter=800, atol=1e-10, rtol=1e-8)
rho_ss = heom_reduced_state(stationary_hierarchy)
```

Stationary solving requires an autonomous system generator. As for any finite
HEOM calculation, convergence of `rho_ss` with hierarchy depth remains the
user's responsibility.

## Reliability checklist

Check separately:

1. convergence of the bath exponential decomposition;
2. consistency of the left/right correlation decomposition, especially when
   `right_coefficients` is explicit;
3. convergence in `max_depth`;
4. convergence of the time step or Krylov residual;
5. preservation of `tr(heom_reduced_state(state))`;
6. Hermiticity and physicality of the root ADO.

The implementation never normalizes the hierarchy, clips eigenvalues,
symmetrizes invalid couplings, changes explicit bath coefficients, or inserts
a terminator. Default conjugate-pole completion is part of the documented
correlation representation, not a state repair. The constructor rejects nonfinite coefficients, nondecaying
frequencies, non-Hermitian fixed couplings, incompatible PI bases, and scalar
types that would narrow the prepared system Liouvillian.

## Current limitations

- The implemented hierarchy is the unscaled bosonic Gaussian-bath HEOM for
  fixed Hermitian system couplings. Fermionic sign hierarchies and
  non-Hermitian coupling pairs require a different construction.
- Each declared environment couples through one global PI operator `Q_b`.
  A common bath coupled to a collective observable is therefore supported,
  but this is not yet the specialized PI hierarchy for `N` independent local
  non-Markovian baths.
- Only a factorized initial hierarchy is constructed automatically. A user
  may supply a complete `HEOMState` for a correlated initial condition, but
  imaginary-time thermal preparation is not implemented.
- `terminator=:none` is the only boundary rule. Markovian residue corrections,
  low-temperature corrections, counterterms, and bath decompositions are not
  inferred; include the appropriate system correction explicitly when the
  chosen physical derivation requires one.
- The hierarchy is unscaled. Deep, strongly coupled calculations may become
  poorly conditioned even before their storage becomes prohibitive. A future
  dynamically scaled hierarchy could improve that regime, but silently
  rescaling only some ADOs would change the equations and is not done here.
- The number of ADOs grows as `binomial(K+D,D)`. PI compression removes the
  exponential system Hilbert space, not the combinatorial hierarchy growth
  caused by many bath poles and a large depth.
- There is currently no HEOM-specific adjoint application or block
  preconditioner. `heom_steady_state` uses the general forward matrix-free
  trace-fixed GMRES implementation.
- Fixed-step RK4 is the dedicated time-domain interface. There is no
  HEOM-specific SciML problem wrapper; time-step convergence must be assessed
  explicitly when using this path.

The complete executable dephasing comparison is in
[`examples/pi_heom.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_heom.jl),
with discussion in
[`examples/pi_heom.md`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_heom.md).

## API

```@docs
HEOMBath
HEOMPlan
HEOMWorkspace
HEOMEvolutionWorkspace
HEOMState
heom_number_ados
heom_multiindices
heom_initial_state
heom_ado
heom_reduced_state
heom_evolve!
heom_evolve
heom_time_evolution
heom_depth_convergence
heom_liouvillian
heom_steady_state
```
