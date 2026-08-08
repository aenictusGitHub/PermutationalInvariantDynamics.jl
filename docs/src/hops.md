# Permutationally invariant HOPS

The hierarchy of pure states (HOPS) backend provides a stochastic,
wave-function counterpart to PI--HEOM. It is useful when the system couples
to one or more *shared* structured bosonic environments and propagating pure
Schur-sector amplitudes is cheaper than propagating a hierarchy of PI density
operators.

HOPS introduces two independent numerical limits:

- the finite hierarchy depth;
- the number of statistically independent noise realizations.

Neither limit replaces the time-step convergence check. A result should be
reported as converged only after all three have been varied independently.
The HOPS root is an unnormalized auxiliary direct-sum amplitude, not the
physical [`SymmetricKet`](@ref) used for closed fully symmetric dynamics; see
[Symmetric pure kets and block-resolved
entropy](symmetric_kets_and_block_entropy.md) for a side-by-side comparison.

## Physical convention

For bath $b$, [`HOPSBath`](@ref) represents

```math
C_b(t)=\sum_{k\in b}c_k e^{-\nu_k t},\qquad t\geq0,
```

where $\mathrm{Re}\,\nu_k\gt 0$. Its PI coupling operator is $L_b$.
Different baths have independent proper complex Gaussian processes,

```math
\mathbb E[z_b(t)]=\mathbb E[z_b(t)z_{b'}(s)]=0,\qquad
\mathbb E[z_b(t)z_{b'}^*(s)]
 =\delta_{bb'}C_b(t-s).
```

After flattening all exponential terms into one pole index $k$, the linear
hierarchy is

```math
\begin{aligned}
\dot\psi_{\boldsymbol n}
={}&\left[-iH-\sum_k n_k\nu_k+
          \sum_b L_bz_b^*(t)\right]\psi_{\boldsymbol n}\\
 &+\sum_k n_kc_kL_{b(k)}
       \psi_{\boldsymbol n-\boldsymbol e_k}
  -\sum_k L_{b(k)}^\dagger
       \psi_{\boldsymbol n+\boldsymbol e_k}.
\end{aligned}
```

Initially only the root is occupied:
$\psi_{\boldsymbol 0}(0)=\psi_0$ and
$\psi_{\boldsymbol n}(0)=0$ for $\boldsymbol n\ne0$. The root of a
linear HOPS trajectory is deliberately **not normalized**. The unbiased
density estimator is

```math
\rho(t)=\mathbb E\!\left[
 |\psi_{\boldsymbol0}(t)\rangle
 \langle\psi_{\boldsymbol0}(t)|\right],
```

not the average of individually normalized projectors. Calling
[`hops_density`](@ref) applies this rule.

This convention is the one introduced by Süß, Eisfeld, and Strunz,
[*Phys. Rev. Lett.* **113**, 150403 (2014)](https://doi.org/10.1103/PhysRevLett.113.150403).
The underlying
non-Markovian quantum-state-diffusion construction is described by Diósi,
Gisin, and Strunz,
[*Phys. Rev. A* **58**, 1699 (1998)](https://doi.org/10.1103/PhysRevA.58.1699).

## Prepared PI workflow

For one collective bath, the basic workflow is

```julia
basis = PIBasis(N, 2)
spin = spin_matrices()
H = collective_operator(basis, spin.jx)
Q = collective_operator(basis, spin.jz)

bath = HOPSBath(Q, coefficient, decay)
plan = HOPSPlan(H, bath; max_depth=4, scaling=:scaled)
workspace = HOPSWorkspace(plan)

rho0 = iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2))
psi0 = weak_pi_pseudoket(rho0)
times = range(0.0, 2.0; length=101)

path = hops_trajectory(
    plan, psi0, times;
    dt=0.005, rng=Random.Xoshiro(11), workspace)
rho_path = hops_density(path, lastindex(times))

mean_states = hops_average(
    plan, psi0, times, 1000;
    dt=0.005, seed=11, threaded=true)
```

[`HOPSPlan`](@ref) owns the immutable hierarchy topology, physical Schur
blocks, decay factors, and scaling data. [`HOPSWorkspace`](@ref) owns
trajectory-local hierarchy, integration, and noise scratch. A workspace must
not be shared by simultaneous tasks.

No state or operator of dimension $d^N$ is constructed. Every PI
Hamiltonian and coupling acts blockwise on the direct sum of Schur irreps used
by [`WeakPIPseudoKet`](@ref). The multiplicity spaces are spectators.

A general mixed PI state can be passed directly to [`hops_average`](@ref).
The library prepares a [`HOPSInitialEnsemble`](@ref) by diagonalizing each
multiplicity-weighted Schur block and samples its positive eigencomponents:

```julia
initial_ensemble = hops_initial_ensemble(plan, rho0_mixed)
result = hops_average(
    plan, initial_ensemble, times, 1000;
    dt=0.005, return_info=true)
```

The spectral preparation retains only PI-sized block data, never clips a
negative eigenvalue, and does not sample multiplicity tableaux. The optional
result reports Hilbert--Schmidt Monte Carlo spread and standard error at every
saved time. Those statistics do not include time-step or hierarchy error.

A physical [`HEOMBath`](@ref) with zero white-noise residue can be passed
directly to `HOPSPlan` or converted with `HOPSBath(heom_bath)`. Conversion
checks by default that its right correlation is the conjugate of its left
correlation. A nonzero HEOM residue requires a separate Markovian unraveling
and is therefore rejected by this linear colored-noise backend.

## Instantaneous PI pulses

An ideal control pulse must act on every hierarchy auxiliary; restarting
`hops_trajectory` after a pulse would reset both the auxiliaries and their
colored-noise memory and therefore solve a different problem. Prepare the
Schur action once:

```julia
Ux = exp(-im * pi * spin.jx)
pulse = PIUnitaryPulse(basis, Ux)
sequence = HierarchyPulseSequence(pulse_times, pulse)

result = hops_average(
    plan, psi0, times, trajectories;
    dt=0.005, pulses=sequence, return_info=true)
```

A local `d`-by-`d` matrix is lifted as $U^{\otimes N}$ without constructing
that full-Hilbert matrix. A compatible `PIOperator` may be supplied instead.
The pulse maps every auxiliary pseudo-ket as
$\psi_{\boldsymbol n}\mapsto U\psi_{\boldsymbol n}$, while the OU state and
the other hierarchy data remain continuous.

Pulse times are finite, ordered events. The driver splits a nominal step
exactly at every event. It applies events in `(start, stop]`, so a pulse at a
saved time is applied before saving and is not applied again in the adjacent
interval. Equal-time events are applied in input order. These are ideal
instantaneous unitaries; a finite pulse shape requires a model and evolution
route that explicitly represent its time-dependent Hamiltonian rather than
interpreting its duration as zero.

The published Platonic Eulerian sequences can be prepared without transcribing
their Cayley-graph words:

```julia
tedd = tetrahedral_pulse_sequence(basis, tau0)
oedd = octahedral_pulse_sequence(basis, tau0)
iedd = icosahedral_pulse_sequence(basis, tau0)
```

They contain 24, 48, and 120 equally spaced ideal pulses per cycle. The
constructors reuse their two immutable global rotation generators and return
ordinary `HierarchyPulseSequence` objects, so the same schedules work with
both PI--HOPS and PI--HEOM. See
[Read, Serrano-Ensástiga, and Martin, *Quantum* **9**, 1661
(2025)](https://doi.org/10.22331/q-2025-03-12-1661) for the decoupling and
anisotropy conditions. The instantaneous hierarchy backend does not by itself
simulate the finite-width control Hamiltonians considered in the robustness
analysis.

## What “PI HOPS” means

The exact reduced backend applies to a shared bath whose system coupling
$L_b$ is itself PI. Such an operator is block diagonal in the Schur
decomposition and acts as the identity on each multiplicity space. The HOPS
path therefore remains representable by Schur-irrep amplitudes.

This condition excludes independent local colored noises,

```math
\sum_{i=1}^N L_i z_i(t),
```

even when all $z_i$ have identical statistics. A single realization has
different values of $z_i$, breaks permutation symmetry, and generally acts
nontrivially on the Schur multiplicity spaces. The ensemble-averaged density
may be PI, but an individual path is not a weak-PI pseudo-ket. Use PI--HEOM or
the finite-cutoff local-pseudomode supersite backend for that problem.
Replacing the independent $z_i$ by one common process would describe a
different, collectively correlated environment.

Likewise, a `WeakPIPseudoKet` is an auxiliary state in the direct sum of
Schur irreps. It is not a labeled-particle vector in the full Hilbert space,
and relative phases between different Schur sectors are not physical PI
data.

## Finite-temperature correlations

For a self-adjoint coupling $Q=Q^\dagger$, the linear equation above can
use the complete physical thermal correlation as one sum process. This is the
finite-temperature construction described in the original HOPS work. The
two-sided kernel must nevertheless be a valid Gaussian covariance:
$C(-t)=C(t)^*$, $C(0)$ must be real and nonnegative, and every finite
covariance matrix sampled from it must be positive semidefinite.

An arbitrary [`HEOMBath`](@ref) is therefore not automatically interchangeable
with a `HOPSBath`. HEOM can evolve a user-supplied left/right exponential
decomposition algebraically, whereas HOPS must also realize its left
correlation as a stochastic covariance. In particular, an inconsistent
explicit right correlation, a truncated fit that is not positive
semidefinite, or an unapplied white-noise residue must not be hidden by the
noise generator.

For strong coupling, Hartmann and Strunz recommend an alternative exact
thermal construction for Hermitian coupling: use the zero-temperature
correlation in the quantum hierarchy and add the thermal part as an
independent Hermitian stochastic potential. That formulation can converge at
substantially smaller hierarchy depth. It is not silently inferred from a
finite-temperature pole list and is not a separate evolution mode in this
initial linear backend. Use the full physical covariance or PI--HEOM instead.
A non-Hermitian finite-temperature coupling requires the thermofield-doubled
construction rather than the one-process Hermitian shortcut.

## Noise generation

The built-in fast generator is an exact discrete-time stationary
Ornstein--Uhlenbeck construction for exponential terms whose coefficients
are real and nonnegative. For one term,

```math
x(t+\Delta t)=e^{-\nu\Delta t}x(t)
 +\sqrt{c\left(1-e^{-2\mathrm{Re}(\nu)\Delta t}\right)}\,\eta,
```

where $\eta$ is a proper unit-variance complex Gaussian. The initial value
is sampled from the stationary distribution,
$x(0)=\sqrt c\,\eta_0$; setting it to zero would generate the wrong
nonstationary covariance. Components belonging to the same bath are summed
before applying $L_b$.

A finite exponential representation may contain complex or negative
individual coefficients even though its *total* correlation is physical.
Taking a complex square root of each coefficient does not generate the
required Gaussian process. For such a decomposition, pass an explicitly
prepared `noise` realization to [`hops_trajectory`](@ref). A callable has the
form `noise(time, bath_number)` and must return the same value whenever the
integrator revisits a time. In stochastic use it should interpolate one
realization prepared before propagation, and its ensemble must obey the total
two-time covariance. Spectral synthesis is a standard construction; Hartmann
and Strunz give an FFT implementation in
[*J. Chem. Theory Comput.* **13**, 5834 (2017)](https://doi.org/10.1021/acs.jctc.7b00751).

Noise must be fixed before a path is integrated. Drawing a new value whenever
an adaptive right-hand side is evaluated changes the stochastic process and
does not implement HOPS. The built-in path integrator therefore uses one
explicit grid and a fixed `dt`. Repeat with a smaller `dt`.

## Scaling and hierarchy truncation

With `scaling=:scaled`, the stored auxiliary is

```math
\widehat\psi_{\boldsymbol n}
=\frac{\psi_{\boldsymbol n}}{
 \prod_k\sqrt{n_k!\,a_k^{n_k}}},
```

with positive prepared pole scales $a_k$. The corresponding downward and
upward coefficients are

```math
\sqrt{n_k}\frac{c_k}{\sqrt{a_k}},
\qquad
\sqrt{(n_k+1)a_k},
```

respectively. This is an exact diagonal similarity transformation of the
retained hierarchy. It changes neither the root nor the modeled bath.
`scaling=:unscaled` is useful when comparing equations term by term.
[`hops_coordinate_scale`](@ref) exposes $s_{\boldsymbol n}$ for a retained
node. Exact duplicate poles within one bath are combined before these scales
and the hierarchy are built; exact cancellations are removed.

`max_depth=D` retains multi-indices satisfying
$\sum_k n_k\le D$. For $P$ exponential poles, the number of retained
auxiliary pure states is

```math
\binom{P+D}{D}.
```

Missing upward neighbors are zero in the finite hierarchy. This hard
truncation is an approximation, and convergence must be checked by increasing
`max_depth`. A deeper hierarchy is commonly needed for stronger coupling,
slower bath decay, or longer propagation. Alternative HOPS truncations and
their convergence behavior are analyzed by Zhang, Bentley, and Eisfeld,
[*J. Chem. Phys.* **148**, 134103 (2018)](https://doi.org/10.1063/1.5022225).

A positive `importance_cutoff` additionally retains a deterministic
downward-closed order ideal. Its score is a heuristic, not an error bound.
The setup is budget-bounded by retained and frontier nodes, so it can prepare
a small retained hierarchy even when the exact unpruned binomial count does
not fit `Int`. [`hops_auxiliary_importances`](@ref) returns the retained
scores. Report and converge this cutoff separately.

For one Schur irrep of dimension $m_\nu$, a trajectory stores quantities
scaling as $m_\nu\binom{P+D}{D}$, rather than the
$m_\nu^2\binom{P+D}{D}$ density coordinates of an analogous HEOM
hierarchy. HOPS exchanges that per-path reduction for Monte Carlo sampling.
Which method is faster depends on the Schur support, hierarchy depth,
sampling accuracy, and requested observables.

Within one path, every prepared Schur block acts on all hierarchy nodes in a
matrix--matrix kernel. Edges are grouped by bath, and one hierarchy-sized
action buffer is reused for $L$ and $L^\dagger$. A self-adjoint coupling
uses one combined edge pass; a non-Hermitian coupling overwrites the same
buffer and uses separate downward/upward edge passes. Across paths,
`hops_average(...; threaded=true)` uses deterministic balanced partitions
with one workspace and random stream per active task.
Float32, Float64, and BigFloat inputs retain their selected scalar type.
BigFloat plans capture the required precision and rounding context and restore
it for workspace construction, conditioned application, integration, and
ensemble reduction.

## Reliability checklist

For a research calculation:

1. verify that every bath coupling is PI and that each noise represents a
   shared environment;
2. check that the exponential fit is accurate throughout the simulated
   memory interval;
3. decrease `dt`;
4. increase `max_depth`;
5. increase the number of independent trajectories and report a statistical
   uncertainty;
6. compare against PI--HEOM at a smaller size or shallower hierarchy when
   possible.

Agreement between two trajectory counts is not a hierarchy convergence test.
Agreement between two hierarchy depths is not a sampling-error estimate.
Initial system--bath correlations are outside the factorized Gaussian-bath
derivation and are not inferred by the backend.

## Runnable examples

The HOPS examples separate four complementary workflows:

- [`examples/pi_hops.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_hops.jl)
  introduces a pure-state, one-bath collective-dephasing calculation and
  compares the HOPS mean with deterministic PI--HEOM and the analytical
  coherence.
- [`examples/pi_hops_collective_emission.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_hops_collective_emission.jl)
  uses the non-Hermitian coupling $L=J_-$ in a one-excitation manifold. It
  validates the exact depth-one closure, prescribed conditioned noise,
  `hops_density`, hierarchy inspection, and deterministic `hops_rhs!`.
- [`examples/pi_hops_mixed_multibath.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_hops_mixed_multibath.jl)
  starts from a genuinely mixed, multi-sector PI state and two independent
  shared baths. It demonstrates `hops_initial_ensemble`,
  `HOPSBatchWorkspace`, `return_info=true`, Monte Carlo error contraction,
  and setup-only importance pruning.
- [`examples/nonmarkovian_dynamical_decoupling.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/nonmarkovian_dynamical_decoupling.jl)
  applies ideal CPMG and UDD4 pulses to every hierarchy auxiliary without
  restarting the colored noise. It compares the HOPS ensemble with PI--HEOM
  and the exact filter function for the same full-line one-pole Lorentzian
  correlation, alongside a separately integrated positive-frequency
  analytical curve.

All four scripts retain numerical assertions when CairoMakie is unavailable.
With the examples environment they also render the already validated arrays;
plotting never triggers another HOPS solve.

## API

```@docs
PIUnitaryPulse
HierarchyPulseSequence
platonic_pulse_sequence
tetrahedral_pulse_sequence
octahedral_pulse_sequence
icosahedral_pulse_sequence
apply_hierarchy_pulse!
HOPSBath
HOPSPlan
HOPSWorkspace
HOPSBatchWorkspace
HOPSInitialEnsemble
HOPSRootKet
HOPSTrajectory
HOPSEnsembleResult
hops_number_auxiliaries
hops_multiindices
hops_hierarchy_metadata
hops_auxiliary_importances
hops_coordinate_scale
hops_initial_ensemble
hops_trajectory
hops_density
hops_average
hops_rhs!
```
