# All-to-all Ising spins with local pseudomodes

Source:
[`debecker2026_all_to_all_ising_pseudomodes.jl`](debecker2026_all_to_all_ising_pseudomodes.jl)

## Scope of the example

The maintainer-supplied Debecker *et al.* draft, *Generating spatial
correlations in Ising chains via bath engineering* (2026), studies a
nearest-neighbour periodic Ising chain in which every spin is coupled to its
own damped pseudomode. A nearest-neighbour ring is translation invariant, but
it is not invariant under every permutation of the sites. The executable
example therefore does **not** claim to reproduce that chain or digitize one
of its figures.

Instead, it implements the specialization requested for this package: every
unordered spin pair has the same Ising interaction. The object permuted by the
symmetric group is the complete local supersite

```math
\mathcal H_{\mathrm{site}}
=\mathcal H_{\mathrm{spin}}\otimes\mathcal H_{\mathrm{mode}}.
```

All supersites have the same Hamiltonian, spin--mode coupling, and mode-loss
channel, while the Ising coefficient is identical for every pair. The
spin--pseudomode model is consequently exactly PI at any fixed pseudomode
cutoff. No spatial separation remains: every distinct pair has the same
correlator, and a spatial correlation length is not defined. Relations in the
nearest-neighbour manuscript that depend on momentum, distance, or the
one-dimensional Ising dispersion must not be transferred to this model.

At fixed cutoff, the enlarged spin-plus-pseudomode state obeys a Markovian
Lindblad equation. Tracing out the local modes yields the intended
finite-memory, generally non-Markovian reduced spin dynamics.

## Hamiltonian and damping convention

After truncating each pseudomode to occupations `0:nmax`, one supersite has
dimension

```math
d=2(n_{\max}+1).
```

The example evolves

```math
H=-J_{\mathrm{pair}}\sum_{i<j}X_iX_j
  +\omega_c\sum_i a_i^\dagger a_i
  +g\sum_i\left(L_i a_i^\dagger+L_i^\dagger a_i\right),
\qquad
g=\sqrt{\gamma\kappa},
```

with one local pseudomode-loss term at every site. Here `X` acts on the spin,
`a` acts on its truncated mode, and `L` is selected with
`coupling=:minus` or `coupling=:z`. In the latter case `L` is Hermitian and
the interaction reduces to a longitudinal coupling proportional to
`sigma_z*(a+a')`.

The package defines

```math
\mathcal D[A]\rho
=A\rho A^\dagger-
\frac{1}{2}\left\{A^\dagger A,\rho\right\}.
```

The manuscript instead puts the factor of two inside its dissipator. Its
coefficient `kappa` is therefore represented by
`LocalJump(a; rate=2kappa)`. The helper also uses the manuscript relation
`g=sqrt(gamma*kappa)`; neither conversion should be applied a second time in a
calling script.

`debecker2026_all_to_all_ising_pseudomode_model` interprets `Jpair` literally:
it is the coefficient of each unordered pair and the constructor inserts no
system-size scaling. The executable uses the extensive Curie--Weiss choice

```math
J_{\mathrm{pair}}=\frac{J}{N-1}.
```

Thus the `J` shown on its axes is a collective interaction scale, not the
literal `PBodyHamiltonian` rate. To study an unscaled all-pair model, pass the
desired coefficient directly as `Jpair`.

The pair term is lowered with `PBodyHamiltonian`. The script independently
checks the identity

```math
\sum_{i<j}X_iX_j
=\frac{1}{2}\left[\left(\sum_iX_i\right)^2-NI\right].
```

The scalar identity drops out of the commutator, so this supplies a separate
collective-square validation of the compiled generator.

## Model-to-solution workflow

The reusable helpers in [`paper_models.jl`](paper_models.jl) keep the tensor
ordering, cutoff matrices, rate conversion, and coupling convention together:

```julia
using LinearAlgebra
using PermutationalInvariantDynamics

include("examples/paper_models.jl")
using .PaperModels

N = 3
nmax = 1
J = 0.25
omega_c = 1.0
gamma = 0.05
kappa = 2.0

operators = debecker2026_pseudomode_operators(nmax)
basis = PIBasis(N, operators.dsite)
model = debecker2026_all_to_all_ising_pseudomode_model(
    basis, operators;
    Jpair=J / (N - 1), omega_c, gamma, kappa,
    coupling=:minus,
)

local_ket = zeros(ComplexF64, operators.dsite)
local_ket[1] = 1 # |g> tensor |0>
rho0 = iid_pure_state(basis, local_ket)

pair_geometry = PBodyGeometry(basis, 2)
pair_sum = pbody_collective_operator(
    basis, kron(operators.x_site, operators.x_site), 2;
    cache=pair_geometry,
)

times = collect(range(0.0, 10.0; length=101))
prepared = compile(model; backend=:matrixfree)
solution = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times,
    steps_per_interval=8,
    observables=(pair_sum=pair_sum,),
    save_states=false,
)
Cxx = real.(solution.observables[:pair_sum]) / binomial(N, 2)
```

This compact snippet illustrates the ordinary `compile`/`solve_dynamics`
workflow over a short interval. The executable uses a different integrator for
the manuscript-style curves extending to `omega_c*t=100`. At
`kappa/omega_c=20`, simply halving an explicit RK4 step did not provide a
reliable refinement of the broad decay. Each plotted long curve therefore
uses a matrix-free Liouvillian, one reusable dimension-30
`KrylovExpvWorkspace`, and preallocated `krylov_expv!` exponential actions
between successive output times. It retains the current PI state and the two
observable series rather than a state history. With 201 output points, the
default run performs 6000 Liouvillian applications per curve.

The broad `kappa/omega_c=20` curve is repeated independently with Krylov
dimension 40 and tighter absolute and relative tolerances. The script checks
the pointwise `Cxx` difference and the accumulated exponential-action error;
the current default comparison is at approximately `8e-16`, well inside its
regression tolerance. This Krylov refinement, rather than an unstable
fixed-step comparison, is the convergence evidence for the long dynamics
panel.

The initial state is `|g,0>^N`, for which the distinct-pair correlation
vanishes. `pbody_collective_operator` returns the sum over all unordered
pairs, so division by `binomial(N,2)` gives the common physical correlator

```math
C_{xx}=\langle X_iX_j\rangle,\qquad i\ne j.
```

The complete PI basis is required: local mode loss and local spin--mode
exchange generally transfer weight between Schur sectors. Restricting the
basis to the fully symmetric sector would silently change the dynamics and is
not a valid optimization for this model.

For the small default stationary grid, the script compiles a sparse generator
and uses `DirectAlgorithm`. One reference point is solved independently with
a matrix-free `GMRESAlgorithm`, and the common-pair correlators are compared.
For a larger feasible PI dimension, prepare the matrix-free model once, reuse
its Krylov workspace and Schur-sector preconditioner, and continuation-seed
neighbouring parameter points. Solver convergence does not replace the cutoff
study described below.

## Spin-only negativity

The physical question is the entanglement of two spins after their two local
pseudomodes have been traced out. Applying the package's supersite
`negativity` directly would answer a different question because each side
would still contain a truncated mode.

The example instead evaluates all 16 distinct-particle Pauli moments with
operators `sigma_mu tensor I_mode` and reconstructs

```math
\rho_{\mathrm{spin}}^{(2)}
=\frac{1}{4}\sum_{\mu,\nu=0}^{3}
\left\langle\sigma_\mu^{(1)}\sigma_\nu^{(2)}\right\rangle
\sigma_\mu\otimes\sigma_\nu.
```

It checks Hermiticity, unit trace, and positivity before partially
transposing the second physical qubit. The plotted value is

```math
\mathcal N
=\frac{\left\|\left(\rho_{\mathrm{spin}}^{(2)}\right)^{T_2}\right\|_1-1}{2}.
```

This distinction is essential when comparing spin entanglement with an
explicit-environment embedding.

## Longitudinal coupling and the spin GHZ witness

The manuscript's other coupling choice, `coupling=:z`, retains the strong
global spin-parity symmetry

```math
\Pi_z=\prod_{i=1}^{N}\sigma_z^{(i)}.
```

An unrestricted trace-fixed stationary solver would therefore be asked to
choose among symmetry-resolved stationary components. The example avoids that
ambiguity. It starts from `|e,0>^N`, which has `Pi_z=+1` in the package spin
convention, and obtains the parity-even long-time state by adaptive
matrix-free `krylov_expv` propagation to `omega_c*t=1600`.

The plotted quantity is the largest fidelity obtained by optimizing only the
relative phase of the spin x-GHZ family

```math
|\mathrm{GHZ}_x(\phi)\rangle
=\frac{|{+x}\rangle^{\otimes N}
       +e^{i\phi}|{-x}\rangle^{\otimes N}}{\sqrt{2}}.
```

After tracing every pseudomode, define

```math
P_\pm=\langle {\pm x}|^{\otimes N}
\rho_{\mathrm{spin}}
|{\pm x}\rangle^{\otimes N},
\qquad
C=\langle {+x}|^{\otimes N}
\rho_{\mathrm{spin}}
|{-x}\rangle^{\otimes N}.
```

The phase-optimized fidelity is then

```math
F_{\mathrm{GHZ}_x}^{\max}
=\frac{P_++P_-}{2}+|C|.
```

`spin_x_ghz_fidelity` evaluates these three quantities as exact ordered
`N`-particle local moments. Every local projector or coherence is tensored
with the mode identity, so the pseudomodes are traced without constructing a
full spin density matrix. This is an `N`-body observable and becomes more
expensive than the pair diagnostics as `N` grows.

For every point in the displayed grid, the script checks state diagnostics,
the adaptive exponential-action error estimate, and the stationary residual
`norm(L*rho)`. At one grid corner it repeats the propagation for half the
settling time and requires the GHZ witness to agree. These checks establish
time and Krylov convergence for the finite calculation while respecting the
strong symmetry; they do not establish cutoff or thermodynamic convergence.
The contour `F_GHZx=0.5`, when crossed by the plotted range, is the standard
GHZ-fidelity witness threshold for genuine multipartite spin entanglement.

## Cutoff and scaling

A complete `N`-supersite PI operator has

```math
n_{\mathrm{PI}}
=\binom{N+d^2-1}{N},
\qquad d=2(n_{\max}+1),
```

coordinates. The default `N=3` calculation therefore has 816 coordinates at
`nmax=1`, but 8436 at `nmax=2`. The method avoids an explicit `d^N` Hilbert
state while still growing rapidly with the local mode cutoff. Estimate this
dimension before launching a scan; increasing `N` and `nmax` simultaneously
can become impractical even in PI coordinates.

Because the manuscript does not prescribe a cutoff for this all-to-all
specialization, the example repeats one dynamical curve at `nmax=1` and 2. It
checks both the change in `Cxx(t)` and the population of the highest retained
mode level in the wider calculation. A small boundary population is useful
evidence but is not by itself a proof of convergence. Production results
should be repeated at successively larger cutoffs for every observable and
time interval being claimed. The built-in `nmax=1,2` comparison applies to the
`coupling=:minus` correlation curve. It does not certify the cutoff of the
separate `coupling=:z` GHZ map; repeat that map or selected points at a wider
cutoff before using it as a research result. This short cutoff comparison
still uses `solve_dynamics` with fixed-step RK4 and is separately bounded by
its short time interval and step choice; it should not be confused with the
Krylov convergence check for the long manuscript-style curves.

## Figures and interpretation

The main Makie figure follows the manuscript's presentation style without
claiming a nearest-neighbour reproduction:

- the first panel shows `Cxx(t)` for `kappa/omega_c = 1, 5, 20`, using the
  single distinct-pair PI correlator in place of a distance-resolved one and
  reusable matrix-free Krylov exponential actions for the long sweep;
- the second panel maps the stationary `Cxx` over `J/omega_c` and
  `kappa/omega_c`;
- the third panel maps the spin-only two-spin negativity over the same grid.

The annotation states the Kac scaling and explicitly notes that no spatial
correlation length exists. The finite `N=3` heat maps are solver and model
demonstrations, not evidence for a thermodynamic phase boundary. PDF and PNG
copies are saved as
`debecker2026_all_to_all_ising_pseudomodes.*`.

A second figure overlays the `nmax=1` and 2 correlation curves and shows the
wider-cutoff highest-level population. It is saved as
`debecker2026_all_to_all_ising_pseudomodes_cutoff.*`.

A third, manuscript-Fig.-4-style figure shows the parity-selected
`coupling=:z` spin x-GHZ fidelity. Its black `0.5` contour marks the usual GHZ
witness threshold, and its annotation reports the largest long-time
Liouvillian residual on the grid. It is saved as
`debecker2026_all_to_all_ising_pseudomodes_ghz.*`. As with the first heat maps,
this is the uniform-all-pair analogue and not a spatial-chain reproduction.

The executable defaults to a compact stationary grid. Set
`PI_PSEUDOMODE_FULL_SCAN=1` for the denser built-in grid:

```sh
PI_PSEUDOMODE_FULL_SCAN=1 \
  julia --project=examples examples/debecker2026_all_to_all_ising_pseudomodes.jl
```

Run the checked compact version, including its Makie output, with:

```sh
julia --project=examples examples/debecker2026_all_to_all_ising_pseudomodes.jl
```

The numerical assertions also run under the root project; without the
examples-only CairoMakie dependency, only figure generation is skipped.
