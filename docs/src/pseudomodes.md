# Local pseudomodes and PI supersites

A finite-cutoff pseudomode turns a selected non-Markovian environment into a
larger time-local Lindblad problem. When every physical system has the same
local pseudomodes, the object permuted by the symmetric group is the complete
local tuple

```math
\text{system}_i\otimes\text{mode}_{i,1}\otimes\cdots
\otimes\text{mode}_{i,M}.
```

The package treats this tuple as one identical **supersite**. If the physical
system has dimension ``d_S`` and mode ``\mu`` retains occupations
``0:n_{\max,\mu}``, one supersite has dimension

```math
D=d_S\prod_{\mu=1}^{M}(n_{\max,\mu}+1).
```

The exact complete PI operator space then has

```math
\binom{N+D^2-1}{N}
```

coordinates. This is polynomial in ``N`` at fixed ``D`` and never constructs
a ``D^N`` state, but it can still grow rapidly with the number and cutoff of
the local modes. Always preflight and converge the cutoffs.

This construction describes identical **independent local** pseudomodes. It is
not the same as a [shared cavity](global_pseudomodes.md) or a common
[`HEOMBath`](@ref): either common environment can contain cross-correlations
between different particles. See [Global pseudomodes and shared
cavities](global_pseudomodes.md) and [PI--HEOM non-Markovian
dynamics](heom.md) for those distinct workflows.

The complete generator must still be invariant under every permutation of
the supersites. Identical onsite terms and local damping satisfy this
automatically; a many-system interaction must be a uniform symmetric
`p`-body sum, such as an all-to-all Ising term. Translation invariance or
periodic boundary conditions alone do not make a nearest-neighbour chain PI.

## Internal ordering

[`PISupersite`](@ref) follows Julia's Kronecker-product ordering. Factors are
declared as

```text
(system, mode1, mode2, ...)
```

and a local tensor operator is

```julia
kron(system_operator, mode1_operator, mode2_operator, ...)
```

so the last internal factor is fastest. This local convention is different
from [`CompositePIBasis`](@ref), which combines complete global operator
spaces. `CompositePIBasis` cannot preserve the pairing between system `i` and
its own mode `i`; local pseudomodes therefore belong inside one
`PISupersite`.

The low-level supersite layer is also useful for finite ancillas that are not
bosonic:

```julia
using LinearAlgebra
using PermutationalInvariantDynamics

site = PISupersite(
    3, (3, 2);
    labels=(:system, :ancilla),
)

A = Matrix{ComplexF64}(I, 3, 3)
B = ComplexF64[0 1; 1 0]

A_on_site = lift_system_operator(site, A)
B_on_site = lift_supersite_operator(site, B; factor=:ancilla)
AB = supersite_tensor_operator(site, A, B)

@assert AB == A_on_site * B_on_site
```

These helpers allocate only a `site.basis.d`-dimensional local matrix. They do
not lift it to the labeled-particle Hilbert space.

When a basis has already been prepared, use
`PISupersite(basis, factor_dimensions; ...)` or
`pseudomode_supersite(basis, system_dimension, modes...)`. The wrapper keeps
that exact basis object, including an explicitly retained sector set, and
validates that its local dimension matches the factorization.

## One pseudomode per system

The following model contains `N` qubits, one damped local mode per qubit, and
a uniform all-to-all Ising interaction. The local basis is
`qubit ⊗ mode`.

```julia
using LinearAlgebra
using PermutationalInvariantDynamics

N = 3
spin = spin_matrices(2)
sigma_x = 2spin.jx
sigma_z = 2spin.jz
sigma_minus = spin.jm

Omega = 0.35
J = 0.8
omega_c = 1.0
kappa = 0.4
g = 0.18

mode = BosonicPseudomode(
    2;
    label=:cavity,
    frequency=omega_c,
    damping=kappa,
    thermal_occupation=0,
)

coupling = PseudomodeCoupling(
    sigma_minus;
    mode=:cavity,
    strength=g,
)

# This rate is the literal coefficient of every unordered pair. The package
# never inserts a Kac factor, so it is written explicitly here.
pair_term = PBodyHamiltonian(
    kron(sigma_x, sigma_x), 2;
    rate=-J / (N - 1),
)

# Separate geometry from physical rates when the same cutoff is reused.
site = pseudomode_supersite(N, 2, mode)

embedding = pseudomode_model(
    site, (Omega / 2) * sigma_z;
    couplings=(coupling,),
    system_terms=(pair_term,),
)

basis = embedding.basis
model = embedding.model
```

The builder retains separate fixed operators and scalar rates for the system
Hamiltonian, mode frequency, couplings, damping channels, and lifted system
terms. Compilation can therefore share their Schur geometry and
[`compile_family`](@ref) can specialize compatible scalar-rate scans.
`embedding.site_hamiltonian` (equivalently
`embedding.base_site_hamiltonian`) contains the one-supersite system,
frequency, and coupling terms. Collective and `p`-body entries cannot be
represented by that local matrix; inspect `embedding.lifted_system_terms` and
`embedding.supersite_terms` for those contributions. The aggregate
construction estimate is available as `embedding.resource_estimates`.
Calling `pseudomode_model(site, ...)` again preserves exact basis identity;
the `frequencies`, `dampings`, and `thermal_occupations` keywords can override
the immutable mode defaults for a fixed-cutoff scan.

For the fastest scan, construct one prototype and pass the indices of the
rates that actually vary to `compile_family`; `specialize` then changes only
those scalars. If a varied component is zero at the prototype point, construct
the prototype with `retain_zero_terms=true` so its fixed operator geometry is
kept explicitly. The default omits fixed zero components to keep an ordinary
single-point model small.

For example, this changes physical rates without reconstructing the basis:

```julia
scan_point = pseudomode_model(
    site, (Omega / 2) * sigma_z;
    couplings=(
        PseudomodeCoupling(
            sigma_minus;
            mode=:cavity,
            strength=0.21,
        ),
    ),
    system_terms=(pair_term,),
    frequencies=1.05,
    dampings=0.45,
    thermal_occupations=0.02,
)

@assert scan_point.basis === basis
```

The interaction specified by `strength=g` is

```math
g\,L a^\dagger + g^* L^\dagger a.
```

`L` need not be Hermitian and `g` may be complex. The optional
`counterrotating_strength=h` adds

```math
h\,L a + h^* L^\dagger a^\dagger.
```

Internally, a complex strength is split into real rates multiplying Hermitian
quadratures; Hamiltonian rates are never made complex. For Hermitian `L`, the
rotating expression with a real strength is already proportional to
``L(a+a^\dagger)``. Adding an equal real counter-rotating strength would add
the same operator again.

The damping convention is the package convention

```math
\mathcal D[a](\rho)
=a\rho a^\dagger-\frac12\{a^\dagger a,\rho\}.
```

Thus `damping=kappa` produces
``\kappa(\bar n+1)\mathcal D[a]`` and
``\kappa\bar n\mathcal D[a^\dagger]``. The mode amplitude decays at
``\kappa/2``. Convert explicitly when a paper defines its dissipator with an
extra factor of two.

## Product initial states

[`pseudomode_product_state`](@ref) tensors only the local factor states and
then calls the PI tensor-power constructor. Modes start in their vacuum kets
unless `mode_states` is supplied:

```julia
excited = ComplexF64[0, 1]
rho0 = pseudomode_product_state(site, excited)
validate_state(rho0)
```

A mixed system state or a finite-temperature initial mode is also accepted.
For a single truncated mode, a normalized truncated geometric state can be
constructed explicitly:

```julia
nbar_initial = 0.2
q = nbar_initial / (nbar_initial + 1)
weights = q .^ (0:mode.nmax)
weights ./= sum(weights)
rho_mode = Diagonal(ComplexF64.(weights))

rho_system = ComplexF64[0.7 0; 0 0.3]
rho0_thermal = pseudomode_product_state(
    site, rho_system;
    mode_states=rho_mode,
)
```

No normalization is performed silently: each supplied local state is
validated by the ordinary PI product-state constructors.

If the state of one supersite already contains system--mode correlations,
construct that local vector or density matrix explicitly and use
[`supersite_iid_state`](@ref):

```julia
ground = ComplexF64[1, 0]
one_photon = zeros(ComplexF64, mode.levels)
one_photon[2] = 1

local_correlated = (
    kron(ground, mode.vacuum) +
    kron(excited, one_photon)
) / sqrt(2)

rho0_locally_correlated = supersite_iid_state(
    site, local_correlated,
)
```

This makes `N` identical copies of the correlated local supersite state; it
does not claim that the different supersites are initially entangled.

## Matrix-free dynamics and stationary states

The supersite model is an ordinary [`PIModel`](@ref), so the standard prepared
workflow applies unchanged:

```julia
prepared = compile(model; backend=:matrixfree)

times = range(0.0, 12.0; length=121)
solution = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times,
    steps_per_interval=8,
)
```

For an autonomous stationary problem, explicitly selecting GMRES keeps the
solve matrix-free:

```julia
steady = stationary_state(
    prepared;
    algorithm=GMRESAlgorithm(krylovdim=40, maxiter=1000),
    atol=1e-11,
    rtol=1e-9,
    return_info=true,
)

rho_ss = steady.state
```

Inspect `steady.info.residual` and the state diagnostics rather than treating
solver return alone as a uniqueness or cutoff certificate. Use
[`recommend_solver`](@ref) before a large solve or materialization and reuse
explicit Krylov/Liouvillian workspaces in ordered scans.

Independent mode damping is represented by [`LocalJump`](@ref), not
[`CollectiveJump`](@ref). It generally populates multiple Schur sectors even
when the initial supersite state lies in the fully symmetric sector.
Consequently a complete `PIBasis(N,D)` is normally required.
`pseudomode_model(...; sectors=...)` accepts a restriction only when
[`PIModel`](@ref) can certify that every requested local process stays inside
it; it never silently projects the dynamics.

## System observables and all-to-all interactions

A system-only one-body observable must first be lifted to the local supersite:

```julia
sigma_x_site = lift_system_operator(site, sigma_x)
mean_x = real(collective_expectation(rho_ss, sigma_x_site)) / N
```

The model builder applies the same lift automatically to each fixed built-in
entry of `system_terms`. For a general system-only `p`-body term, it uses
[`lift_system_pbody_operator`](@ref) to interleave the auxiliary identities
particle by particle:

```text
(system1 ⊗ modes1) ⊗ ... ⊗ (systemp ⊗ modesp).
```

This is not the grouped ordering obtained from
`kron(system_pbody_operator, I_modes)`. The lifted matrix is sparse, uses the
exact support of the input, and is guarded by `memory_budget`.
The lower-level [`PISupersite`](@ref) constructor permits arbitrary factor
labels, but `lift_system_pbody_operator` requires the `:system` factor to be
first. [`pseudomode_supersite`](@ref) always establishes that ordering.

The lower-level equivalent of the `system_terms` call above is:

```julia
lifted_pair_term = lift_system_term(site, pair_term)
@assert lifted_pair_term isa PBodyHamiltonian
```

`lift_system_term` supports fixed microscopic local, collective, and symmetric
`p`-body built-in terms. A direct Schur-space term is already tied to another
basis and cannot be lifted this way.

## Multiple local pseudomodes

Give every mode a unique label and select it from each coupling:

```julia
mode_fast = BosonicPseudomode(
    1;
    label=:fast,
    frequency=1.4,
    damping=0.7,
)

mode_slow = BosonicPseudomode(
    2;
    label=:slow,
    frequency=0.45,
    damping=0.12,
    thermal_occupation=0.05,
)

couplings = (
    PseudomodeCoupling(
        sigma_minus;
        mode=:fast,
        strength=0.16,
    ),
    PseudomodeCoupling(
        sigma_minus;
        mode=:slow,
        strength=0.10,
        counterrotating_strength=0.025im,
    ),
)

N_multi = 2
pair_term_multi = PBodyHamiltonian(
    kron(sigma_x, sigma_x), 2;
    rate=-J / (N_multi - 1),
)

two_mode = pseudomode_model(
    N_multi, (Omega / 2) * sigma_z, (mode_fast, mode_slow);
    couplings=couplings,
    system_terms=(pair_term_multi,),
)

@assert two_mode.supersite.factor_dimensions == (2, 2, 3)
@assert two_mode.basis.d == 12

rho0_two_mode = pseudomode_product_state(
    two_mode.supersite, excited,
) # both modes use their vacuum kets
```

`two_mode.mode_operators[1]` and `[2]` contain the lifted annihilation,
creation, number, parity, and top-level-projector matrices. The equivalent
label-based lookup is

```julia
slow_operators = pseudomode_operators(two_mode.supersite, :slow)
```

Each additional mode multiplies the local dimension before the PI coordinate
count is evaluated. Increasing several cutoffs simultaneously can therefore
be much more expensive than increasing one cutoff.

## Arbitrary qudits and system channels

Nothing in the supersite construction is qubit-specific. Here is a qutrit
with local and collective system relaxation:

```julia
d = 3
Delta = 0.6
Hq = Diagonal(ComplexF64[0, Delta, 2 * Delta])
lower = ComplexF64[
    0 1 0
    0 0 sqrt(2)
    0 0 0
]

qutrit_mode = BosonicPseudomode(
    1;
    label=:qutrit_bath,
    frequency=1.1,
    damping=0.3,
)

qutrit_embedding = pseudomode_model(
    N, Hq, qutrit_mode;
    couplings=(
        PseudomodeCoupling(
            lower;
            mode=:qutrit_bath,
            strength=0.08 + 0.02im,
        ),
    ),
    system_terms=(
        LocalJump(lower; rate=0.01),
        CollectiveJump(lower; rate=0.02 / N),
    ),
)

qutrit_excited = ComplexF64[0, 0, 1]
rho0_qutrit = pseudomode_product_state(
    qutrit_embedding.supersite, qutrit_excited,
)
```

The distinction between the two relaxation terms is preserved after lifting:
`LocalJump` remains a sum of independent channels, whereas `CollectiveJump`
remains one coherent summed jump.

## Tracing all modes back to the physical systems

The system-only state is obtained directly in PI coordinates. Prepare the
rectangular trace map once, and give every concurrent task its own workspace:

```julia
trace_plan = pseudomode_trace_plan(site)
trace_work = LocalFactorTraceWorkspace(trace_plan)

rho_system_ss = trace_pseudomodes(
    rho_ss, site;
    plan=trace_plan,
    workspace=trace_work,
)

@assert rho_system_ss.basis === trace_plan.output_basis
@assert rho_system_ss.basis.d == 2
@assert isapprox(trace(rho_system_ss), 1; atol=1e-10)

mean_z = real(
    collective_expectation(rho_system_ss, sigma_z),
) / N
```

For a repeated scan, preallocate the output as well:

```julia
rho_system_buffer = PIState(trace_plan.output_basis)
trace_pseudomodes!(
    rho_system_buffer, rho_ss, site, trace_plan, trace_work;
    check=false, # only when the surrounding workflow already validates rho_ss
)
```

All trailing pseudomodes are grouped into one local auxiliary factor before
applying [`LocalFactorTracePlan`](@ref). The trace keeps all `N` physical
systems; it is different from [`reduced_state`](@ref), which discards
particles at fixed local dimension. The prepared plan is tied to the exact
supersite basis, so a different cutoff needs a different plan.

## Cutoff convergence

The occupation of the highest retained mode level is a useful boundary
diagnostic:

```julia
mode_ops = pseudomode_operators(site, :cavity)
top_population = real(
    collective_expectation(rho_ss, mode_ops.top_projector),
) / N
```

A small `top_population` is necessary but not sufficient. The physical
quantity of interest must also stabilize when `nmax` is increased. Since
supersite states at different cutoffs belong to different bases, do not
compare their raw PI coordinate vectors. Trace the modes or compare a common
observable instead:

```julia
function cutoff_point(nmax)
    local_mode = BosonicPseudomode(
        nmax;
        label=:cavity,
        frequency=omega_c,
        damping=kappa,
    )
    local_coupling = PseudomodeCoupling(
        sigma_minus;
        mode=:cavity,
        strength=g,
    )
    local_embedding = pseudomode_model(
        N, (Omega / 2) * sigma_z, local_mode;
        couplings=(local_coupling,),
        system_terms=(pair_term,),
    )
    local_prepared = compile(
        local_embedding.model;
        backend=:matrixfree,
    )
    local_steady = stationary_state(
        local_prepared;
        algorithm=GMRESAlgorithm(krylovdim=40, maxiter=1000),
    )

    operators = only(local_embedding.mode_operators)
    boundary = real(
        collective_expectation(
            local_steady, operators.top_projector,
        ),
    ) / N

    physical = trace_pseudomodes(
        local_steady, local_embedding.supersite,
    )
    magnetization = real(
        collective_expectation(physical, sigma_z),
    ) / N

    (;nmax, boundary, magnetization)
end

cutoff_data = cutoff_point.((1, 2, 3))
```

Repeat this refinement for dynamics at common physical times when the desired
result is time dependent. Solver tolerance, time-step convergence, finite
`N`, and pseudomode cutoff are separate numerical claims.

## Performance checklist

- Compute `D` and `commutant_dimension(N,D)` before building a large model.
- Keep independent local damping on a complete Schur basis unless a requested
  restricted basis is explicitly validated.
- Use `compile(...; backend=:matrixfree)` when sparse materialization would
  dominate memory.
- Reuse the compiled model, Krylov workspaces, observable plans, and one
  `pseudomode_trace_plan` plus task-owned trace workspace per exact
  cutoff/basis; use `trace_pseudomodes!` when the output can also be reused.
- Keep the coupling strengths and frequencies as scalar term rates so
  [`compile_family`](@ref) can reuse fixed geometry in parameter scans.
- Monitor the top-level population and converge every reported system
  observable in the cutoff.
- Use a wider floating type rather than allowing a required scale to underflow
  or overflow.

The historical convenience call

```julia
independent_local_pseudomode_model(
    N, Hsystem, L;
    nmax, frequency, coupling_strength, damping,
)
```

remains available for one mode. The `BosonicPseudomode` /
`PseudomodeCoupling` workflow is the general route for multiple modes,
counter-rotating terms, system-term lifting, state preparation, and prepared
mode tracing.
