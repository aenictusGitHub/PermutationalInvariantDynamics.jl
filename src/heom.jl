"""
    HEOMBath(coupling, coefficients, frequencies;
             right_coefficients=nothing, residue=0, metadata=(;),
             check=true, atol=0, rtol=sqrt(eps(...)))

Finite exponential representation of one bosonic bath correlation function,

```math
C(t)=\\sum_k c_k\\exp(-\\nu_k t),\\qquad t\\geq0.
```

`coupling` is the fixed Hermitian PI system operator ``Q`` in
``H_\\mathrm{int}=Q\\otimes B``. `coefficients[k]` is the left coefficient
``c_k`` and `frequencies[k]` is ``\\nu_k``. By default the constructor
represents the conjugate correlation on the same pole list: real poles use
`conj(c_k)`, exact complex-conjugate pairs are cross-paired, and an absent
conjugate pole is appended with zero left coefficient. Pass
`right_coefficients` to specify the same-pole right coefficients explicitly
and disable this completion. Every pole must have a finite strictly positive
real part. A scalar coefficient/frequency pair is accepted as a convenience.
Integer-valued bath data must be exactly representable in the prepared scalar
type; otherwise preparation raises and requests an explicitly wider floating
input instead of rounding it. `residue` is an explicit real coefficient for
the white-noise approximation to omitted correlation terms. It is ignored by
`HEOMPlan(...; terminator=:none)` and included as
``2\\,\\mathtt{residue}\\,\\mathcal D[Q]`` only when
`terminator=:residue`. `metadata` must be an immutable named tuple.
"""
struct HEOMBath{O,C,R,F,D,M}
    coupling::O
    coefficients::C
    right_coefficients::R
    frequencies::F
    residue::D
    metadata::M
end

@inline _heom_isfinite(z::Number)=isfinite(real(z))&&isfinite(imag(z))

function _heom_complete_conjugate_correlation(coefficients,frequencies)
    original_count=length(coefficients)
    left=copy(coefficients);poles=copy(frequencies)
    right=[zero(value) for value in coefficients]
    matched=falses(original_count)
    for index in 1:original_count
        matched[index]&&continue
        pole=frequencies[index]
        if iszero(imag(pole))
            right[index]=conj(coefficients[index]);matched[index]=true
            continue
        end
        partner=findfirst(1:original_count) do candidate
            candidate!=index&&!matched[candidate]&&
                frequencies[candidate]==conj(pole)
        end
        if partner===nothing
            right[index]=zero(coefficients[index]);matched[index]=true
            push!(left,zero(coefficients[index]))
            push!(right,conj(coefficients[index]))
            push!(poles,conj(pole))
        else
            right[index]=conj(coefficients[partner])
            right[partner]=conj(coefficients[index])
            matched[index]=true;matched[partner]=true
        end
    end
    left,right,poles
end

function HEOMBath(coupling::PIOperator,coefficients::AbstractVector,
                  frequencies::AbstractVector;right_coefficients=nothing,
                  residue=0,metadata::NamedTuple=(;),
                  check::Bool=true,
                  atol::Real=0,
                  rtol::Real=sqrt(eps(_real_float_type(eltype(coupling.data)))))
    length(coefficients)==length(frequencies)||throw(DimensionMismatch(
        "bath coefficients and frequencies must have equal lengths"))
    isempty(coefficients)&&throw(ArgumentError(
        "an HEOM bath must contain at least one exponential term"))
    isfinite(atol)&&atol>=0||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    isfinite(rtol)&&rtol>=0||throw(ArgumentError(
        "rtol must be finite and nonnegative"))
    check&&!ishermitian(coupling;atol,rtol)&&throw(ArgumentError(
        "an HEOM bath coupling operator must be Hermitian"))
    cs=collect(coefficients);nus=collect(frequencies)
    for (index,c) in pairs(cs)
        c isa Number&&_heom_isfinite(c)||throw(ArgumentError(
            "bath coefficient $index must be a finite number"))
    end
    for (index,nu) in pairs(nus)
        nu isa Number&&_heom_isfinite(nu)||throw(ArgumentError(
            "bath frequency $index must be a finite number"))
        real(nu)>0||throw(ArgumentError(
            "bath frequency $index must have a strictly positive real part"))
    end
    rights=if right_coefficients===nothing
        cs,completed_rights,nus=
            _heom_complete_conjugate_correlation(cs,nus)
        completed_rights
    else
        values=right_coefficients isa Number ?
            [right_coefficients] : collect(right_coefficients)
        length(values)==length(cs)||throw(DimensionMismatch(
            "right bath coefficients and frequencies must have equal lengths"))
        values
    end
    for (index,c) in pairs(rights)
        c isa Number&&_heom_isfinite(c)||throw(ArgumentError(
            "right bath coefficient $index must be a finite number"))
    end
    residue isa Real&&isfinite(residue)||throw(ArgumentError(
        "bath residue must be a finite real number"))
    bath_metadata=merge((model=:finite_exponential,
                         decomposition=:explicit,
                         inverse_temperature=nothing,
                         approximate=false),metadata)
    HEOMBath{typeof(coupling),typeof(cs),typeof(rights),typeof(nus),
             typeof(residue),typeof(bath_metadata)}(
        copy(coupling),cs,rights,nus,residue,bath_metadata)
end

HEOMBath(coupling::PIOperator,coefficient::Number,frequency::Number;kwargs...)=
    HEOMBath(coupling,[coefficient],[frequency];kwargs...)

show(io::IO,bath::HEOMBath)=print(io,
    "HEOMBath(N=$(bath.coupling.basis.N), d=$(bath.coupling.basis.d), " *
    "model=$(bath.metadata.model), " *
    "exponentials=$(length(bath.coefficients)), residue=$(bath.residue))")

"""Return the immutable physical/decomposition metadata attached to `bath`."""
heom_bath_metadata(bath::HEOMBath)=bath.metadata

"""Return the explicit real white-noise residue coefficient of `bath`."""
heom_bath_residue(bath::HEOMBath)=bath.residue

function _physical_heom_parameters(values...)
    all(value->value isa Real&&isfinite(value),values)||throw(ArgumentError(
        "physical HEOM bath parameters must be finite real numbers"))
    promote_type(map(value->typeof(float(value)),values)...)
end

function _physical_heom_positive(value,name;allow_zero::Bool=false)
    valid=allow_zero ? value>=0 : value>0
    valid||throw(ArgumentError("$name must be " *
        (allow_zero ? "nonnegative" : "strictly positive")))
    value
end

function _physical_heom_term_count(value)
    value isa Integer&&value>=0||throw(ArgumentError(
        "matsubara_terms must be a nonnegative integer"))
    BigInt(value)<=typemax(Int)||throw(ArgumentError(
        "matsubara_terms must be representable as an Int"))
    Int(value)
end

# SPDX-SnippetBegin
# SPDX-SnippetCopyrightText: 2011-2026 QuTiP developers and contributors
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from QuTiP's qutip/core/environment.py Padé routines at revision
# e5dbb0195fdbf37fb39d4e52e27c80594f8eb655. Translated and modified for this
# package's precision, validation, indexing, and memory contracts. The complete
# upstream notice and exact source link are in THIRD_PARTY_NOTICES.md.
function _drude_pade_parameters(terms::Int,::Type{R},memory_budget) where R
    R in (Float32,Float64)||throw(ArgumentError(
        "Drude--Lorentz Padé decomposition currently requires Float32 or Float64 physical parameters; use decomposition=:matsubara for generic precision"))
    iszero(terms)&&return (R[],R[])
    BigInt(2)*BigInt(terms)<=typemax(Int)||throw(ArgumentError(
        "the Drude Padé eigensystem dimension exceeds Int indexing"))
    dimension=Base.checked_mul(2,terms)
    estimated_bytes=_performance_array_bytes(dimension,R,6)
    _require_performance_budget(
        "Drude--Lorentz Padé preparation",estimated_bytes,memory_budget;
        guidance="Reduce matsubara_terms, use decomposition=:matsubara, or pass memory_budget=Inf explicitly.")
    alpha=zeros(R,dimension,dimension)
    for index in 1:dimension-1
        k=R(index-1)
        value=inv(sqrt((R(2)*k+R(5))*(R(2)*k+R(3))))
        alpha[index,index+1]=value;alpha[index+1,index]=value
    end
    eigenvalues=eigvals(Symmetric(alpha))
    epsilon=R[-R(2)/eigenvalues[index] for index in 1:terms]
    chi=R[]
    if terms>1
        dimension_chi=dimension-1
        alpha_chi=zeros(R,dimension_chi,dimension_chi)
        for index in 1:dimension_chi-1
            k=R(index-1)
            value=inv(sqrt((R(2)*k+R(7))*(R(2)*k+R(5))))
            alpha_chi[index,index+1]=value
            alpha_chi[index+1,index]=value
        end
        eigenvalues_chi=eigvals(Symmetric(alpha_chi))
        chi=R[-R(2)/eigenvalues_chi[index] for index in 1:terms-1]
    end
    kappa=Vector{R}(undef,terms)
    prefactor=R(terms)*(R(2)*R(terms)+R(3))/R(2)
    for j in 1:terms
        value=prefactor
        for k in 1:terms-1
            denominator=epsilon[k]^2-epsilon[j]^2+(j==k ? one(R) : zero(R))
            value*=(chi[k]^2-epsilon[j]^2)/denominator
        end
        denominator=epsilon[terms]^2-epsilon[j]^2+
                    (j==terms ? one(R) : zero(R))
        value/=denominator
        isfinite(value)||throw(ArgumentError(
            "Drude Padé weight $j is not finite in $R"))
        kappa[j]=value
    end
    kappa,epsilon
end
# SPDX-SnippetEnd

function _heom_real_residue(value,scale,::Type{R},description) where R
    tolerance=R(64)*eps(R)*max(one(R),R(abs(scale)),R(abs(value)))
    abs(imag(value))<=tolerance||throw(ArgumentError(
        "$description produced a non-real omitted-correlation residue; " *
        "increase precision or use an explicit HEOMBath decomposition"))
    residue=R(real(value))
    isfinite(residue)||throw(ArgumentError(
        "$description residue is not finite in $R"))
    residue
end

"""
    drude_lorentz_bath(coupling, reorganization_energy, cutoff,
                       inverse_temperature; matsubara_terms=0,
                       decomposition=:matsubara, memory_budget=512MiB)

Construct the finite Matsubara decomposition of the Drude--Lorentz spectral
density

```math
J(\\omega)=\\frac{2\\lambda\\gamma\\omega}{\\gamma^2+\\omega^2}.
```

Units use ``\\hbar=k_B=1`` and `inverse_temperature` is ``\\beta``. The
returned [`HEOMBath`](@ref) contains the Drude pole and exactly
`matsubara_terms` positive thermal poles. `decomposition=:matsubara` supports
generic floating precision. `decomposition=:pade` uses the Hu--Xu--Yan
tridiagonal-eigenvalue construction and currently requires Float32 or
Float64 physical parameters. Its `residue` is the real
zero-frequency discrepancy of the omitted terms. Opt into the associated
white-noise correction explicitly with
`HEOMPlan(...; terminator=:residue)`. Singular coincident Drude/Matsubara
poles are rejected rather than evaluated as `Inf - Inf`.
The Padé eigensolve obeys `memory_budget`; `Inf` is the explicit opt-out.
"""
function drude_lorentz_bath(coupling::PIOperator,reorganization_energy,
                            cutoff,inverse_temperature;
                            matsubara_terms::Integer=0,
                            decomposition::Symbol=:matsubara,
                            memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
                            check::Bool=true,atol::Real=0,
                            rtol::Real=sqrt(eps(_real_float_type(
                                eltype(coupling.data)))))
    R=_physical_heom_parameters(
        reorganization_energy,cutoff,inverse_temperature)
    lambda=R(reorganization_energy);gamma=R(cutoff)
    beta=R(inverse_temperature)
    _physical_heom_positive(lambda,"reorganization_energy";allow_zero=true)
    _physical_heom_positive(gamma,"cutoff")
    _physical_heom_positive(beta,"inverse_temperature")
    K=_physical_heom_term_count(matsubara_terms)
    decomposition in (:matsubara,:pade)||throw(ArgumentError(
        "decomposition must be :matsubara or :pade"))
    piR=R(pi);half=one(R)/R(2)
    argument=beta*gamma*half
    tangent=tan(argument)
    isfinite(tangent)&&!iszero(tangent)||throw(ArgumentError(
        "the Drude pole is singular in this Matsubara representation; " *
        "change cutoff/inverse_temperature or use an explicitly combined limiting decomposition"))
    coefficients=Complex{R}[Complex{R}(
        lambda*gamma/tangent,-lambda*gamma)]
    frequencies=Complex{R}[Complex{R}(gamma)]
    if decomposition===:matsubara
        for k in 1:K
            nu=R(2)*piR*R(k)/beta
            denominator=nu*nu-gamma*gamma
            !iszero(denominator)&&isfinite(denominator)||throw(ArgumentError(
                "Matsubara pole $k coincides with the Drude pole; use an explicitly combined limiting decomposition"))
            coefficient=(R(4)*lambda*gamma/beta)*nu/denominator
            isfinite(coefficient)||throw(ArgumentError(
                "Drude Matsubara coefficient $k is not finite in $R"))
            push!(coefficients,Complex{R}(coefficient))
            push!(frequencies,Complex{R}(nu))
        end
    else
        kappa,epsilon=_drude_pade_parameters(K,R,memory_budget)
        temperature=inv(beta)
        for k in 1:K
            nu=epsilon[k]*temperature
            denominator=nu*nu-gamma*gamma
            !iszero(denominator)&&isfinite(denominator)||throw(ArgumentError(
                "Padé pole $k coincides with the Drude pole; use a different truncation or explicit limiting decomposition"))
            coefficient=(kappa[k]*temperature)*R(4)*lambda*gamma*nu/
                        denominator
            isfinite(coefficient)||throw(ArgumentError(
                "Drude Padé coefficient $k is not finite in $R"))
            push!(coefficients,Complex{R}(coefficient))
            push!(frequencies,Complex{R}(nu))
        end
    end
    target=Complex{R}(R(2)*lambda/(beta*gamma),-lambda)
    integral=sum(coefficients[index]/frequencies[index]
                 for index in eachindex(coefficients))
    residue_description=decomposition===:pade ?
        "Drude--Lorentz Pade decomposition" :
        "Drude--Lorentz Matsubara decomposition"
    residue=_heom_real_residue(target-integral,target,R,
                               residue_description)
    metadata=(model=:drude_lorentz,decomposition,
              inverse_temperature=beta,temperature=inv(beta),
              spectral_density_convention=:two_lambda_gamma,
              parameters=(reorganization_energy=lambda,cutoff=gamma),
              expansion_terms=K,approximate=true,
              residue_target=target,residue_interpretation=:white_noise)
    HEOMBath(coupling,coefficients,frequencies;
             residue,metadata,check,atol,rtol)
end

"""
    underdamped_brownian_bath(coupling, coupling_strength, damping,
                              resonance, inverse_temperature;
                              matsubara_terms=0)

Construct a Matsubara decomposition of the underdamped Brownian spectral
density

```math
J(\\omega)=\\frac{\\lambda^2\\gamma\\omega}
{(\\omega_0^2-\\omega^2)^2+\\gamma^2\\omega^2}.
```

This exact convention matters: `coupling_strength` is ``\\lambda`` and enters
quadratically. The constructor requires ``0<\\gamma<2\\omega_0`` and retains
the conjugate damped-oscillator poles plus `matsubara_terms` thermal poles.
The finite-tail residue is stored exactly as for
[`drude_lorentz_bath`](@ref).
"""
function underdamped_brownian_bath(coupling::PIOperator,coupling_strength,
                                   damping,resonance,inverse_temperature;
                                   matsubara_terms::Integer=0,
                                   check::Bool=true,atol::Real=0,
                                   rtol::Real=sqrt(eps(_real_float_type(
                                       eltype(coupling.data)))))
    R=_physical_heom_parameters(
        coupling_strength,damping,resonance,inverse_temperature)
    lambda=R(coupling_strength);gamma=R(damping);omega0=R(resonance)
    beta=R(inverse_temperature)
    _physical_heom_positive(lambda,"coupling_strength";allow_zero=true)
    _physical_heom_positive(gamma,"damping")
    _physical_heom_positive(omega0,"resonance")
    _physical_heom_positive(beta,"inverse_temperature")
    gamma<R(2)*omega0||throw(ArgumentError(
        "underdamped_brownian_bath requires damping < 2*resonance"))
    K=_physical_heom_term_count(matsubara_terms)
    Gamma=gamma/R(2)
    Omega=sqrt(omega0*omega0-Gamma*Gamma)
    z=beta*Complex{R}(Omega,Gamma)/R(2)
    amplitude=lambda*lambda/(R(4)*Omega)
    coefficients=Complex{R}[
        amplitude*(inv(tanh(z))-one(R)),
        amplitude*(inv(tanh(conj(z)))+one(R))]
    frequencies=Complex{R}[
        Complex{R}(Gamma,-Omega),Complex{R}(Gamma,Omega)]
    piR=R(pi)
    for k in 1:K
        nu=R(2)*piR*R(k)/beta
        denominator=((Complex{R}(Omega,Gamma))^2+nu^2)*
                    ((Complex{R}(Omega,-Gamma))^2+nu^2)
        !iszero(denominator)&&_heom_isfinite(denominator)||throw(ArgumentError(
            "Brownian Matsubara denominator $k is singular or nonfinite"))
        coefficient=-(R(2)*lambda^2*gamma/beta)*nu/denominator
        _heom_isfinite(coefficient)||throw(ArgumentError(
            "Brownian Matsubara coefficient $k is not finite in $R"))
        push!(coefficients,Complex{R}(coefficient))
        push!(frequencies,Complex{R}(nu))
    end
    target=gamma*lambda^2/(beta*omega0^4)
    integral=sum(real(coefficients[index]/frequencies[index])
                 for index in eachindex(coefficients))
    residue=_heom_real_residue(target-integral,target,R,
                               "underdamped Brownian Matsubara decomposition")
    metadata=(model=:underdamped_brownian,decomposition=:matsubara,
              inverse_temperature=beta,temperature=inv(beta),
              spectral_density_convention=:lambda_squared,
              parameters=(coupling_strength=lambda,damping=gamma,
                          resonance=omega0),
              matsubara_terms=K,approximate=true,
              residue_target=target,residue_interpretation=:white_noise)
    HEOMBath(coupling,coefficients,frequencies;
             residue,metadata,check,atol,rtol)
end

"""
    independent_local_pseudomode_model(N, system_hamiltonian, coupling;
        nmax, frequency, coupling_strength, damping,
        thermal_occupation=0, memory_budget=512*1024^2)

Build a finite-cutoff PI supersite model for `N` identical, *independent*
local damped pseudomodes. Each supersite contains one `d`-level system and one
oscillator truncated to occupations `0:nmax`, in `system ⊗ mode` basis
order. Its local Hamiltonian is

```math
H_{\\mathrm{site}}=H_S+\\omega a^\\dagger a
 +g\\left(L a^\\dagger+L^\\dagger a\\right),
```

and its independent Lindblad rates are
``\\kappa(\\bar n+1)\\mathcal D[a]`` and
``\\kappa\\bar n\\mathcal D[a^\\dagger]``. At zero occupation this realizes the
physical pole ``c=g^2`` and ``\\nu=\\kappa/2+i\\omega`` for channel ``L``.
Hermitian ``L=Q`` reduces to ``gQ(a+a^\\dagger)``. The returned named tuple
contains `basis`, `model`, the lifted local operators, and immutable cutoff
metadata so additional PI interactions can be built on the exact same basis.
For several local modes, automatic lifting of system-only `p`-body terms, and
prepared local-mode tracing, use the generalized [`pseudomode_model`](@ref)
workflow.

This is a time-local pseudomode embedding, not a collective-coupling
[`HEOMPlan`](@ref). It is the safe existing-coordinate route for identical
independent local environments, but only represents pole decompositions that
admit such a positive damped-mode realization. The oscillator cutoff is an
explicit approximation and must be converged in `nmax`.
"""
function independent_local_pseudomode_model(
        N::Integer,system_hamiltonian::AbstractMatrix,
        coupling::AbstractMatrix;nmax::Integer,frequency,coupling_strength,
        damping,thermal_occupation=0,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    N isa Integer&&!(N isa Bool)||throw(ArgumentError(
        "N must be an integer"))
    N>=1||throw(ArgumentError("N must be positive"))
    BigInt(N)<=typemax(Int)||throw(ArgumentError(
        "N must be representable as an Int"))
    nmax isa Integer&&!(nmax isa Bool)||throw(ArgumentError(
        "nmax must be an integer"))
    nmax>=1||throw(ArgumentError("nmax must be positive"))
    BigInt(nmax)<typemax(Int)||throw(ArgumentError(
        "nmax is too large for local indexing"))
    d=size(system_hamiltonian,1)
    size(system_hamiltonian)==(d,d)&&d>0||throw(DimensionMismatch(
        "system_hamiltonian must be a nonempty square matrix"))
    size(coupling)==(d,d)||throw(DimensionMismatch(
        "coupling must match system_hamiltonian"))
    ishermitian(system_hamiltonian)||throw(ArgumentError(
        "system_hamiltonian must be Hermitian"))
    all(_heom_isfinite,_supersite_stored_values(system_hamiltonian))&&
        all(_heom_isfinite,_supersite_stored_values(coupling))||
        throw(ArgumentError(
        "system_hamiltonian and coupling must contain only finite values"))
    all(value->value isa Real&&!(value isa Bool)&&isfinite(value),
        (frequency,coupling_strength,damping,thermal_occupation))||
        throw(ArgumentError(
        "pseudomode parameters must be finite real numbers"))
    _physical_heom_positive(
        frequency,"frequency";allow_zero=true)
    _physical_heom_positive(
        coupling_strength,"coupling_strength";allow_zero=true)
    _physical_heom_positive(damping,"damping")
    _physical_heom_positive(
        thermal_occupation,"thermal_occupation";allow_zero=true)
    R=promote_type(_real_float_type(eltype(system_hamiltonian)),
                   _real_float_type(eltype(coupling)))
    for value in (frequency,coupling_strength,damping,
                  thermal_occupation)
        R=_supersite_promote_parameter_type(R,value)
    end
    mode=BosonicPseudomode(
        Int(nmax);frequency,damping,
        thermal_occupation,label=:mode,T=R,memory_budget)
    coupling_specification=PseudomodeCoupling(
        coupling;mode=:mode,strength=coupling_strength,
        memory_budget)
    generalized=pseudomode_model(
        Int(N),system_hamiltonian,mode;
        couplings=coupling_specification,memory_budget)
    supersite=generalized.supersite
    basis=generalized.basis
    site_hamiltonian=generalized.site_hamiltonian
    lifted_system_hamiltonian=
        generalized.lifted_system_hamiltonian
    operators=only(generalized.mode_operators)

    precision_bits=generalized.resource_estimates.precision_bits
    components=(coupling_specification.operator,mode.identity)
    extra_output_bytes=_supersite_sparse_csc_bytes(
        supersite.basis.d,
        BigInt(_supersite_structural_nnz(
            coupling_specification.operator))*mode.levels,
        Complex{R};bigfloat_precision=precision_bits)
    extra_peak=_supersite_tensor_peak_bytes(
        supersite.factor_dimensions,components,Complex{R};
        bigfloat_precision=precision_bits)
    legacy_peak=max(
        generalized.resource_estimates.setup_peak_bytes,
        generalized.resource_estimates.site_bytes+
        generalized.resource_estimates.generated_retained_bytes+
        extra_output_bytes+extra_peak)
    _require_performance_budget(
        "legacy pseudomode model construction",legacy_peak,memory_budget;
        guidance="Reduce N, nmax, or the system-operator support.")
    lifted_coupling=lift_system_operator(
        supersite,coupling_specification.operator;memory_budget=Inf)

    # Preserve the historical three-term layout (one combined Hamiltonian and
    # one or two jumps) while all physical lowering comes from the generalized
    # builder above.
    terms=(LocalHamiltonian(site_hamiltonian;check=false),
           generalized.damping_terms...)
    model=PIModel(basis,terms)
    omega=mode.frequency
    g=real(coupling_specification.strength)
    kappa=mode.damping
    occupation=mode.thermal_occupation
    dsite=basis.d
    resource_estimates=merge(
        generalized.resource_estimates,
        (legacy_setup_peak_bytes=legacy_peak,
         legacy_lifted_coupling_bytes=extra_output_bytes))
    metadata=(embedding=:independent_local_pseudomode,
              exact_permutation_symmetry=true,
              oscillator_cutoff=Int(nmax),cutoff_approximation=true,
              system_dimension=d,local_dimension=dsite,
              frequency=omega,coupling_strength=g,damping=kappa,
              thermal_occupation=occupation,
              coupling_hermitian=ishermitian(
                  coupling_specification.operator),
              zero_temperature_pole=(coefficient=abs2(g),
                  frequency=Complex{R}(kappa/R(2),omega)),
              generalized_metadata=generalized.metadata,
              resource_estimates)
    (;supersite,basis,model,site_hamiltonian,lifted_system_hamiltonian,
      lifted_coupling,annihilation=operators.annihilation,
      number_operator=operators.number_operator,
      resource_estimates,metadata)
end

_heom_basis(source)=_operator_basis(source)

_prepare_heom_system(source::PIModel)=compile(source;backend=:matrixfree)
_prepare_heom_system(source::AbstractMatrix)=copy(source)
_prepare_heom_system(source)=source

function _heom_bath_tuple(baths)
    baths isa HEOMBath&&return (baths,)
    tuple=Tuple(baths)
    all(bath->bath isa HEOMBath,tuple)||throw(ArgumentError(
        "every bath must be an HEOMBath"))
    tuple
end

function _heom_compositions!(output,current,position,remaining)
    if position>length(current)
        iszero(remaining)&&push!(output,copy(current))
        return output
    end
    for occupation in 0:remaining
        current[position]=occupation
        _heom_compositions!(output,current,position+1,remaining-occupation)
    end
    output
end

function _heom_multiindices(exponents::Int,max_depth::Int)
    if iszero(exponents)
        return [Int[]]
    end
    output=Vector{Vector{Int}}()
    current=zeros(Int,exponents)
    for depth in 0:max_depth
        _heom_compositions!(output,current,1,depth)
    end
    output
end

function _heom_checked_convert(::Type{T},value,description) where T
    converted=try
        T(value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("$description is not representable in $T; use a wider scalar type"))
    end
    _heom_isfinite(converted)||throw(ArgumentError(
        "$description is not finite in $T; use a wider scalar type"))
    if real(value) isa Integer&&imag(value) isa Integer&&converted!=value
        throw(ArgumentError(
            "$description is not exactly representable in $T; pass it in a wider floating type"))
    end
    converted
end

function _heom_scalar_type(system,baths)
    T=_complex_float_type(eltype(system))
    for bath in baths
        T=promote_type(T,eltype(bath.coupling.data))
        for value in bath.coefficients
            T=_heom_promote_value_type(T,value)
        end
        for value in bath.right_coefficients
            T=_heom_promote_value_type(T,value)
        end
        for value in bath.frequencies
            T=_heom_promote_value_type(T,value)
        end
        T=_heom_promote_value_type(T,bath.residue)
    end
    _complex_float_type(T)
end

function _heom_promote_value_type(::Type{T},value) where T
    if real(value) isa Integer&&imag(value) isa Integer
        converted=try
            T(value)
        catch
            nothing
        end
        converted!==nothing&&_heom_isfinite(converted)&&converted==value&&
            return T
    end
    promote_type(T,_complex_float_type(typeof(value)))
end

function _heom_coupling_blocks(basis,coupling,::Type{T}) where T
    R=_real_float_type(T)
    [Matrix{T}(_divide_by_schur_multiplicity_scale(
         Matrix{T}(coefficient_block(coupling,partition)),R,partition))
     for partition in basis.sectors]
end

"""
    HEOMPlan(system, baths; max_depth, basis=nothing, terminator=:none,
             scaling=:unscaled, scaling_factors=nothing,
             importance_cutoff=0,
             importance_metric=:normalized_coupling_decay)

Prepare a finite, matrix-free hierarchy of equations of motion for a PI
system. `system` may be a `PIModel`, compiled PI model, Liouvillian plan,
matrix-free Liouvillian, or square matrix. Pass `basis` only when it cannot be
inferred from `system`.

For flattened exponential index ``k`` and hierarchy multi-index ``n``, the
implemented convention is

```math
\\dot\\rho_n=\\mathcal L_S\\rho_n-\\sum_k n_k\\nu_k\\rho_n
-i\\sum_k[Q_{b(k)},\\rho_{n+e_k}]
-i\\sum_k n_k\\left(\\ell_kQ_{b(k)}\\rho_{n-e_k}
-r_k\\rho_{n-e_k}Q_{b(k)}\\right).
```

Here ``\\ell_k`` and ``r_k`` are the prepared left and right correlation
coefficients on the common pole list. They are not assumed to be termwise
complex conjugates when ``\\nu_k`` is complex.

Indices with total depth larger than `max_depth` are identically zero.
`terminator=:none` is the original hard truncation. `terminator=:residue`
also adds ``2\\Delta_b\\mathcal D[Q_b]`` for each bath's explicitly stored real
white-noise residue ``\\Delta_b``; it does not infer one from arbitrary pole
data. Convergence must still be checked by increasing `max_depth` and the
number of correlation poles.

A positive `importance_cutoff` is an explicit additional approximation. It
retains a deterministic downward-closed order ideal using the documented
`:normalized_coupling_decay` score; all missing upward neighbors are zero.
Zero preserves the complete total-depth hierarchy bit for bit. Setup and
application scale with the PI dimension, never with ``d^N``.

The default `scaling=:unscaled` preserves the conventional ADOs above. With
`scaling=:scaled`, the stored coordinate is

```math
\\widehat\\rho_n=\\rho_n/s_n,\\qquad
s_n=\\prod_k\\sqrt{n_k!\\,a_k^{n_k}}.
```

The positive pole factors ``a_k`` default to
``\\max(|\\ell_k|,|r_k|)`` (and exactly one when both coefficients vanish), or
may be supplied explicitly through `scaling_factors`. This is an exact
diagonal similarity transformation, not a hierarchy approximation;
[`heom_ado`](@ref) always returns the corresponding unscaled physical ADO.
"""
struct _PackedHEOMTopology{I}
    lower::Vector{I}
    upper::Vector{I}
    term::Vector{I}
    level::Vector{I}
    ado_edge_ptr::Vector{I}
    incident_edges::Vector{I}
    coupling_groups::Int
end

function _heom_topology_index_type(maximum_index::Int)
    maximum_index<=typemax(Int32) ? Int32 : Int
end

function _heom_packed_topology(multiindices,lookup,exponent_baths,
                               max_depth::Int)
    number_ados=length(multiindices);K=length(exponent_baths)
    BigInt(K)*BigInt(number_ados)<=typemax(Int)||throw(ArgumentError(
        "the requested HEOM edge topology exceeds Int indexing"))
    lower_int=Int[];upper_int=Int[];term_int=Int[];level_int=Int[]
    sizehint!(lower_int,min(K*number_ados,typemax(Int)))
    sizehint!(upper_int,length(lower_int));sizehint!(term_int,length(lower_int))
    sizehint!(level_int,length(lower_int))
    candidate=zeros(Int,K)
    for lower_ado in eachindex(multiindices)
        occupations=multiindices[lower_ado]
        sum(occupations)<max_depth||continue
        for term in 1:K
            copyto!(candidate,occupations);candidate[term]+=1
            upper_ado=get(lookup,Tuple(candidate),0)
            iszero(upper_ado)&&continue
            push!(lower_int,lower_ado);push!(upper_int,upper_ado)
            push!(term_int,term);push!(level_int,occupations[term]+1)
        end
    end
    number_edges=length(lower_int)
    2BigInt(number_edges)+1<=typemax(Int)||throw(ArgumentError(
        "the retained HEOM incidence topology exceeds Int indexing"))
    I=_heom_topology_index_type(max(number_ados,K,max_depth,
                                    2number_edges+1))
    lower=I.(lower_int);upper=I.(upper_int)
    term_index=I.(term_int);level=I.(level_int)
    degrees=zeros(Int,number_ados)
    for edge in 1:number_edges
        degrees[Int(lower[edge])]+=1;degrees[Int(upper[edge])]+=1
    end

    ado_edge_ptr=Vector{I}(undef,number_ados+1)
    cursor=1
    for ado in 1:number_ados
        ado_edge_ptr[ado]=I(cursor)
        cursor+=degrees[ado]
    end
    ado_edge_ptr[end]=I(cursor)
    cursor==2number_edges+1||error("internal HEOM incidence-count mismatch")
    incident_edges=Vector{I}(undef,2number_edges)
    next_position=Int.(view(ado_edge_ptr,1:number_ados))
    for current_edge in 1:number_edges
        lower_ado=Int(lower[current_edge]);upper_ado=Int(upper[current_edge])
        incident_edges[next_position[lower_ado]]=I(current_edge)
        next_position[lower_ado]+=1
        incident_edges[next_position[upper_ado]]=I(current_edge)
        next_position[upper_ado]+=1
    end

    coupling_groups=0
    for ado in 1:number_ados
        first_edge=Int(ado_edge_ptr[ado])
        last_edge=Int(ado_edge_ptr[ado+1])-1
        if first_edge<=last_edge
            segment=view(incident_edges,first_edge:last_edge)
            sort!(segment;by=current_edge->begin
                edge_number=Int(current_edge)
                term=Int(term_index[edge_number])
                (exponent_baths[term],term,Int(lower[edge_number]),
                 Int(upper[edge_number]))
            end)
            previous_bath=0
            for current_edge in segment
                term=Int(term_index[Int(current_edge)])
                bath=exponent_baths[term]
                if bath!=previous_bath
                    coupling_groups+=1;previous_bath=bath
                end
            end
        end
    end
    _PackedHEOMTopology{I}(lower,upper,term_index,level,ado_edge_ptr,
                           incident_edges,coupling_groups)
end

function _heom_importance_weights(coefficients,right_coefficients,
                                  frequencies,::Type{R}) where R
    weights=Vector{R}(undef,length(coefficients))
    for term in eachindex(weights)
        amplitude=max(abs(coefficients[term]),abs(right_coefficients[term]))
        frequency=abs(frequencies[term])
        # sqrt(amplitude / frequency^2), evaluated without squaring the pole.
        ratio=sqrt(R(amplitude))/R(frequency)
        isfinite(ratio)||throw(ArgumentError(
            "HEOM importance ratio $term is not finite in $R"))
        # Map the dimensional estimate monotonically into [0,1), keeping each
        # occupation step non-increasing and hence the retained set downward
        # closed under every hierarchy parent.
        weights[term]=ratio/hypot(one(R),ratio)
    end
    weights
end

function _heom_multiindex_importance(index,weights,::Type{R}) where R
    value=one(R)
    for term in eachindex(index)
        for occupation in 1:index[term]
            value*=weights[term]/sqrt(R(occupation))
        end
    end
    value
end

function _heom_pruned_multiindices(coefficients,right_coefficients,
                                   frequencies,max_depth::Int,cutoff,
                                   metric::Symbol,::Type{R}) where R
    metric===:normalized_coupling_decay||throw(ArgumentError(
        "importance_metric must be :normalized_coupling_decay"))
    cutoff isa Real&&isfinite(cutoff)&&0<=cutoff<=1||throw(ArgumentError(
        "importance_cutoff must be finite and lie between zero and one"))
    cutoffR=R(cutoff)
    cutoffR==cutoff||throw(ArgumentError(
        "importance_cutoff is not exactly representable in $R"))
    K=length(coefficients)
    if iszero(cutoffR)
        indices=_heom_multiindices(K,max_depth)
        # Preserve the original complete-hierarchy setup path. Scores are
        # diagnostic only and are evaluated lazily if requested.
        return indices,R[],cutoffR
    end
    weights=_heom_importance_weights(
        coefficients,right_coefficients,frequencies,R)
    root=zeros(Int,K)
    indices=Vector{Vector{Int}}([root]);importances=R[one(R)]
    lookup=Dict{Tuple,Int}(Tuple(root)=>1)
    frontier=Vector{Vector{Int}}([root])
    for depth in 1:max_depth
        candidates=Set{Tuple}()
        for parent in frontier,term in 1:K
            candidate=copy(parent);candidate[term]+=1
            push!(candidates,Tuple(candidate))
        end
        ordered=sort!(collect(candidates))
        next_frontier=Vector{Vector{Int}}()
        for key in ordered
            candidate=collect(key)
            # Requiring every immediate parent makes the retained hierarchy an
            # order ideal even in the presence of finite-precision ties.
            all_parents=true
            for term in 1:K
                iszero(candidate[term])&&continue
                parent=copy(candidate);parent[term]-=1
                if !haskey(lookup,Tuple(parent))
                    all_parents=false;break
                end
            end
            all_parents||continue
            importance=_heom_multiindex_importance(candidate,weights,R)
            importance>=cutoffR||continue
            push!(indices,candidate);push!(importances,importance)
            lookup[key]=length(indices);push!(next_frontier,candidate)
        end
        frontier=next_frontier
        isempty(frontier)&&break
    end
    indices,importances,cutoffR
end

"""
    HEOMPlan(system, baths; max_depth, basis=nothing,
             terminator=:none, scaling=:unscaled,
             scaling_factors=nothing, importance_cutoff=0,
             importance_metric=:normalized_coupling_decay)

Prepared finite-exponential PI hierarchy of equations of motion. The root ADO
is the physical reduced state, while all auxiliary ADOs use the same PI
coefficient coordinates. `max_depth` applies a hard hierarchy truncation;
convergence must be checked by increasing it. `scaling=:scaled` applies the
documented exact diagonal similarity transform and does not approximate the
hierarchy.

The plan owns immutable system, coupling, pole, multi-index, and packed-edge
topology data. Numerical application scratch belongs to [`HEOMWorkspace`](@ref)
or [`HEOMEvolutionWorkspace`](@ref). Production application is matrix-free and
never constructs a `d^N` system object. See [`HEOMBath`](@ref) for the
finite-correlation convention and [`heom_depth_convergence`](@ref) for an
explicit truncation study.
"""
struct HEOMPlan{B,S,C,E,I,D,G,V,R,F,T}
    basis::B
    system::S
    coupling_blocks::C
    exponent_baths::E
    coefficients::V
    right_coefficients::V
    frequencies::V
    multiindices::I
    index::D
    topology::G
    upward_level_factors::Matrix{R}
    downward_level_factors::Matrix{R}
    ado_scales::Vector{R}
    pole_scales::Vector{R}
    residue_coefficients::Vector{R}
    bath_metadata::Tuple
    ado_importances::Vector{R}
    decays::V
    full_ado_count::BigInt
    max_depth::Int
    npi::Int
    tracevec::F
    Ttype::Type{T}
    autonomous::Bool
    terminator::Symbol
    scaling::Symbol
    importance_cutoff::R
    importance_metric::Symbol
end

function _heom_scaling_data(coefficients,right_coefficients,multiindices,
                            lookup,max_depth,scaling,scaling_factors,
                            ::Type{T}) where T
    scaling in (:unscaled,:scaled)||throw(ArgumentError(
        "scaling must be :unscaled or :scaled"))
    R=_real_float_type(T);K=length(coefficients)
    if scaling===:unscaled
        scaling_factors===nothing||throw(ArgumentError(
            "scaling_factors may be supplied only with scaling=:scaled"))
        pole_scales=ones(R,K)
    else
        raw=scaling_factors===nothing ?
            [begin
                 value=max(abs(coefficients[k]),abs(right_coefficients[k]))
                 iszero(value) ? one(R) : value
             end for k in 1:K] : collect(scaling_factors)
        length(raw)==K||throw(DimensionMismatch(
            "scaling_factors must contain one value per prepared HEOM pole"))
        pole_scales=Vector{R}(undef,K)
        for k in 1:K
            value=raw[k]
            value isa Real&&isfinite(value)&&value>0||throw(ArgumentError(
                "HEOM scaling factor $k must be finite, real, and positive"))
            converted=try
                R(value)
            catch error
                error isa InexactError||error isa OverflowError||rethrow()
                throw(ArgumentError(
                    "HEOM scaling factor $k is not representable in $R"))
            end
            isfinite(converted)&&converted>0||throw(ArgumentError(
                "HEOM scaling factor $k is not positive and finite in $R"))
            !iszero(value)&&iszero(converted)&&throw(ArgumentError(
                "HEOM scaling factor $k underflows in $R; use wider precision"))
            converted==value||throw(ArgumentError(
                "HEOM scaling factor $k is not exactly representable in $R; prepare the system or bath at wider precision"))
            pole_scales[k]=converted
        end
    end
    number_ados=length(multiindices)
    ado_scales=ones(R,number_ados)
    if scaling===:scaled
        parent_index=zeros(Int,K)
        for ado in 2:number_ados
            occupations=multiindices[ado]
            term=findfirst(>(0),occupations)
            term===nothing&&error("internal scaled-HEOM hierarchy mismatch")
            copyto!(parent_index,occupations);parent_index[term]-=1
            parent=get(lookup,Tuple(parent_index),0)
            parent>0||error("internal scaled-HEOM parent mismatch")
            ratio=sqrt(R(occupations[term]))*sqrt(pole_scales[term])
            isfinite(ratio)&&ratio>0||throw(ArgumentError(
                "scaled HEOM coordinate ratio for ADO $ado is not representable in $R; use wider precision"))
            scale=ado_scales[parent]*ratio
            isfinite(scale)&&scale>0||throw(ArgumentError(
                "scaled HEOM coordinate factor for ADO $ado is not representable in $R; reduce max_depth or use wider precision"))
            ado_scales[ado]=scale
        end
    end
    upward_level_factors=Matrix{R}(undef,max_depth,K)
    downward_level_factors=Matrix{R}(undef,max_depth,K)
    for term in 1:K,occupation in 1:max_depth
        upward_factor=scaling===:scaled ?
            sqrt(R(occupation))*sqrt(pole_scales[term]) : one(R)
        downward_factor=scaling===:scaled ?
            sqrt(R(occupation))/sqrt(pole_scales[term]) : R(occupation)
        isfinite(upward_factor)&&upward_factor>0||throw(ArgumentError(
            "scaled HEOM upward factor for occupation $occupation and pole $term is not representable in $R; use wider precision"))
        isfinite(downward_factor)&&downward_factor>0||throw(ArgumentError(
            "scaled HEOM downward factor for occupation $occupation and pole $term is not representable in $R; use wider precision"))
        upward_level_factors[occupation,term]=upward_factor
        downward_level_factors[occupation,term]=downward_factor
        end
    ado_scales,pole_scales,upward_level_factors,downward_level_factors
end

function HEOMPlan(source,baths;max_depth::Integer,basis=nothing,
                  terminator::Symbol=:none,scaling=:unscaled,
                  scaling_factors=nothing,importance_cutoff::Real=0,
                  importance_metric::Symbol=:normalized_coupling_decay)
    max_depth>=0||throw(ArgumentError("max_depth must be nonnegative"))
    BigInt(max_depth)<=typemax(Int)||throw(ArgumentError(
        "max_depth must be representable as an Int"))
    terminator in (:none,:residue)||throw(ArgumentError(
        "terminator must be :none or :residue"))
    prepared=_prepare_heom_system(source)
    inferred=_heom_basis(prepared)
    selected_basis=basis===nothing ? inferred : basis
    selected_basis isa PIBasis||throw(ArgumentError(
        "the PI basis cannot be inferred from this system; pass basis=..."))
    inferred===nothing||inferred===selected_basis||throw(ArgumentError(
        "the supplied basis is not the exact basis owned by the system"))
    size(prepared)==(length(selected_basis),length(selected_basis))||
        throw(DimensionMismatch("system Liouvillian and PI basis dimensions differ"))
    length(selected_basis)>0||throw(ArgumentError(
        "HEOM requires a nonempty retained PI basis"))
    bath_tuple=_heom_bath_tuple(baths)
    for (number,bath) in pairs(bath_tuple)
        bath.coupling.basis===selected_basis||throw(ArgumentError(
            "HEOM bath $number uses a different PI basis"))
    end
    T=_heom_scalar_type(prepared,bath_tuple)
    _check_liouvillian_source_precision(prepared,T,"HEOM coordinate")
    coefficients=T[];right_coefficients=T[];frequencies=T[]
    R=_real_float_type(T)
    residue_coefficients=Vector{R}(undef,length(bath_tuple))
    bath_metadata=map(bath->bath.metadata,bath_tuple)
    exponent_baths=Int[]
    coupling_blocks=Vector{Vector{Matrix{T}}}(undef,length(bath_tuple))
    for (bath_number,bath) in pairs(bath_tuple)
        coupling_blocks[bath_number]=_heom_coupling_blocks(
            selected_basis,bath.coupling,T)
        residue=_heom_checked_convert(
            T,bath.residue,"bath $bath_number white-noise residue")
        iszero(imag(residue))||throw(ArgumentError(
            "bath $bath_number white-noise residue is not real in $T"))
        residue_coefficients[bath_number]=R(real(residue))
        for term in eachindex(bath.coefficients)
            push!(coefficients,_heom_checked_convert(
                T,bath.coefficients[term],"bath $bath_number coefficient $term"))
            push!(right_coefficients,_heom_checked_convert(
                T,bath.right_coefficients[term],
                "bath $bath_number right coefficient $term"))
            frequency=_heom_checked_convert(
                T,bath.frequencies[term],"bath $bath_number frequency $term")
            real(frequency)>0||throw(ArgumentError(
                "bath $bath_number frequency $term lost its positive real part in $T; use a wider scalar type"))
            push!(frequencies,frequency)
            push!(exponent_baths,bath_number)
        end
    end
    K=length(coefficients);D=Int(max_depth)
    number_big=exact_binomial(BigInt(K)+BigInt(D),BigInt(D))
    iszero(importance_cutoff)&&number_big>typemax(Int)&&throw(ArgumentError(
        "the requested HEOM hierarchy has $number_big ADOs, exceeding Int indexing; an explicit positive importance_cutoff may retain a smaller approximate hierarchy"))
    npi=length(selected_basis)
    iszero(importance_cutoff)&&number_big*BigInt(npi)>typemax(Int)&&throw(ArgumentError(
        "the requested HEOM coordinate dimension exceeds Int indexing"))
    multiindices,ado_importances,prepared_cutoff=
        _heom_pruned_multiindices(coefficients,right_coefficients,
                                  frequencies,D,importance_cutoff,
                                  importance_metric,R)
    iszero(prepared_cutoff)&&length(multiindices)!=Int(number_big)&&error(
        "internal HEOM hierarchy enumeration mismatch")
    BigInt(length(multiindices))*BigInt(npi)<=typemax(Int)||throw(ArgumentError(
        "the retained HEOM coordinate dimension exceeds Int indexing"))
    lookup=Dict{Tuple,Int}(Tuple(index)=>position
                           for (position,index) in pairs(multiindices))
    topology=_heom_packed_topology(
        multiindices,lookup,exponent_baths,D)
    ado_scales,pole_scales,upward_level_factors,downward_level_factors=
        _heom_scaling_data(coefficients,right_coefficients,multiindices,
                           lookup,D,scaling,scaling_factors,T)
    decays=zeros(T,length(multiindices))
    for (position,index) in pairs(multiindices)
        value=zero(T)
        @inbounds for term in 1:K
            decay_term=index[term]*frequencies[term]
            _heom_isfinite(decay_term)||throw(ArgumentError(
                "HEOM decay for ADO $position and pole $term overflows in $T; reduce max_depth or use wider bath/system precision"))
            value+=decay_term
            _heom_isfinite(value)||throw(ArgumentError(
                "accumulated HEOM decay for ADO $position overflows in $T; reduce max_depth or use wider bath/system precision"))
            if index[term]>0
                _heom_isfinite(index[term]*coefficients[term])&&
                    _heom_isfinite(index[term]*right_coefficients[term])||
                    throw(ArgumentError(
                    "HEOM downward coefficient for ADO $position and pole $term overflows in $T; reduce max_depth or use wider bath precision"))
            end
        end
        decays[position]=value
    end
    tracevec=_resize_trace_functional(
        _trace_functional(selected_basis,T),
        npi*length(multiindices))
    HEOMPlan{typeof(selected_basis),typeof(prepared),typeof(coupling_blocks),
             typeof(exponent_baths),typeof(multiindices),typeof(lookup),
             typeof(topology),typeof(coefficients),eltype(ado_scales),
             typeof(tracevec),T}(
        selected_basis,prepared,coupling_blocks,exponent_baths,
        coefficients,right_coefficients,frequencies,multiindices,lookup,
        topology,upward_level_factors,downward_level_factors,
        ado_scales,pole_scales,residue_coefficients,bath_metadata,
        ado_importances,decays,number_big,
        D,npi,tracevec,T,isautonomous(prepared),terminator,scaling,
        prepared_cutoff,importance_metric)
end

size(plan::HEOMPlan)=(length(plan.tracevec),length(plan.tracevec))
size(plan::HEOMPlan,index::Integer)=index in (1,2) ? length(plan.tracevec) : 1
eltype(plan::HEOMPlan)=plan.Ttype
isautonomous(plan::HEOMPlan)=plan.autonomous

show(io::IO,plan::HEOMPlan)=print(io,
    "HEOMPlan(N=$(plan.basis.N), d=$(plan.basis.d), " *
    "exponentials=$(length(plan.coefficients)), max_depth=$(plan.max_depth), " *
    "ADOs=$(length(plan.multiindices))/$(plan.full_ado_count), " *
    "dimension=$(size(plan,1)), scaling=$(plan.scaling), " *
    "terminator=$(plan.terminator), autonomous=$(plan.autonomous))")

"""Return the number of auxiliary density operators retained by `plan`."""
heom_number_ados(plan::HEOMPlan)=length(plan.multiindices)

"""
    heom_multiindices(plan)

Return detached copies of all retained hierarchy multi-indices, ordered first
by total depth and then by ascending lexicographic occupation order.
"""
heom_multiindices(plan::HEOMPlan)=map(copy,plan.multiindices)

"""
    heom_ado_importances(plan)

Return a detached vector of the dimensionless heuristic scores used by
importance pruning. The root score is one. Scores are diagnostic estimates,
not error bounds.
"""
function heom_ado_importances(plan::HEOMPlan)
    isempty(plan.ado_importances)||return copy(plan.ado_importances)
    R=_real_float_type(plan.Ttype)
    weights=_heom_importance_weights(
        plan.coefficients,plan.right_coefficients,plan.frequencies,R)
    R[_heom_multiindex_importance(index,weights,R)
      for index in plan.multiindices]
end

"""
    heom_hierarchy_metadata(plan)

Return immutable metadata describing full versus retained ADO counts,
importance pruning, scaling, and the explicit residue terminator. The
`full_ados` field is an exact `BigInt`.
"""
heom_hierarchy_metadata(plan::HEOMPlan)=(
    full_ados=plan.full_ado_count,
    retained_ados=heom_number_ados(plan),
    pruned_ados=plan.full_ado_count-BigInt(heom_number_ados(plan)),
    importance_cutoff=plan.importance_cutoff,
    importance_metric=plan.importance_metric,
    pruning_approximation=!iszero(plan.importance_cutoff),
    terminator=plan.terminator,
    residue_coefficients=copy(plan.residue_coefficients),
    bath_metadata=plan.bath_metadata,
    scaling=plan.scaling)

"""
    heom_coordinate_scale(plan, label)

Return the positive scalar ``s_n`` relating the stored ADO coordinate to the
conventional unscaled ADO, ``rho_n = s_n * rhohat_n``. `label` may be the
one-based ADO position or its occupation vector. The value is exactly one for
an unscaled plan and for the root ADO.
"""
heom_coordinate_scale(plan::HEOMPlan,label)=
    plan.ado_scales[_heom_ado_index(plan,label)]

_heom_system_workspace(system)=_linear_operator_workspace(system)

const _HEOM_SYSTEM_BATCH_WIDTH=16

"""
    HEOMWorkspace(plan; batch_columns=1)

Task-owned mutable storage for matrix-free HEOM generator application. A
workspace may be reused sequentially but not
concurrently. RK4 stage arrays live separately in
[`HEOMEvolutionWorkspace`](@ref), so spectral and steady-state calculations do
not retain three hierarchy-sized evolution vectors. `batch_columns` prepares
the bounded system-action scratch for that many hierarchy right-hand sides;
the retained system batch width is capped by the implementation's fixed
chunk width.
"""
struct HEOMWorkspace{T,P,S}
    plan::P
    system::S
    left::Vector{T}
    right::Vector{T}
    mixed::Vector{T}
end

function HEOMWorkspace(plan::HEOMPlan;batch_columns::Integer=1)
    batch_columns isa Integer&&!(batch_columns isa Bool)&&batch_columns>0||
        throw(ArgumentError("HEOM batch_columns must be a positive integer"))
    BigInt(batch_columns)<=typemax(Int)||throw(ArgumentError(
        "HEOM batch_columns must be representable as an Int"))
    T=plan.Ttype
    system_work=_heom_system_workspace(plan.system)
    system_columns=Int(min(
        BigInt(heom_number_ados(plan))*BigInt(batch_columns),
        BigInt(_HEOM_SYSTEM_BATCH_WIDTH)))
    system_work isa LiouvillianWorkspace&&_ensure_batch_capacity!(
        system_work.batch,system_columns)
    HEOMWorkspace{T,typeof(plan),typeof(system_work)}(
        plan,system_work,zeros(T,plan.npi),zeros(T,plan.npi),
        zeros(T,plan.npi))
end

# Let generic matrix-free consumers (Floquet/response/control) request
# task-owned HEOM scratch instead of falling back to the synchronized
# compatibility callback retained by `heom_liouvillian`.
_linear_operator_workspace(plan::HEOMPlan)=HEOMWorkspace(plan)
_linear_operator_batch_workspace(
    plan::HEOMPlan,columns::Integer,::Type{T}) where T=
    HEOMWorkspace(plan;batch_columns=columns)
function _linear_operator_workspace(
        L::MatrixFreeLiouvillian{F,T,V,P}) where {F,T,V,P<:HEOMPlan}
    HEOMWorkspace(L.plan)
end
function _linear_operator_batch_workspace(
        L::MatrixFreeLiouvillian{F,T,V,P},columns::Integer,::Type{S}) where
        {F,T,V,P<:HEOMPlan,S}
    HEOMWorkspace(L.plan;batch_columns=columns)
end
_operator_trace_functional(plan::HEOMPlan)=plan.tracevec
_operator_has_adjoint(::HEOMPlan)=true

function _performance_heom_workspace_bytes(
        plan::HEOMPlan,batch_columns::Integer=1)
    batch_columns>0||throw(ArgumentError(
        "HEOM batch column count must be positive"))
    system_columns=Int(min(
        BigInt(heom_number_ados(plan))*BigInt(batch_columns),
        BigInt(_HEOM_SYSTEM_BATCH_WIDTH)))
    _performance_array_bytes(plan.npi,plan.Ttype,0;linear_arrays=3)+
        _performance_batched_operator_workspace_bytes(
            plan.system,system_columns)
end
_performance_linear_operator_workspace_bytes(plan::HEOMPlan;
    batch_columns::Integer=0)=_performance_heom_workspace_bytes(
        plan,batch_columns==0 ? 1 : batch_columns)
function _performance_linear_operator_workspace_bytes(
        L::MatrixFreeLiouvillian{F,T,V,P};
        batch_columns::Integer=0) where {F,T,V,P<:HEOMPlan}
    _performance_heom_workspace_bytes(
        L.plan,batch_columns==0 ? 1 : batch_columns)
end
_performance_source_action_bytes(plan::HEOMPlan,::Type{T}) where T=
    _performance_source_action_bytes(plan.system,T)
function _performance_source_action_bytes(
        L::MatrixFreeLiouvillian{F,T,V,P,W},::Type{S}) where
        {F,T,V,P<:HEOMPlan,W<:HEOMWorkspace,S}
    _performance_source_action_bytes(L.plan.system,S)
end

function apply!(destination::AbstractVector,
        L::MatrixFreeLiouvillian{F,T,V,P},source::AbstractVector,
        time,parameters,work::HEOMWorkspace) where {F,T,V,P<:HEOMPlan}
    work.plan===L.plan||throw(ArgumentError(
        "HEOM workspace belongs to a different matrix-free adapter"))
    apply!(destination,L.plan,source,time,parameters,work)
end

function apply!(destination::AbstractMatrix,
        L::MatrixFreeLiouvillian{F,T,V,P},source::AbstractMatrix,
        time,parameters,work::HEOMWorkspace) where {F,T,V,P<:HEOMPlan}
    size(source,1)==size(L,2)&&
        size(destination)==(size(L,1),size(source,2))||
        throw(DimensionMismatch("HEOM batch has incompatible dimensions"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "HEOM batch source and destination must not share storage"))
    work.plan===L.plan||throw(ArgumentError(
        "HEOM workspace belongs to a different matrix-free adapter"))
    apply!(destination,L.plan,source,time,parameters,work)
end

function _check_heom_workspace(work::HEOMWorkspace,plan::HEOMPlan)
    work.plan===plan||throw(ArgumentError(
        "HEOM workspace belongs to a different plan"))
    length(work.left)==plan.npi&&length(work.right)==plan.npi&&
        length(work.mixed)==plan.npi||
        throw(DimensionMismatch("HEOM workspace has incompatible dimensions"))
    eltype(work.left)===plan.Ttype||throw(ArgumentError(
        "HEOM workspace has an incompatible scalar type"))
    work
end

"""
    HEOMEvolutionWorkspace(plan)

Task-owned HEOM application scratch plus three full hierarchy vectors for the
low-storage classical RK4 path: a stage state, one derivative, and a weighted
stage accumulator. The legacy `k3` and `k4` fields remain empty compatibility
placeholders in newly constructed workspaces; a field-constructed legacy
workspace with full arrays remains accepted. Reuse the workspace for repeated
fixed-step evolution, but do not share it between concurrent tasks.
"""
struct HEOMEvolutionWorkspace{T,A}
    application::A
    temporary::Vector{T}
    k1::Vector{T}
    k2::Vector{T}
    k3::Vector{T}
    k4::Vector{T}
end

function HEOMEvolutionWorkspace(plan::HEOMPlan)
    T=plan.Ttype;n=size(plan,1)
    application=HEOMWorkspace(plan)
    HEOMEvolutionWorkspace{T,typeof(application)}(
        application,zeros(T,n),zeros(T,n),zeros(T,n),T[],T[])
end

function _check_heom_evolution_workspace(work::HEOMEvolutionWorkspace,
                                         plan::HEOMPlan)
    _check_heom_workspace(work.application,plan)
    n=size(plan,1)
    all(vector->length(vector)==n,
        (work.temporary,work.k1,work.k2))&&
        all(vector->length(vector) in (0,n),(work.k3,work.k4))||
        throw(DimensionMismatch("HEOM evolution workspace has incompatible dimensions"))
    eltype(work.temporary)===plan.Ttype||throw(ArgumentError(
        "HEOM evolution workspace has an incompatible scalar type"))
    work
end

function _check_heom_evolution_aliases(work::HEOMEvolutionWorkspace,
                                       destination)
    active=(work.temporary,work.k1,work.k2)
    for i in eachindex(active)
        Base.mightalias(active[i],destination)&&throw(ArgumentError(
            "HEOM evolution destination must not alias workspace scratch"))
        for j in 1:i-1
            Base.mightalias(active[i],active[j])&&throw(ArgumentError(
                "HEOM evolution workspace scratch arrays must not alias"))
        end
    end
    nothing
end

function _apply_heom_hierarchy_pulse_unchecked!(
        data,plan::HEOMPlan,pulse::PIUnitaryPulse,
        work::HEOMEvolutionWorkspace)
    intermediate=work.k1
    destination=work.temporary
    npi=plan.npi
    for ado in 1:heom_number_ados(plan)
        base=(ado-1)*npi
        for sector in eachindex(plan.basis.sectors)
            dimension=length(plan.basis.patterns[sector])
            first_coordinate=base+plan.basis.offsets[sector]
            range=first_coordinate:first_coordinate+dimension^2-1
            source_block=reshape(view(data,range),dimension,dimension)
            intermediate_block=reshape(
                view(intermediate,range),dimension,dimension)
            destination_block=reshape(
                view(destination,range),dimension,dimension)
            unitary=pulse.blocks[sector]
            mul!(intermediate_block,unitary,source_block)
            mul!(destination_block,intermediate_block,adjoint(unitary))
        end
    end
    copyto!(data,destination)
    data
end

function _apply_heom_hierarchy_pulse!(
        data::AbstractVector,plan::HEOMPlan,pulse::PIUnitaryPulse,
        work::HEOMEvolutionWorkspace)
    _check_hierarchy_pulse(pulse,plan.basis,plan.Ttype)
    _check_heom_evolution_workspace(work,plan)
    _check_heom_evolution_aliases(work,data)
    length(data)==size(plan,1)||throw(DimensionMismatch(
        "HEOM hierarchy pulse source has the wrong coordinate dimension"))
    _apply_heom_hierarchy_pulse_unchecked!(data,plan,pulse,work)
end

function _heom_system_apply_batch!(destination,system,source,time,parameters,work)
    if work===nothing
        if system isa LiouvillianPlan
            return apply!(destination,system,source,time,parameters)
        elseif system isa MatrixFreeLiouvillian||system isa CompiledPIModel||
                system isa SpecializedPIModel
            return apply!(destination,system,source,time,parameters)
        end
        return mul!(destination,system,source)
    end
    apply!(destination,system,source,time,parameters,work)
end

function _heom_apply_prepared_pi_chunks!(
        destination,plan::LiouvillianPlan,source,time,parameters,
        work::LiouvillianWorkspace,width::Int)
    n=length(plan.basis)
    size(source,1)==n&&size(destination)==size(source)||
        throw(DimensionMismatch(
            "HEOM system batch has incompatible PI dimensions"))
    _check_liouvillian_workspace(work,plan)
    _check_liouvillian_apply_types(destination,source,plan)
    if plan.kernels===nothing
        # Evaluate an allocating fallback schedule once for all ADO/RHS
        # columns.
        return mul!(destination,_matrix_at(plan.fallback_model,time,parameters),
                    source)
    end
    width=min(width,work.batch.capacity)
    width>0||throw(ArgumentError(
        "HEOM system workspace has zero batch capacity"))
    _prepare_kernels!(
        plan.kernels,work.kernel_workspaces,plan.basis,time,parameters)
    for first_column in 1:width:size(source,2)
        last_column=min(first_column+width-1,size(source,2))
        output=view(destination,:,first_column:last_column)
        input=view(source,:,first_column:last_column)
        fill!(output,zero(eltype(output)))
        _apply_batch_kernels!(
            output,input,plan.kernels,work.kernel_workspaces,
            plan.basis,time,parameters,work)
    end
    destination
end

function _heom_system_apply_hierarchy!(destination,system,source,time,
                                       parameters,work)
    columns=size(source,2)
    width=work===nothing ? max(columns,1) :
          min(max(columns,1),_HEOM_SYSTEM_BATCH_WIDTH)
    if columns<=width
        return _heom_system_apply_batch!(
            destination,system,source,time,parameters,work)
    end
    for first_column in 1:width:columns
        last_column=min(first_column+width-1,columns)
        _heom_system_apply_batch!(
            view(destination,:,first_column:last_column),system,
            view(source,:,first_column:last_column),time,parameters,work)
    end
    destination
end

function _heom_system_apply_hierarchy!(
        destination,system::LiouvillianPlan,source,time,parameters,
        work::LiouvillianWorkspace)
    width=min(max(size(source,2),1),_HEOM_SYSTEM_BATCH_WIDTH)
    _heom_apply_prepared_pi_chunks!(
        destination,system,source,time,parameters,work,width)
end

function _heom_system_apply_hierarchy!(
        destination,system::CompiledPIModel,source,time,parameters,
        work::LiouvillianWorkspace)
    system.backend===:matrixfree||return mul!(
        destination,system.operator,source)
    width=min(max(size(source,2),1),_HEOM_SYSTEM_BATCH_WIDTH)
    _heom_apply_prepared_pi_chunks!(
        destination,system.plan,source,time,parameters,work,width)
end

function _heom_system_apply_hierarchy!(
        destination,system::SpecializedPIModel,source,time,parameters,
        work::LiouvillianWorkspace)
    system.backend===:matrixfree||return mul!(
        destination,system.operator,source)
    width=min(max(size(source,2),1),_HEOM_SYSTEM_BATCH_WIDTH)
    _heom_apply_prepared_pi_chunks!(
        destination,system.plan,source,time,system.rates,work,width)
end

function _heom_system_apply_hierarchy!(
        destination,system::MatrixFreeLiouvillian,source,time,parameters,
        work::LiouvillianWorkspace)
    system.plan isa LiouvillianPlan||return apply!(
        destination,system,source,time,parameters)
    width=min(max(size(source,2),1),_HEOM_SYSTEM_BATCH_WIDTH)
    _heom_apply_prepared_pi_chunks!(
        destination,system.plan,source,time,parameters,work,width)
end

function _heom_coupling_actions!(left,right,plan::HEOMPlan,source,bath_number)
    blocks=plan.coupling_blocks[bath_number]
    for sector in eachindex(plan.basis.sectors)
        range=plan.basis.offsets[sector]:(plan.basis.offsets[sector+1]-1)
        dimension=length(plan.basis.patterns[sector])
        source_block=reshape(view(source,range),dimension,dimension)
        left_block=reshape(view(left,range),dimension,dimension)
        right_block=reshape(view(right,range),dimension,dimension)
        coupling=blocks[sector]
        mul!(left_block,coupling,source_block)
        mul!(right_block,source_block,coupling)
    end
    left,right
end

function _heom_residue_action!(destination,plan::HEOMPlan,source,
                               bath_number,work::HEOMWorkspace)
    delta=plan.residue_coefficients[bath_number]
    iszero(delta)&&return destination
    blocks=plan.coupling_blocks[bath_number]
    for sector in eachindex(plan.basis.sectors)
        range=plan.basis.offsets[sector]:(plan.basis.offsets[sector+1]-1)
        dimension=length(plan.basis.patterns[sector])
        source_block=reshape(view(source,range),dimension,dimension)
        destination_block=reshape(view(destination,range),dimension,dimension)
        left_block=reshape(view(work.left,range),dimension,dimension)
        right_block=reshape(view(work.right,range),dimension,dimension)
        mixed_block=reshape(view(work.mixed,range),dimension,dimension)
        coupling=blocks[sector]
        mul!(left_block,coupling,source_block)
        mul!(right_block,source_block,coupling)
        mul!(mixed_block,coupling,left_block)
        @. destination_block=destination_block-delta*mixed_block
        mul!(mixed_block,right_block,coupling)
        @. destination_block=destination_block-delta*mixed_block
        mul!(mixed_block,left_block,coupling)
        @. destination_block=destination_block+2delta*mixed_block
    end
    destination
end

@inline function _heom_apply_columns!(
        destination,plan::HEOMPlan,source,right_hand_sides::Int,
        time,parameters,work::HEOMWorkspace)
    npi=plan.npi
    number_ados=length(plan.multiindices)
    total_columns=number_ados*right_hand_sides
    length(source)==npi*total_columns&&length(destination)==length(source)||
        throw(DimensionMismatch("flattened HEOM storage has incompatible dimensions"))
    source_ados=reshape(source,npi,total_columns)
    destination_ados=reshape(destination,npi,total_columns)
    size(source_ados)==(npi,total_columns)&&
        size(destination_ados)==(npi,total_columns)||
        throw(DimensionMismatch("flattened HEOM columns have incompatible dimensions"))
    _heom_system_apply_hierarchy!(
        destination_ados,plan.system,source_ados,time,parameters,work.system)
    if plan.terminator===:residue
        for column in 1:total_columns
            for bath in eachindex(plan.residue_coefficients)
                _heom_residue_action!(
                    view(destination_ados,:,column),plan,
                    view(source_ados,:,column),bath,work)
            end
        end
    end
    for rhs in 1:right_hand_sides,ado in 1:number_ados
        column=ado+(rhs-1)*number_ados
        output=view(destination_ados,:,column)
        input=view(source_ados,:,column)
        decay=plan.decays[ado]
        if !iszero(decay)
            @inbounds @simd for coordinate in eachindex(output,input)
                output[coordinate]-=decay*input[coordinate]
            end
        end
    end

    topology=plan.topology
    for rhs in 1:right_hand_sides,source_ado in 1:number_ados
        source_column=source_ado+(rhs-1)*number_ados
        position=Int(topology.ado_edge_ptr[source_ado])
        stop=Int(topology.ado_edge_ptr[source_ado+1])-1
        while position<=stop
            edge=Int(topology.incident_edges[position])
            term=Int(topology.term[edge])
            bath=plan.exponent_baths[term]
            _heom_coupling_actions!(
                work.left,work.right,plan,
                view(source_ados,:,source_column),bath)
            while position<=stop
                edge=Int(topology.incident_edges[position])
                term=Int(topology.term[edge])
                plan.exponent_baths[term]==bath||break
                lower=Int(topology.lower[edge])
                upper=Int(topology.upper[edge])
                occupation=Int(topology.level[edge])
                if source_ado==lower
                    output=view(destination_ados,:,
                                upper+(rhs-1)*number_ados)
                    factor=plan.downward_level_factors[occupation,term]
                    coefficient=plan.coefficients[term]
                    right_coefficient=plan.right_coefficients[term]
                    @inbounds @simd for coordinate in eachindex(output)
                        output[coordinate]+=-im*factor*(
                            coefficient*work.left[coordinate]-
                            right_coefficient*work.right[coordinate])
                    end
                else
                    source_ado==upper||error(
                        "internal HEOM incidence mismatch")
                    output=view(destination_ados,:,
                                lower+(rhs-1)*number_ados)
                    factor=plan.upward_level_factors[occupation,term]
                    @inbounds @simd for coordinate in eachindex(output)
                        output[coordinate]+=-im*factor*(
                            work.left[coordinate]-work.right[coordinate])
                    end
                end
                position+=1
            end
        end
    end
    destination_ados
end

"""
    apply!(destination, plan::HEOMPlan, source, time, parameters, workspace)
    apply!(destination, plan::HEOMPlan, source, workspace)

Apply the truncated HEOM generator without constructing its matrix. The
explicit-time method supports a driven system Liouvillian. The shorter method
requires an autonomous system.
"""
function apply!(destination::AbstractVector,plan::HEOMPlan,
                source::AbstractVector,time,parameters,
                work::HEOMWorkspace)
    length(source)==size(plan,1)&&length(destination)==size(plan,1)||
        throw(DimensionMismatch("HEOM vector has the wrong length"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "HEOM generator source and destination must not share storage"))
    _check_heom_workspace(work,plan)
    promote_type(plan.Ttype,eltype(source))===plan.Ttype||throw(ArgumentError(
        "HEOM source scalar type is wider than the prepared plan"))
    promote_type(plan.Ttype,eltype(source),eltype(destination))===eltype(destination)||
        throw(ArgumentError("HEOM destination scalar type cannot represent the result"))
    _heom_apply_columns!(
        destination,plan,source,1,time,parameters,work)
    destination
end

function apply!(destination::AbstractMatrix,plan::HEOMPlan,
                source::AbstractMatrix,time,parameters,
                work::HEOMWorkspace)
    size(source,1)==size(plan,1)&&size(destination)==size(source)||
        throw(DimensionMismatch("HEOM matrix batch has incompatible dimensions"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "HEOM generator source and destination must not share storage"))
    _check_heom_workspace(work,plan)
    promote_type(plan.Ttype,eltype(source))===plan.Ttype||throw(ArgumentError(
        "HEOM source scalar type is wider than the prepared plan"))
    promote_type(plan.Ttype,eltype(source),eltype(destination))===
        eltype(destination)||throw(ArgumentError(
        "HEOM destination scalar type cannot represent the result"))
    _heom_apply_columns!(
        destination,plan,source,size(source,2),time,parameters,work)
    destination
end

function apply!(destination,plan::HEOMPlan,source,work::HEOMWorkspace)
    _require_autonomous(plan,"HEOM application")
    apply!(destination,plan,source,zero(_real_float_type(plan.Ttype)),nothing,work)
end

apply!(destination,plan::HEOMPlan,source,time,parameters)=
    apply!(destination,plan,source,time,parameters,HEOMWorkspace(plan))

function apply!(destination,plan::HEOMPlan,source)
    _require_autonomous(plan,"HEOM application")
    apply!(destination,plan,source,HEOMWorkspace(plan))
end

function _heom_system_adjoint_apply_batch!(destination,system,source,time,
                                           parameters,work)
    if work===nothing
        if system isa LiouvillianPlan||system isa MatrixFreeLiouvillian||
                system isa CompiledPIModel||system isa SpecializedPIModel
            return apply_adjoint!(destination,system,source,time,parameters)
        end
        return mul!(destination,adjoint(system),source)
    end
    apply_adjoint!(destination,system,source,time,parameters,work)
end

function _heom_system_adjoint_apply_hierarchy!(destination,system,source,time,
                                               parameters,work)
    columns=size(source,2)
    width=work===nothing ? max(columns,1) :
          min(max(columns,1),_HEOM_SYSTEM_BATCH_WIDTH)
    if columns<=width
        return _heom_system_adjoint_apply_batch!(
            destination,system,source,time,parameters,work)
    end
    for first_column in 1:width:columns
        last_column=min(first_column+width-1,columns)
        _heom_system_adjoint_apply_batch!(
            view(destination,:,first_column:last_column),system,
            view(source,:,first_column:last_column),time,parameters,work)
    end
    destination
end

function _heom_apply_prepared_pi_adjoint_chunks!(
        destination,plan::LiouvillianPlan,source,time,parameters,
        work::LiouvillianWorkspace,width::Int)
    n=length(plan.basis)
    size(source,1)==n&&size(destination)==size(source)||
        throw(DimensionMismatch(
            "adjoint HEOM system batch has incompatible PI dimensions"))
    _check_liouvillian_workspace(work,plan)
    _check_liouvillian_apply_types(destination,source,plan)
    if plan.kernels===nothing
        matrix=_matrix_at(plan.fallback_model,time,parameters)
        return mul!(destination,adjoint(matrix),source)
    end
    width=min(width,work.batch.capacity)
    width>0||throw(ArgumentError(
        "adjoint HEOM system workspace has zero batch capacity"))
    _prepare_kernels!(
        plan.kernels,work.kernel_workspaces,plan.basis,time,parameters)
    for first_column in 1:width:size(source,2)
        last_column=min(first_column+width-1,size(source,2))
        output=view(destination,:,first_column:last_column)
        input=view(source,:,first_column:last_column)
        fill!(output,zero(eltype(output)))
        _apply_adjoint_batch_kernels!(
            output,input,plan.kernels,work.kernel_workspaces,
            plan.basis,time,parameters,work)
    end
    destination
end

function _heom_system_adjoint_apply_hierarchy!(
        destination,system::LiouvillianPlan,source,time,parameters,
        work::LiouvillianWorkspace)
    width=min(max(size(source,2),1),_HEOM_SYSTEM_BATCH_WIDTH)
    _heom_apply_prepared_pi_adjoint_chunks!(
        destination,system,source,time,parameters,work,width)
end

function _heom_system_adjoint_apply_hierarchy!(
        destination,system::CompiledPIModel,source,time,parameters,
        work::LiouvillianWorkspace)
    system.backend===:matrixfree||return mul!(
        destination,adjoint(system.operator),source)
    width=min(max(size(source,2),1),_HEOM_SYSTEM_BATCH_WIDTH)
    _heom_apply_prepared_pi_adjoint_chunks!(
        destination,system.plan,source,time,parameters,work,width)
end

function _heom_system_adjoint_apply_hierarchy!(
        destination,system::SpecializedPIModel,source,time,parameters,
        work::LiouvillianWorkspace)
    system.backend===:matrixfree||return mul!(
        destination,adjoint(system.operator),source)
    width=min(max(size(source,2),1),_HEOM_SYSTEM_BATCH_WIDTH)
    _heom_apply_prepared_pi_adjoint_chunks!(
        destination,system.plan,source,time,system.rates,work,width)
end

function _heom_system_adjoint_apply_hierarchy!(
        destination,system::MatrixFreeLiouvillian,source,time,parameters,
        work::LiouvillianWorkspace)
    system.plan isa LiouvillianPlan||return apply_adjoint!(
        destination,system,source,time,parameters)
    width=min(max(size(source,2),1),_HEOM_SYSTEM_BATCH_WIDTH)
    _heom_apply_prepared_pi_adjoint_chunks!(
        destination,system.plan,source,time,parameters,work,width)
end

@inline function _heom_apply_adjoint_columns!(
        destination,plan::HEOMPlan,source,right_hand_sides::Int,
        time,parameters,work::HEOMWorkspace)
    npi=plan.npi
    number_ados=length(plan.multiindices)
    total_columns=number_ados*right_hand_sides
    length(source)==npi*total_columns&&length(destination)==length(source)||
        throw(DimensionMismatch(
            "flattened adjoint HEOM storage has incompatible dimensions"))
    source_ados=reshape(source,npi,total_columns)
    destination_ados=reshape(destination,npi,total_columns)
    size(source_ados)==(npi,total_columns)&&
        size(destination_ados)==(npi,total_columns)||
        throw(DimensionMismatch(
            "flattened adjoint HEOM columns have incompatible dimensions"))
    _heom_system_adjoint_apply_hierarchy!(
        destination_ados,plan.system,source_ados,time,parameters,work.system)
    if plan.terminator===:residue
        for column in 1:total_columns
            for bath in eachindex(plan.residue_coefficients)
                _heom_residue_action!(
                    view(destination_ados,:,column),plan,
                    view(source_ados,:,column),bath,work)
            end
        end
    end
    for rhs in 1:right_hand_sides,ado in 1:number_ados
        column=ado+(rhs-1)*number_ados
        output=view(destination_ados,:,column)
        input=view(source_ados,:,column)
        decay=conj(plan.decays[ado])
        if !iszero(decay)
            @inbounds @simd for coordinate in eachindex(output,input)
                output[coordinate]-=decay*input[coordinate]
            end
        end
    end
    topology=plan.topology
    for rhs in 1:right_hand_sides,source_ado in 1:number_ados
        source_column=source_ado+(rhs-1)*number_ados
        position=Int(topology.ado_edge_ptr[source_ado])
        stop=Int(topology.ado_edge_ptr[source_ado+1])-1
        while position<=stop
            edge=Int(topology.incident_edges[position])
            term=Int(topology.term[edge])
            bath=plan.exponent_baths[term]
            _heom_coupling_actions!(
                work.left,work.right,plan,
                view(source_ados,:,source_column),bath)
            while position<=stop
                edge=Int(topology.incident_edges[position])
                term=Int(topology.term[edge])
                plan.exponent_baths[term]==bath||break
                lower=Int(topology.lower[edge])
                upper=Int(topology.upper[edge])
                occupation=Int(topology.level[edge])
                if source_ado==lower
                    output=view(destination_ados,:,
                                upper+(rhs-1)*number_ados)
                    factor=plan.upward_level_factors[occupation,term]
                    @inbounds @simd for coordinate in eachindex(output)
                        output[coordinate]+=im*factor*(
                            work.left[coordinate]-work.right[coordinate])
                    end
                else
                    source_ado==upper||error(
                        "internal HEOM incidence mismatch")
                    output=view(destination_ados,:,
                                lower+(rhs-1)*number_ados)
                    factor=plan.downward_level_factors[occupation,term]
                    coefficient=conj(plan.coefficients[term])
                    right_coefficient=conj(plan.right_coefficients[term])
                    @inbounds @simd for coordinate in eachindex(output)
                        output[coordinate]+=im*factor*(
                            coefficient*work.left[coordinate]-
                            right_coefficient*work.right[coordinate])
                    end
                end
                position+=1
            end
        end
    end
    destination_ados
end

"""
    apply_adjoint!(destination, plan::HEOMPlan, source, time, parameters,
                   workspace)
    apply_adjoint!(destination, plan::HEOMPlan, source, workspace)

Apply the exact Hilbert--Schmidt adjoint of the prepared HEOM generator in its
stored (scaled or unscaled) coordinate basis. The explicit-time method
supports a driven system Liouvillian. Source and destination storage must not
overlap, and one [`HEOMWorkspace`](@ref) may be reused sequentially.
"""
function apply_adjoint!(destination::AbstractVector,plan::HEOMPlan,
                        source::AbstractVector,time,parameters,
                        work::HEOMWorkspace)
    length(source)==size(plan,1)&&length(destination)==size(plan,1)||
        throw(DimensionMismatch("HEOM vector has the wrong length"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "HEOM adjoint source and destination must not share storage"))
    _check_heom_workspace(work,plan)
    promote_type(plan.Ttype,eltype(source))===plan.Ttype||throw(ArgumentError(
        "HEOM adjoint source scalar type is wider than the prepared plan"))
    promote_type(plan.Ttype,eltype(source),eltype(destination))===
        eltype(destination)||throw(ArgumentError(
        "HEOM adjoint destination scalar type cannot represent the result"))
    _heom_apply_adjoint_columns!(
        destination,plan,source,1,time,parameters,work)
    destination
end

function apply_adjoint!(destination::AbstractMatrix,plan::HEOMPlan,
                        source::AbstractMatrix,time,parameters,
                        work::HEOMWorkspace)
    size(source,1)==size(plan,1)&&size(destination)==size(source)||
        throw(DimensionMismatch(
            "adjoint HEOM matrix batch has incompatible dimensions"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "HEOM adjoint source and destination must not share storage"))
    _check_heom_workspace(work,plan)
    promote_type(plan.Ttype,eltype(source))===plan.Ttype||throw(ArgumentError(
        "HEOM adjoint source scalar type is wider than the prepared plan"))
    promote_type(plan.Ttype,eltype(source),eltype(destination))===
        eltype(destination)||throw(ArgumentError(
        "HEOM adjoint destination scalar type cannot represent the result"))
    _heom_apply_adjoint_columns!(
        destination,plan,source,size(source,2),time,parameters,work)
    destination
end

function apply_adjoint!(destination,plan::HEOMPlan,source,
                        work::HEOMWorkspace)
    _require_autonomous(plan,"HEOM adjoint application")
    apply_adjoint!(destination,plan,source,
                   zero(_real_float_type(plan.Ttype)),nothing,work)
end

apply_adjoint!(destination,plan::HEOMPlan,source,time,parameters)=
    apply_adjoint!(destination,plan,source,time,parameters,HEOMWorkspace(plan))

function apply_adjoint!(destination,plan::HEOMPlan,source)
    _require_autonomous(plan,"HEOM adjoint application")
    apply_adjoint!(destination,plan,source,HEOMWorkspace(plan))
end

function apply_adjoint!(destination::AbstractVector,
        L::MatrixFreeLiouvillian{F,T,V,P},source::AbstractVector,
        time,parameters,work::HEOMWorkspace) where {F,T,V,P<:HEOMPlan}
    work.plan===L.plan||throw(ArgumentError(
        "HEOM workspace belongs to a different matrix-free adapter"))
    apply_adjoint!(destination,L.plan,source,time,parameters,work)
end

function apply_adjoint!(destination::AbstractMatrix,
        L::MatrixFreeLiouvillian{F,T,V,P},source::AbstractMatrix,
        time,parameters,work::HEOMWorkspace) where {F,T,V,P<:HEOMPlan}
    size(source,1)==size(L,1)&&
        size(destination)==(size(L,2),size(source,2))||
        throw(DimensionMismatch(
            "adjoint HEOM batch has incompatible dimensions"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "adjoint HEOM batch source and destination must not share storage"))
    work.plan===L.plan||throw(ArgumentError(
        "HEOM workspace belongs to a different matrix-free adapter"))
    apply_adjoint!(destination,L.plan,source,time,parameters,work)
end

"""
    HEOMState(plan, data)

State of a prepared hierarchy in ADO-major order. Each ADO stores one complete
PI coefficient vector in `plan.scaling`; [`heom_ado`](@ref) converts selected
auxiliaries back to conventional unscaled coordinates. Construction copies
`data` and rejects scalar narrowing; it does not normalize or repair the
reduced state.
"""
struct HEOMState{T<:Number,P<:HEOMPlan}
    plan::P
    data::Vector{T}
    function HEOMState{T,P}(plan::P,data::Vector{T},::Val{:owned}) where
            {T<:Number,P<:HEOMPlan}
        new{T,P}(plan,data)
    end
end

function HEOMState(plan::HEOMPlan,data::AbstractVector)
    length(data)==size(plan,1)||throw(DimensionMismatch(
        "HEOM state vector has the wrong length"))
    promote_type(plan.Ttype,eltype(data))===plan.Ttype||throw(ArgumentError(
        "HEOM state data would be narrowed to the plan scalar type"))
    values=Vector{plan.Ttype}(undef,length(data))
    for index in eachindex(data)
        values[index]=_heom_checked_convert(
            plan.Ttype,data[index],"HEOM state entry $index")
    end
    HEOMState{plan.Ttype,typeof(plan)}(plan,values,Val(:owned))
end

Base.copy(state::HEOMState)=HEOMState(state.plan,state.data)
Base.length(state::HEOMState)=length(state.data)
Base.eltype(state::HEOMState)=eltype(state.data)

function show(io::IO,state::HEOMState)
    root_trace=dot(state.plan.tracevec,state.data)
    print(io,"HEOMState(ADOs=$(heom_number_ados(state.plan)), " *
             "dimension=$(length(state.data)), root_trace=$root_trace)")
end

"""
    apply_hierarchy_pulse!(state, pulse, workspace)

Apply a prepared instantaneous system unitary to every ADO of `state`,
in place, as `rho_n -> U*rho_n*U'`. The hierarchy scaling and every
system--bath memory auxiliary are retained. `workspace` must be a task-owned
[`HEOMEvolutionWorkspace`](@ref) for the same plan.
"""
function apply_hierarchy_pulse!(
        state::HEOMState,pulse::PIUnitaryPulse,
        workspace::HEOMEvolutionWorkspace)
    state.plan===workspace.application.plan||throw(ArgumentError(
        "HEOM state and pulse workspace use different plans"))
    _apply_heom_hierarchy_pulse!(
        state.data,state.plan,pulse,workspace)
    state
end

"""
    heom_initial_state(plan, rho)

Construct the standard factorized HEOM initial condition: the root ADO is
`rho` and every auxiliary ADO is exactly zero.
"""
function heom_initial_state(plan::HEOMPlan,rho::PIState)
    rho.basis===plan.basis||throw(ArgumentError(
        "the initial PI state uses a different basis"))
    promote_type(plan.Ttype,eltype(rho.data))===plan.Ttype||throw(ArgumentError(
        "the initial PI state would be narrowed to the HEOM scalar type"))
    values=zeros(plan.Ttype,size(plan,1))
    values[1:plan.npi].=rho.data
    HEOMState(plan,values)
end

function _heom_validate_inverse_temperature(plan::HEOMPlan,beta)
    R=_real_float_type(plan.Ttype)
    betaR=_heom_checked_time(R,beta,"inverse_temperature")
    betaR>0||throw(ArgumentError(
        "inverse_temperature must be strictly positive"))
    for (bath,metadata) in pairs(plan.bath_metadata)
        haskey(metadata,:inverse_temperature)||continue
        bath_beta=metadata.inverse_temperature
        bath_beta===nothing&&continue
        converted=_heom_checked_time(R,bath_beta,
            "bath $bath inverse_temperature metadata")
        converted==betaR||throw(ArgumentError(
            "bath $bath was prepared at inverse_temperature=$converted, not $betaR"))
    end
    betaR
end

"""
    heom_thermal_state(plan, Hamiltonian, inverse_temperature;
                       preparation=:factorized,
                       relaxation_time=nothing, steps=256,
                       parameters=nothing, return_info=false, kwargs...)

Prepare a hierarchy from the bare-system Gibbs state. With
`preparation=:factorized`, only the root ADO is populated. The explicit
`preparation=:relaxation` route evolves that hierarchy for `relaxation_time`
and therefore creates correlated auxiliary ADOs. `preparation=:stationary`
uses the same Gibbs hierarchy only as the initial guess of
[`heom_steady_state`](@ref); `kwargs` are forwarded to that solver.

The latter two routes are real-time relaxation/stationary preparations, not
imaginary-time HEOM. A stationary result represents thermal equilibrium only
when the supplied system generator and physical bath decomposition satisfy
the appropriate equilibrium assumptions. Known physical-bath temperature
metadata is checked exactly, but the library cannot certify detailed balance
of an arbitrary `system` Liouvillian.
"""
function heom_thermal_state(plan::HEOMPlan,Hamiltonian::PIOperator,
                            inverse_temperature;
                            preparation::Symbol=:factorized,
                            relaxation_time=nothing,steps::Integer=256,
                            parameters=nothing,return_info::Bool=false,
                            kwargs...)
    Hamiltonian.basis===plan.basis||throw(ArgumentError(
        "the thermal Hamiltonian uses a different PI basis"))
    beta=_heom_validate_inverse_temperature(plan,inverse_temperature)
    rho=thermal_state(Hamiltonian,beta)
    factorized=heom_initial_state(plan,rho)
    if preparation===:factorized
        relaxation_time===nothing||throw(ArgumentError(
            "relaxation_time applies only to preparation=:relaxation"))
        isempty(kwargs)||throw(ArgumentError(
            "solver keywords apply only to preparation=:stationary"))
        return_info&&throw(ArgumentError(
            "return_info applies only to preparation=:stationary"))
        return factorized
    elseif preparation===:relaxation
        relaxation_time===nothing&&throw(ArgumentError(
            "preparation=:relaxation requires relaxation_time"))
        isempty(kwargs)||throw(ArgumentError(
            "solver keywords apply only to preparation=:stationary"))
        return_info&&throw(ArgumentError(
            "return_info applies only to preparation=:stationary"))
        R=_real_float_type(plan.Ttype)
        duration=_heom_checked_time(R,relaxation_time,"relaxation_time")
        duration>0||throw(ArgumentError(
            "relaxation_time must be strictly positive"))
        return heom_evolve(plan,factorized,(zero(R),duration);
                           steps,parameters)
    elseif preparation===:stationary
        relaxation_time===nothing||throw(ArgumentError(
            "relaxation_time applies only to preparation=:relaxation"))
        parameters===nothing||throw(ArgumentError(
            "preparation=:stationary does not accept dynamics parameters; freeze a driven system first"))
        return heom_steady_state(plan;initial_state=factorized,return_info,
                                 kwargs...)
    end
    throw(ArgumentError(
        "preparation must be :factorized, :relaxation, or :stationary"))
end

function _heom_ado_index(plan::HEOMPlan,label::Integer)
    1<=label<=heom_number_ados(plan)||throw(BoundsError(plan.multiindices,label))
    Int(label)
end

function _heom_ado_index(plan::HEOMPlan,label)
    values=collect(label)
    length(values)==length(plan.coefficients)||throw(DimensionMismatch(
        "HEOM multi-index has the wrong length"))
    all(value->value isa Integer&&value>=0,values)||throw(ArgumentError(
        "HEOM occupations must be nonnegative integers"))
    index=get(plan.index,Tuple(Int.(values)),0)
    iszero(index)&&throw(ArgumentError(
        "HEOM multi-index is outside the retained hierarchy"))
    index
end

"""
    heom_ado(state, label)

Return a detached `PIOperator` holding the ADO selected by its one-based
hierarchy position or occupation-vector `label`. Auxiliary ADOs are not
density matrices and are therefore deliberately returned as operators.
"""
function heom_ado(state::HEOMState,label)
    index=_heom_ado_index(state.plan,label);npi=state.plan.npi
    range=(index-1)*npi+1:index*npi
    R=_real_float_type(eltype(state.data))
    scale=state.plan.ado_scales[index]
    PIOperator(state.plan.basis,
               Complex{R}.(scale .* view(state.data,range)))
end

"""
    heom_reduced_state(state)

Return a detached `PIState` containing the physical reduced system density
operator, i.e. the root ADO. No normalization or positivity repair is made.
"""
function heom_reduced_state(state::HEOMState)
    R=_real_float_type(eltype(state.data))
    PIState(state.plan.basis,Complex{R}.(view(state.data,1:state.plan.npi)))
end

function _heom_checked_time(::Type{R},time,description) where R
    converted=try
        R(time)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("$description is not representable in $R"))
    end
    isfinite(converted)||throw(ArgumentError("$description must be finite"))
    converted==time||throw(ArgumentError(
        "$description is not exactly representable in $R; prepare the HEOM plan at wider precision"))
    converted
end

function _heom_rk4_step!(destination,plan::HEOMPlan,time,endpoint,
                         parameters,work::HEOMEvolutionWorkspace)
    step=endpoint-time
    midpoint=time+step/2
    apply!(work.k1,plan,destination,time,parameters,work.application)
    copyto!(work.k2,work.k1)
    @. work.temporary=destination+(step/2)*work.k1
    apply!(work.k1,plan,work.temporary,midpoint,parameters,work.application)
    @. work.k2=work.k2+2work.k1
    @. work.temporary=destination+(step/2)*work.k1
    apply!(work.k1,plan,work.temporary,midpoint,parameters,work.application)
    @. work.k2=work.k2+2work.k1
    @. work.temporary=destination+step*work.k1
    apply!(work.k1,plan,work.temporary,endpoint,parameters,work.application)
    @. work.k2=work.k2+work.k1
    @. destination=destination+(step/6)*work.k2
    destination
end

"""
    heom_evolve!(destination, plan, source, tspan;
                 steps=256, parameters=nothing, workspace=nothing,
                 pulses=nothing)

Propagate a hierarchy with preallocated, three-scratch fixed-step RK4.
`destination` may alias `source`, but it must not alias workspace scratch; one
`HEOMEvolutionWorkspace` may be reused sequentially. Increase `steps` and
`plan.max_depth` independently to check integration and hierarchy truncation
errors. A [`HierarchyPulseSequence`](@ref) passed as `pulses` splits RK4
steps exactly at every event in `(tspan[1], tspan[2]]`; events at the final
time are applied before return.
"""
function heom_evolve!(destination::AbstractVector,plan::HEOMPlan,
                      source::AbstractVector,tspan;steps::Integer=256,
                      parameters=nothing,workspace=nothing,pulses=nothing)
    length(source)==size(plan,1)&&length(destination)==size(plan,1)||
        throw(DimensionMismatch("HEOM state vector has the wrong length"))
    steps>0||throw(ArgumentError("steps must be positive"))
    BigInt(steps)<=typemax(Int)||throw(ArgumentError(
        "steps must be representable as an Int"))
    promote_type(plan.Ttype,eltype(source))===plan.Ttype||throw(ArgumentError(
        "HEOM evolution source scalar type is wider than the prepared plan"))
    promote_type(plan.Ttype,eltype(source),eltype(destination))===
        eltype(destination)||throw(ArgumentError(
        "HEOM evolution destination scalar type cannot represent the result without narrowing"))
    destination===source||!Base.mightalias(destination,source)||
        throw(ArgumentError(
            "HEOM evolution permits exact in-place use but not partially overlapping source and destination storage"))
    length(tspan)==2||throw(ArgumentError("tspan must contain exactly two times"))
    R=_real_float_type(plan.Ttype)
    t0=_heom_checked_time(R,first(tspan),"initial time")
    t1=_heom_checked_time(R,last(tspan),"final time")
    sequence=if pulses===nothing
        nothing
    else
        pulses isa HierarchyPulseSequence||throw(ArgumentError(
            "pulses must be a HierarchyPulseSequence or nothing"))
        checked=_check_hierarchy_pulse_sequence(
            pulses,plan.basis,plan.Ttype)
        _hierarchy_pulse_event_range(checked,t0,t1)
        checked
    end
    step_count=Int(steps)
    step_count_R=_heom_checked_time(R,step_count,"step count")
    interval=t1-t0
    isfinite(interval)||throw(ArgumentError(
        "HEOM evolution time interval is not finite in $R"))
    step=interval/step_count_R
    !iszero(interval)&&iszero(step)&&throw(ArgumentError(
        "HEOM evolution step underflows in $R; use fewer steps or wider precision"))
    work=workspace===nothing ? HEOMEvolutionWorkspace(plan) :
                              _check_heom_evolution_workspace(workspace,plan)
    _check_heom_evolution_aliases(work,destination)
    destination===source||copyto!(destination,source)
    events=sequence===nothing ? nothing :
        _hierarchy_pulse_event_range(sequence,t0,t1)
    if events===nothing||isempty(events)
        # Preserve the historical arithmetic order exactly when no pulse is
        # active in this span. In particular, the nominal `step` is reused at
        # the final RK stage instead of recomputing it from the endpoint.
        for step_index in 0:step_count-1
            time=t0+R(step_index)*step
            midpoint=time+step/2
            endpoint=step_index==step_count-1 ? t1 : time+step
            apply!(work.k1,plan,destination,time,parameters,work.application)
            copyto!(work.k2,work.k1)
            @. work.temporary=destination+(step/2)*work.k1
            apply!(work.k1,plan,work.temporary,midpoint,parameters,work.application)
            @. work.k2=work.k2+2work.k1
            @. work.temporary=destination+(step/2)*work.k1
            apply!(work.k1,plan,work.temporary,midpoint,parameters,work.application)
            @. work.k2=work.k2+2work.k1
            @. work.temporary=destination+step*work.k1
            apply!(work.k1,plan,work.temporary,endpoint,parameters,work.application)
            @. work.k2=work.k2+work.k1
            @. destination=destination+(step/6)*work.k2
        end
        return destination
    end
    event_index=first(events)
    last_event=last(events)
    for step_index in 0:step_count-1
        time=t0+R(step_index)*step
        endpoint=step_index==step_count-1 ? t1 : time+step
        segment_start=time
        while event_index<=last_event&&
                sequence.times[event_index]<=endpoint
            event_time=sequence.times[event_index]
            event_time>segment_start&&_heom_rk4_step!(
                destination,plan,segment_start,event_time,parameters,work)
            _apply_heom_hierarchy_pulse_unchecked!(
                destination,plan,sequence.pulses[event_index],work)
            segment_start=event_time
            event_index+=1
        end
        segment_start<endpoint&&_heom_rk4_step!(
            destination,plan,segment_start,endpoint,parameters,work)
    end
    destination
end

function heom_evolve!(destination::HEOMState,plan::HEOMPlan,
                      source::HEOMState,tspan;kwargs...)
    destination.plan===plan&&source.plan===plan||throw(ArgumentError(
        "HEOM states and plan are incompatible"))
    heom_evolve!(destination.data,plan,source.data,tspan;kwargs...)
    destination
end

"""Allocating convenience wrapper for [`heom_evolve!`](@ref)."""
function heom_evolve(plan::HEOMPlan,source::HEOMState,tspan;kwargs...)
    source.plan===plan||throw(ArgumentError("HEOM state and plan are incompatible"))
    destination=copy(source)
    heom_evolve!(destination,plan,source,tspan;kwargs...)
end

function heom_evolve(plan::HEOMPlan,source::PIState,tspan;kwargs...)
    initial=heom_initial_state(plan,source)
    heom_evolve(plan,initial,tspan;kwargs...)
end

"""
    heom_time_evolution(plan, initial, times;
                        steps_per_interval=64, parameters=nothing,
                        pulses=nothing)

Return saved hierarchy states at ordered `times`, reusing one RK4 workspace.
`initial` may be a `PIState` (factorized hierarchy) or an `HEOMState`.
Pulse-time states follow the post-pulse convention of
[`HierarchyPulseSequence`](@ref).
"""
function heom_time_evolution(plan::HEOMPlan,initial,times;
                             steps_per_interval::Integer=64,parameters=nothing,
                             pulses=nothing)
    steps_per_interval>0||throw(ArgumentError(
        "steps_per_interval must be positive"))
    state=initial isa PIState ? heom_initial_state(plan,initial) : copy(initial)
    state isa HEOMState&&state.plan===plan||throw(ArgumentError(
        "initial must be a compatible PIState or HEOMState"))
    raw_values=collect(times)
    isempty(raw_values)&&return typeof(state)[]
    R=_real_float_type(plan.Ttype)
    values=R[_heom_checked_time(R,value,"time $index")
             for (index,value) in pairs(raw_values)]
    all(diff(values).>=zero(R))||throw(ArgumentError(
        "times must be nondecreasing"))
    workspace=HEOMEvolutionWorkspace(plan);output=typeof(state)[copy(state)]
    for index in 2:length(values)
        values[index]==values[index-1]||heom_evolve!(
            state,plan,state,(values[index-1],values[index]);
            steps=steps_per_interval,parameters,workspace,pulses)
        push!(output,copy(state))
    end
    output
end

"""
    heom_problem(plan, initial, tspan; parameters=nothing)

Construct an in-place `SciMLBase.ODEProblem` for a prepared HEOM hierarchy.
`initial` may be a compatible [`PIState`](@ref), which creates the factorized
root-only hierarchy, or a complete [`HEOMState`](@ref). The problem owns one
[`HEOMWorkspace`](@ref), copies its initial coordinate vector, and supports a
driven system Liouvillian through SciML's time and parameter arguments.

The state vectors use `plan.scaling`; use [`heom_reduced_state`](@ref) on an
`HEOMState(plan, solution.u[i])` to recover the physical reduced density
operator. Construct a separate problem for every concurrent solve because its
captured workspace is mutable. Both time-span endpoints must be exactly
representable in the plan's real scalar type; the returned problem stores
that precision rather than silently widening a fixed-precision plan.
"""
function heom_problem(plan::HEOMPlan,initial,tspan;parameters=nothing)
    state=if initial isa PIState
        heom_initial_state(plan,initial)
    elseif initial isa HEOMState
        initial.plan===plan||throw(ArgumentError(
            "initial HEOM state belongs to a different plan"))
        initial
    else
        throw(ArgumentError(
            "initial must be a compatible PIState or HEOMState"))
    end
    length(tspan)==2||throw(ArgumentError(
        "tspan must contain exactly two times"))
    R=_real_float_type(plan.Ttype)
    prepared_tspan=(
        _heom_checked_time(R,first(tspan),"initial time"),
        _heom_checked_time(R,last(tspan),"final time"))
    work=HEOMWorkspace(plan)
    f! = (du,u,p,t)->apply!(du,plan,u,t,p,work)
    SciMLBase.ODEProblem(f!,copy(state.data),prepared_tspan,parameters)
end

function _heom_prefix_plan(template::HEOMPlan,depth::Int)
    depth==template.max_depth&&return template
    0<=depth<template.max_depth||throw(ArgumentError(
        "prefix depth must lie below the template depth"))
    K=length(template.coefficients)
    count=something(findlast(index->sum(index)<=depth,
                             template.multiindices),0)
    count>0||error("internal HEOM prefix hierarchy lost its root ADO")
    multiindices=map(copy,view(template.multiindices,1:count))
    lookup=Dict{Tuple,Int}(Tuple(index)=>position
                           for (position,index) in pairs(multiindices))
    topology=_heom_packed_topology(
        multiindices,lookup,template.exponent_baths,depth)
    ado_scales=copy(view(template.ado_scales,1:count))
    ado_importances=isempty(template.ado_importances) ?
        eltype(template.ado_scales)[] :
        copy(view(template.ado_importances,1:count))
    decays=copy(view(template.decays,1:count))
    full_ado_count=exact_binomial(BigInt(K)+BigInt(depth),BigInt(depth))
    tracevec=_resize_trace_functional(
        template.tracevec,template.npi*count)
    HEOMPlan{typeof(template.basis),typeof(template.system),
             typeof(template.coupling_blocks),typeof(template.exponent_baths),
             typeof(multiindices),typeof(lookup),typeof(topology),
             typeof(template.coefficients),eltype(ado_scales),
             typeof(tracevec),template.Ttype}(
        template.basis,template.system,template.coupling_blocks,
        template.exponent_baths,template.coefficients,
        template.right_coefficients,template.frequencies,
        multiindices,lookup,topology,template.upward_level_factors,
        template.downward_level_factors,
        ado_scales,template.pole_scales,template.residue_coefficients,
        template.bath_metadata,ado_importances,decays,full_ado_count,
        depth,template.npi,
        tracevec,template.Ttype,template.autonomous,template.terminator,
        template.scaling,template.importance_cutoff,
        template.importance_metric)
end

function _validated_heom_depths(depths)
    values=collect(depths)
    length(values)>=2||throw(ArgumentError(
        "depths must contain at least two values"))
    all(value->value isa Integer&&value>=0,values)||throw(ArgumentError(
        "depths must contain nonnegative integers"))
    all(diff(values).>0)||throw(ArgumentError(
        "depths must be strictly increasing"))
    int_depths=Int[]
    for value in values
        BigInt(value)<=typemax(Int)||throw(ArgumentError(
            "every hierarchy depth must be representable as an Int"))
        push!(int_depths,Int(value))
    end
    int_depths
end

function _heom_depth_convergence(template::HEOMPlan,initial::PIState,tspan,
        int_depths;steps::Integer,parameters,atol::Real,rtol,
        consecutive::Integer,require_convergence::Bool)
    steps>0||throw(ArgumentError("steps must be positive"))
    length(tspan)==2||throw(ArgumentError(
        "tspan must contain exactly two times"))
    isfinite(atol)&&atol>=0||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    rtol===nothing||rtol isa Real&&isfinite(rtol)&&rtol>=0||
        throw(ArgumentError("rtol must be nothing or a finite nonnegative real"))
    consecutive>0||throw(ArgumentError("consecutive must be positive"))
    last(int_depths)<=template.max_depth||throw(ArgumentError(
        "requested depth $(last(int_depths)) exceeds template.max_depth=$(template.max_depth)"))
    initial.basis===template.basis||throw(ArgumentError(
        "the initial PI state uses a different basis"))
    R=_real_float_type(template.Ttype)
    atolR=R(atol)
    isfinite(atolR)||throw(ArgumentError(
        "atol is not finite in $R"))
    !iszero(atol)&&iszero(atolR)&&throw(ArgumentError(
        "atol underflows in $R; use a wider prepared scalar type"))
    if rtol!==nothing
        rtolR=R(rtol)
        isfinite(rtolR)||throw(ArgumentError("rtol is not finite in $R"))
        !iszero(rtol)&&iszero(rtolR)&&throw(ArgumentError(
            "rtol underflows in $R; use a wider prepared scalar type"))
    end
    initial_trace=template.Ttype(trace(initial))
    _heom_isfinite(initial_trace)||throw(ArgumentError(
        "the initial PI state has a nonfinite trace"))
    function evaluate_depth(depth)
        plan=_heom_prefix_plan(template,depth)
        started=time_ns()
        hierarchy=heom_evolve(plan,initial,tspan;
                              steps,parameters)
        elapsed_seconds=(time_ns()-started)/1e9
        all(_heom_isfinite,hierarchy.data)||throw(ArgumentError(
            "HEOM evolution produced a nonfinite hierarchy at depth $depth; " *
            "refine the time step and inspect the bath decomposition"))
        reduced=heom_reduced_state(hierarchy)
        trace_error=R(abs(trace(reduced)-initial_trace))
        retained_hierarchy=depth==last(int_depths) ? hierarchy : nothing
        (state=reduced,depth,ado_count=heom_number_ados(plan),
         dimension=size(plan,1),trace_error,elapsed_seconds,
         hierarchy=retained_hierarchy,system_prepared_once=true,
         coupling_blocks_shared=true,initial_trace,
         template_max_depth=template.max_depth,terminator=template.terminator,
         importance_cutoff=template.importance_cutoff,
         full_ado_count=plan.full_ado_count)
    end
    hierarchy_depth_convergence(evaluate_depth,int_depths;
        estimate=result->result.state,
        diagnostics=identity,atol=atolR,rtol,consecutive,
        require_convergence)
end


"""
    heom_depth_convergence(system, baths, initial, tspan;
                           depths, steps=256, parameters=nothing,
                           scaling=:unscaled, scaling_factors=nothing,
                           atol=0, rtol=nothing, consecutive=2,
                           require_convergence=false)
    heom_depth_convergence(template, initial, tspan; depths, ...)

Propagate the same factorized initial PI state at a strictly increasing set of
hierarchy `depths` and compare successive final reduced states in the exact
Hilbert--Schmidt norm of the PI coefficient basis. A pair converges when

```math
\\lVert\\rho_D-\\rho_{D'}\\rVert_2\\leq
\\mathrm{atol}+\\mathrm{rtol}\\max(\\lVert\\rho_D\\rVert_2,
                                 \\lVert\\rho_{D'}\\rVert_2).
```

The system is compiled once and all depths share the prepared system and
coupling blocks; each evolution still owns dimension-matched scratch. An
existing `HEOMPlan` may instead be passed as `template`, in which case every
requested depth must not exceed `template.max_depth`.

The return value is the package-wide [`ConvergenceStudyResult`](@ref).
Intermediate raw results contain only a detached reduced `state` and plain
diagnostics; only `last(report.results).hierarchy` retains a complete
hierarchy. The shared convergence rule requires the final `consecutive`
successive comparisons to pass (two by default). Set
`require_convergence=true` to raise otherwise.

This report isolates hierarchy truncation only: every depth uses the same
fixed RK4 `steps`, so time-step convergence must be checked separately. No
state is normalized, clipped, or symmetrized for the comparison.
"""
function heom_depth_convergence(source,baths,initial::PIState,tspan;
        depths,steps::Integer=256,parameters=nothing,
        scaling=:unscaled,scaling_factors=nothing,
        terminator::Symbol=:none,importance_cutoff::Real=0,
        importance_metric::Symbol=:normalized_coupling_decay,
        atol::Real=0,rtol=nothing,consecutive::Integer=2,
        require_convergence::Bool=false)
    int_depths=_validated_heom_depths(depths)
    prepared=_prepare_heom_system(source)
    template=HEOMPlan(prepared,baths;max_depth=last(int_depths),
                      basis=initial.basis,scaling,scaling_factors,terminator,
                      importance_cutoff,importance_metric)
    _heom_depth_convergence(template,initial,tspan,int_depths;
        steps,parameters,atol,rtol,consecutive,require_convergence)
end

function heom_depth_convergence(template::HEOMPlan,initial::PIState,tspan;
        depths,steps::Integer=256,parameters=nothing,
        atol::Real=0,rtol=nothing,consecutive::Integer=2,
        require_convergence::Bool=false)
    int_depths=_validated_heom_depths(depths)
    _heom_depth_convergence(template,initial,tspan,int_depths;
        steps,parameters,atol,rtol,consecutive,require_convergence)
end

"""
    heom_liouvillian(plan)

Return a synchronized `MatrixFreeLiouvillian` adapter for an HEOM plan. Its
trace functional acts only on the root ADO, enabling use of the package's
existing matrix-free Krylov routines without constructing the hierarchy
matrix. The adapter publishes synchronized vector and matrix-RHS callbacks
for both forward and adjoint application. In parallel hot loops, call
`apply!` with one explicit `HEOMWorkspace` per task instead.
"""
function heom_liouvillian(plan::HEOMPlan)
    workspace=HEOMWorkspace(plan)
    action! = (destination,source,time,parameters)->
        apply!(destination,plan,source,time,parameters,workspace)
    adjoint_action! = (destination,source,time,parameters)->
        apply_adjoint!(destination,plan,source,time,parameters,workspace)
    batched_action! = (destination,source,time,parameters)->
        apply!(destination,plan,source,time,parameters,workspace)
    batched_adjoint_action! = (destination,source,time,parameters)->
        apply_adjoint!(destination,plan,source,time,parameters,workspace)
    MatrixFreeLiouvillian(size(plan,1),action!,plan.Ttype,copy(plan.tracevec);
                          autonomous=plan.autonomous,plan,workspace,
                          adjoint_action!,batched_action!,
                          batched_adjoint_action!)
end

"""
    HEOMBlockPreconditioner

Reusable guarded Schur-shift or LU factors for the ADO-diagonal blocks of the
normalized, trace-fixed HEOM stationary operator. LAPACK complex precisions
retain one Schur form and compact per-ADO shifts; ill-conditioned shifts and
unsupported scalar types use exact block LU factors. Construct one with
[`heom_block_preconditioner`](@ref) and apply it through `ldiv!`.
"""
struct HEOMBlockPreconditioner{T,F,I,Z,U,R,S,M}
    factors::Vector{F}
    factor_indices::Vector{I}
    shifts::Vector{T}
    schur_vectors::Z
    schur_triangular::U
    schur_rhs::Vector{T}
    schur_solution::Vector{T}
    schur_lock::ReentrantLock
    n::Int
    npi::Int
    regularization::T
    residual_tolerance::R
    operator_scale::S
    metadata::M
end

size(P::HEOMBlockPreconditioner)=(P.n,P.n)
size(P::HEOMBlockPreconditioner,index::Integer)=
    index in (1,2) ? P.n : 1
eltype(::HEOMBlockPreconditioner{T}) where T=T

"""Return setup, storage, and application-cost metadata for `P`."""
preconditioner_cost(P::HEOMBlockPreconditioner)=P.metadata
_preconditioner_operator_scale(P::HEOMBlockPreconditioner)=P.operator_scale
_preconditioner_metadata(P::HEOMBlockPreconditioner)=P.metadata

function _heom_shifted_schur_solve!(destination,P::HEOMBlockPreconditioner,
                                    source,shift)
    Z=P.schur_vectors;U=P.schur_triangular
    Z===nothing&&error("internal HEOM Schur-preconditioner mismatch")
    mul!(P.schur_rhs,adjoint(Z),source)
    rhs_norm=norm(P.schur_rhs)
    copyto!(P.schur_solution,P.schur_rhs)
    n=P.npi
    @inbounds for row in n:-1:1
        value=P.schur_solution[row]
        for column in row+1:n
            value-=U[row,column]*P.schur_solution[column]
        end
        pivot=U[row,row]+shift
        _heom_isfinite(pivot)&&!iszero(pivot)||throw(ErrorException(
            "guarded HEOM Schur solve encountered a singular or nonfinite shifted pivot"))
        P.schur_solution[row]=value/pivot
        _heom_isfinite(P.schur_solution[row])||throw(ErrorException(
            "guarded HEOM Schur solve produced a nonfinite transformed solution"))
    end

    # The Schur vectors are unitary, so this transformed residual has exactly
    # the same two-norm as the physical-coordinate block residual.
    @inbounds for row in 1:n
        value=shift*P.schur_solution[row]
        for column in row:n
            value+=U[row,column]*P.schur_solution[column]
        end
        P.schur_rhs[row]-=value
    end
    residual_norm=norm(P.schur_rhs)
    allowed=P.residual_tolerance*rhs_norm
    isfinite(residual_norm)&&residual_norm<=allowed||throw(ErrorException(
        "guarded HEOM Schur solve failed its residual check; rebuild the preconditioner with shift_backend=:lu"))
    mul!(destination,Z,P.schur_solution)
    all(_heom_isfinite,destination)||throw(ErrorException(
        "guarded HEOM Schur solve produced a nonfinite output"))
    destination
end

function _heom_block_ldiv_unlocked!(destination,P::HEOMBlockPreconditioner)
    for ado in eachindex(P.factor_indices)
        range=(ado-1)*P.npi+1:ado*P.npi
        factor_index=Int(P.factor_indices[ado])
        if iszero(factor_index)
            _heom_shifted_schur_solve!(view(destination,range),P,
                                       view(destination,range),P.shifts[ado])
        else
            factor=P.factors[factor_index]
            ldiv!(view(destination,range),factor,view(destination,range))
        end
    end
    destination
end

function ldiv!(destination::AbstractVector,P::HEOMBlockPreconditioner,
               source::AbstractVector)
    length(destination)==P.n&&length(source)==P.n||throw(DimensionMismatch(
        "HEOM preconditioner vector has the wrong length"))
    destination===source||copyto!(destination,source)
    if P.schur_vectors===nothing
        return _heom_block_ldiv_unlocked!(destination,P)
    end
    lock(P.schur_lock)
    try
        _heom_block_ldiv_unlocked!(destination,P)
    finally
        unlock(P.schur_lock)
    end
end

"""
    heom_block_preconditioner(plan; regularization=0,
                              operator_scale=nothing,
                              shift_backend=:auto,
                              schur_rcond_threshold=nothing,
                              schur_residual_tolerance=nothing,
                              expected_reuses=1,
                              expected_solve_applications=30,
                              warn_unamortized=true)

Construct a hierarchy-aware left preconditioner for trace-fixed HEOM GMRES.
The common dense `npi`-by-`npi` system block is extracted once, after which
every non-root diagonal block is formed by a scalar shift. With
`shift_backend=:auto`, `ComplexF32` and `ComplexF64` plans retain one LAPACK
Schur factorization. A triangular reciprocal-condition estimate guards every
distinct shift; unsafe shifts receive a conventional LU factor. Other scalar
types use duplicate-aware LU factors throughout. Pass `shift_backend=:lu` to
force that generic route, or `:schur` to require a LAPACK-supported plan.
`schur_rcond_threshold` and the application-time transformed-residual tolerance
default to precision-aware values. All hierarchy couplings are omitted from
the approximation. The root always has its own LU factor containing the
trace-fixing rank-one term, and every block uses the same reproducible
operator normalization as [`krylov_steady_state`](@ref).

Setup uses at most `plan.npi` system right-hand sides, evaluated in bounded
batches, rather than `heom_number_ados(plan) * plan.npi` complete HEOM
applications. Reuse the returned object across solves when that cost
amortizes. A positive `regularization` is an explicit numerical modification
of the preconditioner only; it never changes the HEOM generator. The guarded
Schur route retains two scratch vectors protected by a lock, so concurrent
applications of one preconditioner serialize; construct one preconditioner per
concurrent solve when that matters. The LU-only route retains no shared
numerical scratch and does not serialize.
"""
function heom_block_preconditioner(plan::HEOMPlan;
        regularization::Real=0,operator_scale=nothing,
        shift_backend::Symbol=:auto,schur_rcond_threshold=nothing,
        schur_residual_tolerance=nothing,
        expected_reuses::Integer=1,
        expected_solve_applications::Integer=30,
        warn_unamortized::Bool=true)
    _require_autonomous(plan,"HEOM block-preconditioner construction")
    shift_backend in (:auto,:schur,:lu)||throw(ArgumentError(
        "shift_backend must be :auto, :schur, or :lu"))
    isfinite(regularization)&&regularization>=0||throw(ArgumentError(
        "regularization must be finite and nonnegative"))
    operator_scale===nothing||_validated_operator_scale(operator_scale)
    expected_reuses>0||throw(ArgumentError(
        "expected_reuses must be positive"))
    expected_solve_applications>0||throw(ArgumentError(
        "expected_solve_applications must be positive"))
    BigInt(expected_reuses)<=typemax(Int)||throw(ArgumentError(
        "expected_reuses must be representable as an Int"))
    BigInt(expected_solve_applications)<=typemax(Int)||throw(ArgumentError(
        "expected_solve_applications must be representable as an Int"))
    expected_reuses_int=Int(expected_reuses)
    expected_solve_applications_int=Int(expected_solve_applications)
    started=time_ns();T=plan.Ttype;Rscale=_real_float_type(T)
    n=size(plan,1);npi=plan.npi
    schur_supported=T===ComplexF32||T===ComplexF64
    shift_backend===:schur&&!schur_supported&&throw(ArgumentError(
        "shift_backend=:schur requires a ComplexF32 or ComplexF64 HEOM plan; use :auto or :lu for generic precision"))
    use_schur=shift_backend!==:lu&&schur_supported
    default_rcond=sqrt(eps(Rscale))
    rcond_threshold=schur_rcond_threshold===nothing ? default_rcond :
        _heom_checked_time(Rscale,schur_rcond_threshold,
                           "schur_rcond_threshold")
    residual_tolerance=schur_residual_tolerance===nothing ? default_rcond :
        _heom_checked_time(Rscale,schur_residual_tolerance,
                           "schur_residual_tolerance")
    0<=rcond_threshold<=1||throw(ArgumentError(
        "schur_rcond_threshold must lie between zero and one"))
    residual_tolerance>=0||throw(ArgumentError(
        "schur_residual_tolerance must be nonnegative"))
    operator_scaleT=operator_scale===nothing ? nothing :
        _heom_checked_time(Rscale,_validated_operator_scale(operator_scale),
                           "operator_scale")
    regularizationT=try
        T(regularization)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError(
            "regularization is not representable in $(plan.Ttype); prepare the HEOM plan at wider precision"))
    end
    !iszero(regularization)&&iszero(regularizationT)&&throw(ArgumentError(
        "regularization underflows in $(plan.Ttype); use wider precision"))
    _heom_isfinite(regularizationT)&&regularizationT==regularization||
        throw(ArgumentError(
        "regularization is not exactly representable in $(plan.Ttype); prepare the HEOM plan at wider precision"))
    t=plan.tracevec
    v=_normalized_trace_functional(t)
    e=zeros(T,n);image=zeros(T,n)
    adapter=heom_liouvillian(plan)
    scale_probes=operator_scaleT===nothing ? 3 : 0
    Lscale=_validated_operator_scale(operator_scaleT===nothing ?
        _estimated_operator_scale!(adapter,e,image;
                                   probes=max(scale_probes,1)) :
        operator_scaleT)
    work=HEOMWorkspace(plan)
    system_block=if plan.system isa AbstractMatrix
        Matrix{T}(plan.system)
    else
        input=Matrix{T}(I,npi,npi)
        output=zeros(T,npi,npi)
        _heom_system_apply_hierarchy!(output,plan.system,input,
            zero(Rscale),nothing,work.system)
        output
    end
    system_rhs_applications=plan.system isa AbstractMatrix ? 0 : npi
    system_batch_applications=plan.system isa AbstractMatrix ? 0 :
        (work.system===nothing ? 1 : cld(npi,_HEOM_SYSTEM_BATCH_WIDTH))
    normalized_system=system_block./Lscale
    function factor_block(block)
        try
            lu(block;check=true)
        catch error
            error isa SingularException||rethrow()
            throw(ArgumentError(
                "an HEOM diagonal preconditioner block is singular; pass a small positive regularization"))
        end
    end
    root_block=copy(normalized_system)
    _add_trace_border_block!(root_block,v,t,0)
    if !iszero(regularizationT)
        @inbounds for diagonal in axes(root_block,1)
            root_block[diagonal,diagonal]+=regularizationT
        end
    end
    root_factor=factor_block(root_block)
    factors=typeof(root_factor)[root_factor]
    number_ados=heom_number_ados(plan)
    Iindex=eltype(plan.topology.ado_edge_ptr)
    factor_indices=Vector{Iindex}(undef,number_ados)
    factor_indices[1]=Iindex(1)
    shifts=zeros(T,number_ados)
    schur_vectors=nothing;schur_triangular=nothing
    condition_work=nothing
    if use_schur
        decomposition=LinearAlgebra.schur(normalized_system)
        schur_vectors=Matrix{T}(decomposition.Z)
        schur_triangular=Matrix{T}(decomposition.T)
        all(_heom_isfinite,schur_vectors)&&
            all(_heom_isfinite,schur_triangular)||throw(ArgumentError(
            "the normalized HEOM system block has a nonfinite Schur factorization"))
        condition_work=similar(schur_triangular)
    end
    shift_entries=Dict{T,Tuple{Int,Rscale}}()
    condition_estimates=Rscale[]
    for ado in 2:number_ados
        shift=regularizationT-plan.decays[ado]/Lscale
        shifts[ado]=shift
        entry=get(shift_entries,shift,nothing)
        if entry===nothing
            reciprocal_condition=zero(Rscale)
            safe_schur=false
            if use_schur
                copyto!(condition_work,schur_triangular)
                @inbounds for diagonal in axes(condition_work,1)
                    condition_work[diagonal,diagonal]+=shift
                end
                reciprocal_condition=try
                    LinearAlgebra.LAPACK.trcon!(
                        'I','U','N',condition_work)
                catch error
                    error isa LinearAlgebra.LAPACKException||rethrow()
                    zero(Rscale)
                end
                safe_schur=isfinite(reciprocal_condition)&&
                           reciprocal_condition>rcond_threshold
            end
            factor_index=0
            if !safe_schur
                block=copy(normalized_system)
                @inbounds for diagonal in axes(block,1)
                    block[diagonal,diagonal]+=shift
                end
                push!(factors,factor_block(block))
                factor_index=length(factors)
            end
            entry=(factor_index,reciprocal_condition)
            shift_entries[shift]=entry
            use_schur&&push!(condition_estimates,reciprocal_condition)
        end
        factor_indices[ado]=Iindex(first(entry))
    end
    F=eltype(factors)
    schur_shift_blocks=count(iszero,view(factor_indices,2:number_ados))
    fallback_shift_blocks=(number_ados-1)-schur_shift_blocks
    unique_shift_blocks=length(shift_entries)
    duplicate_shift_blocks=(number_ados-1)-unique_shift_blocks
    retain_schur=use_schur&&schur_shift_blocks>0
    if !retain_schur
        schur_vectors=nothing;schur_triangular=nothing
    end
    schur_rhs=retain_schur ? zeros(T,npi) : T[]
    schur_solution=retain_schur ? zeros(T,npi) : T[]
    schur_lock=ReentrantLock()
    setup_application_batches=scale_probes+system_batch_applications
    recommended_reuses=max(2,cld(max(setup_application_batches,1),
                                  expected_solve_applications_int))
    metadata=(setup_seconds=(time_ns()-started)/1e9,
              setup_liouvillian_applications=scale_probes,
              setup_system_rhs_applications=system_rhs_applications,
              setup_system_batch_applications=system_batch_applications,
              setup_application_batches,
              requested_shift_backend=shift_backend,
              shift_backend=retain_schur ? :schur : :lu,
              schur_supported,
              schur_attempted=use_schur,
              schur_rcond_threshold=rcond_threshold,
              schur_residual_tolerance=residual_tolerance,
              minimum_shift_rcond=isempty(condition_estimates) ? nothing :
                  minimum(condition_estimates),
              setup_lu_factorizations=length(factors),
              setup_schur_factorizations=use_schur ? 1 : 0,
              setup_factorizations=length(factors)+(use_schur ? 1 : 0),
              ado_blocks=number_ados,
              unique_shift_blocks,
              duplicate_shift_blocks,
              reused_shift_blocks=duplicate_shift_blocks,
              schur_shift_blocks,
              fallback_shift_blocks,
              stored_factor_coefficients=length(factors)*npi^2,
              stored_schur_coefficients=retain_schur ? 2npi^2 : 0,
              stored_shift_coefficients=length(shifts),
              stored_coefficients=length(factors)*npi^2+
                  (retain_schur ? 2npi^2+2npi : 0)+length(shifts),
              stored_bytes=Base.summarysize((factors,factor_indices,shifts,
                  schur_vectors,schur_triangular,schur_rhs,schur_solution,
                  schur_lock)),
              apply_triangular_solves=number_ados,
              apply_schur_residual_checks=schur_shift_blocks,
              apply_serialized=retain_schur,
              apply_flop_estimate=(number_ados-schur_shift_blocks)*2npi^2+
                                  schur_shift_blocks*5npi^2,
              recommended_minimum_reuses=recommended_reuses,
              expected_reuses=expected_reuses_int,
              amortization_expected=expected_reuses_int>=recommended_reuses)
    if warn_unamortized&&!metadata.amortization_expected
        @warn "HEOM block-preconditioner setup is unlikely to amortize for the declared reuse count" expected_reuses recommended_reuses setup_application_batches maxlog=1
    end
    HEOMBlockPreconditioner{T,F,Iindex,typeof(schur_vectors),
        typeof(schur_triangular),typeof(residual_tolerance),typeof(Lscale),
        typeof(metadata)}(
        factors,factor_indices,shifts,schur_vectors,schur_triangular,
        schur_rhs,schur_solution,schur_lock,n,npi,regularizationT,
        residual_tolerance,Lscale,metadata)
end

"""
    heom_steady_state(plan; initial_state=nothing, return_info=false,
                      preconditioner=nothing,
                      preconditioner_regularization=0, kwargs...)

Compute the trace-fixed stationary hierarchy with the existing restarted
matrix-free GMRES solver. `kwargs` are forwarded to `krylov_steady_state`.
The physical stationary density operator is `heom_reduced_state(result)`.
Set `preconditioner=:block` to construct and use one
[`HEOMBlockPreconditioner`](@ref), or pass a compatible reusable
preconditioner directly.
"""
function heom_steady_state(plan::HEOMPlan;initial_state=nothing,
                           return_info::Bool=false,preconditioner=nothing,
                           preconditioner_regularization::Real=0,
                           operator_scale=nothing,kwargs...)
    _require_autonomous(plan,"HEOM steady-state solving")
    initial=initial_state isa HEOMState ? begin
        initial_state.plan===plan||throw(ArgumentError(
            "initial HEOM state belongs to a different plan"))
        initial_state.data
    end : initial_state isa PIState ? heom_initial_state(plan,initial_state).data :
          initial_state
    prepared_preconditioner=if preconditioner===:block
        heom_block_preconditioner(plan;
            regularization=preconditioner_regularization,operator_scale,
            warn_unamortized=false)
    elseif preconditioner isa Symbol
        throw(ArgumentError("unknown HEOM preconditioner $preconditioner"))
    else
        iszero(preconditioner_regularization)||throw(ArgumentError(
            "preconditioner_regularization applies only to preconditioner=:block"))
        preconditioner
    end
    selected_scale=prepared_preconditioner isa HEOMBlockPreconditioner ?
        prepared_preconditioner.operator_scale : operator_scale
    result=krylov_steady_state(heom_liouvillian(plan);
        trace_vector=plan.tracevec,initial_state=initial,
        return_info=return_info,preconditioner=prepared_preconditioner,
        operator_scale=selected_scale,kwargs...)
    if return_info
        return merge(result,(state=HEOMState(plan,result.state),))
    end
    HEOMState(plan,result)
end
