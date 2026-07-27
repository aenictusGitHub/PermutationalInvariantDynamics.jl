"""
    Models

Curated, convention-explicit model recipes built from the stable PI term API.
These constructors are the single source used by flagship documentation and
are intended to prevent literature examples from carrying subtly different
normalizations of the same model.

Use `Models.catalog()` for machine-readable discovery.
"""
module Models

using LinearAlgebra
import ..PermutationalInvariantDynamics:
    PIBasis,PIModel,LocalHamiltonian,CollectiveHamiltonian,
    LocalJump,CollectiveJump,DirectPIHamiltonian,
    qubit_ensemble_model,spin_matrices,collective_operator,OneBodyGeometry,
    _checked_mul_exact_ratio

export catalog,find,describe,example,driven_qubits,
       independent_dephasing,local_pump_decay,
       one_axis_twisting,steady_superradiance,boundary_time_crystal

const _CATALOG=(
    driven_qubits=(
        title="Driven qubits with local pumping and decay",
        citation=nothing,
        difficulty=:beginner,
        sectors=:complete,
        tasks=(:dynamics,:steady_state,:trajectories),
        script="driven_qubits.jl",
        guide="driven_qubits.md",
        constructor=:driven_qubits),
    independent_dephasing=(
        title="Independent Markovian qubit dephasing",
        citation="Huelga et al., Phys. Rev. Lett. 79, 3865 (1997)",
        difficulty=:beginner,
        sectors=:complete,
        tasks=(:dynamics,:qfi),
        script="independent_dephasing_coherence.jl",
        guide="independent_dephasing_coherence.md",
        constructor=:independent_dephasing),
    local_pump_decay=(
        title="Independent incoherent pumping and emission",
        citation="Shammah et al., Phys. Rev. A 98, 063815 (2018)",
        difficulty=:beginner,
        sectors=:complete,
        tasks=(:steady_state,:populations),
        script="local_pumping.jl",
        guide="local_pumping.md",
        constructor=:local_pump_decay),
    one_axis_twisting=(
        title="One-axis twisting",
        citation="Kitagawa and Ueda, Phys. Rev. A 47, 5138 (1993)",
        difficulty=:intermediate,
        sectors=:complete,
        tasks=(:dynamics,:squeezing,:entanglement),
        script="one_axis_twisting.jl",
        guide="one_axis_twisting.md",
        constructor=:one_axis_twisting),
    steady_superradiance=(
        title="Steady-state superradiance",
        citation="Meiser and Holland, Phys. Rev. A 81, 033847 (2010)",
        difficulty=:intermediate,
        sectors=:complete,
        tasks=(:steady_state,:observables,:trajectories),
        script="steady_superradiance.jl",
        guide="steady_superradiance.md",
        constructor=:steady_superradiance),
    boundary_time_crystal=(
        title="Boundary time crystal",
        citation="Iemini et al., Phys. Rev. Lett. 121, 035301 (2018)",
        difficulty=:advanced,
        sectors=:symmetric,
        tasks=(:spectrum,:gap,:dynamics),
        script="boundary_time_crystal.jl",
        guide="boundary_time_crystal.md",
        constructor=:boundary_time_crystal),
)

"""Return detached metadata for the curated built-in model recipes."""
catalog()=deepcopy(_CATALOG)

"""
    find(; task=nothing, difficulty=nothing, sectors=nothing)

Return detached metadata for built-in model recipes matching the requested
discovery filters. Filters may be symbols or strings. `task` matches one of a
recipe's documented tasks; `difficulty` and `sectors` match their corresponding
metadata exactly. With no filters this is equivalent to [`catalog`](@ref).
"""
function find(;task=nothing,difficulty=nothing,sectors=nothing)
    normalize(value)=value===nothing ? nothing : Symbol(value)
    selected_task=normalize(task)
    selected_difficulty=normalize(difficulty)
    selected_sectors=normalize(sectors)
    entries=Pair{Symbol,Any}[]
    for name in keys(_CATALOG)
        metadata=getproperty(_CATALOG,name)
        selected_task===nothing||selected_task in metadata.tasks||continue
        selected_difficulty===nothing||
            selected_difficulty===metadata.difficulty||continue
        selected_sectors===nothing||selected_sectors===metadata.sectors||continue
        push!(entries,name=>deepcopy(metadata))
    end
    (;entries...)
end

"""
    describe(name)

Return detached convention, task, citation, and example metadata for one
built-in recipe. `name` may be a symbol or string. Unknown names raise and list
the available recipes instead of silently returning an empty result.
"""
function describe(name)
    key=Symbol(name)
    hasproperty(_CATALOG,key)||throw(ArgumentError(
        "unknown model recipe $name; choose one of "*
        join(string.(keys(_CATALOG)),", ")))
    deepcopy(getproperty(_CATALOG,key))
end

"""
    example(name)

Return the installed runnable-example paths and exact root-environment command
for a built-in model recipe. This function is read-only: it never starts a
process, opens an editor, or changes the active project.
"""
function example(name)
    metadata=describe(name)
    root=normpath(joinpath(@__DIR__,".."))
    script=joinpath(root,"examples",metadata.script)
    guide=joinpath(root,"examples",metadata.guide)
    isfile(script)||throw(ArgumentError(
        "the installed example script is missing: $script"))
    isfile(guide)||throw(ArgumentError(
        "the installed example guide is missing: $guide"))
    (;name=Symbol(name),script,guide,
      command=`$(Base.julia_cmd()) --project=$root $script`,
      metadata)
end

function _recipe_scalar(value,::Type{T},name::AbstractString;
        nonnegative::Bool=false,positive::Bool=false) where
        T<:AbstractFloat
    isconcretetype(T)||throw(ArgumentError(
        "T must be a concrete AbstractFloat type, got $T"))
    value isa Real&&!(value isa Bool)||throw(ArgumentError(
        "$name must be a real number"))
    isfinite(value)||throw(ArgumentError("$name must be finite"))
    nonnegative&&value<zero(value)&&throw(ArgumentError(
        "$name must be nonnegative"))
    positive&&value<=zero(value)&&throw(ArgumentError(
        "$name must be positive"))
    converted=try
        T(value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError(
            "$name is not representable in requested recipe precision $T"))
    end
    isfinite(converted)||throw(ArgumentError(
        "$name is not finite in requested recipe precision $T"))
    !iszero(value)&&iszero(converted)&&throw(ArgumentError(
        "$name is nonzero but underflows in requested recipe precision $T; " *
        "choose a wider T"))
    converted
end

@inline function _recipe_exact_scale(value::T,numerator::Integer,
        denominator::Integer,name::AbstractString) where T<:AbstractFloat
    # Ordinary IEEE recipe coefficients stay on a native, allocation-free
    # setup path. Endpoint, underflow, overflow, or large-integer cases fall
    # through to the exact binary-scaled implementation for certification.
    if T===Float16||T===Float32||T===Float64
        converted_numerator=try
            T(numerator)
        catch
            nothing
        end
        converted_denominator=try
            T(denominator)
        catch
            nothing
        end
        if converted_numerator!==nothing&&converted_denominator!==nothing&&
           isfinite(converted_numerator)&&isfinite(converted_denominator)&&
           !iszero(converted_denominator)
            result=value*(converted_numerator/converted_denominator)
            if iszero(value)
                return result
            end
            endpoint=isfinite(result)&&!iszero(result)&&(
                abs(result)==floatmax(T)||
                abs(result)==nextfloat(zero(T)))
            isfinite(result)&&!iszero(result)&&!endpoint&&return result
        end
    end
    _checked_mul_exact_ratio(
        T,value,numerator,denominator;
        context="$name in requested recipe precision $T")
end

"""
    driven_qubits(N; drive=0.7, detuning=0, emission=0.12,
                  pumping=0.02, dephasing=0, sectors=nothing, T=Float64)

Standard complete-PI qubit ensemble in local order `(|g>,|e>)`, with
`H = drive*jx + detuning*jz` summed over particles and independent Lindblad
channels. Rates use the package convention `D[L]ρ=LρL†-{L†L,ρ}/2`.
"""
function driven_qubits(N::Integer;drive=0.7,detuning=0,
        emission=0.12,pumping=0.02,dephasing=0,sectors=nothing,
        T::Type{<:AbstractFloat}=Float64)
    drive_T=_recipe_scalar(drive,T,"drive")
    detuning_T=_recipe_scalar(detuning,T,"detuning")
    emission_T=_recipe_scalar(
        emission,T,"emission";nonnegative=true)
    pumping_T=_recipe_scalar(
        pumping,T,"pumping";nonnegative=true)
    dephasing_T=_recipe_scalar(
        dephasing,T,"dephasing";nonnegative=true)
    spin=spin_matrices(2;T)
    # For a qubit, 2*jx and 2*jz contain only exact zero/unit entries. Form the
    # physical half-amplitudes with the checked exact-ratio route first, so a
    # nonzero drive cannot disappear when `T` is a narrow floating type.
    drive_half=_recipe_exact_scale(drive_T,1,2,"drive/2")
    detuning_half=_recipe_exact_scale(detuning_T,1,2,"detuning/2")
    hamiltonian=
        drive_half*(T(2)*spin.jx)+detuning_half*(T(2)*spin.jz)
    qubit_ensemble_model(
        N;sectors,hamiltonian,emission=emission_T,
        pumping=pumping_T,dephasing=dephasing_T,T)
end

"""
    independent_dephasing(N; gamma=1, sectors=nothing, T=Float64)

Independent dephasing normalized so one-qubit coherence decays as
`exp(-gamma*t)`.
"""
function independent_dephasing(N::Integer;gamma=1.0,sectors=nothing,
                               T::Type{<:AbstractFloat}=Float64)
    gamma_T=_recipe_scalar(gamma,T,"gamma";nonnegative=true)
    basis=PIBasis(N,2;sectors)
    spin=spin_matrices(2;T)
    rate=_recipe_exact_scale(gamma_T,1,2,"independent-dephasing rate")
    PIModel(basis,(LocalJump(T(2)*spin.jz;rate),))
end

"""
    local_pump_decay(N; down=1, up=0.25, sectors=nothing, T=Float64)

Independent emission and pumping with the exact product stationary excited
population `up/(down+up)` when `down+up>0`.
"""
function local_pump_decay(N::Integer;down=1.0,up=0.25,sectors=nothing,
                          T::Type{<:AbstractFloat}=Float64)
    down_T=_recipe_scalar(down,T,"down";nonnegative=true)
    up_T=_recipe_scalar(up,T,"up";nonnegative=true)
    down_T+up_T>zero(T)||throw(ArgumentError(
        "at least one of down and up must be positive"))
    basis=PIBasis(N,2;sectors)
    spin=spin_matrices(2;T)
    PIModel(basis,(
        LocalJump(spin.jm;rate=down_T),
        LocalJump(spin.jp;rate=up_T)))
end

"""
    one_axis_twisting(N; chi=1, sectors=nothing, T=Float64)

Kitagawa--Ueda Hamiltonian `H=chi*Jz^2`. The complete PI basis is retained by
default so this recipe can be combined safely with later local channels.
"""
function one_axis_twisting(N::Integer;chi=1.0,sectors=nothing,
                           T::Type{<:AbstractFloat}=Float64)
    chi_T=_recipe_scalar(chi,T,"chi")
    basis=PIBasis(N,2;sectors)
    spin=spin_matrices(2;T)
    geometry=OneBodyGeometry(basis,T)
    Jz=collective_operator(basis,spin.jz;cache=geometry)
    PIModel(basis,(DirectPIHamiltonian(Jz*Jz;rate=chi_T),))
end

"""
    steady_superradiance(N; collective_decay=1, pump=1, T=Float64)

Meiser--Holland collective emission with independent incoherent pumping.
Every Schur sector is retained because the local pump changes total spin.
"""
function steady_superradiance(N::Integer;collective_decay=1.0,pump=1.0,
                              T::Type{<:AbstractFloat}=Float64)
    decay_T=_recipe_scalar(
        collective_decay,T,"collective_decay";nonnegative=true)
    pump_T=_recipe_scalar(pump,T,"pump";nonnegative=true)
    basis=PIBasis(N,2)
    spin=spin_matrices(2;T)
    PIModel(basis,(
        CollectiveJump(spin.jm;rate=decay_T),
        LocalJump(spin.jp;rate=pump_T)))
end

"""
    boundary_time_crystal(N; omega=1.5, kappa=1, T=Float64)

Iemini *et al.* symmetric-sector boundary-time-crystal generator with
`H=omega*Jx` and package-convention collective rate `2kappa/N`.
"""
function boundary_time_crystal(N::Integer;omega=1.5,kappa=1.0,
                               T::Type{<:AbstractFloat}=Float64)
    N>0||throw(ArgumentError("N must be positive"))
    omega_T=_recipe_scalar(omega,T,"omega")
    kappa_T=_recipe_scalar(kappa,T,"kappa";nonnegative=true)
    basis=PIBasis(N,2;sectors=[(N,0)])
    spin=spin_matrices(2;T)
    rate=_recipe_exact_scale(
        kappa_T,2,N,"boundary-time-crystal collective rate")
    PIModel(basis,(
        CollectiveHamiltonian(spin.jx;rate=omega_T),
        CollectiveJump(spin.jm;rate)))
end

end # module Models
