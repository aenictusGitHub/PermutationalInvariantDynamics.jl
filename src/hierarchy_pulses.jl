"""
    PIUnitaryPulse(basis, unitary;
                   atol=0, rtol=sqrt(eps(...)), cache=nothing,
                   memory_budget=512MiB)

Prepare one instantaneous PI system unitary for hierarchy dynamics.
`unitary` may be either a local `d`-by-`d` matrix, which is lifted as
`unitary^⊗N`, or a `PIOperator` on the exact `basis`. Only physical Schur
blocks are retained; no PI-coordinate superoperator or full-Hilbert matrix is
constructed.

The prepared pulse is immutable and shareable. Applying it to HEOM maps every
ADO as `rho_n -> U*rho_n*U'`; applying it to HOPS maps every auxiliary ket as
`psi_n -> U*psi_n`. Mutable application scratch belongs to the corresponding
hierarchy workspace.
"""
struct PIUnitaryPulse{B,T,P,Q}
    basis::B
    blocks::P
    precision_bits::Int
    rounding_mode::Q
end

eltype(::PIUnitaryPulse{B,T}) where {B,T}=T

function show(io::IO,pulse::PIUnitaryPulse)
    print(io,"PIUnitaryPulse(N=$(pulse.basis.N), d=$(pulse.basis.d), " *
             "sectors=$(length(pulse.blocks)))")
end

@inline _hierarchy_pulse_value_precision(value)=0
@inline _hierarchy_pulse_value_precision(value::BigFloat)=precision(value)
@inline _hierarchy_pulse_value_precision(value::Complex{BigFloat})=
    max(precision(real(value)),precision(imag(value)))

function _hierarchy_pulse_array_precision(array)
    _real_float_type(eltype(array))===BigFloat||return 0
    maximum(_hierarchy_pulse_value_precision,array;init=0)
end

function _with_hierarchy_pulse_precision(
        f,::Type{R},precision_bits::Integer,rounding_mode) where
        R<:AbstractFloat
    R===BigFloat||return f()
    setrounding(BigFloat,rounding_mode) do
        setprecision(BigFloat,Int(precision_bits)) do
            f()
        end
    end
end

function _hierarchy_pulse_geometry_estimate(
        basis::PIBasis,::Type{R},precision_bits::Integer) where
        R<:AbstractFloat
    if _has_single_fully_symmetric_sector(basis)&&
            !_needs_wide_collective(basis,R)
        _estimate_symmetric_collective_geometry(
            basis,R;bigfloat_precision=precision_bits)
    else
        _estimate_onebody_geometry(
            basis,R;diagonal_only=true,
            bigfloat_precision=precision_bits)
    end
end

function _hierarchy_pulse_geometry(
        basis::PIBasis,::Type{R}) where R<:AbstractFloat
    if _has_single_fully_symmetric_sector(basis)&&
            !_needs_wide_collective(basis,R)
        _SymmetricCollectiveGeometry(basis,R)
    else
        _diagonal_onebody_geometry(basis,R)
    end
end

function PIUnitaryPulse(basis::PIBasis,unitary;
        atol::Real=0,rtol=nothing,cache=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    unitary isa Union{AbstractMatrix,PIOperator}||throw(ArgumentError(
        "a PI unitary pulse must be a local matrix or PIOperator"))
    source=unitary isa PIOperator ? unitary.data : unitary
    T=_complex_float_type(eltype(source))
    R=_real_float_type(T)
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),_hierarchy_pulse_array_precision(source)) :
        precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    retained_bytes=_performance_entries_bytes(
        length(basis),T;bigfloat_precision=precision_bits)
    setup_bytes=if unitary isa AbstractMatrix&&cache===nothing
        estimate=_hierarchy_pulse_geometry_estimate(
            basis,R,precision_bits)
        retained_bytes+BigInt(estimate.setup_bytes)
    else
        retained_bytes
    end
    _require_performance_budget(
        "PI hierarchy pulse preparation",setup_bytes,memory_budget;
        guidance="Use a restricted invariant Schur basis when physically appropriate.")
    _with_hierarchy_pulse_precision(
            R,precision_bits,rounding_mode) do
        resolved_rtol=rtol===nothing ? sqrt(eps(R)) : rtol
        prepared_cache=unitary isa AbstractMatrix&&cache===nothing ?
            _hierarchy_pulse_geometry(basis,R) : cache
        raw_blocks=_pi_unitary_blocks(
            basis,unitary;atol,rtol=resolved_rtol,
            cache=prepared_cache)
        blocks=Vector{Matrix{T}}(undef,length(raw_blocks))
        for index in eachindex(raw_blocks)
            block=Matrix{T}(raw_blocks[index])
            all(isfinite,block)||throw(ArgumentError(
                "PI hierarchy pulse block $index contains nonfinite values"))
            blocks[index]=block
        end
        PIUnitaryPulse{typeof(basis),T,typeof(blocks),
                       typeof(rounding_mode)}(
            basis,blocks,precision_bits,rounding_mode)
    end
end

"""
    HierarchyPulseSequence(times, pulse)
    HierarchyPulseSequence(times, pulses)

Prepare ordered instantaneous events for HEOM or HOPS evolution. The first
form reuses one [`PIUnitaryPulse`](@ref) at every time. The second accepts one
compatible prepared pulse per time. Times must be finite and nondecreasing;
equal times are allowed and their pulses are applied deterministically in
input order.

Evolution applies events in the half-open convention `(start, stop]`. Thus a
pulse exactly at a saved time is applied before that state is saved, while a
pulse at the initial time is not applied implicitly. This convention prevents
double application when adjacent propagation intervals share an endpoint.
Events outside a requested propagation span are retained but ignored.
"""
struct HierarchyPulseSequence{B,R,P}
    basis::B
    times::Vector{R}
    pulses::Vector{P}
end

function show(io::IO,sequence::HierarchyPulseSequence)
    print(io,"HierarchyPulseSequence($(length(sequence.times)) events, " *
             "N=$(sequence.basis.N), d=$(sequence.basis.d))")
end

function _hierarchy_pulse_time(::Type{R},value,index::Int) where
        R<:AbstractFloat
    value isa Real&&!(value isa Bool)&&isfinite(value)||throw(ArgumentError(
        "hierarchy pulse time $index must be a finite real number"))
    converted=try
        R(value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError(
            "hierarchy pulse time $index is not representable in $R"))
    end
    isfinite(converted)||throw(ArgumentError(
        "hierarchy pulse time $index is not finite in $R"))
    converted==value||throw(ArgumentError(
        "hierarchy pulse time $index is not exactly representable in $R"))
    converted
end

function HierarchyPulseSequence(times,pulse::PIUnitaryPulse)
    raw=collect(times)
    R=_real_float_type(eltype(pulse))
    converted=Vector{R}(undef,length(raw))
    for index in eachindex(raw)
        converted[index]=_hierarchy_pulse_time(R,raw[index],index)
    end
    issorted(converted)||throw(ArgumentError(
        "hierarchy pulse times must be nondecreasing"))
    pulses=fill(pulse,length(converted))
    HierarchyPulseSequence{typeof(pulse.basis),R,typeof(pulse)}(
        pulse.basis,converted,pulses)
end

function HierarchyPulseSequence(times,pulses_input::AbstractVector)
    pulses=collect(pulses_input)
    isempty(pulses)&&throw(ArgumentError(
        "an empty per-event pulse vector has no basis; use " *
        "HierarchyPulseSequence(times, pulse)"))
    first_pulse=first(pulses)
    first_pulse isa PIUnitaryPulse||throw(ArgumentError(
        "every hierarchy pulse event must be a PIUnitaryPulse"))
    all(pulse->pulse isa typeof(first_pulse),pulses)||throw(ArgumentError(
        "all hierarchy pulse events must use the same prepared scalar type"))
    all(pulse->pulse.basis===first_pulse.basis,pulses)||throw(ArgumentError(
        "all hierarchy pulse events must use the exact same PI basis"))
    raw=collect(times)
    length(raw)==length(pulses)||throw(DimensionMismatch(
        "hierarchy pulse times and pulses must have equal lengths"))
    R=_real_float_type(eltype(first_pulse))
    converted=Vector{R}(undef,length(raw))
    for index in eachindex(raw)
        converted[index]=_hierarchy_pulse_time(R,raw[index],index)
    end
    issorted(converted)||throw(ArgumentError(
        "hierarchy pulse times must be nondecreasing"))
    typed=Vector{typeof(first_pulse)}(pulses)
    HierarchyPulseSequence{typeof(first_pulse.basis),R,
                           typeof(first_pulse)}(
        first_pulse.basis,converted,typed)
end

const _TETRAHEDRAL_DD_WORD="abaababbbaababbbaababbaa"
const _OCTAHEDRAL_DD_WORD=
    "abaaabbbabaabbbaababbaaa" *
    "ababbbabaabbaaaababbbabb"
const _ICOSAHEDRAL_DD_WORD=
    "baaabbaabaaaaabbaaab" *
    "abbbabaabbaabbabbabb" *
    "abbbaaaababbbaaababb" *
    "baaababbbaababbaabba" *
    "abbaabbbabbbaababbba" *
    "ababbbaababbbabaaaaa"

function _platonic_pulse_spec(group,::Type{R}) where R<:AbstractFloat
    one_R=one(R)
    zero_R=zero(R)
    pi_R=R(pi)
    if group===:tetrahedral
        axis_a=(zero_R,zero_R,one_R)
        axis_b=(sqrt(R(2))/R(3),sqrt(R(2)/R(3)),one_R/R(3))
        angle_a=R(2)*pi_R/R(3)
        angle_b=angle_a
        word=_TETRAHEDRAL_DD_WORD
    elseif group===:octahedral
        axis_a=(zero_R,zero_R,one_R)
        inverse_sqrt_three=inv(sqrt(R(3)))
        axis_b=(inverse_sqrt_three,inverse_sqrt_three,
                inverse_sqrt_three)
        angle_a=pi_R/R(2)
        angle_b=R(2)*pi_R/R(3)
        word=_OCTAHEDRAL_DD_WORD
    elseif group===:icosahedral
        phi=(one_R+sqrt(R(5)))/R(2)
        axis_a=(zero_R,-one_R/sqrt(phi+R(2)),
                phi/sqrt(phi+R(2)))
        inverse_sqrt_three=inv(sqrt(R(3)))
        axis_b=((one_R-phi)*inverse_sqrt_three,zero_R,
                phi*inverse_sqrt_three)
        angle_a=R(2)*pi_R/R(5)
        angle_b=R(2)*pi_R/R(3)
        word=_ICOSAHEDRAL_DD_WORD
    else
        throw(ArgumentError(
            "Platonic group must be :tetrahedral, :octahedral, or " *
            ":icosahedral"))
    end
    (;axis_a,axis_b,angle_a,angle_b,word)
end

function _platonic_cycle_count(cycles)
    cycles isa Integer&&!(cycles isa Bool)||throw(ArgumentError(
        "Platonic sequence cycles must be a nonnegative integer"))
    cycles>=0||throw(ArgumentError(
        "Platonic sequence cycles must be nonnegative"))
    try
        Int(cycles)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError(
            "Platonic sequence cycles must be representable as an Int"))
    end
end

function _platonic_event_count(word::AbstractString,cycles::Int)
    try
        Base.checked_mul(ncodeunits(word),cycles)
    catch error
        error isa OverflowError||rethrow()
        throw(ArgumentError(
            "the requested Platonic sequence has too many pulse events"))
    end
end

function _platonic_pulse_times(
        ::Type{R},start_time,pulse_interval,event_count::Int) where
        R<:AbstractFloat
    start=_hierarchy_pulse_time(R,start_time,0)
    interval=_hierarchy_pulse_time(R,pulse_interval,0)
    interval>zero(R)||throw(ArgumentError(
        "the Platonic pulse interval must be positive"))
    times=Vector{R}(undef,event_count)
    for event in eachindex(times)
        event_R=_hierarchy_pulse_time(R,event,event)
        time=start+event_R*interval
        isfinite(time)||throw(ArgumentError(
            "Platonic pulse time $event is not finite in $R"))
        times[event]=time
    end
    times
end

function _platonic_local_rotation(
        spin,axis::NTuple{3,R},angle::R) where R<:AbstractFloat
    generator=axis[1]*spin.jx+axis[2]*spin.jy+axis[3]*spin.jz
    exp(-complex(zero(R),one(R))*angle*generator)
end

"""
    platonic_pulse_sequence(basis, group, pulse_interval;
                            cycles=1, start_time=0, T=Float64,
                            atol=0, rtol=nothing, cache=nothing,
                            memory_budget=512MiB)

Construct one of the ideal Eulerian Platonic dynamical-decoupling schedules
of Read, Serrano-Ensástiga, and Martin, *Quantum* **9**, 1661 (2025).
`group` must be `:tetrahedral`, `:octahedral`, or `:icosahedral`, producing
the published TEDD, OEDD, or IEDD word of 24, 48, or 120 pulses per cycle.

Every pulse is a global spin rotation, represented locally as
`exp(-im*angle*(axis_x*jx + axis_y*jy + axis_z*jz))` for the spin
`j=(basis.d-1)/2`. The first event occurs after one positive
`pulse_interval`; a free interval precedes every event, including the final
cyclic pulse. Repeated cycles concatenate the published word without
repreparing its two generators. `start_time` shifts the complete schedule.
The built-in axis--angle matrix preparation supports `T=Float32` and
`T=Float64`; wider custom rotations can still be supplied to
[`PIUnitaryPulse`](@ref) as prepared PI operators and scheduled with
[`HierarchyPulseSequence`](@ref).

The returned [`HierarchyPulseSequence`](@ref) acts directly on every HEOM ADO
or HOPS auxiliary. These are instantaneous pulses. Their Eulerian
finite-duration robustness does not model a nonzero pulse width; finite
pulses require an explicit time-dependent control Hamiltonian.
"""
function platonic_pulse_sequence(
        basis::PIBasis,group,pulse_interval;
        cycles=1,start_time=0,
        T::Type{<:AbstractFloat}=Float64,
        atol::Real=0,rtol=nothing,cache=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    isconcretetype(T)||throw(ArgumentError(
        "Platonic pulse scalar type must be concrete, got $T"))
    T in (Float32,Float64)||throw(ArgumentError(
        "built-in Platonic rotations support T=Float32 or T=Float64; " *
        "for wider precision, prepare compatible PIUnitaryPulse objects " *
        "from PI operators and pass them to HierarchyPulseSequence"))
    cycle_count=_platonic_cycle_count(cycles)
    spec=_platonic_pulse_spec(group,T)
    event_count=_platonic_event_count(spec.word,cycle_count)
    precision_bits=T===BigFloat ? precision(BigFloat) : precision(T)
    pulse_scalar=Complex{T}
    pulse_bytes=big(2)*_performance_entries_bytes(
        length(basis),pulse_scalar;bigfloat_precision=precision_bits)
    schedule_bytes=_performance_entries_bytes(
        event_count,T;bigfloat_precision=precision_bits)+
        BigInt(event_count)*BigInt(sizeof(Ptr{Cvoid}))
    geometry_estimate=cache===nothing ?
        _hierarchy_pulse_geometry_estimate(
            basis,T,precision_bits) : nothing
    geometry_setup=geometry_estimate===nothing ?
        big(0) : BigInt(geometry_estimate.setup_bytes)
    _require_performance_budget(
        "Platonic hierarchy pulse preparation",
        pulse_bytes+schedule_bytes+geometry_setup,memory_budget;
        guidance="Reduce the cycle count or use a restricted invariant Schur basis.")

    times=_platonic_pulse_times(
        T,start_time,pulse_interval,event_count)
    prepared_cache=cache===nothing ?
        _hierarchy_pulse_geometry(basis,T) : cache
    spin=spin_matrices(basis.d;T)
    unitary_a=_platonic_local_rotation(
        spin,spec.axis_a,spec.angle_a)
    unitary_b=_platonic_local_rotation(
        spin,spec.axis_b,spec.angle_b)
    # The combined preflight above includes both blocks, the shared geometry,
    # and the complete schedule; do not apply two smaller independent guards.
    pulse_a=PIUnitaryPulse(
        basis,unitary_a;atol,rtol,cache=prepared_cache,
        memory_budget=Inf)
    pulse_b=PIUnitaryPulse(
        basis,unitary_b;atol,rtol,cache=prepared_cache,
        memory_budget=Inf)
    event_count==0&&return HierarchyPulseSequence(times,pulse_a)

    pulses=Vector{typeof(pulse_a)}(undef,event_count)
    word_length=ncodeunits(spec.word)
    for event in eachindex(pulses)
        letter=codeunit(spec.word,mod1(event,word_length))
        pulses[event]=letter==UInt8('a') ? pulse_a : pulse_b
    end
    HierarchyPulseSequence(times,pulses)
end

"""
    tetrahedral_pulse_sequence(basis, pulse_interval; kwargs...)

Construct the published 24-pulse tetrahedral Eulerian DD sequence (TEDD).
See [`platonic_pulse_sequence`](@ref) for timing, precision, and memory
keywords.
"""
tetrahedral_pulse_sequence(
    basis::PIBasis,pulse_interval;kwargs...)=
    platonic_pulse_sequence(
        basis,:tetrahedral,pulse_interval;kwargs...)

"""
    octahedral_pulse_sequence(basis, pulse_interval; kwargs...)

Construct the published 48-pulse octahedral Eulerian DD sequence (OEDD).
See [`platonic_pulse_sequence`](@ref) for timing, precision, and memory
keywords.
"""
octahedral_pulse_sequence(
    basis::PIBasis,pulse_interval;kwargs...)=
    platonic_pulse_sequence(
        basis,:octahedral,pulse_interval;kwargs...)

"""
    icosahedral_pulse_sequence(basis, pulse_interval; kwargs...)

Construct the published 120-pulse icosahedral Eulerian DD sequence (IEDD).
See [`platonic_pulse_sequence`](@ref) for timing, precision, and memory
keywords.
"""
icosahedral_pulse_sequence(
    basis::PIBasis,pulse_interval;kwargs...)=
    platonic_pulse_sequence(
        basis,:icosahedral,pulse_interval;kwargs...)

function _check_hierarchy_pulse(
        pulse::PIUnitaryPulse,basis::PIBasis,::Type{T};
        precision_bits=nothing,rounding_mode=nothing) where T
    pulse.basis===basis||throw(ArgumentError(
        "hierarchy pulse and evolution plan use different PI bases"))
    eltype(pulse)===T||throw(ArgumentError(
        "hierarchy pulse scalar type $(eltype(pulse)) does not match " *
        "evolution scalar type $T"))
    R=_real_float_type(T)
    if R===BigFloat
        precision(BigFloat)==pulse.precision_bits||throw(ArgumentError(
            "hierarchy pulse requires BigFloat precision " *
            "$(pulse.precision_bits), not $(precision(BigFloat))"))
        rounding(BigFloat)==pulse.rounding_mode||throw(ArgumentError(
            "hierarchy pulse requires BigFloat rounding mode " *
            "$(pulse.rounding_mode)"))
        precision_bits===nothing||precision_bits==pulse.precision_bits||
            throw(ArgumentError(
            "hierarchy pulse and evolution plan use different BigFloat precisions"))
        rounding_mode===nothing||rounding_mode==pulse.rounding_mode||
            throw(ArgumentError(
            "hierarchy pulse and evolution plan use different BigFloat rounding modes"))
    end
    pulse
end

function _check_hierarchy_pulse_sequence(
        sequence::HierarchyPulseSequence,basis::PIBasis,::Type{T};
        precision_bits=nothing,rounding_mode=nothing) where T
    sequence.basis===basis||throw(ArgumentError(
        "hierarchy pulse sequence and evolution plan use different PI bases"))
    for pulse in sequence.pulses
        _check_hierarchy_pulse(
            pulse,basis,T;precision_bits,rounding_mode)
    end
    sequence
end

@inline function _hierarchy_pulse_event_range(
        sequence::HierarchyPulseSequence,start,stop)
    stop>=start||throw(ArgumentError(
        "hierarchy pulse evolution requires a nondecreasing time span"))
    first_event=searchsortedlast(sequence.times,start)+1
    last_event=searchsortedlast(sequence.times,stop)
    first_event:last_event
end
