"""
    HEOMBath(coupling, coefficients, frequencies;
             right_coefficients=nothing,
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
input instead of rounding it.
"""
struct HEOMBath{O,C,R,F}
    coupling::O
    coefficients::C
    right_coefficients::R
    frequencies::F
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
    HEOMBath{typeof(coupling),typeof(cs),typeof(rights),typeof(nus)}(
        copy(coupling),cs,rights,nus)
end

HEOMBath(coupling::PIOperator,coefficient::Number,frequency::Number;kwargs...)=
    HEOMBath(coupling,[coefficient],[frequency];kwargs...)

show(io::IO,bath::HEOMBath)=print(io,
    "HEOMBath(N=$(bath.coupling.basis.N), d=$(bath.coupling.basis.d), " *
    "exponentials=$(length(bath.coefficients)))")

_heom_basis(source::PIModel)=source.basis
_heom_basis(source::LiouvillianPlan)=source.basis
_heom_basis(source::CompiledPIModel)=source.plan.basis
function _heom_basis(source::MatrixFreeLiouvillian)
    source.plan isa LiouvillianPlan ? source.plan.basis : nothing
end
_heom_basis(::Any)=nothing

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
    HEOMPlan(system, baths; max_depth, basis=nothing, terminator=:none)

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

Indices with total depth larger than `max_depth` are identically zero. The
only implemented terminator is therefore `:none` (hard truncation); convergence
must be checked by increasing `max_depth`. Setup and application scale with
the PI dimension, never with ``d^N``.
"""
struct HEOMPlan{B,S,C,E,I,D,U,V,T}
    basis::B
    system::S
    coupling_blocks::C
    exponent_baths::E
    coefficients::V
    right_coefficients::V
    frequencies::V
    multiindices::I
    index::D
    upward::U
    downward::U
    decays::V
    max_depth::Int
    npi::Int
    tracevec::V
    Ttype::Type{T}
    autonomous::Bool
    terminator::Symbol
end

function HEOMPlan(source,baths;max_depth::Integer,basis=nothing,
                  terminator::Symbol=:none)
    max_depth>=0||throw(ArgumentError("max_depth must be nonnegative"))
    BigInt(max_depth)<=typemax(Int)||throw(ArgumentError(
        "max_depth must be representable as an Int"))
    terminator===:none||throw(ArgumentError(
        "only terminator=:none is implemented"))
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
    exponent_baths=Int[]
    coupling_blocks=Vector{Vector{Matrix{T}}}(undef,length(bath_tuple))
    for (bath_number,bath) in pairs(bath_tuple)
        coupling_blocks[bath_number]=_heom_coupling_blocks(
            selected_basis,bath.coupling,T)
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
    number_big<=typemax(Int)||throw(ArgumentError(
        "the requested HEOM hierarchy has $number_big ADOs, exceeding Int indexing"))
    npi=length(selected_basis)
    number_big*BigInt(npi)<=typemax(Int)||throw(ArgumentError(
        "the requested HEOM coordinate dimension exceeds Int indexing"))
    multiindices=_heom_multiindices(K,D)
    length(multiindices)==Int(number_big)||error(
        "internal HEOM hierarchy enumeration mismatch")
    lookup=Dict{Tuple,Int}(Tuple(index)=>position
                           for (position,index) in pairs(multiindices))
    upward=zeros(Int,length(multiindices),K)
    downward=zeros(Int,length(multiindices),K)
    candidate=zeros(Int,K)
    for (position,index) in pairs(multiindices),term in 1:K
        copyto!(candidate,index);candidate[term]+=1
        upward[position,term]=get(lookup,Tuple(candidate),0)
        copyto!(candidate,index)
        if index[term]>0
            candidate[term]-=1
            downward[position,term]=get(lookup,Tuple(candidate),0)
        end
    end
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
    tracevec=zeros(T,npi*length(multiindices))
    tracevec[1:npi].=_trace_vector(selected_basis,T)
    HEOMPlan{typeof(selected_basis),typeof(prepared),typeof(coupling_blocks),
             typeof(exponent_baths),typeof(multiindices),typeof(lookup),
             typeof(upward),typeof(coefficients),T}(
        selected_basis,prepared,coupling_blocks,exponent_baths,
        coefficients,right_coefficients,frequencies,multiindices,lookup,
        upward,downward,decays,
        D,npi,tracevec,T,isautonomous(prepared),terminator)
end

size(plan::HEOMPlan)=(length(plan.tracevec),length(plan.tracevec))
size(plan::HEOMPlan,index::Integer)=index in (1,2) ? length(plan.tracevec) : 1
eltype(plan::HEOMPlan)=plan.Ttype
isautonomous(plan::HEOMPlan)=plan.autonomous

show(io::IO,plan::HEOMPlan)=print(io,
    "HEOMPlan(N=$(plan.basis.N), d=$(plan.basis.d), " *
    "exponentials=$(length(plan.coefficients)), max_depth=$(plan.max_depth), " *
    "ADOs=$(length(plan.multiindices)), dimension=$(size(plan,1)), " *
    "autonomous=$(plan.autonomous))")

"""Return the number of auxiliary density operators retained by `plan`."""
heom_number_ados(plan::HEOMPlan)=length(plan.multiindices)

"""
    heom_multiindices(plan)

Return detached copies of all retained hierarchy multi-indices, ordered first
by total depth and then by ascending lexicographic occupation order.
"""
heom_multiindices(plan::HEOMPlan)=map(copy,plan.multiindices)

function _heom_system_workspace(system)
    if system isa LiouvillianPlan
        return LiouvillianWorkspace(system)
    elseif system isa MatrixFreeLiouvillian
        return system.plan isa LiouvillianPlan ? LiouvillianWorkspace(system.plan) : nothing
    elseif system isa CompiledPIModel
        return system.backend===:matrixfree ? LiouvillianWorkspace(system.plan) : nothing
    end
    nothing
end

"""
    HEOMWorkspace(plan)

Task-owned mutable storage for matrix-free HEOM generator application. A
workspace may be reused sequentially but not
concurrently. RK4 stage arrays live separately in
[`HEOMEvolutionWorkspace`](@ref), so spectral and steady-state calculations do
not retain five hierarchy-sized evolution vectors.
"""
struct HEOMWorkspace{T,P,S}
    plan::P
    system::S
    left::Vector{T}
    right::Vector{T}
end

function HEOMWorkspace(plan::HEOMPlan)
    T=plan.Ttype
    system_work=_heom_system_workspace(plan.system)
    HEOMWorkspace{T,typeof(plan),typeof(system_work)}(
        plan,system_work,zeros(T,plan.npi),zeros(T,plan.npi))
end

function _check_heom_workspace(work::HEOMWorkspace,plan::HEOMPlan)
    work.plan===plan||throw(ArgumentError(
        "HEOM workspace belongs to a different plan"))
    length(work.left)==plan.npi&&length(work.right)==plan.npi||
        throw(DimensionMismatch("HEOM workspace has incompatible dimensions"))
    eltype(work.left)===plan.Ttype||throw(ArgumentError(
        "HEOM workspace has an incompatible scalar type"))
    work
end

"""
    HEOMEvolutionWorkspace(plan)

Task-owned HEOM application scratch plus four RK4 stages and one temporary
hierarchy vector. Reuse it for repeated fixed-step evolution; do not share it
between concurrent tasks.
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
        application,zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n))
end

function _check_heom_evolution_workspace(work::HEOMEvolutionWorkspace,
                                         plan::HEOMPlan)
    _check_heom_workspace(work.application,plan)
    all(vector->length(vector)==size(plan,1),
        (work.temporary,work.k1,work.k2,work.k3,work.k4))||
        throw(DimensionMismatch("HEOM evolution workspace has incompatible dimensions"))
    eltype(work.temporary)===plan.Ttype||throw(ArgumentError(
        "HEOM evolution workspace has an incompatible scalar type"))
    work
end

function _heom_system_apply!(destination,system,source,time,parameters,work)
    if work===nothing
        if system isa LiouvillianPlan
            return apply!(destination,system,source,time,parameters)
        elseif system isa MatrixFreeLiouvillian||system isa CompiledPIModel
            return apply!(destination,system,source,time,parameters)
        end
        return mul!(destination,system,source)
    end
    apply!(destination,system,source,time,parameters,work)
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
    npi=plan.npi;K=length(plan.coefficients)
    for ado in eachindex(plan.multiindices)
        range=(ado-1)*npi+1:ado*npi
        output=view(destination,range);input=view(source,range)
        _heom_system_apply!(output,plan.system,input,time,parameters,work.system)
        decay=plan.decays[ado]
        if !iszero(decay)
            @inbounds @simd for coordinate in eachindex(output,input)
                output[coordinate]-=decay*input[coordinate]
            end
        end
        occupations=plan.multiindices[ado]
        for term in 1:K
            bath=plan.exponent_baths[term]
            upper=plan.upward[ado,term]
            if !iszero(upper)
                upper_range=(upper-1)*npi+1:upper*npi
                _heom_coupling_actions!(work.left,work.right,plan,
                                        view(source,upper_range),bath)
                @inbounds @simd for coordinate in eachindex(output)
                    output[coordinate]+=-im*(work.left[coordinate]-work.right[coordinate])
                end
            end
            occupation=occupations[term]
            lower=plan.downward[ado,term]
            if occupation>0
                iszero(lower)&&error("internal HEOM downward-neighbor mismatch")
                lower_range=(lower-1)*npi+1:lower*npi
                _heom_coupling_actions!(work.left,work.right,plan,
                                        view(source,lower_range),bath)
                coefficient=plan.coefficients[term]
                right_coefficient=plan.right_coefficients[term]
                @inbounds @simd for coordinate in eachindex(output)
                    output[coordinate]+=-im*occupation*(
                        coefficient*work.left[coordinate]-
                        right_coefficient*work.right[coordinate])
                end
            end
        end
    end
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

"""
    HEOMState(plan, data)

State of a prepared hierarchy in ADO-major order. Each ADO stores one complete
PI coefficient vector. Construction copies `data` and rejects scalar
narrowing; it does not normalize or repair the reduced state.
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
    values=plan.Ttype.(data)
    HEOMState{plan.Ttype,typeof(plan)}(plan,values,Val(:owned))
end

Base.copy(state::HEOMState)=HEOMState(state.plan,state.data)
Base.length(state::HEOMState)=length(state.data)
Base.eltype(state::HEOMState)=eltype(state.data)

function show(io::IO,state::HEOMState)
    root_trace=dot(view(state.plan.tracevec,1:state.plan.npi),
                   view(state.data,1:state.plan.npi))
    print(io,"HEOMState(ADOs=$(heom_number_ados(state.plan)), " *
             "dimension=$(length(state.data)), root_trace=$root_trace)")
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
    PIOperator(state.plan.basis,Complex{R}.(view(state.data,range)))
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

"""
    heom_evolve!(destination, plan, source, tspan;
                 steps=256, parameters=nothing, workspace=nothing)

Propagate a hierarchy with preallocated fixed-step RK4. `destination` may
alias `source`; one `HEOMEvolutionWorkspace` may be reused sequentially. Increase
`steps` and `plan.max_depth` independently to check integration and hierarchy
truncation errors.
"""
function heom_evolve!(destination::AbstractVector,plan::HEOMPlan,
                      source::AbstractVector,tspan;steps::Integer=256,
                      parameters=nothing,workspace=nothing)
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
    work=workspace===nothing ? HEOMEvolutionWorkspace(plan) :
                              _check_heom_evolution_workspace(workspace,plan)
    R=_real_float_type(plan.Ttype)
    t0=_heom_checked_time(R,first(tspan),"initial time")
    t1=_heom_checked_time(R,last(tspan),"final time")
    step_count=Int(steps)
    step_count_R=_heom_checked_time(R,step_count,"step count")
    interval=t1-t0
    isfinite(interval)||throw(ArgumentError(
        "HEOM evolution time interval is not finite in $R"))
    step=interval/step_count_R
    !iszero(interval)&&iszero(step)&&throw(ArgumentError(
        "HEOM evolution step underflows in $R; use fewer steps or wider precision"))
    destination===source||copyto!(destination,source)
    for step_index in 0:step_count-1
        time=t0+R(step_index)*step
        midpoint=time+step/2
        endpoint=step_index==step_count-1 ? t1 : time+step
        apply!(work.k1,plan,destination,time,parameters,work.application)
        @. work.temporary=destination+(step/2)*work.k1
        apply!(work.k2,plan,work.temporary,midpoint,parameters,work.application)
        @. work.temporary=destination+(step/2)*work.k2
        apply!(work.k3,plan,work.temporary,midpoint,parameters,work.application)
        @. work.temporary=destination+step*work.k3
        apply!(work.k4,plan,work.temporary,endpoint,parameters,work.application)
        @. destination=destination+(step/6)*(
            work.k1+2work.k2+2work.k3+work.k4)
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
                        steps_per_interval=64, parameters=nothing)

Return saved hierarchy states at ordered `times`, reusing one RK4 workspace.
`initial` may be a `PIState` (factorized hierarchy) or an `HEOMState`.
"""
function heom_time_evolution(plan::HEOMPlan,initial,times;
                             steps_per_interval::Integer=64,parameters=nothing)
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
            steps=steps_per_interval,parameters,workspace)
        push!(output,copy(state))
    end
    output
end

function _heom_prefix_plan(template::HEOMPlan,depth::Int)
    depth==template.max_depth&&return template
    0<=depth<template.max_depth||throw(ArgumentError(
        "prefix depth must lie below the template depth"))
    K=length(template.coefficients)
    count=Int(exact_binomial(BigInt(K)+BigInt(depth),BigInt(depth)))
    multiindices=map(copy,view(template.multiindices,1:count))
    lookup=Dict{Tuple,Int}(Tuple(index)=>position
                           for (position,index) in pairs(multiindices))
    upward=Matrix(view(template.upward,1:count,:))
    @inbounds for index in eachindex(upward)
        upward[index]>count&&(upward[index]=0)
    end
    downward=Matrix(view(template.downward,1:count,:))
    decays=copy(view(template.decays,1:count))
    tracevec=zeros(template.Ttype,template.npi*count)
    tracevec[1:template.npi].=view(template.tracevec,1:template.npi)
    HEOMPlan{typeof(template.basis),typeof(template.system),
             typeof(template.coupling_blocks),typeof(template.exponent_baths),
             typeof(multiindices),typeof(lookup),typeof(upward),
             typeof(template.coefficients),template.Ttype}(
        template.basis,template.system,template.coupling_blocks,
        template.exponent_baths,template.coefficients,
        template.right_coefficients,template.frequencies,
        multiindices,lookup,upward,downward,decays,depth,template.npi,
        tracevec,template.Ttype,template.autonomous,template.terminator)
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
         template_max_depth=template.max_depth,terminator=template.terminator)
    end
    hierarchy_depth_convergence(evaluate_depth,int_depths;
        estimate=result->result.state,
        diagnostics=identity,atol=atolR,rtol,consecutive,
        require_convergence)
end


"""
    heom_depth_convergence(system, baths, initial, tspan;
                           depths, steps=256, parameters=nothing,
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
        atol::Real=0,rtol=nothing,consecutive::Integer=2,
        require_convergence::Bool=false)
    int_depths=_validated_heom_depths(depths)
    prepared=_prepare_heom_system(source)
    template=HEOMPlan(prepared,baths;max_depth=last(int_depths),
                      basis=initial.basis)
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
matrix. In parallel hot loops, call `apply!` with one explicit
`HEOMWorkspace` per task instead.
"""
function heom_liouvillian(plan::HEOMPlan)
    workspace=HEOMWorkspace(plan)
    action! = (destination,source,time,parameters)->
        apply!(destination,plan,source,time,parameters,workspace)
    MatrixFreeLiouvillian(size(plan,1),action!,plan.Ttype,copy(plan.tracevec);
                          autonomous=plan.autonomous,plan,workspace)
end

"""
    heom_steady_state(plan; initial_state=nothing, return_info=false, kwargs...)

Compute the trace-fixed stationary hierarchy with the existing restarted
matrix-free GMRES solver. `kwargs` are forwarded to `krylov_steady_state`.
The physical stationary density operator is `heom_reduced_state(result)`.
"""
function heom_steady_state(plan::HEOMPlan;initial_state=nothing,
                           return_info::Bool=false,kwargs...)
    _require_autonomous(plan,"HEOM steady-state solving")
    initial=initial_state isa HEOMState ? begin
        initial_state.plan===plan||throw(ArgumentError(
            "initial HEOM state belongs to a different plan"))
        initial_state.data
    end : initial_state isa PIState ? heom_initial_state(plan,initial_state).data :
          initial_state
    result=krylov_steady_state(heom_liouvillian(plan);
        trace_vector=plan.tracevec,initial_state=initial,
        return_info=return_info,kwargs...)
    if return_info
        return merge(result,(state=HEOMState(plan,result.state),))
    end
    HEOMState(plan,result)
end
