# Pure states in the sole fully symmetric Schur sector.
#
# This backend is intentionally distinct from `WeakPIPseudoKet`: relative
# phases are physical here because there is exactly one Hilbert-space irrep,
# with symmetric-group multiplicity one.  Include this file after
# `local_factor_trace.jl` and `liouvillian.jl`.

"""
    SymmetricKet(basis, data; atol=100eps(T), rtol=100eps(T))

Normalized physical ket in the fully symmetric irrep `(N,0,...,0)`.  `basis`
must retain that sector and no other sector.  The amplitudes follow the
package's GT-pattern order and have length
`binomial(N+d-1,d-1)`, rather than `d^N`.

Construction copies and validates `data`; it never normalizes it.  Use one
`SymmetricKetHamiltonianPlan` as immutable shared data and one
`SymmetricKetWorkspace` per task for repeated evolution. Amplitude storage is
mutable for in-place kernels; public analysis and conversion routines validate
the norm by default, and evolution reports excessive norm drift rather than
renormalizing it.
"""
struct SymmetricKet{T<:AbstractFloat,B<:PIBasis}
    basis::B
    data::Vector{Complex{T}}
    function SymmetricKet(
            basis::B,data::AbstractVector{Complex{T}};
            atol=nothing,rtol=nothing) where {T<:AbstractFloat,B<:PIBasis}
        _symmetric_ket_check_basis(basis)
        Base.require_one_based_indexing(data)
        length(data)==symmetric_ket_dimension(basis)||throw(DimensionMismatch(
            "symmetric-ket data have length $(length(data)); expected " *
            "$(symmetric_ket_dimension(basis))"))
        if T===BigFloat
            bounds=_symmetric_ket_bigfloat_bounds(
                data,"symmetric-ket amplitudes")
            if precision(BigFloat)!=bounds[1]
                return setprecision(BigFloat,bounds[1]) do
                    SymmetricKet(basis,data;atol,rtol)
                end
            end
        end
        actual_atol=atol===nothing ? T(100)*eps(T) : atol
        actual_rtol=rtol===nothing ? T(100)*eps(T) : rtol
        actual_atol isa Real&&actual_rtol isa Real||throw(ArgumentError(
            "symmetric-ket tolerances must be real numbers or nothing"))
        atolT,rtolT=_symmetric_ket_tolerances(
            T,actual_atol,actual_rtol)
        all(isfinite,data)||throw(ArgumentError(
            "symmetric-ket amplitudes must be finite"))
        value=norm(data)
        isfinite(value)||throw(ArgumentError(
            "symmetric-ket norm is nonfinite"))
        abs(value-one(T))<=atolT+rtolT||throw(ArgumentError(
            "symmetric ket must have unit norm; norm=$value"))
        new{T,B}(basis,collect(data))
    end
    # State propagation may accumulate a documented integration error.  A
    # copy must preserve that numerical state verbatim rather than applying a
    # second, potentially different validation tolerance.
    function SymmetricKet(
            basis::B,data::Vector{Complex{T}},::Val{:unchecked}) where
            {T<:AbstractFloat,B<:PIBasis}
        new{T,B}(basis,data)
    end
end

function SymmetricKet(
        basis::PIBasis,data::AbstractVector{S};kwargs...) where S<:Number
    T=_real_float_type(S)
    if T===BigFloat
        bounds=_symmetric_ket_bigfloat_bounds(
            data,"symmetric-ket amplitudes")
        if precision(BigFloat)!=bounds[1]
            return setprecision(BigFloat,bounds[1]) do
                SymmetricKet(basis,data;kwargs...)
            end
        end
    end
    converted=Complex{T}[
        _symmetric_ket_checked_complex(T,value,"symmetric-ket amplitude")
        for value in data]
    SymmetricKet(basis,converted;kwargs...)
end

function _symmetric_ket_check_basis(basis::PIBasis)
    _has_single_fully_symmetric_sector(basis)||throw(ArgumentError(
        "a physical SymmetricKet requires a PIBasis retaining only the " *
        "fully symmetric sector ($(basis.N),0,...)"))
    basis
end

function _symmetric_ket_tolerance(::Type{T},value,label) where
        T<:AbstractFloat
    converted=_checked_prepared_real(value,T,label)
    converted>=zero(T)||throw(ArgumentError(
        "$label must be nonnegative"))
    converted
end

function _symmetric_ket_tolerances(::Type{T},atol,rtol) where
        T<:AbstractFloat
    (_symmetric_ket_tolerance(T,atol,"atol"),
     _symmetric_ket_tolerance(T,rtol,"rtol"))
end

@inline function _symmetric_ket_validate_memory_budget(memory_budget)
    memory_budget isa Real&&!(memory_budget isa Bool)||throw(ArgumentError(
        "memory_budget must be a real number of bytes, not a Bool"))
    isnan(memory_budget)&&throw(ArgumentError("memory_budget cannot be NaN"))
    memory_budget>=0||throw(ArgumentError(
        "memory_budget must be nonnegative"))
    nothing
end

"""Return the physical symmetric-ket dimension, rejecting other bases."""
function symmetric_ket_dimension(basis::PIBasis)
    _symmetric_ket_check_basis(basis)
    length(only(basis.patterns))
end

copy(state::SymmetricKet)=SymmetricKet(
    state.basis,copy(state.data),Val(:unchecked))
eltype(state::SymmetricKet)=eltype(state.data)
length(state::SymmetricKet)=length(state.data)
show(io::IO,state::SymmetricKet)=print(io,
    "SymmetricKet(N=$(state.basis.N), d=$(state.basis.d), " *
    "dimension=$(length(state)))")

"""
    validate_symmetric_ket(state; atol=100eps(T), rtol=100eps(T))

Validate finiteness and unit norm without modifying `state`.
"""
function validate_symmetric_ket(
        state::SymmetricKet{T};
        atol=nothing,rtol=nothing) where T<:AbstractFloat
    _symmetric_ket_check_basis(state.basis)
    if T===BigFloat
        bounds=_symmetric_ket_bigfloat_bounds(
            state.data,"symmetric-ket amplitudes")
        if precision(BigFloat)!=bounds[1]
            return setprecision(BigFloat,bounds[1]) do
                validate_symmetric_ket(state;atol,rtol)
            end
        end
    end
    actual_atol=atol===nothing ? T(100)*eps(T) : atol
    actual_rtol=rtol===nothing ? T(100)*eps(T) : rtol
    actual_atol isa Real&&actual_rtol isa Real||throw(ArgumentError(
        "symmetric-ket tolerances must be real numbers or nothing"))
    atolT,rtolT=_symmetric_ket_tolerances(
        T,actual_atol,actual_rtol)
    all(isfinite,state.data)||throw(ArgumentError(
        "symmetric-ket amplitudes must be finite"))
    value=norm(state.data)
    isfinite(value)&&abs(value-one(T))<=atolT+rtolT||
        throw(ArgumentError(
            "symmetric ket must have unit norm; norm=$value"))
    state
end

"""
    symmetric_occupation_ket(basis, occupations; T=Float64)

Construct a normalized symmetric occupation ket directly from one-based local
level counts.  Its exact rank costs `O(d)` and no complete block is scanned.
"""
function symmetric_occupation_ket(
        basis::PIBasis,occupations;
        T::Type{<:AbstractFloat}=Float64)
    scalar_type=_concrete_pi_real_type(T)
    _symmetric_ket_check_basis(basis)
    counts=_symmetric_occupation_counts(basis,occupations)
    index=_symmetric_occupation_index(counts)
    patterns=only(basis.patterns)
    1<=index<=length(patterns)&&content(patterns[index])==counts||error(
        "internal error: symmetric occupation rank disagrees with GT ordering")
    data=zeros(Complex{scalar_type},length(patterns))
    data[index]=one(scalar_type)
    SymmetricKet(basis,data;atol=zero(scalar_type),rtol=zero(scalar_type))
end

function _symmetric_product_ket_impl(
        basis::PIBasis,psi::AbstractVector,atol::Real,rtol::Real,
        ::Type{T}) where T<:AbstractFloat
    _symmetric_ket_check_basis(basis)
    length(psi)==basis.d||throw(DimensionMismatch(
        "one-particle ket must have length $(basis.d)"))
    atolT,rtolT=_symmetric_ket_tolerances(T,atol,rtol)
    W=_iid_amplitude_work_type(T)
    psi_norm=norm(Complex{W}.(psi))
    isapprox(psi_norm,one(W);atol=W(atolT),rtol=W(rtolT))||throw(ArgumentError(
        "psi must be normalized; norm evaluated in $W is $psi_norm"))
    tensor_trace=abs2(psi_norm^basis.N)
    tensor_tolerance=W(atolT)+W(rtolT)
    isfinite(tensor_trace)&&abs(tensor_trace-one(W))<=tensor_tolerance||
        throw(ArgumentError(
            "the accepted one-particle normalization error is amplified at " *
            "N=$(basis.N): the tensor-power trace would be $tensor_trace " *
            "(tolerance $tensor_tolerance); supply a more accurately " *
            "normalized vector or use a wider scalar type"))

    magnitudes=W[abs(value) for value in psi]
    tails=zeros(W,basis.d+1)
    for level in basis.d:-1:1
        tails[level]=hypot(magnitudes[level],tails[level+1])
    end
    phases=Complex{W}[iszero(magnitudes[level]) ? one(Complex{W}) :
        Complex{W}(psi[level])/magnitudes[level] for level in 1:basis.d]
    overall_scale=tails[1]^basis.N
    isfinite(overall_scale)&&!iszero(overall_scale)||throw(ArgumentError(
        "the tensor-power amplitude scale is outside the nonzero finite " *
        "range of $W; use a wider scalar type"))

    tables=Dict{Tuple{Int,Int},Vector{W}}()
    patterns=only(basis.patterns)
    data=zeros(Complex{T},length(patterns))
    for (index,pattern) in pairs(patterns)
        occupations=content(pattern)
        any(level->iszero(magnitudes[level])&&occupations[level]>0,
            1:basis.d)&&continue
        remaining=basis.N
        amplitude=Complex{W}(overall_scale)
        for level in 1:basis.d-1
            iszero(remaining)&&break
            tail=tails[level]
            iszero(tail)&&begin
                amplitude=zero(amplitude)
                break
            end
            table=get!(tables,(level,remaining)) do
                _normalized_binomial_amplitudes(
                    remaining,magnitudes[level]/tail,
                    tails[level+1]/tail)
            end
            amplitude*=table[occupations[level]+1]
            remaining-=occupations[level]
        end
        for level in 1:basis.d
            occupations[level]>0&&
                (amplitude*=phases[level]^occupations[level])
        end
        converted=Complex{T}(amplitude)
        isfinite(converted)||throw(ArgumentError(
            "an occupation amplitude is not finite in $T; use a wider " *
            "scalar type"))
        !iszero(amplitude)&&iszero(converted)&&throw(ArgumentError(
            "a nonzero occupation amplitude underflows in $T; use a wider " *
            "scalar type"))
        data[index]=converted
    end
    SymmetricKet(basis,data;atol=atolT,rtol=rtolT)
end

"""
    symmetric_product_ket(basis, psi; atol=0, rtol=sqrt(eps(T)))

Construct the physical ket `psi^tensor N` in normalized symmetric occupation
coordinates.  Conditional-binomial recurrences avoid explicit multinomial
coefficients and remain polynomial at fixed local dimension.
"""
function symmetric_product_ket(
        basis::PIBasis,psi::AbstractVector;
        atol::Real=0,
        rtol=nothing)
    T=_real_float_type(eltype(psi))
    if T===BigFloat
        input_precision=_symmetric_ket_bigfloat_bounds(
            psi,"one-particle ket")[1]
        return setprecision(BigFloat,input_precision) do
            actual_rtol=rtol===nothing ? sqrt(eps(T)) : rtol
            actual_rtol isa Real||throw(ArgumentError(
                "rtol must be a real number or nothing"))
            _symmetric_product_ket_impl(basis,psi,atol,actual_rtol,T)
        end
    end
    actual_rtol=rtol===nothing ? sqrt(eps(T)) : rtol
    actual_rtol isa Real||throw(ArgumentError(
        "rtol must be a real number or nothing"))
    _symmetric_product_ket_impl(basis,psi,atol,actual_rtol,T)
end

"""
    symmetric_ket_density!(rho, ket; check=true, atol, rtol,
                           memory_budget=512*1024^2)
    symmetric_ket_density(ket; check=true, atol, rtol,
                          memory_budget=512*1024^2)

Form the pure `PIState` represented by `ket`.  This explicit conversion costs
`O(g^2)` storage for `g=length(ket)`; ket-native Hamiltonian evolution and
local-factor tracing do not perform it. The allocating form enforces the
package's default memory budget before creating the density coordinates;
pass `memory_budget=Inf` only as an explicit opt-out.
"""
function symmetric_ket_density!(rho::PIState,ket::SymmetricKet;
        check::Bool=true,atol=nothing,rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _symmetric_ket_validate_memory_budget(memory_budget)
    rho.basis===ket.basis||throw(ArgumentError(
        "density destination and symmetric ket use different PIBasis objects"))
    eltype(rho.data)===eltype(ket.data)||throw(ArgumentError(
        "density destination and symmetric ket must have the same scalar type"))
    T=_real_float_type(eltype(ket.data))
    if T===BigFloat
        ket_bounds=_symmetric_ket_bigfloat_bounds(
            ket.data,"symmetric-ket amplitudes")
        rho_bounds=_symmetric_ket_bigfloat_bounds(
            rho.data,"density destination")
        rho_bounds==ket_bounds||throw(ArgumentError(
            "density destination has precision range $rho_bounds, but the " *
            "symmetric ket has precision range $ket_bounds"))
        if precision(BigFloat)!=ket_bounds[1]
            return setprecision(BigFloat,ket_bounds[1]) do
                symmetric_ket_density!(
                    rho,ket;check,atol,rtol,memory_budget)
            end
        end
    end
    actual_atol=atol===nothing ? _analysis_atol(rho) : atol
    actual_rtol=rtol===nothing ? _state_rtol(rho) : rtol
    actual_atol isa Real&&actual_rtol isa Real||throw(ArgumentError(
        "symmetric-ket density tolerances must be real numbers or nothing"))
    check&&validate_symmetric_ket(
        ket;atol=actual_atol,rtol=actual_rtol)
    dimension=length(ket)
    @inbounds for column in 1:dimension,row in 1:dimension
        rho.data[row+(column-1)*dimension]=row==column ?
            _symmetric_ket_checked_abs2(
                ket.data[row],"symmetric-ket density diagonal",
                memory_budget) :
            _symmetric_ket_checked_triple_product(
                one(eltype(ket.data)),ket.data[row],conj(ket.data[column]),
                "symmetric-ket density entry",memory_budget)
    end
    if T===BigFloat
        bounds=_symmetric_ket_bigfloat_bounds(rho.data,"density destination")
        bounds[1]==precision(BigFloat)||error(
            "internal error: symmetric-ket density changed BigFloat precision")
    end
    rho
end

"""
    symmetric_ket_density(ket; memory_budget=512*1024^2)

Allocating wrapper for [`symmetric_ket_density!`](@ref). The quadratic
PI-density allocation is guarded by `memory_budget`; ket-native evolution,
expectations, and local-factor tracing avoid this conversion.
"""
function symmetric_ket_density(ket::SymmetricKet{T};
        check::Bool=true,atol=nothing,rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where T<:AbstractFloat
    if T===BigFloat
        bounds=_symmetric_ket_bigfloat_bounds(
            ket.data,"symmetric-ket amplitudes")
        if precision(BigFloat)!=bounds[1]
            return setprecision(BigFloat,bounds[1]) do
                symmetric_ket_density(
                    ket;check,atol,rtol,memory_budget)
            end
        end
    end
    actual_atol=atol===nothing ? T(100)*eps(T) : atol
    actual_rtol=rtol===nothing ? T(100)*eps(T) : rtol
    actual_atol isa Real&&actual_rtol isa Real||throw(ArgumentError(
        "symmetric-ket density tolerances must be real numbers or nothing"))
    check&&validate_symmetric_ket(
        ket;atol=actual_atol,rtol=actual_rtol)
    # The public zero-state constructor defensively copies its input vector,
    # so account both simultaneous g^2 arrays before either is allocated.
    peak=_performance_array_bytes(
        length(ket.basis),Complex{T},0;linear_arrays=2)
    _require_performance_budget(
        "symmetric-ket density conversion",peak,memory_budget;guidance=
        "Use ket-native expectations or local_factor_trace for large runs.")
    rho=PIState(ket.basis;T)
    symmetric_ket_density!(
        rho,ket;check=false,atol=actual_atol,rtol=actual_rtol,memory_budget)
end

"""
    symmetric_ket(rho; atol, rtol, memory_budget=512*1024^2)

Recover a physical symmetric ket from a pure state in the sole symmetric
sector.  Mixed or lower-rank states are rejected; the global phase is fixed
by making the largest-amplitude component real and nonnegative.
"""
function symmetric_ket(
        rho::PIState{T};
        atol=nothing,rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where T<:AbstractFloat
    _symmetric_ket_check_basis(rho.basis)
    if T===BigFloat
        bounds=_symmetric_ket_bigfloat_bounds(
            rho.data,"symmetric density state")
        if precision(BigFloat)!=bounds[1]
            return setprecision(BigFloat,bounds[1]) do
                symmetric_ket(rho;atol,rtol,memory_budget)
            end
        end
    end
    actual_atol=atol===nothing ? _analysis_atol(rho) : atol
    actual_rtol=rtol===nothing ? _state_rtol(rho) : rtol
    actual_atol isa Real&&actual_rtol isa Real||throw(ArgumentError(
        "symmetric-ket conversion tolerances must be real numbers or nothing"))
    atolT,rtolT=_symmetric_ket_tolerances(
        T,actual_atol,actual_rtol)
    block=coefficient_block(rho,only(rho.basis.sectors))
    all(isfinite,block)||throw(ArgumentError(
        "symmetric-ket conversion requires finite density data"))
    trace_value=trace(rho)
    trace_tolerance=atolT+rtolT
    abs(trace_value-one(trace_value))<=trace_tolerance||throw(ArgumentError(
        "state is not trace one within tolerance: trace=$trace_value"))
    hermiticity_error=zero(T)
    hermiticity_scale=zero(T)
    @inbounds for row in axes(block,1)
        error_row=zero(T)
        scale_row=zero(T)
        for column in axes(block,2)
            value=block[row,column]
            error_row+=abs(value-conj(block[column,row]))
            scale_row+=abs(value)
        end
        hermiticity_error=max(hermiticity_error,error_row)
        hermiticity_scale=max(hermiticity_scale,scale_row)
    end
    hermiticity_error<=atolT+rtolT*hermiticity_scale||throw(ArgumentError(
        "state is not Hermitian within tolerance: error=$hermiticity_error"))
    n=size(block,1)
    peak=_performance_array_bytes(n,Complex{T},7;linear_arrays=6)
    _require_performance_budget(
        "symmetric-ket rank-one conversion",peak,memory_budget;guidance=
        "Retain the ket during pure-state evolution or raise the explicit budget.")
    hermitian_block=Matrix{Complex{T}}(undef,n,n)
    @inbounds for column in axes(block,2),row in axes(block,1)
        hermitian_block[row,column]=(
            block[row,column]+conj(block[column,row]))/T(2)
    end
    eig=_hermitian_eigen(Hermitian(hermitian_block);
        operation="symmetric-ket conversion")
    values=eig.values
    scale=maximum(abs,values;init=zero(T))
    tolerance=atolT+rtolT*max(scale,one(T))
    largest=last(values)
    abs(largest-one(T))<=tolerance||throw(ArgumentError(
        "state is not a normalized rank-one symmetric state: largest " *
        "eigenvalue=$largest"))
    length(values)>1&&maximum(abs,view(values,1:length(values)-1))>tolerance&&
        throw(ArgumentError(
            "state is mixed and cannot be represented by one SymmetricKet"))
    data=Vector{Complex{T}}(view(eig.vectors,:,length(values)))
    pivot=firstindex(data)
    pivot_magnitude=abs(data[pivot])
    @inbounds for index in Iterators.drop(eachindex(data),1)
        magnitude=abs(data[index])
        if magnitude>pivot_magnitude
            pivot=index
            pivot_magnitude=magnitude
        end
    end
    !iszero(data[pivot])&&(data .*= conj(data[pivot])/abs(data[pivot]))
    ket=SymmetricKet(rho.basis,data;
        atol=max(atolT,T(100)*eps(T)),
        rtol=max(rtolT,T(100)*eps(T)))
    # Validate reconstruction without allocating a second g^2 PIState.
    residual_squared=zero(T)
    @inbounds for column in axes(block,2),row in axes(block,1)
        difference=ket.data[row]*conj(ket.data[column])-block[row,column]
        residual_squared+=abs2(difference)
    end
    residual=sqrt(residual_squared)
    residual<=atolT+rtolT*max(norm(rho.data),one(T))||throw(ArgumentError(
        "state is not representable by one SymmetricKet; reconstruction " *
        "residual=$residual"))
    ket
end

function _symmetric_ket_checked_complex(::Type{R},value,label) where
        R<:AbstractFloat
    value isa Number&&isfinite(value)||throw(ArgumentError(
        "$label must contain only finite numbers"))
    converted=try
        R===BigFloat ? complex(
            BigFloat(real(value);precision=precision(BigFloat)),
            BigFloat(imag(value);precision=precision(BigFloat))) :
            Complex{R}(value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("$label is not representable in Complex{$R}"))
    end
    isfinite(converted)||throw(ArgumentError(
        "$label overflows in Complex{$R}; use a wider scalar type"))
    for (source,destination) in
            ((real(value),real(converted)),(imag(value),imag(converted)))
        !iszero(source)&&iszero(destination)&&throw(ArgumentError(
            "$label has a nonzero component that underflows in Complex{$R}; " *
            "use a wider scalar type"))
        if (R===Float16||R===Float32||R===Float64)&&!iszero(source)
            magnitude=abs(destination)
            if magnitude in (nextfloat(zero(R)),floatmax(R))
                exact_magnitude=abs(Rational{BigInt}(source))
                Rational{BigInt}(nextfloat(zero(R)))<=exact_magnitude<=
                    Rational{BigInt}(floatmax(R))||throw(ArgumentError(
                        "$label has a component outside the nonzero finite " *
                        "range of Complex{$R}; use a wider scalar type"))
            end
        end
    end
    converted
end

function _symmetric_ket_checked_matrix(
        matrix::SparseMatrixCSC,::Type{R},label) where R<:AbstractFloat
    values=Complex{R}[_symmetric_ket_checked_complex(R,value,label)
        for value in nonzeros(matrix)]
    SparseMatrixCSC(size(matrix,1),size(matrix,2),copy(matrix.colptr),
        copy(rowvals(matrix)),values)
end

function _symmetric_ket_checked_matrix(
        matrix::Diagonal,::Type{R},label) where R<:AbstractFloat
    Diagonal(Complex{R}[
        _symmetric_ket_checked_complex(R,value,label) for value in matrix.diag])
end

function _symmetric_ket_checked_matrix(
        matrix::AbstractMatrix,::Type{R},label) where R<:AbstractFloat
    output=Matrix{Complex{R}}(undef,size(matrix))
    for index in eachindex(matrix)
        output[index]=_symmetric_ket_checked_complex(R,matrix[index],label)
    end
    output
end

function _symmetric_ket_check_hamiltonian(matrix,atol,rtol)
    R=_real_float_type(eltype(matrix))
    atolR,rtolR=_symmetric_ket_tolerances(R,atol,rtol)
    error=zero(R);scale=zero(R)
    if matrix isa Diagonal
        @inbounds for value in matrix.diag
            scale=max(scale,abs(value))
            error=max(error,abs(value-conj(value)))
        end
    elseif matrix isa SparseMatrixCSC
        row_error=zeros(R,size(matrix,1))
        row_scale=zeros(R,size(matrix,1))
        @inbounds for column in axes(matrix,2)
            for pointer in nzrange(matrix,column)
                row=matrix.rowval[pointer]
                value=matrix.nzval[pointer]
                row_scale[row]+=abs(value)
                row_error[row]+=abs(value-conj(matrix[column,row]))
            end
        end
        error=maximum(row_error;init=zero(R))
        scale=maximum(row_scale;init=zero(R))
    else
        @inbounds for row in axes(matrix,1)
            row_error=zero(R);row_scale=zero(R)
            for column in axes(matrix,2)
                value=matrix[row,column]
                row_scale+=abs(value)
                row_error+=abs(value-conj(matrix[column,row]))
            end
            error=max(error,row_error);scale=max(scale,row_scale)
        end
    end
    error<=atolR+rtolR*scale||throw(ArgumentError(
        "symmetric-ket Hamiltonian must be Hermitian: error=$error"))
    matrix
end

"""
    SymmetricKetHamiltonianPlan(basis, H;
        representation=:local, rate=1, hbar=1, T=nothing,
        check=true, atol=0, rtol=0, memory_budget=512*1024^2)
    SymmetricKetHamiltonianPlan(H::PIOperator; rate=1, hbar=1, ...)

Prepare the Schrödinger action `dpsi/dt = -im*(rate/hbar)*H*psi` in the sole
fully symmetric sector.  With `representation=:local`, `H` is a `d`-by-`d`
one-particle Hamiltonian and is lifted directly onto exact sparse occupation
support.  With `representation=:block`, it is an already-lifted physical
Schur block.  A `PIOperator` constructor reuses its sole physical block.

A callable `rate(time, parameters)` is checked on every application.  Fixed
or callable operator-valued schedules are deliberately unsupported: prepare a
new immutable plan, or use density-matrix `PIModel` dynamics, rather than
placing mutable operator storage in a shared plan.
"""
struct SymmetricKetHamiltonianPlan{R<:AbstractFloat,B<:PIBasis,H,S,Q}
    basis::B
    hamiltonian::H
    scale::S
    precision_bits::Int
    rounding_mode::Q
end

eltype(plan::SymmetricKetHamiltonianPlan)=eltype(plan.hamiltonian)
size(plan::SymmetricKetHamiltonianPlan)=size(plan.hamiltonian)
size(plan::SymmetricKetHamiltonianPlan,index::Integer)=
    size(plan.hamiltonian,index)
show(io::IO,plan::SymmetricKetHamiltonianPlan{R}) where R=print(io,
    "SymmetricKetHamiltonianPlan(N=$(plan.basis.N), d=$(plan.basis.d), " *
    "dimension=$(size(plan,1)), scalar_type=$R, " *
    "storage=$(typeof(plan.hamiltonian)))")

_symmetric_ket_matrix_values(matrix::SparseMatrixCSC)=nonzeros(matrix)
_symmetric_ket_matrix_values(matrix::Diagonal)=matrix.diag
_symmetric_ket_matrix_values(matrix)=matrix

function _symmetric_ket_bigfloat_bounds(values,label)
    bounds=_local_factor_precision_bounds(values)
    bounds[1]==bounds[2]||throw(ArgumentError(
        "$label BigFloat storage has mixed precision range $bounds"))
    bounds
end

function _symmetric_ket_matrix_storage_bytes(matrix,::Type{R}) where
        R<:AbstractFloat
    if matrix isa SparseMatrixCSC
        pointers=BigInt(size(matrix,2)+1)
        entries=BigInt(nnz(matrix))
        return _performance_entries_bytes(entries,Complex{R})+
            BigInt(sizeof(Int))*(pointers+entries)+256
    elseif matrix isa Diagonal
        return _performance_entries_bytes(
            BigInt(length(matrix.diag)),Complex{R})
    end
    _performance_entries_bytes(BigInt(length(matrix)),Complex{R})
end

struct _SymmetricKetHamiltonianRate{S,H}
    schedule::S
    hbar::H
end

function _symmetric_ket_plan_precision(
        operator_type,rate,hbar,requested_type)
    R=_real_float_type(operator_type)
    if requested_type!==nothing
        requested_type isa Type&&requested_type<:AbstractFloat&&
            isconcretetype(requested_type)||throw(ArgumentError(
                "T must be a concrete AbstractFloat type or nothing"))
        promote_type(requested_type,R)===requested_type||throw(ArgumentError(
            "T=$requested_type would narrow Hamiltonian scalar type $R"))
        R=requested_type
    end
    if rate isa Number
        return _prepared_rate_precision(R,rate,hbar)
    end
    _check_finite_real(hbar,"Hamiltonian hbar";nonzero=true)
    _prepared_rate_precision(_rate_schedule_precision(rate,R),hbar)
end

function _symmetric_ket_target_bigfloat_precision(
        H,rate,hbar,requested_type)
    precisions=Int[]
    if _real_float_type(eltype(H))===BigFloat
        bounds=_symmetric_ket_bigfloat_bounds(
            _symmetric_ket_matrix_values(H),"Hamiltonian input")
        push!(precisions,bounds[1])
    end
    requested_type===BigFloat&&push!(precisions,precision(BigFloat))
    rate isa BigFloat&&push!(precisions,precision(rate))
    hbar isa BigFloat&&push!(precisions,precision(hbar))
    isempty(precisions) ? nothing : maximum(precisions)
end

function _symmetric_ket_checked_rate_input(value,plan,label)
    R=_symmetric_ket_plan_real_type(plan)
    if R===BigFloat&&value isa BigFloat
        bits=precision(value)
        bits<=plan.precision_bits||throw(ArgumentError(
            "$label has precision $bits bits, but the prepared symmetric-ket " *
            "Hamiltonian plan has only $(plan.precision_bits) bits; prepare " *
            "the plan at the wider precision"))
        return BigFloat(value;precision=plan.precision_bits)
    end
    value
end

function SymmetricKetHamiltonianPlan(
        basis::PIBasis,H::AbstractMatrix;
        representation::Symbol=:local,rate=1,hbar=1,T=nothing,
        check::Bool=true,atol::Real=0,rtol::Real=0,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _symmetric_ket_check_basis(basis)
    representation in (:local,:block)||throw(ArgumentError(
        "representation must be :local or :block"))
    expected=representation===:local ? (basis.d,basis.d) :
        (symmetric_ket_dimension(basis),symmetric_ket_dimension(basis))
    size(H)==expected||throw(DimensionMismatch(
        "Hamiltonian has size $(size(H)); expected $expected for " *
        "representation=$representation"))
    target_precision=_symmetric_ket_target_bigfloat_precision(
        H,rate,hbar,T)
    if target_precision!==nothing
        if precision(BigFloat)!=target_precision
            return setprecision(BigFloat,target_precision) do
                SymmetricKetHamiltonianPlan(
                    basis,H;representation,rate,hbar,T,check,atol,rtol,
                    memory_budget)
            end
        end
    end
    R=_symmetric_ket_plan_precision(eltype(H),rate,hbar,T)
    scale=rate isa Number ? _checked_rate_quotient(rate,hbar,R) :
        _SymmetricKetHamiltonianRate(rate,hbar)
    input_bytes=BigInt(Base.summarysize(H))
    converted_bytes=_symmetric_ket_matrix_storage_bytes(H,R)
    hermiticity_scratch=_performance_entries_bytes(
        2BigInt(maximum(size(H))),R)
    setup_peak=if representation===:local
        geometry_estimate=_estimate_symmetric_collective_geometry(basis,R)
        input_bytes+converted_bytes+2BigInt(geometry_estimate.setup_bytes)+
            hermiticity_scratch
    else
        input_bytes+converted_bytes+hermiticity_scratch
    end
    _require_performance_budget(
        "symmetric-ket Hamiltonian preparation",setup_peak,memory_budget;
        guidance="Use sparse block storage or raise the explicit budget.")
    converted=_symmetric_ket_checked_matrix(H,R,"symmetric-ket Hamiltonian")
    check&&_symmetric_ket_check_hamiltonian(converted,atol,rtol)
    block=if representation===:local
        geometry=_SymmetricCollectiveGeometry(basis,R)
        _collective_sparse_block(
            basis,converted,only(basis.sectors);cache=geometry)
    else
        converted
    end
    block_values=block isa SparseMatrixCSC ? nonzeros(block) : block
    all(isfinite,block_values)||throw(ArgumentError(
        "lifted symmetric-ket Hamiltonian contains nonfinite entries"))
    precision_bits=R===BigFloat ? precision(BigFloat) : 0
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    SymmetricKetHamiltonianPlan{
        R,typeof(basis),typeof(block),typeof(scale),typeof(rounding_mode)}(
        basis,block,scale,precision_bits,rounding_mode)
end

function SymmetricKetHamiltonianPlan(
        H::PIOperator;
        rate=1,hbar=1,T=nothing,check::Bool=true,
        atol::Real=0,rtol::Real=0,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _symmetric_ket_check_basis(H.basis)
    SymmetricKetHamiltonianPlan(
        H.basis,coefficient_block(H,only(H.basis.sectors));
        representation=:block,rate,hbar,T,check,atol,rtol,memory_budget)
end

function _symmetric_ket_plan_real_type(
        ::SymmetricKetHamiltonianPlan{R}) where R
    R
end

function _symmetric_ket_evaluated_scale(plan,t,p)
    R=_symmetric_ket_plan_real_type(plan)
    if plan.scale isa _SymmetricKetHamiltonianRate
        raw_rate=value_at(plan.scale.schedule,t,p)
        checked_rate=_symmetric_ket_checked_rate_input(
            raw_rate,plan,"symmetric-ket Hamiltonian rate")
        checked_hbar=_symmetric_ket_checked_rate_input(
            plan.scale.hbar,plan,"symmetric-ket Hamiltonian hbar")
        return _checked_rate_quotient(checked_rate,checked_hbar,R)
    end
    value=_symmetric_ket_checked_rate_input(
        plan.scale,plan,"symmetric-ket Hamiltonian rate/hbar")
    _checked_prepared_real(value,R,"symmetric-ket Hamiltonian rate/hbar")
end

"""
    apply_symmetric_hamiltonian!(destination, plan, source,
                                 time=0, parameters=nothing)

Apply the prepared Schrödinger right-hand side without allocating.  Source and
destination must not overlap.
"""
function apply_symmetric_hamiltonian!(
        destination::AbstractVector,
        plan::SymmetricKetHamiltonianPlan{R},
        source::AbstractVector,time=0,parameters=nothing) where R
    if R===BigFloat&&(precision(BigFloat)!=plan.precision_bits||
            rounding(BigFloat)!=plan.rounding_mode)
        return setrounding(BigFloat,plan.rounding_mode) do
            setprecision(BigFloat,plan.precision_bits) do
                apply_symmetric_hamiltonian!(
                    destination,plan,source,time,parameters)
            end
        end
    end
    n=size(plan,1)
    length(source)==n||throw(DimensionMismatch(
        "symmetric-ket source has the wrong dimension"))
    length(destination)==n||throw(DimensionMismatch(
        "symmetric-ket destination has the wrong dimension"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "symmetric-ket source and destination must not alias"))
    eltype(source)===eltype(plan)||throw(ArgumentError(
        "source scalar type $(eltype(source)) does not match plan scalar " *
        "type $(eltype(plan))"))
    eltype(destination)===eltype(plan)||throw(ArgumentError(
        "destination scalar type $(eltype(destination)) does not match plan " *
        "scalar type $(eltype(plan))"))
    if R===BigFloat
        bounds=_symmetric_ket_bigfloat_bounds(source,"source")
        bounds==(plan.precision_bits,plan.precision_bits)||
            throw(ArgumentError(
                "source BigFloat storage has precision range $bounds, but " *
                "the Hamiltonian plan requires $(plan.precision_bits) bits"))
    end
    scale=_symmetric_ket_evaluated_scale(plan,time,parameters)
    mul!(destination,plan.hamiltonian,source)
    @. destination=(-im*scale)*destination
    if R===BigFloat
        bounds=_symmetric_ket_bigfloat_bounds(destination,"destination")
        bounds==(plan.precision_bits,plan.precision_bits)||
            throw(ArgumentError(
                "destination BigFloat storage has precision range $bounds, " *
                "but the Hamiltonian plan requires $(plan.precision_bits) bits"))
    end
    destination
end

apply!(destination::AbstractVector,plan::SymmetricKetHamiltonianPlan,
       source::AbstractVector,time=0,parameters=nothing)=
    apply_symmetric_hamiltonian!(
        destination,plan,source,time,parameters)

function LinearAlgebra.mul!(
        destination::AbstractVector,
        plan::SymmetricKetHamiltonianPlan,
        source::AbstractVector)
    plan.scale isa Number||throw(ArgumentError(
        "matrix-free Krylov evolution requires an autonomous symmetric-ket " *
        "Hamiltonian plan; use evolve_symmetric_ket! for a callable rate"))
    apply_symmetric_hamiltonian!(destination,plan,source,0,nothing)
end

"""
    SymmetricKetWorkspace(plan; memory_budget=512*1024^2)

Task-owned three-vector RK4 storage for one symmetric-ket Hamiltonian plan.
The immutable plan may be shared; a workspace may not be used concurrently.
"""
struct SymmetricKetWorkspace{P,V}
    plan::P
    temporary::V
    derivative::V
    accumulator::V
end

function SymmetricKetWorkspace(plan::SymmetricKetHamiltonianPlan;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    if _symmetric_ket_plan_real_type(plan)===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return setrounding(BigFloat,plan.rounding_mode) do
            setprecision(BigFloat,plan.precision_bits) do
                SymmetricKetWorkspace(plan;memory_budget)
            end
        end
    end
    T=eltype(plan);n=size(plan,1)
    peak=_performance_array_bytes(n,T,0;linear_arrays=3)
    _require_performance_budget(
        "symmetric-ket RK4 workspace",peak,memory_budget;guidance=
        "Use the autonomous Krylov route or raise the explicit budget.")
    SymmetricKetWorkspace(plan,zeros(T,n),zeros(T,n),zeros(T,n))
end

function _check_symmetric_ket_workspace(
        workspace::SymmetricKetWorkspace,
        plan::SymmetricKetHamiltonianPlan,destination,source)
    workspace.plan===plan||throw(ArgumentError(
        "SymmetricKetWorkspace was prepared for a different plan"))
    n=size(plan,1)
    all(vector->length(vector)==n,
        (workspace.temporary,workspace.derivative,
         workspace.accumulator))||throw(DimensionMismatch(
        "symmetric-ket workspace has the wrong dimension"))
    active=(workspace.temporary,workspace.derivative,
            workspace.accumulator)
    for index in eachindex(active)
        Base.mightalias(active[index],destination)&&throw(ArgumentError(
            "symmetric-ket destination must not alias workspace scratch"))
        Base.mightalias(active[index],source)&&throw(ArgumentError(
            "symmetric-ket source must not alias workspace scratch"))
        for previous in 1:index-1
            Base.mightalias(active[index],active[previous])&&
                throw(ArgumentError(
                    "symmetric-ket workspace arrays must not alias"))
        end
    end
    if _symmetric_ket_plan_real_type(plan)===BigFloat
        for (label,values) in (("workspace temporary",workspace.temporary),
                ("workspace derivative",workspace.derivative),
                ("workspace accumulator",workspace.accumulator))
            bounds=_symmetric_ket_bigfloat_bounds(values,label)
            bounds==(plan.precision_bits,plan.precision_bits)||
                throw(ArgumentError(
                    "$label has precision range $bounds, but the plan " *
                    "requires $(plan.precision_bits) bits"))
        end
    end
    workspace
end

function _symmetric_ket_checked_time(::Type{R},value,label) where
        R<:AbstractFloat
    value isa Real||throw(ArgumentError("$label must be real"))
    _checked_prepared_real(value,R,label)
end

"""
    evolve_symmetric_ket!(destination, plan, source, tspan;
                          steps=256, parameters=nothing, workspace=nothing,
                          check=true, atol, rtol,
                          memory_budget=512*1024^2)

Propagate a symmetric ket with a preallocated fixed-step RK4 kernel.  Exact
in-place use (`destination === source`) is supported.  The routine never
renormalizes. By default it validates the final norm and raises if the RK4
error exceeds `atol`/`rtol`; `check=false` explicitly returns an unchecked
approximation.
"""
function evolve_symmetric_ket!(
        destination::SymmetricKet{T},
        plan::SymmetricKetHamiltonianPlan{T},
        source::SymmetricKet{T},tspan;
        steps::Integer=256,parameters=nothing,workspace=nothing,
        check::Bool=true,
        atol::Real=max(T(1e-12),T(100)*eps(T)),
        rtol::Real=max(T(1e-10),T(100)*eps(T)),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where
        T<:AbstractFloat
    if T===BigFloat&&(precision(BigFloat)!=plan.precision_bits||
            rounding(BigFloat)!=plan.rounding_mode)
        return setrounding(BigFloat,plan.rounding_mode) do
            setprecision(BigFloat,plan.precision_bits) do
                evolve_symmetric_ket!(
                    destination,plan,source,tspan;steps,parameters,workspace,
                    check,atol,rtol,memory_budget)
            end
        end
    end
    destination.basis===plan.basis&&source.basis===plan.basis||
        throw(ArgumentError(
            "plan, source, and destination must use the same PIBasis object"))
    steps>0||throw(ArgumentError("steps must be positive"))
    atolT,rtolT=_symmetric_ket_tolerances(T,atol,rtol)
    check&&validate_symmetric_ket(source;atol=atolT,rtol=rtolT)
    destination===source||!Base.mightalias(destination.data,source.data)||
        throw(ArgumentError(
            "evolution permits exact in-place use but not partially " *
            "overlapping source and destination storage"))
    R=T
    t0=_symmetric_ket_checked_time(R,first(tspan),"initial time")
    t1=_symmetric_ket_checked_time(R,last(tspan),"final time")
    h=(t1-t0)/R(steps)
    isfinite(h)||throw(ArgumentError(
        "symmetric-ket time step is not finite"))
    t0!=t1&&iszero(h)&&throw(ArgumentError(
        "symmetric-ket time step underflows in $R"))
    workspace_peak=_performance_array_bytes(
        length(source),eltype(source),0;linear_arrays=3)
    _require_performance_budget(
        "symmetric-ket RK4 workspace",workspace_peak,memory_budget;guidance=
        "Use the autonomous Krylov route or raise the explicit budget.")
    work=workspace===nothing ?
        SymmetricKetWorkspace(plan;memory_budget=Inf) : workspace
    work isa SymmetricKetWorkspace||throw(ArgumentError(
        "workspace must be a SymmetricKetWorkspace"))
    _check_symmetric_ket_workspace(
        work,plan,destination.data,source.data)
    destination===source||copyto!(destination.data,source.data)
    t=t0
    for _ in 1:steps
        apply_symmetric_hamiltonian!(
            work.derivative,plan,destination.data,t,parameters)
        copyto!(work.accumulator,work.derivative)
        @. work.temporary=destination.data+(h/R(2))*work.derivative
        apply_symmetric_hamiltonian!(
            work.derivative,plan,work.temporary,t+h/R(2),parameters)
        @. work.accumulator=work.accumulator+R(2)*work.derivative
        @. work.temporary=destination.data+(h/R(2))*work.derivative
        apply_symmetric_hamiltonian!(
            work.derivative,plan,work.temporary,t+h/R(2),parameters)
        @. work.accumulator=work.accumulator+R(2)*work.derivative
        @. work.temporary=destination.data+h*work.derivative
        apply_symmetric_hamiltonian!(
            work.derivative,plan,work.temporary,t+h,parameters)
        @. work.accumulator=work.accumulator+work.derivative
        @. destination.data=destination.data+(h/R(6))*work.accumulator
        t+=h
    end
    check&&validate_symmetric_ket(
        destination;atol=atolT,rtol=rtolT)
    destination
end

"""Return a propagated copy of `source` after a combined output/workspace guard."""
function time_evolve_symmetric_ket(
        plan::SymmetricKetHamiltonianPlan{T},
        source::SymmetricKet{T},tspan;
        steps::Integer=256,parameters=nothing,workspace=nothing,
        check::Bool=true,
        atol::Real=max(T(1e-12),T(100)*eps(T)),
        rtol::Real=max(T(1e-10),T(100)*eps(T)),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where T<:AbstractFloat
    if T===BigFloat&&(precision(BigFloat)!=plan.precision_bits||
            rounding(BigFloat)!=plan.rounding_mode)
        return setrounding(BigFloat,plan.rounding_mode) do
            setprecision(BigFloat,plan.precision_bits) do
                time_evolve_symmetric_ket(
                    plan,source,tspan;steps,parameters,workspace,check,atol,
                    rtol,memory_budget)
            end
        end
    end
    source.basis===plan.basis||throw(ArgumentError(
        "plan and source must use the same PIBasis object"))
    steps>0||throw(ArgumentError("steps must be positive"))
    atolT,rtolT=_symmetric_ket_tolerances(T,atol,rtol)
    check&&validate_symmetric_ket(source;atol=atolT,rtol=rtolT)
    _symmetric_ket_checked_time(T,first(tspan),"initial time")
    _symmetric_ket_checked_time(T,last(tspan),"final time")
    workspace===nothing||workspace isa SymmetricKetWorkspace||
        throw(ArgumentError("workspace must be a SymmetricKetWorkspace"))
    peak=_performance_array_bytes(length(source),eltype(source),0;
        linear_arrays=4)
    _require_performance_budget(
        "allocating symmetric-ket RK4 evolution",peak,memory_budget;guidance=
        "Supply caller-owned destination/workspace storage or raise the budget.")
    work=workspace===nothing ?
        SymmetricKetWorkspace(plan;memory_budget=Inf) : workspace
    _check_symmetric_ket_workspace(work,plan,source.data,source.data)
    destination=copy(source)
    evolve_symmetric_ket!(destination,plan,source,tspan;
        steps,parameters,workspace=work,check,atol,rtol,memory_budget=Inf)
end

function _check_symmetric_krylov_workspace(
        workspace::KrylovExpvWorkspace,plan::SymmetricKetHamiltonianPlan,
        destination,source)
    n=length(source)
    m=size(workspace.H,2)
    size(workspace.V)==(n,m+1)&&size(workspace.H)==(m+1,m)&&
        size(workspace.small)==(m+1,m+1)&&
        length(workspace.w)==n&&length(workspace.current)==n&&
        length(workspace.trial)==n||throw(DimensionMismatch(
            "KrylovExpvWorkspace has incompatible dimensions"))
    eltype(workspace.V)===eltype(plan)||throw(ArgumentError(
        "KrylovExpvWorkspace has the wrong scalar type"))
    arrays=(workspace.V,workspace.H,workspace.small,workspace.w,
            workspace.current,workspace.trial)
    for index in eachindex(arrays)
        Base.mightalias(arrays[index],source)&&throw(ArgumentError(
            "symmetric-ket source must not alias Krylov workspace scratch"))
        Base.mightalias(arrays[index],destination)&&throw(ArgumentError(
            "symmetric-ket destination must not alias Krylov workspace scratch"))
        for previous in 1:index-1
            Base.mightalias(arrays[index],arrays[previous])&&
                throw(ArgumentError("Krylov workspace arrays must not alias"))
        end
    end
    if _symmetric_ket_plan_real_type(plan)===BigFloat
        for (label,values) in (("Krylov basis",workspace.V),
                ("Krylov Hessenberg",workspace.H),
                ("Krylov projected matrix",workspace.small),
                ("Krylov residual",workspace.w),
                ("Krylov current state",workspace.current),
                ("Krylov trial state",workspace.trial))
            bounds=_symmetric_ket_bigfloat_bounds(values,label)
            bounds==(plan.precision_bits,plan.precision_bits)||
                throw(ArgumentError(
                    "$label has precision range $bounds, but the plan " *
                    "requires $(plan.precision_bits) bits"))
        end
    end
    workspace
end

"""
    krylov_evolve_symmetric_ket!(destination, plan, source, time;
        workspace=nothing, krylovdim=30, memory_budget=512*1024^2,
        kwargs...)

Apply the autonomous propagator `exp(time*(-im*rate*H/hbar))` with the
package's adaptive restarted-Arnoldi exponential action.  The Hamiltonian
remains sparse or matrix-free through `plan`; no exponential or density
matrix is formed.  A supplied `KrylovExpvWorkspace` is task-owned and reused;
its capacity is included in `memory_budget` whether supplied or allocated.
Callable rates are rejected because an exponential action is valid only for
an autonomous generator. The propagated norm is checked by default and is
never repaired.
"""
function krylov_evolve_symmetric_ket!(
        destination::SymmetricKet{T},
        plan::SymmetricKetHamiltonianPlan{T},
        source::SymmetricKet{T},time::Real;
        workspace=nothing,krylovdim::Integer=30,
        atol::Real=T(1e-10),rtol::Real=T(1e-8),
        check::Bool=true,norm_atol::Real=atol,norm_rtol::Real=rtol,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...) where
        T<:AbstractFloat
    if T===BigFloat&&(precision(BigFloat)!=plan.precision_bits||
            rounding(BigFloat)!=plan.rounding_mode)
        return setrounding(BigFloat,plan.rounding_mode) do
            setprecision(BigFloat,plan.precision_bits) do
                krylov_evolve_symmetric_ket!(
                    destination,plan,source,time;workspace,krylovdim,atol,rtol,
                    check,norm_atol,norm_rtol,memory_budget,kwargs...)
            end
        end
    end
    destination.basis===plan.basis&&source.basis===plan.basis||
        throw(ArgumentError(
            "plan, source, and destination must use the same PIBasis object"))
    plan.scale isa Number||throw(ArgumentError(
        "Krylov exponential evolution requires an autonomous symmetric-ket " *
        "Hamiltonian plan"))
    _symmetric_ket_checked_time(T,time,"evolution time")
    _symmetric_ket_tolerances(T,atol,rtol)
    norm_atolT,norm_rtolT=_symmetric_ket_tolerances(
        T,norm_atol,norm_rtol)
    check&&validate_symmetric_ket(
        source;atol=norm_atolT,rtol=norm_rtolT)
    workspace===nothing||workspace isa KrylovExpvWorkspace||
        throw(ArgumentError("workspace must be a KrylovExpvWorkspace"))
    effective_dimension=workspace===nothing ? krylovdim : size(workspace.H,2)
    peak=_performance_krylov_expv_workspace_bytes(
        length(source),eltype(plan),effective_dimension)
    _require_performance_budget(
        "symmetric-ket Krylov workspace",peak,memory_budget;guidance=
        "Reduce krylovdim or raise the explicit budget.")
    work=workspace===nothing ? KrylovExpvWorkspace(
        eltype(plan),length(source),krylovdim) : workspace
    _check_symmetric_krylov_workspace(
        work,plan,destination.data,source.data)
    result=krylov_expv!(
        destination.data,plan,source.data,time,work;atol,rtol,kwargs...)
    check&&validate_symmetric_ket(
        destination;atol=norm_atol,rtol=norm_rtol)
    merge(result,(state=destination,))
end

"""Allocating result wrapper for `krylov_evolve_symmetric_ket!`."""
function krylov_time_evolve_symmetric_ket(
        plan::SymmetricKetHamiltonianPlan{T},
        source::SymmetricKet{T},time::Real;
        workspace=nothing,krylovdim::Integer=30,
        atol::Real=T(1e-10),rtol::Real=T(1e-8),
        check::Bool=true,norm_atol::Real=atol,norm_rtol::Real=rtol,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...) where
        T<:AbstractFloat
    if T===BigFloat&&(precision(BigFloat)!=plan.precision_bits||
            rounding(BigFloat)!=plan.rounding_mode)
        return setrounding(BigFloat,plan.rounding_mode) do
            setprecision(BigFloat,plan.precision_bits) do
                krylov_time_evolve_symmetric_ket(
                    plan,source,time;workspace,krylovdim,atol,rtol,check,
                    norm_atol,norm_rtol,memory_budget,kwargs...)
            end
        end
    end
    source.basis===plan.basis||throw(ArgumentError(
        "plan and source must use the same PIBasis object"))
    _symmetric_ket_checked_time(T,time,"evolution time")
    _symmetric_ket_tolerances(T,atol,rtol)
    _symmetric_ket_tolerances(T,norm_atol,norm_rtol)
    check&&validate_symmetric_ket(source;atol=norm_atol,rtol=norm_rtol)
    workspace===nothing||workspace isa KrylovExpvWorkspace||
        throw(ArgumentError("workspace must be a KrylovExpvWorkspace"))
    effective_dimension=workspace===nothing ? krylovdim : size(workspace.H,2)
    peak=_performance_krylov_expv_workspace_bytes(
        length(source),eltype(plan),effective_dimension)+
        _performance_entries_bytes(length(source),eltype(source))
    _require_performance_budget(
        "allocating symmetric-ket Krylov evolution",peak,memory_budget;
        guidance="Supply caller-owned destination/workspace storage or raise the budget.")
    work=workspace===nothing ? KrylovExpvWorkspace(
        eltype(plan),length(source),krylovdim) : workspace
    _check_symmetric_krylov_workspace(work,plan,source.data,source.data)
    destination=copy(source)
    krylov_evolve_symmetric_ket!(
        destination,plan,source,time;workspace=work,krylovdim,
        atol,rtol,check,norm_atol,norm_rtol,memory_budget=Inf,kwargs...)
end

"""
    symmetric_ket_expectation(ket, operator;
                              workspace=nothing,
                              memory_budget=512*1024^2)

Evaluate a `PIOperator`, physical Schur block, or prepared Hamiltonian directly
against a symmetric ket without constructing a density matrix. For a
`SymmetricKetHamiltonianPlan`, this returns the expectation of the stored
Hamiltonian block itself; the separate evolution scale `rate/hbar` is not
included.
"""
function symmetric_ket_expectation(
        ket::SymmetricKet,operator;
        workspace=nothing,
        check::Bool=true,
        atol=nothing,rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    R=_real_float_type(eltype(ket))
    ket_precision=if R===BigFloat
        _symmetric_ket_bigfloat_bounds(
            ket.data,"symmetric-ket amplitudes")[1]
    else
        0
    end
    if R===BigFloat
        big_plan=operator isa SymmetricKetHamiltonianPlan&&
            _symmetric_ket_plan_real_type(operator)===BigFloat
        target_precision=big_plan ?
            operator.precision_bits : ket_precision
        target_rounding=big_plan ?
            operator.rounding_mode : rounding(BigFloat)
        ket_precision==target_precision||throw(ArgumentError(
            "symmetric-ket amplitudes have precision $ket_precision bits, " *
            "but the observable plan requires $target_precision bits"))
        if precision(BigFloat)!=target_precision||
                rounding(BigFloat)!=target_rounding
            return setrounding(BigFloat,target_rounding) do
                setprecision(BigFloat,target_precision) do
                    symmetric_ket_expectation(
                        ket,operator;workspace,check,atol,rtol,memory_budget)
                end
            end
        end
    end
    check&&validate_symmetric_ket(ket;atol,rtol)
    matrix=if operator isa PIOperator
        operator.basis===ket.basis||throw(ArgumentError(
            "operator and ket use different PIBasis objects"))
        coefficient_block(operator,only(operator.basis.sectors))
    elseif operator isa SymmetricKetHamiltonianPlan
        operator.basis===ket.basis||throw(ArgumentError(
            "Hamiltonian plan and ket use different PIBasis objects"))
        operator.hamiltonian
    elseif operator isa AbstractMatrix
        size(operator)==(length(ket),length(ket))||throw(DimensionMismatch(
            "observable block has the wrong dimensions"))
        operator
    else
        throw(ArgumentError(
            "operator must be a PIOperator, SymmetricKetHamiltonianPlan, or " *
            "physical Schur-block matrix"))
    end
    eltype(matrix)===eltype(ket.data)||throw(ArgumentError(
        "observable and symmetric ket must have the same scalar type"))
    if R===BigFloat
        matrix_precision=_symmetric_ket_bigfloat_bounds(
            _symmetric_ket_matrix_values(matrix),"observable block")[1]
        matrix_precision==ket_precision||throw(ArgumentError(
            "observable block has precision $matrix_precision bits, but the " *
            "symmetric ket has precision $ket_precision bits"))
    end
    peak=_performance_entries_bytes(length(ket),eltype(ket.data))
    _require_performance_budget(
        "symmetric-ket expectation workspace",peak,memory_budget;
        guidance="Pass a reusable ket-sized workspace.")
    temporary=workspace===nothing ? similar(ket.data) : workspace
    temporary isa AbstractVector&&length(temporary)==length(ket)||
        throw(DimensionMismatch(
            "expectation workspace must be a vector of length $(length(ket))"))
    eltype(temporary)===eltype(ket.data)||throw(ArgumentError(
        "expectation workspace has the wrong scalar type"))
    Base.mightalias(temporary,ket.data)&&throw(ArgumentError(
        "expectation workspace must not alias ket storage"))
    mul!(temporary,matrix,ket.data)
    if R===BigFloat
        workspace_precision=_symmetric_ket_bigfloat_bounds(
            temporary,"expectation workspace")[1]
        workspace_precision==ket_precision||throw(ArgumentError(
            "expectation workspace has precision $workspace_precision bits, " *
            "but the symmetric ket has precision $ket_precision bits"))
    end
    dot(ket.data,temporary)
end

# Ket-native contraction through the already prepared exact sparse
# local-factor trace map.  The source coordinate at flattened index q is
# psi[row]*conj(psi[column]); it is evaluated only where a retained sparse
# transform entry requests it.  Thus no g-by-g density block is formed.
@inline function _symmetric_ket_complex_product_risk(
        a::T,b::T,c::T,d::T,ac::T,bd::T,ad::T,bc::T,
        result::Complex{T}) where T<:AbstractFloat
    ((!iszero(a)&&!iszero(c)&&iszero(ac))||
     (!iszero(b)&&!iszero(d)&&iszero(bd))||
     (!iszero(a)&&!iszero(d)&&iszero(ad))||
     (!iszero(b)&&!iszero(c)&&iszero(bc)))&&return true
    isfinite(result)||return true
    real_scale=abs(ac)+abs(bd)
    imaginary_scale=abs(ad)+abs(bc)
    isfinite(real_scale)&&isfinite(imaginary_scale)||return true
    roundoff_guard=T(8)*eps(T)
    real_cancellation=!iszero(real_scale)&&
        abs(real(result))<=roundoff_guard*real_scale
    imaginary_cancellation=!iszero(imaginary_scale)&&
        abs(imag(result))<=roundoff_guard*imaginary_scale
    if real_cancellation&&
            !(iszero(real(result))&&
              _symmetric_ket_exact_zero_product_difference(
                  a,c,ac,b,d,bd))
        return true
    end
    if imaginary_cancellation&&
            !(iszero(imag(result))&&
              _symmetric_ket_exact_zero_product_sum(
                  a,d,ad,b,c,bc))
        return true
    end
    if T===Float16||T===Float32||T===Float64
        smallest=nextfloat(zero(T));largest=floatmax(T)
        real_magnitude=abs(real(result))
        imaginary_magnitude=abs(imag(result))
        (real_magnitude==smallest||real_magnitude==largest||
         imaginary_magnitude==smallest||imaginary_magnitude==largest)&&
            return true
    end
    false
end

# With an FMA, `fma(x,y,-x*y)` is the error-free residual of one normal,
# finite binary product.  It lets common algebraic zeros such as
# `z*conj(z)` remain on the allocation-free path without treating a rounded
# cancellation as an exact zero.  Products touching the subnormal range stay
# on the conservative wide-precision path.
@inline function _symmetric_ket_normal_product(value::T) where
        T<:AbstractFloat
    # A product may be normal while its exact FMA residual is still below the
    # subnormal range.  Requiring `precision(T)` additional exponent bits
    # makes every possible nonzero residual of two T-valued operands
    # representable; smaller products use the wide fallback.
    iszero(value)||abs(value)>=ldexp(floatmin(T),precision(T))
end

@inline function _symmetric_ket_exact_zero_product_difference(
        a::T,c::T,ac::T,b::T,d::T,bd::T) where T<:AbstractFloat
    (T===Float16||T===Float32||T===Float64)||return false
    _symmetric_ket_normal_product(ac)&&
        _symmetric_ket_normal_product(bd)||return false
    fma(a,c,-ac)==fma(b,d,-bd)
end

@inline function _symmetric_ket_exact_zero_product_sum(
        a::T,d::T,ad::T,b::T,c::T,bc::T) where T<:AbstractFloat
    (T===Float16||T===Float32||T===Float64)||return false
    _symmetric_ket_normal_product(ad)&&
        _symmetric_ket_normal_product(bc)||return false
    fma(a,d,-ad)==-fma(b,c,-bc)
end

@inline function _symmetric_ket_complex_product_and_risk(
        left::Complex{T},right::Complex{T}) where T<:AbstractFloat
    a=real(left);b=imag(left);c=real(right);d=imag(right)
    ac=a*c;bd=b*d;ad=a*d;bc=b*c
    result=Complex{T}(ac-bd,ad+bc)
    result,_symmetric_ket_complex_product_risk(
        a,b,c,d,ac,bd,ad,bc,result)
end

@inline function _symmetric_ket_complex_product_requires_wide(
        left::Complex{T},right::Complex{T},result::Complex{T}) where
        T<:AbstractFloat
    a=real(left);b=imag(left);c=real(right);d=imag(right)
    ac=a*c;bd=b*d;ad=a*d;bc=b*c
    _symmetric_ket_complex_product_risk(
        a,b,c,d,ac,bd,ad,bc,result)
end

function _symmetric_ket_triple_guard_precision(
        ::Type{T},first,second,third,memory_budget) where T<:AbstractFloat
    minimum_exponent=typemax(Int)
    maximum_exponent=typemin(Int)
    for value in (first,second,third),component in (real(value),imag(value))
        iszero(component)&&continue
        _,component_exponent=frexp(component)
        minimum_exponent=min(minimum_exponent,component_exponent)
        maximum_exponent=max(maximum_exponent,component_exponent)
    end
    span=minimum_exponent==typemax(Int) ? 0 :
        BigInt(maximum_exponent)-BigInt(minimum_exponent)
    source_precision=T===BigFloat ? precision(BigFloat) : precision(T)
    # Every expanded real or imaginary component is a sum of four products of
    # three binary floating-point components.  This precision can represent
    # those products and their complete exponent span before the final checked
    # conversion, including cancellation such as (1-delta)*(1+delta).
    guard_big=max(BigInt(256),3BigInt(source_precision)+3span+64)
    guard_big<=typemax(Int)||throw(ArgumentError(
        "symmetric-ket contraction certification requires an unsupported " *
        "BigFloat precision; rescale the state or operator"))
    guard=Int(guard_big)
    scratch_bytes=32*cld(guard_big,8)+1024
    _require_performance_budget(
        "symmetric-ket contraction certification",scratch_bytes,
        memory_budget;guidance=
        "Rescale the state/operator or use inputs with a smaller exponent span.")
    guard
end

Base.@noinline function _symmetric_ket_wide_triple_product(
        first::Complex{T},second::Complex{T},third::Complex{T},context,
        memory_budget) where
        T<:AbstractFloat
    guard=_symmetric_ket_triple_guard_precision(
        T,first,second,third,memory_budget)
    wide=setprecision(BigFloat,guard) do
        Complex{BigFloat}(first)*Complex{BigFloat}(second)*
            Complex{BigFloat}(third)
    end
    _symmetric_ket_checked_complex(T,wide,context)
end

@inline function _symmetric_ket_checked_triple_product(
        first::Complex{T},second::Complex{T},third::Complex{T},context,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where
        T<:AbstractFloat
    isfinite(first)&&isfinite(second)&&isfinite(third)||throw(ArgumentError(
        "$context contains a nonfinite multiplicative input"))
    (iszero(first)||iszero(second)||iszero(third))&&return zero(Complex{T})
    pair,pair_requires_wide=_symmetric_ket_complex_product_and_risk(
        first,second)
    value,value_requires_wide=_symmetric_ket_complex_product_and_risk(
        pair,third)
    !pair_requires_wide&&!value_requires_wide&&!iszero(value)&&return value
    _symmetric_ket_wide_triple_product(
        first,second,third,context,memory_budget)
end

@inline function _symmetric_ket_checked_abs2(
        value::Complex{T},context,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where
        T<:AbstractFloat
    isfinite(value)||throw(ArgumentError(
        "$context contains a nonfinite amplitude"))
    iszero(value)&&return zero(Complex{T})
    real_square=real(value)*real(value)
    imaginary_square=imag(value)*imag(value)
    result=real_square+imaginary_square
    requires_wide=(!iszero(real(value))&&iszero(real_square))||
        (!iszero(imag(value))&&iszero(imaginary_square))||!isfinite(result)
    if !requires_wide&&(T===Float16||T===Float32||T===Float64)
        requires_wide=result==nextfloat(zero(T))||result==floatmax(T)
    end
    !requires_wide&&!iszero(result)&&return Complex{T}(result,zero(T))
    _symmetric_ket_wide_triple_product(
        value,conj(value),one(Complex{T}),context,memory_budget)
end

function _symmetric_ket_local_factor_occupations!(
        destination,ket::SymmetricKet,plan::LocalFactorTracePlan,
        memory_budget)
    lifted=plan.lifted_columns
    lifted isa SparseMatrixCSC||error(
        "LocalFactorTracePlan lifted_columns must use SparseMatrixCSC storage")
    dimension=length(ket)
    size(lifted,1)==dimension^2||throw(DimensionMismatch(
        "local-factor trace source dimension is incompatible with the " *
        "symmetric ket"))
    length(destination)==size(lifted,2)||throw(DimensionMismatch(
        "local-factor occupation workspace has the wrong dimension"))
    @inbounds for column in axes(lifted,2)
        value=zero(eltype(destination))
        correction=zero(eltype(destination))
        for pointer in nzrange(lifted,column)
            coordinate=lifted.rowval[pointer]
            row=mod1(coordinate,dimension)
            ket_column=(coordinate-1)÷dimension+1
            contribution=if row==ket_column
                density_entry=_symmetric_ket_checked_abs2(
                    ket.data[row],
                    "symmetric-ket local-factor trace diagonal",
                    memory_budget)
                _symmetric_ket_checked_triple_product(
                    conj(lifted.nzval[pointer]),density_entry,
                    one(eltype(ket.data)),
                    "symmetric-ket local-factor trace contribution",
                    memory_budget)
            else
                _symmetric_ket_checked_triple_product(
                    conj(lifted.nzval[pointer]),ket.data[row],
                    conj(ket.data[ket_column]),
                    "symmetric-ket local-factor trace contribution",
                    memory_budget)
            end
            updated=value+contribution
            correction+=abs(value)>=abs(contribution) ?
                (value-updated)+contribution :
                (contribution-updated)+value
            value=updated
        end
        result=value+correction
        isfinite(result)||throw(ArgumentError(
            "symmetric-ket local-factor trace accumulation is nonfinite; " *
            "use a wider scalar type"))
        destination[column]=result
    end
    destination
end

"""
    local_factor_trace!(output, ket, plan, workspace;
                        check=true, atol, rtol,
                        memory_budget=512*1024^2)
    local_factor_trace(ket, plan; workspace=nothing,
                       memory_budget=512*1024^2, ...)

Trace one internal local factor directly from a physical symmetric ket.  This
method reuses `LocalFactorTracePlan` but contracts each retained sparse source
coordinate against `psi*psi'` on demand, avoiding the `O(g^2)` source
`PIState` allocation.
"""
function local_factor_trace!(
        output::PIState,ket::SymmetricKet,
        plan::LocalFactorTracePlan,
        workspace::LocalFactorTraceWorkspace;
        check::Bool=true,atol=nothing,rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _symmetric_ket_validate_memory_budget(memory_budget)
    ket.basis===plan.basis||throw(ArgumentError(
        "LocalFactorTracePlan was prepared for a different source PIBasis"))
    output.basis===plan.output_basis||throw(ArgumentError(
        "output state must use the LocalFactorTracePlan output_basis object"))
    workspace.plan===plan||throw(ArgumentError(
        "LocalFactorTraceWorkspace was prepared for a different plan"))
    T=eltype(plan.lifted_columns)
    eltype(ket.data)===T||throw(ArgumentError(
        "symmetric ket scalar type $(eltype(ket.data)) does not match " *
        "LocalFactorTracePlan scalar type $T"))
    eltype(output.data)===T&&
        eltype(workspace.occupation_coordinates)===T||throw(ArgumentError(
        "local-factor output and workspace must match plan scalar type $T"))
    if _real_float_type(T)===BigFloat
        precision_bits=plan.estimates.precision_bits
        rounding_mode=plan.estimates.rounding_mode
        if precision(BigFloat)!=precision_bits||
                rounding(BigFloat)!=rounding_mode
            return setrounding(BigFloat,rounding_mode) do
                setprecision(BigFloat,precision_bits) do
                    local_factor_trace!(
                        output,ket,plan,workspace;
                        check,atol,rtol,memory_budget)
                end
            end
        end
        for (name,values) in (("symmetric ket",ket.data),
                ("output state",output.data),
                ("workspace",workspace.occupation_coordinates))
            bounds=_local_factor_precision_bounds(values)
            bounds==(precision_bits,precision_bits)||throw(ArgumentError(
                "$name BigFloat storage has precision range $bounds, but " *
                "the LocalFactorTracePlan requires $precision_bits bits"))
        end
    end
    actual_atol=atol===nothing ? _analysis_atol(output) : atol
    actual_rtol=rtol===nothing ? _state_rtol(output) : rtol
    actual_atol isa Real&&actual_rtol isa Real||throw(ArgumentError(
        "local-factor trace tolerances must be real numbers or nothing"))
    _symmetric_ket_tolerances(
        _real_float_type(eltype(ket.data)),actual_atol,actual_rtol)
    check&&validate_symmetric_ket(
        ket;atol=actual_atol,rtol=actual_rtol)
    _symmetric_ket_local_factor_occupations!(
        workspace.occupation_coordinates,ket,plan,memory_budget)
    mul!(output.data,plan.output_columns,
         workspace.occupation_coordinates)
    check&&validate_state(output;atol=actual_atol,rtol=actual_rtol)
    output
end

function local_factor_trace(
        ket::SymmetricKet{T},plan::LocalFactorTracePlan;
        workspace=nothing,check::Bool=true,atol=nothing,rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where
        T<:AbstractFloat
    if T===BigFloat
        precision_bits=plan.estimates.precision_bits
        rounding_mode=plan.estimates.rounding_mode
        if precision(BigFloat)!=precision_bits||
                rounding(BigFloat)!=rounding_mode
            return setrounding(BigFloat,rounding_mode) do
                setprecision(BigFloat,precision_bits) do
                    local_factor_trace(
                        ket,plan;workspace,check,atol,rtol,memory_budget)
                end
            end
        end
    end
    work=workspace===nothing ? LocalFactorTraceWorkspace(plan) : workspace
    work isa LocalFactorTraceWorkspace||throw(ArgumentError(
        "workspace must be a LocalFactorTraceWorkspace"))
    actual_atol=atol===nothing ? T(100)*eps(T) : atol
    actual_rtol=rtol===nothing ? T(100)*eps(T) : rtol
    actual_atol isa Real&&actual_rtol isa Real||throw(ArgumentError(
        "local-factor trace tolerances must be real numbers or nothing"))
    output=PIState(plan.output_basis;T)
    local_factor_trace!(
        output,ket,plan,work;
        check,atol=actual_atol,rtol=actual_rtol,memory_budget)
end
