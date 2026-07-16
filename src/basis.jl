"""
    PIBasis(N, d; sectors=nothing)

Construct the equation-(7) PI operator basis for `N` identical `d`-level
systems. By default all partitions of `N` into `d` padded parts are retained;
`sectors` may select a restricted set. Sectors use descending lexicographic
order, GT patterns use ascending stored-entry order, and each sector matrix is
flattened in Julia column-major order. `length(basis)` is the retained PI
coordinate dimension ``sum_nu dim(U_nu)^2``.
"""
struct PIBasis{D,L}
    N::Int
    d::Int
    sectors::Vector{Partition{D}}
    patterns::Vector{Vector{GTPattern{D,L}}}
    offsets::Vector{Int}
    index::Dict{Partition{D},Int}
end

function PIBasis(N::Integer,d::Integer;sectors=nothing)
    N>=0 || throw(ArgumentError("N must be nonnegative")); d>=1 || throw(ArgumentError("d must be positive"))
    D=Int(d);L=D*(D+1)÷2
    ps = if sectors===nothing
        partitions(N,D)
    else
        Partition{D}[Partition(Tuple(x)) for x in sectors]
    end
    all(p->weight(p)==N,ps) || throw(ArgumentError("all sectors must partition N into d parts"))
    length(unique(ps))==length(ps) || throw(ArgumentError("duplicate sector"))
    sort!(ps;by=p->p.parts,rev=true)
    pats=Vector{Vector{GTPattern{D,L}}}(undef,length(ps))
    for i in eachindex(ps);pats[i]=gt_patterns(ps[i]);end
    offs=Int[1]; for gs in pats; push!(offs,offs[end]+length(gs)^2); end
    PIBasis{D,L}(Int(N),D,ps,pats,offs,Dict(p=>i for (i,p) in pairs(ps)))
end
length(b::PIBasis)=b.offsets[end]-1
show(io::IO,b::PIBasis)=print(io,"PIBasis(N=$(b.N), d=$(b.d), sectors=$(length(b.sectors)), dimension=$(length(b)))")

abstract type AbstractPIOperator{T} end

"""
    PIOperator(basis, data)
    PIOperator(basis; T=Float64)

An operator in orthonormal PI coefficient coordinates. `data` is partitioned
into column-major stored blocks ``C_nu``; use [`physical_block`](@ref) for the
corresponding Schur block ``C_nu/sqrt(f^nu)``. The zero constructor selects
the real scalar type with `T`.
"""
struct PIOperator{T<:AbstractFloat,B<:PIBasis} <: AbstractPIOperator{T}
    basis::B
    data::Vector{Complex{T}}
    function PIOperator(b::PIBasis,data::AbstractVector{Complex{T}}) where T<:AbstractFloat
        length(data)==length(b)||throw(DimensionMismatch("coefficient vector has wrong length")); new{T,typeof(b)}(b,collect(data))
    end
end

"""
    PIState(basis, data)
    PIState(basis; T=Float64)

A density-state container in the same PI coefficient convention as
[`PIOperator`](@ref). Construction checks only basis-coordinate dimensions;
it does not normalize, symmetrize, or impose positivity. Use
[`validate_state`](@ref) when physical-state validation is required.
"""
struct PIState{T<:AbstractFloat,B<:PIBasis} <: AbstractPIOperator{T}
    basis::B
    data::Vector{Complex{T}}
    function PIState(b::PIBasis,data::AbstractVector{Complex{T}}) where T<:AbstractFloat
        length(data)==length(b)||throw(DimensionMismatch("coefficient vector has wrong length")); new{T,typeof(b)}(b,collect(data))
    end
end
PIOperator(b::PIBasis;T=Float64)=PIOperator(b,zeros(Complex{T},length(b)))
PIState(b::PIBasis;T=Float64)=PIState(b,zeros(Complex{T},length(b)))
# The public constructors already make one defensive `collect` copy. Passing
# the source vector directly here therefore preserves `copy` semantics without
# first creating a second, immediately discarded vector. This matters for
# trajectory output, where every saved state is copied.
copy(a::PIOperator)=PIOperator(a.basis,a.data); copy(a::PIState)=PIState(a.basis,a.data)
eltype(a::AbstractPIOperator)=eltype(a.data)
function _sidx(b::PIBasis,p::Partition); get(b.index,p,0)>0||throw(ArgumentError("sector $p is absent")); b.index[p]; end

"""
    coefficient_block(A, partition)

Return a mutable matrix view of the stored equation-(7) coefficient block
``C_nu``. Mutating the view mutates `A`.
"""
function coefficient_block(a::AbstractPIOperator,p::Partition)
    s=_sidx(a.basis,p); n=length(a.basis.patterns[s]); reshape(view(a.data,a.basis.offsets[s]:a.basis.offsets[s+1]-1),n,n)
end

"""Alias for [`coefficient_block`](@ref)."""
sector_view=coefficient_block

function _schur_multiplicity_scale(::Type{T},partition) where T<:AbstractFloat
    _checked_sqrt_exact_integer(T,symmetric_group_dimension(partition);
        context="square root of the sector multiplicity for $partition")
end

function _schur_inverse_multiplicity_scale(::Type{T},partition) where T<:AbstractFloat
    _checked_sqrt_exact_ratio(T,one(BigInt),symmetric_group_dimension(partition);
        context="inverse square root of the sector multiplicity for $partition")
end

function _multiply_by_schur_multiplicity_scale(value,::Type{T},partition) where T<:AbstractFloat
    multiplicity=symmetric_group_dimension(partition)
    scale=try
        _checked_sqrt_exact_integer(T,multiplicity;
            context="square root of the sector multiplicity for $partition")
    catch error
        error isa ArgumentError||rethrow()
        nothing
    end
    if scale!==nothing
        result=scale.*value
        _ordinary_scaled_value_safe(result,value)&&return result
    end
    _checked_mul_sqrt_exact_ratio(T,value,multiplicity,one(BigInt);
        context="coefficient Schur-block scaling for $partition")
end

function _ordinary_scaled_component_safe(result::T,input::T) where T<:AbstractFloat
    isfinite(result)&&
        (iszero(input)||!iszero(result))&&
        abs(result)!=floatmax(T)&&abs(result)!=nextfloat(zero(T))
end
_ordinary_scaled_value_safe(result::Real,input::Real)=
    _ordinary_scaled_component_safe(result,input)
_ordinary_scaled_value_safe(result::Complex,input::Complex)=
    _ordinary_scaled_component_safe(real(result),real(input))&&
    _ordinary_scaled_component_safe(imag(result),imag(input))
function _ordinary_scaled_value_safe(result::AbstractArray,input::AbstractArray)
    all(index->_ordinary_scaled_value_safe(result[index],input[index]),eachindex(result,input))
end

function _divide_by_schur_multiplicity_scale(value,::Type{T},partition) where T<:AbstractFloat
    multiplicity=symmetric_group_dimension(partition)
    scale=try
        _checked_sqrt_exact_integer(T,multiplicity;
            context="square root of the sector multiplicity for $partition")
    catch error
        error isa ArgumentError||rethrow()
        nothing
    end
    if scale!==nothing
        result=value/scale
        _ordinary_scaled_value_safe(result,value)&&return result
    end
    _checked_mul_sqrt_exact_ratio(T,value,one(BigInt),multiplicity;
        context="physical Schur-block scaling for $partition")
end

"""
    physical_block(A, partition)

Return the physical Schur block ``C_nu/sqrt(f^nu)`` corresponding to the
stored coefficient block of `A`.  The returned matrix is detached from `A`;
the exact multiplicity is square-rooted with binary scaling, and an
unrepresentable final scale raises rather than producing a nonfinite value.
"""
function physical_block(a::AbstractPIOperator,p::Partition)
    T=typeof(real(zero(eltype(a.data))))
    # Public per-copy blocks require the equation-(7) divisor itself to be
    # representable, even for an all-zero coefficient block.  Internal
    # contractions may use `_divide_by_schur_multiplicity_scale` directly to
    # fuse an exceptional inverse with other finite factors.
    multiplicity=symmetric_group_dimension(p)
    scale=_checked_sqrt_exact_integer(T,multiplicity;
        context="square root of the sector multiplicity for $p")
    value=coefficient_block(a,p)
    result=value/scale
    _ordinary_scaled_value_safe(result,value)&&return result
    _checked_mul_sqrt_exact_ratio(T,value,one(BigInt),multiplicity;
        context="physical Schur-block scaling for $p")
end

function _schur_block_representation(representation::Symbol)
    representation in (:physical,:coefficient)||throw(ArgumentError(
        "representation must be :physical or :coefficient"))
    representation
end

"""
    each_schur_block(A; representation=:physical)

Iterate lazily over `partition => block` pairs in the retained sector order of
`A.basis`.  With the default `representation=:physical`, each returned block
is detached storage for the physical Schur matrix
``C_nu/sqrt(f^nu)``; mutating it does not modify `A`.  With
`representation=:coefficient`, each block is the mutable stored
[`coefficient_block`](@ref) view, so mutating it does modify `A`.

The iterator can be passed directly to [`operator_from_schur_blocks`](@ref) or
[`state_from_schur_blocks`](@ref) for a round trip, using the same
`representation` keyword on both sides.
"""
function each_schur_block(a::AbstractPIOperator;
                          representation::Symbol=:physical)
    resolved=_schur_block_representation(representation)
    if resolved===:physical
        return (p=>physical_block(a,p) for p in a.basis.sectors)
    end
    (p=>coefficient_block(a,p) for p in a.basis.sectors)
end

function _sector_metadata_entry(b::PIBasis,s::Int,p::Partition,spin)
    dimension=length(b.patterns[s])
    multiplicity=symmetric_group_dimension(p)
    (index=s,partition=p,block_dimension=dimension,
     coordinate_dimension=dimension^2,
     coordinate_range=b.offsets[s]:(b.offsets[s+1]-1),
     multiplicity,
     hilbert_dimension=multiplicity*big(dimension),spin)
end

"""
    sector_metadata(basis)

Return one named tuple per retained Schur sector, in basis order.  Each entry
contains the one-based sector `index`, `partition`, `block_dimension`
``dim(U_nu)``, `coordinate_dimension`, flattened `coordinate_range`, exact
`BigInt` symmetric-group `multiplicity` ``f^nu``, and exact retained
`hilbert_dimension = f^nu dim(U_nu)``.  For qubits, `spin` is the exact
total-spin label ``j=(nu_1-nu_2)/2``; it is `missing` for other local
dimensions.

No full ``d^N`` object is constructed.  For a complete basis, summing the
`hilbert_dimension` fields gives exactly ``d^N``; for a restricted basis it
gives the retained Hilbert-space dimension.
"""
function sector_metadata(b::PIBasis)
    if b.d==2
        return [_sector_metadata_entry(b,s,p,(p[1]-p[2])//2)
                for (s,p) in pairs(b.sectors)]
    end
    [_sector_metadata_entry(b,s,p,missing)
     for (s,p) in pairs(b.sectors)]
end

function _samebasis(a,b); a.basis===b.basis||throw(ArgumentError("incompatible PI bases")); end

"""
    identity_operator(basis; T=Float64)

Construct the identity on the Hilbert space represented by `basis`. Its stored
block in sector `nu` is ``sqrt(f^nu) I``.
"""
function identity_operator(b::PIBasis;T=Float64)
    a=PIOperator(b;T=T)
    for p in b.sectors; C=coefficient_block(a,p); C[diagind(C)].=_schur_multiplicity_scale(T,p); end
    a
end

"""
    maximally_mixed_state(basis; T=Float64)

Construct the normalized maximally mixed state on the retained Hilbert-space
sectors. For a complete basis this is ``I/d^N``.
"""
function maximally_mixed_state(b::PIBasis;T=Float64)
    retained_dimension=sum((symmetric_group_dimension(p)*
        big(length(b.patterns[s])) for (s,p) in pairs(b.sectors));init=big(0))
    iszero(retained_dimension)&&throw(ArgumentError(
        "cannot construct a maximally mixed state on an empty retained Hilbert space"))
    denominator=retained_dimension^2
    a=PIState(b;T=T)
    for p in b.sectors
        multiplicity=symmetric_group_dimension(p)
        coefficient=_checked_sqrt_exact_ratio(T,multiplicity,denominator;
            context="maximally mixed coefficient in sector $p")
        C=coefficient_block(a,p);C[diagind(C)].=coefficient
    end
    a
end

"""
    trace(A)

Return the physical trace
``sum_nu sqrt(f^nu) tr(C_nu)`` from the stored PI coefficient blocks.
"""
function trace(a::AbstractPIOperator)
    T=typeof(real(zero(eltype(a.data))))
    total=zero(Complex{T});correction=zero(Complex{T})
    for p in a.basis.sectors
        multiplicity=symmetric_group_dimension(p)
        block_trace=LinearAlgebra.tr(coefficient_block(a,p))
        scale=try
            _checked_sqrt_exact_integer(T,multiplicity;
                context="square root of the sector multiplicity for $p")
        catch error
            error isa ArgumentError||rethrow()
            nothing
        end
        contribution=scale===nothing ?
            _checked_mul_sqrt_exact_ratio(T,block_trace,multiplicity,one(BigInt);
                context="trace contribution of sector $p") : scale*block_trace
        if scale!==nothing&&!_ordinary_scaled_value_safe(contribution,block_trace)
            contribution=_checked_mul_sqrt_exact_ratio(
                T,block_trace,multiplicity,one(BigInt);
                context="trace contribution of sector $p")
        end
        # Neumaier-style compensated summation costs only O(number of sectors)
        # and protects traces of general (not necessarily positive) operators.
        updated=total+contribution
        correction+=abs(total)>=abs(contribution) ?
            (total-updated)+contribution : (contribution-updated)+total
        total=updated
    end
    total+correction
end
"""Return the density-operator purity `tr(rho^2)` in orthonormal PI coordinates."""
function purity(a::PIState)
    result=real(dot(a.data,a.data))
    isfinite(result)&&(!iszero(result)||all(iszero,a.data))&&return result
    scale=norm(a.data)
    isfinite(scale)&&!iszero(scale)||throw(ArgumentError(
        "state purity is outside the nonzero finite range of $(_real_float_type(eltype(a.data))); use a wider scalar type"))
    result=scale*scale
    isfinite(result)&&!iszero(result)||throw(ArgumentError(
        "state purity is outside the nonzero finite range of $(_real_float_type(eltype(a.data))); use a wider scalar type"))
    result
end

"""
    normalize!(rho)

Divide the stored coordinates of `rho` by its physical trace and return
`rho`. A zero trace raises; no Hermiticity or positivity repair is performed.
"""
function normalize!(a::PIState); z=trace(a); iszero(z)&&throw(ArgumentError("zero trace")); a.data./=z; a; end

_real_float_type(::Type{T}) where T=typeof(float(real(zero(T))))
_complex_float_type(::Type{T}) where T=
    promote_type(T,Complex{_real_float_type(T)})
_default_rtol(a::AbstractPIOperator)=sqrt(eps(_real_float_type(eltype(a.data))))
_analysis_atol(a::AbstractPIOperator)=max(_real_float_type(eltype(a.data))(1e-12),
                                          100eps(_real_float_type(eltype(a.data))))
_state_rtol(a::AbstractPIOperator)=max(_real_float_type(eltype(a.data))(1e-10),
                                       100eps(_real_float_type(eltype(a.data))))

function _hermitian_eigen(A::Hermitian;operation::AbstractString="Hermitian spectral analysis")
    try
        eigen(A)
    catch error
        error isa MethodError||rethrow()
        R=_real_float_type(eltype(A))
        throw(ArgumentError("$operation is unavailable for scalar type $R with Julia's active LinearAlgebra backend; use Float32/Float64 data or load a compatible generic eigensolver"))
    end
end

function _hermitian_eigvals(A::Hermitian;operation::AbstractString="Hermitian spectral analysis")
    try
        eigvals(A)
    catch error
        error isa MethodError||rethrow()
        R=_real_float_type(eltype(A))
        throw(ArgumentError("$operation is unavailable for scalar type $R with Julia's active LinearAlgebra backend; use Float32/Float64 data or load a compatible generic eigensolver"))
    end
end

# Hermiticity and positivity are invariant under the positive scalar relating a
# coefficient block to its physical Schur block.  Work in stored coordinates
# so validation remains available when a physical eigenvalue underflows.
function _hermiticity_metrics(a::AbstractPIOperator)
    T=_real_float_type(eltype(a.data));err=zero(T);scale=zero(T)
    for p in a.basis.sectors
        C=coefficient_block(a,p)
        err=max(err,norm(C-C',Inf))
        scale=max(scale,norm(C,Inf))
    end
    (;error=err,scale)
end

function ishermitian(a::AbstractPIOperator;atol::Real=0,rtol::Real=_default_rtol(a))
    atol>=0||throw(ArgumentError("atol must be nonnegative"))
    rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
    h=_hermiticity_metrics(a)
    h.error<=atol+rtol*h.scale
end

# This helper is only called after tolerant Hermiticity has been established.
# Averaging then removes roundoff-level skew-Hermitian noise before `eigvals`;
# it is not used to repair an invalid input.
function _spectral_positivity_metrics(a::AbstractPIOperator)
    minimum_eigenvalue=nothing;scale=nothing
    for p in a.basis.sectors
        C=Matrix(coefficient_block(a,p));vals=_hermitian_eigvals(Hermitian((C+C')/2);
            operation="positivity eigenvalue diagnostics")
        pmin=minimum(vals);pscale=maximum(abs,vals)
        minimum_eigenvalue=minimum_eigenvalue===nothing ? pmin : min(minimum_eigenvalue,pmin)
        scale=scale===nothing ? pscale : max(scale,pscale)
    end
    (;minimum_eigenvalue,scale)
end

function _positivity_scale(a::AbstractPIOperator)
    T=_real_float_type(eltype(a.data));scale=zero(T)
    for p in a.basis.sectors
        C=coefficient_block(a,p)
        scale=max(scale,norm(C,Inf))
    end
    scale
end

# Complete diagonal pivoting provides a scalar-generic PSD decision for the
# small local matrices and exceptional failed-Cholesky blocks that cannot use
# LAPACK's Hermitian eigensolvers. `tolerance` is applied as a diagonal shift;
# the matrix itself is never modified.
function _scalar_generic_psd_check(A::Hermitian{Complex{R},<:AbstractMatrix},
                                   tolerance::R) where R<:AbstractFloat
    S=Matrix(A);n=size(S,1)
    for index in 1:n;S[index,index]+=tolerance;end
    scale=max(norm(S,Inf),one(R));roundoff=R(32)*R(max(n,1))*eps(R)*scale
    for k in 1:n
        pivot_index=k;pivot_value=real(S[k,k])
        for index in k+1:n
            candidate=real(S[index,index])
            candidate>pivot_value&&(pivot_index=index;pivot_value=candidate)
        end
        if pivot_index!=k
            S[[k,pivot_index],:]=S[[pivot_index,k],:]
            S[:,[k,pivot_index]]=S[:,[pivot_index,k]]
        end
        pivot=real(S[k,k]);pivot>=-roundoff||return false
        if pivot<=roundoff
            # Complete diagonal pivoting chose the largest remaining diagonal.
            # For a PSD matrix, a numerically zero diagonal implies its whole
            # row/column is numerically zero. Gershgorin lower bounds cannot be
            # used conversely here: a rank-one block delta*ones(n,n) is PSD but
            # has a negative Gershgorin bound for n>2.
            trailing_max=maximum(abs,@view(S[k:n,k:n]);init=zero(R))
            return trailing_max<=roundoff
        end
        for column in k+1:n
            column_pivot=S[column,k]
            for row in column:n
                value=S[row,column]-S[row,k]*conj(column_pivot)/pivot
                S[row,column]=value;S[column,row]=conj(value)
            end
        end
    end
    true
end

"""
    positivity_diagnostics(A; method=:auto, dense_threshold=256,
                           atol=_analysis_atol(A), rtol=0)

Check positivity sector by sector without constructing a full `d^N` matrix.
`:eigen` returns the exact numerical minimum of every stored Schur block.
`:cholesky` instead factorizes each block after the requested tolerance shift;
it avoids a full eigenspectrum for accepted states and falls back to `eigmin`
only when a factorization fails on LAPACK-supported scalar types. Other scalar
types use a pivoted PSD decision without narrowing precision. `:auto` selects
the Cholesky path when the largest irrep block exceeds `dense_threshold` or a
Hermitian eigensolver is unavailable.

On the Cholesky path `minimum_eigenvalue` is `missing`, because a successful
factorization is a lower-bound certificate rather than an eigenvalue
calculation. For LAPACK-supported scalar types, a negative direction stores
the failed block's numerical minimum in `witness_eigenvalue`; a generic
pivoted-PSD failure is reported with a missing numerical witness. The input is
never modified.
"""
function positivity_diagnostics(a::AbstractPIOperator;method::Symbol=:auto,
                                dense_threshold::Integer=256,
                                atol::Real=_analysis_atol(a),rtol::Real=0)
    atol>=0||throw(ArgumentError("atol must be nonnegative"))
    rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
    dense_threshold>=1||throw(ArgumentError("dense_threshold must be positive"))
    method in (:auto,:eigen,:cholesky)||
        throw(ArgumentError("method must be :auto, :eigen, or :cholesky"))
    h=_hermiticity_metrics(a)
    hermitian=h.error<=atol+rtol*h.scale
    hermitian||return (;positive=false,hermitian=false,method=:none,
        minimum_eigenvalue=missing,witness_eigenvalue=missing,
        certified_lower_bound=missing,scale=h.scale,tolerance=missing,
        factorized_sectors=0,fallback_eigensolves=0)

    largest=maximum(length(patterns) for patterns in a.basis.patterns;init=0)
    R=_real_float_type(eltype(a.data))
    lapack_scalar=R===Float32||R===Float64
    selected=method===:auto ? (largest<=dense_threshold&&lapack_scalar ? :eigen : :cholesky) : method
    if selected===:eigen
        p=_spectral_positivity_metrics(a);tol=atol+rtol*p.scale
        return (;positive=p.minimum_eigenvalue>=-tol,hermitian=true,method=:eigen,
            minimum_eigenvalue=p.minimum_eigenvalue,
            witness_eigenvalue=p.minimum_eigenvalue < -tol ? p.minimum_eigenvalue : missing,
            certified_lower_bound=p.minimum_eigenvalue,scale=p.scale,tolerance=tol,
            factorized_sectors=0,fallback_eigensolves=0)
    end

    T=R;scale=_positivity_scale(a)
    tol=T(atol)+T(rtol)*scale
    factorized=0;fallbacks=0;witness=missing
    for p in a.basis.sectors
        C=Matrix(coefficient_block(a,p));B=(C+C')/2
        @inbounds for i in axes(B,1);B[i,i]+=tol;end
        factor=cholesky!(Hermitian(B);check=false)
        if isposdef(factor)
            factorized+=1
            continue
        end
        # POTRF is deliberately strict at a zero pivot. Re-evaluate only this
        # exceptional block to decide the tolerance-boundary case reliably.
        fallbacks+=1
        original=Matrix(coefficient_block(a,p));H=Hermitian((original+original')/2)
        if lapack_scalar
            lambda=eigmin(H);invalid=lambda < -tol
            invalid&&(witness=lambda)
        else
            invalid=!_scalar_generic_psd_check(H,tol)
        end
        if invalid
            return (;positive=false,hermitian=true,method=:cholesky,
                minimum_eigenvalue=missing,witness_eigenvalue=witness,
                certified_lower_bound=missing,scale,tolerance=tol,
                factorized_sectors=factorized,fallback_eigensolves=fallbacks)
        end
    end
    (;positive=true,hermitian=true,method=:cholesky,
      minimum_eigenvalue=missing,witness_eigenvalue=witness,
      certified_lower_bound=-tol,scale,tolerance=tol,
      factorized_sectors=factorized,fallback_eigensolves=fallbacks)
end

"""
    ispositive(A; atol=_analysis_atol(A), rtol=0,
               method=:auto, dense_threshold=256)

Return whether every physical Schur block is Hermitian and positive
semidefinite within tolerance. See [`positivity_diagnostics`](@ref) for the
eigenvalue and scalable shifted-Cholesky backends and their detailed report.
"""
function ispositive(a::AbstractPIOperator;atol::Real=_analysis_atol(a),rtol::Real=0,
                    method::Symbol=:auto,dense_threshold::Integer=256)
    positivity_diagnostics(a;atol=atol,rtol=rtol,method=method,
                           dense_threshold=dense_threshold).positive
end

"""
    state_diagnostics(rho; atol=_analysis_atol(rho), rtol=_state_rtol(rho),
                      positivity_method=:auto, dense_threshold=256)

Return trace, Hermiticity, and positivity diagnostics for `rho`.  Invalid
inputs are reported but never normalized, symmetrized, or eigenvalue-clipped.
The minimum eigenvalue is `missing` when the state is not Hermitian within the
requested tolerance, because positivity is then not mathematically defined,
or when the scalable Cholesky certificate succeeds without diagonalization.
In the latter case `positivity_method` and `positivity_witness` distinguish the
certificate from a failed spectral check.
"""
function state_diagnostics(rho::PIState;atol::Real=_analysis_atol(rho),
                           rtol::Real=_state_rtol(rho),
                           positivity_method::Symbol=:auto,
                           dense_threshold::Integer=256)
    atol>=0||throw(ArgumentError("atol must be nonnegative"))
    rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
    z=trace(rho);trace_error=abs(z-one(z));trace_tolerance=atol+rtol
    h=_hermiticity_metrics(rho);hermiticity_tolerance=atol+rtol*h.scale
    hermitian=h.error<=hermiticity_tolerance
    if hermitian
        p=positivity_diagnostics(rho;atol=atol,rtol=rtol,
            method=positivity_method,dense_threshold=dense_threshold)
        positivity_tolerance=p.tolerance
        minimum_eigenvalue=p.minimum_eigenvalue
        positive=p.positive
        positivity_method_used=p.method
        positivity_witness=p.witness_eigenvalue
    else
        minimum_eigenvalue=missing;positivity_tolerance=missing;positive=false
        positivity_method_used=:none;positivity_witness=missing
    end
    trace_one=trace_error<=trace_tolerance
    (;trace_value=z,trace_error,hermiticity_error=h.error,minimum_eigenvalue,
      trace_tolerance,hermiticity_tolerance,positivity_tolerance,
      positivity_method=positivity_method_used,positivity_witness,
      trace_one,hermitian,positive,valid=trace_one&&hermitian&&positive)
end

"""
    validate_state(rho; trace_one=true, hermitian=true, positive=true,
                   atol=_analysis_atol(rho), rtol=_state_rtol(rho),
                   positivity_method=:auto, dense_threshold=256)

Validate selected density-state conditions and return `rho`.  A failed check
throws `ArgumentError`; this function never silently repairs its argument.
Set individual Boolean keywords to `false` only when a subnormalized or
operator-like `PIState` is intentionally being analysed.
"""
function validate_state(rho::PIState;trace_one::Bool=true,hermitian::Bool=true,
                        positive::Bool=true,atol::Real=_analysis_atol(rho),
                        rtol::Real=_state_rtol(rho),
                        positivity_method::Symbol=:auto,
                        dense_threshold::Integer=256)
    atol>=0||throw(ArgumentError("atol must be nonnegative"))
    rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
    if trace_one
        z=trace(rho);err=abs(z-one(z));tol=atol+rtol
        err<=tol||throw(ArgumentError("state is not trace one: trace=$z, error=$err, tolerance=$tol"))
    end
    if positive
        # `positivity_diagnostics` already performs the tolerant Hermiticity
        # check required for positivity.  Reusing that result avoids scanning
        # and differencing every Schur block twice on every validated analysis
        # call.  Recompute the detailed metric only on the exceptional error
        # path so the established diagnostic remains informative.
        p=positivity_diagnostics(rho;atol=atol,rtol=rtol,
            method=positivity_method,dense_threshold=dense_threshold)
        if !p.hermitian
            h=_hermiticity_metrics(rho);tol=atol+rtol*h.scale
            throw(ArgumentError(!hermitian ?
                "state positivity requires a Hermitian input within tolerance" :
                "state is not Hermitian: error=$(h.error), tolerance=$tol"))
        end
        p.positive||throw(ArgumentError("state is not positive: witness eigenvalue=$(p.witness_eigenvalue), tolerance=$(p.tolerance)"))
    elseif hermitian
        h=_hermiticity_metrics(rho);tol=atol+rtol*h.scale
        h.error<=tol||throw(ArgumentError(
            "state is not Hermitian: error=$(h.error), tolerance=$tol"))
    end
    rho
end

# Shared validated block spectrum used by entropy and entanglement routines.
# Hermitian averaging is performed only after the tolerance check succeeds.
function _sector_eigenvalues(rho::PIState,p;atol=_analysis_atol(rho),
                             rtol=_state_rtol(rho))
    R=Matrix(physical_block(rho,p));herr=norm(R-R',Inf);hscale=norm(R,Inf)
    herr<=atol+rtol*hscale||throw(ArgumentError("state is not Hermitian in sector $p: error=$herr"))
    vals=_hermitian_eigvals(Hermitian((R+R')/2);
        operation="density-block spectral analysis");scale=maximum(abs,vals)
    minimum(vals)>=-(atol+rtol*scale)||throw(ArgumentError("state has a negative eigenvalue in sector $p"))
    vals
end

"""
    isphysical(rho; atol=_analysis_atol(rho), rtol=_state_rtol(rho))

Return whether `rho` is trace one, Hermitian, and positive within the requested
tolerances. The input is never modified.
"""
isphysical(a::PIState;atol::Real=_analysis_atol(a),rtol::Real=_state_rtol(a))=state_diagnostics(a;atol=atol,rtol=rtol).valid

"""
    sector_population(A, partition)

Return ``sqrt(f^nu) tr(C_nu)``, the physical trace contribution of one Schur
sector. For a valid state this is its sector probability.
"""
function sector_population(a::AbstractPIOperator,p::Partition)
    T=typeof(real(zero(eltype(a.data))))
    multiplicity=symmetric_group_dimension(p)
    block_trace=LinearAlgebra.tr(coefficient_block(a,p))
    scale=try
        _checked_sqrt_exact_integer(T,multiplicity;
            context="square root of the sector multiplicity for $p")
    catch error
        error isa ArgumentError||rethrow()
        nothing
    end
    if scale!==nothing
        result=scale*block_trace
        _ordinary_scaled_value_safe(result,block_trace)&&return result
    end
    _checked_mul_sqrt_exact_ratio(T,block_trace,multiplicity,one(BigInt);
        context="population of sector $p")
end

"""Return a dictionary of [`sector_population`](@ref) values for all retained sectors."""
sector_populations(a::AbstractPIOperator)=Dict(p=>sector_population(a,p) for p in a.basis.sectors)

"""
    basis_state(basis, partition, W; T=Float64)

Construct the trace-one PI state supported on GT pattern `W` in `partition`,
uniformly over the sector's symmetric-group multiplicity copies.
"""
function basis_state(b::PIBasis,p::Partition,W::GTPattern;T=Float64)
    s=_sidx(b,p); i=findfirst(==(W),b.patterns[s]); i===nothing&&throw(ArgumentError("pattern not in sector"))
    a=PIState(b;T=T)
    coefficient_block(a,p)[i,i]=_schur_inverse_multiplicity_scale(T,p)
    a
end

"""
    sector_density_matrix(basis, partition, B)

Embed `B` as the physical Schur block of `partition`, with every other sector
zero. The stored block is ``sqrt(f^nu) B``; the caller is responsible for the
desired global trace and positivity.
"""
function sector_density_matrix(b::PIBasis,p::Partition,B::AbstractMatrix)
    n=length(b.patterns[_sidx(b,p)]); size(B)==(n,n)||throw(DimensionMismatch())
    T=typeof(float(real(zero(eltype(B)))));a=PIState(b;T=T)
    coefficient_block(a,p).=_multiply_by_schur_multiplicity_scale(B,T,p)
    a
end

function _schur_block_partition(b::PIBasis{D},label) where D
    partition=if label isa Partition
        length(label)==D||throw(ArgumentError(
            "sector label $label has length $(length(label)); expected $D"))
        label
    elseif label isa Tuple||label isa AbstractVector
        length(label)==D||throw(ArgumentError(
            "sector label has length $(length(label)); expected $D"))
        all(value->value isa Integer,label)||throw(ArgumentError(
            "sector tuple/vector labels must contain integers"))
        try
            Partition(ntuple(index->Int(label[index]),D))
        catch error
            error isa InexactError||rethrow()
            throw(ArgumentError("sector label entries must be representable as Int"))
        end
    else
        throw(ArgumentError(
            "Schur-block labels must be Partition objects or integer tuples/vectors"))
    end
    _sidx(b,partition)
    partition
end

function _collect_schur_block_entries(b::PIBasis,blocks,requested_type)
    entries=NamedTuple[]
    real_types=DataType[]
    seen=Set{eltype(b.sectors)}()
    for entry in blocks
        entry isa Pair||throw(ArgumentError(
            "blocks must be an iterable of partition => matrix pairs"))
        partition=_schur_block_partition(b,first(entry))
        partition in seen&&throw(ArgumentError("duplicate Schur block for sector $partition"))
        push!(seen,partition)
        block=last(entry)
        block isa AbstractMatrix||throw(ArgumentError(
            "the block for sector $partition must be an AbstractMatrix"))
        Base.require_one_based_indexing(block)
        sector_index=b.index[partition]
        dimension=length(b.patterns[sector_index])
        size(block)==(dimension,dimension)||throw(DimensionMismatch(
            "block for sector $partition has size $(size(block)); expected ($dimension, $dimension)"))
        real_type=try
            _real_float_type(eltype(block))
        catch
            throw(ArgumentError(
                "cannot infer a floating-point scalar type from block eltype $(eltype(block))"))
        end
        real_type<:AbstractFloat||throw(ArgumentError(
            "block eltype $(eltype(block)) does not promote to an AbstractFloat scalar"))
        push!(entries,(partition=partition,block=block))
        push!(real_types,real_type)
    end
    scalar_type=if requested_type===nothing
        isempty(real_types) ? Float64 : foldl(promote_type,real_types)
    else
        requested_type isa Type&&requested_type<:AbstractFloat||throw(ArgumentError(
            "T must be an AbstractFloat type or nothing"))
        all(real_type->promote_type(requested_type,real_type)===requested_type,
            real_types)||throw(ArgumentError(
            "one or more Schur blocks would be narrowed by conversion to $requested_type"))
        requested_type
    end
    entries,scalar_type
end

function _from_schur_blocks(::Type{S},b::PIBasis,blocks;
                            representation::Symbol,T=nothing) where S<:AbstractPIOperator
    resolved=_schur_block_representation(representation)
    entries,scalar_type=_collect_schur_block_entries(b,blocks,T)
    result=S(b;T=scalar_type)
    for entry in entries
        destination=coefficient_block(result,entry.partition)
        if resolved===:physical
            destination.=_multiply_by_schur_multiplicity_scale(
                entry.block,scalar_type,entry.partition)
        else
            destination.=entry.block
        end
    end
    result
end

"""
    operator_from_schur_blocks(basis, blocks; representation=:physical,
                               T=nothing)

Construct a `PIOperator` from an iterable of `partition => matrix` pairs.
Input order is irrelevant and unspecified retained sectors are exactly zero.
By default each matrix is interpreted as a physical Schur block and is stored
as ``sqrt(f^nu)`` times that matrix.  With `representation=:coefficient`, the
matrices are interpreted directly as stored equation-(7) coefficient blocks.

Every label must belong to `basis`, duplicate labels and wrong block sizes
raise, and input matrices are copied rather than aliased.  The output scalar
type is the promotion of the input blocks' real floating component types; an
empty collection produces a zero `ComplexF64` operator. Pass an
`AbstractFloat` type as `T` to select the output type explicitly; conversion
that would narrow any supplied block raises. Tuple and integer-vector
partition labels are accepted as conveniences. No full ``d^N`` operator is
constructed.
"""
function operator_from_schur_blocks(b::PIBasis,blocks;
                                    representation::Symbol=:physical,T=nothing)
    _from_schur_blocks(PIOperator,b,blocks;representation,T)
end

"""
    state_from_schur_blocks(basis, blocks; representation=:physical,
                            T=nothing, validate=false,
                            validation_keywords...)

Construct a `PIState` from an iterable of `partition => matrix` pairs, using
the copying, sector-membership, dimension, representation, and scalar-
promotion rules of [`operator_from_schur_blocks`](@ref).  Unspecified retained
sectors are exactly zero.  The function never normalizes, symmetrizes, clips,
or otherwise repairs its input.

With `validate=true`, [`validate_state`](@ref) is called on the completed state
and remaining keywords such as `atol`, `rtol`, `trace_one`, or
`positivity_method` are forwarded to it.  Validation keywords supplied while
`validate=false` raise instead of being silently ignored.
"""
function state_from_schur_blocks(b::PIBasis,blocks;
        representation::Symbol=:physical,T=nothing,
        validate::Bool=false,kwargs...)
    !validate&&!isempty(kwargs)&&throw(ArgumentError(
        "validation keywords require validate=true"))
    state=_from_schur_blocks(PIState,b,blocks;representation,T)
    if validate
        validate_state(state;kwargs...)
    end
    state
end

# Product-state occupation amplitudes are assembled from conditional binomial
# distributions. This remains finite at and beyond the point where their
# individual exact binomial coefficients no longer fit in the output type.
_iid_amplitude_work_type(::Type{Float16})=Float64
_iid_amplitude_work_type(::Type{Float32})=Float64
_iid_amplitude_work_type(::Type{T}) where T<:AbstractFloat=T

function _normalized_binomial_amplitudes(M::Int,a::W,b::W) where W<:AbstractFloat
    values=zeros(W,M+1)
    if M==0
        values[1]=one(W)
        return values
    elseif iszero(a)
        values[1]=one(W)
        return values
    elseif iszero(b)
        values[end]=one(W)
        return values
    end

    mode=clamp(floor(Int,W(M+1)*a*a),0,M)
    values[mode+1]=one(W)
    for occupation in mode+1:M
        ratio=sqrt(W(M-occupation+1)/W(occupation))*(a/b)
        values[occupation+1]=values[occupation]*ratio
    end
    for occupation in mode-1:-1:0
        ratio=sqrt(W(occupation+1)/W(M-occupation))*(b/a)
        values[occupation+1]=values[occupation+2]*ratio
    end
    scale=norm(values)
    isfinite(scale)&&!iszero(scale)||throw(ArgumentError(
        "conditional-binomial amplitudes are not representable in $W; use a wider scalar type"))
    values./=scale
    values
end

function _iid_pure_state_impl(b::PIBasis,psi::AbstractVector,atol::Real,
                              rtol::Real,::Type{T}) where T<:AbstractFloat
    p=Partition(ntuple(i->i==1 ? b.N : 0,b.d));s=_sidx(b,p)
    atol>=0||throw(ArgumentError("atol must be nonnegative"));rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
    W=_iid_amplitude_work_type(T)
    # Validate in the recurrence precision, not in a low-precision norm that
    # may round to exactly one.  The accepted one-site error must also remain
    # acceptable after N tensor factors; otherwise returning the unnormalized
    # tensor power would silently amplify it exponentially.
    psi_norm=norm(Complex{W}.(psi))
    isapprox(psi_norm,one(W);atol=atol,rtol=rtol)||throw(ArgumentError(
        "psi must be normalized; norm evaluated in $W is $psi_norm"))
    tensor_trace=abs2(psi_norm^b.N)
    tensor_tolerance=W(atol)+W(rtol)
    isfinite(tensor_trace)&&abs(tensor_trace-one(W))<=tensor_tolerance||throw(ArgumentError(
        "the accepted one-particle normalization error is amplified at N=$(b.N): " *
        "the tensor-power trace would be $tensor_trace (tolerance $tensor_tolerance); " *
        "supply a more accurately normalized vector or use a wider scalar type"))

    magnitudes=W[abs(value) for value in psi]
    tails=zeros(W,b.d+1)
    for level in b.d:-1:1
        tails[level]=hypot(magnitudes[level],tails[level+1])
    end
    phases=Complex{W}[iszero(magnitudes[level]) ? one(Complex{W}) :
        Complex{W}(psi[level])/magnitudes[level] for level in 1:b.d]
    overall_scale=tails[1]^b.N
    isfinite(overall_scale)&&!iszero(overall_scale)||throw(ArgumentError(
        "the tensor-power amplitude scale is outside the nonzero finite range of $W; use a wider scalar type"))

    tables=Dict{Tuple{Int,Int},Vector{W}}()
    v=zeros(Complex{T},length(b.patterns[s]))
    for (k,g) in pairs(b.patterns[s])
        occupations=content(g)
        any(level->iszero(magnitudes[level])&&occupations[level]>0,1:b.d)&&continue
        remaining=b.N
        amplitude=Complex{W}(overall_scale)
        for level in 1:b.d-1
            iszero(remaining)&&break
            tail=tails[level]
            iszero(tail)&&begin
                amplitude=zero(amplitude)
                break
            end
            table=get!(tables,(level,remaining)) do
                _normalized_binomial_amplitudes(
                    remaining,magnitudes[level]/tail,tails[level+1]/tail)
            end
            amplitude*=table[occupations[level]+1]
            remaining-=occupations[level]
        end
        for level in 1:b.d
            occupations[level]>0&&(amplitude*=phases[level]^occupations[level])
        end
        converted=Complex{T}(amplitude)
        isfinite(converted)||throw(ArgumentError(
            "an occupation amplitude is not finite in $T; use a wider scalar type"))
        v[k]=converted
    end
    sector_density_matrix(b,p,v*v')
end

_bigfloat_input_precision(value::BigFloat)=precision(value)
_bigfloat_input_precision(value::Complex{BigFloat})=
    max(precision(real(value)),precision(imag(value)))

"""
    iid_pure_state(basis, psi; atol=0, rtol=sqrt(eps(...)))

Construct the product state ``|psi><psi|^otimes N`` directly in the fully
symmetric Schur sector. `psi` must be normalized and that sector must be
present in `basis`; no full ``d^N`` vector is formed. Occupation amplitudes
are evaluated by normalized conditional-binomial recurrences, so large
binomial or multinomial coefficients are never converted to the state scalar
type.
"""
function iid_pure_state(b::PIBasis,psi::AbstractVector;atol::Real=0,
                        rtol::Real=sqrt(eps(_real_float_type(eltype(psi)))))
    length(psi)==b.d||throw(DimensionMismatch())
    T=_real_float_type(eltype(psi))
    if T===BigFloat
        input_precision=maximum(_bigfloat_input_precision,psi;
            init=precision(BigFloat))
        return setprecision(BigFloat,input_precision) do
            _iid_pure_state_impl(b,psi,atol,rtol,T)
        end
    end
    _iid_pure_state_impl(b,psi,atol,rtol,T)
end
