function _operator_product_exponent_span(block,::Type{T}) where T<:AbstractFloat
    minimum_exponent=typemax(Int);maximum_exponent=typemin(Int)
    for value in block,component in (real(value),imag(value))
        converted=T(component)
        isfinite(converted)||throw(ArgumentError(
            "PI operator multiplication requires finite coefficient blocks"))
        iszero(converted)&&continue
        _,exponent=frexp(converted)
        minimum_exponent=min(minimum_exponent,exponent)
        maximum_exponent=max(maximum_exponent,exponent)
    end
    minimum_exponent==typemax(Int) ? 0 :
        maximum_exponent-minimum_exponent
end

function _convert_checked_operator_product(::Type{T},value,p) where T<:AbstractFloat
    converted=try
        Complex{T}(value)
    catch error
        throw(ArgumentError(
            "PI operator product in sector $p is not representable in $T: " *
            sprint(showerror,error)))
    end
    isfinite(converted)||throw(ArgumentError(
        "PI operator product in sector $p is outside the finite range of $T; " *
        "use a wider scalar type"))
    for (source,destination) in
        ((real(value),real(converted)),(imag(value),imag(converted)))
        iszero(source)&&continue
        iszero(destination)&&throw(ArgumentError(
            "PI operator product in sector $p is outside the nonzero finite " *
            "range of $T; use a wider scalar type"))
        if T===Float16||T===Float32||T===Float64
            magnitude=abs(destination)
            if magnitude==nextfloat(zero(T))||magnitude==floatmax(T)
                exact_magnitude=abs(Rational{BigInt}(source))
                Rational{BigInt}(nextfloat(zero(T)))<=exact_magnitude<=
                    Rational{BigInt}(floatmax(T))||throw(ArgumentError(
                        "PI operator product in sector $p is outside the nonzero " *
                        "finite range of $T; use a wider scalar type"))
            end
        end
    end
    converted
end

@inline function _complex_product_cancellation_possible(left::Complex,
                                                         right::Complex)
    a=real(left);b=imag(left);c=real(right);d=imag(right)
    real_cancellation=!iszero(a)&&!iszero(c)&&!iszero(b)&&!iszero(d)&&
        ((signbit(a)⊻signbit(c))==(signbit(b)⊻signbit(d)))
    imaginary_cancellation=!iszero(a)&&!iszero(d)&&!iszero(b)&&!iszero(c)&&
        ((signbit(a)⊻signbit(d))!=(signbit(b)⊻signbit(c)))
    real_cancellation||imaginary_cancellation
end

# Exceptional fallback for `C_A*C_B/sqrt(f)`.  The ordinary BLAS multiplication
# remains the first path.  This routine is reached only when that intermediate
# overflowed or the subsequent scaling could not be represented, even though
# the combined physical product may still be finite.  Native fused products
# usually resolve the case without allocating wider matrices; guarded
# BigFloat multiplication is reserved for an unrepresentable contribution or
# a severely cancelled dot product.
function _fused_schur_operator_product(left,right,::Type{T},p::Partition) where
        T<:AbstractFloat
    multiplicity=symmetric_group_dimension(p)
    inverse_scale=_prepare_exact_scale(T,one(BigInt),multiplicity,Val(true);
        context="PI operator product in sector $p")
    rows,inner=size(left);inner==size(right,1)||throw(DimensionMismatch())
    columns=size(right,2);result=zeros(Complex{T},rows,columns)
    severe=false
    for column in 1:columns,row in 1:rows
        value=zero(Complex{T});absolute_sum=zero(T)
        for index in 1:inner
            # The fused complex helper protects exponent range, but its
            # `ac-bd` and `ad+bc` combinations still round in T.  Mark any
            # sign pattern capable of cancellation so the exceptional block
            # is recomputed at the guarded precision below; otherwise an
            # exactly representable small component could disappear before
            # the outer dot-product cancellation check sees it.
            severe|=_complex_product_cancellation_possible(
                Complex{T}(left[row,index]),Complex{T}(right[index,column]))
            contribution=try
                _apply_prepared_exact_scale_product(
                    left[row,index],right[index,column],inverse_scale;
                    context="PI operator product in sector $p")
            catch error
                error isa ArgumentError||rethrow()
                severe=true
                break
            end
            value+=contribution;absolute_sum+=abs(contribution)
        end
        result[row,column]=value
        severe|=(!isfinite(value)||!isfinite(absolute_sum)||
            (!iszero(absolute_sum)&&absolute_sum>T(8)*abs(value)))
    end
    severe||return result

    span=_operator_product_exponent_span(left,T)+
         _operator_product_exponent_span(right,T)
    guard=max(precision(BigFloat),256,2*precision(T)+span+
              ndigits(big(max(inner,1));base=2)+32)
    setprecision(BigFloat,guard) do
        wide_left=Matrix{Complex{BigFloat}}(left)
        wide_right=Matrix{Complex{BigFloat}}(right)
        wide_product=wide_left*wide_right
        wide_scale=_prepare_exact_scale(BigFloat,one(BigInt),multiplicity,
            Val(true);context="wide PI operator product in sector $p")
        for index in eachindex(result,wide_product)
            scaled=_apply_prepared_exact_scale(
                wide_product[index],wide_scale;
                context="wide PI operator product in sector $p")
            result[index]=_convert_checked_operator_product(T,scaled,p)
        end
    end
    result
end

function *(a::PIOperator,b::PIOperator)
    _samebasis(a,b)
    T=promote_type(real(eltype(a.data)),real(eltype(b.data)))
    c=PIOperator(a.basis;T=T)
    for p in a.basis.sectors
        left=coefficient_block(a,p);right=coefficient_block(b,p)
        product=left*right
        block=try
            _divide_by_schur_multiplicity_scale(product,T,p)
        catch error
            error isa ArgumentError||rethrow()
            _fused_schur_operator_product(left,right,T,p)
        end
        coefficient_block(c,p).=block
    end
    c
end
function *(x::Number,a::PIOperator); PIOperator(a.basis,x.*a.data); end
*(a::PIOperator,x::Number)=x*a
+(a::PIOperator,b::PIOperator)=(_samebasis(a,b);PIOperator(a.basis,a.data+b.data))
-(a::PIOperator,b::PIOperator)=(_samebasis(a,b);PIOperator(a.basis,a.data-b.data))
-(a::PIOperator)=(-1)*a
adjoint(a::PIOperator)=begin c=copy(a); for p in a.basis.sectors coefficient_block(c,p).=coefficient_block(a,p)';end;c end
# Recursive sector evaluator for `iid_state`. It implements
# `D_lambda(rho) = C' * (rho ⊗ D_mu(rho)) * C` one removable corner at a
# time, retaining only the sector-sized parent and destination matrices.
function _iid_schur_block(A::AbstractMatrix{Complex{R}},lambda::Partition{D},
                          blocks::Dict{Partition{D},Matrix{Complex{R}}},
                          isometries::Dict{Tuple{Partition{D},Partition{D}},Array{R,3}}) where {R<:AbstractFloat,D}
    get!(blocks,lambda) do
        if iszero(weight(lambda))
            return reshape(Complex{R}[one(R)],1,1)
        end
        mu=remove_corner(lambda,first(removable_corners(lambda)))
        parent=_iid_schur_block(A,mu,blocks,isometries)
        U=get!(isometries,(mu,lambda)) do
            _path_isometry(Partition{D}[mu,lambda],R)
        end
        block_dimension,parent_dimension,local_dimension=size(U)
        local_dimension==size(A,1)||throw(ErrorException(
            "internal Schur-recurrence local dimension mismatch"))
        block=zeros(Complex{R},block_dimension,block_dimension)
        left_parent=zeros(Complex{R},block_dimension,parent_dimension)
        rotated_isometry=zeros(Complex{R},block_dimension,parent_dimension)
        contribution=zeros(Complex{R},block_dimension,block_dimension)
        for input_label in 1:local_dimension
            fill!(rotated_isometry,zero(Complex{R}))
            for output_label in 1:local_dimension
                coefficient=conj(A[input_label,output_label])
                iszero(coefficient)&&continue
                @views rotated_isometry .+= coefficient.*U[:,:,output_label]
            end
            @views mul!(left_parent,U[:,:,input_label],parent)
            mul!(contribution,left_parent,adjoint(rotated_isometry))
            block .+= contribution
        end
        block
    end
end

"""
    iid_state(basis, rho; atol=0, rtol=sqrt(eps(...)))

Construct the PI representation of `rho`⊗`basis.N` for a normalized
one-particle density matrix. Each physical Schur block is evaluated by the
one-box Clebsch--Gordan recurrence

`D_lambda(rho) = C' * (rho ⊗ D_mu(rho)) * C`,

where `mu` is any partition obtained by removing one corner from `lambda`.
The recurrence is polynomial in `N` at fixed local dimension and only forms
sector-sized matrices. It therefore also supports rank-deficient (including
pure) density matrices without a matrix logarithm, an eigendecomposition, or
a numerical regularizer.

All Schur sectors carrying nonzero weight must be present in `basis`; an
incomplete restricted basis is rejected instead of silently projecting and
renormalizing the product state. The input scalar precision is retained, and
the input is validated without normalization or eigenvalue clipping.
"""
function iid_state(b::PIBasis{D},rho::AbstractMatrix;atol::Real=0,
                   rtol::Real=sqrt(eps(_real_float_type(eltype(rho))))) where D
    size(rho)==(b.d,b.d)||throw(DimensionMismatch("rho must have size ($(b.d), $(b.d))"))
    atol>=0||throw(ArgumentError("atol must be nonnegative"))
    rtol>=0||throw(ArgumentError("rtol must be nonnegative"))

    R=_real_float_type(eltype(rho))
    hermiticity_error=norm(rho-rho',Inf)
    hermiticity_scale=norm(rho,Inf)
    hermiticity_error<=atol+rtol*hermiticity_scale||
        throw(ArgumentError("rho must be Hermitian: error=$hermiticity_error"))

    # Hermitian averaging only removes skew-Hermitian roundoff already accepted
    # by the requested tolerance; the validated matrix is not spectrally
    # clipped or normalized before entering the polynomial recurrence.
    A=Matrix{Complex{R}}((rho+rho')/2)
    spectral_scale=norm(A,Inf)
    positivity_tolerance=R(atol)+R(rtol)*spectral_scale
    _scalar_generic_psd_check(Hermitian(A),positivity_tolerance)||
        throw(ArgumentError("rho must be positive within tolerance $positivity_tolerance"))
    trace_value=real(LinearAlgebra.tr(A))
    abs(trace_value-one(trace_value))<=atol+rtol||
        throw(ArgumentError("rho must have trace one: trace=$trace_value"))
    W=_iid_amplitude_work_type(R)
    tensor_trace=W(trace_value)^b.N
    tensor_tolerance=W(atol)+W(rtol)
    isfinite(tensor_trace)&&abs(tensor_trace-one(W))<=tensor_tolerance||
        throw(ArgumentError(
            "the accepted one-particle trace error is amplified at N=$(b.N): " *
            "the tensor-power trace would be $tensor_trace (tolerance $tensor_tolerance); " *
            "supply a more accurately normalized density matrix or use a wider scalar type"))

    out=PIState(b;T=R)
    b.N==0 && begin
        coefficient_block(out,only(b.sectors))[1,1]=one(Complex{R})
        return out
    end

    blocks=Dict{Partition{D},Matrix{Complex{R}}}()
    isometries=Dict{Tuple{Partition{D},Partition{D}},Array{R,3}}()
    for sector in b.sectors
        block=_iid_schur_block(A,sector,blocks,isometries)
        coefficient_block(out,sector).=
            _multiply_by_schur_multiplicity_scale(block,R,sector)
    end

    product_trace=trace(out)
    abs(product_trace-one(product_trace))<=atol+rtol||
        throw(ArgumentError("basis omits Schur-sector support of rho⊗N: represented trace=$product_trace"))
    out
end
"""Construct `exp(-beta*H)/Z`, rejecting a non-Hermitian PI Hamiltonian."""
function thermal_state(H::PIOperator,beta::Real;atol::Real=0,rtol::Real=_default_rtol(H))
    ishermitian(H;atol=atol,rtol=rtol)||throw(ArgumentError("thermal_state requires a Hermitian Hamiltonian"))
    T=real(eltype(H.data));a=PIState(H.basis;T=T)
    for p in H.basis.sectors
        B=Matrix(physical_block(H,p))
        # Hermiticity was checked globally; averaging only removes tolerated
        # skew-Hermitian roundoff before the Hermitian matrix exponential.
        coefficient_block(a,p).=_multiply_by_schur_multiplicity_scale(
            exp(-beta*Hermitian((B+B')/2)),T,p)
    end
    normalize!(a)
end

"""
    local_kernel_operator(basis, X, Y; representation=:sparse,
                          cache=OneBodyGeometry(basis))

Construct the sparse PI-coordinate matrix of the local map
``rho -> sum_i X_i rho Y_i^dagger``. Reusing `cache` avoids rebuilding the
one-body CG geometry. The `representation` keyword is retained for API
compatibility; the current result is sparse.
"""
function local_kernel_operator(b::PIBasis,X,Y;representation=:sparse,cache=OneBodyGeometry(b))
    _check_geometry_basis(cache,b)
    T=promote_type(Complex{geometry_scalar_type(cache)},eltype(X),eltype(Y))
    I=Int[];J=Int[];V=T[]
    for (li,l) in pairs(b.sectors),(ni,n) in pairs(b.sectors)
        isempty(cache.connections[(li,ni)])&&continue
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        for a in 1:nl,bb in 1:nl,c in 1:nn,d in 1:nn
            z=local_kernel_element(cache,X,Y,l,a,bb,n,c,d); iszero(z)&&continue
            push!(I,b.offsets[li]+a-1+(bb-1)*nl);push!(J,b.offsets[ni]+c-1+(d-1)*nn);push!(V,z)
        end
    end
    sparse(I,J,V,length(b),length(b))
end
