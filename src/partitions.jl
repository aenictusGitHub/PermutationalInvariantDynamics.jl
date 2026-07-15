"""A partition padded to exactly `D` nonnegative parts."""
struct Partition{D}
    parts::NTuple{D,Int}
    function Partition(parts::NTuple{D,<:Integer}) where D
        p = ntuple(i -> Int(parts[i]), D)
        all(>=(0), p) || throw(ArgumentError("partition parts must be nonnegative"))
        issorted(p; rev=true) || throw(ArgumentError("partition parts must be nonincreasing"))
        new{D}(p)
    end
end
Partition(xs::AbstractVector{<:Integer}) = Partition(Tuple(xs))
show(io::IO, p::Partition) = print(io, "Partition", p.parts)
getindex(p::Partition, i::Integer) = p.parts[i]
length(::Partition{D}) where D = D

"""Return the number of boxes (the sum of all parts) of `p`."""
weight(p::Partition) = sum(p.parts)

"""Return the number of nonzero rows of the Young diagram `p`."""
length_nonzero(p::Partition) = count(!=(0), p.parts)

"""
    partitions(N, d)

Enumerate the partitions of `N` into exactly `d` nonnegative, nonincreasing
parts, padding with zeros. The result uses the package's descending
lexicographic sector order.
"""
function partitions(N::Integer, d::Integer)
    N >= 0 || throw(ArgumentError("N must be nonnegative"))
    d >= 1 || throw(ArgumentError("d must be positive"))
    out = Partition{d}[]
    function rec(prefix::Vector{Int}, left::Int, cap::Int)
        k = length(prefix)
        if k == d
            left == 0 && push!(out, Partition(Tuple(prefix)))
            return
        end
        for x in min(cap, left):-1:0
            push!(prefix, x); rec(prefix, left-x, x); pop!(prefix)
        end
    end
    rec(Int[], Int(N), Int(N))
    out
end

"""
    removable_corners(p)

Return the one-based row indices from which one box can be removed while
leaving a valid partition.
"""
removable_corners(p::Partition{D}) where D = [i for i in 1:D if p[i] > (i == D ? 0 : p[i+1])]

"""
    addable_corners(p)

Return the one-based row indices to which one box can be added while leaving a
valid length-`D` partition.
"""
addable_corners(p::Partition{D}) where D = [i for i in 1:D if i == 1 || p[i] < p[i-1]]

"""
    remove_corner(p, row)

Remove one box from the one-based `row`. Throw `ArgumentError` if that row is
not a removable corner.
"""
function remove_corner(p::Partition{D}, row::Integer) where D
    row in removable_corners(p) || throw(ArgumentError("row $row is not a removable corner of $p"))
    Partition(ntuple(i -> p[i] - (i == row), D))
end

"""
    add_corner(p, row)

Add one box to the one-based `row`. Throw `ArgumentError` if that row is not
an addable corner.
"""
function add_corner(p::Partition{D}, row::Integer) where D
    row in addable_corners(p) || throw(ArgumentError("row $row is not an addable corner of $p"))
    Partition(ntuple(i -> p[i] + (i == row), D))
end

"""
    minus_plus_neighbors(p)

Return, in sector order, the distinct partitions obtained by removing one box
from `p` and adding one box back. These are the sectors coupled to `p` by a
general local one-particle process.
"""
function minus_plus_neighbors(p::Partition)
    unique(sort!([add_corner(q, r) for i in removable_corners(p) for q in (remove_corner(p,i),)
                  for r in addable_corners(q)]; by=x->x.parts, rev=true))
end

"""
    reachable_sectors(p, n)

Return the sectors reachable from `p` after exactly `n` successive
remove-then-add neighbor steps. The returned partitions remain in descending
lexicographic sector order.
"""
function reachable_sectors(p::Partition, n::Integer)
    n >= 0 || throw(ArgumentError("p must be nonnegative"))
    s = Set([p])
    for _ in 1:n
        s = Set(q for x in s for q in minus_plus_neighbors(x))
    end
    sort!(collect(s); by=x->x.parts, rev=true)
end

"""
    symmetric_group_dimension(p)

Return the exact `BigInt` dimension ``f^p`` of the symmetric-group irrep with
Young shape `p`, evaluated with the hook-length formula.
"""
function symmetric_group_dimension(p::Partition)
    n = weight(p); n == 0 && return big(1)
    den = big(1)
    for i in 1:length(p), j in 1:p[i]
        below = count(k -> p[k] >= j, i+1:length(p))
        den *= p[i] - j + below + 1
    end
    factorial(big(n)) ÷ den
end

# Convert exact combinatorial ratios only after every cancellation has taken
# place.  `T(q)` already converts a Rational{BigInt} without first converting
# its numerator and denominator separately; the explicit checks make
# underflow/overflow of a nonzero final factor a hard error rather than a
# silently deleted or nonfinite PI coupling.
function _checked_exact_ratio(::Type{T},numerator_value::Integer,
                              denominator_value::Integer;
                              context::AbstractString="combinatorial factor") where T<:AbstractFloat
    isconcretetype(T)||throw(ArgumentError(
        "$context requires a concrete floating-point type, got $T"))
    numerator_value>=0||throw(ArgumentError("$context numerator must be nonnegative"))
    denominator_value>0||throw(ArgumentError("$context denominator must be positive"))
    iszero(numerator_value)&&return zero(T)
    exact_value=big(numerator_value)//big(denominator_value)
    value=try
        T(exact_value)
    catch error
        throw(ArgumentError("$context is not representable in $T: $(sprint(showerror,error))"))
    end
    isfinite(value)&&!iszero(value)||throw(ArgumentError(
        "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    # IEEE conversion rounds values just outside the mathematical finite range
    # back to `floatmax(T)` (and values below the least subnormal back to that
    # subnormal) in a narrow boundary interval.  Those are not representable
    # nonzero factors under the package's checked semantics.
    if T===Float16||T===Float32||T===Float64
        maximum_exact=Rational{BigInt}(floatmax(T))
        minimum_exact=Rational{BigInt}(nextfloat(zero(T)))
        exact_value<=maximum_exact&&exact_value>=minimum_exact||throw(ArgumentError(
            "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    end
    value
end

# Binary representation of an exact positive rational scale.  The `Root`
# parameter distinguishes q from sqrt(q) without a run-time branch in hot
# scalar applications.  `direct=true` is the ordinary small-coefficient path;
# otherwise `mantissa*2^exponent` is formed only jointly with the value being
# scaled, so a large combinatorial factor may cancel a small physical rate.
struct _PreparedExactScale{T<:AbstractFloat,Root}
    numerator::BigInt
    denominator::BigInt
    direct::Bool
    factor::T
    mantissa::T
    exponent::Int
end

function _exact_ratio_binary_parts(::Type{T},numerator_big::BigInt,
                                   denominator_big::BigInt) where T<:AbstractFloat
    exponent=ndigits(numerator_big;base=2)-ndigits(denominator_big;base=2)
    below_power=exponent>=0 ?
        numerator_big<(denominator_big<<exponent) :
        (numerator_big<<(-exponent))<denominator_big
    below_power&&(exponent-=1)
    scaled=exponent>=0 ? numerator_big//(denominator_big<<exponent) :
                         (numerator_big<<(-exponent))//denominator_big
    mantissa=T(scaled)
    isfinite(mantissa)&&mantissa>=one(T)&&mantissa<=T(2)||error(
        "internal error: exact-ratio mantissa left its bounded range")
    (;mantissa,exponent)
end

function _sqrt_exact_ratio_binary_parts(::Type{T},numerator_big::BigInt,
                                        denominator_big::BigInt) where T<:AbstractFloat
    ratio_parts=_exact_ratio_binary_parts(T,numerator_big,denominator_big)
    root_exponent=fld(ratio_parts.exponent,2)
    shift=2root_exponent
    scaled=shift>=0 ? numerator_big//(denominator_big<<shift) :
                      (numerator_big<<(-shift))//denominator_big
    bounded=T(scaled)
    isfinite(bounded)&&bounded>=one(T)&&bounded<=T(4)||error(
        "internal error: square-root mantissa left its bounded range")
    (;mantissa=sqrt(bounded),exponent=root_exponent)
end

function _prepare_exact_scale(::Type{T},numerator_value::Integer,
                              denominator_value::Integer,::Val{Root};
                              context::AbstractString) where {T<:AbstractFloat,Root}
    isconcretetype(T)||throw(ArgumentError(
        "$context requires a concrete floating-point type, got $T"))
    numerator_value>=0||throw(ArgumentError("$context numerator must be nonnegative"))
    denominator_value>0||throw(ArgumentError("$context denominator must be positive"))
    numerator_big=big(numerator_value);denominator_big=big(denominator_value)
    if iszero(numerator_big)
        return _PreparedExactScale{T,Root}(
            numerator_big,denominator_big,true,zero(T),zero(T),0)
    end
    direct_factor=try
        Root ? _checked_sqrt_exact_ratio(T,numerator_big,denominator_big;
                                         context) :
               _checked_exact_ratio(T,numerator_big,denominator_big;context)
    catch error
        error isa ArgumentError||rethrow()
        nothing
    end
    if direct_factor!==nothing
        return _PreparedExactScale{T,Root}(
            numerator_big,denominator_big,true,direct_factor,direct_factor,0)
    end
    parts=Root ? _sqrt_exact_ratio_binary_parts(T,numerator_big,denominator_big) :
                 _exact_ratio_binary_parts(T,numerator_big,denominator_big)
    _PreparedExactScale{T,Root}(numerator_big,denominator_big,false,zero(T),
                                parts.mantissa,parts.exponent)
end

_scale_real_type(::Type{Complex{T}}) where T<:AbstractFloat=T
_scale_real_type(::Type{T}) where T<:AbstractFloat=T
_scale_real_type(::Type{T}) where T=typeof(float(real(zero(T))))

function _strict_scaled_product_boundary(::Type{T},result::T,input::T,
        scale::_PreparedExactScale{T,false},context) where T<:AbstractFloat
    (T===Float16||T===Float32||T===Float64)||return result
    magnitude=abs(result)
    (magnitude==floatmax(T)||magnitude==nextfloat(zero(T)))||return result
    exact_magnitude=abs(Rational{BigInt}(input))*scale.numerator//scale.denominator
    minimum_exact=Rational{BigInt}(nextfloat(zero(T)))
    maximum_exact=Rational{BigInt}(floatmax(T))
    minimum_exact<=exact_magnitude<=maximum_exact||throw(ArgumentError(
        "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    result
end

function _strict_scaled_product_boundary(::Type{T},result::T,input::T,
        scale::_PreparedExactScale{T,true},context) where T<:AbstractFloat
    (T===Float16||T===Float32||T===Float64)||return result
    magnitude=abs(result)
    (magnitude==floatmax(T)||magnitude==nextfloat(zero(T)))||return result
    input_exact=abs(Rational{BigInt}(input))
    squared_magnitude=input_exact^2*scale.numerator//scale.denominator
    minimum_squared=Rational{BigInt}(nextfloat(zero(T)))^2
    maximum_squared=Rational{BigInt}(floatmax(T))^2
    minimum_squared<=squared_magnitude<=maximum_squared||throw(ArgumentError(
        "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    result
end

function _apply_prepared_exact_scale(input::Real,
        scale::_PreparedExactScale{T,Root};context::AbstractString) where {T<:AbstractFloat,Root}
    value=try
        T(input)
    catch error
        throw(ArgumentError("$context input is not representable in $T: $(sprint(showerror,error))"))
    end
    isfinite(value)||throw(ArgumentError("$context input must be finite"))
    iszero(scale.numerator)&&return zero(T)*value
    iszero(value)&&return value
    result = if scale.direct
        value*scale.factor
    else
        # Multiplication first preserves a least-subnormal result that a
        # downscale-first evaluation could round to zero before a mantissa
        # near two rescues it. Downscale first only when that multiplication
        # could overflow while a negative exponent rescues the final value.
        multiply_first=abs(value)<=floatmax(T)/scale.mantissa
        multiply_first ? ldexp(value*scale.mantissa,scale.exponent) :
                         ldexp(value,scale.exponent)*scale.mantissa
    end
    isfinite(result)&&!iszero(result)||throw(ArgumentError(
        "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    _strict_scaled_product_boundary(T,result,value,scale,context)
end

function _apply_prepared_exact_scale(input::Complex,
        scale::_PreparedExactScale{T,Root};context::AbstractString) where {T<:AbstractFloat,Root}
    complex(_apply_prepared_exact_scale(real(input),scale;context),
            _apply_prepared_exact_scale(imag(input),scale;context))
end

function _apply_prepared_exact_scale(input::AbstractArray,
        scale::_PreparedExactScale{T,Root};context::AbstractString) where {T<:AbstractFloat,Root}
    S=eltype(input)<:Complex ? Complex{T} : T
    output=similar(input,S)
    @inbounds for index in eachindex(input,output)
        output[index]=_apply_prepared_exact_scale(input[index],scale;context)
    end
    output
end

# Apply one prepared scale jointly with a product of two floating values. This
# is needed by Appendix-D gain maps: `left*right` may underflow before a large
# exact path weight rescales the physically finite result. Binary mantissas are
# combined first and the exponent is restored only once.
function _prepared_scale_binary_parts(scale::_PreparedExactScale{T}) where T
    value=scale.direct ? scale.factor : scale.mantissa
    mantissa,exponent=frexp(value)
    mantissa,exponent+(scale.direct ? 0 : scale.exponent)
end

function _scaled_real_product_parts(left::Real,right::Real,
        scale::_PreparedExactScale{T};context) where T<:AbstractFloat
    a=T(left);b=T(right)
    isfinite(a)&&isfinite(b)||throw(ArgumentError("$context inputs must be finite"))
    (iszero(a)||iszero(b)||iszero(scale.numerator))&&return (zero(T),0)
    ma,ea=frexp(a);mb,eb=frexp(b);ms,es=_prepared_scale_binary_parts(scale)
    (ma*mb*ms,ea+eb+es)
end

function _combine_binary_parts(first::Tuple{T,Int},second::Tuple{T,Int}) where T
    iszero(first[1])&&return second
    iszero(second[1])&&return first
    exponent=max(first[2],second[2])
    mantissa=ldexp(first[1],first[2]-exponent)+
             ldexp(second[1],second[2]-exponent)
    iszero(mantissa)&&return (zero(T),0)
    normalized,shift=frexp(mantissa)
    (normalized,exponent+shift)
end

function _checked_binary_parts_value(parts::Tuple{T,Int},context) where T<:AbstractFloat
    iszero(parts[1])&&return zero(T)
    result=ldexp(parts[1],parts[2])
    isfinite(result)&&!iszero(result)||throw(ArgumentError(
        "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    result
end


function _strict_scaled_exact_input_boundary(::Type{T},result::T,
        exact_input::Rational{BigInt},scale::_PreparedExactScale{T,false},
        context) where T<:AbstractFloat
    (T===Float16||T===Float32||T===Float64)||return result
    magnitude=abs(result)
    (magnitude==floatmax(T)||magnitude==nextfloat(zero(T)))||return result
    exact_magnitude=abs(exact_input)*scale.numerator//scale.denominator
    Rational{BigInt}(nextfloat(zero(T)))<=exact_magnitude<=
        Rational{BigInt}(floatmax(T))||throw(ArgumentError(
            "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    result
end

function _strict_scaled_exact_input_boundary(::Type{T},result::T,
        exact_input::Rational{BigInt},scale::_PreparedExactScale{T,true},
        context) where T<:AbstractFloat
    (T===Float16||T===Float32||T===Float64)||return result
    magnitude=abs(result)
    (magnitude==floatmax(T)||magnitude==nextfloat(zero(T)))||return result
    squared_magnitude=exact_input^2*scale.numerator//scale.denominator
    Rational{BigInt}(nextfloat(zero(T)))^2<=squared_magnitude<=
        Rational{BigInt}(floatmax(T))^2||throw(ArgumentError(
            "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    result
end

function _apply_prepared_exact_scale_product(left::Real,right::Real,
        scale::_PreparedExactScale{T};context::AbstractString) where T<:AbstractFloat
    result=_checked_binary_parts_value(
        _scaled_real_product_parts(left,right,scale;context),context)
    magnitude=abs(result)
    if (T===Float16||T===Float32||T===Float64)&&
       (magnitude==floatmax(T)||magnitude==nextfloat(zero(T)))
        exact_input=Rational{BigInt}(T(left))*Rational{BigInt}(T(right))
        _strict_scaled_exact_input_boundary(T,result,exact_input,scale,context)
    end
    result
end

function _apply_prepared_exact_scale_product(left::Complex,right::Complex,
        scale::_PreparedExactScale{T};context::AbstractString) where T<:AbstractFloat
    ac=_scaled_real_product_parts(real(left),real(right),scale;context)
    bd=_scaled_real_product_parts(imag(left),imag(right),scale;context)
    ad=_scaled_real_product_parts(real(left),imag(right),scale;context)
    bc=_scaled_real_product_parts(imag(left),real(right),scale;context)
    real_parts=_combine_binary_parts(ac,(-bd[1],bd[2]))
    imaginary_parts=_combine_binary_parts(ad,bc)
    real_result=_checked_binary_parts_value(real_parts,context)
    imaginary_result=_checked_binary_parts_value(imaginary_parts,context)
    fixed_ieee=T===Float16||T===Float32||T===Float64
    real_endpoint=fixed_ieee&&(
        abs(real_result)==floatmax(T)||abs(real_result)==nextfloat(zero(T)))
    imaginary_endpoint=fixed_ieee&&(
        abs(imaginary_result)==floatmax(T)||
        abs(imaginary_result)==nextfloat(zero(T)))
    if fixed_ieee&&(real_endpoint||imaginary_endpoint)
        a=Rational{BigInt}(T(real(left)));b=Rational{BigInt}(T(imag(left)))
        c=Rational{BigInt}(T(real(right)));d=Rational{BigInt}(T(imag(right)))
        real_endpoint&&_strict_scaled_exact_input_boundary(
            T,real_result,a*c-b*d,scale,context)
        imaginary_endpoint&&_strict_scaled_exact_input_boundary(
            T,imaginary_result,a*d+b*c,scale,context)
    end
    complex(real_result,imaginary_result)
end

function _checked_mul_exact_ratio(::Type{T},value,numerator_value::Integer,
                                  denominator_value::Integer;
                                  context::AbstractString="scaled combinatorial factor") where T<:AbstractFloat
    scale=_prepare_exact_scale(T,numerator_value,denominator_value,Val(false);context)
    _apply_prepared_exact_scale(value,scale;context)
end
function _checked_mul_exact_ratio(value,numerator_value::Integer,
                                  denominator_value::Integer;
                                  context::AbstractString="scaled combinatorial factor")
    T=_scale_real_type(value isa AbstractArray ? eltype(value) : typeof(value))
    _checked_mul_exact_ratio(T,value,numerator_value,denominator_value;context)
end

function _checked_mul_sqrt_exact_ratio(::Type{T},value,numerator_value::Integer,
                                       denominator_value::Integer;
                                       context::AbstractString="scaled square root of combinatorial factor") where T<:AbstractFloat
    scale=_prepare_exact_scale(T,numerator_value,denominator_value,Val(true);context)
    _apply_prepared_exact_scale(value,scale;context)
end
function _checked_mul_sqrt_exact_ratio(value,numerator_value::Integer,
                                       denominator_value::Integer;
                                       context::AbstractString="scaled square root of combinatorial factor")
    T=_scale_real_type(value isa AbstractArray ? eltype(value) : typeof(value))
    _checked_mul_sqrt_exact_ratio(T,value,numerator_value,denominator_value;context)
end

# Return sqrt(numerator/denominator) without ever converting the unscaled
# rational.  If e=floor(log2(q)), q/2^(2*floor(e/2)) lies in [1,4), so only a
# bounded exact rational is converted to T.  `ldexp` restores the binary scale
# after the square root.  This is the large-coefficient fallback used when an
# exact binomial, multinomial, or Schur multiplicity exceeds floatmax(T) even
# though its square root remains representable.
function _checked_sqrt_exact_ratio(::Type{T},numerator_value::Integer,
                                   denominator_value::Integer;
                                   context::AbstractString="square root of combinatorial factor") where T<:AbstractFloat
    isconcretetype(T)||throw(ArgumentError(
        "$context requires a concrete floating-point type, got $T"))
    numerator_value>=0||throw(ArgumentError("$context numerator must be nonnegative"))
    denominator_value>0||throw(ArgumentError("$context denominator must be positive"))
    iszero(numerator_value)&&return zero(T)
    numerator_big=big(numerator_value);denominator_big=big(denominator_value)
    exponent=ndigits(numerator_big;base=2)-ndigits(denominator_big;base=2)
    below_power = exponent>=0 ?
        numerator_big < (denominator_big<<exponent) :
        (numerator_big<<(-exponent)) < denominator_big
    below_power&&(exponent-=1)
    root_exponent=fld(exponent,2)
    shift=2root_exponent
    scaled = if shift>=0
        numerator_big//(denominator_big<<shift)
    else
        (numerator_big<<(-shift))//denominator_big
    end
    bounded=try
        T(scaled)
    catch error
        throw(ArgumentError("$context is not representable in $T: $(sprint(showerror,error))"))
    end
    isfinite(bounded)&&bounded>zero(T)||error(
        "internal error: binary-scaled combinatorial ratio left its bounded range")
    value=ldexp(sqrt(bounded),root_exponent)
    isfinite(value)&&!iszero(value)||throw(ArgumentError(
        "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    if T===Float16||T===Float32||T===Float64
        exact_ratio=numerator_big//denominator_big
        minimum_squared=Rational{BigInt}(nextfloat(zero(T)))^2
        maximum_squared=Rational{BigInt}(floatmax(T))^2
        minimum_squared<=exact_ratio<=maximum_squared||throw(ArgumentError(
            "$context is outside the nonzero finite range of $T; use a wider scalar type"))
    end
    value
end

function _checked_sqrt_exact_integer(::Type{T},value::Integer;
        context::AbstractString="square root of combinatorial coefficient") where T<:AbstractFloat
    value>=0||throw(ArgumentError("$context must be nonnegative"))
    iszero(value)&&return zero(T)
    # Preserve the old allocation-light path whenever the exact integer fits.
    # Binary scaling is needed only after this conversion ceases to be finite.
    direct=try
        T(value)
    catch
        T(Inf)
    end
    # Conversion immediately above `floatmax(T)` may round back to the finite
    # endpoint.  In that boundary band use exact binary scaling so the square
    # root is rounded from the true integer, not from the saturated endpoint.
    direct_in_range=if T===Float16||T===Float32||T===Float64
        if isfinite(direct)&&direct<floatmax(T)
            true
        elseif direct==floatmax(T)
            big(value)<=Rational{BigInt}(floatmax(T))
        else
            false
        end
    else
        isfinite(direct)
    end
    direct_in_range&&isfinite(direct)&&!iszero(direct)&&return sqrt(direct)
    _checked_sqrt_exact_ratio(T,value,one(value);context=context)
end

# Exact branching factor |lambda| f^mu/f^lambda for removing one corner.
# The shifted-row formula is O(d) and avoids constructing either (possibly
# enormous) hook dimension.  Keeping this rational exact is the key to stable
# one- and p-body geometry prefactors.
function _one_box_branch_weight(lambda::Partition{D},mu::Partition{D}) where D
    weight(mu)==weight(lambda)-1||throw(ArgumentError(
        "$mu must contain exactly one box fewer than $lambda"))
    row=findfirst(index->lambda[index]==mu[index]+1,1:D)
    row===nothing&&throw(ArgumentError("$mu is not obtained by removing one box from $lambda"))
    all(index->index==row ? lambda[index]==mu[index]+1 :
                              lambda[index]==mu[index],1:D)||throw(ArgumentError(
        "$mu is not obtained by removing one box from $lambda"))
    row in removable_corners(lambda)||throw(ArgumentError(
        "$mu is not obtained by removing a removable corner from $lambda"))
    shifted=ntuple(index->big(lambda[index])+D-index,D)
    pivot=shifted[row]
    result=pivot//big(1)
    for index in 1:row-1
        difference=shifted[index]-pivot
        result*=((difference+1)//difference)
    end
    for index in row+1:D
        difference=pivot-shifted[index]
        result*=((difference-1)//difference)
    end
    result
end

# Exact weight binomial(N,p) f^center/f^endpoint associated with one ordered
# removal path stored as center -> ... -> endpoint.
function _subset_path_weight(path::AbstractVector{<:Partition})
    isempty(path)&&throw(ArgumentError("a removal path cannot be empty"))
    result=big(1)//big(1)
    for index in 1:length(path)-1
        result*=_one_box_branch_weight(path[index+1],path[index])
    end
    result/factorial(big(length(path)-1))
end

"""
    exact_binomial(n, k)

Return `binomial(n,k)` as an exact `BigInt`, without machine-integer overflow.
`n` must be nonnegative; an out-of-range `k` returns zero, following Julia's
`binomial` convention.
"""
function exact_binomial(n::Integer,k::Integer)
    n>=0||throw(ArgumentError("n must be nonnegative"))
    binomial(big(n),big(k))
end

"""
    exact_multinomial(counts)

Return ``(sum_i n_i)!/prod_i n_i!`` as an exact `BigInt`. Sequential exact
binomials are used, so no factorial intermediate exceeds the final
multinomial coefficient. Numerical PI kernels combine or scale such factors
before floating conversion rather than converting this potentially enormous
integer directly.
"""
function exact_multinomial(counts)
    collected=collect(counts)
    all(count->count isa Integer&&count>=0,collected)||throw(ArgumentError(
        "multinomial counts must be nonnegative integers"))
    remaining=sum(big(count) for count in collected;init=big(0))
    coefficient=big(1)
    for count in collected
        count_big=big(count)
        coefficient*=exact_binomial(remaining,count_big)
        remaining-=count_big
    end
    coefficient
end

"""
    unitary_group_dimension(p)

Return the exact `BigInt` dimension of the `U(D)` irrep labeled by the
length-`D` partition `p`, evaluated with the Weyl dimension formula.
"""
function unitary_group_dimension(p::Partition{D}) where D
    q = big(1)//big(1)
    for i in 1:D-1, j in i+1:D
        q *= (big(p[i]-p[j]+j-i)//big(j-i))
    end
    denominator(q) == 1 || error("internal Weyl-dimension error")
    numerator(q)
end

"""
    commutant_dimension(N, d)

Return the exact number of PI operator coordinates for `N` identical
`d`-level systems, ``binomial(N+d^2-1,N)``. This is the dimension of a complete
`PIBasis`, not the Hilbert-space dimension.
"""
commutant_dimension(N::Integer, d::Integer) = binomial(big(N)+big(d)^2-1, big(N))

"""Legacy alias for [`commutant_dimension`](@ref)."""
estimate_basis_size(N::Integer, d::Integer) = commutant_dimension(N,d)

_bigfloat_component_count(::Type)=0
_bigfloat_component_count(::Type{BigFloat})=1
_bigfloat_component_count(::Type{Complex{BigFloat}})=2

# Per-array-element retained-storage bound shared by public memory estimates.
# Fixed-size isbits scalars preserve exact `sizeof(T)` accounting. BigFloat's
# type does not encode its MPFR precision, so callers state the assumption.
# The bound adds a complete requested-precision limb allocation and object/
# header allowance to `summarysize` of an ordinary scalar. It does not mutate
# process precision and is deliberately conservative rather than an ABI
# promise. Generic heap-backed values receive a padded sample-based estimate,
# since their payload need not be bounded by their type.
function _scalar_retained_bytes(::Type{T};
                                bigfloat_precision::Integer=precision(BigFloat)) where T
    isconcretetype(T)||throw(ArgumentError(
        "scalar-storage estimates require a concrete type, got $T"))
    components=_bigfloat_component_count(T)
    if iszero(components)
        isbitstype(T)&&return big(sizeof(T))
        sample=try
            zero(T)
        catch error
            throw(ArgumentError(
                "cannot estimate retained scalar storage for $T: $(sprint(showerror,error))"))
        end
        return big(sizeof(Ptr{Cvoid}))+2big(Base.summarysize(sample))
    end
    2<=bigfloat_precision<=typemax(Int)||throw(ArgumentError(
        "bigfloat_precision must lie between 2 and typemax(Int)"))
    requested_precision=Int(bigfloat_precision)
    sample_bytes=Base.summarysize(zero(BigFloat))
    word_bytes=sizeof(UInt)
    limb_bytes=big(cld(requested_precision,8word_bytes))*word_bytes
    component_bound=big(sample_bytes)+limb_bytes+4word_bytes
    box_allowance=big(sizeof(Ptr{Cvoid})+4word_bytes)
    box_allowance+components*component_bound
end

function _scalar_storage_estimate(::Type{T}) where T
    !iszero(_bigfloat_component_count(T))&&return :conservative_retained_bound
    isbitstype(T) ? :exact_inline : :sample_based_retained_estimate
end
_scalar_precision_assumption(::Type{T},p) where T=
    iszero(_bigfloat_component_count(T)) ? nothing : Int(p)

"""
    estimate_memory(N, d; T=ComplexF64,
                    bigfloat_precision=precision(BigFloat))

Legacy estimate for one complete PI coordinate vector; prefer
[`estimate_state_bytes`](@ref) for an already constructed, possibly
sector-restricted basis. Fixed-size isbits values retain the historical exact
inline formula; BigFloat values use a conservative bound at the stated
precision.
"""
estimate_memory(N::Integer,d::Integer;T=ComplexF64,
                bigfloat_precision::Integer=precision(BigFloat))=
    commutant_dimension(N,d)*_scalar_retained_bytes(T;bigfloat_precision)
