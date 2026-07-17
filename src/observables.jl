"""
    expectation(rho, A)

Return the Hilbert--Schmidt expectation `tr(A' * rho)` of a `PIOperator` on
the exact same `PIBasis`. For a Hermitian observable this is its ordinary
quantum expectation value.
"""
expectation(rho::PIState,A::PIOperator)=(_samebasis(rho,A);dot(A.data,rho.data))

"""Return `real(expectation(rho, A*A) - expectation(rho, A)^2)`."""
variance(rho,A)=real(expectation(rho,A*A)-expectation(rho,A)^2)

"""Return the symmetrized covariance of two PI observables."""
function covariance(rho,A::PIOperator,B::PIOperator)
    R=_real_float_type(promote_type(eltype(rho.data),eltype(A.data),eltype(B.data)))
    expectation(rho,(A*B+B*A)*(one(R)/2))-expectation(rho,A)*expectation(rho,B)
end

# Multiplicity-weighted Schur coordinates are the numerically natural
# coordinates for state analysis:
#
#     rho_bar_nu = f^nu rho_nu = sqrt(f^nu) C_nu.
#
# Their trace is the sector probability, so their entries remain O(1) for a
# normalized state even when the physical eigenvalues rho_nu are far below the
# working type's absolute tolerance (or its nonzero range).  Retain the old
# single floating multiply for ordinary sectors.  Only the exceptional case
# where sqrt(f^nu) itself is not representable uses the exact binary-scaled
# value-times-factor helper from `partitions.jl`.
function _multiplicity_weighted_block(A::AbstractPIOperator,p::Partition)
    C=coefficient_block(A,p);R=_real_float_type(eltype(A.data))
    scale=try
        _schur_multiplicity_scale(R,p)
    catch error
        error isa ArgumentError||rethrow()
        nothing
    end
    if scale!==nothing
        B=Matrix(C)
        B.*=scale
        _ordinary_scaled_value_safe(B,C)&&return B
    end
    f=symmetric_group_dimension(p)
    Matrix(_checked_mul_sqrt_exact_ratio(C,f,one(f);
        context="multiplicity-weighted Schur block for $p"))
end

function _multiplicity_weighted_scalar(value,p::Partition,::Type{R};
                                       context="multiplicity-weighted Schur contraction") where R<:AbstractFloat
    scale=try
        _schur_multiplicity_scale(R,p)
    catch error
        error isa ArgumentError||rethrow()
        nothing
    end
    if scale!==nothing
        result=value*scale
        _ordinary_scaled_value_safe(result,value)&&return result
    end
    _checked_mul_sqrt_exact_ratio(value,symmetric_group_dimension(p),big(1);
        context=context)
end

function _store_physical_block!(C::AbstractMatrix,H::AbstractMatrix,
                                p::Partition,::Type{R};
                                context="stored physical Schur block") where R<:AbstractFloat
    scale=try
        _schur_multiplicity_scale(R,p)
    catch error
        error isa ArgumentError||rethrow()
        nothing
    end
    if scale!==nothing
        C.=scale.*H
        _ordinary_scaled_value_safe(C,H)&&return C
    end
    f=symmetric_group_dimension(p)
    C.=_checked_mul_sqrt_exact_ratio(H,f,one(f);context=context)
    C
end

# `log(f^nu)` is needed by entropy, but converting f^nu itself can overflow
# long before the logarithm does.  Keep direct conversion as the small-system
# fast path and otherwise convert only a bounded binary mantissa.
function _log_schur_multiplicity(::Type{R},p::Partition) where R<:AbstractFloat
    f=symmetric_group_dimension(p)
    direct=try
        R(f)
    catch
        R(Inf)
    end
    isfinite(direct)&&return log(direct)
    exponent=ndigits(f;base=2)-1
    # A Float16 exponent can overflow even while exponent*log(2), the desired
    # logarithm, remains representable.  Evaluate that exceptional path in a
    # wider work type and only then perform a strict range-checked conversion.
    W=R===Float16 ? Float64 : R
    mantissa=W(f//(big(1)<<exponent))
    wide_result=log(mantissa)+W(exponent)*log(W(2))
    isfinite(wide_result)&&wide_result<=W(floatmax(R))||throw(ArgumentError(
        "logarithm of the sector multiplicity for $p is not representable in $R; use a wider scalar type"))
    result=R(wide_result)
    isfinite(result)&&(!iszero(wide_result)||!iszero(result))||throw(ArgumentError(
        "logarithm of the sector multiplicity for $p is not representable in $R; use a wider scalar type"))
    result
end

function _exact_particle_count(::Type{R},N::T) where {R<:AbstractFloat,T<:Base.BitInteger}
    count=try
        R(N)
    catch
        R(Inf)
    end
    isfinite(count)||return count,false
    roundtrip=try
        T(count)
    catch
        return count,false
    end
    count,roundtrip==N
end

function _exact_particle_count(::Type{R},N::Integer) where R<:AbstractFloat
    count=try
        R(N)
    catch
        R(Inf)
    end
    exact=isfinite(count)&&isinteger(count)&&BigInt(count)==big(N)
    count,exact
end

function _divide_by_particle_pair_factor(value,N::Integer;
                                         ordered_distinct::Bool,
                                         context::AbstractString)
    N>=0||throw(ArgumentError("particle count must be nonnegative"))
    R=_real_float_type(typeof(value))
    count,count_is_exact=_exact_particle_count(R,N)
    denominator=ordered_distinct ? count*(count-one(R)) : count*count
    if count_is_exact&&isfinite(denominator)&&!iszero(denominator)
        result=value/denominator
        _ordinary_scaled_value_safe(result,value)&&return result
    end
    exact_denominator=ordered_distinct ? big(N)*big(N-1) : big(N)^2
    exact_denominator>0||throw(ArgumentError("$context has a zero denominator"))
    _checked_mul_exact_ratio(value,one(BigInt),exact_denominator;context=context)
end

function _inverse_particle_count(::Type{R},N::Integer) where R<:AbstractFloat
    N>0||throw(ArgumentError("particle count must be positive"))
    count,count_is_exact=_exact_particle_count(R,N)
    if count_is_exact
        inverse=inv(count)
        _ordinary_scaled_value_safe(inverse,one(R))&&return inverse
    end
    _checked_exact_ratio(R,one(BigInt),big(N);
        context="inverse particle count in the one-body RDM")
end

function _checked_particle_count_ratio(value::Real,numerator_value::Integer,
                                       denominator_value::Integer,
                                       ::Type{R};context::AbstractString) where
        R<:AbstractFloat
    denominator_value>0||throw(ArgumentError("$context has a zero denominator"))
    source=R(value)
    isfinite(source)||throw(ArgumentError("$context requires a finite input"))
    iszero(source)&&return source
    exact=Rational{BigInt}(source)*big(numerator_value)//big(denominator_value)
    result=try
        R(exact)
    catch caught
        throw(ArgumentError("$context is not representable in $R: " *
                            sprint(showerror,caught)))
    end
    isfinite(result)&&!iszero(result)||throw(ArgumentError(
        "$context is outside the nonzero finite range of $R; use a wider scalar type"))
    if R===Float16||R===Float32||R===Float64
        magnitude=abs(exact)
        Rational{BigInt}(nextfloat(zero(R)))<=magnitude<=
            Rational{BigInt}(floatmax(R))||throw(ArgumentError(
                "$context is outside the nonzero finite range of $R; use a wider scalar type"))
    end
    result
end

function _checked_particle_count_ratio(value::Complex,numerator_value::Integer,
                                       denominator_value::Integer,
                                       ::Type{R};context::AbstractString) where
        R<:AbstractFloat
    complex(_checked_particle_count_ratio(real(value),numerator_value,
                denominator_value,R;context),
            _checked_particle_count_ratio(imag(value),numerator_value,
                denominator_value,R;context))
end

function _divide_by_particle_count(value,N::Integer;
                                   prefactor::Integer=1,
                                   context::AbstractString)
    N>0||throw(ArgumentError("particle count must be positive"))
    R=_real_float_type(typeof(value));count,count_is_exact=_exact_particle_count(R,N)
    if count_is_exact
        result=(R(prefactor)*value)/count
        _ordinary_scaled_value_safe(result,value)&&return result
    end
    _checked_particle_count_ratio(value,prefactor,N,R;context)
end

function _multiply_by_particle_count(value,N::Integer;
                                     context::AbstractString)
    N>=0||throw(ArgumentError("particle count must be nonnegative"))
    R=_real_float_type(typeof(value));count,count_is_exact=_exact_particle_count(R,N)
    if count_is_exact
        result=count*value
        (iszero(N)||_ordinary_scaled_value_safe(result,value))&&return result
    end
    _checked_particle_count_ratio(value,N,one(N),R;context)
end

function _weighted_sector_eigen(rho::PIState,p::Partition;
                                atol::Real=_analysis_atol(rho),
                                rtol::Real=_state_rtol(rho),
                                operation::AbstractString="multiplicity-weighted density-block spectral analysis")
    B=_multiplicity_weighted_block(rho,p)
    R=_real_float_type(eltype(B));scale=max(norm(B,Inf),zero(R))
    tolerance=R(atol)+R(rtol)*scale
    herr=norm(B-B',Inf)
    herr<=tolerance||throw(ArgumentError(
        "state is not Hermitian in sector $p: weighted error=$herr, tolerance=$tolerance"))
    E=_hermitian_eigen(Hermitian((B+B')/2);operation=operation)
    spectral_scale=maximum(abs,E.values)
    spectral_tolerance=R(atol)+R(rtol)*spectral_scale
    minimum(E.values)>=-spectral_tolerance||throw(ArgumentError(
        "state has a negative eigenvalue in sector $p in multiplicity-weighted coordinates"))
    # Denominators and numerical-rank decisions are relative to this sector,
    # not to its global probability.  Consequently a sector of weight 1e-30
    # is treated exactly like a normalized sector after factoring out that
    # weight; the public `atol` remains a global state-validation tolerance.
    zero_tolerance=R(32)*R(max(length(E.values),1))*eps(R)*spectral_scale
    (;values=E.values,vectors=E.vectors,tolerance=spectral_tolerance,
      zero_tolerance,block=B)
end

function _check_local_observable(rho::PIState,X::AbstractMatrix)
    size(X)==(rho.basis.d,rho.basis.d)||throw(DimensionMismatch(
        "local observable must be $(rho.basis.d)×$(rho.basis.d)"))
end

"""
    CollectiveObservablePlan(basis, X; cache=OneBodyGeometry(basis))

Prepare the physical Schur blocks of the collective observable
`sum_n X^(n)`. Treat the prepared blocks as read-only and reuse the plan for
many states on the exact same `PIBasis`, avoiding both representation-geometry
construction and block assembly on every expectation, variance, covariance,
or QFI call.

When preparing several observables, construct one `OneBodyGeometry` and pass
it through `cache`.  The plan retains only one matrix per Schur sector, not the
larger CG geometry.  A one-off contraction can instead receive the geometry
directly through `cache=` without retaining prepared observable blocks.
"""
struct CollectiveObservablePlan{T,B<:PIBasis}
    basis::B
    local_operator::Matrix{T}
    blocks::Vector{Matrix{T}}
end
show(io::IO,p::CollectiveObservablePlan)=print(io,"CollectiveObservablePlan(N=$(p.basis.N), d=$(p.basis.d), sectors=$(length(p.blocks)))")

function CollectiveObservablePlan(b::PIBasis,X::AbstractMatrix;cache=nothing)
    size(X)==(b.d,b.d)||throw(DimensionMismatch("local observable must be $(b.d)×$(b.d)"))
    cache===nothing&&(cache=OneBodyGeometry(b;T=_real_float_type(eltype(X))))
    _check_geometry_basis(cache,b)
    T=promote_type(Complex{geometry_scalar_type(cache)},eltype(X));local_operator=Matrix{T}(X)
    blocks=Matrix{T}[collective_block(b,local_operator,p;cache=cache) for p in b.sectors]
    CollectiveObservablePlan{T,typeof(b)}(b,local_operator,blocks)
end

collective_block(plan::CollectiveObservablePlan,p::Partition)=plan.blocks[_sidx(plan.basis,p)]
function collective_operator(plan::CollectiveObservablePlan)
    T=_real_float_type(eltype(plan.local_operator));A=PIOperator(plan.basis;T=T)
    for (s,p) in pairs(plan.basis.sectors)
        _store_physical_block!(coefficient_block(A,p),plan.blocks[s],p,T;
            context="stored collective-observable block for sector $p")
    end
    A
end

function _check_collective_plan(plan::CollectiveObservablePlan,b::PIBasis,X=nothing)
    plan.basis===b||throw(ArgumentError("CollectiveObservablePlan was prepared for a different PIBasis"))
    X===nothing||(size(X)==size(plan.local_operator)&&X==plan.local_operator)||
        throw(ArgumentError("CollectiveObservablePlan was prepared for a different local observable"))
    plan
end

function _resolve_collective_plan(rho::PIState,X,cache,plan)
    _check_local_observable(rho,X)
    if plan===nothing
        cache===nothing&&(cache=OneBodyGeometry(rho.basis;T=promote_type(
            _real_float_type(eltype(rho.data)),_real_float_type(eltype(X)))))
        return CollectiveObservablePlan(rho.basis,X;cache=cache)
    end
    plan isa CollectiveObservablePlan||throw(ArgumentError("plan must be a CollectiveObservablePlan"))
    cache===nothing||throw(ArgumentError("provide either cache or plan, not both"))
    _check_collective_plan(plan,rho.basis,X)
end

function _resolve_collective_plans(rho::PIState,generators,cache,plans)
    gs=collect(generators)
    if plans===nothing&&all(g->g isa CollectiveObservablePlan,gs)
        cache===nothing||throw(ArgumentError("prepared generators cannot be combined with a cache"))
        for g in gs;_check_collective_plan(g,rho.basis);end
        return gs,gs
    end
    if plans===nothing
        if cache===nothing
            types=(_real_float_type(eltype(g)) for g in gs)
            T=foldl(promote_type,types;init=_real_float_type(eltype(rho.data)))
            cache=OneBodyGeometry(rho.basis;T=T)
        end
        return gs,[CollectiveObservablePlan(rho.basis,g;cache=cache) for g in gs]
    end
    cache===nothing||throw(ArgumentError("provide either cache or plans, not both"))
    ps=collect(plans);length(ps)==length(gs)||throw(DimensionMismatch("one CollectiveObservablePlan is required per generator"))
    for (g,p) in zip(gs,ps)
        p isa CollectiveObservablePlan||throw(ArgumentError("plans must contain CollectiveObservablePlan objects"))
        _check_collective_plan(p,rho.basis,g)
    end
    gs,ps
end

"""
    collective_moments(rho, X)

Return `(mean, second_moment)` for the collective observable
`Xc = sum(n=1:N) X^(n)`, without assembling `Xc` as a flattened `PIOperator`.
The equation-(31) physical blocks are constructed once and contracted as

`sum_nu f^nu * tr(rho_nu * Xc_nu)` and
`sum_nu f^nu * tr(rho_nu * Xc_nu^2)`.
"""
function collective_moments(rho::PIState,plan::CollectiveObservablePlan)
    _check_collective_plan(plan,rho.basis)
    T=promote_type(eltype(rho.data),eltype(plan.local_operator));mean=zero(T);second=zero(T)
    for (s,p) in pairs(rho.basis.sectors)
        B=_multiplicity_weighted_block(rho,p);G=plan.blocks[s]
        mean += LinearAlgebra.tr(B*G)
        second += LinearAlgebra.tr(B*G*G)
    end
    (;mean,second_moment=second)
end
function collective_moments(rho::PIState,X::AbstractMatrix;cache=nothing,plan=nothing)
    collective_moments(rho,_resolve_collective_plan(rho,X,cache,plan))
end

"""
    collective_expectation(rho, X)

Efficiently return the expectation value of `sum_n X^(n)` directly from its
physical Schur blocks.
"""
function collective_expectation(rho::PIState,plan::CollectiveObservablePlan)
    _check_collective_plan(plan,rho.basis)
    T=promote_type(eltype(rho.data),eltype(plan.local_operator));mean=zero(T)
    for (s,p) in pairs(rho.basis.sectors)
        B=_multiplicity_weighted_block(rho,p);G=plan.blocks[s]
        mean += LinearAlgebra.tr(B*G)
    end
    mean
end
function collective_expectation(rho::PIState,X::AbstractMatrix;cache=nothing,plan=nothing)
    collective_expectation(rho,_resolve_collective_plan(rho,X,cache,plan))
end

"""
    collective_variance(rho, X; atol=_analysis_atol(rho))

Efficiently return the variance of the Hermitian collective observable
`sum_n X^(n)`. Tiny imaginary roundoff up to `atol` is discarded; a larger
imaginary component raises an error.
"""
function collective_variance(rho::PIState,plan::CollectiveObservablePlan;
                             atol::Real=_analysis_atol(rho))
    LinearAlgebra.ishermitian(plan.local_operator)||throw(ArgumentError("collective variance requires a Hermitian local observable"))
    m=collective_moments(rho,plan);v=m.second_moment-m.mean^2
    abs(imag(v))<=atol||throw(ArgumentError("variance has a non-negligible imaginary component $(imag(v))"))
    real(v)
end
function collective_variance(rho::PIState,X::AbstractMatrix;
                             atol::Real=_analysis_atol(rho),cache=nothing,
                             plan=nothing)
    collective_variance(rho,_resolve_collective_plan(rho,X,cache,plan);atol=atol)
end

"""Symmetrized covariance of two collective one-particle observables."""
function collective_covariance(rho::PIState,X::CollectiveObservablePlan,Y::CollectiveObservablePlan)
    _check_collective_plan(X,rho.basis);_check_collective_plan(Y,rho.basis)
    T=promote_type(eltype(rho.data),eltype(X.local_operator),eltype(Y.local_operator))
    ex=zero(T);ey=zero(T);sym=zero(T);Rtype=_real_float_type(T)
    for (s,p) in pairs(rho.basis.sectors)
        B=_multiplicity_weighted_block(rho,p);GX=X.blocks[s];GY=Y.blocks[s]
        ex+=LinearAlgebra.tr(B*GX);ey+=LinearAlgebra.tr(B*GY)
        sym+=LinearAlgebra.tr(B*(GX*GY+GY*GX)/2)
    end
    real(sym-ex*ey)
end
function collective_covariance(rho::PIState,X::AbstractMatrix,Y::AbstractMatrix;cache=nothing,plans=nothing)
    _,ps=_resolve_collective_plans(rho,(X,Y),cache,plans)
    collective_covariance(rho,ps[1],ps[2])
end

function _collective_covariance_matrix(rho::PIState,plans)
    all(p->LinearAlgebra.ishermitian(p.local_operator),plans)||throw(ArgumentError("covariance generators must be Hermitian"))
    n=length(plans);T=foldl(promote_type,(eltype(p.local_operator) for p in plans);
        init=eltype(rho.data));Rtype=_real_float_type(T)
    C=zeros(Rtype,n,n);means=zeros(T,n)
    for (s,p) in pairs(rho.basis.sectors)
        B=_multiplicity_weighted_block(rho,p);G=[plan.blocks[s] for plan in plans]
        for i in 1:n
            means[i]+=LinearAlgebra.tr(B*G[i])
            for j in i:n
                C[i,j]+=real(LinearAlgebra.tr(B*(G[i]*G[j]+G[j]*G[i]))/2)
            end
        end
    end
    for i in 1:n,j in i:n;C[i,j]-=real(means[i]*means[j]);C[j,i]=C[i,j];end
    C
end

"""
    collective_covariance_matrix(rho, generators; cache=nothing, plans=nothing)

Return the real symmetric covariance matrix of Hermitian collective
one-particle generators. Supply prepared `CollectiveObservablePlan` objects,
or reuse shared one-body geometry through `cache`, to avoid rebuilding Schur
blocks across repeated calls.
"""
function collective_covariance_matrix(rho::PIState,generators;cache=nothing,plans=nothing)
    _,ps=_resolve_collective_plans(rho,generators,cache,plans)
    _collective_covariance_matrix(rho,ps)
end
function collective_covariance_matrix(rho::PIState,plans::AbstractVector{<:CollectiveObservablePlan})
    for p in plans;_check_collective_plan(p,rho.basis);end
    _collective_covariance_matrix(rho,plans)
end

function _qubit_squeezing_data(rho::PIState;cache=nothing,plans=nothing)
    rho.basis.d==2||throw(ArgumentError("spin-squeezing helpers currently use the qubit normalization"))
    Rtype=if plans===nothing
        cache===nothing ? _real_float_type(eltype(rho.data)) :
            promote_type(_real_float_type(eltype(rho.data)),geometry_scalar_type(cache))
    else
        foldl(promote_type,(_real_float_type(eltype(p.local_operator)) for p in plans);
              init=_real_float_type(eltype(rho.data)))
    end
    Ctype=Complex{Rtype};half=one(Rtype)/2
    sx=Ctype[0 1;1 0]*half;sy=Ctype[0 -im;im 0]*half;sz=Ctype[1 0;0 -1]*half
    gs=[sx,sy,sz];_,ps=_resolve_collective_plans(rho,gs,cache,plans)
    m=real.([collective_expectation(rho,p) for p in ps]);C=_collective_covariance_matrix(rho,ps)
    nrm=norm(m)
    if nrm<=sqrt(eps(Rtype))
        vals=_hermitian_eigvals(Hermitian(complex.(C));operation="spin squeezing")
        return (;mean=m,covariance=C,min_transverse=minimum(vals),mean_norm=nrm)
    end
    n=m/nrm
    seed=abs(n[1])<Rtype(0.9) ? Rtype[1,0,0] : Rtype[0,1,0]
    u=normalize(seed-dot(seed,n)*n);v=cross(n,u);P=hcat(u,v)
    vals=_hermitian_eigvals(Hermitian(complex.(P'*C*P));operation="spin squeezing")
    (;mean=m,covariance=C,min_transverse=minimum(vals),mean_norm=nrm)
end

"""Kitagawa--Ueda qubit squeezing parameter `4*min(Delta J_perp^2)/N`."""
function kitagawa_ueda_squeezing(rho::PIState;kwargs...)
    value=_qubit_squeezing_data(rho;kwargs...).min_transverse
    _divide_by_particle_count(value,rho.basis.N;prefactor=4,
        context="Kitagawa--Ueda squeezing normalization")
end

"""Wineland qubit squeezing parameter `N*min(Delta J_perp^2)/|<J>|^2`."""
function wineland_squeezing(rho::PIState;kwargs...)
    x=_qubit_squeezing_data(rho;kwargs...)
    iszero(x.mean_norm)&&return typeof(x.mean_norm)(Inf)
    count,count_is_exact=_exact_particle_count(typeof(x.mean_norm),rho.basis.N)
    count_is_exact&&return count*x.min_transverse/x.mean_norm^2
    ratio=x.min_transverse/x.mean_norm^2
    _multiply_by_particle_count(ratio,rho.basis.N;
        context="Wineland squeezing normalization")
end

"""Return the identical two-particle reduced state."""
two_body_rdm(rho::PIState;kwargs...)=rho.basis.N>=2 ? reduced_state(rho,2;kwargs...) : throw(ArgumentError("two particles are required"))

"""Expectation of `X ⊗ Y` on an ordered pair of distinct particles."""
function two_body_expectation(rho::PIState,X::AbstractMatrix,Y::AbstractMatrix;cache=nothing)
    N=rho.basis.N;N>=2||throw(ArgumentError("two particles are required"));_check_local_observable(rho,X);_check_local_observable(rho,Y)
    cache===nothing&&(cache=OneBodyGeometry(rho.basis;T=promote_type(
        _real_float_type(eltype(rho.data)),_real_float_type(eltype(X)),
        _real_float_type(eltype(Y)))))
    T=promote_type(eltype(rho.data),eltype(X),eltype(Y),Complex{geometry_scalar_type(cache)})
    cross=zero(T);Rtype=_real_float_type(T)
    for p in rho.basis.sectors
        B=_multiplicity_weighted_block(rho,p);GX=collective_block(rho.basis,X,p;cache=cache);GY=collective_block(rho.basis,Y,p;cache=cache)
        cross+=LinearAlgebra.tr(B*GX*GY)
    end
    numerator=cross-collective_expectation(rho,X*Y;cache=cache)
    _divide_by_particle_pair_factor(numerator,N;ordered_distinct=true,
        context="normalization of an ordered two-body expectation")
end

"""
    connected_two_body_correlation(rho, X, Y; cache=nothing)

Return the connected correlation of `X` and `Y` on two distinct particles,
`expectation(X tensor Y) - expectation(X) * expectation(Y)`, using identical
one-particle marginals implied by permutation invariance.
"""
function connected_two_body_correlation(rho::PIState,X::AbstractMatrix,Y::AbstractMatrix;cache=nothing)
    cache===nothing&&(cache=OneBodyGeometry(rho.basis;T=promote_type(
        _real_float_type(eltype(rho.data)),_real_float_type(eltype(X)),
        _real_float_type(eltype(Y)))))
    product=collective_expectation(rho,X;cache=cache)*
        collective_expectation(rho,Y;cache=cache)
    disconnected=_divide_by_particle_pair_factor(product,rho.basis.N;
        ordered_distinct=false,
        context="normalization of a disconnected two-body expectation")
    two_body_expectation(rho,X,Y;cache=cache)-disconnected
end

"""Normalized identical-particle `g^(2)` for a local lowering operator `L`."""
function normalized_second_order_correlation(rho::PIState,L::AbstractMatrix;cache=nothing)
    cache===nothing&&(cache=OneBodyGeometry(rho.basis;T=promote_type(
        _real_float_type(eltype(rho.data)),_real_float_type(eltype(L)))))
    n=L'*L
    mean=_divide_by_particle_count(collective_expectation(rho,n;cache=cache),
        rho.basis.N;context="normalization of the one-particle intensity")
    iszero(mean)&&return _real_float_type(typeof(mean))(NaN)
    real(two_body_expectation(rho,n,n;cache=cache)/mean^2)
end

"""Whether the Wineland parameter certifies particle entanglement."""
spin_squeezing_entangled(rho::PIState;atol=_analysis_atol(rho),kwargs...)=
    wineland_squeezing(rho;kwargs...)<1-atol

"""
qfi_entanglement_depth(rho, X)

Lower bound on entanglement depth from collective-generator QFI. The standard
`k`-producible bound `s*k^2+r^2`, multiplied by the squared spectral range of
the local generator, is used.
"""
function qfi_entanglement_depth(rho::PIState,X::AbstractMatrix;
                                atol=_state_rtol(rho))
    N=rho.basis.N
    vals=_hermitian_eigvals(Hermitian(X);operation="QFI entanglement-depth bound")
    spectral_range=maximum(vals)-minimum(vals)
    _qfi_entanglement_depth_from_value(qfi(rho,X),spectral_range,N,atol)
end

function _qfi_depth_bound(range_squared,N::Int,k::Int)
    R=_real_float_type(typeof(range_squared))
    s=fld(N,k)
    r=N-s*k
    coefficient=if N<=isqrt(typemax(Int))
        s*k^2+r^2
    else
        big(s)*big(k)^2+big(r)^2
    end
    converted=try
        R(coefficient)
    catch
        R(Inf)
    end
    if isfinite(converted)
        bound=converted*range_squared
        isfinite(bound)&&(iszero(range_squared)||!iszero(bound))&&return bound
    end
    try
        _checked_mul_exact_ratio(range_squared,coefficient,one(coefficient);
            context="k-producible QFI bound")
    catch error
        error isa ArgumentError||rethrow()
        # The coefficient is at least one, so multiplying a finite nonzero
        # range squared cannot underflow.  A failed fused conversion therefore
        # means that the positive bound exceeds this scalar type; no finite QFI
        # can violate it.
        R(Inf)
    end
end

function _qfi_entanglement_depth_from_value(F,spectral_range,N::Int,atol)
    N>=1||throw(ArgumentError("at least one particle is required"))
    R=promote_type(_real_float_type(typeof(F)),
                   _real_float_type(typeof(spectral_range)),
                   _real_float_type(typeof(atol)))
    fisher=R(F)
    range_squared=R(spectral_range)^2
    tolerance=R(atol)
    depth=1
    lower=1
    upper=N-1
    while lower<=upper
        k=lower+(upper-lower)÷2
        bound=_qfi_depth_bound(range_squared,N,k)
        if fisher>bound+tolerance
            depth=k+1
            lower=k+1
        else
            upper=k-1
        end
    end
    depth
end

@doc raw"""
    quantum_fisher_information(rho, generator; atol=_analysis_atol(rho))

Compute the quantum Fisher information for the unitary family
`exp(-im*theta*generator) * rho * exp(im*theta*generator)`.

`generator` may be a PI operator or a local `d×d` matrix. A local matrix is
interpreted as the collective observable `sum_n generator^(n)`.  The
implementation diagonalizes the multiplicity-weighted blocks
`rho_bar_nu=f^nu*rho_nu`, whose eigenvalues `q_a=f^nu*p_a` remain on the
sector-probability scale, and evaluates

```math
F_Q[\rho,G] = 2\sum_{\nu,a,b:\,q_a+q_b>0}
\frac{(q_a-q_b)^2}{q_a+q_b}|G_{ab}|^2.
```

This is algebraically identical to the physical-eigenvalue formula, without
forming an exponentially small `p_a` or converting `f^nu` to a machine float.
No full-Hilbert-space matrix is constructed.
"""
function quantum_fisher_information(rho::PIState,generator;
                                    atol::Real=_analysis_atol(rho),cache=nothing,plan=nothing)
    abs(trace(rho)-1)<=atol||throw(ArgumentError("quantum Fisher information requires a trace-one state"))
    ishermitian(rho;atol=atol,rtol=0)||throw(ArgumentError("quantum Fisher information requires a Hermitian state"))
    local_generator=generator isa Union{AbstractMatrix,CollectiveObservablePlan}
    G=generator;prepared=nothing
    if generator isa CollectiveObservablePlan
        cache===nothing&&plan===nothing||throw(ArgumentError("a prepared generator cannot be combined with cache or plan keywords"))
        prepared=_check_collective_plan(generator,rho.basis)
        LinearAlgebra.ishermitian(prepared.local_operator)||throw(ArgumentError("quantum Fisher generator must be Hermitian"))
    elseif generator isa AbstractMatrix
        LinearAlgebra.ishermitian(G)||throw(ArgumentError("quantum Fisher generator must be Hermitian"))
        prepared=_resolve_collective_plan(rho,G,cache,plan)
    else
        plan===nothing||throw(ArgumentError("plan is only valid for a local-matrix generator"))
        cache===nothing||throw(ArgumentError("cache is only valid for a local-matrix generator"))
        G isa PIOperator||throw(ArgumentError("generator must be a local matrix or PIOperator"));_samebasis(rho,G)
        ishermitian(G;atol=atol,rtol=0)||throw(ArgumentError("quantum Fisher generator must be Hermitian"))
    end
    Rtype=_real_float_type(promote_type(eltype(rho.data),
        generator isa PIOperator ? eltype(generator.data) :
        eltype(prepared.local_operator)))
    result=zero(Rtype)
    for (s,p) in pairs(rho.basis.sectors)
        E=_weighted_sector_eigen(rho,p;atol=atol,rtol=0,
            operation="quantum Fisher information")
        Gp=local_generator ? prepared.blocks[s] : Matrix(physical_block(G,p))
        Ge=E.vectors'*Gp*E.vectors
        sector=zero(Rtype)
        for a in eachindex(E.values),b in eachindex(E.values)
            den=E.values[a]+E.values[b]
            den>E.zero_tolerance||continue
            sector += 2*(E.values[a]-E.values[b])^2/den*abs2(Ge[a,b])
        end
        result += sector
    end
    max(zero(Rtype),real(result))
end

"""Alias for [`quantum_fisher_information`](@ref)."""
qfi(args...;kwargs...)=quantum_fisher_information(args...;kwargs...)

function _qfim_shared_geometry(rho::PIState,generators)
    geometry_type=foldl(promote_type,
        (_real_float_type(eltype(G)) for G in generators
         if G isa AbstractMatrix);
        init=_real_float_type(eltype(rho.data)))
    OneBodyGeometry(rho.basis;T=geometry_type)
end

@doc raw"""
    quantum_fisher_information_matrix(rho, generators;
                                      atol=_analysis_atol(rho))

Compute the symmetric quantum Fisher information matrix (QFIM) for the
multiparameter unitary model generated by `generators`:

```math
[F_Q]_{\mu\nu}=2\sum_{a,b:p_a+p_b>0}
\frac{(p_a-p_b)^2}{p_a+p_b}
\mathrm{Re}[(G_\mu)_{ab}(G_\nu)_{ba}].
```

Each generator may be a local `d×d` matrix, interpreted as its collective
sum, or a `PIOperator`. Multiplicity-weighted density blocks are diagonalized
only once per sector, and local collective blocks share one
representation-geometry cache.
"""
function quantum_fisher_information_matrix(rho::PIState,generators;
                                           atol::Real=_analysis_atol(rho),cache=nothing,plans=nothing)
    gs=collect(generators);npar=length(gs)
    npar>0||throw(ArgumentError("at least one QFIM generator is required"))
    abs(trace(rho)-1)<=atol||throw(ArgumentError("QFIM requires a trace-one state"))
    ishermitian(rho;atol=atol,rtol=0)||throw(ArgumentError("QFIM requires a Hermitian state"))
    for G in gs
        if G isa CollectiveObservablePlan
            _check_collective_plan(G,rho.basis);LinearAlgebra.ishermitian(G.local_operator)||throw(ArgumentError("every QFIM generator must be Hermitian"))
        elseif G isa AbstractMatrix
            _check_local_observable(rho,G)
            LinearAlgebra.ishermitian(G)||throw(ArgumentError("every QFIM generator must be Hermitian"))
        elseif G isa PIOperator
            _samebasis(rho,G)
            ishermitian(G;atol=atol,rtol=0)||throw(ArgumentError("every QFIM generator must be Hermitian"))
        else
            throw(ArgumentError("QFIM generators must be local matrices or PIOperators"))
        end
    end
    prepared=Vector{Union{Nothing,CollectiveObservablePlan}}(undef,npar)
    if plans===nothing
        if any(G->G isa AbstractMatrix,gs)&&cache===nothing
            cache=_qfim_shared_geometry(rho,gs)
        end
        for (i,G) in pairs(gs)
            prepared[i]=G isa CollectiveObservablePlan ? G : G isa AbstractMatrix ? CollectiveObservablePlan(rho.basis,G;cache=cache) : nothing
        end
    else
        cache===nothing||throw(ArgumentError("provide either cache or plans, not both"))
        ps=collect(plans);length(ps)==npar||throw(DimensionMismatch("one plan entry is required per QFIM generator"))
        for (i,G) in pairs(gs)
            if G isa AbstractMatrix
                ps[i] isa CollectiveObservablePlan||throw(ArgumentError("local QFIM generators require CollectiveObservablePlan entries"))
                prepared[i]=_check_collective_plan(ps[i],rho.basis,G)
            elseif G isa CollectiveObservablePlan
                ps[i]===nothing||throw(ArgumentError("use `nothing` for an already prepared QFIM generator"));prepared[i]=G
            else
                ps[i]===nothing||throw(ArgumentError("PIOperator QFIM generators require `nothing` plan entries"));prepared[i]=nothing
            end
        end
    end
    Rtype=foldl(promote_type,(_real_float_type(G isa PIOperator ? eltype(G.data) :
        G isa CollectiveObservablePlan ? eltype(G.local_operator) : eltype(G)) for G in gs);
        init=_real_float_type(eltype(rho.data)))
    F=zeros(Rtype,npar,npar)
    for (s,p) in pairs(rho.basis.sectors)
        E=_weighted_sector_eigen(rho,p;atol=atol,rtol=0,
            operation="quantum Fisher information matrix")
        transformed=Matrix{Complex{Rtype}}[]
        for (i,G) in pairs(gs)
            block=G isa Union{AbstractMatrix,CollectiveObservablePlan} ? prepared[i].blocks[s] : Matrix(physical_block(G,p))
            push!(transformed,Matrix{Complex{Rtype}}(E.vectors'*block*E.vectors))
        end
        for a in eachindex(E.values),b in eachindex(E.values)
            den=E.values[a]+E.values[b];den>E.zero_tolerance||continue
            weight=2*(E.values[a]-E.values[b])^2/den
            for mu in 1:npar,nu in mu:npar
                F[mu,nu]+=weight*real(transformed[mu][a,b]*transformed[nu][b,a])
            end
        end
    end
    for mu in 1:npar,nu in 1:mu-1;F[mu,nu]=F[nu,mu];end
    F
end

"""Alias for [`quantum_fisher_information_matrix`](@ref)."""
qfim(args...;kwargs...)=quantum_fisher_information_matrix(args...;kwargs...)

"""QFIM from tangent density operators `d rho / d theta_mu`."""
function qfim_from_derivatives(rho::PIState,derivatives;
                               atol::Real=_analysis_atol(rho))
    ds=collect(derivatives);all(d->d isa PIState&&d.basis===rho.basis,ds)||throw(ArgumentError("derivatives must be PIState tangents on the same basis"))
    Rtype=foldl(promote_type,(_real_float_type(eltype(d.data)) for d in ds);
                init=_real_float_type(eltype(rho.data)))
    m=length(ds);F=zeros(Rtype,m,m)
    for p in rho.basis.sectors
        E=_weighted_sector_eigen(rho,p;atol=atol,rtol=0,
            operation="derivative quantum Fisher information matrix")
        D=[E.vectors'*_multiplicity_weighted_block(d,p)*E.vectors for d in ds]
        for a in eachindex(E.values),b in eachindex(E.values)
            den=E.values[a]+E.values[b];den>E.zero_tolerance||continue
            for mu in 1:m,nu in mu:m;F[mu,nu]+=2real(D[mu][a,b]*D[nu][b,a])/den;end
        end
    end
    for i in 1:m,j in 1:i-1;F[i,j]=F[j,i];end;F
end

"""Return SLD `PIOperator`s for unitary parameters generated by `generators`."""
function symmetric_logarithmic_derivatives(rho::PIState,generators;
                                           atol::Real=_analysis_atol(rho))
    gs=collect(generators);isempty(gs)&&throw(ArgumentError("at least one generator is required"))
    for (mu,G) in pairs(gs)
        if G isa AbstractMatrix;_check_local_observable(rho,G);LinearAlgebra.ishermitian(G)||throw(ArgumentError("generators must be Hermitian"))
        elseif G isa PIOperator;_samebasis(rho,G);ishermitian(G)||throw(ArgumentError("generators must be Hermitian"))
        else;throw(ArgumentError("generators must be local matrices or PIOperators"));end
    end
    Rtype=foldl(promote_type,(_real_float_type(G isa PIOperator ? eltype(G.data) : eltype(G)) for G in gs);
                init=_real_float_type(eltype(rho.data)))
    ops=PIOperator[PIOperator(rho.basis;T=Rtype) for _ in gs]
    cache=any(G->G isa AbstractMatrix,gs) ? OneBodyGeometry(rho.basis;T=Rtype) : nothing
    for p in rho.basis.sectors
        E=_weighted_sector_eigen(rho,p;atol=atol,rtol=0,
            operation="symmetric logarithmic derivatives")
        for (mu,G) in pairs(gs)
            B=G isa AbstractMatrix ? collective_block(rho.basis,G,p;cache=cache) : Matrix(physical_block(G,p));Ge=E.vectors'*B*E.vectors
            Le=zeros(Complex{Rtype},size(Ge))
            for a in eachindex(E.values),b in eachindex(E.values)
                den=E.values[a]+E.values[b];den>E.zero_tolerance&&
                    (Le[a,b]=2im*(E.values[a]-E.values[b])/den*Ge[a,b])
            end
            L=E.vectors*Le*E.vectors'
            C=coefficient_block(ops[mu],p);H=(L+L')/2
            _store_physical_block!(C,H,p,Rtype;
                context="stored SLD coefficient for sector $p")
        end
    end
    ops
end

"""Matrix `(1/2i) tr(rho [L_mu,L_nu])` diagnosing SLD incompatibility."""
function sld_commutator_matrix(rho::PIState,generators;
                               atol::Real=_analysis_atol(rho))
    L=symmetric_logarithmic_derivatives(rho,generators;atol=atol);n=length(L)
    Rtype=foldl(promote_type,(_real_float_type(eltype(A.data)) for A in L);
                init=_real_float_type(eltype(rho.data)))
    C=zeros(Rtype,n,n)
    for mu in 1:n,nu in mu+1:n
        z=zero(Complex{Rtype})
        for p in rho.basis.sectors
            z+=LinearAlgebra.tr(_multiplicity_weighted_block(rho,p)*
                physical_block(L[mu],p)*physical_block(L[nu],p))
        end
        C[mu,nu]=imag(z);C[nu,mu]=-C[mu,nu]
    end
    C
end

"""Mean Uhlmann-curvature convention `U = sld_commutator_matrix/2`."""
mean_uhlmann_curvature(rho::PIState,generators;kwargs...)=sld_commutator_matrix(rho,generators;kwargs...)/2

"""Whether the weak SLD commutativity condition holds within tolerance."""
multiparameter_compatible(rho::PIState,generators;atol::Real=1e-10)=maximum(abs,sld_commutator_matrix(rho,generators;atol=atol))<=atol

function _one_body_rdm_geometry(rho::PIState,cache::OneBodyGeometry)
    b=rho.basis;_check_geometry_basis(cache,b)
    T=promote_type(eltype(rho.data),Complex{geometry_scalar_type(cache)})
    Rtype=_real_float_type(T);R1=zeros(T,b.d,b.d)
    inverse_particle_count=_inverse_particle_count(Rtype,b.N)
    for (s,p) in pairs(b.sectors)
        B=_multiplicity_weighted_block(rho,p)
        for a in axes(B,1),c in axes(B,2),mu in cache.connections[(s,s)]
            key=(s,mu,s)
            prefactor=inverse_particle_count*B[a,c]*cache.scales[key]
            # tr(R G[E_ji]) uses G[c,a].  A contraction entry `(u,v,z)`
            # contributes conj(z) to E_{u,v}; therefore rho[v,u] receives it.
            for (u,v,z) in cache.contractions[key][c,a]
                R1[v,u]+=prefactor*conj(z)
            end
        end
    end
    R1
end

"""
    one_body_rdm(rho; cache=nothing, plan=nothing,
                 atol=_analysis_atol(rho), rtol=_state_rtol(rho))

Return the one-particle density matrix in computational-label order.  Passing
a reusable `OneBodyGeometry` contracts all matrix units in one traversal.
Alternatively, a `ReductionPlan(basis,1)` reuses fixed-bipartition recoupling
data.  Supplying both `cache` and `plan` is an error.
"""
function one_body_rdm(rho::PIState;cache=nothing,plan=nothing,
                      atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho))
    rho.basis.N>=1||throw(ArgumentError("one particle is required"))
    if plan===nothing&&cache!==nothing
        validate_state(rho;atol=atol,rtol=rtol)
        return _one_body_rdm_geometry(rho,cache)
    elseif plan===nothing
        validate_state(rho;atol=atol,rtol=rtol)
        T=_real_float_type(eltype(rho.data))
        return _one_body_rdm_geometry(rho,OneBodyGeometry(rho.basis;T=T))
    end
    cache===nothing||throw(ArgumentError("provide either cache or plan, not both"))
    reduced=reduced_state(rho,1;plan=plan,atol=atol,rtol=rtol)
    p=only(reduced.basis.sectors);R=Matrix(physical_block(reduced,p));patterns=only(reduced.basis.patterns)
    # GT patterns are sorted by stored entries, not by local computational
    # label. Map each one-box content vector back to the public 1:d ordering.
    order=Vector{Int}(undef,reduced.basis.d)
    for i in eachindex(order)
        q=findfirst(g->content(g)[i]==1,patterns)
        q===nothing&&throw(ErrorException("one-box GT pattern for local label $i is missing"))
        order[i]=q
    end
    R[order,order]
end
"""Return `abs(trace(rho) - 1)`."""
trace_error(rho)=abs(trace(rho)-1)

"""Return the largest Frobenius Hermiticity residual among coefficient blocks."""
hermiticity_error(rho)=maximum(norm(coefficient_block(rho,p)-coefficient_block(rho,p)') for p in rho.basis.sectors)

"""Return the smallest eigenvalue among all physical Schur blocks."""
minimum_sector_eigenvalue(rho)=minimum(minimum(_hermitian_eigvals(
    Hermitian(Matrix(physical_block(rho,p)));operation="minimum sector eigenvalue"))
    for p in rho.basis.sectors)

"""
    check_generator(model)

Return structural diagnostics for a `PIModel`: sparse trace-preservation
residual when available, static-Hamiltonian Hermiticity, the presence of
negative constant rates, and sector connectivity. For a driven or otherwise
non-materializable model the trace-preservation entry is `missing`.
"""
function check_generator(model::PIModel)
    L=try liouvillian(model;representation=:sparse) catch; nothing end
    trace_preservation_error=if L===nothing
        missing
    else
        Rtype=_real_float_type(eltype(L));tau=zeros(Complex{Rtype},length(model.basis))
        for (s,p) in pairs(model.basis.sectors)
            n=length(model.basis.patterns[s])
            for i in 1:n
                tau[model.basis.offsets[s]+i-1+(i-1)*n]=
                    _schur_multiplicity_scale(Rtype,p)
            end
        end
        norm(transpose(tau)*L)
    end
    (trace_preservation_error=trace_preservation_error,
     static_hamiltonians_hermitian=all(t->!(t isa Union{LocalHamiltonian,CollectiveHamiltonian,DirectPIHamiltonian,PBodyHamiltonian})||t.operator isa Function||ishermitian(t.operator),model.terms),
     negative_constant_rates=any(t->t.rate isa Number&&t.rate<0,model.terms),connectivity=[minus_plus_neighbors(p) for p in model.basis.sectors])
end
