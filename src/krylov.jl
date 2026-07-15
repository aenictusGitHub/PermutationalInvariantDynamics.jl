@inline function _promote_krylov_array_type(::Type{T},x) where T
    S=eltype(x)
    S<:Number&&isconcretetype(S)||throw(ArgumentError(
        "iterative-solver inputs must expose a concrete numeric eltype"))
    promote_type(T,_complex_float_type(S))
end
@inline function _promote_krylov_operator_type(::Type{T},x) where T
    S=eltype(x)
    S<:Number ? promote_type(T,_complex_float_type(S)) : T
end
function _promote_krylov_scalar_type(::Type{T},x) where T
    x===nothing&&return T
    x isa Integer&&iszero(x)&&return T
    S=_complex_float_type(typeof(x))
    if x isa Integer
        R=_real_float_type(S);converted=R(x)
        if !(isfinite(converted)&&BigInt(converted)==BigInt(x))
            converted=BigFloat(x)
            BigInt(converted)==BigInt(x)||throw(ArgumentError(
                "integer iterative-solver input $x is not exactly representable at the active BigFloat precision"))
            S=Complex{BigFloat}
        end
    end
    promote_type(T,S)
end

function _check_krylov_matvec_precision(L,::Type{T}) where T
    _check_liouvillian_source_precision(L,T,"Krylov vector")
end

function _projected_eigen(A,args...)
    try
        eigen(A,args...)
    catch error
        error isa MethodError||rethrow()
        R=_real_float_type(eltype(A))
        throw(ArgumentError(
            "projected Krylov eigensolves are unavailable for scalar type $R with Julia's active LinearAlgebra backend; use Float32/Float64 data or load a compatible generic eigensolver"))
    end
end

function _projected_eigen!(A,args...)
    try
        eigen!(A,args...)
    catch error
        error isa MethodError||rethrow()
        R=_real_float_type(eltype(A))
        throw(ArgumentError(
            "projected Krylov eigensolves are unavailable for scalar type $R with Julia's active LinearAlgebra backend; use Float32/Float64 data or load a compatible generic eigensolver"))
    end
end

function _check_finite_krylov_target(target)
    target===nothing&&return target
    target isa Number&&isfinite(real(target))&&isfinite(imag(target))||
        throw(ArgumentError("Krylov target must be a finite number"))
    target
end

"""Reusable storage for restarted GMRES on a PI Liouville-space vector."""
mutable struct KrylovWorkspace{T}
    V::Matrix{T}
    H::Matrix{T}
    cs::Vector{Float64}
    sn::Vector{T}
    g::Vector{T}
    w::Vector{T}
    r::Vector{T}
    z::Vector{T}
    p::Vector{T}
    y::Vector{T}
end

function KrylovWorkspace(::Type{T}, n::Integer, krylovdim::Integer) where T
    n > 0 || throw(ArgumentError("dimension must be positive"))
    krylovdim > 0 || throw(ArgumentError("krylovdim must be positive"))
    m=min(Int(krylovdim),Int(n))
    KrylovWorkspace(zeros(T,n,m+1),zeros(T,m+1,m),zeros(Float64,m),
                    zeros(T,m),zeros(T,m+1),zeros(T,n),zeros(T,n),zeros(T,n),
                    zeros(T,n),zeros(T,m))
end

KrylovWorkspace(L, krylovdim::Integer=30)=
    KrylovWorkspace(_complex_float_type(eltype(L)),size(L,1),krylovdim)

"""
    ArnoldiWorkspace(L, krylovdim)

Reusable basis, image, pencil, and vector storage shared by ordinary,
harmonic, and implicit-QR Arnoldi. A workspace is mutable and must be owned by
one task at a time.
"""
mutable struct ArnoldiWorkspace{T}
    V::Matrix{T}
    H::Matrix{T}
    LV::Matrix{T}
    Z::Matrix{T}
    A::Matrix{T}
    B::Matrix{T}
    X::Matrix{T}
    LX::Matrix{T}
    q::Vector{T}
    tmp::Vector{T}
end

function ArnoldiWorkspace(::Type{T},n::Integer,krylovdim::Integer) where T
    n>0||throw(ArgumentError("dimension must be positive"))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    m=min(Int(n),Int(krylovdim))
    ArnoldiWorkspace(zeros(T,n,m+1),zeros(T,m+1,m),zeros(T,n,m),
                     zeros(T,n,m),zeros(T,m,m),zeros(T,m,m),
                     zeros(T,n,m),zeros(T,n,m),zeros(T,n),zeros(T,n))
end

ArnoldiWorkspace(L,krylovdim::Integer=max(20,min(size(L,1),40)))=
    ArnoldiWorkspace(_complex_float_type(eltype(L)),size(L,1),krylovdim)

function _check_arnoldi_workspace(ws::ArnoldiWorkspace,n,m)
    size(ws.V,1)==n&&size(ws.V,2)>=m+1||throw(DimensionMismatch("Arnoldi workspace basis is too small"))
    size(ws.H,1)>=m+1&&size(ws.H,2)>=m||throw(DimensionMismatch("Arnoldi workspace Hessenberg array is too small"))
    for M in (ws.LV,ws.Z,ws.X,ws.LX)
        size(M,1)==n&&size(M,2)>=m||throw(DimensionMismatch("Arnoldi workspace matrix is too small"))
    end
    size(ws.A,1)>=m&&size(ws.A,2)>=m&&size(ws.B,1)>=m&&size(ws.B,2)>=m||
        throw(DimensionMismatch("Arnoldi workspace pencil is too small"))
    length(ws.q)==n&&length(ws.tmp)==n||throw(DimensionMismatch("Arnoldi workspace vector has the wrong dimension"))
    ws
end

function _estimated_operator_scale!(L,x,y;probes::Integer=3)
    if L isa AbstractMatrix
        s=opnorm(L,Inf)
        return iszero(s) ? zero(float(real(one(eltype(L))))) : float(real(s))
    end
    probes>0||throw(ArgumentError("operator-scale probes must be positive"))
    n=length(x);rng=Random.MersenneTwister(0x51ca1e)
    R=_real_float_type(promote_type(eltype(x),eltype(y)));scale=zero(R)
    for _ in 1:probes
        randn!(rng,x);nx=norm(x);mul!(y,L,x)
        scale=max(scale,norm(y)/nx)
    end
    scale
end

function _validated_operator_scale(scale)
    isfinite(scale)&&scale>0||throw(ArgumentError("operator_scale must be finite and positive; a numerically zero Liouvillian does not define a unique trace-fixed Krylov solve"))
    float(real(scale))
end

"""
    SchurSectorPreconditioner

Block-diagonal approximation to the trace-fixed Liouvillian in Schur-sector
Liouville coordinates. Each block is LU-factorized once and can be reused by
matrix-free GMRES solves.
"""
struct SchurSectorPreconditioner{T,F,R,S,M}
    factors::Vector{F}
    ranges::Vector{R}
    n::Int
    regularization::T
    operator_scale::S
    metadata::M
end

size(P::SchurSectorPreconditioner)=(P.n,P.n)
size(P::SchurSectorPreconditioner,i::Integer)=i in (1,2) ? P.n : 1
eltype(::SchurSectorPreconditioner{T}) where T=T
"""Return setup, storage, apply-cost, and amortization metadata for `P`."""
preconditioner_cost(P::SchurSectorPreconditioner)=P.metadata

function ldiv!(dest::AbstractVector,P::SchurSectorPreconditioner,src::AbstractVector)
    length(dest)==P.n&&length(src)==P.n||throw(DimensionMismatch("preconditioner vector has wrong length"))
    dest===src || copyto!(dest,src)
    for (fac,r) in zip(P.factors,P.ranges)
        ldiv!(view(dest,r),fac,view(dest,r))
    end
    dest
end

"""
    schur_sector_preconditioner(L, basis; trace_vector=nothing,
                                regularization=0)

Construct a left preconditioner from the diagonal Schur-sector blocks of the
scale-normalized trace-fixed operator `L/s + v*t'`. Only `sum_s n_s^2` coefficients are retained,
where `n_s` is a sector's Liouville dimension. Construction uses matrix-free
applications of `L`; the resulting LU factors should be reused across solves.
The implementation normalizes `L` by a reproducible operator-scale estimate,
matching `krylov_steady_state`. Set a small positive `regularization` only when
a sector block is singular. Cost and amortization estimates are available as
`preconditioner_cost(P)`; `expected_reuses` controls the setup warning.
"""
function schur_sector_preconditioner(L,basis::PIBasis;trace_vector=nothing,
                                     regularization::Real=0,operator_scale=nothing,
                                     expected_reuses::Integer=1,
                                     expected_solve_applications::Integer=30,
                                     warn_unamortized::Bool=true)
    started=time_ns()
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("L must be square"))
    length(basis)==n||throw(DimensionMismatch("basis and Liouvillian dimensions differ"))
    isfinite(regularization)&&regularization>=0||throw(ArgumentError(
        "regularization must be finite and nonnegative"))
    operator_scale===nothing||_validated_operator_scale(operator_scale)
    expected_reuses>0||throw(ArgumentError("expected_reuses must be positive"))
    expected_solve_applications>0||throw(ArgumentError("expected_solve_applications must be positive"))
    raw_t=trace_vector===nothing ? _trace_vector(basis,_complex_float_type(eltype(L))) :
                                  collect(trace_vector)
    T=promote_type(_complex_float_type(eltype(L)),eltype(raw_t))
    T=_promote_krylov_scalar_type(T,regularization)
    T=_promote_krylov_scalar_type(T,operator_scale)
    _check_krylov_matvec_precision(L,T)
    t=T.(raw_t)
    length(t)==n||throw(DimensionMismatch("trace vector has wrong length"))
    v=t/dot(t,t);e=zeros(T,n);y=zeros(T,n)
    scale_probes=operator_scale===nothing && !(L isa AbstractMatrix) ? 3 : 0
    Lscale=_validated_operator_scale(operator_scale===nothing ?
        _estimated_operator_scale!(L,e,y;probes=max(scale_probes,1)) : operator_scale)
    ranges=[basis.offsets[s]:basis.offsets[s+1]-1 for s in eachindex(basis.sectors)]
    blocks=Matrix{T}[]
    for r in ranges
        B=zeros(T,length(r),length(r))
        for (j,gj) in enumerate(r)
            fill!(e,zero(T));e[gj]=one(T);mul!(y,L,e)
            @views B[:,j].=y[r]./Lscale
        end
        @views B .+= v[r]*adjoint(t[r])
        if !iszero(regularization)
            @inbounds for i in axes(B,1);B[i,i]+=regularization;end
        end
        push!(blocks,B)
    end
    factors=map(B->try
            lu(B;check=true)
        catch err
            err isa SingularException||rethrow()
            throw(ArgumentError("a Schur-sector preconditioner block is singular; pass a small regularization"))
        end,blocks)
    F=eltype(factors);R=eltype(ranges)
    stored_coefficients=sum(length(r)^2 for r in ranges)
    recommended_reuses=max(2,cld(n+scale_probes,expected_solve_applications))
    metadata=(setup_seconds=(time_ns()-started)/1e9,
              setup_liouvillian_applications=n+scale_probes,
              setup_factorizations=length(factors),
              stored_coefficients,
              stored_bytes=Base.summarysize(factors),
              apply_triangular_solves=length(factors),
              apply_flop_estimate=sum(2*length(r)^2 for r in ranges),
              recommended_minimum_reuses=recommended_reuses,
              expected_reuses=Int(expected_reuses),
              amortization_expected=expected_reuses>=recommended_reuses)
    if warn_unamortized&&!metadata.amortization_expected
        @warn "Schur-sector preconditioner setup is unlikely to amortize for the declared reuse count; reuse the returned object across solves or prefer unpreconditioned GMRES" expected_reuses recommended_reuses setup_applications=metadata.setup_liouvillian_applications maxlog=1
    end
    SchurSectorPreconditioner{T,F,R,typeof(Lscale),typeof(metadata)}(
        factors,ranges,n,T(regularization),Lscale,metadata)
end

function schur_sector_preconditioner(model::PIModel;representation=:matrixfree,kwargs...)
    L=liouvillian(model;representation=representation)
    schur_sector_preconditioner(L,model.basis;kwargs...)
end

function _complex_givens(a,b)
    iszero(b) && return (1.0,zero(promote_type(typeof(a),typeof(b))))
    iszero(a) && return (0.0,one(promote_type(typeof(a),typeof(b))))
    scale=abs(a)+abs(b); normab=scale*sqrt(abs2(a/scale)+abs2(b/scale))
    alpha=a/abs(a); (abs(a)/normab,alpha*conj(b)/normab)
end

function _upper_triangular_solve!(y,H,g,k::Integer)
    length(y)>=k||throw(DimensionMismatch("GMRES triangular-solve scratch is too small"))
    for i in k:-1:1
        s=g[i]
        @inbounds for j in i+1:k
            s-=H[i,j]*y[j]
        end
        piv=H[i,i]
        iszero(piv)&&throw(LinearAlgebra.SingularException(i))
        y[i]=s/piv
    end
    y
end

function _gmres!(x,apply!,b,ws::KrylovWorkspace;atol,rtol,maxiter,preconditioner=nothing)
    n=length(x);length(b)==n||throw(DimensionMismatch("right-hand side has wrong length"))
    size(ws.V,1)==n||throw(DimensionMismatch("workspace has wrong dimension"))
    m=size(ws.H,2)
    length(ws.y)>=m||throw(DimensionMismatch("workspace triangular-solve scratch is too small"))
    raw_bnorm=norm(b);rawtol=atol+rtol*raw_bnorm
    if preconditioner===nothing
        bnorm=norm(b)
    else
        ldiv!(ws.p,preconditioner,b);bnorm=norm(ws.p)
    end
    tol=atol+rtol*bnorm;iterations=0
    apply!(ws.w,x);@. ws.r=b-ws.w;rawbeta=norm(ws.r);beta=rawbeta
    if preconditioner!==nothing;ldiv!(ws.p,preconditioner,ws.r);beta=norm(ws.p);end
    beta<=tol&&rawbeta<=rawtol&&return (converged=true,iterations=0,restarts=0,residual=beta,raw_residual=rawbeta)
    restarts=0
    while iterations<maxiter
        fill!(ws.H,zero(eltype(ws.H)));fill!(ws.g,zero(eltype(ws.g)))
        if preconditioner===nothing
            @views ws.V[:,1].=ws.r./beta
        else
            @views ws.V[:,1].=ws.p./beta
        end
        ws.g[1]=beta;kdone=0
        for j in 1:min(m,maxiter-iterations)
            apply!(ws.w,view(ws.V,:,j))
            if preconditioner!==nothing
                ldiv!(ws.p,preconditioner,ws.w);copyto!(ws.w,ws.p)
            end
            # Modified Gram--Schmidt with one reorthogonalization pass.
            for pass in 1:2, i in 1:j
                hij=dot(view(ws.V,:,i),ws.w);ws.H[i,j]+=hij
                @views @. ws.w=ws.w-hij*ws.V[:,i]
            end
            ws.H[j+1,j]=norm(ws.w)
            if !iszero(ws.H[j+1,j]);@views ws.V[:,j+1].=ws.w./ws.H[j+1,j];end
            for i in 1:j-1
                h1=ws.H[i,j];h2=ws.H[i+1,j];c=ws.cs[i];s=ws.sn[i]
                ws.H[i,j]=c*h1+s*h2;ws.H[i+1,j]=-conj(s)*h1+c*h2
            end
            c,s=_complex_givens(ws.H[j,j],ws.H[j+1,j]);ws.cs[j]=c;ws.sn[j]=s
            h1=ws.H[j,j];h2=ws.H[j+1,j];ws.H[j,j]=c*h1+s*h2;ws.H[j+1,j]=0
            g1=ws.g[j];g2=ws.g[j+1];ws.g[j]=c*g1+s*g2;ws.g[j+1]=-conj(s)*g1+c*g2
            iterations+=1;kdone=j
            abs(ws.g[j+1])<=tol&&break
        end
        _upper_triangular_solve!(ws.y,ws.H,ws.g,kdone)
        mul!(ws.z,view(ws.V,:,1:kdone),view(ws.y,1:kdone));x .+= ws.z
        restarts+=1
        apply!(ws.w,x);@. ws.r=b-ws.w;rawbeta=norm(ws.r);beta=rawbeta
        if preconditioner!==nothing;ldiv!(ws.p,preconditioner,ws.r);beta=norm(ws.p);end
        beta<=tol&&rawbeta<=rawtol&&return (converged=true,iterations=iterations,restarts,restart_residual=beta,residual=beta,raw_residual=rawbeta)
    end
    (converged=false,iterations=iterations,restarts,restart_residual=beta,residual=beta,raw_residual=rawbeta)
end

"""
    krylov_steady_state(L; trace_vector=nothing, basis=nothing, ...)

Solve the trace-fixed stationary equation with restarted GMRES, using only
`mul!` applications of `L`. The equivalent nonsingular system is
`(L/s + v*t')rho = v`, where `s` is a reproducible operator-scale estimate,
`t` is the physical trace functional, and
`dot(t,v)=1`. No Liouvillian matrix or factorization is constructed.
"""
function krylov_steady_state(L;basis=nothing,trace_vector=nothing,
                             initial_state=nothing,krylovdim::Integer=30,
                             maxiter::Integer=500,atol::Real=1e-10,
                             rtol::Real=1e-8,workspace=nothing,preconditioner=nothing,
                             operator_scale=nothing,return_info::Bool=false)
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("L must be square"))
    operator_scale===nothing||_validated_operator_scale(operator_scale)
    t=trace_vector!==nothing ? collect(trace_vector) : basis!==nothing ?
      _trace_vector(basis,_complex_float_type(eltype(L))) :
      L isa MatrixFreeLiouvillian ? collect(L.tracevec) : nothing
    t===nothing&&throw(ArgumentError("the physical trace is ambiguous; pass basis=... or trace_vector=..."))
    length(t)==n||throw(DimensionMismatch("trace vector has wrong length"))
    initial_type=initial_state isa PIState ? eltype(initial_state.data) :
                 initial_state===nothing ? eltype(t) : eltype(initial_state)
    T=promote_type(_complex_float_type(eltype(L)),eltype(t),initial_type)
    T=_promote_krylov_scalar_type(T,operator_scale)
    preconditioner===nothing||(T=_promote_krylov_operator_type(T,preconditioner))
    _check_krylov_matvec_precision(L,T)
    tc=T.(t);v=tc/dot(tc,tc)
    x=initial_state isa PIState ? T.(initial_state.data) : initial_state===nothing ? copy(v) : T.(initial_state)
    length(x)==n||throw(DimensionMismatch("initial_state has wrong length"))
    ws=workspace===nothing ? KrylovWorkspace(T,n,krylovdim) : workspace
    promote_type(eltype(ws.V),T)==eltype(ws.V)||throw(ArgumentError(
        "Krylov workspace scalar type cannot represent the solver inputs"))
    _check_krylov_matvec_precision(L,eltype(ws.V))
    preconditioner===nothing||size(preconditioner)==(n,n)||throw(DimensionMismatch("preconditioner has wrong dimension"))
    Lscale = if preconditioner isa SchurSectorPreconditioner
        if operator_scale!==nothing
            requested=_validated_operator_scale(operator_scale)
            scale_type=_real_float_type(typeof(preconditioner.operator_scale))
            isapprox(requested,preconditioner.operator_scale;atol=zero(scale_type),
                     rtol=sqrt(eps(scale_type)))||
                throw(ArgumentError("operator_scale is inconsistent with the Schur-sector preconditioner"))
        end
        preconditioner.operator_scale
    else
        _validated_operator_scale(operator_scale===nothing ?
            _estimated_operator_scale!(L,ws.z,ws.w) : operator_scale)
    end
    invscale=inv(Lscale)
    function apply!(y,z)
        mul!(y,L,z);α=dot(tc,z)
        @inbounds @simd for i in eachindex(y);y[i]=invscale*y[i]+v[i]*α;end
        y
    end
    RT=_real_float_type(T);atolT=RT(atol);rtolT=RT(rtol)
    result=_gmres!(x,apply!,v,ws;atol=atolT,rtol=rtolT,maxiter=maxiter,
                   preconditioner=preconditioner)
    mul!(ws.w,L,x);residual=norm(ws.w);normalized_residual=residual/Lscale;terr=abs(dot(tc,x)-1)
    tol=atolT+rtolT*max(norm(x),one(RT))
    converged=result.converged&&normalized_residual<=tol&&terr<=atolT+rtolT
    converged||throw(ArgumentError("matrix-free Krylov steady-state solve did not converge in $(result.iterations) iterations; normalized_residual=$normalized_residual, trace_error=$terr"))
    info=(state=x,residual=residual,trace_error=terr,nullity=nothing,method=:krylov,
          iterations=result.iterations,eigenvalue=nothing,converged=true,
          linear_residual=result.residual,unpreconditioned_linear_residual=result.raw_residual,
          normalized_residual,operator_scale=Lscale,
          restarts=result.restarts,krylov_dimension=size(ws.H,2),
          preconditioned=preconditioner!==nothing,
          preconditioner=preconditioner===nothing ? nothing : typeof(preconditioner),
          preconditioner_cost=preconditioner isa SchurSectorPreconditioner ?
              preconditioner.metadata : nothing)
    return_info ? info : x
end

function _ritz_order(values,which)
    which===:LR&&return sortperm(values;by=real,rev=true)
    which===:LM&&return sortperm(values;by=abs,rev=true)
    which===:SM&&return sortperm(values;by=abs)
    throw(ArgumentError("which must be :LR, :LM, or :SM"))
end

"""
    krylov_liouvillian_spectrum(L; nev=6, krylovdim=max(20, 2nev+4), which=:LR)

Approximate selected Liouvillian eigenpairs by matrix-free Arnoldi iteration.
The returned residual estimates are `norm(L*v-lambda*v)` Ritz estimates.
Pass a reusable `ArnoldiWorkspace` to avoid reallocating the large basis and
Hessenberg arrays. Residual tolerances scale with the sampled operator action,
without an absolute unit-scale floor.
"""
function krylov_liouvillian_spectrum(L;nev::Integer=6,krylovdim::Integer=max(20,2nev+4),
                                     which=:LR,initial_vector=nothing,
                                     atol::Real=1e-10,rtol::Real=1e-8,
                                     vectors::Bool=false,rng=Random.default_rng(),
                                     require_convergence::Bool=true,
                                     workspace=nothing)
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("L must be square"))
    0<nev<=n||throw(ArgumentError("nev must lie between 1 and the operator dimension"))
    m=min(n,max(Int(krylovdim),Int(nev)+1));T=_complex_float_type(eltype(L))
    initial_vector===nothing||(T=_promote_krylov_array_type(T,initial_vector))
    _check_krylov_matvec_precision(L,T)
    ws=workspace===nothing ? ArnoldiWorkspace(T,n,m) : _check_arnoldi_workspace(workspace,n,m)
    promote_type(eltype(ws.V),T)==eltype(ws.V)||throw(ArgumentError("Arnoldi workspace scalar type cannot represent the operator"))
    _check_krylov_matvec_precision(L,eltype(ws.V))
    V=view(ws.V,:,1:m+1);H=view(ws.H,1:m+1,1:m);w=ws.tmp;q=ws.q
    fill!(H,zero(eltype(H)))
    if initial_vector===nothing
        randn!(rng,q)
    else
        length(initial_vector)==n||throw(DimensionMismatch("initial_vector has wrong length"))
        q.=initial_vector
    end
    nq=norm(q);nq>0||throw(ArgumentError("initial_vector must be nonzero"))
    @views V[:,1].=q./nq;k=0
    RT=typeof(real(zero(eltype(V))));breakfactor=eps(RT)
    for j in 1:m
        mul!(w,L,view(V,:,j))
        image_norm=norm(w)
        for pass in 1:2, i in 1:j
            hij=dot(view(V,:,i),w);H[i,j]+=hij
            @views @. w=w-hij*V[:,i]
        end
        H[j+1,j]=norm(w);k=j
        (j==m||abs(H[j+1,j])<=breakfactor*max(image_norm,floatmin(RT)))&&break
        @views V[:,j+1].=w./H[j+1,j]
    end
    E=_projected_eigen(Matrix(view(H,1:k,1:k)));order=_ritz_order(E.values,which)[1:min(nev,k)]
    vals=E.values[order];Y=E.vectors[:,order]
    residuals=abs.(H[k+1,k].*Y[end,:])
    scale=max(maximum((norm(view(H,1:k+1,j)) for j in 1:k);init=zero(RT)),floatmin(RT))
    tolerance=RT(atol)+RT(rtol)*scale;converged=residuals .<= tolerance
    require_convergence&&!all(converged)&&throw(ArgumentError("Arnoldi spectrum did not converge; maximum requested Ritz residual=$(maximum(residuals)); increase krylovdim"))
    info=(values=vals,residuals=residuals,converged=converged,iterations=k,
          krylov_dimension=m,dimension=n,which=which,residual_scale=scale,
          residual_tolerance=tolerance,workspace_reused=workspace!==nothing)
    vectors ? merge(info,(vectors=Matrix(view(V,:,1:k))*Y,)) : info
end

function _project_vector!(dest,projector,src,work=nothing)
    if projector===nothing
        copyto!(dest,src)
    elseif work===nothing
        mul!(dest,projector,src)
    else
        apply!(dest,projector,src,work)
    end
end

function _orthogonalize!(q,V,k)
    for _ in 1:2, i in 1:k
        α=dot(view(V,:,i),q);@views @. q=q-α*V[:,i]
    end
    norm(q)
end

function _orthonormalize_columns!(R,ncols)
    RT=typeof(real(zero(eltype(R))));kept=0;threshold=sqrt(eps(RT))
    for j in 1:ncols
        col=view(R,:,j);initial=norm(col)
        for _ in 1:2, i in 1:kept
            qi=view(R,:,i);α=dot(qi,col)
            @. col=col-α*qi
        end
        β=norm(col)
        β<=threshold*max(initial,floatmin(RT))&&continue
        kept+=1
        kept==j||copyto!(view(R,:,kept),col)
        view(R,:,kept)./=β
    end
    kept
end

function _selected_ritz_order(values,which,target)
    target===nothing ? _ritz_order(values,which) : sortperm(abs.(values .- target))
end

function _arnoldi_expand!(L,V,H,first_column,last_column,tmp)
    k=first_column-1
    RT=typeof(real(zero(eltype(V))));breakfactor=eps(RT)
    for j in first_column:last_column
        mul!(tmp,L,view(V,:,j));image_norm=norm(tmp)
        fill!(view(H,1:j,j),zero(eltype(H)))
        for pass in 1:2, i in 1:j
            hij=dot(view(V,:,i),tmp);H[i,j]+=hij
            @views @. tmp=tmp-hij*V[:,i]
        end
        H[j+1,j]=norm(tmp);k=j
        abs(H[j+1,j])<=breakfactor*max(image_norm,floatmin(RT))&&break
        @views V[:,j+1].=tmp./H[j+1,j]
    end
    k
end

"""
    implicitly_restarted_arnoldi_spectrum(L; nev=6, krylovdim=40,
                                           retained_dimension=nev,
                                           maxrestarts=20, which=:LR,
                                           target=nothing)

Compute selected eigenpairs using matrix-free implicitly restarted Arnoldi.
Every restart applies exact unwanted Ritz shifts by implicit QR transformations
of the Hessenberg matrix; the transformed Arnoldi factorization is truncated
and expanded without materializing `L`. Set `target` to order Ritz values by
distance from an interior point, or leave it as `nothing` and select with
`which=:LR`, `:LM`, or `:SM`.

This routine is most effective for spectral-edge modes. For difficult interior
or strongly nonnormal clusters, use `jacobi_davidson_spectrum`, whose
correction equations can be preconditioned.
"""
function implicitly_restarted_arnoldi_spectrum(L;nev::Integer=6,
        krylovdim::Integer=max(30,3nev+6),
        retained_dimension::Integer=max(nev,min(2nev,krylovdim-1)),
        maxrestarts::Integer=20,which=:LR,target=nothing,
        initial_vector=nothing,atol::Real=1e-10,rtol::Real=1e-8,
        vectors::Bool=false,rng=Random.default_rng(),
        require_convergence::Bool=true,workspace=nothing)
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("L must be square"))
    0<nev<=n||throw(ArgumentError("nev must lie between 1 and the operator dimension"))
    which in (:LR,:LM,:SM)||throw(ArgumentError("which must be :LR, :LM, or :SM"))
    maxrestarts>=0||throw(ArgumentError("maxrestarts must be nonnegative"))
    _check_finite_krylov_target(target)
    m=min(n,max(Int(krylovdim),Int(nev)+1));keep=min(m-1,max(Int(retained_dimension),Int(nev)))
    T=_complex_float_type(eltype(L))
    initial_vector===nothing||(T=_promote_krylov_array_type(T,initial_vector))
    T=_promote_krylov_scalar_type(T,target)
    _check_krylov_matvec_precision(L,T)
    ws=workspace===nothing ? ArnoldiWorkspace(T,n,m) : _check_arnoldi_workspace(workspace,n,m)
    promote_type(eltype(ws.V),T)==eltype(ws.V)||throw(ArgumentError("Arnoldi workspace scalar type cannot represent the operator"))
    _check_krylov_matvec_precision(L,eltype(ws.V))
    V=view(ws.V,:,1:m+1);H=view(ws.H,1:m+1,1:m);tmp=ws.tmp;q=ws.q
    fill!(H,zero(eltype(H)))
    if initial_vector===nothing
        randn!(rng,q)
    else
        length(initial_vector)==n||throw(DimensionMismatch("initial_vector has wrong length"))
        q.=initial_vector
    end
    nq=norm(q);nq>0||throw(ArgumentError("initial_vector must be nonzero"));V[:,1].=q./nq
    RT=typeof(real(zero(eltype(V))));history=NamedTuple[];applications=0
    k=_arnoldi_expand!(L,V,H,1,m,tmp);applications+=k;best=nothing
    for cycle in 0:maxrestarts
        E=_projected_eigen(Matrix(view(H,1:k,1:k)));requested_target=target===nothing ? nothing : Complex{RT}(target)
        order=_selected_ritz_order(E.values,which,requested_target);sel=order[1:min(nev,k)]
        vals=E.values[sel];Y=E.vectors[:,sel];residuals=abs.(H[k+1,k].*Y[end,:])
        scale=max(maximum((norm(view(H,1:k+1,j)) for j in 1:k);init=zero(RT)),floatmin(RT))
        tolerance=RT(atol)+RT(rtol)*scale;converged=residuals .<= tolerance
        push!(history,(cycle,subspace_dimension=k,converged=count(converged),
                       maximum_residual=maximum(residuals),residual_scale=scale))
        best=(values=vals,residuals=residuals,converged=converged,
              iterations=applications,restarts=cycle,krylov_dimension=m,
              retained_dimension=keep,dimension=n,which=which,target=requested_target,
              algorithm=:implicit_qr_arnoldi,residual_scale=scale,
              residual_tolerance=tolerance,restart_history=copy(history),
              workspace_reused=workspace!==nothing)
        if length(vals)>=nev&&all(converged)
            return vectors ? merge(best,(vectors=Matrix(view(V,:,1:k))*Y,)) : best
        end
        cycle==maxrestarts&&break
        k>keep||break

        # Exact unwanted Ritz shifts. The implicit-Q theorem guarantees that
        # the first `keep` transformed columns retain an Arnoldi factorization.
        shifts=E.values[order[keep+1:end]]
        Hsmall=Matrix(view(H,1:k,1:k));Qtotal=Matrix{eltype(H)}(I,k,k)
        for shift in shifts
            Q=Matrix(qr(Hsmall-shift*I).Q);Hsmall=adjoint(Q)*Hsmall*Q;Qtotal=Qtotal*Q
        end
        Vplus=view(ws.Z,:,1:k);mul!(Vplus,view(V,:,1:k),Qtotal)
        # The transformed residual combines the retained Hessenberg coupling
        # with the previous Arnoldi residual (implicit-Q theorem).
        copyto!(q,view(Vplus,:,keep+1));q .*= Hsmall[keep+1,keep]
        if !iszero(H[k+1,k]);@views @. q=q+(H[k+1,k]*Qtotal[k,keep])*V[:,k+1];end
        β=_orthogonalize!(q,Vplus,keep)
        β>sqrt(eps(RT))*max(scale,floatmin(RT))||break
        V[:,1:keep].=view(Vplus,:,1:keep);V[:,keep+1].=q./β
        fill!(H,zero(eltype(H)));H[1:keep,1:keep].=view(Hsmall,1:keep,1:keep);H[keep+1,keep]=β
        k=_arnoldi_expand!(L,V,H,keep+1,m,tmp);applications+=k-keep
    end
    require_convergence&&throw(ArgumentError("implicitly restarted Arnoldi did not converge in $maxrestarts restarts; maximum requested Ritz residual=$(maximum(best.residuals))"))
    if vectors
        E=_projected_eigen(Matrix(view(H,1:k,1:k)));order=_selected_ritz_order(E.values,which,target===nothing ? nothing : Complex{RT}(target))
        sel=order[1:min(nev,k)];return merge(best,(vectors=Matrix(view(V,:,1:k))*E.vectors[:,sel],))
    end
    best
end

"""Reusable large-vector storage for `jacobi_davidson_spectrum`."""
mutable struct JacobiDavidsonWorkspace{T}
    arnoldi::ArnoldiWorkspace{T}
    correction::KrylovWorkspace{T}
    u::Vector{T}
    au::Vector{T}
    residual::Vector{T}
    correction_vector::Vector{T}
    work::Vector{T}
    image::Vector{T}
end

function JacobiDavidsonWorkspace(::Type{T},n::Integer,subspace_dim::Integer,
                                 correction_krylovdim::Integer) where T
    n>0||throw(ArgumentError("dimension must be positive"));m=min(Int(n),Int(subspace_dim))
    m>1||throw(ArgumentError("subspace_dim must exceed one"))
    JacobiDavidsonWorkspace(ArnoldiWorkspace(T,n,m),KrylovWorkspace(T,n,correction_krylovdim),
        zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n))
end

JacobiDavidsonWorkspace(L,subspace_dim::Integer=max(30,min(size(L,1),40)),
                        correction_krylovdim::Integer=min(size(L,1),20))=
    JacobiDavidsonWorkspace(_complex_float_type(eltype(L)),size(L,1),
                            subspace_dim,correction_krylovdim)

function _project_complement!(dest,src,locked,nlocked,current=nothing)
    dest===src||copyto!(dest,src)
    for _ in 1:2
        for j in 1:nlocked
            q=view(locked,:,j);α=dot(q,dest);@. dest=dest-α*q
        end
        if current!==nothing;α=dot(current,dest);@. dest=dest-α*current;end
    end
    dest
end

struct _JDProjectedPreconditioner{P,M,V}
    base::P
    locked::M
    nlocked::Int
    current::V
end
size(P::_JDProjectedPreconditioner)=size(P.base)
size(P::_JDProjectedPreconditioner,i::Integer)=size(P.base,i)
function ldiv!(dest::AbstractVector,P::_JDProjectedPreconditioner,src::AbstractVector)
    ldiv!(dest,P.base,src);_project_complement!(dest,dest,P.locked,P.nlocked,P.current)
end

function _random_complement!(dest,rng,locked,nlocked;attempts=8)
    RT=typeof(real(zero(eltype(dest))))
    for _ in 1:attempts
        randn!(rng,dest);initial=norm(dest);_project_complement!(dest,dest,locked,nlocked);β=norm(dest)
        β>sqrt(eps(RT))*max(initial,floatmin(RT))&&(dest./=β;return dest)
    end
    throw(ArgumentError("could not construct a starting vector outside the hard-locked invariant subspace"))
end

"""
    jacobi_davidson_spectrum(L; nev=6, target=0, subspace_dim=40,
                             preconditioner=nothing)

Compute matrix-free eigenpairs nearest `target` with restarted
Jacobi--Davidson. The inexact correction equation
`P*(L-theta*I)*P*t=-r` is solved by restarted GMRES, where `P` removes the
current Ritz vector and the converged invariant subspace. Converged Schur
directions are hard locked. Pass any reusable object supporting
`ldiv!(y,P,x)` as `preconditioner`; it is projected consistently inside the
correction solve.

Returned right vectors are reconstructed from the final locked Ritz block and
validated with explicit `norm(L*v-lambda*v)` residuals. Only `mul!` and, when
requested, `ldiv!` actions are used.
"""
function jacobi_davidson_spectrum(L;nev::Integer=6,target=0,
        subspace_dim::Integer=max(30,3nev+6),maxiter::Integer=200,
        correction_krylovdim::Integer=min(size(L,1),20),
        correction_maxiter::Integer=60,correction_atol::Real=0,
        correction_rtol::Real=0.1,preconditioner=nothing,
        initial_vector=nothing,atol::Real=1e-10,rtol::Real=1e-8,
        vectors::Bool=false,rng=Random.default_rng(),
        require_convergence::Bool=true,operator_scale=nothing,workspace=nothing)
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("L must be square"))
    0<nev<=n||throw(ArgumentError("nev must lie between 1 and the operator dimension"))
    maxiter>0||throw(ArgumentError("maxiter must be positive"));correction_maxiter>0||throw(ArgumentError("correction_maxiter must be positive"))
    correction_atol>=0&&correction_rtol>=0||throw(ArgumentError("correction tolerances must be nonnegative"))
    _check_finite_krylov_target(target)
    operator_scale===nothing||(operator_scale isa Real&&isfinite(operator_scale)&&
        operator_scale>=0)||throw(ArgumentError(
        "Jacobi--Davidson operator_scale must be finite and nonnegative"))
    m=min(n,max(Int(subspace_dim),Int(nev)+2));T=_complex_float_type(eltype(L))
    initial_vector===nothing||(T=_promote_krylov_array_type(T,initial_vector))
    T=_promote_krylov_scalar_type(T,target)
    T=_promote_krylov_scalar_type(T,operator_scale)
    preconditioner===nothing||(T=_promote_krylov_operator_type(T,preconditioner))
    _check_krylov_matvec_precision(L,T)
    ws=workspace===nothing ? JacobiDavidsonWorkspace(T,n,m,correction_krylovdim) : workspace
    ws isa JacobiDavidsonWorkspace||throw(ArgumentError(
        "workspace must be a JacobiDavidsonWorkspace"))
    workspace_type=eltype(ws.arnoldi.V)
    RT=typeof(real(zero(workspace_type)))
    promote_type(workspace_type,T)===workspace_type||throw(ArgumentError(
        "Jacobi--Davidson workspace scalar type $workspace_type cannot represent operator scalar type $T"))
    _check_krylov_matvec_precision(L,workspace_type)
    aws=_check_arnoldi_workspace(ws.arnoldi,n,m);cws=ws.correction
    size(cws.V,1)==n||throw(DimensionMismatch("Jacobi--Davidson correction workspace has the wrong dimension"))
    eltype(cws.V)===workspace_type||throw(ArgumentError(
        "Jacobi--Davidson Arnoldi and correction workspaces use incompatible scalar types"))
    size(cws.H,2)>=min(n,correction_krylovdim)||throw(DimensionMismatch(
        "Jacobi--Davidson correction workspace Krylov basis is too small"))
    all(vector->eltype(vector)===workspace_type,
        (ws.u,ws.au,ws.residual,ws.correction_vector,ws.work,ws.image))||
        throw(ArgumentError("Jacobi--Davidson vector scratch uses an incompatible scalar type"))
    preconditioner===nothing||size(preconditioner)==(n,n)||throw(DimensionMismatch("preconditioner has wrong dimension"))
    V=view(aws.V,:,1:m);LV=view(aws.LV,:,1:m);locked=view(aws.X,:,1:m)
    locked_images=view(aws.LX,:,1:m);smallA=view(aws.A,1:m,1:m)
    targetT=Complex{RT}(target)
    raw_scale=operator_scale===nothing ?
        _estimated_operator_scale!(L,ws.work,ws.image) : operator_scale
    raw_scale isa Real&&isfinite(raw_scale)&&raw_scale>=0||throw(ArgumentError(
        "Jacobi--Davidson operator_scale must be finite and nonnegative"))
    scale=float(raw_scale)
    if iszero(scale)
        values=fill(zero(workspace_type),nev)
        residuals=zeros(RT,nev);converged=trues(nev)
        history=NamedTuple[]
        info=(values,residuals,converged,iterations=0,
              operator_applications=0,correction_iterations=0,
              correction_failures=0,restarts=0,krylov_dimension=m,
              dimension=n,target=targetT,hard_locked=nev,
              algorithm=:jacobi_davidson,
              preconditioned=preconditioner!==nothing,
              residual_scale=scale,residual_tolerance=RT(atol),
              restart_history=history,workspace_reused=workspace!==nothing)
        if vectors
            U=zeros(workspace_type,n,nev)
            for index in 1:nev;U[index,index]=one(workspace_type);end
            return merge(info,(vectors=U,))
        end
        return info
    end
    tolerance=RT(atol)+RT(rtol)*scale;nlocked=0;k=1;restarts=0;applications=Ref(0)
    correction_iterations=0;correction_failures=0;history=NamedTuple[]
    if initial_vector===nothing
        _random_complement!(view(V,:,1),rng,locked,0)
    else
        length(initial_vector)==n||throw(DimensionMismatch("initial_vector has wrong length"));V[:,1].=initial_vector
        nq=norm(view(V,:,1));nq>0||throw(ArgumentError("initial_vector must be nonzero"));V[:,1]./=nq
    end

    for outer in 1:maxiter
        mul!(ws.image,L,view(V,:,k));applications[]+=1;_project_complement!(view(LV,:,k),ws.image,locked,nlocked)
        Vk=view(V,:,1:k);LVk=view(LV,:,1:k);H=view(smallA,1:k,1:k)
        mul!(H,adjoint(Vk),LVk);E=_projected_eigen(Matrix(H));idx=argmin(abs.(E.values .- targetT))
        theta=E.values[idx];y=view(E.vectors,:,idx);mul!(ws.u,Vk,y);mul!(ws.au,LVk,y)
        nu=norm(ws.u);ws.u./=nu;ws.au./=nu;@. ws.residual=ws.au-theta*ws.u
        residual_norm=norm(ws.residual)
        push!(history,(outer,subspace_dimension=k,hard_locked=nlocked,value=theta,residual=residual_norm))
        if residual_norm<=tolerance
            nlocked+=1;locked[:,nlocked].=ws.u
            _project_complement!(view(locked,:,nlocked),view(locked,:,nlocked),locked,nlocked-1)
            view(locked,:,nlocked)./=norm(view(locked,:,nlocked))
            mul!(view(locked_images,:,nlocked),L,view(locked,:,nlocked));applications[]+=1
            k=0;nlocked>=nev&&break
            k=1;_random_complement!(view(V,:,1),rng,locked,nlocked);continue
        end
        if nlocked+k>=m
            V[:,1].=ws.u;k=1;restarts+=1
            # `V[:,1]` no longer matches its cached image after the collapse.
            # Refresh it before the next projected Ritz extraction.
            mul!(ws.image,L,view(V,:,1));applications[]+=1
            _project_complement!(view(LV,:,1),ws.image,locked,nlocked)
        end

        @. ws.residual=-ws.residual;fill!(ws.correction_vector,zero(T))
        locked_count=nlocked
        function correction_action!(dest,z)
            _project_complement!(ws.work,z,locked,locked_count,ws.u);mul!(ws.image,L,ws.work);applications[]+=1
            @. ws.image=ws.image-theta*ws.work;_project_complement!(dest,ws.image,locked,locked_count,ws.u)
        end
        projected_preconditioner=preconditioner===nothing ? nothing : _JDProjectedPreconditioner(preconditioner,locked,nlocked,ws.u)
        correction_info=_gmres!(ws.correction_vector,correction_action!,ws.residual,cws;
            atol=correction_atol,rtol=correction_rtol,maxiter=correction_maxiter,preconditioner=projected_preconditioner)
        correction_iterations+=correction_info.iterations;correction_info.converged||(correction_failures+=1)
        _project_complement!(ws.correction_vector,ws.correction_vector,locked,nlocked,ws.u)
        β=_orthogonalize!(ws.correction_vector,V,k)
        if β<=sqrt(eps(RT))*max(norm(ws.residual),floatmin(RT))
            copyto!(ws.correction_vector,ws.residual)
            preconditioner!==nothing&&ldiv!(ws.correction_vector,projected_preconditioner,ws.residual)
            _project_complement!(ws.correction_vector,ws.correction_vector,locked,nlocked,ws.u)
            β=_orthogonalize!(ws.correction_vector,V,k)
        end
        if β<=sqrt(eps(RT))*max(norm(ws.residual),floatmin(RT))
            _random_complement!(ws.correction_vector,rng,locked,nlocked);β=_orthogonalize!(ws.correction_vector,V,k)
        end
        β>0||throw(ArgumentError("Jacobi--Davidson expansion lost the search direction"))
        k+=1;V[:,k].=ws.correction_vector./β
    end

    # Explicitly reconstruct and validate right Ritz vectors from the final
    # invariant/search subspace.
    total=nlocked+(nlocked>=nev ? 0 : k);W=view(aws.Z,:,1:total)
    total>0||throw(ArgumentError("Jacobi--Davidson did not construct a search subspace"))
    nlocked>0&&(W[:,1:nlocked].=view(locked,:,1:nlocked))
    total>nlocked&&(W[:,nlocked+1:total].=view(V,:,1:k))
    AW=view(aws.LX,:,1:total)
    for j in nlocked+1:total;mul!(view(AW,:,j),L,view(W,:,j));applications[]+=1;end
    H=Matrix(adjoint(W)*AW);E=_projected_eigen(H);order=sortperm(abs.(E.values .- targetT));sel=order[1:min(nev,total)]
    vals=E.values[sel];Y=E.vectors[:,sel];U=Matrix(W*Y);AU=Matrix(AW*Y)
    residuals=Vector{RT}(undef,length(sel))
    for j in eachindex(sel)
        nu=norm(view(U,:,j));U[:,j]./=nu;AU[:,j]./=nu;@views @. ws.work=AU[:,j]-vals[j]*U[:,j]
        residuals[j]=norm(ws.work)
    end
    converged=residuals .<= tolerance
    info=(values=vals,residuals=residuals,converged=converged,iterations=length(history),
          operator_applications=applications[],correction_iterations,correction_failures,restarts,
          krylov_dimension=m,dimension=n,target=targetT,hard_locked=nlocked,
          algorithm=:jacobi_davidson,preconditioned=preconditioner!==nothing,
          residual_scale=scale,residual_tolerance=tolerance,restart_history=history,
          workspace_reused=workspace!==nothing)
    require_convergence&&!(length(vals)>=nev&&all(converged))&&throw(ArgumentError("Jacobi--Davidson did not converge in $maxiter outer iterations; hard_locked=$nlocked, maximum requested Ritz residual=$(maximum(residuals))"))
    vectors ? merge(info,(vectors=U,)) : info
end

function _projector_workspace(projector,work)
    work!==nothing&&return work
    projector===nothing&&return nothing
    if isdefined(@__MODULE__,:MatrixFreeSymmetryProjector)&&
       projector isa getfield(@__MODULE__,:MatrixFreeSymmetryProjector)
        return getfield(@__MODULE__,:SymmetryProjectorWorkspace)(projector)
    end
    nothing
end

"""
    harmonic_arnoldi_spectrum(L; nev=6, krylovdim=40, thickdim=12,
                              maxrestarts=20, projector=nothing)

Compute eigenvalues nearest `target` with thick-restarted harmonic Arnoldi.
Converged and best unconverged harmonic Ritz vectors are retained between
cycles. An optional matrix-free orthogonal `projector` restricts every basis
and residual vector to a weak-symmetry charge sector.
`workspace` reuses Arnoldi and pencil storage; `projector_workspace` supplies
caller-owned symmetry scratch. Restart reports include locked-mode counts and
per-cycle residual history.
"""
function harmonic_arnoldi_spectrum(L;nev::Integer=6,krylovdim::Integer=max(30,3nev+6),
                                   thickdim::Integer=max(nev+2,2nev),
                                   maxrestarts::Integer=20,target=0,
                                   initial_vector=nothing,projector=nothing,
                                   atol::Real=1e-10,rtol::Real=1e-8,
                                   vectors::Bool=false,rng=Random.default_rng(),
                                   require_convergence::Bool=true,
                                   workspace=nothing,projector_workspace=nothing)
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("L must be square"))
    0<nev<=n||throw(ArgumentError("nev must lie between 1 and the operator dimension"))
    m=min(n,max(Int(krylovdim),Int(nev)+2));keep=min(m-1,max(Int(thickdim),Int(nev)))
    maxrestarts>=0||throw(ArgumentError("maxrestarts must be nonnegative"))
    projector===nothing||size(projector)==(n,n)||throw(DimensionMismatch("projector has wrong dimension"))
    _check_finite_krylov_target(target)
    T=_complex_float_type(eltype(L))
    initial_vector===nothing||(T=_promote_krylov_array_type(T,initial_vector))
    T=_promote_krylov_scalar_type(T,target)
    projector===nothing||(T=_promote_krylov_operator_type(T,projector))
    _check_krylov_matvec_precision(L,T)
    ws=workspace===nothing ? ArnoldiWorkspace(T,n,m) : _check_arnoldi_workspace(workspace,n,m)
    promote_type(eltype(ws.V),T)==eltype(ws.V)||throw(ArgumentError("Arnoldi workspace scalar type cannot represent the operator"))
    _check_krylov_matvec_precision(L,eltype(ws.V))
    V=view(ws.V,:,1:m);LV=view(ws.LV,:,1:m);Zwork=view(ws.Z,:,1:m)
    Xwork=view(ws.X,:,1:m);LXwork=view(ws.LX,:,1:m);q=ws.q;tmp=ws.tmp
    pwork=_projector_workspace(projector,projector_workspace)
    if initial_vector===nothing
        randn!(rng,tmp)
    else
        length(initial_vector)==n||throw(DimensionMismatch("initial_vector has wrong length"))
        tmp.=initial_vector
    end
    seednorm=norm(tmp);_project_vector!(q,projector,tmp,pwork);nq=norm(q)
    RT=typeof(real(zero(eltype(V))));breakfactor=sqrt(eps(RT))
    nq>breakfactor*max(seednorm,floatmin(RT))||throw(ArgumentError("initial_vector has zero component in the selected symmetry sector"))
    V[:,1].=q./nq;nkeep=1;applications=0;best=nothing;history=NamedTuple[]
    for cycle in 0:maxrestarts
        k=nkeep
        # Images of retained vectors are recomputed once per restart. Expansion
        # uses the newest image as a block-Krylov continuation direction.
        for j in 1:nkeep
            mul!(tmp,L,view(V,:,j));applications+=1
            _project_vector!(view(LV,:,j),projector,tmp,pwork)
        end
        while k<m
            copyto!(q,view(LV,:,k));image_norm=norm(q);β=_orthogonalize!(q,V,k)
            if β<=breakfactor*max(image_norm,floatmin(RT))
                randn!(rng,tmp);seednorm=norm(tmp)
                _project_vector!(q,projector,tmp,pwork);projected_norm=norm(q)
                β=_orthogonalize!(q,V,k)
                β<=breakfactor*max(projected_norm,seednorm*eps(RT),floatmin(RT))&&break
            end
            k+=1;V[:,k].=q./β
            mul!(tmp,L,view(V,:,k));applications+=1
            _project_vector!(view(LV,:,k),projector,tmp,pwork)
        end
        Vk=view(V,:,1:k);LVk=view(LV,:,1:k)
        requested_target=Complex{RT}(target)
        action_scale=max(maximum((norm(view(LVk,:,j)) for j in 1:k);init=zero(RT)),floatmin(RT))
        # At an exact eigenvalue (notably the stationary zero mode), the
        # harmonic pencil has a common null vector. A tiny offset makes the
        # pencil regular without changing the requested tolerance scale.
        sigma=iszero(requested_target) ? Complex{RT}(-max(sqrt(eps(RT))*action_scale,atol)) : requested_target
        # Harmonic Rayleigh--Ritz condition:
        # (A V)'(A V y - μ V y)=0 for A=L-σI.
        Z=view(Zwork,:,1:k);@. Z=LVk-sigma*Vk
        A=view(ws.A,1:k,1:k);B=view(ws.B,1:k,1:k)
        mul!(A,adjoint(Z),Z);mul!(B,adjoint(Z),Vk)
        E=_projected_eigen!(A,B);finite=findall(i->isfinite(real(E.values[i]))&&isfinite(imag(E.values[i])),eachindex(E.values))
        isempty(finite)&&throw(ArgumentError("harmonic Ritz pencil is singular; enlarge krylovdim or change the initial vector"))
        lambda_all=sigma .+ E.values
        order=finite[sortperm(abs.(lambda_all[finite].-requested_target))];sel=order[1:min(nev,length(order))]
        vals=sigma .+ E.values[sel];Y=view(E.vectors,:,sel);nsel=length(sel)
        X=view(Xwork,:,1:nsel);LX=view(LXwork,:,1:nsel)
        mul!(X,Vk,Y);mul!(LX,LVk,Y)
        for j in 1:nsel
            nx=norm(view(X,:,j));X[:,j]./=nx;LX[:,j]./=nx
        end
        residuals=Vector{RT}(undef,nsel)
        for j in 1:nsel
            copyto!(q,view(LX,:,j));lambda=vals[j]
            @views @. q=q-lambda*X[:,j]
            residuals[j]=norm(q)
        end
        tolerance=RT(atol)+RT(rtol)*action_scale;converged=residuals .<= tolerance
        locked=count(converged)
        push!(history,(cycle,subspace_dimension=k,locked,
                       maximum_residual=maximum(residuals),residual_scale=action_scale))
        best=(values=vals,residuals=residuals,converged=converged,
              restarts=cycle,iterations=applications,krylov_dimension=m,
              retained_dimension=keep,final_retained_dimension=nkeep,
              locked,dimension=n,target=requested_target,
              harmonic_shift=sigma,algorithm=:harmonic,
              residual_scale=action_scale,residual_tolerance=tolerance,
              restart_history=copy(history),workspace_reused=workspace!==nothing)
        if length(vals)>=nev&&all(converged)
            return vectors ? merge(best,(vectors=copy(X),)) : best
        end
        cycle==maxrestarts&&break
        retain=Int[]
        for (j,idx) in pairs(sel)
            converged[j]&&push!(retain,idx)
        end
        for idx in order
            idx in retain&&continue
            push!(retain,idx);length(retain)>=min(keep,length(order))&&break
        end
        R=view(Zwork,:,1:length(retain));mul!(R,Vk,view(E.vectors,:,retain))
        nkeep=_orthonormalize_columns!(R,length(retain))
        nkeep>0||throw(ArgumentError("harmonic restart lost the retained invariant subspace"))
        V[:,1:nkeep].=view(R,:,1:nkeep)
    end
    require_convergence&&throw(ArgumentError("thick-restarted harmonic Arnoldi did not converge in $maxrestarts restarts; maximum requested Ritz residual=$(maximum(best.residuals))"))
    vectors ? merge(best,(vectors=copy(view(Xwork,:,1:length(best.values))),)) : best
end
