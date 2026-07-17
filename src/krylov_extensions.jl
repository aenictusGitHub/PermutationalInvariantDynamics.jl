# Shared matrix-free helpers -------------------------------------------------

@inline function _advanced_krylov_eltype(A)
    S=try
        eltype(A)
    catch
        Any
    end
    S isa Type&&S<:Number&&isconcretetype(S) ? S : nothing
end

function _advanced_krylov_apply!(dest,A,src)
    if applicable(A,dest,src)
        A(dest,src)
    elseif applicable(A,src)
        value=A(src)
        length(value)==length(dest)||throw(DimensionMismatch(
            "callable Krylov operator returned a vector of the wrong length"))
        copyto!(dest,value)
    elseif applicable(mul!,dest,A,src)
        mul!(dest,A,src)
    else
        throw(ArgumentError(
            "operator must support mul!(dest, A, src), A(dest, src), or A(src)"))
    end
    dest
end

function _advanced_krylov_ldiv!(dest,P,src)
    if P===nothing
        dest===src||copyto!(dest,src)
    elseif applicable(ldiv!,dest,P,src)
        ldiv!(dest,P,src)
    elseif applicable(P,dest,src)
        P(dest,src)
    else
        throw(ArgumentError(
            "preconditioner must support ldiv!(dest, P, src) or P(dest, src)"))
    end
    dest
end

function _advanced_krylov_dimension(A,n::Integer)
    dims=try
        size(A)
    catch
        nothing
    end
    dims===nothing&&return n
    length(dims)==2||throw(DimensionMismatch("Krylov operator must be two-dimensional"))
    dims==(n,n)||throw(DimensionMismatch(
        "Krylov operator has size $dims but vectors have length $n"))
    n
end

function _advanced_promote_scalar_type(::Type{T},x) where T
    if x isa Integer
        R=_real_float_type(T)
        converted=try
            R(x)
        catch
            nothing
        end
        if converted!==nothing&&isfinite(converted)&&BigInt(converted)==BigInt(x)
            return T
        end
    end
    _promote_krylov_scalar_type(T,x)
end

function _advanced_krylov_type(A,arrays...;scalars=())
    S=_advanced_krylov_eltype(A)
    T=S===nothing ? Union{} : _complex_float_type(S)
    for array in arrays
        eltype(array)<:Number&&isconcretetype(eltype(array))||throw(ArgumentError(
            "Krylov inputs must expose a concrete numeric eltype"))
        Ta=_complex_float_type(eltype(array))
        T=T===Union{} ? Ta : promote_type(T,Ta)
    end
    T===Union{}&&(T=ComplexF64)
    for scalar in scalars
        T=_advanced_promote_scalar_type(T,scalar)
    end
    T
end

function _advanced_check_operator_precision(A,::Type{T}) where T
    S=_advanced_krylov_eltype(A)
    if S!==nothing
        operator_type=_complex_float_type(S)
        promote_type(T,operator_type)===T||throw(ArgumentError(
            "Krylov workspace scalar type $T cannot represent operator scalar type $operator_type"))
        _check_krylov_matvec_precision(A,T)
    end
    T
end

function _advanced_check_tolerances(atol,rtol,::Type{R}) where R
    atol isa Real&&rtol isa Real&&isfinite(atol)&&isfinite(rtol)&&
        atol>=0&&rtol>=0||throw(ArgumentError(
        "Krylov tolerances must be finite and nonnegative"))
    a=R(atol);r=R(rtol)
    isfinite(a)&&isfinite(r)&&a>=zero(R)&&r>=zero(R)||throw(ArgumentError(
        "Krylov tolerances must be finite and nonnegative"))
    a,r
end

function _advanced_projected_least_squares(A,B)
    try
        qr(A,ColumnNorm())\B
    catch error
        error isa Union{MethodError,SingularException}||rethrow()
        R=_real_float_type(eltype(A))
        throw(ArgumentError(
            "rank-revealing projected least squares is unavailable for scalar type $R with Julia's active LinearAlgebra backend; use Float32/Float64 data or load a compatible generic factorization backend"))
    end
end

function _advanced_expv_step_control(::Type{R},value,label;
        allow_zero::Bool) where R<:AbstractFloat
    value isa Real&&isfinite(value)||throw(ArgumentError(
        "$label must be finite and real"))
    valid_sign=allow_zero ? value>=0 : value>0
    valid_sign||throw(ArgumentError(
        "$label must be $(allow_zero ? "nonnegative" : "positive")"))
    if value isa Integer
        converted=R(value)
        isfinite(converted)&&try
            BigInt(converted)==BigInt(value)
        catch
            false
        end||throw(ArgumentError(
            "$label is not exactly representable in $R"))
        return converted
    end
    promote_type(R,typeof(value))===R||throw(ArgumentError(
        "$label scalar type $(typeof(value)) would narrow in $R precision"))
    converted=R(value)
    converted_sign=allow_zero ? converted>=zero(R) : converted>zero(R)
    isfinite(converted)&&converted_sign||throw(ArgumentError(
        "$label is not representable as a finite $(allow_zero ? "nonnegative" : "positive") value in $R"))
    converted
end

function _advanced_check_preconditioner_dimension(P,n)
    P===nothing&&return P
    dims=try
        size(P)
    catch error
        error isa MethodError||rethrow()
        nothing
    end
    dims===nothing||dims==(n,n)||throw(DimensionMismatch(
        "preconditioner has size $dims, expected ($n, $n)"))
    P
end

function _advanced_check_preconditioner_precision(P,::Type{T}) where T
    P===nothing&&return P
    S=_advanced_krylov_eltype(P)
    S===nothing&&return P # A callable's numerical precision is caller-owned.
    preconditioner_type=_complex_float_type(S)
    promote_type(T,preconditioner_type)===T||throw(ArgumentError(
        "Krylov workspace scalar type $T cannot represent preconditioner scalar type $preconditioner_type"))
    P
end

@inline function _advanced_checked_norm(value,description)
    result=norm(value)
    isfinite(result)||throw(ArgumentError(
        "$description has a nonfinite norm; rescale the problem or use wider precision"))
    result
end


# Block GMRES ---------------------------------------------------------------

"""
    BlockGMRESWorkspace(T, n, nrhs, block_krylovdim)
    BlockGMRESWorkspace(A, nrhs, block_krylovdim)

Reusable storage for [`block_gmres!`](@ref). `block_krylovdim` counts block
Arnoldi steps, so the dominant basis contains at most
`nrhs * (block_krylovdim + 1)` vectors. A workspace is mutable and must be
owned by one task at a time.
"""
mutable struct BlockGMRESWorkspace{T}
    V::Matrix{T}
    H::Matrix{T}
    W::Matrix{T}
    residual::Matrix{T}
    projected_residual::Matrix{T}
    image::Matrix{T}
    G::Matrix{T}
    correction::Matrix{T}
end

function BlockGMRESWorkspace(::Type{T},n::Integer,nrhs::Integer,
                             block_krylovdim::Integer) where T
    n>0||throw(ArgumentError("dimension must be positive"))
    nrhs>0||throw(ArgumentError("nrhs must be positive"))
    block_krylovdim>0||throw(ArgumentError(
        "block_krylovdim must be positive"))
    p=Int(nrhs);m=Int(block_krylovdim);cols=p*m;rows=p*(m+1)
    BlockGMRESWorkspace(zeros(T,n,rows),zeros(T,rows,cols),zeros(T,n,p),
        zeros(T,n,p),zeros(T,n,p),zeros(T,n,p),zeros(T,rows,p),
        zeros(T,n,p))
end

function BlockGMRESWorkspace(A,nrhs::Integer,block_krylovdim::Integer=10)
    n=size(A,1);size(A,2)==n||throw(DimensionMismatch("operator must be square"))
    S=_advanced_krylov_eltype(A)
    S===nothing&&throw(ArgumentError(
        "use BlockGMRESWorkspace(T, n, nrhs, block_krylovdim) for a callable operator"))
    BlockGMRESWorkspace(_complex_float_type(S),n,nrhs,block_krylovdim)
end

function _check_block_workspace(ws::BlockGMRESWorkspace,n,p)
    size(ws.V,1)==n||throw(DimensionMismatch("block-GMRES workspace has the wrong dimension"))
    size(ws.W)==(n,p)&&size(ws.residual)==(n,p)&&
        size(ws.projected_residual)==(n,p)&&size(ws.image)==(n,p)&&
        size(ws.correction)==(n,p)||throw(DimensionMismatch(
            "block-GMRES workspace has the wrong right-hand-side count"))
    size(ws.H,1)==size(ws.V,2)&&size(ws.G,1)==size(ws.V,2)&&
        size(ws.H,2)+p==size(ws.V,2)||throw(DimensionMismatch(
            "block-GMRES workspace arrays are inconsistent"))
    ws
end

# Factor the columns of W as Q*R, dropping dependent columns. Q may alias
# neither W nor R. The no-pivot order is deterministic and preserves the
# original right-hand-side ordering in R.
function _advanced_column_factor!(Q,W,R;breakdown_factor=nothing)
    n,p=size(W);size(Q,1)==n||throw(DimensionMismatch("factor basis has the wrong row count"))
    size(Q,2)>=p||throw(DimensionMismatch("factor basis has too few columns"))
    size(R,1)>=p&&size(R,2)>=p||throw(DimensionMismatch("factor array is too small"))
    fill!(R,zero(eltype(R)))
    RT=typeof(real(zero(eltype(W))))
    breakdown_factor===nothing||(
        breakdown_factor isa Real&&isfinite(breakdown_factor)&&
        breakdown_factor>=0)||throw(ArgumentError(
        "breakdown tolerance must be finite and nonnegative"))
    factor=breakdown_factor===nothing ? sqrt(eps(RT)) : RT(breakdown_factor)
    factor>=zero(RT)&&isfinite(factor)||throw(ArgumentError(
        "breakdown tolerance must be finite and nonnegative"))
    scale=maximum((_advanced_checked_norm(view(W,:,j),
        "block-Arnoldi input column $j") for j in 1:p);init=zero(RT))
    threshold=factor*max(scale,floatmin(RT));rank=0
    for j in 1:p
        w=view(W,:,j)
        for pass in 1:2, i in 1:rank
            q=view(Q,:,i);alpha=dot(q,w);R[i,j]+=alpha
            @. w=w-alpha*q
        end
        beta=_advanced_checked_norm(w,"block-Arnoldi remainder column $j")
        beta<=threshold&&continue
        rank+=1;R[rank,j]=beta
        @views Q[:,rank].=w./beta
    end
    rank
end

function _advanced_apply_columns!(dest,A,src)
    size(dest)==size(src)||throw(DimensionMismatch("operator block arrays differ in size"))
    if !(A isa Function)&&applicable(mul!,dest,A,src)
        try
            mul!(dest,A,src)
            return dest
        catch error
            error isa MethodError||rethrow()
        end
    end
    for j in axes(src,2)
        _advanced_krylov_apply!(view(dest,:,j),A,view(src,:,j))
    end
    dest
end

function _advanced_precondition_columns!(dest,P,src)
    size(dest)==size(src)||throw(DimensionMismatch("preconditioner block arrays differ in size"))
    if P===nothing
        dest===src||copyto!(dest,src)
    else
        if !(P isa Function)&&applicable(ldiv!,dest,P,src)
            try
                ldiv!(dest,P,src)
                return dest
            catch error
                error isa MethodError||rethrow()
            end
        end
        for j in axes(src,2)
            _advanced_krylov_ldiv!(view(dest,:,j),P,view(src,:,j))
        end
    end
    dest
end

function _advanced_block_residuals!(ws,A,X,B,P)
    _advanced_apply_columns!(ws.image,A,X)
    @. ws.residual=B-ws.image
    _advanced_precondition_columns!(ws.projected_residual,P,ws.residual)
    raw=[_advanced_checked_norm(view(ws.residual,:,j),
        "block-GMRES residual column $j") for j in axes(B,2)]
    projected=[_advanced_checked_norm(view(ws.projected_residual,:,j),
        "block-GMRES projected residual column $j") for j in axes(B,2)]
    raw,projected
end

"""
    block_gmres!(X, A, B, workspace; ...)

Solve `A*X = B` with restarted block GMRES using only matrix-free vector
applications. Linearly dependent residual columns are deflated rather than
duplicated in the block basis. A fixed left `preconditioner` may be supplied
through `ldiv!`; convergence is always checked against both the projected and
the original residual of every right-hand side.

The returned named tuple contains `solution`, per-column `residuals`,
`projected_residuals`, `converged`, block `iterations`, `restarts`, and the
number of vector `operator_applications`. Failure raises unless
`require_convergence=false` is explicitly requested.
"""
function block_gmres!(X::AbstractMatrix,A,B::AbstractMatrix,
                      ws::BlockGMRESWorkspace;atol::Real=1e-10,
                      rtol::Real=1e-8,maxiter::Integer=100,
                      preconditioner=nothing,breakdown_tol=nothing,
                      require_convergence::Bool=true)
    n,p=size(B);size(X)==(n,p)||throw(DimensionMismatch(
        "solution and right-hand-side blocks must have the same size"))
    Base.mightalias(X,B)&&throw(ArgumentError("X and B must not alias"))
    _advanced_krylov_dimension(A,n);maxiter>0||throw(ArgumentError(
        "maxiter must be positive"))
    _advanced_check_preconditioner_dimension(preconditioner,n)
    _check_block_workspace(ws,n,p)
    T=eltype(ws.V);promote_type(T,eltype(X),eltype(B))===T&&
        promote_type(eltype(X),T)===eltype(X)||throw(ArgumentError(
        "block-GMRES workspace scalar type cannot represent the inputs"))
    _advanced_check_operator_precision(A,T)
    _advanced_check_preconditioner_precision(preconditioner,T)
    RT=typeof(real(zero(T)));atolT,rtolT=_advanced_check_tolerances(atol,rtol,RT)
    btol=[atolT+rtolT*_advanced_checked_norm(view(B,:,j),
        "block-GMRES right-hand-side column $j") for j in 1:p]
    all(isfinite,btol)||throw(ArgumentError(
        "block-GMRES residual tolerance overflowed; rescale the problem or use wider precision"))
    _advanced_precondition_columns!(ws.W,preconditioner,B)
    projected_btol=[atolT+rtolT*_advanced_checked_norm(view(ws.W,:,j),
        "block-GMRES preconditioned right-hand-side column $j") for j in 1:p]
    all(isfinite,projected_btol)||throw(ArgumentError(
        "block-GMRES projected tolerance overflowed; rescale the problem or use wider precision"))
    raw,projected=_advanced_block_residuals!(ws,A,X,B,preconditioner)
    applications=p;iterations=0;restarts=0
    all(j->raw[j]<=btol[j]&&projected[j]<=projected_btol[j],1:p)&&return (
        solution=X,residuals=raw,projected_residuals=projected,
        converged=trues(p),iterations,restarts,operator_applications=applications,
        block_krylov_dimension=size(ws.H,2)÷p,preconditioned=preconditioner!==nothing,
        workspace_reused=true)

    maxblocks=size(ws.H,2)÷p
    while iterations<maxiter
        fill!(ws.H,zero(T));fill!(ws.G,zero(T))
        # R0 = V1*S. The initial block may have rank below p.
        copyto!(ws.W,ws.projected_residual)
        qfirst=_advanced_column_factor!(view(ws.V,:,1:p),ws.W,
            view(ws.G,1:p,1:p);breakdown_factor=breakdown_tol)
        qfirst>0||throw(ArgumentError(
            "left preconditioning annihilated a nonconverged residual block"))
        basis_end=qfirst;current_start=1;current_end=qfirst
        search_end=0;row_count=qfirst;blocks_done=0
        while blocks_done<maxblocks&&iterations<maxiter
            qcurrent=current_end-current_start+1
            columns=search_end+1:search_end+qcurrent
            # Apply the left-preconditioned operator to the current block.
            _advanced_apply_columns!(view(ws.image,:,1:qcurrent),A,
                view(ws.V,:,current_start:current_end));applications+=qcurrent
            _advanced_precondition_columns!(view(ws.W,:,1:qcurrent),
                preconditioner,view(ws.image,:,1:qcurrent))
            # Two-pass block MGS against the complete existing basis.
            for local_index in 1:qcurrent
                w=view(ws.W,:,local_index);column=first(columns)+local_index-1
                for pass in 1:2, i in 1:basis_end
                    vi=view(ws.V,:,i);alpha=dot(vi,w);ws.H[i,column]+=alpha
                    @. w=w-alpha*vi
                end
            end
            available=min(p,size(ws.V,2)-basis_end)
            qnext=0
            if available>0
                qnext=_advanced_column_factor!(
                    view(ws.V,:,basis_end+1:basis_end+available),
                    view(ws.W,:,1:qcurrent),
                    view(ws.H,basis_end+1:basis_end+available,columns);
                    breakdown_factor=breakdown_tol)
            end
            search_end+=qcurrent;row_count=basis_end+qnext
            iterations+=1;blocks_done+=1
            qnext==0&&break
            current_start=basis_end+1;current_end=basis_end+qnext
            basis_end+=qnext
        end
        search_end>0||throw(ArgumentError("block Arnoldi produced no search direction"))
        Hbar=Matrix(view(ws.H,1:row_count,1:search_end))
        Gbar=Matrix(view(ws.G,1:row_count,1:p))
        Y=_advanced_projected_least_squares(Hbar,Gbar)
        mul!(ws.correction,view(ws.V,:,1:search_end),Y)
        X .+= ws.correction
        restarts+=1
        raw,projected=_advanced_block_residuals!(ws,A,X,B,preconditioner)
        applications+=p
        all(j->raw[j]<=btol[j]&&projected[j]<=projected_btol[j],1:p)&&break
    end
    flags=BitVector(raw[j]<=btol[j]&&projected[j]<=projected_btol[j] for j in 1:p)
    all_converged=all(flags)
    if require_convergence&&!all_converged
        throw(ArgumentError(
            "block GMRES did not converge in $iterations block iterations; maximum residual=$(maximum(raw))"))
    end
    (solution=X,residuals=raw,projected_residuals=projected,
     converged=flags,iterations,restarts,operator_applications=applications,
     block_krylov_dimension=maxblocks,preconditioned=preconditioner!==nothing,
     workspace_reused=true)
end

"""
    block_gmres(A, B; initial_guess=nothing, workspace=nothing, ...)

Allocating wrapper for [`block_gmres!`](@ref). It returns the same diagnostic
named tuple, with the solution available as `result.solution`.
"""
function block_gmres(A,B::AbstractMatrix;initial_guess=nothing,
                     block_krylovdim::Integer=10,workspace=nothing,kwargs...)
    n,p=size(B);_advanced_krylov_dimension(A,n)
    inferred=initial_guess===nothing ? _advanced_krylov_type(A,B) :
        _advanced_krylov_type(A,B,initial_guess)
    T=workspace===nothing ? inferred : eltype(workspace.V)
    promote_type(T,inferred)===T||throw(ArgumentError(
        "block-GMRES workspace scalar type cannot represent the inputs"))
    X=initial_guess===nothing ? zeros(T,n,p) : T.(initial_guess)
    size(X)==(n,p)||throw(DimensionMismatch("initial_guess has the wrong size"))
    reused=workspace!==nothing
    ws=reused ? workspace : BlockGMRESWorkspace(T,n,p,block_krylovdim)
    merge(block_gmres!(X,A,B,ws;kwargs...),(workspace_reused=reused,))
end


# Shared-Arnoldi multi-shift GMRES -----------------------------------------

"""
    MultiShiftGMRESWorkspace(T, n, nshifts, krylovdim)
    MultiShiftGMRESWorkspace(A, nshifts, krylovdim)

Reusable Arnoldi and residual storage for [`multishift_gmres!`](@ref).
"""
mutable struct MultiShiftGMRESWorkspace{T}
    V::Matrix{T}
    H::Matrix{T}
    shifted_H::Matrix{T}
    w::Vector{T}
    rhs::Vector{T}
    residual::Vector{T}
    nshifts::Int
end

function MultiShiftGMRESWorkspace(::Type{T},n::Integer,nshifts::Integer,
                                  krylovdim::Integer) where T
    n>0||throw(ArgumentError("dimension must be positive"))
    nshifts>0||throw(ArgumentError("nshifts must be positive"))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    m=min(Int(n),Int(krylovdim))
    MultiShiftGMRESWorkspace(zeros(T,n,m+1),zeros(T,m+1,m),
        zeros(T,m+1,m),zeros(T,n),zeros(T,m+1),zeros(T,n),Int(nshifts))
end

function MultiShiftGMRESWorkspace(A,nshifts::Integer,krylovdim::Integer=30)
    n=size(A,1);size(A,2)==n||throw(DimensionMismatch("operator must be square"))
    S=_advanced_krylov_eltype(A);S===nothing&&throw(ArgumentError(
        "use MultiShiftGMRESWorkspace(T, n, nshifts, krylovdim) for a callable operator"))
    MultiShiftGMRESWorkspace(_complex_float_type(S),n,nshifts,krylovdim)
end

"""
    multishift_gmres!(X, A, b, shifts, workspace; ...)

Solve all systems `(A - shifts[j]*I) * X[:,j] = b` from one unrestarted
Arnoldi factorization. Each shifted projected problem is the GMRES
minimum-residual least-squares problem in the shared Krylov space.

Shift invariance requires a zero common initial guess and no generic left or
right preconditioner; nonzero `X` is therefore rejected. Increase
`krylovdim` when a requested shifted system is not converged. The returned
tuple reports actual full-space residuals and vector-application counts, and
failure raises unless `require_convergence=false`.
"""
function multishift_gmres!(X::AbstractMatrix,A,b::AbstractVector,shifts,
                           ws::MultiShiftGMRESWorkspace;atol::Real=1e-10,
                           rtol::Real=1e-8,preconditioner=nothing,
                           require_convergence::Bool=true)
    n=length(b);s=collect(shifts);ns=length(s)
    ns>0||throw(ArgumentError("at least one shift is required"))
    size(X)==(n,ns)||throw(DimensionMismatch("solution block has the wrong size"))
    Base.mightalias(X,b)&&throw(ArgumentError("X and b must not alias"))
    preconditioner===nothing||throw(ArgumentError(
        "generic preconditioning does not preserve the shared shifted Krylov space"))
    ws.nshifts==ns||throw(DimensionMismatch(
        "multi-shift workspace has the wrong shift count"))
    all(iszero,X)||throw(ArgumentError(
        "shared-Arnoldi shifted GMRES currently requires a zero initial guess"))
    _advanced_krylov_dimension(A,n)
    size(ws.V,1)==n&&size(ws.H,2)+1==size(ws.V,2)||throw(DimensionMismatch(
        "multi-shift workspace has the wrong dimension"))
    T=eltype(ws.V);promote_type(T,eltype(X),eltype(b))===T&&
        promote_type(eltype(X),T)===eltype(X)||throw(ArgumentError(
        "multi-shift workspace scalar type cannot represent vector inputs"))
    for shift in s
        isfinite(real(shift))&&isfinite(imag(shift))||throw(ArgumentError(
            "all shifts must be finite"))
        _advanced_promote_scalar_type(T,shift)===T||throw(ArgumentError(
            "multi-shift workspace scalar type cannot represent all shifts"))
    end
    _advanced_check_operator_precision(A,T)
    RT=typeof(real(zero(T)));atolT,rtolT=_advanced_check_tolerances(atol,rtol,RT)
    beta=_advanced_checked_norm(b,"multi-shift right-hand side")
    tol=atolT+rtolT*beta
    isfinite(tol)||throw(ArgumentError(
        "multi-shift residual tolerance overflowed; rescale the problem or use wider precision"))
    if iszero(beta)
        return (solutions=X,residuals=zeros(RT,ns),converged=trues(ns),
            iterations=0,operator_applications=0,krylov_dimension=size(ws.H,2),
            shared_arnoldi=true,workspace_reused=true)
    end
    V=ws.V;H=ws.H;fill!(H,zero(T));V[:,1].=b./beta
    m=size(H,2);k=0;has_extra=false
    for j in 1:m
        _advanced_krylov_apply!(ws.w,A,view(V,:,j))
        _advanced_checked_norm(ws.w,"multi-shift Arnoldi image")
        for pass in 1:2, i in 1:j
            alpha=dot(view(V,:,i),ws.w);H[i,j]+=alpha
            @views @. ws.w=ws.w-alpha*V[:,i]
        end
        H[j+1,j]=_advanced_checked_norm(ws.w,"multi-shift Arnoldi remainder");k=j
        # Only an exact invariant-subspace closure is a happy breakdown.
        # A small but representable remainder can still dominate a requested
        # tight residual tolerance and must therefore be retained.
        if iszero(H[j+1,j])||j==n
            has_extra=false;break
        end
        V[:,j+1].=ws.w./H[j+1,j];has_extra=true
    end
    rows=k+(has_extra ? 1 : 0);fill!(ws.rhs,zero(T));ws.rhs[1]=beta
    estimates=zeros(RT,ns)
    for ell in 1:ns
        S=view(ws.shifted_H,1:rows,1:k);copyto!(S,view(H,1:rows,1:k))
        shift=T(s[ell]);@inbounds for i in 1:k;S[i,i]-=shift;end
        y=_advanced_projected_least_squares(
            Matrix(S),Vector(view(ws.rhs,1:rows)))
        mul!(view(X,:,ell),view(V,:,1:k),y)
        estimates[ell]=norm(view(ws.rhs,1:rows)-S*y)
    end
    residuals=Vector{RT}(undef,ns);applications=k
    for ell in 1:ns
        _advanced_krylov_apply!(ws.residual,A,view(X,:,ell));applications+=1
        shift=T(s[ell])
        @views @. ws.residual=b-ws.residual+shift*X[:,ell]
        residuals[ell]=_advanced_checked_norm(ws.residual,
            "multi-shift full residual $ell")
    end
    flags=residuals .<= tol;all_converged=all(flags)
    if require_convergence&&!all_converged
        throw(ArgumentError(
            "shared-Arnoldi multi-shift GMRES did not converge with krylovdim=$m; maximum residual=$(maximum(residuals))"))
    end
    (solutions=X,residuals,projected_residuals=estimates,converged=flags,
     iterations=k,operator_applications=applications,krylov_dimension=m,
     shared_arnoldi=true,workspace_reused=true)
end

"""
    multishift_gmres(A, b, shifts; workspace=nothing, krylovdim=30, ...)

Allocating wrapper for [`multishift_gmres!`](@ref).
"""
function multishift_gmres(A,b::AbstractVector,shifts;workspace=nothing,
                          krylovdim::Integer=30,initial_guess=nothing,
                          preconditioner=nothing,kwargs...)
    s=collect(shifts);isempty(s)&&throw(ArgumentError("at least one shift is required"))
    inferred=initial_guess===nothing ? _advanced_krylov_type(A,b;scalars=s) :
        _advanced_krylov_type(A,b,initial_guess;scalars=s)
    T=workspace===nothing ? inferred : eltype(workspace.V)
    promote_type(T,inferred)===T||throw(ArgumentError(
        "multi-shift workspace scalar type cannot represent the inputs"))
    X=initial_guess===nothing ? zeros(T,length(b),length(s)) : T.(initial_guess)
    size(X)==(length(b),length(s))||throw(DimensionMismatch(
        "initial_guess has the wrong size"))
    reused=workspace!==nothing
    ws=reused ? workspace : MultiShiftGMRESWorkspace(T,length(b),length(s),krylovdim)
    merge(multishift_gmres!(X,A,b,s,ws;preconditioner,kwargs...),
        (workspace_reused=reused,))
end


# Recycled GMRES / GCRO -----------------------------------------------------

"""
    RecycledGMRESWorkspace(T, n, krylovdim, recycle_dim)
    RecycledGMRESWorkspace(A, krylovdim, recycle_dim)

Reusable GCRO-style projected-GMRES storage. The workspace retains up to
`recycle_dim` near-target Ritz directions after a solve and reuses them in
subsequent calls to [`recycled_gmres!`](@ref). It must not be shared between
tasks.
"""
mutable struct RecycledGMRESWorkspace{T}
    V::Matrix{T}
    H::Matrix{T}
    U::Matrix{T}
    C::Matrix{T}
    candidate::Matrix{T}
    AU::Matrix{T}
    coupling::Matrix{T}
    smallR::Matrix{T}
    w::Vector{T}
    image::Vector{T}
    residual::Vector{T}
    projected_residual::Vector{T}
    correction::Vector{T}
    nrecycle::Int
end

function RecycledGMRESWorkspace(::Type{T},n::Integer,krylovdim::Integer,
                                recycle_dim::Integer) where T
    n>0||throw(ArgumentError("dimension must be positive"))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    recycle_dim>=0||throw(ArgumentError("recycle_dim must be nonnegative"))
    m=min(Int(n),Int(krylovdim));k=min(Int(recycle_dim),max(Int(n)-1,0))
    RecycledGMRESWorkspace(zeros(T,n,m+1),zeros(T,m+1,m),zeros(T,n,k),
        zeros(T,n,k),zeros(T,n,k),zeros(T,n,k),zeros(T,k,m),zeros(T,k,k),
        zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n),zeros(T,n),0)
end

function RecycledGMRESWorkspace(A,krylovdim::Integer=30,recycle_dim::Integer=8)
    n=size(A,1);size(A,2)==n||throw(DimensionMismatch("operator must be square"))
    S=_advanced_krylov_eltype(A);S===nothing&&throw(ArgumentError(
        "use RecycledGMRESWorkspace(T, n, krylovdim, recycle_dim) for a callable operator"))
    RecycledGMRESWorkspace(_complex_float_type(S),n,krylovdim,recycle_dim)
end

function _recycled_apply!(dest,A,src,P,image)
    _advanced_krylov_apply!(image,A,src)
    _advanced_krylov_ldiv!(dest,P,image)
end

function _prepare_recycle!(ws,A,P)
    k=ws.nrecycle;k==0&&return (0,false,0)
    applications=0
    for j in 1:k
        _recycled_apply!(view(ws.AU,:,j),A,view(ws.U,:,j),P,ws.image)
        applications+=1
    end
    rank=_advanced_column_factor!(view(ws.C,:,1:k),view(ws.AU,:,1:k),
        view(ws.smallR,1:k,1:k))
    if rank<k
        ws.nrecycle=0
        return (0,true,applications)
    end
    copyto!(view(ws.candidate,:,1:k),view(ws.U,:,1:k))
    rdiv!(view(ws.candidate,:,1:k),UpperTriangular(view(ws.smallR,1:k,1:k)))
    copyto!(view(ws.U,:,1:k),view(ws.candidate,:,1:k))
    k,false,applications
end

function _update_recycle!(ws,k,target)
    capacity=size(ws.U,2);capacity==0&&return 0
    E=_projected_eigen(Matrix(view(ws.H,1:k,1:k)))
    keep=min(capacity,k);order=sortperm(abs.(E.values .- target))[1:keep]
    Y=view(E.vectors,:,order)
    mul!(view(ws.candidate,:,1:keep),view(ws.V,:,1:k),Y)
    old=ws.nrecycle
    if old>0
        coeff=view(ws.smallR,1:old,1:keep)
        mul!(coeff,view(ws.coupling,1:old,1:k),Y)
        mul!(view(ws.AU,:,1:keep),view(ws.U,:,1:old),coeff)
        view(ws.candidate,:,1:keep).-=
            view(ws.AU,:,1:keep)
    end
    copyto!(view(ws.AU,:,1:keep),view(ws.candidate,:,1:keep))
    rank=_advanced_column_factor!(view(ws.U,:,1:keep),view(ws.AU,:,1:keep),
        view(ws.smallR,1:keep,1:keep))
    ws.nrecycle=rank
    rank
end

"""
    recycled_gmres!(x, A, b, workspace; ...)

Solve `A*x=b` with a GCRO-style projected restarted GMRES method. On later
calls, the workspace rebuilds `C=A*U` for the current operator, projects the
residual onto that image, and solves in the complementary Krylov space. At
the end it retains near-`target` Ritz directions constructed from the GCRO
correction basis. This makes the workspace suitable for continuation through
a sequence of slowly varying operators.

A fixed left preconditioner is supported. Full unpreconditioned residuals are
always validated. The returned named tuple records whether recycling was
used, the retained dimension, residuals, restarts, iterations, and operator
applications. Failure raises unless `require_convergence=false`.
"""
function recycled_gmres!(x::AbstractVector,A,b::AbstractVector,
                         ws::RecycledGMRESWorkspace;atol::Real=1e-10,
                         rtol::Real=1e-8,maxiter::Integer=500,target=0,
                         preconditioner=nothing,require_convergence::Bool=true)
    n=length(b);length(x)==n||throw(DimensionMismatch(
        "solution and right-hand side have different lengths"))
    Base.mightalias(x,b)&&throw(ArgumentError("x and b must not alias"))
    _advanced_krylov_dimension(A,n);size(ws.V,1)==n||throw(DimensionMismatch(
        "recycled-GMRES workspace has the wrong dimension"))
    maxiter>0||throw(ArgumentError("maxiter must be positive"))
    _advanced_check_preconditioner_dimension(preconditioner,n)
    isfinite(real(target))&&isfinite(imag(target))||throw(ArgumentError(
        "recycling target must be finite"))
    T=eltype(ws.V);promote_type(T,eltype(x),eltype(b))===T&&
        promote_type(eltype(x),T)===eltype(x)&&
        _advanced_promote_scalar_type(T,target)===T||throw(ArgumentError(
        "recycled-GMRES workspace scalar type cannot represent the inputs"))
    _advanced_check_operator_precision(A,T)
    _advanced_check_preconditioner_precision(preconditioner,T)
    RT=typeof(real(zero(T)));atolT,rtolT=_advanced_check_tolerances(atol,rtol,RT)
    bnorm=_advanced_checked_norm(b,"recycled-GMRES right-hand side")
    rawtol=atolT+rtolT*bnorm
    isfinite(rawtol)||throw(ArgumentError(
        "recycled-GMRES residual tolerance overflowed; rescale the problem or use wider precision"))
    targetT=T(target)
    recycled_requested=ws.nrecycle;nr,recycle_reset,applications=
        _prepare_recycle!(ws,A,preconditioner)
    _advanced_krylov_apply!(ws.image,A,x);applications+=1
    @. ws.residual=b-ws.image
    _advanced_krylov_ldiv!(ws.projected_residual,preconditioner,ws.residual)
    if nr>0
        alpha=view(ws.smallR,1:nr,1)
        mul!(alpha,adjoint(view(ws.C,:,1:nr)),ws.projected_residual)
        mul!(ws.correction,view(ws.U,:,1:nr),alpha);x .+= ws.correction
        mul!(ws.correction,view(ws.C,:,1:nr),alpha)
        ws.projected_residual .-= ws.correction
        _advanced_krylov_apply!(ws.image,A,x);applications+=1
        @. ws.residual=b-ws.image
    end
    rawres=_advanced_checked_norm(ws.residual,"recycled-GMRES residual")
    pres=_advanced_checked_norm(ws.projected_residual,
        "recycled-GMRES projected residual")
    projected_bnorm=preconditioner===nothing ? bnorm :
        (_advanced_krylov_ldiv!(ws.w,preconditioner,b);
         _advanced_checked_norm(ws.w,"preconditioned right-hand side"))
    ptol=atolT+rtolT*projected_bnorm
    isfinite(ptol)||throw(ArgumentError(
        "recycled-GMRES projected tolerance overflowed; rescale the problem or use wider precision"))
    iterations=0;restarts=0;lastk=0
    m=size(ws.H,2);breakfactor=sqrt(eps(RT))
    while !(rawres<=rawtol&&pres<=ptol)&&iterations<maxiter
        fill!(ws.H,zero(T));fill!(ws.coupling,zero(T))
        beta=_advanced_checked_norm(ws.projected_residual,
            "recycled-GMRES projected residual")
        beta>zero(RT)||break
        ws.V[:,1].=ws.projected_residual./beta;k=0;has_extra=false
        for j in 1:min(m,maxiter-iterations)
            _recycled_apply!(ws.w,A,view(ws.V,:,j),preconditioner,ws.image)
            applications+=1
            image_norm=_advanced_checked_norm(ws.w,
                "recycled-GMRES Arnoldi image")
            if nr>0
                for i in 1:nr
                    alpha=dot(view(ws.C,:,i),ws.w);ws.coupling[i,j]=alpha
                    @views @. ws.w=ws.w-alpha*ws.C[:,i]
                end
            end
            for pass in 1:2, i in 1:j
                alpha=dot(view(ws.V,:,i),ws.w);ws.H[i,j]+=alpha
                @views @. ws.w=ws.w-alpha*ws.V[:,i]
            end
            ws.H[j+1,j]=_advanced_checked_norm(ws.w,
                "recycled-GMRES Arnoldi remainder");k=j;iterations+=1
            if abs(ws.H[j+1,j])<=breakfactor*max(image_norm,floatmin(RT))||j==n
                has_extra=false;break
            end
            ws.V[:,j+1].=ws.w./ws.H[j+1,j];has_extra=true
        end
        lastk=k;rows=k+(has_extra ? 1 : 0)
        g=zeros(T,rows);g[1]=beta
        y=_advanced_projected_least_squares(
            Matrix(view(ws.H,1:rows,1:k)),g)
        mul!(ws.correction,view(ws.V,:,1:k),y);x .+= ws.correction
        if nr>0
            alpha=view(ws.smallR,1:nr,1)
            mul!(alpha,view(ws.coupling,1:nr,1:k),y)
            mul!(ws.correction,view(ws.U,:,1:nr),alpha);x .-= ws.correction
        end
        restarts+=1
        _advanced_krylov_apply!(ws.image,A,x);applications+=1
        @. ws.residual=b-ws.image
        _advanced_krylov_ldiv!(ws.projected_residual,preconditioner,ws.residual)
        if nr>0
            alpha=view(ws.smallR,1:nr,1)
            mul!(alpha,adjoint(view(ws.C,:,1:nr)),ws.projected_residual)
            mul!(ws.correction,view(ws.U,:,1:nr),alpha);x .+= ws.correction
            mul!(ws.correction,view(ws.C,:,1:nr),alpha)
            ws.projected_residual .-= ws.correction
            # Refresh the unpreconditioned residual after the coarse correction.
            _advanced_krylov_apply!(ws.image,A,x);applications+=1
            @. ws.residual=b-ws.image
        end
        rawres=_advanced_checked_norm(ws.residual,"recycled-GMRES residual")
        pres=_advanced_checked_norm(ws.projected_residual,
            "recycled-GMRES projected residual")
    end
    lastk>0&&_update_recycle!(ws,lastk,targetT)
    converged=rawres<=rawtol&&pres<=ptol
    if require_convergence&&!converged
        throw(ArgumentError(
            "recycled GMRES did not converge in $iterations iterations; residual=$rawres"))
    end
    (solution=x,residual=rawres,projected_residual=pres,converged,
     iterations,restarts,operator_applications=applications,
     recycled_initially=recycled_requested>0,recycle_reset,
     recycle_dimension=ws.nrecycle,krylov_dimension=m,target=targetT,
     preconditioned=preconditioner!==nothing,workspace_reused=true)
end

"""
    recycled_gmres(A, b; initial_guess=nothing, workspace=nothing,
                   krylovdim=30, recycle_dim=8, ...)

Allocating wrapper for [`recycled_gmres!`](@ref). Reuse the returned call's
workspace explicitly on later systems to obtain recycling; the result itself
contains only the solution and diagnostics.
"""
function recycled_gmres(A,b::AbstractVector;initial_guess=nothing,
                        workspace=nothing,krylovdim::Integer=30,
                        recycle_dim::Integer=8,kwargs...)
    target=get(kwargs,:target,0)
    inferred=initial_guess===nothing ? _advanced_krylov_type(A,b;scalars=(target,)) :
        _advanced_krylov_type(A,b,initial_guess;scalars=(target,))
    T=workspace===nothing ? inferred : eltype(workspace.V)
    promote_type(T,inferred)===T||throw(ArgumentError(
        "recycled-GMRES workspace scalar type cannot represent the inputs"))
    x=initial_guess===nothing ? zeros(T,length(b)) : T.(initial_guess)
    length(x)==length(b)||throw(DimensionMismatch("initial_guess has the wrong length"))
    reused=workspace!==nothing
    ws=reused ? workspace : RecycledGMRESWorkspace(T,length(b),krylovdim,recycle_dim)
    merge(recycled_gmres!(x,A,b,ws;kwargs...),(workspace_reused=reused,))
end


# Adaptive exponential action ----------------------------------------------

"""
    KrylovExpvWorkspace(T, n, krylovdim)
    KrylovExpvWorkspace(A, krylovdim)

Reusable Arnoldi and vector storage for [`krylov_expv!`](@ref).
"""
mutable struct KrylovExpvWorkspace{T}
    V::Matrix{T}
    H::Matrix{T}
    small::Matrix{T}
    w::Vector{T}
    current::Vector{T}
    trial::Vector{T}
end

function KrylovExpvWorkspace(::Type{T},n::Integer,krylovdim::Integer) where T
    n>0||throw(ArgumentError("dimension must be positive"))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    m=min(Int(n),Int(krylovdim))
    KrylovExpvWorkspace(zeros(T,n,m+1),zeros(T,m+1,m),zeros(T,m+1,m+1),
        zeros(T,n),zeros(T,n),zeros(T,n))
end

function KrylovExpvWorkspace(A,krylovdim::Integer=30)
    n=size(A,1);size(A,2)==n||throw(DimensionMismatch("operator must be square"))
    S=_advanced_krylov_eltype(A);S===nothing&&throw(ArgumentError(
        "use KrylovExpvWorkspace(T, n, krylovdim) for a callable operator"))
    KrylovExpvWorkspace(_complex_float_type(S),n,krylovdim)
end

function _expv_trial!(ws,A,h)
    T=eltype(ws.V);RT=typeof(real(zero(T)))
    beta=_advanced_checked_norm(ws.current,"exponential-action state")
    iszero(beta)&&(fill!(ws.trial,zero(T));return (zero(RT),0,0,true))
    V=ws.V;H=ws.H;fill!(H,zero(T));V[:,1].=ws.current./beta
    m=size(H,2);k=0;exact_breakdown=false
    near_breakdown=false;breakfactor=sqrt(eps(RT))
    for j in 1:m
        _advanced_krylov_apply!(ws.w,A,view(V,:,j))
        image_norm=_advanced_checked_norm(ws.w,"exponential-action Arnoldi image")
        for pass in 1:2, i in 1:j
            alpha=dot(view(V,:,i),ws.w);H[i,j]+=alpha
            @views @. ws.w=ws.w-alpha*V[:,i]
        end
        H[j+1,j]=_advanced_checked_norm(ws.w,
            "exponential-action Arnoldi remainder");k=j
        if iszero(H[j+1,j])||j==size(V,1)
            exact_breakdown=true;break
        end
        near_breakdown|=abs(H[j+1,j])<=
            breakfactor*max(image_norm,floatmin(RT))
        # A small representable remainder is not a happy breakdown. Keep its
        # direction so a later Arnoldi step can either verify an invariant
        # subspace or expose the remaining defect.
        V[:,j+1].=ws.w./H[j+1,j]
    end
    q=k+(exact_breakdown ? 0 : 1)
    S=view(ws.small,1:q,1:q);fill!(S,zero(T))
    view(S,1:k,1:k).=view(H,1:k,1:k)
    exact_breakdown||(S[k+1,k]=H[k+1,k])

    # If the last retained remainder was close to breakdown, use its
    # normalized direction in the approximation and spend one extra matvec
    # to estimate the defect of that enlarged space. This avoids declaring a
    # small but physically relevant component either exact or impossible to
    # converge under time-step refinement.
    used=k;enlarged=false;next_remainder=zero(RT)
    if !exact_breakdown&&near_breakdown
        enlarged=true;used+=1
        _advanced_krylov_apply!(ws.w,A,view(V,:,q))
        for pass in 1:2, i in 1:q
            alpha=dot(view(V,:,i),ws.w);S[i,q]+=alpha
            @views @. ws.w=ws.w-alpha*V[:,i]
        end
        next_remainder=_advanced_checked_norm(ws.w,
            "exponential-action enlarged Arnoldi remainder")
    end

    E=exp(h*Matrix(S));coeff_count=enlarged ? q : k
    coeff=view(E,1:coeff_count,1)
    mul!(ws.trial,view(V,:,1:coeff_count),coeff);ws.trial .*= beta
    completed=exact_breakdown||(enlarged&&
        (q==size(V,1)||iszero(next_remainder)))
    error=if completed
        zero(RT)
    elseif enlarged
        RT(beta*abs(h)*next_remainder*abs(E[q,1]))
    else
        RT(beta*abs(E[k+1,1]))
    end
    error,k,used,completed
end

"""
    krylov_expv!(y, A, b, t, workspace; ...)

Compute `y = exp(t*A)*b` without materializing `A` or its exponential.
Restarted Arnoldi time slices are accepted or rejected using the augmented
Hessenberg defect estimate. `initial_step`, `minimum_step`, and `maximum_step`
are expressed in the same units as the finite real time `t`.

The result reports accepted/rejected steps, total matrix-free operator
applications, the accumulated local error estimate, and `reached_time`.
Exhausting `max_steps` or the minimum step raises unless
`require_convergence=false`, in which case `converged=false` and
`reached_time` make the partial result explicit.
"""
function krylov_expv!(y::AbstractVector,A,b::AbstractVector,t::Real,
                      ws::KrylovExpvWorkspace;atol::Real=1e-10,
                      rtol::Real=1e-8,initial_step=nothing,
                      minimum_step=nothing,maximum_step=nothing,
                      max_steps::Integer=10_000,safety::Real=0.9,
                      require_convergence::Bool=true)
    n=length(b);length(y)==n||throw(DimensionMismatch(
        "destination and source have different lengths"))
    _advanced_krylov_dimension(A,n);size(ws.V,1)==n||throw(DimensionMismatch(
        "exponential-action workspace has the wrong dimension"))
    isfinite(t)||throw(ArgumentError("time must be finite"))
    max_steps>0||throw(ArgumentError("max_steps must be positive"))
    safety isa Real&&isfinite(safety)&&0<safety<1||throw(ArgumentError(
        "safety must lie strictly between zero and one"))
    T=eltype(ws.V);promote_type(T,eltype(y),eltype(b))===T&&
        promote_type(eltype(y),T)===eltype(y)&&
        _advanced_promote_scalar_type(T,t)===T||throw(ArgumentError(
        "exponential-action workspace scalar type cannot represent the inputs"))
    _advanced_check_operator_precision(A,T)
    RT=typeof(real(zero(T)));atolT,rtolT=_advanced_check_tolerances(atol,rtol,RT)
    safetyT=RT(safety)
    zero(RT)<safetyT<one(RT)||throw(ArgumentError(
        "safety rounds outside (0,1) in exponential-action precision $RT"))
    time=RT(t);total=abs(time)
    if iszero(total)
        copyto!(y,b)
        return (value=y,converged=true,reached_time=time,accepted_steps=0,
            rejected_steps=0,operator_applications=0,estimated_error=zero(RT),
            krylov_dimension=size(ws.H,2),workspace_reused=true)
    end
    hmax=maximum_step===nothing ? total :
        _advanced_expv_step_control(RT,maximum_step,"maximum_step";
            allow_zero=false)
    hmin=minimum_step===nothing ? zero(RT) :
        _advanced_expv_step_control(RT,minimum_step,"minimum_step";
            allow_zero=true)
    h0=initial_step===nothing ? hmax :
        _advanced_expv_step_control(RT,initial_step,"initial_step";
            allow_zero=false)
    isfinite(hmax)&&hmax>0||throw(ArgumentError("maximum_step must be finite and positive"))
    isfinite(hmin)&&hmin>=0||throw(ArgumentError("minimum_step must be finite and nonnegative"))
    isfinite(h0)&&h0>0||throw(ArgumentError("initial_step must be finite and positive"))
    hmax=min(hmax,total);h=min(h0,hmax);copyto!(ws.current,b)
    hmin<=hmax||throw(ArgumentError(
        "minimum_step must not exceed the effective maximum step"))
    h0>=hmin||throw(ArgumentError(
        "initial_step must not be smaller than minimum_step"))
    reached=zero(RT);accepted=0;rejected=0;applications=0;error_sum=zero(RT)
    converged=false;attempts=0;direction=sign(time)
    while reached<total&&attempts<max_steps
        attempts+=1;h=min(h,total-reached)
        error,k,used,breakdown=_expv_trial!(ws,A,direction*h);applications+=used
        finite_trial=all(z->isfinite(real(z))&&isfinite(imag(z)),ws.trial)&&
            isfinite(error)
        trial_norm=finite_trial ? _advanced_checked_norm(ws.trial,
            "exponential-action trial state") : _advanced_checked_norm(
            ws.current,"exponential-action state")
        fraction=h/total;localtol=fraction*(atolT+rtolT*trial_norm)
        isfinite(localtol)||throw(ArgumentError(
            "exponential-action local tolerance overflowed; rescale the state or use wider precision"))
        accept=finite_trial&&(breakdown||error<=localtol)
        if accept
            copyto!(ws.current,ws.trial)
            remaining=total-reached
            next_reached=h==remaining ? total : reached+h
            next_reached>reached||throw(ArgumentError(
                "exponential-action time step no longer advances in $RT; use fewer steps or wider precision"))
            reached=next_reached;accepted+=1;error_sum+=error
            isfinite(error_sum)||throw(ArgumentError(
                "exponential-action accumulated defect overflowed; rescale the state or use wider precision"))
        else
            rejected+=1
        end
        if !finite_trial
            factor=RT(0.2)
        elseif breakdown
            factor=RT(5)
        elseif iszero(error)
            factor=RT(2)
        else
            exponent=inv(RT(max(k,1)))
            factor=safetyT*(localtol/error)^exponent
            factor=clamp(factor,RT(0.2),RT(5))
        end
        !accept&&(factor=min(factor,RT(0.5)))
        hnew=min(hmax,h*factor)
        if !accept&&(hnew<hmin||iszero(hnew)||
                (reached>zero(RT)&&reached+hnew==reached))
            break
        end
        h=max(hnew,hmin)
    end
    converged=reached==total;copyto!(y,ws.current)
    if require_convergence&&!converged
        throw(ArgumentError(
            "adaptive Krylov exponential action reached time $(direction*reached) of $time after $attempts attempts"))
    end
    (value=y,converged,reached_time=direction*reached,accepted_steps=accepted,
     rejected_steps=rejected,operator_applications=applications,
     estimated_error=error_sum,krylov_dimension=size(ws.H,2),
     workspace_reused=true)
end

"""
    krylov_expv(A, b, t; workspace=nothing, krylovdim=30, ...)

Allocating wrapper for [`krylov_expv!`](@ref).
"""
function krylov_expv(A,b::AbstractVector,t::Real;workspace=nothing,
                     krylovdim::Integer=30,kwargs...)
    inferred=_advanced_krylov_type(A,b;scalars=(t,))
    T=workspace===nothing ? inferred : eltype(workspace.V)
    promote_type(T,inferred)===T||throw(ArgumentError(
        "exponential-action workspace scalar type cannot represent the inputs"))
    y=zeros(T,length(b))
    reused=workspace!==nothing
    ws=reused ? workspace : KrylovExpvWorkspace(T,length(b),krylovdim)
    merge(krylov_expv!(y,A,b,t,ws;kwargs...),(workspace_reused=reused,))
end
