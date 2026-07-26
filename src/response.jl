_dense_liouvillian(L::AbstractMatrix)=Matrix(L)
_dense_liouvillian(L)=Matrix(_materialize(L))

"""
    ResponseWorkspace(source; krylovdim=30, expv_krylovdim=30,
                      mode=:both, memory_budget=512*1024^2)

Reusable matrix-free response scratch. `mode=:linear` retains two restarted-
GMRES workspaces and response vectors, `mode=:evolution` retains only adaptive
Krylov exponential-action storage, and `mode=:both` supports every response
route. A workspace is tied to its prepared source and may be reused
sequentially, never concurrently. A [`RestrictedLiouvillian`](@ref) with the
`:lowered` backend retains only its reduced Schur-block application scratch;
the response layer does not recreate ambient PI vectors.
Construction preflights Krylov, response-vector, and matrix-free action
scratch; `memory_budget=Inf` is the explicit opt-out.
"""
struct ResponseWorkspace{S,K,A,E,V,W}
    source::S
    forward::K
    adjoint::A
    exponential::E
    x::V
    y::V
    z::V
    rhs::V
    action_workspace::W
    mode::Symbol
end

_response_source(source::PIModel)=compile(source;backend=:matrixfree)
_response_source(source)=source
_response_source(source::PIModel,memory_budget)=
    compile(source;backend=:matrixfree,memory_budget)
_response_source(source,memory_budget)=source

function _response_source_preflight(source,memory_budget)
    _performance_memory_limit(memory_budget)
    source isa PIModel&&_require_model_preparation_budget(source,memory_budget;
        operation="matrix-free response model preparation")
    nothing
end

_response_action_budget(source,::Type{T}) where T=
    _performance_linear_operator_workspace_bytes(source)+
    _performance_source_action_bytes(source,T)
_response_retained_source_bytes(source,::Type{T}) where T=
    _performance_source_action_bytes(source,T)

function _response_tolerance(::Type{R},value,label::AbstractString;
        positive::Bool=false) where R<:AbstractFloat
    value isa Real&&isfinite(value)||throw(ArgumentError(
        "$label must be a finite real number"))
    valid=positive ? value>0 : value>=0
    valid||throw(ArgumentError(
        "$label must be $(positive ? "positive" : "nonnegative")"))
    if value isa Integer
        converted=R(value)
        isfinite(converted)&&try
            BigInt(converted)==BigInt(value)
        catch
            false
        end||throw(ArgumentError(
            "$label is not exactly representable in workspace precision $R"))
        return converted
    end
    promote_type(R,typeof(value))===R||throw(ArgumentError(
        "$label scalar type $(typeof(value)) would narrow in workspace precision $R"))
    converted=R(value)
    isfinite(converted)||throw(ArgumentError(
        "$label is not finite in workspace precision $R"))
    !iszero(value)&&iszero(converted)&&throw(ArgumentError(
        "$label underflows in workspace precision $R; use wider precision"))
    converted
end

function _response_action_workspace(source)
    source isa FloquetMap&&return(FloquetWorkspace(source))
    _linear_operator_workspace(source)
end

function ResponseWorkspace(source0;krylovdim::Integer=30,
        expv_krylovdim::Integer=krylovdim,mode::Symbol=:both,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    mode in (:both,:linear,:evolution)||throw(ArgumentError(
        "response workspace mode must be :both, :linear, or :evolution"))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    expv_krylovdim>0||throw(ArgumentError(
        "expv_krylovdim must be positive"))
    BigInt(krylovdim)<=typemax(Int)||throw(ArgumentError(
        "krylovdim must be representable as an Int"))
    BigInt(expv_krylovdim)<=typemax(Int)||throw(ArgumentError(
        "expv_krylovdim must be representable as an Int"))
    _response_source_preflight(source0,memory_budget)
    source=_response_source(source0,memory_budget);n=size(source,1)
    size(source,2)==n||throw(DimensionMismatch(
        "response source must be square"))
    T=_complex_float_type(eltype(source))
    expm=min(n,Int(expv_krylovdim))
    exponential_entries=BigInt(n)*(expm+4)+BigInt(expm+1)*expm+
                        BigInt(expm+1)^2
    estimate=(mode in (:both,:linear) ?
        2*_performance_gmres_bytes(n,T,krylovdim)+
        _performance_array_bytes(n,T,0;linear_arrays=4) : big(0))+
        (mode in (:both,:evolution) ?
        _performance_entries_bytes(exponential_entries,T) : big(0))+
        _response_action_budget(source,T)
    _require_performance_budget("response workspace construction",estimate,
        memory_budget;guidance="Reduce Krylov dimensions or increase the budget.")
    linear=mode in (:both,:linear)
    evolution=mode in (:both,:evolution)
    forward=linear ? KrylovWorkspace(T,n,krylovdim) : nothing
    adjoint_work=linear ? KrylovWorkspace(T,n,krylovdim) : nothing
    exponential=evolution ? KrylovExpvWorkspace(T,n,expv_krylovdim) : nothing
    vector=linear ? zeros(T,n) : nothing
    ResponseWorkspace(source,forward,adjoint_work,exponential,vector,
        linear ? similar(vector) : nothing,linear ? similar(vector) : nothing,
        linear ? similar(vector) : nothing,
        _response_action_workspace(source),mode)
end

function _prepared_response_source(source,workspace,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    workspace===nothing&&return _response_source(source,memory_budget)
    workspace isa ResponseWorkspace||return _response_source(source,memory_budget)
    if workspace.source===source
        return source
    elseif source isa PIModel&&workspace.source isa CompiledPIModel&&
           workspace.source.model===source
        return workspace.source
    end
    throw(ArgumentError("response workspace belongs to a different source"))
end

function _check_response_workspace(work::ResponseWorkspace,source;
        linear::Bool=false,evolution::Bool=false)
    work.source===source||throw(ArgumentError(
        "response workspace belongs to a different prepared source"))
    linear&&work.mode===:evolution&&throw(ArgumentError(
        "this response operation requires mode=:linear or :both"))
    evolution&&work.mode===:linear&&throw(ArgumentError(
        "this response operation requires mode=:evolution or :both"))
    n=size(source,1);size(source,2)==n||throw(DimensionMismatch(
        "response source must be square"))
    T=_complex_float_type(eltype(source))
    if linear
        all(vector->vector!==nothing&&length(vector)==n&&eltype(vector)===T,
            (work.x,work.y,work.z,work.rhs))||throw(DimensionMismatch(
            "response linear workspace has incompatible vectors"))
        for (label,krylov) in (("forward",work.forward),
                              ("adjoint",work.adjoint))
            krylov isa KrylovWorkspace||throw(ArgumentError(
                "response $label Krylov workspace is missing or invalid"))
            m=size(krylov.H,2)
            size(krylov.V)==(n,m+1)&&size(krylov.H)==(m+1,m)&&
                length(krylov.cs)==m&&length(krylov.sn)==m&&
                length(krylov.g)==m+1&&length(krylov.y)>=m&&
                all(vector->length(vector)==n,
                    (krylov.w,krylov.r,krylov.z,krylov.p))||
                throw(DimensionMismatch(
                    "response $label Krylov workspace has incompatible dimensions"))
            all(array->eltype(array)===T,
                (krylov.V,krylov.H,krylov.sn,krylov.g,krylov.w,
                 krylov.r,krylov.z,krylov.p,krylov.y))||
                throw(ArgumentError(
                    "response $label Krylov workspace has an incompatible scalar type"))
        end
    end
    if evolution
        exponential=work.exponential
        exponential isa KrylovExpvWorkspace||throw(ArgumentError(
            "response exponential workspace is missing or invalid"))
        m=size(exponential.H,2)
        size(exponential.V)==(n,m+1)&&size(exponential.H)==(m+1,m)&&
            size(exponential.small)==(m+1,m+1)&&
            all(vector->length(vector)==n,
                (exponential.w,exponential.current,exponential.trial))||
            throw(DimensionMismatch(
                "response exponential workspace has incompatible dimensions"))
        all(array->eltype(array)===T,
            (exponential.V,exponential.H,exponential.small,
             exponential.w,exponential.current,exponential.trial))||
            throw(ArgumentError(
                "response exponential workspace has an incompatible scalar type"))
    end
    if source isa FloquetMap
        work.action_workspace isa FloquetWorkspace||throw(ArgumentError(
            "response workspace is missing Floquet action scratch"))
        _check_floquet_workspace(work.action_workspace,source)
    elseif source isa RestrictedLiouvillian
        if source.backend===:compressed
            work.action_workspace===nothing||throw(ArgumentError(
                "a compressed restricted response source does not use action scratch"))
        elseif source.backend===:lowered
            work.action_workspace isa _RestrictedKernelWorkspace||
                throw(ArgumentError(
                    "response workspace is missing reduced restricted action scratch"))
            _check_restricted_kernel_workspace(
                work.action_workspace,source.compressed_source.plan)
        else
            work.action_workspace isa RestrictedLiouvillianWorkspace||
                throw(ArgumentError(
                    "response workspace is missing ambient restricted action scratch"))
            _check_restricted_workspace(work.action_workspace,source.source,
                                        source.restriction)
        end
    end
    work
end

function _check_response_workspace(work,source;kwargs...)
    throw(ArgumentError("workspace must be a ResponseWorkspace"))
end

function _response_apply!(destination,source,input,work)
    if source isa AbstractMatrix
        mul!(destination,source,input)
    elseif source isa FloquetMap
        apply!(destination,source,input,work.action_workspace)
    elseif source isa RestrictedLiouvillian
        work.action_workspace===nothing ?
            mul!(destination,source,input) :
            apply!(destination,source,input,
                zero(_real_float_type(eltype(source))),nothing,
                work.action_workspace)
    elseif work.action_workspace===nothing
        apply!(destination,source,input)
    else
        apply!(destination,source,input,
               zero(_real_float_type(eltype(source))),nothing,
               work.action_workspace)
    end
end

function _response_apply_adjoint!(destination,source,input,work)
    if source isa AbstractMatrix
        mul!(destination,adjoint(source),input)
    elseif source isa FloquetMap
        apply_adjoint!(destination,source,input,work.action_workspace)
    elseif source isa RestrictedLiouvillian
        work.action_workspace===nothing ?
            mul!(destination,adjoint(source),input) :
            apply_adjoint!(destination,source,input,
                zero(_real_float_type(eltype(source))),nothing,
                work.action_workspace)
    elseif work.action_workspace===nothing
        apply_adjoint!(destination,source,input)
    else
        apply_adjoint!(destination,source,input,
                       zero(_real_float_type(eltype(source))),nothing,
                       work.action_workspace)
    end
end

struct _ResponseActionOperator{T,S,W}
    source::S
    workspace::W
    adjoint_action::Bool
end
Base.size(operator::_ResponseActionOperator)=size(operator.source)
Base.size(operator::_ResponseActionOperator,index::Integer)=
    index in (1,2) ? size(operator.source,index) : 1
Base.eltype(::_ResponseActionOperator{T}) where T=T
function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_ResponseActionOperator,input::AbstractVector)
    operator.adjoint_action ?
        _response_apply_adjoint!(destination,operator.source,input,
                                 operator.workspace) :
        _response_apply!(destination,operator.source,input,operator.workspace)
end
(operator::_ResponseActionOperator)(destination,input)=
    mul!(destination,operator,input)

struct _ResponseShiftedOperator{T,S,W,Z}
    source::S
    workspace::W
    shift::Z
    adjoint_action::Bool
end
Base.size(operator::_ResponseShiftedOperator)=size(operator.source)
Base.size(operator::_ResponseShiftedOperator,index::Integer)=
    index in (1,2) ? size(operator.source,index) : 1
Base.eltype(::_ResponseShiftedOperator{T}) where T=T
function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_ResponseShiftedOperator,input::AbstractVector)
    if operator.adjoint_action
        _response_apply_adjoint!(destination,operator.source,input,
                                 operator.workspace)
        @. destination=conj(operator.shift)*input-destination
    else
        _response_apply!(destination,operator.source,input,operator.workspace)
        @. destination=operator.shift*input-destination
    end
    destination
end
(operator::_ResponseShiftedOperator)(destination,input)=
    mul!(destination,operator,input)

function _response_shifted_solve!(solution,source,rhs,shift,work;
        adjoint_action::Bool=false,atol,rtol,maxiter)
    n=size(source,1)
    length(solution)==n&&length(rhs)==n||throw(DimensionMismatch(
        "shifted response vectors have incompatible dimensions"))
    eltype(solution)===eltype(work.x)&&
        promote_type(eltype(solution),eltype(rhs))===eltype(solution)||
        throw(ArgumentError(
            "shifted response vectors have incompatible scalar types"))
    maxiter>0||throw(ArgumentError("maxiter must be positive"))
    BigInt(maxiter)<=typemax(Int)||throw(ArgumentError(
        "maxiter must be representable as an Int"))
    fill!(solution,zero(eltype(solution)))
    operator=_ResponseShiftedOperator{eltype(solution),typeof(source),
        typeof(work),typeof(shift)}(source,work,shift,adjoint_action)
    krylov=adjoint_action ? work.adjoint : work.forward
    result=_gmres!(solution,operator,rhs,krylov;
        atol,rtol,maxiter)
    result.converged||throw(ArgumentError(
        "shifted response GMRES did not converge in $(result.iterations) iterations; raw residual=$(result.raw_residual)"))
    result
end

function _response_backend_error(operation::AbstractString,A,error)
    error isa MethodError||rethrow(error)
    R=_real_float_type(eltype(A))
    throw(ArgumentError("$operation is unavailable for scalar type $R with Julia's active LinearAlgebra backend; use Float32/Float64 data or load a compatible generic linear-algebra backend"))
end

function _response_eigen(A;operation::AbstractString)
    try
        eigen(A)
    catch error
        _response_backend_error(operation,A,error)
    end
end

function _response_svdvals(A;operation::AbstractString)
    try
        svdvals(A)
    catch error
        _response_backend_error(operation,A,error)
    end
end

function _response_pinv(A;operation::AbstractString,kwargs...)
    try
        pinv(A;kwargs...)
    catch error
        _response_backend_error(operation,A,error)
    end
end

function _response_matrix_exp(A;operation::AbstractString)
    try
        exp(A)
    catch error
        _response_backend_error(operation,A,error)
    end
end

function _response_scalar_type(x)
    T=try
        eltype(x)
    catch
        return nothing
    end
    T isa Type&&T<:Number ? T : nothing
end

function _response_real_type(seed::Type,args...)
    R=_real_float_type(seed)
    for x in args
        T=_response_scalar_type(x)
        T===nothing||(R=promote_type(R,_real_float_type(T)))
    end
    R
end

function _response_dense_bytes(source,square_arrays::Integer,inputs...;
        linear_arrays::Integer=8)
    n=size(source,1);size(source,2)==n||throw(DimensionMismatch(
        "response operator must be square"))
    T=promote_type(_complex_float_type(eltype(source)),ComplexF64)
    for input in inputs
        S=_response_scalar_type(input)
        S===nothing||
            (T=promote_type(T,_complex_float_type(S)))
    end
    _performance_array_bytes(n,T,square_arrays;linear_arrays)
end

function _response_dense_choice(source,method,memory_budget,estimate,
        iterative::Symbol,operation::AbstractString)
    if method===:auto
        return source isa AbstractMatrix&&
               _performance_budget_fits(estimate,memory_budget) ? :dense :
               iterative
    end
    method===:dense&&_require_performance_budget(operation,estimate,
        memory_budget;guidance=
        "Use method=$iterative for bounded matrix-free work.")
    method
end

function _response_linear_workspace_bytes(source,krylovdim,workspace)
    n=size(source,1)
    if workspace isa ResponseWorkspace&&workspace.forward!==nothing
        T=eltype(workspace.forward.V)
        m=size(workspace.forward.V,2)-1
    else
        T=_complex_float_type(eltype(source));m=min(n,Int(krylovdim))
    end
    action=workspace isa ResponseWorkspace&&
           workspace.action_workspace!==nothing ?
        BigInt(Base.summarysize(workspace.action_workspace))+
        _response_retained_source_bytes(source,T) :
        _response_action_budget(source,T)
    2*_performance_gmres_bytes(n,T,m)+
        _performance_array_bytes(n,T,0;linear_arrays=4)+action
end

function _response_evolution_workspace_bytes(source,krylovdim,workspace)
    n=size(source,1)
    if workspace isa ResponseWorkspace&&workspace.exponential!==nothing
        expv=workspace.exponential;T=eltype(expv.V);m=size(expv.H,2)
    else
        T=_complex_float_type(eltype(source));m=min(n,Int(krylovdim))
    end
    entries=BigInt(n)*(m+4)+BigInt(m+1)*m+BigInt(m+1)^2
    action=workspace isa ResponseWorkspace&&
           workspace.action_workspace!==nothing ?
        BigInt(Base.summarysize(workspace.action_workspace))+
        _response_retained_source_bytes(source,T) :
        _response_action_budget(source,T)
    _performance_entries_bytes(entries,T)+action
end

"""
    liouvillian_modes(L; k=6, which=:largest_real, method=:auto, ...)

Return selected Liouvillian eigenvalues and right decay modes. Explicit
matrices retain the dense compatibility path under `method=:auto` only when
its work-array estimate fits `memory_budget`; otherwise ordinary Arnoldi is
selected and its workspace is checked against the same budget. `method=:iram` and `:jd` expose
the corresponding bounded-memory solvers. Iterative results retain residuals,
convergence flags, and partial spectral scope.

Ordinary Arnoldi advances one initial vector, so an exact degenerate
eigenspace may contribute only one Ritz direction. Use
`pi_liouvillian_spectrum(...; method=:block_arnoldi)` when resolving spectral
multiplicities is required.
"""
function liouvillian_modes(L0;k::Integer=6,which=:largest_real,
        method::Symbol=:auto,target=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    _response_source_preflight(L0,memory_budget)
    L=_response_source(L0,memory_budget)
    _require_autonomous(L,"Liouvillian mode analysis")
    n=size(L,1);k>0||throw(ArgumentError("k must be positive"))
    requested_count=Int(min(BigInt(k),BigInt(n)))
    selection=which===:largest_real ? :LR :
        which===:smallest_magnitude ? :SM : which
    selection in (:LR,:LM,:SM)||throw(ArgumentError(
        "which must be :largest_real, :smallest_magnitude, :LR, :LM, or :SM"))
    dense_estimate=_response_dense_bytes(L,7)
    chosen=_response_dense_choice(L,method,memory_budget,dense_estimate,
        :arnoldi,"dense Liouvillian mode analysis")
    chosen in (:dense,:arnoldi,:iram,:jd)||throw(ArgumentError(
        "mode method must be :auto, :dense, :arnoldi, :iram, or :jd"))
    if chosen===:dense
        M=_dense_liouvillian(L)
        E=_response_eigen(M;operation="Liouvillian mode analysis")
        order=_ritz_order(E.values,selection)
        idx=order[1:min(requested_count,length(order))]
        values=E.values[idx];vectors=E.vectors[:,idx]
        R=_real_float_type(eltype(values));residuals=R[
            norm(M*view(vectors,:,column)-values[column]*view(vectors,:,column))
            for column in axes(vectors,2)]
        return (;values,vectors,residuals,converged=trues(length(values)),
            method=:dense,dimension=n,partial_scope=length(values)<n,
            selection)
    end
    solver_krylovdim=Int(chosen===:jd ?
        get(kwargs,:subspace_dim,max(30,3requested_count+6)) :
        get(kwargs,:krylovdim,max(20,2requested_count+4)))
    workspace_estimate=_selected_spectrum_workspace_bytes(L,chosen,
        solver_krylovdim,requested_count;vectors=true,target,kwargs...)
    _require_performance_budget("selected Liouvillian mode workspace",
        workspace_estimate,memory_budget;guidance=
        "Reduce k/krylovdim or increase the budget.")
    result=if chosen===:arnoldi
        target===nothing||throw(ArgumentError(
            "ordinary Arnoldi does not accept target; use method=:jd"))
        krylov_liouvillian_spectrum(L;nev=requested_count,which=selection,
                                     vectors=true,
                                     kwargs...)
    elseif chosen===:iram
        implicitly_restarted_arnoldi_spectrum(L;nev=requested_count,
            which=selection,
            target,vectors=true,kwargs...)
    else
        requested_target=target===nothing ? zero(eltype(L)) : target
        jacobi_davidson_spectrum(L;nev=requested_count,
                                  target=requested_target,
                                  vectors=true,
                                  kwargs...)
    end
    merge(result,(method=chosen,partial_scope=length(result.values)<n,
                  selection=chosen===:jd ? :near_target : selection))
end

"""Liouvillian decay modes together with Hilbert--Schmidt observable overlaps."""
function observable_decay_modes(L,A::PIOperator;k=6,which=:largest_real,kwargs...)
    M=liouvillian_modes(L;k=k,which=which,kwargs...)
    merge(M,(overlaps=[dot(A.data,M.vectors[:,j])
        for j in axes(M.vectors,2)],))
end

_response_has_adjoint(source)=_operator_has_adjoint(source)
_response_has_adjoint(map::FloquetMap)=_operator_has_adjoint(map.source)
_response_has_adjoint(operator::RestrictedLiouvillian)=
    operator.compressed_source!==nothing||
    _operator_has_adjoint(operator.source)

"""
    resolvent_norm(L, z; method=:auto, krylovdim=30, ...)

Return the spectral norm of `(z*I-L)^(-1)`. Explicit matrices retain the dense
SVD path under `method=:auto` only when it fits `memory_budget`. Otherwise
matrix-free power iteration is selected and its forward/adjoint GMRES
workspace is checked against the same bound. Matrix-free sources use power iteration on
`R'R`, with both shifted systems solved by restarted GMRES. This route requires
a certified adjoint action; a custom matrix-free source without one is
rejected rather than materialized implicitly.

`return_info=true` reports power/GMRES convergence. The iterative value is a
converged numerical estimate, not a certified upper bound on the resolvent.
At a singular or insufficiently resolved shift, a failed GMRES solve raises;
the routine never reports a finite norm from a nonconverged shifted solve.
"""
function resolvent_norm(L0,z::Number;method::Symbol=:auto,
        krylovdim::Integer=30,maxiter::Integer=500,
        max_power_iterations::Integer=40,atol=nothing,
        rtol=nothing,power_atol=nothing,power_rtol=nothing,
        initial_vector=nothing,rng=Random.default_rng(),workspace=nothing,
        return_info::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    isfinite(real(z))&&isfinite(imag(z))||throw(ArgumentError(
        "resolvent shift must be finite"))
    _response_source_preflight(L0,memory_budget)
    L=_prepared_response_source(L0,workspace,memory_budget)
    _require_autonomous(L,"Liouvillian resolvent norm")
    dense_estimate=_response_dense_bytes(L,8,z)
    chosen=_response_dense_choice(L,method,memory_budget,dense_estimate,
        :krylov,"dense Liouvillian resolvent norm")
    chosen in (:dense,:krylov,:gmres)||throw(ArgumentError(
        "resolvent method must be :auto, :dense, or :krylov"))
    if chosen===:dense
        M=_dense_liouvillian(L)
        s=_response_svdvals(z*I-M;operation="Liouvillian resolvent norm")
        Rtype=_real_float_type(eltype(s));value=iszero(s[end]) ?
            Rtype(Inf) : inv(s[end])
        return return_info ? (;value,method=:dense,converged=true,
            iterations=1,dimension=size(M,1)) : value
    end
    iterative_estimate=_response_linear_workspace_bytes(L,krylovdim,workspace)
    _require_performance_budget("matrix-free resolvent workspace",
        iterative_estimate,memory_budget;guidance=
        "Reduce krylovdim or increase the budget.")
    _response_has_adjoint(L)||throw(ArgumentError(
        "matrix-free resolvent norm requires an explicit adjoint action"))
    max_power_iterations>0||throw(ArgumentError(
        "max_power_iterations must be positive"))
    maxiter>0||throw(ArgumentError("maxiter must be positive"))
    BigInt(maxiter)<=typemax(Int)&&BigInt(max_power_iterations)<=typemax(Int)||
        throw(ArgumentError(
            "iteration limits must be representable as Int values"))
    work=workspace===nothing ? ResponseWorkspace(
        L;krylovdim,mode=:linear,memory_budget) :
        _check_response_workspace(workspace,L;linear=true)
    T=eltype(work.x);R=_real_float_type(T)
    promoted=_promote_krylov_scalar_type(T,z)
    promoted===T||throw(ArgumentError(
        "resolvent shift cannot be represented in workspace precision $T"))
    shift=T(z)
    isfinite(real(shift))&&isfinite(imag(shift))||throw(ArgumentError(
        "resolvent shift is not finite in workspace precision $T"))
    default_atol=sqrt(eps(R))*sqrt(sqrt(eps(R)))
    default_rtol=sqrt(eps(R))
    atolT=atol===nothing ? default_atol :
        _response_tolerance(R,atol,"atol")
    rtolT=rtol===nothing ? default_rtol :
        _response_tolerance(R,rtol,"rtol")
    power_atolT=power_atol===nothing ? atolT :
        _response_tolerance(R,power_atol,"power_atol")
    power_rtolT=power_rtol===nothing ? rtolT :
        _response_tolerance(R,power_rtol,"power_rtol")
    if initial_vector===nothing
        randn!(rng,work.x)
    else
        length(initial_vector)==length(work.x)||throw(DimensionMismatch(
            "resolvent initial_vector has the wrong length"))
        promote_type(T,eltype(initial_vector))===T||throw(ArgumentError(
            "resolvent initial_vector cannot be represented in workspace precision"))
        copyto!(work.x,initial_vector)
    end
    initial_norm=norm(work.x);initial_norm>zero(R)||throw(ArgumentError(
        "resolvent initial_vector must be nonzero"))
    work.x./=initial_norm
    previous=R(Inf);estimate=zero(R);converged=false
    forward_result=nothing;adjoint_result=nothing;iterations=0
    for power_iteration in 1:max_power_iterations
        forward_result=_response_shifted_solve!(work.y,L,work.x,shift,work;
            adjoint_action=false,atol=atolT,rtol=rtolT,maxiter)
        estimate=norm(work.y)
        adjoint_result=_response_shifted_solve!(work.z,L,work.y,shift,work;
            adjoint_action=true,atol=atolT,rtol=rtolT,maxiter)
        beta=norm(work.z);beta>zero(R)||throw(ArgumentError(
            "resolvent power iteration reached a zero adjoint image"))
        @. work.x=work.z/beta
        iterations=power_iteration
        if isfinite(previous)&&abs(estimate-previous)<=power_atolT+
                power_rtolT*max(estimate,previous,one(R))
            # Re-evaluate at the newly normalized singular-vector iterate.
            # Convergence metadata refers to this returned estimate, rather
            # than to the previous vector that generated the adjoint update.
            forward_result=_response_shifted_solve!(work.y,L,work.x,shift,work;
                adjoint_action=false,atol=atolT,rtol=rtolT,maxiter)
            final_estimate=norm(work.y)
            if abs(final_estimate-estimate)<=power_atolT+
                    power_rtolT*max(final_estimate,estimate,one(R))
                estimate=final_estimate;converged=true;break
            end
        end
        previous=estimate
    end
    converged||throw(ArgumentError(
        "resolvent power iteration did not converge in $max_power_iterations iterations"))
    info=(value=estimate,method=:krylov,converged,iterations,
        dimension=size(L,1),forward_solve=forward_result,
        adjoint_solve=adjoint_result,workspace_reused=workspace!==nothing)
    return_info ? info : estimate
end

"""Largest real grid coordinate where the resolvent norm exceeds `1/epsilon`."""
function pseudospectral_abscissa(L,epsilon::Real;real_grid,imag_grid,
        method::Symbol=:auto,workspace=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        max_grid_points::Integer=100_000,kwargs...)
    epsilon>0||throw(ArgumentError("epsilon must be positive"))
    _response_source_preflight(L,memory_budget)
    source=_prepared_response_source(L,workspace,memory_budget)
    max_grid_points isa Bool&&throw(ArgumentError(
        "max_grid_points must be an integer, not a Bool"))
    max_grid_points>0||throw(ArgumentError(
        "max_grid_points must be positive"))
    point_count=BigInt(length(real_grid))*BigInt(length(imag_grid))
    point_count<=max_grid_points||throw(ArgumentError(
        "pseudospectral grid contains $point_count points, exceeding "*
        "max_grid_points=$max_grid_points; increase max_grid_points "*
        "explicitly after checking the repeated-solve cost"))
    dense_estimate=_response_dense_bytes(
        source,8,epsilon,real_grid,imag_grid)
    chosen=_response_dense_choice(source,method,memory_budget,dense_estimate,
        :krylov,"dense pseudospectral analysis")
    Rtype=_response_real_type(eltype(source),epsilon)
    for grid in (real_grid,imag_grid)
        T=eltype(grid)
        T<:AbstractFloat&&(Rtype=promote_type(Rtype,T))
    end
    epsilonR=_response_tolerance(Rtype,epsilon,"epsilon";positive=true)
    if chosen!==:dense
        iterative_estimate=_response_linear_workspace_bytes(source,
            get(kwargs,:krylovdim,30),workspace)
        _require_performance_budget("pseudospectral Krylov workspace",
            iterative_estimate,memory_budget;guidance=
            "Reduce krylovdim or increase the budget.")
    end
    local_workspace=if workspace===nothing&&chosen!==:dense
        ResponseWorkspace(source;krylovdim=get(kwargs,:krylovdim,30),
                          mode=:linear,memory_budget)
    else
        workspace
    end
    best=Rtype(-Inf)
    for x in real_grid,y in imag_grid
        z=complex(Rtype(x),Rtype(y))
        value=resolvent_norm(source,z;method=chosen,
                             workspace=local_workspace,memory_budget,kwargs...)
        value>=inv(epsilonR)&&(best=max(best,Rtype(x)))
    end
    best
end

"""
    adjoint_evolve(L, A, t; method=:auto, workspace=nothing, ...)

Evolve a PI observable under the adjoint of an autonomous Liouvillian.
Explicit matrices retain the dense exponential under `method=:auto` only
when its work arrays fit `memory_budget`; otherwise the checked adaptive
[`krylov_expv!`](@ref) workspace is used with an explicit
adjoint action. `return_info=true` includes exponential-action convergence and
error diagnostics.
"""
function adjoint_evolve(L0,A::PIOperator,t::Real;method::Symbol=:auto,
        workspace=nothing,krylovdim::Integer=30,return_info::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    _response_source_preflight(L0,memory_budget)
    L=_prepared_response_source(L0,workspace,memory_budget)
    _require_autonomous(L,"adjoint observable evolution")
    size(L,1)==length(A.basis)||throw(DimensionMismatch())
    dense_estimate=_response_dense_bytes(L,9,A.data,t)
    chosen=_response_dense_choice(L,method,memory_budget,dense_estimate,
        :krylov,"dense adjoint exponential evolution")
    chosen in (:dense,:krylov,:expv)||throw(ArgumentError(
        "adjoint evolution method must be :auto, :dense, or :krylov"))
    if chosen===:dense
        M=_dense_liouvillian(L);Rtype=_response_real_type(eltype(A.data),M,t)
        propagator=_response_matrix_exp(Rtype(t)*adjoint(M);
            operation="adjoint Liouvillian evolution")
        value=PIOperator(A.basis,Complex{Rtype}.(propagator*A.data))
        return return_info ? (;value,method=:dense,converged=true,
            dimension=size(M,1)) : value
    end
    iterative_estimate=_response_evolution_workspace_bytes(
        L,krylovdim,workspace)
    _require_performance_budget("matrix-free adjoint-evolution workspace",
        iterative_estimate,memory_budget;guidance=
        "Reduce krylovdim or increase the budget.")
    _response_has_adjoint(L)||throw(ArgumentError(
        "matrix-free adjoint evolution requires an explicit adjoint action"))
    work=workspace===nothing ? ResponseWorkspace(L;
        expv_krylovdim=krylovdim,mode=:evolution,memory_budget) :
        _check_response_workspace(workspace,L;evolution=true)
    T=eltype(work.exponential.current)
    promote_type(T,eltype(A.data))===T||throw(ArgumentError(
        "observable cannot be represented in response workspace precision"))
    operator=_ResponseActionOperator{T,typeof(L),typeof(work)}(L,work,true)
    destination=zeros(T,length(A.data))
    result=krylov_expv!(destination,operator,A.data,t,work.exponential;kwargs...)
    value=PIOperator(A.basis,destination)
    info=merge(result,(value,method=:krylov,dimension=size(L,1),
        workspace_reused=workspace!==nothing))
    return_info ? info : value
end

_value_operator(x,t,p)=x isa Function ? x(t,p) : x

function _sensitivity_apply_batch!(destination,source,input,t,p,work)
    if source isa AbstractMatrix
        return mul!(destination,source,input)
    elseif work===nothing
        if applicable(apply!,destination,source,input,t,p)
            return apply!(destination,source,input,t,p)
        end
    elseif applicable(apply!,destination,source,input,t,p,work)
        return apply!(destination,source,input,t,p,work)
    end
    # Custom vector-only callbacks remain supported. Built-in PI, composite,
    # restricted, and HEOM sources take one genuine matrix-RHS application
    # through the branches above.
    for column in axes(input,2)
        if work===nothing
            apply!(view(destination,:,column),source,
                   view(input,:,column),t,p)
        else
            apply!(view(destination,:,column),source,
                   view(input,:,column),t,p,work)
        end
    end
    destination
end

function _sensitivity_apply_derivative!(destination,derivative,input,t,p,work)
    if derivative isa AbstractMatrix
        mul!(destination,derivative,input)
    elseif work===nothing
        apply!(destination,derivative,input,t,p)
    else
        apply!(destination,derivative,input,t,p,work)
    end
end

"""
    sensitivity_problem(L, rho0, tspan, dLs; parameters=nothing)

Construct an in-place augmented ODE for the state and tangent states
`d rho / d theta_mu`. `dLs[mu]` is either a static matrix/operator or a
callable `(t,p) -> dL/dtheta_mu`. The augmented state has columns
`[rho, drho/dtheta_1, ...]`. Built-in prepared generators apply `L` to all
columns in one matrix-RHS call, so driven schedules are evaluated once per ODE
right-hand side. Task-owned workspaces for static derivative generators are
also prepared once when the problem is constructed.
"""
function sensitivity_problem(L,rho0::PIState,tspan,dLs;parameters=nothing)
    prepared=L isa PIModel ? compile(L;backend=:matrixfree) : L
    derivatives=collect(dLs);m=length(derivatives);n=length(rho0.data)
    Rtype=_response_real_type(eltype(rho0.data),prepared,derivatives...)
    u0=zeros(Complex{Rtype},n,m+1);u0[:,1].=rho0.data
    work=_linear_operator_batch_workspace(
        prepared,m+1,Complex{Rtype})
    forcing=zeros(Complex{Rtype},n,m)
    derivative_work=map(derivatives) do derivative
        derivative isa Function ? nothing :
            _linear_operator_workspace(derivative)
    end
    function f!(du,u,p,t)
        _sensitivity_apply_batch!(du,prepared,u,t,p,work)
        for mu in 1:m
            D=_value_operator(derivatives[mu],t,p)
            # A callable may return a different prepared source at every
            # evaluation, so only immutable derivative specifications own a
            # reusable workspace.
            _sensitivity_apply_derivative!(
                view(forcing,:,mu),D,view(u,:,1),t,p,derivative_work[mu])
            @views du[:,mu+1].+=forcing[:,mu]
        end
    end
    SciMLBase.ODEProblem(f!,u0,tspan,parameters)
end

"""Extract parameter tangent `mu` from an augmented sensitivity solution."""
sensitivity_state(sol,i::Integer,mu::Integer,basis::PIBasis)=PIState(basis,sol.u[i][:,mu+1])

"""Classical Fisher matrix for PI POVM effects and tangent states."""
function classical_fisher_information(rho::PIState,derivatives,effects;
                                      atol::Real=_analysis_atol(rho))
    ds=collect(derivatives);es=collect(effects);m=length(ds)
    Rtype=_response_real_type(eltype(rho.data),ds...,es...);F=zeros(Rtype,m,m)
    for E in es
        prob=real(expectation(rho,E));prob>=-atol||throw(ArgumentError("negative measurement probability"));prob>atol||continue
        dp=Rtype[real(expectation(d,E)) for d in ds];F .+= dp*dp'/prob
    end
    F
end

function _pi_left_product(A::PIOperator,rho::PIState)
    _samebasis(A,rho)
    Rtype=promote_type(_real_float_type(eltype(A.data)),
                       _real_float_type(eltype(rho.data)))
    out=PIState(rho.basis;T=Rtype)
    for p in rho.basis.sectors
        coefficient_block(out,p).=_divide_by_schur_multiplicity_scale(
            coefficient_block(A,p)*coefficient_block(rho,p),Rtype,p)
    end
    out
end

struct _TraceFixedResponseOperator{T,S,W,F,V,R}
    source::S
    workspace::W
    tracevec::F
    anchor::V
    sign::R
    inverse_scale::R
end
Base.size(operator::_TraceFixedResponseOperator)=size(operator.source)
Base.size(operator::_TraceFixedResponseOperator,index::Integer)=
    index in (1,2) ? size(operator.source,index) : 1
Base.eltype(::_TraceFixedResponseOperator{T}) where T=T
function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_TraceFixedResponseOperator,input::AbstractVector)
    _response_apply!(destination,operator.source,input,operator.workspace)
    alpha=dot(operator.tracevec,input)
    @. destination=operator.sign*operator.inverse_scale*destination+
        operator.anchor*alpha
    destination
end
(operator::_TraceFixedResponseOperator)(destination,input)=
    mul!(destination,operator,input)

function _trace_fixed_response_solve!(solution,source,rhs,anchor,tracevec,work;
        sign,operator_scale,atol,rtol,maxiter)
    n=size(source,1)
    all(vector->length(vector)==n,(solution,rhs,anchor,tracevec))||
        throw(DimensionMismatch(
            "trace-fixed response vectors have incompatible dimensions"))
    T=eltype(solution)
    all(vector->promote_type(T,eltype(vector))===T,
        (rhs,anchor,tracevec))||throw(ArgumentError(
            "trace-fixed response vectors have incompatible scalar types"))
    maxiter>0||throw(ArgumentError("maxiter must be positive"))
    BigInt(maxiter)<=typemax(Int)||throw(ArgumentError(
        "maxiter must be representable as an Int"))
    action=_ResponseActionOperator{eltype(solution),typeof(source),typeof(work)}(
        source,work,false)
    scale=_validated_operator_scale(operator_scale===nothing ?
        _estimated_operator_scale!(action,work.z,work.y) : operator_scale)
    R=_real_float_type(eltype(solution))
    scaleR=_response_tolerance(R,scale,"operator_scale";positive=true)
    inverse_scale=inv(scaleR)
    isfinite(inverse_scale)||throw(ArgumentError(
        "inverse operator_scale is not finite in workspace precision $R"))
    atolR=_response_tolerance(R,atol,"atol")
    rtolR=_response_tolerance(R,rtol,"rtol")
    operator=_TraceFixedResponseOperator{eltype(solution),typeof(source),
        typeof(work),typeof(tracevec),typeof(anchor),R}(
        source,work,tracevec,anchor,R(sign),inverse_scale)
    @. work.rhs=inverse_scale*rhs
    fill!(solution,zero(eltype(solution)))
    result=_gmres!(solution,operator,work.rhs,work.forward;
        atol=atolR,rtol=rtolR,maxiter)
    result.converged||throw(ArgumentError(
        "trace-fixed response GMRES did not converge in $(result.iterations) iterations; raw residual=$(result.raw_residual)"))
    _response_apply!(work.y,source,solution,work)
    @. work.y=sign*work.y-rhs
    residual=norm(work.y);trace_error=abs(dot(tracevec,solution))
    physical_tolerance=scaleR*atolR+rtolR*norm(rhs)
    trace_tolerance=atolR+rtolR*max(norm(solution),one(R))
    residual<=physical_tolerance&&trace_error<=trace_tolerance||
        throw(ArgumentError(
            "trace-fixed response solve failed explicit certification: physical_residual=$residual (tolerance=$physical_tolerance), trace_error=$trace_error (tolerance=$trace_tolerance)"))
    merge(result,(physical_residual=residual,trace_error,
        physical_tolerance,trace_tolerance,certified=true,
        operator_scale=scaleR))
end

"""
    integrated_correlation_time(L, rho, A; method=:auto, ...)

Integrated connected autocorrelation time. Matrix-free sources solve the
trace-zero Poisson equation with restarted GMRES and a rank-one physical-trace
constraint; explicit matrices retain the dense pseudoinverse compatibility
path only when it fits `memory_budget`. The selected dense or Krylov workspace
is checked before allocation. `return_info=true` includes raw residual and
trace diagnostics.
"""
function integrated_correlation_time(L0,rho::PIState,A::PIOperator;
        atol::Real=_analysis_atol(rho),rtol::Real=sqrt(eps(
            _real_float_type(eltype(rho.data)))),method::Symbol=:auto,
        krylovdim::Integer=30,maxiter::Integer=500,workspace=nothing,
        operator_scale=nothing,return_info::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _response_source_preflight(L0,memory_budget)
    mean=expectation(rho,A);x=_pi_left_product(A,rho);x.data.-=mean*rho.data
    L=_prepared_response_source(L0,workspace,memory_budget)
    _require_autonomous(L,"integrated correlation time")
    size(L,1)==length(rho.data)||throw(DimensionMismatch())
    dense_estimate=_response_dense_bytes(L,9,rho.data,A.data)
    chosen=_response_dense_choice(L,method,memory_budget,dense_estimate,
        :krylov,"dense integrated-correlation solve")
    chosen in (:dense,:krylov,:gmres)||throw(ArgumentError(
        "correlation-time method must be :auto, :dense, or :krylov"))
    v=variance(rho,A)
    if chosen===:dense
        M=_dense_liouvillian(L)
        integral=-dot(A.data,_response_pinv(M;
            operation="integrated correlation time",rtol=atol)*x.data)
        Rtype=_response_real_type(eltype(rho.data),A,M)
        value=abs(v)<=atol ? Rtype(Inf) : Rtype(real(integral/v))
        return return_info ? (;value,method=:dense,converged=true,
            residual=zero(Rtype),trace_error=zero(Rtype)) : value
    end
    iterative_estimate=_response_linear_workspace_bytes(L,krylovdim,workspace)
    _require_performance_budget("matrix-free integrated-correlation workspace",
        iterative_estimate,memory_budget;guidance=
        "Reduce krylovdim or increase the budget.")
    work=workspace===nothing ? ResponseWorkspace(
        L;krylovdim,mode=:linear,memory_budget) :
        _check_response_workspace(workspace,L;linear=true)
    T=eltype(work.x);promote_type(T,eltype(x.data),eltype(rho.data))===T||
        throw(ArgumentError(
        "state cannot be represented in response workspace precision"))
    tracevec=_trace_functional(rho.basis,_real_float_type(T))
    anchor=T.(rho.data);rhs=T.(x.data)
    result=_trace_fixed_response_solve!(work.x,L,rhs,anchor,tracevec,work;
        sign=-1,operator_scale,atol,rtol,maxiter)
    integral=dot(A.data,work.x);Rtype=_real_float_type(T)
    value=abs(v)<=atol ? Rtype(Inf) : Rtype(real(integral/v))
    info=merge(result,(value,method=:krylov,converged=true,
        residual=result.physical_residual,workspace_reused=workspace!==nothing))
    return_info ? info : value
end

"""
    steady_state_susceptibility(L, rho, dL; method=:auto, observable=nothing)

Solve the trace-fixed tangent equation `L*dρ = -dL*ρ`. Matrix-free sources use
restarted GMRES with a rank-one physical-trace constraint; explicit matrices
retain the dense compatibility route only when it fits `memory_budget`. The
selected dense or Krylov workspace is checked before allocation. The tangent
is never normalized or projected onto positive states.
"""
function steady_state_susceptibility(L0,rho::PIState,dL0;observable=nothing,
        method::Symbol=:auto,krylovdim::Integer=30,maxiter::Integer=500,
        atol::Real=_analysis_atol(rho),rtol::Real=sqrt(eps(
        _real_float_type(eltype(rho.data)))),workspace=nothing,
        operator_scale=nothing,return_info::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _response_source_preflight(L0,memory_budget)
    _response_source_preflight(dL0,memory_budget)
    L=_prepared_response_source(L0,workspace,memory_budget)
    D=_response_source(dL0,memory_budget)
    _require_autonomous(L,"steady-state susceptibility")
    _require_autonomous(D,"steady-state susceptibility perturbation")
    size(L)==size(D)==(length(rho.data),length(rho.data))||
        throw(DimensionMismatch("response generators have incompatible dimensions"))
    dense_estimate=_response_dense_bytes(
        L,12,rho.data,D,observable)
    if method===:auto
        chosen=L isa AbstractMatrix&&D isa AbstractMatrix&&
               _performance_budget_fits(dense_estimate,memory_budget) ?
               :dense : :krylov
    else
        method===:dense&&_require_performance_budget(
            "dense steady-state susceptibility",dense_estimate,memory_budget;
            guidance="Use method=:krylov for bounded matrix-free work.")
        chosen=method
    end
    chosen in (:dense,:krylov,:gmres)||throw(ArgumentError(
        "susceptibility method must be :auto, :dense, or :krylov"))
    if chosen===:dense
        M=_dense_liouvillian(L);Dmatrix=_dense_liouvillian(D)
        Rtype=_response_real_type(eltype(rho.data),M,Dmatrix)
        Ctype=Complex{Rtype};matrix=Matrix{Ctype}(M)
        rhs=-Matrix{Ctype}(Dmatrix)*rho.data
        tau=_trace_functional(rho.basis,Rtype)
        matrix[end,:].=tau;rhs[end]=zero(Ctype)
        tangent=PIState(rho.basis,matrix\rhs)
        value=observable===nothing ? tangent : real(expectation(tangent,observable))
        return return_info ? (;value,state=tangent,method=:dense,
            converged=true,residual=norm(M*tangent.data+Dmatrix*rho.data),
            trace_error=abs(trace(tangent))) : value
    end
    iterative_estimate=_response_linear_workspace_bytes(L,krylovdim,workspace)
    _require_performance_budget("matrix-free susceptibility workspace",
        iterative_estimate,memory_budget;guidance=
        "Reduce krylovdim or increase the budget.")
    work=workspace===nothing ? ResponseWorkspace(
        L;krylovdim,mode=:linear,memory_budget) :
        _check_response_workspace(workspace,L;linear=true)
    T=eltype(work.x);promote_type(T,eltype(rho.data),eltype(D))===T||
        throw(ArgumentError(
        "perturbation or state cannot be represented in response workspace precision"))
    derivative_work=_response_action_workspace(D)
    derivative_response=ResponseWorkspace(D,
        work.forward,work.adjoint,work.exponential,work.x,work.y,work.z,
        work.rhs,derivative_work,work.mode)
    _response_apply!(work.rhs,D,rho.data,derivative_response)
    @. work.rhs=-work.rhs
    rhs=copy(work.rhs)
    tracevec=_trace_functional(rho.basis,_real_float_type(T))
    anchor=T.(rho.data)
    result=_trace_fixed_response_solve!(work.x,L,rhs,anchor,tracevec,work;
        sign=1,operator_scale,atol,rtol,maxiter)
    tangent=PIState(rho.basis,work.x)
    value=observable===nothing ? tangent : real(expectation(tangent,observable))
    info=merge(result,(value,state=tangent,method=:krylov,converged=true,
        residual=result.physical_residual,workspace_reused=workspace!==nothing))
    return_info ? info : value
end
