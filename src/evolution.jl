"""
Reusable storage for allocation-conscious fourth-order Liouvillian evolution.

New workspaces use three full integration arrays: `tmp` is the stage state,
`k1` is the current derivative, and `k2` is the weighted stage accumulator.
The legacy `k3` and `k4` fields remain as empty compatibility placeholders;
legacy field-constructed workspaces with full arrays also remain accepted.
"""
struct EvolutionWorkspace{V,E,LW}
    tmp::V
    k1::V
    k2::V
    k3::E
    k4::E
    liouvillian::LW
end

EvolutionWorkspace(x::AbstractVector)=EvolutionWorkspace(
    similar(x),similar(x),similar(x),similar(x,0),similar(x,0),nothing)
EvolutionWorkspace(rho::PIState)=EvolutionWorkspace(rho.data)

_evolution_liouvillian(L::PIModel)=compile(L)
_evolution_liouvillian(L)=L

function _checked_evolution_tspan(tspan;ordered::Bool=false)
    length(tspan)==2||throw(ArgumentError("tspan must contain exactly two endpoints"))
    t0,t1=tspan
    all(t->t isa Real&&!(t isa Bool)&&isfinite(t)&&isfinite(float(t)),(t0,t1))||
        throw(ArgumentError("tspan endpoints must be finite real numbers"))
    ordered&&t1<t0&&throw(ArgumentError("tspan must be ordered"))
    t0,t1
end

function _check_evolution_basis(L,rho::PIState)
    basis=_operator_basis(L)
    basis===nothing||basis===rho.basis||throw(ArgumentError(
        "Liouvillian source and initial state use incompatible PI bases"))
    nothing
end

function _check_evolution_precision(L,::Type{S},::Type{T}) where {S,T}
    promote_type(eltype(L),S,T)===T||throw(ArgumentError(
        "evolution destination scalar type $T cannot represent the source and " *
        "Liouvillian precision; use a wider destination and workspace"))
    _check_liouvillian_source_precision(L,T,"evolution state")
    nothing
end

function _prepare_evolution_state(L,rho::PIState)
    _check_evolution_basis(L,rho)
    n=length(rho.data)
    size(L)==(n,n)||throw(DimensionMismatch(
        "Liouvillian and initial state dimensions differ"))
    T=_complex_float_type(promote_type(eltype(L),eltype(rho.data)))
    _check_evolution_precision(L,eltype(rho.data),T)
    eltype(rho.data)===T ? copy(rho) : PIState(rho.basis,T.(rho.data))
end

function EvolutionWorkspace(L0,x::AbstractVector)
    L=_evolution_liouvillian(L0)
    _check_evolution_precision(L,eltype(x),eltype(x))
    EvolutionWorkspace(similar(x),similar(x),similar(x),similar(x,0),
                       similar(x,0),_linear_operator_workspace(L))
end
function EvolutionWorkspace(L,rho::PIState)
    _check_evolution_basis(L,rho)
    EvolutionWorkspace(L,rho.data)
end

function _evolution_action!(y,L,x,t,p,work::EvolutionWorkspace)
    if work.liouvillian===nothing
        if L isa LiouvillianPlan
            apply!(y,L,x,t,p)
        else
            _liouvillian_action!(y,L,x,t,p)
        end
    else
        apply!(y,L,x,t,p,work.liouvillian)
    end
end

function _check_evolution_workspace(w::EvolutionWorkspace,n)
    all(v->length(v)==n,(w.tmp,w.k1,w.k2))&&
        all(v->length(v) in (0,n),(w.k3,w.k4))||throw(DimensionMismatch(
            "evolution workspace has the wrong dimension"))
end

function _check_evolution_scratch_aliases(w::EvolutionWorkspace,destination)
    active=(w.tmp,w.k1,w.k2)
    for i in eachindex(active)
        Base.mightalias(active[i],destination)&&throw(ArgumentError(
            "evolution destination must not alias workspace scratch"))
        for j in 1:i-1
            Base.mightalias(active[i],active[j])&&throw(ArgumentError(
                "evolution workspace scratch arrays must not alias"))
        end
    end
    nothing
end

"""
    evolve!(dest, L, src, tspan; steps=256, parameters=nothing,
            workspace=nothing, progress=false, on_event=nothing,
            cancellation_token=nothing)

Propagate PI coordinates using a preallocated three-scratch fixed-step RK4
kernel. `L` may be a matrix, `MatrixFreeLiouvillian`, or `PIModel`. `dest` may
alias `src`, but it must not alias active workspace scratch. Progress and
cooperative cancellation are observed after each complete RK4 step. On
cancel, [`OperationCancelled`](@ref) is thrown and `dest` contains the last
fully completed step.
Endpoints must be finite real numbers; backward propagation is allowed.
The destination and active workspace arrays must share a scalar type that
can represent both the source and generator without narrowing.
"""
function evolve!(dest::AbstractVector,L0,src::AbstractVector,tspan;
                 steps::Integer=256,parameters=nothing,workspace=nothing,
                 progress=false,on_event=nothing,cancellation_token=nothing)
    t0,t1=_checked_evolution_tspan(tspan)
    L=_evolution_liouvillian(L0);n=length(src)
    length(dest)==n||throw(DimensionMismatch("source and destination dimensions differ"))
    size(L)==(n,n)||throw(DimensionMismatch("Liouvillian and state dimensions differ"))
    steps>0||throw(ArgumentError("steps must be positive"))
    dest===src||!Base.mightalias(dest,src)||throw(ArgumentError(
        "evolution permits exact in-place use but not partially overlapping source and destination storage"))
    h=(t1-t0)/steps
    isfinite(h)&&(t0==t1||!iszero(h))||throw(ArgumentError(
        "RK4 step must be finite and nonzero for a nonzero interval; use wider time precision"))
    _check_evolution_precision(L,eltype(src),eltype(dest))
    w=workspace===nothing ? EvolutionWorkspace(L,dest) : workspace
    _check_evolution_workspace(w,n)
    eltype(w.tmp)===eltype(dest)||throw(ArgumentError(
        "evolution workspace and destination must have the same scalar type"))
    _check_evolution_scratch_aliases(w,dest)
    dest===src||copyto!(dest,src);t=t0
    progress_context=_prepare_progress(:evolve;
        progress,on_event,cancellation_token)
    _progress_emit!(progress_context,:started,0,steps;
        message="RK4 evolution started")
    _progress_throw_if_cancelled!(progress_context,0,steps)
    for step_index in 1:steps
        _evolution_action!(w.k1,L,dest,t,parameters,w)
        copyto!(w.k2,w.k1)
        @. w.tmp=dest+(h/2)*w.k1
        _evolution_action!(w.k1,L,w.tmp,t+h/2,parameters,w)
        @. w.k2=w.k2+2w.k1
        @. w.tmp=dest+(h/2)*w.k1
        _evolution_action!(w.k1,L,w.tmp,t+h/2,parameters,w)
        @. w.k2=w.k2+2w.k1
        @. w.tmp=dest+h*w.k1
        _evolution_action!(w.k1,L,w.tmp,t+h,parameters,w)
        @. w.k2=w.k2+w.k1
        @. dest=dest+(h/6)*w.k2
        t+=h
        if progress_context!==nothing
            _progress_emit!(progress_context,:advanced,step_index,steps;
                message="completed RK4 step $step_index",
                metadata=(time=t,))
            _progress_throw_if_cancelled!(progress_context,step_index,steps)
        end
    end
    _progress_emit!(progress_context,:completed,steps,steps;
        message="RK4 evolution completed",metadata=(time=t,))
    dest
end

function evolve!(dest::PIState,L,src::PIState,tspan;kwargs...)
    dest.basis===src.basis||throw(ArgumentError("source and destination use incompatible PI bases"))
    _check_evolution_basis(L,src)
    evolve!(dest.data,L,src.data,tspan;kwargs...);dest
end

const _EvolutionLiouvillian=Union{AbstractMatrix,MatrixFreeLiouvillian,
                                  LiouvillianPlan,CompiledPIModel,
                                  SpecializedPIModel,PIModel}

"""
Return the propagated PI state, promoting initial coefficients to the
generator's precision when supported.
"""
function time_evolve(L::_EvolutionLiouvillian,rho::PIState,tspan;kwargs...)
    _checked_evolution_tspan(tspan)
    _check_evolution_basis(L,rho)
    prepared=_evolution_liouvillian(L)
    out=_prepare_evolution_state(prepared,rho)
    evolve!(out,prepared,rho,tspan;kwargs...)
end
time_evolve(rho::PIState,L::_EvolutionLiouvillian,tspan;kwargs...)=time_evolve(L,rho,tspan;kwargs...)

"""Return states at ordered sampling times using one reusable workspace."""
function time_evolution(L::_EvolutionLiouvillian,rho::PIState,times;
                        steps_per_interval::Integer=64,parameters=nothing)
    ts=collect(times);isempty(ts)&&return typeof(rho)[]
    all(t->t isa Real&&!(t isa Bool)&&isfinite(t),ts)||throw(ArgumentError(
        "sampling times must be finite real numbers"))
    all(diff(ts).>=0)||throw(ArgumentError("times must be nondecreasing"))
    steps_per_interval>0||throw(ArgumentError("steps_per_interval must be positive"))
    _check_evolution_basis(L,rho)
    prepared=_evolution_liouvillian(L)
    x=_prepare_evolution_state(prepared,rho)
    w=EvolutionWorkspace(prepared,x);out=typeof(x)[copy(x)]
    for i in 2:length(ts)
        ts[i]==ts[i-1]||evolve!(x,prepared,x,(ts[i-1],ts[i]);steps=steps_per_interval,parameters=parameters,workspace=w)
        push!(out,copy(x))
    end
    out
end
time_evolution(rho::PIState,L::_EvolutionLiouvillian,times;kwargs...)=time_evolution(L,rho,times;kwargs...)
