"""Reusable storage for allocation-conscious fourth-order Liouvillian evolution."""
struct EvolutionWorkspace{V,LW}
    tmp::V
    k1::V
    k2::V
    k3::V
    k4::V
    liouvillian::LW
end

EvolutionWorkspace(x::AbstractVector)=EvolutionWorkspace(similar(x),similar(x),similar(x),similar(x),similar(x),nothing)
EvolutionWorkspace(rho::PIState)=EvolutionWorkspace(rho.data)

_evolution_liouvillian(L::PIModel)=compile(L)
_evolution_liouvillian(L)=L

_liouvillian_workspace(::Any)=nothing
_liouvillian_workspace(plan::LiouvillianPlan)=LiouvillianWorkspace(plan)
_liouvillian_workspace(L::MatrixFreeLiouvillian)=L.plan===nothing ? nothing : LiouvillianWorkspace(L.plan)
_liouvillian_workspace(compiled::CompiledPIModel)=LiouvillianWorkspace(compiled.plan)

function EvolutionWorkspace(L0,x::AbstractVector)
    L=_evolution_liouvillian(L0)
    EvolutionWorkspace(similar(x),similar(x),similar(x),similar(x),similar(x),
                       _liouvillian_workspace(L))
end
EvolutionWorkspace(L,rho::PIState)=EvolutionWorkspace(L,rho.data)

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
    all(v->length(v)==n,(w.tmp,w.k1,w.k2,w.k3,w.k4)) || throw(DimensionMismatch("evolution workspace has the wrong dimension"))
end

"""
    evolve!(dest, L, src, tspan; steps=256, parameters=nothing, workspace=nothing)

Propagate PI coordinates using a preallocated fixed-step RK4 kernel. `L` may
be a matrix, `MatrixFreeLiouvillian`, or `PIModel`. `dest` may alias `src`.
"""
function evolve!(dest::AbstractVector,L0,src::AbstractVector,tspan;
                 steps::Integer=256,parameters=nothing,workspace=nothing)
    L=_evolution_liouvillian(L0);n=length(src)
    length(dest)==n||throw(DimensionMismatch("source and destination dimensions differ"))
    size(L)==(n,n)||throw(DimensionMismatch("Liouvillian and state dimensions differ"))
    steps>0||throw(ArgumentError("steps must be positive"))
    t0,t1=tspan;h=(t1-t0)/steps
    w=workspace===nothing ? EvolutionWorkspace(L,src) : workspace
    _check_evolution_workspace(w,n);dest===src||copyto!(dest,src);t=t0
    for _ in 1:steps
        _evolution_action!(w.k1,L,dest,t,parameters,w)
        @. w.tmp=dest+(h/2)*w.k1
        _evolution_action!(w.k2,L,w.tmp,t+h/2,parameters,w)
        @. w.tmp=dest+(h/2)*w.k2
        _evolution_action!(w.k3,L,w.tmp,t+h/2,parameters,w)
        @. w.tmp=dest+h*w.k3
        _evolution_action!(w.k4,L,w.tmp,t+h,parameters,w)
        @. dest=dest+(h/6)*(w.k1+2w.k2+2w.k3+w.k4)
        t+=h
    end
    dest
end

function evolve!(dest::PIState,L,src::PIState,tspan;kwargs...)
    dest.basis===src.basis||throw(ArgumentError("source and destination use incompatible PI bases"))
    evolve!(dest.data,L,src.data,tspan;kwargs...);dest
end

const _EvolutionLiouvillian=Union{AbstractMatrix,MatrixFreeLiouvillian,
                                  LiouvillianPlan,CompiledPIModel,PIModel}

"""Return the PI state obtained by propagating `rho` over `tspan`."""
function time_evolve(L::_EvolutionLiouvillian,rho::PIState,tspan;kwargs...)
    out=copy(rho);evolve!(out,L,rho,tspan;kwargs...)
end
time_evolve(rho::PIState,L::_EvolutionLiouvillian,tspan;kwargs...)=time_evolve(L,rho,tspan;kwargs...)

"""Return states at ordered sampling times using one reusable workspace."""
function time_evolution(L::_EvolutionLiouvillian,rho::PIState,times;
                        steps_per_interval::Integer=64,parameters=nothing)
    ts=collect(times);isempty(ts)&&return PIState[]
    all(diff(ts).>=0)||throw(ArgumentError("times must be nondecreasing"))
    steps_per_interval>0||throw(ArgumentError("steps_per_interval must be positive"))
    prepared=_evolution_liouvillian(L)
    x=copy(rho);w=EvolutionWorkspace(prepared,x);out=typeof(x)[copy(x)]
    for i in 2:length(ts)
        ts[i]==ts[i-1]||evolve!(x,prepared,x,(ts[i-1],ts[i]);steps=steps_per_interval,parameters=parameters,workspace=w)
        push!(out,copy(x))
    end
    out
end
time_evolution(rho::PIState,L::_EvolutionLiouvillian,times;kwargs...)=time_evolution(L,rho,times;kwargs...)
