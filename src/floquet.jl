function _apply_columns!(Y,L,X,t,p,work=nothing)
    if work!==nothing
        apply!(Y,L,X,t,p,work)
    elseif L isa MatrixFreeLiouvillian && L.plan===nothing
        # A user-defined compatibility action is only guaranteed to accept
        # vectors. Compiled PI operators take the batched route below.
        for j in axes(X,2)
            apply!(view(Y,:,j),L,view(X,:,j),t,p)
        end
    else
        apply!(Y,L,X,t,p)
    end
    Y
end

function _check_exact_floquet_integer(::Type{R},value::Integer,label) where R
    converted=R(value)
    isfinite(converted)&&BigInt(converted)==BigInt(value)||throw(ArgumentError(
        "$label=$value is not exactly representable in the Floquet working precision $R; pass an explicit floating-point value with the required precision"))
    converted
end

"""
    floquet_propagator(model_or_L, period; steps=256, t0=0, parameters=nothing)

Integrate one period of a time-dependent PI Liouvillian with a fixed-step RK4
scheme. All stage matrices and matrix-free sector workspaces are preallocated.
Fixed operator terms with time-dependent scalar `rate=(t,p)->...` never
assemble an instantaneous sparse Liouvillian. An `InPlaceTimeOperator` is
evaluated once per RK stage and its dynamic blocks are reused for every
propagator column.
"""
function floquet_propagator(x,period::Real;steps::Integer=256,t0::Real=0,
                            parameters=nothing)
    isfinite(period)&&period>0||throw(ArgumentError("period must be finite and positive"))
    isfinite(t0)||throw(ArgumentError("t0 must be finite"))
    steps>0||throw(ArgumentError("steps must be positive"))
    L=x isa PIModel ? compile(x;backend=:matrixfree) : x
    L isa Union{MatrixFreeLiouvillian,LiouvillianPlan,CompiledPIModel,AbstractMatrix}||
        throw(ArgumentError("unsupported Floquet Liouvillian representation"))
    n=size(L,1);T=_complex_float_type(eltype(L))
    period isa Integer||(T=promote_type(T,_complex_float_type(typeof(period))))
    t0 isa Integer||(T=promote_type(T,_complex_float_type(typeof(t0))))
    R=_real_float_type(T)
    period isa Integer&&_check_exact_floquet_integer(R,period,"period")
    t0 isa Integer&&_check_exact_floquet_integer(R,t0,"t0")
    _check_liouvillian_source_precision(L,T,"Floquet propagator")
    work=_liouvillian_workspace(L)
    U=Matrix{T}(I,n,n);tmp=similar(U);k1=similar(U);k2=similar(U);k3=similar(U);k4=similar(U)
    h=R(period)/R(steps);t=R(t0)
    for step in 1:steps
        _apply_columns!(k1,L,U,t,parameters,work)
        @. tmp=U+(h/2)*k1;_apply_columns!(k2,L,tmp,t+h/2,parameters,work)
        @. tmp=U+(h/2)*k2;_apply_columns!(k3,L,tmp,t+h/2,parameters,work)
        @. tmp=U+h*k3;_apply_columns!(k4,L,tmp,t+h,parameters,work)
        @. U=U+(h/6)*(k1+2k2+2k3+k4);t+=h
    end
    U
end

"""Floquet multipliers, sorted by decreasing modulus."""
floquet_multipliers(F::AbstractMatrix)=sort(eigvals(Matrix(F));by=abs,rev=true)
floquet_multipliers(x,period::Real;kwargs...)=floquet_multipliers(floquet_propagator(x,period;kwargs...))

"""Principal-branch Floquet exponents `log(lambda)/period`."""
floquet_exponents(F::AbstractMatrix,period::Real)=log.(complex.(floquet_multipliers(F)))./period
floquet_exponents(x,period::Real;kwargs...)=floquet_exponents(floquet_propagator(x,period;kwargs...),period)

"""
    floquet_gap(F, period; atol=1e-10)

Return the nonnegative asymptotic decay rate from the subleading Floquet
multiplier. The map must have a fixed multiplier within `atol` of one and no
remaining multiplier with modulus larger than `1 + atol`; otherwise an
`ArgumentError` is thrown. A one-dimensional map has no subleading mode and
returns a precision-matched `NaN`.
"""
function floquet_gap(F::AbstractMatrix,period::Real;atol::Real=1e-10)
    isfinite(period)&&period>0||throw(ArgumentError(
        "period must be finite and positive"))
    isfinite(atol)&&atol>=0||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    vals=floquet_multipliers(F);isempty(vals)&&throw(ArgumentError(
        "Floquet gap requires a nonempty propagator"))
    R=_real_float_type(eltype(vals))
    distances=abs.(vals.-one(eltype(vals)));i=argmin(distances)
    distances[i]<=atol||throw(ArgumentError(
        "Floquet map has no unit multiplier within atol=$atol; closest distance=$(distances[i])"))
    deleteat!(vals,i)
    rate_zero=zero(R)/period
    isempty(vals)&&return oftype(rate_zero,NaN)
    radius=maximum(abs,vals)
    radius<=one(radius)+atol||throw(ArgumentError(
        "Floquet map is unstable within atol=$atol: subleading spectral radius=$radius"))
    rate=-log(radius)/period
    max(zero(rate),rate)
end
function floquet_gap(x,period::Real;atol::Real=1e-10,kwargs...)
    floquet_gap(floquet_propagator(x,period;kwargs...),period;atol=atol)
end

"""Trace-normalized periodic state at the chosen Floquet time origin."""
function floquet_steady_state(model::PIModel,period::Real;return_info::Bool=false,kwargs...)
    F=floquet_propagator(model,period;kwargs...)
    info=steady_state(F-I;basis=model.basis,method=:svd,return_info=true)
    rho=PIState(model.basis,info.state)
    result=(state=rho,propagator=F,residual=norm(F*rho.data-rho.data),trace_error=abs(trace(rho)-1),nullity=info.nullity)
    return_info ? result : rho
end

"""Return PI states after `0:nperiods` applications of a Floquet propagator."""
function stroboscopic_evolution(rho::PIState,F::AbstractMatrix,nperiods::Integer;include_initial::Bool=true)
    nperiods>=0||throw(ArgumentError("nperiods must be nonnegative"))
    n=length(rho.data);size(F)==(n,n)||throw(DimensionMismatch())
    x=_product_destination(F,rho.data,n);copyto!(x,rho.data);y=similar(x)
    R=_real_float_type(eltype(x));out=PIState{R,typeof(rho.basis)}[]
    # `PIState` already makes one defensive copy, so passing `x` directly
    # keeps every saved state detached without an immediately discarded copy.
    include_initial&&push!(out,PIState(rho.basis,x))
    for _ in 1:nperiods
        mul!(y,F,x);x,y=y,x;push!(out,PIState(rho.basis,x))
    end
    out
end

"""State after an integer number of periods."""
function floquet_evolve(rho::PIState,F::AbstractMatrix,nperiods::Integer)
    n=length(rho.data);size(F)==(n,n)||throw(DimensionMismatch())
    # Preserve the established inverse-map semantics for negative periods.
    # The allocation-saving repeated `mul!` path applies to forward periods;
    # an inverse power may fail exactly as ordinary matrix powering did when
    # the supplied map is singular.
    nperiods<0&&return PIState(rho.basis,F^nperiods*rho.data)
    x=_product_destination(F,rho.data,n)
    if iszero(nperiods)
        copyto!(x,rho.data)
        return PIState(rho.basis,x)
    end
    mul!(x,F,rho.data)
    if nperiods>1
        y=similar(x)
        for _ in 2:nperiods
            mul!(y,F,x);x,y=y,x
        end
    end
    PIState(rho.basis,x)
end
