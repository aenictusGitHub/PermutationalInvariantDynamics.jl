_dense_liouvillian(L::AbstractMatrix)=Matrix(L)
_dense_liouvillian(L)=Matrix(_materialize(L))

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

"""Return selected Liouvillian eigenvalues and right decay modes."""
function liouvillian_modes(L;k::Integer=6,which=:largest_real)
    M=_dense_liouvillian(L)
    E=_response_eigen(M;operation="Liouvillian mode analysis")
    order=which===:largest_real ? sortperm(real.(E.values);rev=true) : sortperm(abs.(E.values))
    idx=order[1:min(k,length(order))];(;values=E.values[idx],vectors=E.vectors[:,idx])
end

"""Liouvillian decay modes together with Hilbert--Schmidt observable overlaps."""
function observable_decay_modes(L,A::PIOperator;k=6,which=:largest_real)
    M=liouvillian_modes(L;k=k,which=which);(;values=M.values,overlaps=[dot(A.data,M.vectors[:,j]) for j in axes(M.vectors,2)],vectors=M.vectors)
end

"""Spectral norm of the Liouvillian resolvent `(z*I-L)^(-1)`."""
function resolvent_norm(L,z::Number)
    M=_dense_liouvillian(L)
    s=_response_svdvals(z*I-M;operation="Liouvillian resolvent norm")
    Rtype=_real_float_type(eltype(s));iszero(s[end]) ? Rtype(Inf) : inv(s[end])
end

"""Largest real grid coordinate where the resolvent norm exceeds `1/epsilon`."""
function pseudospectral_abscissa(L,epsilon::Real;real_grid,imag_grid)
    epsilon>0||throw(ArgumentError("epsilon must be positive"))
    M=_dense_liouvillian(L);Rtype=_response_real_type(eltype(M),epsilon)
    for grid in (real_grid,imag_grid)
        T=eltype(grid)
        T<:AbstractFloat&&(Rtype=promote_type(Rtype,T))
    end
    epsilonR=Rtype(epsilon);best=Rtype(-Inf)
    for x in real_grid,y in imag_grid
        z=complex(Rtype(x),Rtype(y))
        s=_response_svdvals(z*I-M;operation="Liouvillian pseudospectrum")
        value=iszero(s[end]) ? Rtype(Inf) : Rtype(inv(s[end]))
        value>=inv(epsilonR)&&(best=max(best,Rtype(x)))
    end
    best
end

"""Evolve a PI observable under the adjoint of a time-independent Liouvillian."""
function adjoint_evolve(L,A::PIOperator,t::Real)
    size(L,1)==length(A.basis)||throw(DimensionMismatch())
    M=_dense_liouvillian(L);Rtype=_response_real_type(eltype(A.data),M,t)
    propagator=_response_matrix_exp(Rtype(t)*adjoint(M);operation="adjoint Liouvillian evolution")
    PIOperator(A.basis,Complex{Rtype}.(propagator*A.data))
end

_value_operator(x,t,p)=x isa Function ? x(t,p) : x

"""
    sensitivity_problem(L, rho0, tspan, dLs; parameters=nothing)

Construct an in-place augmented ODE for the state and tangent states
`d rho / d theta_mu`. `dLs[mu]` is either a static matrix/operator or a
callable `(t,p) -> dL/dtheta_mu`. The augmented state has columns
`[rho, drho/dtheta_1, ...]`.
"""
function sensitivity_problem(L,rho0::PIState,tspan,dLs;parameters=nothing)
    prepared=L isa PIModel ? compile(L;backend=:matrixfree) : L
    derivatives=collect(dLs);m=length(derivatives);n=length(rho0.data)
    Rtype=_response_real_type(eltype(rho0.data),prepared,derivatives...)
    u0=zeros(Complex{Rtype},n,m+1);u0[:,1].=rho0.data
    work=_liouvillian_workspace(prepared);forcing=zeros(Complex{Rtype},n,m)
    function f!(du,u,p,t)
        if work===nothing;apply!(view(du,:,1),prepared,view(u,:,1),t,p)
        else;apply!(view(du,:,1),prepared,view(u,:,1),t,p,work);end
        for mu in 1:m
            if work===nothing;apply!(view(du,:,mu+1),prepared,view(u,:,mu+1),t,p)
            else;apply!(view(du,:,mu+1),prepared,view(u,:,mu+1),t,p,work);end
            D=_value_operator(derivatives[mu],t,p)
            apply!(view(forcing,:,mu),D,view(u,:,1),t,p)
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

"""Integrated connected autocorrelation time from the Liouvillian pseudoinverse."""
function integrated_correlation_time(L,rho::PIState,A::PIOperator;
                                     atol::Real=_analysis_atol(rho))
    mean=expectation(rho,A);x=_pi_left_product(A,rho);x.data.-=mean*rho.data
    M=_dense_liouvillian(L)
    integral=-dot(A.data,_response_pinv(M;operation="integrated correlation time",rtol=atol)*x.data);v=variance(rho,A)
    Rtype=_response_real_type(eltype(rho.data),A,M)
    abs(v)<=atol ? Rtype(Inf) : Rtype(real(integral/v))
end

function _trace_functional(b::PIBasis,::Type{R}) where R<:AbstractFloat
    tau=zeros(Complex{R},length(b))
    for (s,p) in pairs(b.sectors);n=length(b.patterns[s]);for i in 1:n
        tau[b.offsets[s]+i-1+(i-1)*n]=_schur_multiplicity_scale(R,p)
    end;end;tau
end

"""Steady-state tangent under a static generator perturbation `dL`."""
function steady_state_susceptibility(L,rho::PIState,dL;observable=nothing)
    M=_dense_liouvillian(L);D=_dense_liouvillian(dL)
    Rtype=_response_real_type(eltype(rho.data),M,D);Ctype=Complex{Rtype}
    A=Matrix{Ctype}(M);rhs=-Matrix{Ctype}(D)*rho.data
    tau=_trace_functional(rho.basis,Rtype);A[end,:].=tau;rhs[end]=zero(Ctype)
    tangent=PIState(rho.basis,A\rhs)
    observable===nothing ? tangent : real(expectation(tangent,observable))
end
