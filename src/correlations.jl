"""
    CorrelationPlan(L, A, B; right=nothing)

Prepare the quantum-regression correlation

```math
C_{AB}(\\tau)=\\mathrm{tr}\\!\\left[A\\,e^{\\mathcal L\\tau}
    (B\\rho R)\\right],
```

where `R` is the optional `right` operator (the identity when omitted).
`L` may be a `PIModel`, compiled model, matrix-free Liouvillian, plan, or
matrix.  A model is compiled with the matrix-free backend.  The generator
must be autonomous; explicitly driven generators should first be frozen at
the desired time.

The Schur blocks of the insertion operators and the readout vector are copied
once into the immutable plan.  Numerical scratch is deliberately excluded;
construct one [`CorrelationWorkspace`](@ref) per task.

The first operator is *not* implicitly adjointed: the returned contraction is
exactly `tr(A * ...)`.  Thus the standard optical first-order correlation is
obtained with `A=adjoint(c)` and `B=c`.
"""
struct CorrelationPlan{B,L,T,RB}
    basis::B
    generator::L
    readout::Vector{Complex{T}}
    left_blocks::Vector{Matrix{Complex{T}}}
    right_blocks::RB
    tracevec::Vector{Complex{T}}
end

_correlation_generator(model::PIModel)=compile(model;backend=:matrixfree)
_correlation_generator(L)=L

function _correlation_basis(L)
    L isa PIModel&&return L.basis
    L isa CompiledPIModel&&return L.plan.basis
    L isa LiouvillianPlan&&return L.basis
    L isa MatrixFreeLiouvillian&&L.plan!==nothing&&return L.plan.basis
    nothing
end

function _correlation_trace_vector(b::PIBasis,::Type{R}) where R<:AbstractFloat
    tau=zeros(Complex{R},length(b))
    for (s,p) in pairs(b.sectors)
        n=length(b.patterns[s]);scale=_schur_multiplicity_scale(R,p)
        offset=b.offsets[s]
        @inbounds for index in 1:n
            tau[offset+index-1+(index-1)*n]=scale
        end
    end
    tau
end

function _correlation_can_store(::Type{R},::Type{S}) where {R,S}
    promote_type(R,S)===R
end

function _correlation_real_input(::Type{R},value,label) where R<:AbstractFloat
    value isa Real||throw(ArgumentError("$label must be real"))
    if value isa Integer
        converted=R(value)
        isfinite(converted)&&BigInt(converted)==BigInt(value)||throw(ArgumentError(
            "$label=$value is not exactly representable in correlation precision $R"))
        return converted
    end
    promote_type(R,typeof(value))===R||throw(ArgumentError(
        "$label scalar type $(typeof(value)) cannot be represented by " *
        "correlation precision $R without narrowing; use matching inputs " *
        "or prepare the model and operators at a wider precision"))
    converted=R(value)
    isfinite(converted)||throw(ArgumentError("$label must be finite"))
    converted
end

function _correlation_physical_blocks(operator::PIOperator,::Type{R}) where
        R<:AbstractFloat
    blocks=Vector{Matrix{Complex{R}}}(undef,length(operator.basis.sectors))
    for (s,p) in pairs(operator.basis.sectors)
        source=coefficient_block(operator,p)
        block=_divide_by_schur_multiplicity_scale(source,R,p)
        blocks[s]=Matrix{Complex{R}}(block)
        all(isfinite,blocks[s])||throw(ArgumentError(
            "correlation insertion has a nonfinite physical Schur block in sector $p"))
    end
    blocks
end

function CorrelationPlan(L0,A::PIOperator,B::PIOperator;right=nothing)
    _samebasis(A,B)
    right===nothing||begin
        right isa PIOperator||throw(ArgumentError("right must be a PIOperator or nothing"))
        _samebasis(A,right)
    end
    b=A.basis
    known_basis=_correlation_basis(L0)
    known_basis===nothing||known_basis===b||throw(ArgumentError(
        "the correlation operators and generator use incompatible PI bases"))
    L=_correlation_generator(L0)
    size(L)==(length(b),length(b))||throw(DimensionMismatch(
        "Liouvillian and PI correlation dimensions differ"))
    _require_autonomous(L,"quantum-regression correlations")

    LR=_real_float_type(eltype(L))
    operator_types=right===nothing ?
        (_real_float_type(eltype(A.data)),_real_float_type(eltype(B.data))) :
        (_real_float_type(eltype(A.data)),_real_float_type(eltype(B.data)),
         _real_float_type(eltype(right.data)))
    fixed=_fixed_liouvillian_scalar_type(L)
    R=fixed===nothing ? promote_type(LR,operator_types...) : LR
    for S in operator_types
        _correlation_can_store(R,S)||throw(ArgumentError(
            "correlation plan scalar type $R cannot represent operator scalar type $S; " *
            "compile the model and operators at the wider precision"))
    end

    # expectation(rho,A) uses Tr(A' rho).  Storing the coefficients of A'
    # therefore makes dot(readout,x) equal exactly Tr(A*x), as required by QRT.
    readout=Complex{R}.(adjoint(A).data)
    left_blocks=_correlation_physical_blocks(B,R)
    right_blocks=right===nothing ? nothing : _correlation_physical_blocks(right,R)
    CorrelationPlan(b,L,readout,left_blocks,right_blocks,
                    _correlation_trace_vector(b,R))
end

Base.eltype(::CorrelationPlan{B,L,T}) where {B,L,T}=Complex{T}

"""
    CorrelationWorkspace(plan; krylovdim=30)

Caller-owned scratch for time-domain quantum regression and matrix-free
shifted-GMRES spectra.  A workspace is tied to one plan and may be reused
sequentially, but must not be shared concurrently between tasks.
"""
struct CorrelationWorkspace{P,V,E,K}
    plan::P
    state::V
    product::V
    rhs::V
    solution::V
    evolution::E
    krylov::K
end

function CorrelationWorkspace(plan::CorrelationPlan;krylovdim::Integer=30)
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    n=length(plan.basis);T=eltype(plan)
    state=zeros(T,n);product=zeros(T,n);rhs=zeros(T,n);solution=zeros(T,n)
    evolution=EvolutionWorkspace(plan.generator,state)
    krylov=KrylovWorkspace(T,n,krylovdim)
    CorrelationWorkspace(plan,state,product,rhs,solution,evolution,krylov)
end

function _check_correlation_workspace(work::CorrelationWorkspace,
                                      plan::CorrelationPlan)
    work.plan===plan||throw(ArgumentError(
        "correlation workspace belongs to a different plan"))
    n=length(plan.basis)
    all(v->length(v)==n,(work.state,work.product,work.rhs,work.solution))||
        throw(DimensionMismatch("correlation workspace has the wrong dimension"))
    work
end

function _check_correlation_state(plan::CorrelationPlan,rho::PIState)
    rho.basis===plan.basis||throw(ArgumentError(
        "state and correlation plan use incompatible PI bases"))
    R=_real_float_type(eltype(plan));S=_real_float_type(eltype(rho.data))
    _correlation_can_store(R,S)||throw(ArgumentError(
        "correlation plan scalar type $R cannot represent state scalar type $S; " *
        "construct the plan at the wider precision"))
    rho
end

function _correlation_seed!(destination,plan::CorrelationPlan,rho::PIState,
                            work::CorrelationWorkspace)
    for (s,p) in pairs(plan.basis.sectors)
        n=length(plan.basis.patterns[s])
        range=plan.basis.offsets[s]:(plan.basis.offsets[s+1]-1)
        input=reshape(view(rho.data,range),n,n)
        output=reshape(view(destination,range),n,n)
        left=plan.left_blocks[s]
        if plan.right_blocks===nothing
            mul!(output,left,input)
        else
            temporary=reshape(view(work.product,range),n,n)
            mul!(temporary,left,input)
            mul!(output,temporary,plan.right_blocks[s])
        end
    end
    destination
end

@inline _correlation_readout(plan::CorrelationPlan,state)=dot(plan.readout,state)

function _validate_correlation_delays(ts)
    isempty(ts)&&return ts
    previous=first(ts)
    previous isa Real&&isfinite(previous)||throw(ArgumentError(
        "correlation delays must be finite real values"))
    previous>=zero(previous)||throw(ArgumentError(
        "correlation delays must be nonnegative"))
    for index in Iterators.drop(eachindex(ts),1)
        current=ts[index]
        current isa Real&&isfinite(current)||throw(ArgumentError(
            "correlation delays must be finite real values"))
        current>=zero(current)||throw(ArgumentError(
            "correlation delays must be nonnegative"))
        current>=previous||throw(ArgumentError(
            "correlation delays must be nondecreasing"))
        previous=current
    end
    ts
end

function _checked_correlation_delays(delays)
    ts=delays isa AbstractVector ? delays : collect(delays)
    _validate_correlation_delays(ts)
end

function _checked_correlation_delays(plan::CorrelationPlan,delays)
    R=_real_float_type(eltype(plan))
    if delays isa AbstractVector
        S=eltype(delays)
        if S===R
            for value in delays
                _correlation_real_input(R,value,"correlation delay")
            end
            return _validate_correlation_delays(delays)
        end
    end
    converted=R[_correlation_real_input(R,value,"correlation delay")
                for value in delays]
    _validate_correlation_delays(converted)
end

function _correlation_evolve_interval!(state,plan::CorrelationPlan,t0,t1,
                                       steps::Integer,
                                       work::CorrelationWorkspace)
    # CorrelationPlan construction has already required an autonomous
    # generator.  Keeping parameters=nothing here is intentional: a driven
    # QRT requires a two-time propagator with an explicit time origin and is
    # outside this stationary-delay API rather than being approximated by
    # silently freezing a schedule.
    h=(t1-t0)/steps
    evolution=work.evolution
    t=t0
    for _ in 1:steps
        _evolution_action!(evolution.k1,plan.generator,state,t,nothing,evolution)
        @. evolution.tmp=state+(h/2)*evolution.k1
        _evolution_action!(evolution.k2,plan.generator,evolution.tmp,t+h/2,
                           nothing,evolution)
        @. evolution.tmp=state+(h/2)*evolution.k2
        _evolution_action!(evolution.k3,plan.generator,evolution.tmp,t+h/2,
                           nothing,evolution)
        @. evolution.tmp=state+h*evolution.k3
        _evolution_action!(evolution.k4,plan.generator,evolution.tmp,t+h,
                           nothing,evolution)
        @. state=state+(h/6)*(evolution.k1+2evolution.k2+
                              2evolution.k3+evolution.k4)
        t+=h
    end
    state
end

"""
    two_time_correlation!(destination, plan, rho, delays;
                          steps_per_interval=64, workspace=nothing)

Evaluate `tr(A * exp(L*tau) * (B*rho*R))` in the exact PI representation at
ordered nonnegative `delays`, using the quantum-regression theorem and one
preallocated numerical RK4 evolution workspace.  No Liouvillian matrix is
materialized.  `destination` may be a
complex vector of any scalar type that can represent the plan output.
A delay vector in the plan's real precision is reused without allocation;
narrower inputs are converted once so integration still uses the plan
precision. Wider floating inputs and nonrepresentable integers raise instead
of narrowing silently. Plans are autonomous by construction, so every RK4
stage deliberately uses `parameters=nothing`.
"""
function two_time_correlation!(destination::AbstractVector,
        plan::CorrelationPlan,rho::PIState,delays;
        steps_per_interval::Integer=64,workspace=nothing)
    _check_correlation_state(plan,rho)
    steps_per_interval>0||throw(ArgumentError(
        "steps_per_interval must be positive"))
    ts=_checked_correlation_delays(plan,delays)
    length(destination)==length(ts)||throw(DimensionMismatch(
        "destination and delay grid lengths differ"))
    promote_type(eltype(destination),eltype(plan))===eltype(destination)||
        throw(ArgumentError("correlation destination cannot represent the plan scalar type"))
    isempty(ts)&&return destination
    work=workspace===nothing ? CorrelationWorkspace(plan) : workspace
    _check_correlation_workspace(work,plan)
    _correlation_seed!(work.state,plan,rho,work)
    previous=zero(first(ts))
    for index in eachindex(ts)
        delay=ts[index]
        if delay!=previous
            _correlation_evolve_interval!(work.state,plan,previous,delay,
                                          steps_per_interval,work)
        end
        destination[index]=_correlation_readout(plan,work.state)
        previous=delay
    end
    destination
end

"""
    two_time_correlation(plan, rho, delays; kwargs...)
    two_time_correlation(L, rho, A, B, delays; right=nothing, kwargs...)

Return the quantum-regression correlation in the exact PI representation,
`tr(A * exp(L*tau) * (B*rho*R))`.  The convenience form prepares a plan;
reuse a `CorrelationPlan` and `CorrelationWorkspace` for repeated calls.
"""
function two_time_correlation(plan::CorrelationPlan,rho::PIState,delays;kwargs...)
    ts=_checked_correlation_delays(plan,delays)
    output=Vector{eltype(plan)}(undef,length(ts))
    two_time_correlation!(output,plan,rho,ts;kwargs...)
end
function two_time_correlation(L,rho::PIState,A::PIOperator,B::PIOperator,
                              delays;right=nothing,kwargs...)
    plan=CorrelationPlan(L,A,B;right=right)
    two_time_correlation(plan,rho,delays;kwargs...)
end

function _correlation_intensity(rho::PIState,number::PIOperator)
    # number is Hermitian for c'c, but use the explicit standard trace rather
    # than relying on that fact so the convention remains visible.
    dot(adjoint(number).data,rho.data)
end

"""
    delayed_second_order_correlation(L, rho, c, delays;
                                     normalized=true, kwargs...)

Compute

```math
G^{(2)}(\\tau)=\\mathrm{tr}\\!\\left[c^\\dagger c\\,
e^{\\mathcal L\\tau}(c\\rho c^\\dagger)\\right]
```

and, by default, return `g2 = G2 / tr(c'c*rho)^2`.  This normalization is the
stationary one, so `normalized=true` verifies that `rho` is trace one and
stationary.  A zero stationary intensity makes the normalized quantity
undefined and raises `DomainError`; the unnormalized, possibly nonstationary
function remains available with `normalized=false`.
"""
function delayed_second_order_correlation(L,rho::PIState,c::PIOperator,delays;
        normalized::Bool=true,steps_per_interval::Integer=64,
        stationarity_atol=nothing,stationarity_rtol=nothing)
    number=adjoint(c)*c
    plan=CorrelationPlan(L,number,c;right=adjoint(c))
    work=CorrelationWorkspace(plan)
    if normalized
        R=_real_float_type(eltype(plan))
        atol=stationarity_atol===nothing ? R(100)*eps(R) :
            _correlation_real_input(R,stationarity_atol,
                                    "stationarity absolute tolerance")
        rtol=stationarity_rtol===nothing ? sqrt(eps(R)) :
            _correlation_real_input(R,stationarity_rtol,
                                    "stationarity relative tolerance")
        atol>=zero(R)&&rtol>=zero(R)||throw(ArgumentError(
            "stationarity tolerances must be nonnegative"))
        _check_stationary_correlation_state!(work,plan,rho,atol,rtol)
    end
    values=two_time_correlation(plan,rho,delays;
        steps_per_interval=steps_per_interval,workspace=work)
    normalized||return values
    raw_intensity=_correlation_intensity(rho,number)
    scale=max(abs(raw_intensity),one(R))
    abs(imag(raw_intensity))<=atol+rtol*scale||throw(ArgumentError(
        "the stationary intensity is appreciably complex: $raw_intensity"))
    intensity=real(raw_intensity)
    intensity>zero(R)||throw(DomainError(intensity,
        "normalized delayed g2 requires a positive stationary intensity"))
    values./(intensity*intensity)
end

"""Alias for [`delayed_second_order_correlation`](@ref)."""
second_order_correlation(args...;kwargs...)=
    delayed_second_order_correlation(args...;kwargs...)

function _stationary_correlation_seed!(work,plan,rho,connected)
    _correlation_seed!(work.rhs,plan,rho,work)
    baseline=zero(eltype(plan))
    if connected
        seed_trace=dot(plan.tracevec,work.rhs)
        detector_mean=_correlation_readout(plan,rho.data)
        baseline=detector_mean*seed_trace
        @. work.rhs=work.rhs-seed_trace*rho.data
    end
    baseline
end

function _check_stationary_correlation_state!(work,plan,rho,atol,rtol)
    validate_state(rho;atol=atol,rtol=rtol)
    z=dot(plan.tracevec,rho.data)
    abs(z-one(z))<=atol+rtol||throw(ArgumentError(
        "stationary correlation spectra require a trace-one state; trace=$z"))
    _evolution_action!(work.product,plan.generator,rho.data,zero(atol),nothing,
                       work.evolution)
    residual=norm(work.product)
    tolerance=atol+rtol*max(norm(rho.data),one(atol))
    residual<=tolerance||throw(ArgumentError(
        "the supplied state is not stationary within tolerance: residual=$residual, tolerance=$tolerance"))
    residual
end

struct _CorrelationShiftedOperator{P,W,V,R}
    plan::P
    work::W
    stationary_state::V
    omega::R
end


function (operator::_CorrelationShiftedOperator)(destination,source)
    plan=operator.plan;work=operator.work
    R=typeof(operator.omega)
    _evolution_action!(destination,plan.generator,source,zero(R),nothing,
                       work.evolution)
    trace_component=dot(plan.tracevec,source)
    @. destination=im*operator.omega*source-destination+
        trace_component*operator.stationary_state
    destination
end

function _correlation_shifted_solve!(work::CorrelationWorkspace,
        plan::CorrelationPlan,stationary_state,omega,atol,rtol,maxiter)
    fill!(work.solution,zero(eltype(work.solution)))
    shifted=_CorrelationShiftedOperator(plan,work,stationary_state,omega)
    _gmres!(work.solution,shifted,work.rhs,work.krylov;
            atol=atol,rtol=rtol,maxiter=maxiter)
end

"""
    stationary_correlation_spectrum(plan, rho, frequencies;
        connected=true, krylovdim=nothing, maxiter=500, atol=nothing,
        rtol=nothing, workspace=nothing)

Evaluate the one-sided stationary spectrum

```math
S_{AB}(\\omega)=\\int_0^\\infty e^{-i\\omega\\tau}
 C^{\\mathrm{conn}}_{AB}(\\tau)\\,d\\tau
```

with a matrix-free shifted-GMRES solve of
`(im*omega*I - L)x = B*rho*R - rho*tr(B*rho*R)`.  The optional connected
subtraction is exact and removes the stationary delta contribution; no
late-time estimate is used.  A rank-one trace constraint makes the zero
frequency solve nonsingular without changing the trace-zero solution.

Only the connected spectrum is an ordinary function: without subtraction the
stationary component is a Dirac delta distribution.  Consequently
`connected=false` is rejected instead of returning a misleading regularized
number.

The return value is a named tuple containing `frequencies`, complex `values`,
GMRES diagnostics, `connected`, and `convention=:one_sided_exp_minus_iomega_t`.
No factor `2` or real part is applied; users needing a conventional two-sided
Hermitian spectrum may form `2real.(values)`.
Frequencies and explicit tolerances must be representable without narrowing
in the plan's real precision; nonrepresentable integer frequencies also
raise.
"""
function stationary_correlation_spectrum(plan::CorrelationPlan,rho::PIState,
        frequencies;connected::Bool=true,krylovdim=nothing,
        maxiter::Integer=500,atol=nothing,rtol=nothing,workspace=nothing)
    _check_correlation_state(plan,rho)
    _require_autonomous(plan.generator,"stationary correlation spectra")
    omegas=frequencies isa AbstractVector ? frequencies : collect(frequencies)
    all(w->w isa Real&&isfinite(w),omegas)||throw(ArgumentError(
        "spectrum frequencies must be finite real values"))
    connected||throw(ArgumentError(
        "the disconnected stationary spectrum contains a Dirac delta; use connected=true"))
    R=_real_float_type(eltype(plan))
    atolR=atol===nothing ? R(100)*eps(R) :
        _correlation_real_input(R,atol,"spectrum absolute tolerance")
    rtolR=rtol===nothing ? sqrt(eps(R)) :
        _correlation_real_input(R,rtol,"spectrum relative tolerance")
    atolR>=zero(R)&&rtolR>=zero(R)||throw(ArgumentError(
        "spectrum tolerances must be nonnegative"))
    maxiter>0||throw(ArgumentError("maxiter must be positive"))
    if krylovdim!==nothing
        krylovdim isa Integer&&krylovdim>0||throw(ArgumentError(
            "krylovdim must be a positive integer or nothing"))
    end
    work=workspace===nothing ? CorrelationWorkspace(plan;
        krylovdim=krylovdim===nothing ? 30 : krylovdim) : workspace
    _check_correlation_workspace(work,plan)
    if workspace!==nothing&&krylovdim!==nothing
        size(work.krylov.H,2)>=min(krylovdim,length(plan.basis))||
            throw(ArgumentError("correlation workspace Krylov dimension is too small"))
    end
    stationary_residual=_check_stationary_correlation_state!(
        work,plan,rho,atolR,rtolR)
    baseline=_stationary_correlation_seed!(work,plan,rho,connected)
    values=Vector{eltype(plan)}(undef,length(omegas))
    diagnostics=Vector{NamedTuple}(undef,length(omegas))
    for index in eachindex(omegas)
        omega=_correlation_real_input(R,omegas[index],"spectrum frequency")
        result=_correlation_shifted_solve!(
            work,plan,rho.data,omega,atolR,rtolR,maxiter)
        result.converged||throw(ArgumentError(
            "shifted GMRES did not converge at frequency $(omegas[index]) " *
            "in $(result.iterations) iterations; residual=$(result.raw_residual)"))
        values[index]=_correlation_readout(plan,work.solution)
        diagnostics[index]=result
    end
    (;frequencies=omegas,values,diagnostics,connected,baseline,
      stationary_residual,convention=:one_sided_exp_minus_iomega_t,
      method=:matrixfree_shifted_gmres)
end

function stationary_correlation_spectrum(L,rho::PIState,A::PIOperator,
        B::PIOperator,frequencies;right=nothing,kwargs...)
    plan=CorrelationPlan(L,A,B;right=right)
    stationary_correlation_spectrum(plan,rho,frequencies;kwargs...)
end

function _correlation_fft!(data::AbstractVector{Complex{R}}) where
        R<:AbstractFloat
    n=length(data)
    n>0&&(n&(n-1))==0||throw(ArgumentError(
        "FFT length must be a positive power of two"))
    # In-place bit-reversal permutation, using zero-based indices for the
    # standard radix-two update and converting only at array access.
    reversed=0
    for original in 0:n-1
        original<reversed&&begin
            i=original+1;j=reversed+1
            data[i],data[j]=data[j],data[i]
        end
        bit=n>>1
        while bit>0&&(reversed&bit)!=0
            reversed⊻=bit
            bit>>=1
        end
        reversed⊻=bit
    end
    width=2
    while width<=n
        root=cis(-R(2)*R(pi)/R(width))
        half=width>>1
        for start in 1:width:n
            phase=one(Complex{R})
            @inbounds for offset in 0:half-1
                even=data[start+offset]
                odd=phase*data[start+offset+half]
                data[start+offset]=even+odd
                data[start+offset+half]=even-odd
                phase*=root
            end
        end
        width<<=1
    end
    data
end

function _correlation_fft_real_type(delays,values,offset)
    R=_real_float_type(eltype(values))
    T=eltype(delays)
    T<:Number&&(R=promote_type(R,_real_float_type(T)))
    R=promote_type(R,_real_float_type(typeof(offset)))
    R
end

"""
    correlation_spectrum_fft(delays, correlations;
                             offset=0, nfft=nothing)

Compute the finite-time one-sided transform

```math
S_T(\\omega)=\\int_0^T e^{-i\\omega\\tau}
    [C(\\tau)-C_\\infty]\\,d\\tau
```

from a uniform delay grid using trapezoidal endpoint weights and an in-place
radix-two FFT implemented without an additional dependency.  `offset` is the
known `C_infinity`; it is never estimated from the final sample.  `nfft`
defaults to the next power of two and may request additional zero padding.

The returned frequencies are angular frequencies in ascending order.  The
complex values use `exp(-im*omega*tau)`, contain no factor two or real-part
projection, and approximate a finite-window integral rather than the
infinite-time resolvent.
"""
function correlation_spectrum_fft(delays,correlations;offset=0,nfft=nothing)
    ts=collect(delays);values=collect(correlations)
    length(ts)==length(values)||throw(DimensionMismatch(
        "delay and correlation vectors have different lengths"))
    length(ts)>=2||throw(ArgumentError(
        "at least two samples are required for an FFT spectrum"))
    all(t->t isa Real&&isfinite(t),ts)||throw(ArgumentError(
        "FFT delays must be finite real values"))
    all(isfinite,values)||throw(ArgumentError(
        "FFT correlations must be finite"))
    R=_correlation_fft_real_type(ts,values,offset)
    converted_times=R.(ts)
    iszero(converted_times[1])||throw(ArgumentError(
        "the FFT delay grid must start at zero"))
    dt=converted_times[2]-converted_times[1]
    dt>zero(R)||throw(ArgumentError(
        "the FFT delay grid must be strictly increasing"))
    grid_tolerance=R(64)*eps(R)*max(abs(converted_times[end]),dt,one(R))
    for index in 3:length(converted_times)
        step=converted_times[index]-converted_times[index-1]
        abs(step-dt)<=grid_tolerance||throw(ArgumentError(
            "the FFT delay grid must be uniform"))
    end
    requested=nfft===nothing ? nextpow(2,length(values)) : begin
        nfft isa Integer||throw(ArgumentError("nfft must be an integer or nothing"))
        Int(nfft)
    end
    requested>=length(values)||throw(ArgumentError(
        "nfft must be at least the number of samples"))
    requested>0&&(requested&(requested-1))==0||throw(ArgumentError(
        "nfft must be a power of two"))
    transformed=zeros(Complex{R},requested)
    offsetR=Complex{R}(offset)
    @inbounds for index in eachindex(values)
        transformed[index]=Complex{R}(values[index])-offsetR
    end
    half=inv(R(2))
    transformed[1]*=half
    transformed[length(values)]*=half
    _correlation_fft!(transformed)
    transformed .*= dt

    # Map the raw DFT indices to signed angular-frequency bins, assigning the
    # even-length Nyquist point to the negative edge for a unique sorted grid.
    signed_bins=Vector{Int}(undef,requested)
    for raw in 0:requested-1
        signed_bins[raw+1]=raw<cld(requested,2) ? raw : raw-requested
    end
    order=sortperm(signed_bins)
    frequencies=R(2)*R(pi).*(R.(signed_bins[order]))./(R(requested)*dt)
    (;frequencies,values=transformed[order],offset=offsetR,
      duration=converted_times[end],sample_spacing=dt,nfft=requested,
      convention=:finite_one_sided_exp_minus_iomega_t,
      quadrature=:trapezoidal,method=:radix2_fft)
end

"""
    correlation_spectrum_fft(plan, rho, delays; connected=true,
                             workspace=nothing, kwargs...)

Propagate a stationary QRT correlation on `delays` and transform it with
[`correlation_spectrum_fft`](@ref).  With `connected=true`, the exact
stationary product `tr(A*rho)*tr(B*rho*R)` is subtracted before the FFT.
`rho` is checked for unit trace and stationarity; no tail sample is used as a
surrogate for the infinite-time value.
"""
function correlation_spectrum_fft(plan::CorrelationPlan,rho::PIState,delays;
        connected::Bool=true,workspace=nothing,steps_per_interval::Integer=64,
        stationarity_atol=nothing,stationarity_rtol=nothing,nfft=nothing)
    _check_correlation_state(plan,rho)
    work=workspace===nothing ? CorrelationWorkspace(plan) : workspace
    _check_correlation_workspace(work,plan)
    R=_real_float_type(eltype(plan))
    atol=stationarity_atol===nothing ? R(100)*eps(R) :
        _correlation_real_input(R,stationarity_atol,
                                "stationarity absolute tolerance")
    rtol=stationarity_rtol===nothing ? sqrt(eps(R)) :
        _correlation_real_input(R,stationarity_rtol,
                                "stationarity relative tolerance")
    atol>=zero(R)&&rtol>=zero(R)||throw(ArgumentError(
        "stationarity tolerances must be nonnegative"))
    _check_stationary_correlation_state!(work,plan,rho,atol,rtol)
    ts=_checked_correlation_delays(plan,delays)
    values=two_time_correlation(plan,rho,ts;
        steps_per_interval=steps_per_interval,workspace=work)
    _correlation_seed!(work.rhs,plan,rho,work)
    offset=connected ? _correlation_readout(plan,rho.data)*
        dot(plan.tracevec,work.rhs) : zero(eltype(plan))
    result=correlation_spectrum_fft(ts,values;offset=offset,nfft=nfft)
    merge(result,(connected=connected,))
end

"""
    optical_spectrum(L, rho, c, frequencies; kwargs...)
    optical_spectrum(plan, rho, frequencies; kwargs...)

Convenience stationary emission spectrum using
`A=adjoint(c)`, `B=c`.  The result follows the one-sided
`exp(-im*omega*tau)` convention of [`stationary_correlation_spectrum`](@ref).
The plan overload permits reuse of an optical `CorrelationPlan` and its
workspace without constructing another plan.
"""
function optical_spectrum(L,rho::PIState,c::PIOperator,frequencies;kwargs...)
    stationary_correlation_spectrum(L,rho,adjoint(c),c,frequencies;kwargs...)
end
function optical_spectrum(plan::CorrelationPlan,rho::PIState,frequencies;kwargs...)
    stationary_correlation_spectrum(plan,rho,frequencies;kwargs...)
end
