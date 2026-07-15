"""Common supertype for dispatchable high-level solver choices."""
abstract type AbstractPIAlgorithm end

"""Select a conservative algorithm from the problem representation and size."""
struct AutoAlgorithm <: AbstractPIAlgorithm end
"""Trace-bordered direct stationary-state solve."""
struct DirectAlgorithm <: AbstractPIAlgorithm end
"""Dense SVD stationary-manifold solve for diagnostic-sized problems."""
struct SVDAlgorithm <: AbstractPIAlgorithm end
"""Dense eigenvector stationary-state solve."""
struct EigenAlgorithm <: AbstractPIAlgorithm end

"""Sparse shift-invert stationary-state iteration near an explicit or automatic shift."""
struct ShiftInvertAlgorithm{T} <: AbstractPIAlgorithm
    shift::T
    maxiter::Int
end
ShiftInvertAlgorithm(;shift=nothing,maxiter::Integer=200)=
    ShiftInvertAlgorithm(shift,Int(maxiter))

"""Restarted matrix-free GMRES stationary-state algorithm and optional preconditioner."""
struct GMRESAlgorithm{P} <: AbstractPIAlgorithm
    krylovdim::Int
    maxiter::Int
    preconditioner::P
end
GMRESAlgorithm(;krylovdim::Integer=30,maxiter::Integer=500,
               preconditioner=nothing)=
    GMRESAlgorithm(Int(krylovdim),Int(maxiter),preconditioner)

"""Thick-restarted harmonic Arnoldi parameters for modes near zero."""
struct HarmonicArnoldiAlgorithm <: AbstractPIAlgorithm
    nev::Int
    krylovdim::Int
    thickdim::Int
    maxrestarts::Int
end
HarmonicArnoldiAlgorithm(;nev::Integer=6,
    krylovdim::Integer=max(30,3Int(nev)+6),
    thickdim::Integer=max(Int(nev)+2,2Int(nev)),
    maxrestarts::Integer=20)=HarmonicArnoldiAlgorithm(
        Int(nev),Int(krylovdim),Int(thickdim),Int(maxrestarts))

"""Typed high-level stationary-state result."""
struct SteadyStateResult{S,I,A}
    state::S
    info::I
    algorithm::A
end

"""Fixed-step dynamics result with collection semantics over saved states."""
struct DynamicsResult{T,S,A}
    times::Vector{T}
    states::S
    algorithm::A
end
Base.length(sol::DynamicsResult)=length(sol.states)
Base.getindex(sol::DynamicsResult,i::Integer)=sol.states[i]
Base.firstindex(sol::DynamicsResult)=firstindex(sol.states)
Base.lastindex(sol::DynamicsResult)=lastindex(sol.states)
Base.iterate(sol::DynamicsResult,args...)=iterate(sol.states,args...)
state(sol::DynamicsResult,i::Integer)=sol.states[i]
state(sol::DynamicsResult,t::Real)=begin
    i=findmin(abs.(sol.times.-t))[2]
    isapprox(sol.times[i],t)||throw(ArgumentError("time $t was not saved"))
    sol.states[i]
end

"""Typed selected-spectrum result."""
struct SpectrumResult{V,W,I}
    values::V
    vectors::W
    info::I
end

Base.length(sol::PISolution)=length(sol.raw.u)
Base.getindex(sol::PISolution,i::Integer)=state(sol,i)
Base.firstindex(sol::PISolution)=firstindex(sol.raw.u)
Base.lastindex(sol::PISolution)=lastindex(sol.raw.u)
Base.iterate(sol::PISolution,state_index::Int=1)=
    state_index>length(sol) ? nothing : (state(sol,state_index),state_index+1)

function show(io::IO,rho::PIState)
    print(io,"PIState(N=$(rho.basis.N), d=$(rho.basis.d), dimension=$(length(rho.data)), trace=$(trace(rho)))")
end
function show(io::IO,A::PIOperator)
    print(io,"PIOperator(N=$(A.basis.N), d=$(A.basis.d), dimension=$(length(A.data)))")
end
function show(io::IO,model::PIModel)
    print(io,"PIModel(N=$(model.basis.N), d=$(model.basis.d), dimension=$(length(model.basis)), terms=$(length(model.terms)), autonomous=$(isautonomous(model)))")
end
function show(io::IO,prepared::CompiledPIModel)
    print(io,"CompiledPIModel(N=$(prepared.model.basis.N), d=$(prepared.model.basis.d), dimension=$(size(prepared,1)), backend=$(prepared.backend), autonomous=$(isautonomous(prepared)))")
end
function show(io::IO,plan::LiouvillianPlan)
    print(io,"LiouvillianPlan(N=$(plan.basis.N), d=$(plan.basis.d), dimension=$(size(plan,1)), kernels=$(plan.kernels===nothing ? 0 : length(plan.kernels)), autonomous=$(isautonomous(plan)))")
end
function show(io::IO,L::MatrixFreeLiouvillian)
    print(io,"MatrixFreeLiouvillian(dimension=$(size(L,1)), autonomous=$(isautonomous(L)), compiled=$(L.plan!==nothing))")
end
function show(io::IO,result::SteadyStateResult)
    print(io,"SteadyStateResult(method=$(result.info.method), residual=$(result.info.residual), trace_error=$(result.info.trace_error))")
end
function show(io::IO,result::DynamicsResult)
    print(io,"DynamicsResult($(length(result)) states, t=$(first(result.times))…$(last(result.times)), algorithm=$(result.algorithm))")
end
function show(io::IO,result::SpectrumResult)
    print(io,"SpectrumResult($(length(result.values)) values)")
end

function _algorithm_options(algorithm)
    algorithm isa Symbol && return (algorithm,NamedTuple())
    algorithm isa AutoAlgorithm && return (:auto,NamedTuple())
    algorithm isa DirectAlgorithm && return (:direct,NamedTuple())
    algorithm isa SVDAlgorithm && return (:svd,NamedTuple())
    algorithm isa EigenAlgorithm && return (:eigen,NamedTuple())
    algorithm isa ShiftInvertAlgorithm && return (:shiftinvert,
        (;shift=algorithm.shift,maxiter=algorithm.maxiter))
    algorithm isa GMRESAlgorithm && return (:krylov,
        (;krylovdim=algorithm.krylovdim,maxiter=algorithm.maxiter,
          preconditioner=algorithm.preconditioner))
    throw(ArgumentError("unsupported stationary-state algorithm $(typeof(algorithm))"))
end

function _basis_metadata(x,basis)
    basis!==nothing&&return basis
    x isa PIBasis&&return x
    x isa PIModel&&return x.basis
    hasproperty(x,:basis)&&getproperty(x,:basis) isa PIBasis&&return getproperty(x,:basis)
    if hasproperty(x,:model)
        m=getproperty(x,:model)
        m isa PIModel&&return m.basis
    end
    if hasproperty(x,:plan)
        p=getproperty(x,:plan)
        hasproperty(p,:basis)&&return getproperty(p,:basis)
    end
    nothing
end

"""
    stationary_state(x; algorithm=AutoAlgorithm(), basis=nothing,
                     return_info=false, kwargs...)

High-level stationary-state command. Model and compiled-model inputs return a
`PIState`; `return_info=true` returns a `SteadyStateResult`. The existing
`steady_state` function remains the low-level coordinate-vector interface.
"""
function stationary_state(x;algorithm=AutoAlgorithm(),basis=nothing,
                          return_info::Bool=false,kwargs...)
    method,options=_algorithm_options(algorithm)
    b=_basis_metadata(x,basis)
    b===nothing&&throw(ArgumentError("stationary_state requires PI basis metadata"))
    info = if x isa Union{PIModel,CompiledPIModel}
        steady_state(x;method=method,return_info=true,options...,kwargs...)
    else
        steady_state(x;basis=b,method=method,return_info=true,options...,kwargs...)
    end
    rho=PIState(b,info.state)
    result=SteadyStateResult(rho,info,algorithm)
    return_info ? result : rho
end

function _saved_times(tspan,saveat)
    t0,t1=tspan;t1>=t0||throw(ArgumentError("tspan must be ordered"))
    saveat===nothing&&return [float(t0),float(t1)]
    if saveat isa Real
        saveat>0||throw(ArgumentError("saveat must be positive"))
        ts=collect(float(t0):float(saveat):float(t1))
        (isempty(ts)||ts[end]<t1)&&push!(ts,float(t1))
        return ts
    end
    ts=float.(collect(saveat));isempty(ts)&&throw(ArgumentError("saveat cannot be empty"))
    first(ts)==t0&&last(ts)==t1||throw(ArgumentError("explicit saveat times must include both endpoints of tspan"))
    all(diff(ts).>=0)||throw(ArgumentError("saveat times must be nondecreasing"))
    ts
end

"""
    solve_dynamics(x, rho0, tspan; saveat=nothing,
                   steps_per_interval=64, parameters=nothing)

Compile a model once when needed and propagate with the allocation-conscious
fixed-step RK4 path. The result carries saved times and PI states and supports
indexing and iteration. Use `dynamics_problem` directly for adaptive SciML
algorithms.
"""
function solve_dynamics(x,rho0::PIState,tspan;saveat=nothing,
                        steps_per_interval::Integer=64,parameters=nothing)
    steps_per_interval>0||throw(ArgumentError("steps_per_interval must be positive"))
    ts=_saved_times(tspan,saveat)
    source = x isa PIModel && isdefined(@__MODULE__,:compile) ?
        getfield(@__MODULE__,:compile)(x;backend=:matrixfree) : x
    states=time_evolution(source,rho0,ts;steps_per_interval=steps_per_interval,
                          parameters=parameters)
    DynamicsResult(ts,states,:rk4)
end

function _spectrum_algorithm(algorithm,target,n,nev)
    if algorithm isa HarmonicArnoldiAlgorithm
        return (:harmonic,(;nev=algorithm.nev,krylovdim=algorithm.krylovdim,
            thickdim=algorithm.thickdim,maxrestarts=algorithm.maxrestarts))
    elseif algorithm isa Symbol && algorithm!==:auto
        return (algorithm,(;nev=Int(nev)))
    elseif algorithm isa AutoAlgorithm || algorithm===:auto
        method=target===:near_zero ? :harmonic : n<=256 ? :dense : :krylov
        return (method,(;nev=Int(nev)))
    end
    throw(ArgumentError("unsupported spectrum algorithm $(typeof(algorithm))"))
end

"""
    liouvillian_spectrum(x; target=:largest_real, nev=6,
                         algorithm=:auto, vectors=false, return_info=false)

Consistent high-level spectral command. `target` is one of `:largest_real`,
`:near_zero`, or `:largest_magnitude`; method-specific `sortby`/`which`
dialects remain available through the lower-level spectral functions.
"""
function liouvillian_spectrum(x;target=:largest_real,nev::Integer=6,
                              algorithm=:auto,vectors::Bool=false,
                              return_info::Bool=false,kwargs...)
    target in (:largest_real,:near_zero,:largest_magnitude)||
        throw(ArgumentError("target must be :largest_real, :near_zero, or :largest_magnitude"))
    nev>0||throw(ArgumentError("nev must be positive"))
    source=x
    n=pi_dimension(source);method,options=_spectrum_algorithm(algorithm,target,n,nev)
    method in (:jd,:jacobi_davidson)&&target!==:near_zero&&throw(ArgumentError(
        "Jacobi--Davidson is a near-target solver; use target=:near_zero or call jacobi_davidson_spectrum with a numeric target"))
    sortby=target===:largest_real ? :real : :magnitude
    rev=target!==:near_zero
    want_vectors=vectors||return_info
    raw=pi_liouvillian_spectrum(source;method=method,sortby=sortby,rev=rev,
                                vectors=want_vectors,options...,kwargs...)
    if !want_vectors
        return raw[1:min(Int(nev),length(raw))]
    end
    take=1:min(Int(nev),length(raw.values))
    values=raw.values[take];vecs=vectors ? raw.vectors[:,take] : nothing
    info=Base.structdiff(raw,(values=raw.values,vectors=raw.vectors))
    result=SpectrumResult(values,vecs,info)
    return_info ? result : (vectors ? (values=values,vectors=vecs) : values)
end

"""Return the PI-coordinate dimension of a basis, state, model, or operator."""
pi_dimension(b::PIBasis)=length(b)
pi_dimension(x::AbstractPIOperator)=length(x.data)
pi_dimension(x::PIModel)=length(x.basis)
pi_dimension(x)=size(x,1)

"""
    estimate_state_bytes(x; T=ComplexF64,
                         bigfloat_precision=precision(BigFloat))

Estimated retained bytes occupied by one dense PI coordinate vector. Fixed-size
isbits scalar types use the exact inline `sizeof(T)` value. Heap-backed
`BigFloat` scalars use an explicitly conservative per-element bound at
`bigfloat_precision`; pass the largest intended precision when it differs from
the active process precision. Other heap-backed scalar types use a padded
zero-value sample because their payload may not be bounded by their type; this
route is an estimate, not a worst-case guarantee.
"""
estimate_state_bytes(b::PIBasis;T=ComplexF64,
                     bigfloat_precision::Integer=precision(BigFloat))=
    big(length(b))*_scalar_retained_bytes(T;bigfloat_precision)
estimate_state_bytes(x;T=ComplexF64,
                     bigfloat_precision::Integer=precision(BigFloat))=
    big(pi_dimension(x))*_scalar_retained_bytes(T;bigfloat_precision)

"""Retained Julia heap size of an already constructed basis or plan."""
estimate_basis_bytes(b::PIBasis)=Base.summarysize(b)
"""Return the retained Julia heap size in bytes of an assembled or matrix-free Liouvillian object."""
estimate_liouvillian_bytes(L)=Base.summarysize(L)

"""
    estimate_geometry_bytes(basis; T=Float64,
                            bigfloat_precision=precision(BigFloat))

Return a conservative structural memory estimate for construction and
retention of the shared one-body Schur geometry.  The estimate performs no CG
evaluation and reports exact `BigInt` byte counts.  `setup_bytes` is a peak
live-storage upper bound for the sparse transition constructor, whereas
`retained_bytes` describes the resulting read-only cache.  Allocator metadata
and garbage-collector timing remain platform dependent, so benchmark a
representative basis before a long scan. Fixed-size isbits geometry preserves
the historical byte formula; heap-backed BigFloat tuple entries are inflated
with the shared conservative bound at `bigfloat_precision`.
"""
function estimate_geometry_bytes(b::PIBasis;T=Float64,
                                 bigfloat_precision::Integer=precision(BigFloat))
    R=_real_float_type(T)
    R<:AbstractFloat||throw(ArgumentError(
        "geometry scalar type must promote to an AbstractFloat type"))
    raw=_estimate_onebody_geometry(b,R;bigfloat_precision)
    retained_scalar=_scalar_retained_bytes(R;bigfloat_precision)
    merge(raw,(;scalar_type=R,scalar_retained_bytes=retained_scalar,
               scalar_storage_estimate=_scalar_storage_estimate(R),
               bigfloat_precision_assumption=
                   _scalar_precision_assumption(R,bigfloat_precision),
               estimate=:conservative_structural_upper_bound))
end

"""
    estimate_solver_bytes(x; algorithm=:gmres, krylovdim=30, T=ComplexF64,
                          bigfloat_precision=precision(BigFloat))

Estimate solver work-array storage, excluding operator and factorization data.
Fixed-size isbits values retain exact inline byte accounting; heap-backed
BigFloat values use the same conservative precision-aware bound as
[`estimate_state_bytes`](@ref). GMRES real residual/history storage follows the
real component type of `T` rather than assuming `Float64`.
"""
function estimate_solver_bytes(x;algorithm=:gmres,krylovdim::Integer=30,
                               T=ComplexF64,
                               bigfloat_precision::Integer=precision(BigFloat))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    ni=pi_dimension(x);mi=Int(min(big(ni),big(krylovdim)));n=big(ni);m=big(mi)
    scalar_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    real_bytes=_scalar_retained_bytes(_real_float_type(T);bigfloat_precision)
    algorithm in (:gmres,:krylov)&&return scalar_bytes*(n*(m+6)+(m+1)*m+2m+1)+real_bytes*m
    algorithm in (:arnoldi,:harmonic,:iram,:implicit_qr)&&return scalar_bytes*(2n*m+n+3m*m)
    # Arnoldi/JD search arrays plus a same-dimension restarted-GMRES
    # correction workspace and six explicit correction vectors.
    algorithm in (:jd,:jacobi_davidson)&&return scalar_bytes*(6n*m+14n+4m*m+2m)
    algorithm in (:dense,:direct,:svd)&&return scalar_bytes*n*n
    algorithm in (:rk4,:dynamics)&&return 5scalar_bytes*n
    throw(ArgumentError("unknown solver-memory algorithm $algorithm"))
end

function _recommended_geometry_policy(x,basis)
    basis===nothing&&return (include=false,requirement=:unavailable,
                             source=:no_basis_metadata)
    model=x isa PIModel ? x : x isa CompiledPIModel ? x.model : nothing
    if model!==nothing
        required=any(_term_requires_onebody_geometry,model.terms)
        return (include=required,
                requirement=required ? :required : :not_required,
                source=:model_terms)
    end
    # A bare basis, state, operator, or lowered plan no longer carries enough
    # term provenance to distinguish local one-body lowering from direct/p-body
    # blocks. Retain the conservative historical geometry allowance and make
    # that assumption explicit in the returned metadata.
    (include=true,requirement=:conservative_unknown,
     source=x isa PIBasis ? :basis_only : :source_without_term_provenance)
end

"""
    recommend_solver(x; task=:steady_state, memory_budget=512*1024^2,
                     bigfloat_precision=precision(BigFloat))

Return a transparent heuristic recommendation. This performs no assembly and
reports its assumptions; `compile(...; backend=:auto)` makes the final backend
choice from the lowered plan. Model and `CompiledPIModel` inputs include
one-body geometry only when their terms require it. Inputs carrying only basis
metadata retain a conservative geometry allowance, identified by
`geometry_requirement=:conservative_unknown`.
"""
function recommend_solver(x;task=:steady_state,memory_budget::Integer=512*1024^2,
                          krylovdim::Integer=30,T=ComplexF64,
                          bigfloat_precision::Integer=precision(BigFloat))
    memory_budget>0||throw(ArgumentError("memory_budget must be positive"))
    n=pi_dimension(x);autonomous=applicable(isautonomous,x) ? isautonomous(x) : true
    scalar_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    dense_bytes=scalar_bytes*big(n)^2
    gmres_bytes=estimate_solver_bytes(x;algorithm=:gmres,
        krylovdim=krylovdim,T=T,bigfloat_precision)
    arnoldi_bytes=estimate_solver_bytes(x;algorithm=:arnoldi,
        krylovdim=krylovdim,T=T,bigfloat_precision)
    dynamics_bytes=estimate_solver_bytes(x;algorithm=:rk4,T=T,bigfloat_precision)
    task in (:steady_state,:spectrum,:dynamics)||throw(ArgumentError("task must be :steady_state, :spectrum, or :dynamics"))
    backend=(!autonomous||dense_bytes>memory_budget||n>512) ? :matrixfree : :sparse
    algorithm = task===:dynamics ? (autonomous&&backend===:sparse ? :exponential_or_adaptive : :adaptive_or_rk4) :
                task===:steady_state ? (backend===:sparse ? :direct : :gmres) :
                (backend===:sparse ? :dense : :arnoldi)
    reason=!autonomous ? "time-dependent generators require explicit-time matrix-free application" :
           dense_bytes>memory_budget ? "dense PI storage exceeds the requested memory budget" :
           n>512 ? "PI dimension exceeds the conservative sparse/direct crossover" :
           "PI dimension is below the conservative sparse/direct crossover"
    selected_solver_bytes=task===:steady_state ? gmres_bytes :
        task===:spectrum ? arnoldi_bytes : dynamics_bytes
    basis=_basis_metadata(x,nothing)
    geometry_policy=_recommended_geometry_policy(x,basis)
    geometry=geometry_policy.include ? estimate_geometry_bytes(basis;
        T=_real_float_type(T),bigfloat_precision) : nothing
    geometry_retained=basis===nothing ? nothing :
        geometry===nothing ? big(0) : geometry.retained_bytes
    geometry_setup=basis===nothing ? nothing :
        geometry===nothing ? big(0) : geometry.setup_bytes
    state_bytes=estimate_state_bytes(x;T=T,bigfloat_precision)
    estimated_peak=geometry_setup===nothing ? nothing :
        geometry_setup+state_bytes+selected_solver_bytes
    fits_memory=estimated_peak===nothing ? missing : estimated_peak<=memory_budget
    if fits_memory===false
        reason *= iszero(geometry_setup) ?
            "; estimated state and solver vectors exceed the requested memory budget" :
            "; estimated geometry setup plus state and solver vectors exceed the requested memory budget"
    end
    (;task,dimension=n,autonomous,backend,algorithm,reason,memory_budget,
      state_bytes,dense_upper_bytes=dense_bytes,
      scalar_retained_bytes=scalar_bytes,
      scalar_storage_estimate=_scalar_storage_estimate(T),
      bigfloat_precision_assumption=_scalar_precision_assumption(
          T,bigfloat_precision),
      krylov_vector_bytes=task===:spectrum ? arnoldi_bytes : gmres_bytes,
      gmres_vector_bytes=gmres_bytes,arnoldi_vector_bytes=arnoldi_bytes,
      dynamics_workspace_bytes=dynamics_bytes,
      selected_solver_bytes,geometry_retained_upper_bytes=geometry_retained,
      geometry_setup_upper_bytes=geometry_setup,
      geometry_requirement=geometry_policy.requirement,
      geometry_assumption_source=geometry_policy.source,
      estimated_peak_bytes=estimated_peak,fits_memory,
      heuristic=true)
end

"""Unified diagnostics for PI states, models, and compiled/linear operators."""
diagnostics(rho::PIState;kwargs...)=state_diagnostics(rho;kwargs...)
function diagnostics(model::PIModel;kwargs...)
    merge(model_summary(model),check_generator(model),
          (;autonomous=isautonomous(model),basis_bytes=estimate_basis_bytes(model.basis),
            state_bytes=estimate_state_bytes(model.basis)))
end
function diagnostics(prepared::CompiledPIModel;kwargs...)
    merge(prepared.estimates,(;backend=prepared.backend,
        autonomous=isautonomous(prepared),retained_bytes=Base.summarysize(prepared)))
end
function diagnostics(plan::LiouvillianPlan;kwargs...)
    (;dimension=size(plan,1),scalar_type=eltype(plan),autonomous=isautonomous(plan),
      kernels=plan.kernels===nothing ? 0 : length(plan.kernels),
      retained_bytes=Base.summarysize(plan))
end
function diagnostics(x;kwargs...)
    (;dimension=pi_dimension(x),autonomous=applicable(isautonomous,x) ? isautonomous(x) : missing,
      retained_bytes=Base.summarysize(x))
end
