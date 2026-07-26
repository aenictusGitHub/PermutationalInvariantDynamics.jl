"""
    ReductionPlanSet

Immutable collection of [`ReductionPlan`](@ref) objects for several particle
bipartitions of one exact `PIBasis`. Setup shares the complete SU(2)
factorial table for qubits and the Littlewood--Richardson generator/intertwiner
caches for qudits. This avoids repeating the dominant representation setup in
entanglement-scaling studies.
"""
struct ReductionPlanSet{B,K,P,E}
    basis::B
    ks::K
    plans::P
    estimates::E
end

function _reduction_set_tolerance(atol::Real)
    isfinite(atol)&&atol>=zero(atol)||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    converted=try
        Float64(atol)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("atol is not representable in Float64"))
    end
    isfinite(converted)||throw(ArgumentError(
        "atol overflows Float64"))
    !iszero(atol)&&iszero(converted)&&throw(ArgumentError(
        "atol underflows to zero in Float64; use a representable tolerance"))
    converted
end

function Base.show(io::IO,set::ReductionPlanSet)
    print(io,"ReductionPlanSet(N=$(set.basis.N), d=$(set.basis.d), ",
          "bipartitions=$(set.ks), retained_bytes=$(set.estimates.retained_bytes))")
end

function _shared_reduction_plan(
        basis::PIBasis{D},k::Int,atol::Real,
        factorials,lr_cache,generator_cache) where D
    output_basis=k==basis.N ? basis : PIBasis(k,D)
    if k==0||k==basis.N
        T=D==2 ? Float64 : ComplexF64
        I=D==2 ? Matrix{Float64} :
            _PackedLRIntertwiner{ComplexF64,Int}
        Coupling=_ProductSchurCoupling{T,D,I}
        couplings=Coupling[]
        estimates=_reduction_plan_estimates(couplings,D)
        return ReductionPlan{T,D,typeof(basis),typeof(output_basis),
                             typeof(couplings)}(
            basis,k,output_basis,couplings,Float64(atol),estimates)
    end
    if D==2
        couplings=_qubit_reduction_couplings(basis,k,factorials)
        estimates=_reduction_plan_estimates(couplings,D)
        return ReductionPlan{Float64,D,typeof(basis),typeof(output_basis),
                             typeof(couplings)}(
            basis,k,output_basis,couplings,Float64(atol),estimates)
    end
    tolerance=max(atol,2e-11)
    couplings=_qudit_reduction_couplings(
        basis,k,tolerance;cache=lr_cache,gencache=generator_cache)
    estimates=_reduction_plan_estimates(couplings,D)
    ReductionPlan{ComplexF64,D,typeof(basis),typeof(output_basis),
                  typeof(couplings)}(
        basis,k,output_basis,couplings,Float64(tolerance),estimates)
end

"""
    ReductionPlanSet(basis, ks=0:N; atol=2e-11)

Prepare unique requested subsystem sizes in the supplied order. Every `k`
must satisfy `0 ≤ k ≤ N`; duplicates are rejected so workspace ownership and
result ordering remain unambiguous.
"""
function ReductionPlanSet(basis::PIBasis{D},ks=0:basis.N;
                          atol::Real=2e-11) where D
    tolerance=_reduction_set_tolerance(atol)
    requested=Int[]
    for value in ks
        value isa Integer&&!(value isa Bool)||throw(ArgumentError(
            "bipartition sizes must be integers"))
        0<=value<=basis.N||throw(ArgumentError(
            "subsystem size k must satisfy 0 ≤ k ≤ N"))
        push!(requested,Int(value))
    end
    isempty(requested)&&throw(ArgumentError(
        "ReductionPlanSet requires at least one subsystem size"))
    allunique(requested)||throw(ArgumentError(
        "ReductionPlanSet subsystem sizes must be unique"))

    factorials=D==2 ? _SU2FactorialCache(basis.N+1) : nothing
    lr_cache=Dict{Tuple{Partition{D},Partition{D},Partition{D}},
                  Vector{Matrix{ComplexF64}}}()
    generator_cache=Dict{Partition{D},Vector{Matrix{ComplexF64}}}()
    plans=map(requested) do k
        _shared_reduction_plan(
            basis,k,tolerance,factorials,lr_cache,generator_cache)
    end
    lr_intertwiner_count=D==2 ? 0 :
        sum(length,values(lr_cache);init=0)
    lr_intertwiner_entries=D==2 ? 0 :
        sum(intertwiners->sum(length,intertwiners;init=0),
            values(lr_cache);init=0)
    estimates=(;
        retained_bytes=BigInt(Base.summarysize(plans)),
        plan_count=length(plans),
        shared_su2_factorials=D==2 ? length(factorials.values) : 0,
        # Keep the original `_entries` field, but make its name truthful.
        # `_count` exposes the former matrix-count statistic compatibly.
        shared_lr_intertwiner_count=lr_intertwiner_count,
        shared_lr_intertwiner_entries=lr_intertwiner_entries,
        shared_generator_sectors=D==2 ? 0 : length(generator_cache))
    _check_reduction_plan_set(ReductionPlanSet(
        basis,Tuple(requested),Tuple(plans),estimates))
end

function _check_reduction_plan_set(set::ReductionPlanSet)
    length(set.ks)==length(set.plans)||throw(DimensionMismatch(
        "ReductionPlanSet has inconsistent subsystem-size and plan storage"))
    allunique(set.ks)||throw(ArgumentError(
        "ReductionPlanSet subsystem sizes must be unique"))
    for index in eachindex(set.ks)
        k=set.ks[index]
        k isa Integer&&!(k isa Bool)||throw(ArgumentError(
            "ReductionPlanSet subsystem sizes must be integers"))
        0<=k<=set.basis.N||throw(ArgumentError(
            "ReductionPlanSet subsystem size k must satisfy 0 ≤ k ≤ N"))
        plan=set.plans[index]
        plan isa ReductionPlan||throw(ArgumentError(
            "ReductionPlanSet storage contains a non-ReductionPlan entry"))
        _check_reduction_plan(plan,set.basis,k)
    end
    set
end

"""Return the prepared plan for subsystem size `k`."""
function reduction_plan(set::ReductionPlanSet,k::Integer)
    _check_reduction_plan_set(set)
    position=findfirst(==(k),set.ks)
    position===nothing&&throw(ArgumentError(
        "ReductionPlanSet does not contain subsystem size $k"))
    set.plans[position]
end

"""
    ReductionWorkspaceSet(plan_set, rho; mode=:both)

Task-owned numerical scratch matching every plan in a
[`ReductionPlanSet`](@ref). The workspace set must not be shared concurrently;
the immutable plan set may be shared read-only.
"""
struct ReductionWorkspaceSet{P,W}
    plans::P
    workspaces::W
end

function ReductionWorkspaceSet(set::ReductionPlanSet,rho::PIState;
                               mode::Symbol=:both)
    _check_reduction_plan_set(set)
    rho.basis===set.basis||throw(ArgumentError(
        "state and ReductionPlanSet use different PIBasis objects"))
    ReductionWorkspaceSet(
        set,Tuple(ReductionWorkspace(plan,rho;mode) for plan in set.plans))
end

function _check_reduction_workspace_set(
        work::ReductionWorkspaceSet,set::ReductionPlanSet,rho::PIState)
    _check_reduction_plan_set(set)
    work.plans===set||throw(ArgumentError(
        "ReductionWorkspaceSet was prepared for a different ReductionPlanSet"))
    rho.basis===set.basis||throw(ArgumentError(
        "state and ReductionPlanSet use different PIBasis objects"))
    length(work.workspaces)==length(set.plans)||throw(DimensionMismatch(
        "ReductionWorkspaceSet has inconsistent workspace storage"))
    work
end

"""
    reduced_states(rho, plan_set; workspace=nothing, check=true, kwargs...)

Compute all requested reduced density matrices while validating the parent
state once. A supplied [`ReductionWorkspaceSet`](@ref) reuses every numerical
buffer.
"""
function reduced_states(rho::PIState,set::ReductionPlanSet;
        workspace=nothing,check::Bool=true,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho))
    _check_reduction_plan_set(set)
    rho.basis===set.basis||throw(ArgumentError(
        "state and ReductionPlanSet use different PIBasis objects"))
    check&&validate_state(rho;atol,rtol)
    works=workspace===nothing ? ntuple(_->nothing,length(set.plans)) :
        _check_reduction_workspace_set(workspace,set,rho).workspaces
    map(set.plans,works) do plan,work
        reduced_state(rho,plan.k;plan,workspace=work,
                      check=false,atol,rtol)
    end
end

"""
    reduced_purities(rho, plan_set; workspace=nothing, check=true, kwargs...)

Prepared multi-bipartition purity evaluation with one parent-state
validation.
"""
function reduced_purities(rho::PIState,set::ReductionPlanSet;
        workspace=nothing,check::Bool=true,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho))
    _check_reduction_plan_set(set)
    rho.basis===set.basis||throw(ArgumentError(
        "state and ReductionPlanSet use different PIBasis objects"))
    check&&validate_state(rho;atol,rtol)
    works=workspace===nothing ? ntuple(_->nothing,length(set.plans)) :
        _check_reduction_workspace_set(workspace,set,rho).workspaces
    map(set.plans,works) do plan,work
        reduced_purity(rho,plan.k;plan,workspace=work,
                       check=false,atol,rtol)
    end
end

"""
    bipartition_negativities(rho, plan_set; workspace=nothing, kwargs...)

Evaluate negativity in the exact order of `plan_set.ks`. The state is
validated once; a `ReductionWorkspaceSet(mode=:negativity)` or `:both`
workspace avoids per-state scratch allocation.
"""
function bipartition_negativities(rho::PIState,set::ReductionPlanSet;
        workspace=nothing,atol::Real=_analysis_atol(rho),
        rtol::Real=_state_rtol(rho))
    _check_reduction_plan_set(set)
    rho.basis===set.basis||throw(ArgumentError(
        "state and ReductionPlanSet use different PIBasis objects"))
    validate_state(rho;atol,rtol)
    works=workspace===nothing ? ntuple(_->nothing,length(set.plans)) :
        _check_reduction_workspace_set(workspace,set,rho).workspaces
    map(set.plans,works) do plan,work
        if work===nothing
            _plan_negativity(rho,plan)
        else
            _check_reduction_workspace(work,plan,rho)
            _plan_negativity(rho,plan,work;atol,rtol)
        end
    end
end
