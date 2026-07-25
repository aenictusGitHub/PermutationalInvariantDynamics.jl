# One shared finite-cutoff pseudomode coupled to a PI ensemble.
#
# Permutations act only on the physical systems.  The distinguished mode is
# therefore a finite global factor in a CompositePIBasis, never an internal
# factor of a PISupersite.  Composite coordinates keep the PI system factor
# first (fastest) and the mode operator coordinate second.

"""
    GlobalPseudomodeModel

Prepared time-local model for one truncated bosonic pseudomode shared by all
members of a permutationally invariant ensemble.  The system remains in its
exact [`PIBasis`](@ref), the oscillator is a [`FiniteOperatorBasis`](@ref),
and `generator` is a matrix-free sum of factorized superoperators.

`background` contains the supplied PI system generator, the mode frequency,
and coherent system--mode couplings.  It excludes `damping_channels`, so the
two can be passed directly to [`CompositeTrajectoryPlan`](@ref).  `generator`
contains both pieces and is the unconditional deterministic generator.
Prepared numerical data are read-only; obtain a task-owned workspace with
[`global_pseudomode_workspace`](@ref).
"""
struct GlobalPseudomodeModel{
        R<:AbstractFloat,SB,FB,CB,M,SM,SP,BG,DC,G,MO,CO,SR,MR,V,E,MD,Q}
    system_basis::SB
    mode_basis::FB
    basis::CB
    mode::M
    system_model::SM
    system_plan::SP
    background::BG
    damping_channels::DC
    generator::G
    mode_operators::MO
    coupling_operators::CO
    system_reduction_plan::SR
    mode_reduction_plan::MR
    trace_vector::V
    resource_estimates::E
    metadata::MD
    precision_bits::Int
    rounding_mode::Q
end

eltype(::GlobalPseudomodeModel{R}) where R=Complex{R}
size(model::GlobalPseudomodeModel)=size(model.generator)
size(model::GlobalPseudomodeModel,index::Integer)=size(model.generator,index)
pi_dimension(model::GlobalPseudomodeModel)=length(model.basis)
isautonomous(model::GlobalPseudomodeModel)=isautonomous(model.generator)
_operator_requires_matrixfree(::GlobalPseudomodeModel)=true
function _performance_source_action_bytes(
        model::GlobalPseudomodeModel,::Type{T}) where T
    BigInt(model.resource_estimates.workspace_upper_bytes)+
        BigInt(model.resource_estimates.action_transient_upper_bytes)
end

function show(io::IO,model::GlobalPseudomodeModel)
    print(io,
        "GlobalPseudomodeModel(N=$(model.system_basis.N), " *
        "d=$(model.system_basis.d), nmax=$(model.mode.nmax), " *
        "dimension=$(length(model.basis)))")
end

function _global_pseudomode_with_precision(
        f,model::GlobalPseudomodeModel{R}) where R
    _with_supersite_precision(
        f,R,model.precision_bits,model.rounding_mode)
end

function _global_pseudomode_single_parameter(
        value,default,name::AbstractString;nonnegative::Bool=false)
    resolved=value===nothing ? default : value
    if resolved isa Tuple||resolved isa AbstractVector
        length(resolved)==1||throw(DimensionMismatch(
            "$name must contain exactly one value for a global pseudomode"))
        resolved=only(resolved)
    end
    resolved isa Real&&!(resolved isa Bool)||throw(ArgumentError(
        "$name must be a real number"))
    _supersite_isfinite(resolved)||throw(ArgumentError(
        "$name must be finite"))
    nonnegative&&resolved<0&&throw(ArgumentError(
        "$name must be nonnegative"))
    resolved
end

function _global_pseudomode_couplings(couplings)
    _pseudomode_argument_tuple(
        couplings,PseudomodeCoupling,"global pseudomode coupling")
end

function _global_pseudomode_check_selector(
        coupling::PseudomodeCoupling,mode::BosonicPseudomode)
    selected=coupling.mode
    valid=selected isa Integer ? selected==1 : selected==mode.label
    valid||throw(ArgumentError(
        "global pseudomode coupling selects $(repr(selected)), but the " *
        "only shared mode is $(repr(mode.label))"))
    coupling
end

function _global_pseudomode_required_precision(
        system_model::PIModel,mode::BosonicPseudomode,couplings,
        frequency,damping,thermal_occupation,T)
    R=isempty(system_model.terms) ?
        _real_float_type(eltype(mode)) :
        _model_geometry_type(system_model)
    for term in system_model.terms
        R=_pseudomode_promote_term_type(R,term)
    end
    R=promote_type(R,_real_float_type(eltype(mode)))
    for coupling in couplings
        R=promote_type(R,_real_float_type(eltype(coupling)))
    end
    for value in (frequency,damping,thermal_occupation)
        R=_supersite_promote_parameter_type(R,value)
    end
    if T!==nothing
        requested=_real_float_type(T)
        requested<:AbstractFloat||throw(ArgumentError(
            "T must select an AbstractFloat scalar type"))
        promote_type(requested,R)===requested||throw(ArgumentError(
            "T=$requested would narrow global pseudomode data requiring $R"))
        R=requested
    end
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),mode.precision_bits,
            maximum(_pseudomode_term_precision,
                    system_model.terms;init=0),
            maximum(coupling->coupling.precision_bits,couplings;init=0),
            _supersite_value_precision(frequency),
            _supersite_value_precision(damping),
            _supersite_value_precision(thermal_occupation)) :
        precision(R)
    rounding_mode=R===BigFloat ?
        (mode.rounding_mode===nothing ?
            rounding(BigFloat) : mode.rounding_mode) : nothing
    R,precision_bits,rounding_mode
end

function _global_pseudomode_system_plan(
        model::PIModel;coefficient_cache,memory_budget,
        bigfloat_precision::Integer)
    isempty(model.terms)&&return nothing
    _require_model_preparation_budget(
        model,memory_budget;
        operation="global pseudomode system-plan preparation",
        bigfloat_precision,coefficient_cache)
    LiouvillianPlan(model;coefficient_cache)
end

function _global_owned_composite_term(
        basis::CompositePIBasis,actions::Tuple;coefficient=1)
    length(actions)==length(basis.factors)||throw(DimensionMismatch(
        "one action entry is required for each composite factor"))
    for (index,action) in pairs(actions)
        action===nothing&&continue
        size(action)==
            (basis.dimensions[index],basis.dimensions[index])||
            throw(DimensionMismatch(
                "factor $index action has size $(size(action)); expected " *
                "($(basis.dimensions[index]), $(basis.dimensions[index]))"))
        _validate_composite_factor_action(
            basis.factors[index],action,index)
    end
    CompositeSuperoperatorTerm{
        typeof(basis),typeof(actions),typeof(coefficient)}(
        basis,actions,coefficient)
end

function _global_hamiltonian_terms(
        basis::CompositePIBasis,system_operator::PIOperator,
        mode_left,mode_right,rate)
    system_operator.basis===basis.factors[1]||throw(ArgumentError(
        "global coupling operator belongs to a different system basis"))
    left=factor_left_superoperator(
        basis.factors[1],system_operator)
    right=factor_right_superoperator(
        basis.factors[1],system_operator)
    (
        _global_owned_composite_term(
            basis,(left,mode_left);coefficient=-im*rate),
        _global_owned_composite_term(
            basis,(right,mode_right);coefficient=im*rate),
    )
end

function _global_pi_action_nnz(operator::PIOperator)
    basis=operator.basis
    total=big(0)
    for sector in basis.sectors
        block=coefficient_block(operator,sector)
        dimension=size(block,1)
        support=count(!iszero,block)
        total+=BigInt(dimension)*BigInt(support)
    end
    total
end

function _global_sparse_action_bytes(
        dimension::Integer,nonzeros::Integer,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    count=BigInt(nonzeros)
    _performance_entries_bytes(
        count,T;bigfloat_precision)+
        (count+BigInt(dimension)+1)*sizeof(Int)
end

function _global_composite_workspace_upper_bytes(
        generator::CompositeSuperoperator,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    entries=2BigInt(length(generator.basis))
    nested=big(0)
    for term in generator.terms
        for (action,dimension) in zip(
                term.actions,generator.basis.dimensions)
            action===nothing&&continue
            entries+=2BigInt(dimension)
            if action isa LiouvillianPlan
                nested+=_performance_liouvillian_workspace_bytes(
                    action;bigfloat_precision)
            elseif action isa CompiledPIModel
                nested+=_performance_liouvillian_workspace_bytes(
                    action.plan;bigfloat_precision)
            end
        end
    end
    _performance_entries_bytes(entries,T;bigfloat_precision)+nested
end

function _global_pseudomode_basis_preflight(
        N::Integer,d::Integer,sectors,::Type{R},
        memory_budget,precision_bits::Integer) where R<:AbstractFloat
    geometry=_supersite_sector_geometry(Int(N),Int(d),sectors)
    basis_bytes=_supersite_basis_preflight_bytes(Int(d),geometry)
    mode_independent_state=_performance_entries_bytes(
        geometry.coordinate_dimension,Complex{R};
        bigfloat_precision=precision_bits)
    _require_performance_budget(
        "global pseudomode PI-system basis construction",
        basis_bytes+mode_independent_state,memory_budget;
        guidance="Reduce N, the system dimension, or the retained Schur sectors.")
    geometry
end

function _global_pseudomode_remaining_budget(memory_budget,retained_bytes)
    limit=_performance_memory_limit(memory_budget)
    limit===nothing&&return Inf
    remaining=limit-BigInt(retained_bytes)
    remaining>=0||_require_performance_budget(
        "global pseudomode retained preparation",
        BigInt(retained_bytes),memory_budget;
        guidance="Reduce the system PI basis or mode cutoff.")
    min(remaining,BigInt(typemax(Int)))
end

function _global_pseudomode_collective_components(
        system_basis::PIBasis,couplings,mode::BosonicPseudomode,
        ::Type{R},geometry;retain_zero_terms::Bool) where R<:AbstractFloat
    records=()
    components=()
    for (index,coupling) in pairs(couplings)
        _global_pseudomode_check_selector(coupling,mode)
        size(coupling.operator)==(system_basis.d,system_basis.d)||
            throw(DimensionMismatch(
                "global coupling $index operator must be " *
                "$(system_basis.d)×$(system_basis.d)"))
        g=_supersite_checked_complex(
            R,coupling.strength,
            "global rotating-wave coupling strength")
        h=_supersite_checked_complex(
            R,coupling.counterrotating_strength,
            "global counter-rotating coupling strength")
        if iszero(g)&&iszero(h)&&!retain_zero_terms
            records=(records...,(
                collective=nothing,rotating=nothing,
                counterrotating=nothing,strength=g,
                counterrotating_strength=h,mode=mode.label),)
            continue
        end
        geometry===nothing&&throw(ArgumentError(
            "active global pseudomode couplings require prepared " *
            "collective geometry"))
        local_operator=_supersite_converted_component(
            coupling.operator,R;
            context="global pseudomode coupling operator")
        collective=collective_operator(
            system_basis,local_operator;cache=geometry)

        rotating=nothing
        counterrotating=nothing
        if !iszero(g)||retain_zero_terms
            weighted=g*collective
            A=weighted+adjoint(weighted)
            B=-im*(weighted-adjoint(weighted))
            rotating=(x=A,y=B)
            if retain_zero_terms||!all(iszero,A.data)
                components=(
                    components...,
                    (operator=A,mode_quadrature=:x,rate=R(1//2),
                     coupling=index,kind=:rotating_x),)
            end
            if retain_zero_terms||!all(iszero,B.data)
                components=(
                    components...,
                    (operator=B,mode_quadrature=:y,rate=R(1//2),
                     coupling=index,kind=:rotating_y),)
            end
        end
        if !iszero(h)||retain_zero_terms
            weighted=h*collective
            A=weighted+adjoint(weighted)
            B=-im*(weighted-adjoint(weighted))
            counterrotating=(x=A,y=B)
            if retain_zero_terms||!all(iszero,A.data)
                components=(
                    components...,
                    (operator=A,mode_quadrature=:x,rate=R(1//2),
                     coupling=index,kind=:counterrotating_x),)
            end
            if retain_zero_terms||!all(iszero,B.data)
                components=(
                    components...,
                    (operator=B,mode_quadrature=:y,rate=-R(1//2),
                     coupling=index,kind=:counterrotating_y),)
            end
        end
        records=(records...,(
            collective=collective,
            rotating=rotating,
            counterrotating=counterrotating,
            strength=g,counterrotating_strength=h,
            mode=mode.label),)
    end
    Tuple(records),Tuple(components)
end

function _global_pseudomode_coupling_operator_upper_bytes(
        system_basis::PIBasis,couplings,::Type{R};
        retain_zero_terms::Bool,
        bigfloat_precision::Integer=precision(BigFloat)) where
        R<:AbstractFloat
    operators=big(0)
    for coupling in couplings
        rotating=retain_zero_terms||!iszero(coupling.strength)
        counterrotating=retain_zero_terms||
            !iszero(coupling.counterrotating_strength)
        (rotating||counterrotating)||continue
        # One collective operator plus two Hermitian quadratures for each
        # retained rotating/counter-rotating interaction.
        operators+=1+2Int(rotating)+2Int(counterrotating)
    end
    _performance_entries_bytes(
        operators*BigInt(length(system_basis)),Complex{R};
        bigfloat_precision)
end

function _global_pseudomode_damping_channel(
        basis::CompositePIBasis,jump,rate,label::Symbol)
    CompositeJumpChannel(basis,2=>jump;rate,label)
end

"""
    global_pseudomode_model(system_model, mode;
                            couplings=(), frequency=nothing,
                            damping=nothing, thermal_occupation=nothing,
                            retain_zero_terms=false,
                            coefficient_cache=nothing,
                            T=nothing,
                            memory_budget=512*1024^2)

Prepare one truncated pseudomode shared by every member of an existing PI
system model.  A [`PseudomodeCoupling`](@ref) contains a one-particle matrix
`L`; this builder lifts it explicitly to
`J_L = sum(i=1:N) L_i` and uses

`g*J_L⊗a' + conj(g)*J_L'⊗a`

plus the optional counter-rotating interaction.  No Kac or other
`N`-dependent scaling is inserted.

`T` may explicitly select a wider working type for the mode and coupling data,
but it does not widen a nonempty PI system generator after lowering. Such a
system model must already use the requested real type. `T` may not narrow any
input; with `T=nothing`, input scalar types are promoted without narrowing.

The result uses
`CompositePIBasis(system_model.basis, FiniteOperatorBasis(mode.levels))`, so
its coordinate dimension is
`length(system_model.basis) * mode.levels^2`.  It never forms the full
system Hilbert space or the full composite Kronecker superoperator.

The mode dissipator is
`damping*(nbar+1)*D[a] + damping*nbar*D[a']`.  Stochastic trajectories may
use `result.background` with `result.damping_channels`; deterministic
dynamics use `result.generator` or [`global_pseudomode_matrixfree`](@ref).
"""
function global_pseudomode_model(
        system_model::PIModel,mode::BosonicPseudomode;
        couplings=(),frequency=nothing,damping=nothing,
        thermal_occupation=nothing,
        retain_zero_terms::Bool=false,coefficient_cache=nothing,
        T=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    resolved_couplings=_global_pseudomode_couplings(couplings)
    omega=_global_pseudomode_single_parameter(
        frequency,mode.frequency,"frequency")
    kappa=_global_pseudomode_single_parameter(
        damping,mode.damping,"damping";nonnegative=true)
    occupation=_global_pseudomode_single_parameter(
        thermal_occupation,mode.thermal_occupation,
        "thermal_occupation";nonnegative=true)
    R,precision_bits,rounding_mode=
        _global_pseudomode_required_precision(
            system_model,mode,resolved_couplings,
            omega,kappa,occupation,T)
    if R===BigFloat&&
            (precision(BigFloat)!=precision_bits||
             rounding(BigFloat)!=rounding_mode)
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            global_pseudomode_model(
                system_model,mode;
                couplings=resolved_couplings,
                frequency=omega,damping=kappa,
                thermal_occupation=occupation,
                retain_zero_terms,coefficient_cache,T=R,memory_budget)
        end
    end

    omegaR=_supersite_checked_real(R,omega,"global pseudomode frequency")
    kappaR=_supersite_checked_real(R,kappa,"global pseudomode damping")
    occupationR=_supersite_checked_real(
        R,occupation,"global pseudomode thermal occupation")
    bigfloat_precision=R===BigFloat ? precision_bits : precision(R)
    system_plan=_global_pseudomode_system_plan(
        system_model;coefficient_cache,memory_budget,bigfloat_precision)
    if system_plan!==nothing&&
            _real_float_type(eltype(system_plan))!==R
        throw(ArgumentError(
            "the prepared PI system generator uses $(eltype(system_plan)), " *
            "but the shared mode and couplings require Complex{$R}; rebuild " *
            "the system model at the common scalar precision"))
    end

    system_basis=system_model.basis
    mode_basis=FiniteOperatorBasis(
        mode.levels;label=mode.label)
    basis=CompositePIBasis(system_basis,mode_basis)
    coordinate_bytes=_performance_entries_bytes(
        length(basis),Complex{R};
        bigfloat_precision)
    needs_coupling_geometry=!isempty(resolved_couplings)&&(
        retain_zero_terms||any(
            coupling->!iszero(coupling.strength)||
                      !iszero(coupling.counterrotating_strength),
            resolved_couplings))
    geometry_estimate=needs_coupling_geometry ?
        estimate_geometry_bytes(
            system_basis;T=R,bigfloat_precision) : nothing
    coupling_operator_upper_bytes=
        _global_pseudomode_coupling_operator_upper_bytes(
            system_basis,resolved_couplings,R;
            retain_zero_terms,bigfloat_precision)
    setup_before_maps=
        BigInt(Base.summarysize(system_model))+
        BigInt(system_plan===nothing ? 0 : Base.summarysize(system_plan))+
        (geometry_estimate===nothing ? big(0) :
            BigInt(geometry_estimate.setup_bytes))+
        coupling_operator_upper_bytes+
        3coordinate_bytes
    _require_performance_budget(
        "global pseudomode preparation",setup_before_maps,memory_budget;
        guidance="Reduce the system PI basis, mode cutoff, or coupling geometry.")

    # Repeated system/mode partial traces use immutable packed diagonal plans.
    # Give each setup only the uncommitted part of the caller's shared model
    # budget, so their guarded setup peaks include already retained objects.
    reduction_retained_before=BigInt(Base.summarysize((
        system_model,system_plan,basis)))
    reduction_initial_retained=reduction_retained_before
    system_reduction_plan=CompositeReductionPlan(
        basis,1;T=R,memory_budget=_global_pseudomode_remaining_budget(
            memory_budget,reduction_retained_before))
    reduction_retained_before=BigInt(Base.summarysize((
        system_model,system_plan,basis,system_reduction_plan)))
    mode_reduction_plan=CompositeReductionPlan(
        basis,2;T=R,memory_budget=_global_pseudomode_remaining_budget(
            memory_budget,reduction_retained_before))
    reduction_plan_bytes=
        BigInt(system_reduction_plan.estimates.retained_bytes)+
        BigInt(mode_reduction_plan.estimates.retained_bytes)
    reduction_setup_peak=max(
        reduction_initial_retained+
            BigInt(system_reduction_plan.estimates.setup_peak_bytes),
        reduction_retained_before+
            BigInt(mode_reduction_plan.estimates.setup_peak_bytes))

    mode_operators=_with_supersite_precision(
            R,precision_bits,rounding_mode) do
        (
            identity=_supersite_converted_component(
                mode.identity,R;context="global mode identity"),
            annihilation=_supersite_converted_component(
                mode.annihilation,R;context="global mode annihilation"),
            creation=_supersite_converted_component(
                mode.creation,R;context="global mode creation"),
            number_operator=_supersite_converted_component(
                mode.number_operator,R;context="global mode number operator"),
            parity=_supersite_converted_component(
                mode.parity,R;context="global mode parity"),
            top_projector=_supersite_converted_component(
                mode.top_projector,R;context="global mode top projector"),
        )
    end
    xquadrature=mode_operators.annihilation+
        mode_operators.creation
    yquadrature=-im*(mode_operators.annihilation-
        mode_operators.creation)
    geometry=needs_coupling_geometry ?
        _collective_geometry(system_basis,R,nothing) : nothing
    coupling_operators,coupling_components=
        _global_pseudomode_collective_components(
            system_basis,resolved_couplings,mode,R,geometry;
            retain_zero_terms)

    predicted_action_bytes=big(0)
    for component in coupling_components
        map_nonzeros=_global_pi_action_nnz(component.operator)
        predicted_action_bytes+=2*_global_sparse_action_bytes(
            length(system_basis),map_nonzeros,Complex{R};
            bigfloat_precision)
    end
    _require_performance_budget(
        "global pseudomode coupling-map preparation",
        BigInt(system_plan===nothing ? 0 : Base.summarysize(system_plan))+
            reduction_plan_bytes+3coordinate_bytes+
            predicted_action_bytes,
        memory_budget;
        guidance="Reduce the retained Schur sectors or coupling support.")

    terms=()
    if system_plan!==nothing
        terms=(terms...,
            _global_owned_composite_term(
                basis,(system_plan,nothing);coefficient=one(R)))
    end
    mode_left=(
        number=factor_left_superoperator(
            mode_basis,mode_operators.number_operator),
        x=factor_left_superoperator(mode_basis,xquadrature),
        y=factor_left_superoperator(mode_basis,yquadrature),
    )
    mode_right=(
        number=factor_right_superoperator(
            mode_basis,mode_operators.number_operator),
        x=factor_right_superoperator(mode_basis,xquadrature),
        y=factor_right_superoperator(mode_basis,yquadrature),
    )
    if !iszero(omegaR)||retain_zero_terms
        terms=(terms...,
            _global_owned_composite_term(
                basis,(nothing,mode_left.number);
                coefficient=-im*omegaR),
            _global_owned_composite_term(
                basis,(nothing,mode_right.number);
                coefficient=im*omegaR))
    end
    for component in coupling_components
        left=getproperty(mode_left,component.mode_quadrature)
        right=getproperty(mode_right,component.mode_quadrature)
        terms=(terms...,
            _global_hamiltonian_terms(
                basis,component.operator,left,right,
                component.rate)...)
    end
    background=CompositeSuperoperator(
        basis,terms...;T=Complex{R})

    loss_rate=_supersite_checked_product(
        kappaR,_supersite_checked_increment(
            occupationR,"global pseudomode thermal occupation plus one"),
        "global pseudomode loss rate")
    gain_rate=_supersite_checked_product(
        kappaR,occupationR,"global pseudomode thermal-gain rate")
    damping_channels=()
    if !iszero(loss_rate)||retain_zero_terms
        damping_channels=(damping_channels...,
            _global_pseudomode_damping_channel(
                basis,mode_operators.annihilation,loss_rate,
                Symbol(mode.label,:_loss)))
    end
    if !iszero(gain_rate)||retain_zero_terms
        damping_channels=(damping_channels...,
            _global_pseudomode_damping_channel(
                basis,mode_operators.creation,gain_rate,
                Symbol(mode.label,:_gain)))
    end
    generator_terms=background.terms
    for channel in damping_channels
        generator_terms=(
            generator_terms...,
            _composite_channel_generator_terms(basis,channel)...)
    end
    generator=CompositeSuperoperator(
        basis,generator_terms...;T=Complex{R})
    trace_vector=_composite_trace_functional(basis;T=R)
    trace_vector_bytes=_performance_trace_functional_bytes(
        trace_vector;bigfloat_precision)

    workspace_bytes=_global_composite_workspace_upper_bytes(
        generator,Complex{R};bigfloat_precision)
    action_transient_bytes=_performance_composite_action_bytes(
        generator,Complex{R})
    retained_bytes=BigInt(Base.summarysize((
        system_model,system_plan,background,damping_channels,generator,
        mode_operators,coupling_operators,system_reduction_plan,
        mode_reduction_plan,trace_vector)))
    solve_ready_peak=retained_bytes+workspace_bytes+
        action_transient_bytes+coordinate_bytes
    _require_performance_budget(
        "prepared global pseudomode model",solve_ready_peak,memory_budget;
        guidance="Reduce the system PI basis, the mode cutoff, or the " *
                 "number of coupling components.")
    resource_estimates=(
        coordinate_dimension=length(basis),
        state_bytes=coordinate_bytes,
        trace_vector_bytes,
        coupling_action_upper_bytes=predicted_action_bytes,
        coupling_operator_upper_bytes,
        reduction_plan_bytes,
        workspace_upper_bytes=workspace_bytes,
        action_transient_upper_bytes=action_transient_bytes,
        retained_bytes,
        setup_peak_upper_bytes=max(
            setup_before_maps,
            reduction_setup_peak,
            3coordinate_bytes+predicted_action_bytes,
            solve_ready_peak),
        memory_budget=_performance_memory_limit(memory_budget),
        precision_bits=precision_bits,
    )
    metadata=(
        embedding=:single_global_pseudomode,
        exact_system_permutation_symmetry=true,
        cutoff_approximation=true,
        system_count=system_basis.N,
        system_dimension=system_basis.d,
        system_pi_dimension=length(system_basis),
        oscillator_cutoff=mode.nmax,
        mode_levels=mode.levels,
        composite_dimension=length(basis),
        coordinate_order=:system_pi_fastest_then_global_mode,
        coupling_convention=:collective_sum_without_kac_scaling,
        dissipator_convention=:standard,
        frequency=omegaR,damping=kappaR,
        thermal_occupation=occupationR,
        precision_bits,rounding_mode,
        retained_zero_terms=retain_zero_terms,
    )
    GlobalPseudomodeModel{
        R,typeof(system_basis),typeof(mode_basis),typeof(basis),
        typeof(mode),typeof(system_model),typeof(system_plan),
        typeof(background),typeof(damping_channels),typeof(generator),
        typeof(mode_operators),typeof(coupling_operators),
        typeof(system_reduction_plan),typeof(mode_reduction_plan),
        typeof(trace_vector),typeof(resource_estimates),typeof(metadata),
        typeof(rounding_mode)}(
        system_basis,mode_basis,basis,mode,system_model,system_plan,
        background,damping_channels,generator,mode_operators,
        coupling_operators,system_reduction_plan,mode_reduction_plan,
        trace_vector,resource_estimates,metadata,
        precision_bits,rounding_mode)
end

"""
    global_pseudomode_model(N, system_hamiltonian, mode;
                            system_terms=(), system_rate=1,
                            sectors=nothing, kwargs...)

Convenience constructor for a shared pseudomode. `system_hamiltonian` is the
one-particle Hamiltonian summed over all `N` systems, and `system_terms` may
contain additional PI terms in the bare system dimension.  For a prebuilt
system model, prefer the [`PIModel`](@ref) method so its exact basis and
prepared geometry can be reused.
"""
function global_pseudomode_model(
        N::Integer,system_hamiltonian::AbstractMatrix,
        mode::BosonicPseudomode;
        couplings=(),system_terms=(),system_rate=1,sectors=nothing,
        frequency=nothing,damping=nothing,thermal_occupation=nothing,
        retain_zero_terms::Bool=false,coefficient_cache=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    N isa Integer&&!(N isa Bool)||throw(ArgumentError(
        "N must be an integer"))
    N>=1||throw(ArgumentError(
        "a global pseudomode model requires at least one system"))
    rows,columns=size(system_hamiltonian)
    rows==columns&&rows>0||throw(DimensionMismatch(
        "system_hamiltonian must be nonempty and square"))
    ishermitian(system_hamiltonian)||throw(ArgumentError(
        "system_hamiltonian must be Hermitian"))
    all(_supersite_isfinite,
        _supersite_stored_values(system_hamiltonian))||
        throw(ArgumentError(
            "system_hamiltonian must contain only finite values"))
    system_rate isa Real&&!(system_rate isa Bool)||throw(ArgumentError(
        "system_rate must be a real number"))
    _supersite_isfinite(system_rate)||throw(ArgumentError(
        "system_rate must be finite"))
    terms=_pseudomode_argument_tuple(
        system_terms,AbstractPITerm,"system term")
    resolved_couplings=_global_pseudomode_couplings(couplings)
    omega=_global_pseudomode_single_parameter(
        frequency,mode.frequency,"frequency")
    kappa=_global_pseudomode_single_parameter(
        damping,mode.damping,"damping";nonnegative=true)
    occupation=_global_pseudomode_single_parameter(
        thermal_occupation,mode.thermal_occupation,
        "thermal_occupation";nonnegative=true)
    R=_real_float_type(eltype(system_hamiltonian))
    R=promote_type(R,_real_float_type(eltype(mode)))
    R=_supersite_promote_parameter_type(R,system_rate)
    for coupling in resolved_couplings
        R=promote_type(R,_real_float_type(eltype(coupling)))
    end
    for term in terms
        R=_pseudomode_promote_term_type(R,term)
    end
    for value in (omega,kappa,occupation)
        R=_supersite_promote_parameter_type(R,value)
    end
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),mode.precision_bits,
            _supersite_array_precision(system_hamiltonian),
            maximum(coupling->coupling.precision_bits,
                    resolved_couplings;init=0),
            maximum(_pseudomode_term_precision,terms;init=0),
            _supersite_value_precision(system_rate),
            _supersite_value_precision(omega),
            _supersite_value_precision(kappa),
            _supersite_value_precision(occupation)) :
        precision(R)
    rounding_mode=R===BigFloat ?
        (mode.rounding_mode===nothing ?
            rounding(BigFloat) : mode.rounding_mode) : nothing
    if R===BigFloat&&
            (precision(BigFloat)!=precision_bits||
             rounding(BigFloat)!=rounding_mode)
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            global_pseudomode_model(
                N,system_hamiltonian,mode;
                couplings=resolved_couplings,system_terms=terms,
                system_rate,sectors,frequency=omega,damping=kappa,
                thermal_occupation=occupation,retain_zero_terms,
                coefficient_cache,memory_budget)
        end
    end
    _global_pseudomode_basis_preflight(
        N,rows,sectors,R,memory_budget,precision_bits)
    basis=PIBasis(N,rows;sectors)
    H=_supersite_converted_component(
        system_hamiltonian,R;context="global system Hamiltonian")
    rate=_supersite_checked_real(
        R,system_rate,"global system Hamiltonian rate")
    model_terms=terms
    if !iszero(H)&&(!iszero(rate)||retain_zero_terms)
        model_terms=(LocalHamiltonian(H;rate,check=false),model_terms...)
    end
    system_model=PIModel(basis,model_terms)
    global_pseudomode_model(
        system_model,mode;
        couplings=resolved_couplings,frequency=omega,damping=kappa,
        thermal_occupation=occupation,retain_zero_terms,
        coefficient_cache,T=R,memory_budget)
end

"""Alias emphasizing that a global pseudomode is one mode shared by the ensemble."""
shared_pseudomode_model(args...;kwargs...)=
    global_pseudomode_model(args...;kwargs...)

"""
    pseudomode_model(system_model::PIModel, mode::BosonicPseudomode; kwargs...)

Build the global/shared-mode workflow from an existing PI system model.
The `N, system_hamiltonian, mode` signature remains local by default; pass
`topology=:global` there or call [`global_pseudomode_model`](@ref) explicitly.
"""
pseudomode_model(
        system_model::PIModel,mode::BosonicPseudomode;kwargs...)=
    global_pseudomode_model(system_model,mode;kwargs...)

"""
    global_pseudomode_workspace(model; T=eltype(model),
                                memory_budget=512*1024^2)

Allocate one task-owned composite application workspace while preserving the
prepared BigFloat precision and rounding mode. `T` must use the prepared
model's real floating-point type; rebuild the model to change that precision.
"""
function global_pseudomode_workspace(
        model::GlobalPseudomodeModel;T=eltype(model),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    resolved=_composite_coordinate_type(T)
    promote_type(resolved,eltype(model))===resolved||throw(ArgumentError(
        "workspace scalar type $resolved would narrow $(eltype(model))"))
    _real_float_type(resolved)===_real_float_type(eltype(model))||
        throw(ArgumentError(
            "a prepared global pseudomode model must be rebuilt at the " *
            "requested workspace precision"))
    estimate=_global_composite_workspace_upper_bytes(
        model.generator,resolved;
        bigfloat_precision=model.precision_bits)
    _require_performance_budget(
        "global pseudomode workspace",estimate,memory_budget;
        guidance="Reduce the composite dimension or number of coupling terms.")
    _global_pseudomode_with_precision(model) do
        CompositeSuperoperatorWorkspace(
            model.generator;T=resolved)
    end
end

function apply!(
        destination::AbstractVector,model::GlobalPseudomodeModel,
        source::AbstractVector,time,parameters,
        workspace::CompositeSuperoperatorWorkspace)
    _global_pseudomode_with_precision(model) do
        apply!(
            destination,model.generator,source,time,parameters,workspace)
    end
end

function apply!(
        destination::AbstractVector,model::GlobalPseudomodeModel,
        source::AbstractVector,
        workspace::CompositeSuperoperatorWorkspace)
    isautonomous(model)||throw(ArgumentError(
        "autonomous apply! was requested for a driven global pseudomode model"))
    apply!(destination,model,source,0.0,nothing,workspace)
end

function apply_adjoint!(
        destination::AbstractVector,model::GlobalPseudomodeModel,
        source::AbstractVector,time,parameters,
        workspace::CompositeSuperoperatorWorkspace)
    _global_pseudomode_with_precision(model) do
        apply_adjoint!(
            destination,model.generator,source,time,parameters,workspace)
    end
end

function apply_adjoint!(
        destination::AbstractVector,model::GlobalPseudomodeModel,
        source::AbstractVector,
        workspace::CompositeSuperoperatorWorkspace)
    isautonomous(model)||throw(ArgumentError(
        "autonomous adjoint apply! was requested for a driven global " *
        "pseudomode model"))
    apply_adjoint!(
        destination,model,source,0.0,nothing,workspace)
end

"""
    global_pseudomode_matrixfree(model;
                                 workspace=nothing,
                                 memory_budget=512*1024^2)

Return a synchronized [`MatrixFreeLiouvillian`](@ref) for Krylov and response
solvers.  The wrapper owns or reuses exactly one composite workspace and
provides explicit forward and adjoint callbacks.  Parallel hot loops should
instead call `apply!` with one [`global_pseudomode_workspace`](@ref) per task.
The memory preflight includes the workspace and the wrapper's copied sparse
physical trace functional.
"""
function global_pseudomode_matrixfree(
        model::GlobalPseudomodeModel;workspace=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    if workspace!==nothing
        workspace isa CompositeSuperoperatorWorkspace||throw(ArgumentError(
            "workspace must be a CompositeSuperoperatorWorkspace"))
        workspace.superoperator===model.generator||throw(ArgumentError(
            "workspace belongs to a different global pseudomode generator"))
    end
    wrapper_bytes=
        BigInt(model.resource_estimates.workspace_upper_bytes)+
        BigInt(model.resource_estimates.trace_vector_bytes)
    _require_performance_budget(
        "global pseudomode matrix-free wrapper",wrapper_bytes,memory_budget;
        guidance="Reduce the composite dimension or provide a larger " *
                 "explicit memory budget.")
    work=workspace===nothing ?
        global_pseudomode_workspace(model;memory_budget) : workspace
    action! = (y,x,t,p)->apply!(y,model,x,t,p,work)
    adjoint_action! =
        (y,x,t,p)->apply_adjoint!(y,model,x,t,p,work)
    MatrixFreeLiouvillian(
        length(model.basis),action!,eltype(model),
        copy(model.trace_vector);
        autonomous=isautonomous(model),workspace=work,
        adjoint_action!)
end

function liouvillian(
        model::GlobalPseudomodeModel;
        representation=:matrixfree,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        workspace=nothing)
    representation in (:auto,:matrixfree)||throw(ArgumentError(
        "global pseudomode models support representation=:matrixfree; " *
        "a full composite Kronecker superoperator is intentionally not formed"))
    global_pseudomode_matrixfree(
        model;workspace,memory_budget)
end

function steady_state(
        model::GlobalPseudomodeModel;
        method=:krylov,initial_state=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    isautonomous(model)||throw(ArgumentError(
        "steady_state requires an autonomous global pseudomode model"))
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    canonical=_canonical_stationary_algorithm(method)
    canonical in (:auto,:gmres)||throw(ArgumentError(
        "global pseudomode steady states support only matrix-free " *
        "method=:krylov (or :gmres/:auto); a full composite " *
        "superoperator is intentionally not materialized"))
    _global_pseudomode_with_precision(model) do
        initial=if initial_state isa CompositePIState
            initial_state.basis===model.basis||throw(ArgumentError(
                "initial composite state belongs to a different basis"))
            initial_state.data
        else
            initial_state
        end
        operator=global_pseudomode_matrixfree(
            model;memory_budget)
        steady_state(
            operator;trace_vector=model.trace_vector,method=:krylov,
            initial_state=initial,memory_budget,kwargs...)
    end
end

function _global_pseudomode_check_input_precision(
        ::Type{R},::Type{S},context::AbstractString) where
        {R<:AbstractFloat,S}
    input_real=try
        typeof(real(zero(S)))
    catch
        return nothing
    end
    input_real<:AbstractFloat||return nothing
    promote_type(R,input_real)===R||throw(ArgumentError(
        "$context uses $input_real data wider than the prepared $R model; " *
        "rebuild the global pseudomode model at the wider precision"))
    nothing
end

function _global_pseudomode_system_state(
        model::GlobalPseudomodeModel,state,::Type{R}) where
        R<:AbstractFloat
    if state isa PIState
        state.basis===model.system_basis||throw(ArgumentError(
            "system state belongs to a different PIBasis object"))
        _global_pseudomode_check_input_precision(
            R,eltype(state),"system state")
        eltype(state)===Complex{R}&&return state
        data=Vector{Complex{R}}(undef,length(state.data))
        @inbounds for index in eachindex(data,state.data)
            data[index]=_supersite_checked_complex(
                R,state.data[index],
                "global pseudomode system-state coefficient")
        end
        return PIState(model.system_basis,data)
    elseif state isa AbstractVector
        length(state)==model.system_basis.d||throw(DimensionMismatch(
            "one-particle system ket has the wrong length"))
        _global_pseudomode_check_input_precision(
            R,eltype(state),"one-particle system ket")
        ket=Vector{Complex{R}}(undef,length(state))
        @inbounds for index in eachindex(ket,state)
            ket[index]=_supersite_checked_complex(
                R,state[index],"global pseudomode system-ket entry")
        end
        return iid_pure_state(model.system_basis,ket)
    elseif state isa AbstractMatrix
        size(state)==
            (model.system_basis.d,model.system_basis.d)||
            throw(DimensionMismatch(
                "one-particle system density matrix has the wrong size"))
        _global_pseudomode_check_input_precision(
            R,eltype(state),"one-particle system density matrix")
        density=_supersite_converted_component(
            state,R;context="global pseudomode system density matrix")
        return iid_state(model.system_basis,density)
    end
    throw(ArgumentError(
        "system_state must be a PIState, one-particle ket, or " *
        "one-particle density matrix"))
end

function _global_pseudomode_mode_state(
        model::GlobalPseudomodeModel,state,::Type{R}) where
        R<:AbstractFloat
    resolved=state===nothing ? model.mode.vacuum : state
    matrix=if resolved isa AbstractVector
        length(resolved)==model.mode.levels||throw(DimensionMismatch(
            "global pseudomode ket has the wrong length"))
        _global_pseudomode_check_input_precision(
            R,eltype(resolved),"global pseudomode ket")
        ket=Complex{R}[
            _supersite_checked_complex(
                R,value,"global pseudomode ket entry")
            for value in resolved]
        ket*adjoint(ket)
    elseif resolved isa AbstractMatrix
        size(resolved)==(model.mode.levels,model.mode.levels)||
            throw(DimensionMismatch(
                "global pseudomode density matrix has the wrong size"))
        _global_pseudomode_check_input_precision(
            R,eltype(resolved),"global pseudomode density matrix")
        _supersite_converted_component(
            resolved,R;context="global pseudomode density matrix")
    else
        throw(ArgumentError(
            "mode_state must be a ket, density matrix, or nothing"))
    end
    all(_supersite_isfinite,_supersite_stored_values(matrix))||
        throw(ArgumentError(
            "global pseudomode state must contain only finite values"))
    matrix
end

"""
    pseudomode_product_state(model::GlobalPseudomodeModel, system_state;
                             mode_state=nothing,
                             memory_budget=512*1024^2)

Construct a factorized initial state for a shared-mode model. `system_state`
may be a complete `PIState`, a one-particle ket, or a one-particle density
matrix.  A one-particle input is lifted to an iid PI state.  The mode defaults
to its vacuum. Inputs wider than the prepared model are rejected with rebuild
guidance; narrower inputs are converted to the model precision. Inputs are not
normalized or positivity-repaired.
"""
function pseudomode_product_state(
        model::GlobalPseudomodeModel,system_state;
        mode_state=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    output_bytes=_performance_entries_bytes(
        length(model.basis),eltype(model);
        bigfloat_precision=model.precision_bits)
    system_bytes=_performance_entries_bytes(
        length(model.system_basis),eltype(model);
        bigfloat_precision=model.precision_bits)
    mode_matrix_bytes=_performance_entries_bytes(
        BigInt(model.mode.levels)^2,eltype(model);
        bigfloat_precision=model.precision_bits)
    mode_ket_bytes=_performance_entries_bytes(
        model.mode.levels,eltype(model);
        bigfloat_precision=model.precision_bits)
    # The tensor constructor retains one system-sized intermediate while
    # forming the final composite vector. Count a converted/lifted system
    # state, that intermediate, and the finite-mode ket/density scratch.
    peak_bytes=output_bytes+2system_bytes+
        mode_matrix_bytes+mode_ket_bytes
    _require_performance_budget(
        "global pseudomode product state",peak_bytes,memory_budget;
        guidance="Reduce the system PI basis or mode cutoff.")
    _global_pseudomode_with_precision(model) do
        system=_global_pseudomode_system_state(
            model,system_state,_real_float_type(eltype(model)))
        mode_density=_global_pseudomode_mode_state(
            model,mode_state,_real_float_type(eltype(model)))
        composite_tensor_state(
            model.basis,system,mode_density)
    end
end

"""
    trace_pseudomodes(rho::CompositePIState,
                      model::GlobalPseudomodeModel)

Trace the single shared mode and return the system `PIState`.  This is a
factor-coordinate contraction, not a reconstruction of the `d^N` system.
"""
function trace_pseudomodes(
        rho::CompositePIState,
        model::GlobalPseudomodeModel{R}) where R
    rho.basis===model.basis||throw(ArgumentError(
        "state belongs to a different global pseudomode basis"))
    if R===BigFloat
        return _global_pseudomode_with_precision(model) do
            composite_reduced_state(
                rho,model.system_reduction_plan)
        end
    end
    composite_reduced_state(rho,model.system_reduction_plan)
end

function trace_pseudomodes!(
        output::PIState,rho::CompositePIState,
        model::GlobalPseudomodeModel{R}) where R
    rho.basis===model.basis||throw(ArgumentError(
        "state belongs to a different global pseudomode basis"))
    if R===BigFloat
        return _global_pseudomode_with_precision(model) do
            composite_reduced_state!(
                output,rho,model.system_reduction_plan)
        end
    end
    composite_reduced_state!(
        output,rho,model.system_reduction_plan)
end

"""
    global_pseudomode_state(rho, model)

Trace the PI system factor and return the dense reduced density matrix of the
single shared pseudomode.
"""
function global_pseudomode_state(
        rho::CompositePIState,
        model::GlobalPseudomodeModel{R}) where R
    rho.basis===model.basis||throw(ArgumentError(
        "state belongs to a different global pseudomode basis"))
    if R===BigFloat
        return _global_pseudomode_with_precision(model) do
            composite_reduced_state(
                rho,model.mode_reduction_plan)
        end
    end
    composite_reduced_state(rho,model.mode_reduction_plan)
end

"""
    global_pseudomode_state!(output, rho, model)

Trace the PI system factor into the caller-owned mode density matrix `output`.
This is the preallocated counterpart of [`global_pseudomode_state`](@ref);
`output` must have size `model.mode.levels × model.mode.levels`.
"""
function global_pseudomode_state!(
        output::AbstractMatrix,rho::CompositePIState,
        model::GlobalPseudomodeModel{R}) where R
    rho.basis===model.basis||throw(ArgumentError(
        "state belongs to a different global pseudomode basis"))
    if R===BigFloat
        return _global_pseudomode_with_precision(model) do
            composite_reduced_state!(
                output,rho,model.mode_reduction_plan)
        end
    end
    composite_reduced_state!(
        output,rho,model.mode_reduction_plan)
end
