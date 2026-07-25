# Deliberately exponential, qubit-only oracles for N <= 2. These helpers are
# test code only: production global-pseudomode paths must remain in the PI
# system coordinate tensored with one finite mode operator coordinate.

function _global_pm_qubit_schur_transform(N::Int,::Type{T}) where T
    R=typeof(real(zero(T)))
    if N==1
        return T[0 1;1 0]
    elseif N==2
        q=inv(sqrt(R(2)))
        return T[
            0 0 0 1
            0 q q 0
            1 0 0 0
            0 -q q 0
        ]
    end
    throw(ArgumentError(
        "the global-pseudomode dense oracle is bounded to N=1,2"))
end

function _global_pm_reconstruct_qubit_pi(
        basis::PIBasis,coefficients::AbstractVector)
    basis.d==2||throw(ArgumentError(
        "the global-pseudomode dense oracle is qubit-only"))
    T=eltype(coefficients)
    transform=_global_pm_qubit_schur_transform(basis.N,T)
    schur=zeros(T,size(transform))
    if basis.N==1
        length(coefficients)==4||throw(DimensionMismatch(
            "the N=1 PI coefficient slice must have length four"))
        schur.=reshape(coefficients,2,2)
    elseif basis.N==2
        length(coefficients)==10||throw(DimensionMismatch(
            "the N=2 PI coefficient slice must have length ten"))
        schur[1:3,1:3].=reshape(view(coefficients,1:9),3,3)
        schur[4,4]=coefficients[10]
    else
        throw(ArgumentError(
            "the global-pseudomode dense oracle is bounded to N=1,2"))
    end
    adjoint(transform)*schur*transform
end

# The system PI factor is first and fastest. One consecutive PI slice belongs
# to one column-major finite-mode matrix unit.
function _global_pm_reconstruct_composite(
        basis::CompositePIBasis,coefficients::AbstractVector)
    length(basis.factors)==2||throw(ArgumentError(
        "the test oracle supports exactly one global mode"))
    system_basis=basis.factors[1]
    mode_basis=basis.factors[2]
    system_basis isa PIBasis||throw(ArgumentError(
        "the first factor must be the PI system"))
    mode_basis isa FiniteOperatorBasis||throw(ArgumentError(
        "the second factor must be the finite mode"))
    system_coordinates=length(system_basis)
    mode_levels=mode_basis.d
    system_hilbert=system_basis.d^system_basis.N
    length(coefficients)==system_coordinates*mode_levels^2||
        throw(DimensionMismatch("composite coefficient length mismatch"))
    T=eltype(coefficients)
    result=zeros(
        T,system_hilbert*mode_levels,system_hilbert*mode_levels)
    mode_unit=zeros(T,mode_levels,mode_levels)
    for mode_column in 1:mode_levels,mode_row in 1:mode_levels
        mode_coordinate=
            mode_row+(mode_column-1)*mode_levels
        system_slice=view(
            coefficients,
            (mode_coordinate-1)*system_coordinates+1:
                mode_coordinate*system_coordinates)
        system_operator=_global_pm_reconstruct_qubit_pi(
            system_basis,system_slice)
        fill!(mode_unit,zero(T))
        mode_unit[mode_row,mode_column]=one(T)
        result.+=kron(system_operator,mode_unit)
    end
    result
end

function _global_pm_reconstruct_one_supersite(
        coefficients::AbstractVector,dimension::Int)
    length(coefficients)==dimension^2||throw(DimensionMismatch(
        "one-supersite coefficient length mismatch"))
    T=eltype(coefficients)
    transform=reverse(Matrix{T}(I,dimension,dimension);dims=1)
    adjoint(transform)*reshape(coefficients,dimension,dimension)*transform
end

function _global_pm_local_operator(
        operator::AbstractMatrix,N::Int,active::Int)
    dimension=size(operator,1)
    identity=Matrix{eltype(operator)}(I,dimension,dimension)
    reduce(kron,
        (site==active ? operator : identity for site in 1:N))
end

function _global_pm_collective(operator::AbstractMatrix,N::Int)
    dimension=size(operator,1)^N
    sum((_global_pm_local_operator(operator,N,active)
         for active in 1:N);
        init=zeros(eltype(operator),dimension,dimension))
end

function _global_pm_dissipator(jump,density)
    product=adjoint(jump)*jump
    jump*density*adjoint(jump)-
        (product*density+density*product)/2
end

_global_pm_system_trace_allocations(output,rho,model)=
    @allocated(trace_pseudomodes!(output,rho,model))
_global_pm_mode_trace_allocations(output,rho,model)=
    @allocated(global_pseudomode_state!(output,rho,model))

function _global_pm_full_action(
        density,N,system_hamiltonian,independent_jump,
        collective_jump,mode,coupling;
        system_rate,independent_rate,collective_rate,
        frequency=mode.frequency,damping=mode.damping,
        occupation=mode.thermal_occupation)
    T=eltype(density)
    system_hilbert=size(system_hamiltonian,1)^N
    mode_levels=mode.levels
    system_identity=Matrix{T}(I,system_hilbert,system_hilbert)
    mode_identity=Matrix{T}(I,mode_levels,mode_levels)
    system_sum=_global_pm_collective(system_hamiltonian,N)
    coupling_sum=_global_pm_collective(coupling.operator,N)
    annihilation=Matrix{T}(mode.annihilation)
    creation=Matrix{T}(mode.creation)
    number_operator=Matrix{T}(mode.number_operator)
    g=T(coupling.strength)
    h=T(coupling.counterrotating_strength)
    hamiltonian=
        system_rate*kron(system_sum,mode_identity)+
        frequency*kron(system_identity,number_operator)+
        g*kron(coupling_sum,creation)+
        conj(g)*kron(adjoint(coupling_sum),annihilation)+
        h*kron(coupling_sum,annihilation)+
        conj(h)*kron(adjoint(coupling_sum),creation)
    output=-im*(hamiltonian*density-density*hamiltonian)
    for active in 1:N
        jump=kron(
            _global_pm_local_operator(independent_jump,N,active),
            mode_identity)
        output.+=independent_rate*
            _global_pm_dissipator(jump,density)
    end
    collective=kron(
        _global_pm_collective(collective_jump,N),mode_identity)
    output.+=collective_rate*
        _global_pm_dissipator(collective,density)
    loss=kron(system_identity,annihilation)
    gain=kron(system_identity,creation)
    output.=
        output+
        damping*(occupation+1)*
            _global_pm_dissipator(loss,density)+
        damping*occupation*
            _global_pm_dissipator(gain,density)
    output
end

function _global_pm_sparse_oracle(
        system_model::PIModel,mode::BosonicPseudomode,
        coupling::PseudomodeCoupling;
        frequency=mode.frequency,damping=mode.damping,
        occupation=mode.thermal_occupation)
    system_basis=system_model.basis
    mode_basis=FiniteOperatorBasis(
        mode.levels;label=mode.label)
    system_coordinates=length(system_basis)
    mode_coordinates=length(mode_basis)
    R=promote_type(
        typeof(real(zero(eltype(mode)))),
        typeof(real(zero(eltype(coupling)))))
    T=Complex{R}
    system_identity=spdiagm(
        0=>fill(one(T),system_coordinates))
    mode_identity=spdiagm(
        0=>fill(one(T),mode_coordinates))
    result=kron(
        mode_identity,
        sparse(liouvillian(
            system_model;representation=:sparse)))
    result+=frequency*kron(
        commutator_superoperator(mode.number_operator),
        system_identity)
    result+=damping*(occupation+1)*kron(
        dissipator_superoperator(mode.annihilation),
        system_identity)
    result+=damping*occupation*kron(
        dissipator_superoperator(mode.creation),
        system_identity)

    collective=collective_operator(
        system_basis,coupling.operator)
    interactions=(
        (coupling.strength,collective,mode.creation),
        (conj(coupling.strength),adjoint(collective),
         mode.annihilation),
        (coupling.counterrotating_strength,collective,
         mode.annihilation),
        (conj(coupling.counterrotating_strength),
         adjoint(collective),mode.creation),
    )
    for (coefficient,system_operator,mode_operator) in interactions
        iszero(coefficient)&&continue
        result+=-im*coefficient*(
            kron(
                left_superoperator(mode_operator),
                factor_left_superoperator(
                    system_basis,system_operator))-
            kron(
                right_superoperator(mode_operator),
                factor_right_superoperator(
                    system_basis,system_operator)))
    end
    sparse(result)
end

@testset "single global pseudomode scaling and topology" begin
    lowering=ComplexF64[0 1;0 0]
    zero_hamiltonian=zeros(ComplexF64,2,2)
    mode=BosonicPseudomode(
        2;frequency=0.4,damping=0.2,
        thermal_occupation=0.1,label=:shared)
    coupling=PseudomodeCoupling(
        lowering;mode=:shared,strength=0.12)
    model=global_pseudomode_model(
        4,zero_hamiltonian,mode;couplings=coupling)

    @test model isa GlobalPseudomodeModel
    @test model.system_basis===model.basis.factors[1]
    @test model.mode_basis===model.basis.factors[2]
    @test model.mode_basis.d==3
    @test model.basis.dimensions==(35,9)
    @test length(model.basis)==315
    @test BigInt(length(model.basis))==
        commutant_dimension(4,2)*mode.levels^2
    @test commutant_dimension(4,2)*mode.levels^2==
        315
    @test commutant_dimension(4,2*mode.levels)==82251
    @test length(model.basis)<
        commutant_dimension(4,2*mode.levels)
    @test model.metadata.embedding===:single_global_pseudomode
    @test model.metadata.coordinate_order===
        :system_pi_fastest_then_global_mode
    @test model.metadata.coupling_convention===
        :collective_sum_without_kac_scaling
    @test length(model.damping_channels)==2
    @test length(model.generator.terms)>
        length(model.background.terms)
    @test model.resource_estimates.coordinate_dimension==
        length(model.basis)
    @test model.resource_estimates.workspace_upper_bytes>0
    @test model.resource_estimates.coupling_operator_upper_bytes>0
    @test model.resource_estimates.reduction_plan_bytes>0
    @test model.system_reduction_plan isa
        PermutationalInvariantDynamics.CompositeReductionPlan
    @test model.mode_reduction_plan isa
        PermutationalInvariantDynamics.CompositeReductionPlan
    @test model.system_reduction_plan.kept_factor==1
    @test model.mode_reduction_plan.kept_factor==2
    @test pi_dimension(model)==length(model.basis)

    for term in model.generator.terms
        for (factor,action) in pairs(term.actions)
            action===nothing&&continue
            @test size(action)==
                (model.basis.dimensions[factor],
                 model.basis.dimensions[factor])
            @test size(action)!=
                (length(model.basis),length(model.basis))
        end
    end

    symmetric_basis=PIBasis(4,2;sectors=[(4,0)])
    symmetric_system=PIModel(
        symmetric_basis,
        (CollectiveHamiltonian(
            ComplexF64[1 0;0 -1];rate=0.1),))
    symmetric=global_pseudomode_model(
        symmetric_system,mode;couplings=coupling)
    @test symmetric.system_basis===symmetric_basis
    @test symmetric.basis.dimensions==
        (length(symmetric_basis),mode.levels^2)
    @test length(symmetric.basis)==
        length(symmetric_basis)*mode.levels^2

    alias=shared_pseudomode_model(
        PIModel(PIBasis(1,2),()),mode)
    @test alias isa GlobalPseudomodeModel
    overloaded=pseudomode_model(
        alias.system_model,mode)
    @test overloaded isa GlobalPseudomodeModel
    topology=pseudomode_model(
        1,zero_hamiltonian,mode;topology=:global)
    @test topology isa GlobalPseudomodeModel

    zero_coupling=PseudomodeCoupling(
        lowering;mode=:shared,strength=0.0,
        counterrotating_strength=0.0)
    zero_model=global_pseudomode_model(
        4,zero_hamiltonian,mode;couplings=zero_coupling)
    @test only(zero_model.coupling_operators).collective===nothing
    @test zero_model.resource_estimates.coupling_operator_upper_bytes==0

    @test_throws ArgumentError pseudomode_model(
        1,zero_hamiltonian,
        (mode,BosonicPseudomode(1;label=:second));
        topology=:global)
    @test_throws ArgumentError pseudomode_model(
        1,zero_hamiltonian,mode;
        topology=:global,
        supersite_terms=(LocalJump(lowering;rate=0.1),))
    @test_throws ArgumentError global_pseudomode_model(
        0,zero_hamiltonian,mode)
    @test_throws ArgumentError global_pseudomode_model(
        4,zero_hamiltonian,mode;
        couplings=PseudomodeCoupling(
            lowering;mode=:other,strength=0.1))
    @test_throws DimensionMismatch global_pseudomode_model(
        PIModel(PIBasis(1,2),()),mode;
        couplings=PseudomodeCoupling(
            ones(ComplexF64,3,3);mode=:shared))
    @test_throws ArgumentError global_pseudomode_model(
        4,zero_hamiltonian,mode;
        couplings=coupling,memory_budget=1)
end

@testset "global pseudomode bounded N=2 oracles" begin
    N=2
    basis=PIBasis(N,2)
    z=ComplexF64[1 0;0 -1]
    lowering=ComplexF64[0 1;0 0]
    system_rate=0.13
    independent_rate=0.07
    collective_rate=0.025
    system_model=PIModel(
        basis,(
            CollectiveHamiltonian(z;rate=system_rate),
            LocalJump(lowering;rate=independent_rate),
            CollectiveJump(lowering;rate=collective_rate),
        ))
    mode=BosonicPseudomode(
        1;frequency=0.71,damping=0.31,
        thermal_occupation=0.23,label=:cavity)
    coupling=PseudomodeCoupling(
        lowering;mode=:cavity,strength=0.21+0.08im,
        counterrotating_strength=-0.04+0.03im)
    model=global_pseudomode_model(
        system_model,mode;couplings=coupling)
    reference=_global_pm_sparse_oracle(
        system_model,mode,coupling)

    source=ComplexF64[
        sin(0.17index)+im*cos(0.23index)
        for index in 1:length(model.basis)]
    forward=similar(source)
    adjoint_output=similar(source)
    workspace=global_pseudomode_workspace(model)
    nested_workspace_bytes=
        PermutationalInvariantDynamics._performance_liouvillian_workspace_bytes(
            model.system_plan)
    @test model.resource_estimates.workspace_upper_bytes>=
        nested_workspace_bytes
    apply!(forward,model,source,workspace)
    apply_adjoint!(
        adjoint_output,model,source,workspace)
    @test forward≈reference*source atol=8e-12 rtol=8e-12
    @test isapprox(
        adjoint_output,adjoint(reference)*source;
        atol=8e-12,rtol=8e-12)
    @test @allocated(apply!(
        forward,model,source,workspace))==0
    @test @allocated(apply_adjoint!(
        adjoint_output,model,source,workspace))==0

    wrapped=global_pseudomode_matrixfree(
        model;workspace)
    @test model.trace_vector isa SparseVector
    @test wrapped.tracevec isa SparseVector
    @test collect(model.trace_vector)==
        composite_trace_vector(
            model.basis;T=typeof(real(zero(eltype(model)))))
    @test model.resource_estimates.trace_vector_bytes<
        model.resource_estimates.state_bytes
    wrapper_bytes=
        model.resource_estimates.workspace_upper_bytes+
        model.resource_estimates.trace_vector_bytes
    @test_throws ArgumentError global_pseudomode_matrixfree(
        model;memory_budget=wrapper_bytes-1)
    @test_throws ArgumentError global_pseudomode_matrixfree(
        model;workspace,memory_budget=wrapper_bytes-1)
    @test wrapped.plan===nothing
    @test wrapped.adjoint_action! !== nothing
    @test recommend_solver(
        wrapped;task=:steady_state).algorithm===:gmres
    @test recommend_solver(
        wrapped;task=:spectrum).algorithm===:arnoldi
    @test_throws ArgumentError recommend_solver(
        wrapped;task=:steady_state,algorithm=:direct)
    @test isapprox(
        wrapped*source,reference*source;
        atol=8e-12,rtol=8e-12)
    @test isapprox(
        adjoint(wrapped)*source,adjoint(reference)*source;
        atol=8e-12,rtol=8e-12)
    batch=hcat(source,reverse(source))
    batch_forward=similar(batch)
    batch_adjoint=similar(batch)
    apply!(batch_forward,wrapped,batch,0.0,nothing)
    apply_adjoint!(
        batch_adjoint,wrapped,batch,0.0,nothing)
    @test isapprox(
        batch_forward,reference*batch;
        atol=8e-12,rtol=8e-12)
    @test isapprox(
        batch_adjoint,adjoint(reference)*batch;
        atol=8e-12,rtol=8e-12)
    @test isapprox(
        liouvillian(model;representation=:matrixfree)*source,
        reference*source;atol=8e-12,rtol=8e-12)
    @test_throws ArgumentError liouvillian(
        model;representation=:sparse)

    trajectory_plan=CompositeTrajectoryPlan(
        model.background,model.damping_channels...)
    trajectory_output=similar(source)
    apply!(
        trajectory_output,trajectory_plan.generator,source,
        CompositeSuperoperatorWorkspace(
            trajectory_plan.generator,source))
    @test isapprox(
        trajectory_output,forward;
        atol=8e-12,rtol=8e-12)

    reconstructed_source=_global_pm_reconstruct_composite(
        model.basis,source)
    reconstructed_forward=_global_pm_reconstruct_composite(
        model.basis,forward)
    full_reference=_global_pm_full_action(
        reconstructed_source,N,z,lowering,lowering,
        mode,coupling;
        system_rate,independent_rate,collective_rate)
    @test isapprox(
        reconstructed_forward,full_reference;
        atol=9e-12,rtol=9e-12)
    @test isapprox(
        LinearAlgebra.tr(reconstructed_forward),0;
        atol=3e-12,rtol=3e-12)
    @test isapprox(
        dot(model.trace_vector,forward),0;
        atol=3e-12,rtol=3e-12)

    system_density=ComplexF64[
        0.68 0.04+0.07im
        0.04-0.07im 0.32
    ]
    mode_density=ComplexF64[
        0.73 0.11im
        -0.11im 0.27
    ]
    system_state=iid_state(basis,system_density)
    product=pseudomode_product_state(
        model,system_state;mode_state=mode_density)
    @test trace(product)≈1 atol=3e-13 rtol=3e-13
    @test product.data≈composite_tensor_state(
        model.basis,system_state,mode_density).data
    reduced_system=trace_pseudomodes(product,model)
    reduced_mode=global_pseudomode_state(product,model)
    @test reduced_system.basis===basis
    @test isapprox(
        reduced_system.data,system_state.data;
        atol=3e-13,rtol=3e-13)
    @test isapprox(
        reduced_mode,mode_density;
        atol=3e-13,rtol=3e-13)
    system_buffer=PIState(basis)
    mode_buffer=zeros(ComplexF64,mode.levels,mode.levels)
    @test trace_pseudomodes!(
        system_buffer,product,model)===system_buffer
    @test global_pseudomode_state!(
        mode_buffer,product,model)===mode_buffer
    _global_pm_system_trace_allocations(
        system_buffer,product,model)
    _global_pm_mode_trace_allocations(
        mode_buffer,product,model)
    @test _global_pm_system_trace_allocations(
        system_buffer,product,model)==0
    @test _global_pm_mode_trace_allocations(
        mode_buffer,product,model)==0
    @test system_buffer.data≈system_state.data
    @test mode_buffer≈mode_density

    vacuum_product=pseudomode_product_state(
        model,ComplexF64[1,0])
    @test global_pseudomode_state(
        vacuum_product,model)≈
        mode.vacuum*mode.vacuum'
    @test_throws DimensionMismatch pseudomode_product_state(
        model,ones(ComplexF64,3))
    @test_throws DimensionMismatch pseudomode_product_state(
        model,ComplexF64[1,0];
        mode_state=ones(ComplexF64,3))
    @test_throws ArgumentError pseudomode_product_state(
        model,ComplexF64[1,0];memory_budget=1)
end

@testset "global pseudomode stationary-state integration" begin
    lowering=ComplexF64[0 1;0 0]
    excited=ComplexF64[0,1]
    mode=BosonicPseudomode(
        1;frequency=0.37,damping=0.41,
        thermal_occupation=0.0,label=:relaxing_mode)
    system=PIModel(
        PIBasis(1,2),
        (LocalJump(lowering;rate=0.29),))
    model=global_pseudomode_model(system,mode)
    initial=pseudomode_product_state(
        model,excited;mode_state=excited)
    stationary_coordinates=steady_state(
        model;method=:krylov,initial_state=initial,
        krylovdim=12,maxiter=300,
        atol=1e-11,rtol=1e-9)
    @test stationary_coordinates isa Vector{ComplexF64}
    stationary_result=stationary_state(
        model;
        initial_state=initial,krylovdim=12,maxiter=300,
        atol=1e-11,rtol=1e-9,return_info=true)
    stationary=stationary_result.state
    @test stationary_result.info.selected_algorithm===:gmres
    @test recommend_solver(
        model;task=:steady_state).algorithm===:gmres
    @test_throws ArgumentError stationary_state(
        model;algorithm=DirectAlgorithm())
    @test_throws ArgumentError steady_state(
        model;method=:direct)
    @test stationary isa CompositePIState
    @test stationary.basis===model.basis
    @test trace(stationary)≈1 atol=2e-10 rtol=2e-10
    @test stationary.data≈stationary_coordinates atol=2e-9 rtol=2e-9
    residual=similar(stationary.data)
    apply!(
        residual,model,stationary.data,
        global_pseudomode_workspace(model))
    @test norm(residual)<=2e-9

    spectrum=liouvillian_spectrum(
        model;algorithm=:arnoldi,nev=2,krylovdim=16,
        return_info=true)
    @test spectrum isa SpectrumResult
    @test length(spectrum.values)==2
    @test spectrum.info.selected_algorithm===:arnoldi
    @test minimum(abs,spectrum.values)<=1e-10
    @test_throws ArgumentError liouvillian_spectrum(
        model;algorithm=:dense,nev=2)
end

@testset "N=1 global and local pseudomode physical equivalence" begin
    z=ComplexF64[1 0;0 -1]
    lowering=ComplexF64[0 1;0 0]
    mode=BosonicPseudomode(
        2;frequency=0.63,damping=0.24,
        thermal_occupation=0.17,label=:mode)
    coupling=PseudomodeCoupling(
        lowering;mode=:mode,strength=0.19-0.07im,
        counterrotating_strength=0.05+0.02im)
    jump=LocalJump(lowering;rate=0.08)
    global_model=global_pseudomode_model(
        1,z,mode;system_rate=0.12,
        system_terms=(jump,),couplings=coupling)
    local_model=pseudomode_model(
        1,z,mode;system_rate=0.12,
        system_terms=(jump,),couplings=coupling)

    system_density=ComplexF64[
        0.61 0.08+0.03im
        0.08-0.03im 0.39
    ]
    mode_ket=normalize(ComplexF64[1,0.3im,-0.17])
    mode_density=mode_ket*mode_ket'
    global_state=pseudomode_product_state(
        global_model,system_density;mode_state=mode_density)
    local_state=pseudomode_product_state(
        local_model.supersite,system_density;
        mode_states=(mode_density,))

    global_output=similar(global_state.data)
    apply!(
        global_output,global_model,global_state.data,
        global_pseudomode_workspace(global_model))
    local_operator=liouvillian(
        local_model.model;representation=:matrixfree)
    local_output=local_operator*local_state.data
    global_physical=_global_pm_reconstruct_composite(
        global_model.basis,global_output)
    local_physical=_global_pm_reconstruct_one_supersite(
        local_output,2mode.levels)
    @test isapprox(
        global_physical,local_physical;
        atol=7e-12,rtol=7e-12)

    source_physical=_global_pm_reconstruct_composite(
        global_model.basis,global_state.data)
    full_reference=_global_pm_full_action(
        source_physical,1,z,lowering,lowering,
        mode,coupling;
        system_rate=0.12,independent_rate=0.08,
        collective_rate=0.0)
    @test isapprox(
        global_physical,full_reference;
        atol=7e-12,rtol=7e-12)
end

@testset "global pseudomode scalar precision" begin
    z32=ComplexF32[1 0;0 -1]
    lowering32=ComplexF32[0 1;0 0]
    mode32=BosonicPseudomode(
        1;frequency=0.4f0,damping=0.2f0,
        label=:mode32)
    coupling32=PseudomodeCoupling(
        lowering32;mode=:mode32,strength=0.1f0+0.03f0im)
    model32=global_pseudomode_model(
        1,z32,mode32;couplings=coupling32)
    workspace32=global_pseudomode_workspace(model32)
    @test eltype(model32)===ComplexF32
    @test eltype(model32.generator)===ComplexF32
    @test eltype(workspace32.buffer1)===ComplexF32
    @test eltype(global_pseudomode_matrixfree(model32))===
        ComplexF32
    converted_product=pseudomode_product_state(
        model32,ComplexF32[1,0])
    @test eltype(converted_product)===ComplexF32
    @test_throws ArgumentError pseudomode_product_state(
        model32,ComplexF64[1,0])

    prepared=setprecision(BigFloat,256) do
        z=Complex{BigFloat}[1 0;0 -1]
        lowering=Complex{BigFloat}[0 1;0 0]
        mode=BosonicPseudomode(
            1;frequency=big"0.41",damping=big"0.19",
            thermal_occupation=big"0.07",
            label=:wide,T=BigFloat)
        coupling=PseudomodeCoupling(
            lowering;mode=:wide,
            strength=big"0.13"+big"0.02"*im,
            counterrotating_strength=big"0.01")
        system_model=PIModel(
            PIBasis(1,2),(
                CollectiveHamiltonian(z;rate=big"0.17"),
                LocalJump(lowering;rate=big"0.03"),
            ))
        model=global_pseudomode_model(
            system_model,mode;couplings=coupling)
        source=Complex{BigFloat}[
            sin(BigFloat(index)/17)+
            im*cos(BigFloat(index)/19)
            for index in 1:length(model.basis)]
        destination=copy(source)
        reference=_global_pm_sparse_oracle(
            system_model,mode,coupling)
        expected=reference*source
        product=pseudomode_product_state(
            model,Complex{BigFloat}[1,0])
        (;model,source,destination,expected,product)
    end
    @test prepared.model.precision_bits==256

    wide_workspace=setprecision(BigFloat,64) do
        global_pseudomode_workspace(prepared.model)
    end
    @test minimum(
        value->max(
            precision(real(value)),precision(imag(value))),
        wide_workspace.buffer1)==256
    setprecision(BigFloat,64) do
        apply!(
            prepared.destination,prepared.model,
            prepared.source,wide_workspace)
    end
    @test isapprox(
        prepared.destination,prepared.expected;
        atol=big"1e-60",rtol=big"1e-60")
    @test minimum(
        value->max(
            precision(real(value)),precision(imag(value))),
        prepared.destination)==256

    reduced_system,reduced_mode=setprecision(BigFloat,64) do
        (
            trace_pseudomodes(
                prepared.product,prepared.model),
            global_pseudomode_state(
                prepared.product,prepared.model),
        )
    end
    high_ambient_system,high_ambient_mode=
        setprecision(BigFloat,384) do
            (
                trace_pseudomodes(
                    prepared.product,prepared.model),
                global_pseudomode_state(
                    prepared.product,prepared.model),
            )
        end
    @test minimum(
        value->max(
            precision(real(value)),precision(imag(value))),
        reduced_system.data)==256
    @test minimum(
        value->max(
            precision(real(value)),precision(imag(value))),
        reduced_mode)==256
    @test extrema(
        value->max(
            precision(real(value)),precision(imag(value))),
        high_ambient_system.data)==(256,256)
    @test extrema(
        value->max(
            precision(real(value)),precision(imag(value))),
        high_ambient_mode)==(256,256)
    @test high_ambient_system.data==reduced_system.data
    @test high_ambient_mode==reduced_mode

    relaxing=setprecision(BigFloat,256) do
        lowering=Complex{BigFloat}[0 1;0 0]
        mode=BosonicPseudomode(
            1;frequency=BigFloat("0.37"),
            damping=BigFloat("0.41"),
            label=:wide_relaxing,T=BigFloat)
        system=PIModel(
            PIBasis(1,2),
            (LocalJump(
                lowering;rate=BigFloat("0.29")),))
        model=global_pseudomode_model(system,mode)
        excited=Complex{BigFloat}[0,1]
        initial=pseudomode_product_state(
            model,excited;mode_state=excited)
        (;model,initial,tolerance=BigFloat("1e-40"))
    end
    low_level=setprecision(BigFloat,64) do
        steady_state(
            relaxing.model;method=:krylov,
            initial_state=relaxing.initial,
            krylovdim=16,maxiter=300,
            atol=relaxing.tolerance,
            rtol=relaxing.tolerance)
    end
    high_level=setprecision(BigFloat,64) do
        stationary_state(
            relaxing.model;
            initial_state=relaxing.initial,
            krylovdim=16,maxiter=300,
            atol=relaxing.tolerance,
            rtol=relaxing.tolerance)
    end
    @test high_level isa CompositePIState
    @test minimum(
        value->max(
            precision(real(value)),precision(imag(value))),
        low_level)==256
    @test minimum(
        value->max(
            precision(real(value)),precision(imag(value))),
        high_level.data)==256
    low_residual=setprecision(BigFloat,256) do
        output=similar(low_level)
        apply!(
            output,relaxing.model,low_level,
            global_pseudomode_workspace(relaxing.model))
        norm(output)
    end
    high_residual=setprecision(BigFloat,256) do
        output=similar(high_level.data)
        apply!(
            output,relaxing.model,high_level.data,
            global_pseudomode_workspace(relaxing.model))
        norm(output)
    end
    @test low_residual<=relaxing.tolerance
    @test high_residual<=relaxing.tolerance

    @test_throws ArgumentError global_pseudomode_workspace(
        model32;T=BigFloat)
end
