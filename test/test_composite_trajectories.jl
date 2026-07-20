@testset "Composite stochastic systems" begin
    sm=ComplexF64[0 1;0 0]
    excited=ComplexF64[0 0;0 1]
    ground=ComplexF64[1 0;0 0]

    pi_basis=PIBasis(1,2)
    finite_basis=FiniteOperatorBasis(2;label=:ancilla)
    basis=CompositePIBasis(pi_basis,finite_basis)
    rho0=composite_tensor_state(
        basis,iid_state(pi_basis,excited),excited)
    Jpi=collective_operator(pi_basis,sm)
    channel=CompositeJumpChannel(
        basis,1=>Jpi,2=>sm;rate=0.4,label=:pair_loss)
    plan=CompositeTrajectoryPlan(basis,channel)

    @test plan.basis===basis
    @test plan.channels[1].label===:pair_loss
    @test isautonomous(plan)
    @test size(plan)==(length(basis),length(basis))
    @test eltype(plan)===ComplexF64
    @test composite_master_superoperator(plan)===plan.generator
    @test trace(rho0)≈1

    empty_plan=CompositeTrajectoryPlan(basis;T=Float64)
    empty_path=quantum_trajectory(
        empty_plan,rho0,[0.0,0.1];dt=0.01,rng=MersenneTwister(9))
    @test empty_path.states[end].data≈rho0.data atol=2e-14
    zero_channel=CompositeJumpChannel(basis,2=>sm;rate=0.0)
    zero_plan=CompositeTrajectoryPlan(basis,zero_channel)
    zero_path=quantum_trajectory(
        zero_plan,rho0,[0.0,0.1];dt=0.01,rng=MersenneTwister(10))
    @test isempty(zero_path.jump_times)
    @test zero_path.states[end].data≈rho0.data atol=2e-14

    reference=composite_dissipator_superoperator(
        basis,1=>Jpi,2=>sm;rate=0.4)
    destination=zeros(ComplexF64,length(basis))
    reference_destination=similar(destination)
    apply!(destination,plan.generator,rho0.data,0.0,nothing,
           CompositeSuperoperatorWorkspace(plan.generator,rho0.data))
    apply!(reference_destination,reference,rho0.data,0.0,nothing,
           CompositeSuperoperatorWorkspace(reference,rho0.data))
    @test destination≈reference_destination atol=4e-14 rtol=4e-14
    @test abs(trace(CompositePIOperator(basis,destination)))<4e-14

    workspace=CompositeTrajectoryWorkspace(plan,rho0)
    @test sum(length,(workspace.tmp,workspace.k1,workspace.k2))==
          3length(rho0.data)
    @test !hasproperty(workspace,:k3)&&!hasproperty(workspace,:k4)
    internal=PermutationalInvariantDynamics
    rates=internal._composite_channel_intensities!(
        workspace,rho0.data,0.0,nothing)
    @test rates≈[0.4] atol=3e-14
    conditional=similar(rho0.data)
    internal._composite_conditional_action!(
        conditional,rho0.data,workspace,0.0,nothing)
    gain=kron(sandwich_superoperator(sm),
              factor_sandwich_superoperator(pi_basis,Jpi))*rho0.data
    expected_conditional=reference_destination-0.4.*gain+0.4.*rho0.data
    @test conditional≈expected_conditional atol=5e-14 rtol=5e-14
    @test abs(internal._prepared_composite_trace(
        plan.trace_plan,conditional))<4e-14

    # Public constructors own their derived maps. Mutating the raw jump after
    # construction cannot change the prepared stochastic system.
    mutable_jump=copy(sm)
    owned_channel=CompositeJumpChannel(basis,2=>mutable_jump;rate=0.2)
    mutable_jump.=0
    owned_plan=CompositeTrajectoryPlan(basis,owned_channel)
    owned_workspace=CompositeTrajectoryWorkspace(owned_plan,rho0)
    @test internal._composite_channel_intensities!(
        owned_workspace,rho0.data,0.0,nothing)[1]≈0.2

    @testset "single-factor PI equivalence" begin
        composite_basis=CompositePIBasis(pi_basis)
        pi_state=iid_state(pi_basis,excited)
        composite_state=composite_tensor_state(composite_basis,pi_state)
        composite_channel=CompositeJumpChannel(
            composite_basis,1=>Jpi;rate=0.35)
        composite_plan=CompositeTrajectoryPlan(
            composite_basis,composite_channel)
        pi_model=PIModel(pi_basis,(DirectPIJump(Jpi;rate=0.35),))
        times=collect(0.0:0.05:0.5)
        pi_path=quantum_trajectory(
            pi_model,pi_state,times;dt=0.01,rng=MersenneTwister(41))
        composite_path=quantum_trajectory(
            composite_plan,composite_state,times;
            dt=0.01,rng=MersenneTwister(41))
        @test composite_path.jump_times==pi_path.jump_times
        @test composite_path.jump_channels==pi_path.jump_channels
        @test all(index->isapprox(composite_path.states[index].data,
            pi_path.states[index].data;atol=2e-13,rtol=2e-13),
            eachindex(times))
    end

    @testset "multi-sector trace and cross factors" begin
        multi=PIBasis(3,2)
        auxiliary=FiniteOperatorBasis(2)
        multi_basis=CompositePIBasis(multi,auxiliary)
        local_density=ComplexF64[0.62 0.07im;-0.07im 0.38]
        state=composite_tensor_state(
            multi_basis,iid_state(multi,local_density),
            ComplexF64[0.7 0.1;0.1 0.3])
        cross=CompositeJumpChannel(
            multi_basis,
            1=>collective_operator(multi,sm),2=>sm;rate=0.17)
        cross_plan=CompositeTrajectoryPlan(multi_basis,cross)
        @test any(symmetric_group_dimension(partition)>1
                  for partition in multi.sectors)
        @test internal._prepared_composite_trace(
            cross_plan.trace_plan,state.data)≈trace(state) atol=4e-14
        @test internal._prepared_composite_trace(
            cross_plan.trace_plan,state.data)≈dot(
                composite_trace_vector(multi_basis),state.data) atol=4e-14
        cross_workspace=CompositeTrajectoryWorkspace(cross_plan,state)
        cross_rates=internal._composite_channel_intensities!(
            cross_workspace,state.data,0.0,nothing)
        @test length(cross_rates)==1
        @test isfinite(cross_rates[1])&&cross_rates[1]>=0

        two_pi_basis=CompositePIBasis(multi,pi_basis)
        two_pi_state=composite_tensor_state(
            two_pi_basis,iid_state(multi,local_density),
            iid_state(pi_basis,excited))
        two_pi_channel=CompositeJumpChannel(
            two_pi_basis,1=>collective_operator(multi,sm),2=>Jpi;
            rate=0.09)
        two_pi_plan=CompositeTrajectoryPlan(two_pi_basis,two_pi_channel)
        @test internal._prepared_composite_trace(
            two_pi_plan.trace_plan,two_pi_state.data)≈1 atol=4e-14
        @test internal._composite_channel_intensities!(
            CompositeTrajectoryWorkspace(two_pi_plan,two_pi_state),
            two_pi_state.data,0.0,nothing)[1]>=0
    end

    @testset "rates, precision, and validation" begin
        @test_throws ArgumentError CompositeJumpChannel(
            basis,1=>Jpi;rate=-0.1)
        @test_throws ArgumentError CompositeJumpChannel(
            basis,1=>Jpi;rate=0.1im)
        @test_throws ArgumentError CompositeJumpChannel(basis;rate=0.1)
        @test_throws ArgumentError CompositeJumpChannel(
            basis,1=>Jpi,1=>Jpi;rate=0.1)
        @test_throws DimensionMismatch CompositeJumpChannel(
            basis,2=>zeros(3,3);rate=0.1)
        third_rate=CompositeJumpChannel(basis,2=>sm;rate=1//3)
        @test_throws ArgumentError CompositeTrajectoryPlan(basis,third_rate)
        other=CompositePIBasis(PIBasis(1,2),finite_basis)
        @test_throws ArgumentError CompositeTrajectoryPlan(
            other,channel)

        bad_negative=CompositeJumpChannel(
            basis,1=>Jpi;rate=(t,p)->-0.1)
        negative_plan=CompositeTrajectoryPlan(basis,bad_negative)
        @test_throws ArgumentError quantum_trajectory(
            negative_plan,rho0,[0.0,0.1];dt=0.01)
        bad_complex=CompositeJumpChannel(
            basis,1=>Jpi;rate=(t,p)->0.1im)
        complex_plan=CompositeTrajectoryPlan(basis,bad_complex)
        @test_throws ArgumentError quantum_trajectory(
            complex_plan,rho0,[0.0,0.1];dt=0.01)
        bad_nonfinite=CompositeJumpChannel(
            basis,1=>Jpi;rate=(t,p)->Inf)
        nonfinite_plan=CompositeTrajectoryPlan(basis,bad_nonfinite)
        @test_throws ArgumentError quantum_trajectory(
            nonfinite_plan,rho0,[0.0,0.1];dt=0.01)

        scheduled_third=CompositeJumpChannel(
            basis,2=>sm;rate=(t,p)->1//3)
        scheduled_third_plan=CompositeTrajectoryPlan(basis,scheduled_third)
        @test_throws ArgumentError quantum_trajectory(
            scheduled_third_plan,rho0,[0.0,0.1];dt=0.01)
        @test_throws ArgumentError apply!(
            similar(rho0.data),
            composite_master_superoperator(scheduled_third_plan),
            rho0.data,0.0,nothing,
            CompositeSuperoperatorWorkspace(
                composite_master_superoperator(scheduled_third_plan),
                rho0.data))

        driven=CompositeJumpChannel(
            basis,1=>Jpi,2=>sm;rate=(t,p)->p*(1+t))
        driven_plan=CompositeTrajectoryPlan(basis,driven)
        @test !isautonomous(driven_plan)
        driven_path=quantum_trajectory(
            driven_plan,rho0,Float64[0,0.1];dt=0.01,
            parameters=0.2,rng=MersenneTwister(2))
        @test trace(driven_path.states[end])≈1 atol=3e-13

        rate_calls=Ref(0)
        counted=CompositeJumpChannel(
            basis,2=>sm;rate=(t,p)->(rate_calls[]+=1;0.2))
        counted_plan=CompositeTrajectoryPlan(basis,counted)
        counted_workspace=CompositeTrajectoryWorkspace(counted_plan,rho0)
        internal._composite_conditional_action!(
            similar(rho0.data),rho0.data,counted_workspace,0.0,nothing)
        @test rate_calls[]==1

        sharp_channel=CompositeJumpChannel(
            basis,2=>sm;rate=(t,p)->10_000t)
        sharp_plan=CompositeTrajectoryPlan(basis,sharp_channel)
        sharp_workspace=CompositeTrajectoryWorkspace(sharp_plan,rho0)
        sharp_state=copy(rho0.data)
        sharp_h,sharp_hazard,sharp_probability=
            internal._composite_capped_conditional_step!(
                sharp_state,sharp_workspace,0.0,0.01,nothing,0.05,
                -log1p(-0.05))
        @test sharp_h<0.01
        @test sharp_hazard<=-log1p(-0.05)
        @test sharp_probability<=0.05
        @test internal._prepared_composite_trace(
            sharp_plan.trace_plan,sharp_state)≈1 atol=3e-14

        # `log1p` followed by `expm1` may round one ulp upward in Float32.
        # An accepted hazard exactly at the transformed cap must therefore
        # remain usable without weakening the cap for a genuine excess.
        scalar_basis=CompositePIBasis(FiniteOperatorBasis(1))
        scalar_state=CompositePIState(scalar_basis,ComplexF32[1])
        cap32=Float32(1551)/Float32(100001)
        cap_hazard32=-log1p(-cap32)
        scalar_channel=CompositeJumpChannel(
            scalar_basis,1=>ComplexF32[1;;];rate=cap_hazard32)
        scalar_plan=CompositeTrajectoryPlan(scalar_basis,scalar_channel)
        scalar_workspace=CompositeTrajectoryWorkspace(
            scalar_plan,scalar_state)
        _,accepted_hazard32,accepted_probability32=
            internal._composite_capped_conditional_step!(
                copy(scalar_state.data),scalar_workspace,0f0,1f0,nothing,
                cap32,cap_hazard32)
        @test accepted_hazard32<=cap_hazard32
        @test accepted_probability32==cap32
        @test_throws ErrorException internal._composite_capped_conditional_step!(
            copy(scalar_state.data),scalar_workspace,0f0,1f0,nothing,
            prevfloat(cap32),cap_hazard32)

        nonpreserving=CompositeSuperoperator(
            basis,local_superoperator_term(
                basis,2,Matrix{ComplexF64}(I,4,4)))
        nonpreserving_plan=CompositeTrajectoryPlan(
            basis;background=nonpreserving)
        @test_throws ArgumentError quantum_trajectory(
            nonpreserving_plan,rho0,[0.0,0.1];dt=0.01)

        pi32=PIBasis(1,2)
        finite32=FiniteOperatorBasis(2)
        basis32=CompositePIBasis(pi32,finite32)
        excited32=ComplexF32[0 0;0 1]
        sm32=ComplexF32[0 1;0 0]
        rho32=composite_tensor_state(
            basis32,iid_state(pi32,excited32),excited32)
        geometry32=OneBodyGeometry(pi32;T=Float32)
        channel32=CompositeJumpChannel(
            basis32,1=>collective_operator(pi32,sm32;cache=geometry32),2=>sm32;
            rate=Float32(0.2))
        plan32=CompositeTrajectoryPlan(basis32,channel32)
        @test eltype(plan32)===ComplexF32
        path32=quantum_trajectory(
            plan32,rho32,Float32[0,0.1];dt=Float32(0.01),
            rng=MersenneTwister(3))
        @test eltype(path32.states[end].data)===ComplexF32
        @test_throws ArgumentError quantum_trajectory(
            plan32,rho32,Float32[0,0.1];dt=0.01)

        wrong_state=CompositePIState(other,rho0.data)
        @test_throws ArgumentError CompositeTrajectoryWorkspace(
            plan,wrong_state)
        wrong_workspace=CompositeTrajectoryWorkspace(owned_plan,rho0)
        @test_throws ArgumentError quantum_trajectory(
            plan,rho0,[0.0,0.1];dt=0.01,workspace=wrong_workspace)
        nonunit=copy(rho0);nonunit.data .*= 0.9
        @test_throws ArgumentError quantum_trajectory(
            plan,nonunit,[0.0,0.1];dt=0.01)
        @test_throws ArgumentError quantum_trajectory(
            plan,rho0,[0.1,0.0];dt=0.01)
        @test_throws ArgumentError quantum_trajectory(
            plan,rho0,[0.0,0.1];dt=-0.01)
        @test_throws ArgumentError quantum_trajectory(
            plan,rho0,[0.0,0.1];dt=0.01,algorithm=:event)
        @test_throws ArgumentError quantum_trajectories(
            plan,rho0,[0.0,0.1],2;dt=0.01,save_states=false)

        nonhermitian=composite_tensor_operator(
            basis,identity_operator(pi_basis),sm)
        @test_throws ArgumentError quantum_trajectories(
            plan,rho0,[0.0,0.1],2;dt=0.01,
            observables=(bad=nonhermitian,),save_states=false)
    end

    @testset "batches, statistics, and deterministic mean" begin
        times=collect(0.0:0.1:0.8)
        serial=quantum_trajectories(
            plan,rho0,times,64;seed=77,dt=0.02,threaded=false)
        threaded=quantum_trajectories(
            plan,rho0,times,64;seed=77,dt=0.02,threaded=true)
        @test map(path->path.jump_times,serial)==
              map(path->path.jump_times,threaded)
        @test map(path->path.jump_channels,serial)==
              map(path->path.jump_channels,threaded)
        @test all(index->serial[index].states[end].data==
            threaded[index].states[end].data,eachindex(serial))

        batch=CompositeTrajectoryBatchWorkspace(plan,rho0;workers=2)
        first_batch=quantum_trajectories(
            plan,rho0,times,24;seed=19,dt=0.02,workspace=batch)
        second_batch=quantum_trajectories(
            plan,rho0,times,24;seed=19,dt=0.02,workspace=batch)
        @test map(path->path.jump_times,first_batch)==
              map(path->path.jump_times,second_batch)

        identity_pi=identity_operator(pi_basis)
        excitation_observable=composite_tensor_operator(
            basis,identity_pi,excited)
        streamed=quantum_trajectories(
            plan,rho0,times,256;seed=8,dt=0.02,
            observables=(excitation=excitation_observable,),
            save_states=false,jump_statistics=true)
        @test streamed isa TrajectoryEnsembleResult
        @test streamed.trajectories===nothing
        @test streamed.observables.trajectories==256
        @test streamed.jumps.trajectories==256
        @test streamed.jumps.total_jumps<=256

        averages=trajectory_average(serial)
        @test length(averages)==length(times)
        @test all(state->isapprox(trace(state),1;atol=3e-13),averages)
        stats=trajectory_statistics(
            serial;observables=(excitation=excitation_observable,),
            nchannels=1)
        @test stats.jumps.trajectories==length(serial)
        @test haskey(stats.observables.observables,:excitation)
        @test_throws ArgumentError jump_statistics(serial;nchannels=-1)
        @test_throws ArgumentError jump_statistics(serial;nchannels=0)
        malformed_history=CompositeQuantumTrajectory(
            copy(times),copy(serial[1].states),[times[2]],Int[])
        @test_throws ArgumentError jump_statistics([malformed_history])
        malformed_states=copy(serial[1].states)
        other_basis=CompositePIBasis(PIBasis(1,2),finite_basis)
        malformed_states[end]=CompositePIState(
            other_basis,malformed_states[end].data)
        malformed_path=CompositeQuantumTrajectory(
            copy(times),malformed_states,copy(serial[1].jump_times),
            copy(serial[1].jump_channels))
        @test_throws ArgumentError trajectory_average([malformed_path])
        @test_throws ArgumentError trajectory_average(
            CompositeQuantumTrajectory[])

        # This one-jump process has exact survival exp(-gamma*t). Its Monte
        # Carlo mean is compared with the independently propagated master
        # generator at a statistical, not roundoff, tolerance.
        ensemble=quantum_trajectories(
            plan,rho0,[0.0,0.8],2000;seed=1234,dt=0.02)
        stochastic_final=trajectory_average(ensemble)[end]
        deterministic_final=time_evolve(
            composite_master_superoperator(plan),rho0,(0.0,0.8);steps=320)
        @test norm(stochastic_final.data-deterministic_final.data)<0.045
    end

    @testset "prepared memory and warmed application" begin
        second_channel=CompositeJumpChannel(
            basis,2=>sm;rate=0.07,label=:auxiliary_loss)
        two_channel_plan=CompositeTrajectoryPlan(
            basis,channel,second_channel)
        two_channel_workspace=CompositeTrajectoryWorkspace(
            two_channel_plan,rho0)
        conditional=similar(rho0.data)
        internal._composite_conditional_action!(
            conditional,rho0.data,two_channel_workspace,0.0,nothing)
        @test @allocated(internal._composite_conditional_action!(
            conditional,rho0.data,two_channel_workspace,0.0,nothing))<=4096
        @test length(two_channel_workspace.tensor_buffer1)==length(basis)
        @test length(two_channel_workspace.tensor_buffer2)==length(basis)
        @test length(two_channel_workspace.channel_work)==2
    end
end
