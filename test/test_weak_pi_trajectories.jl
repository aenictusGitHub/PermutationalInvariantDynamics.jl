function _weak_pi_hot_allocations(plan,state)
    work=WeakPITrajectoryWorkspace(plan,state)
    copyto!(work.current,state.data)
    PermutationalInvariantDynamics._weak_conditional_rk4!(
        work.current,work,0.0,0.001,nothing)
    rk4=@allocated PermutationalInvariantDynamics._weak_conditional_rk4!(
        work.current,work,0.0,0.001,nothing)
    PermutationalInvariantDynamics._weak_channel_intensities!(
        work,work.current,0.0,nothing)
    PermutationalInvariantDynamics._weak_branch_intensities!(work,work.current)
    branches=@allocated PermutationalInvariantDynamics.
        _weak_branch_intensities!(work,work.current)
    rk4,branches
end

@testset "weak-PI pseudo-ket trajectories" begin
    sm=ComplexF64[0 1;0 0]
    b=PIBasis(3,2)
    rho0=iid_pure_state(b,ComplexF64[0,1])
    psi0=@inferred weak_pi_pseudoket(rho0)

    @test weak_pi_dimension(b)==sum(length,b.patterns)
    @test length(psi0)==weak_pi_dimension(b)<2^b.N
    @test norm(psi0.data)≈1
    @test weak_pi_density(psi0).data≈rho0.data atol=2e-14
    @test weak_pi_expectation(psi0,adjoint(sm)*sm)≈b.N atol=2e-13
    excitation_plan=CollectiveObservablePlan(b,adjoint(sm)*sm)
    @test weak_pi_expectation(psi0,excitation_plan)≈b.N atol=2e-13

    # Exercise every retained sector and nontrivial Schur multiplicity in the
    # history-free density sampler independently of stochastic jump timing.
    multisector_data=ComplexF64.(1:weak_pi_dimension(b))
    multisector_data./=norm(multisector_data)
    multisector_state=WeakPIPseudoKet(b,multisector_data)
    empty_plan=WeakPITrajectoryPlan(PIModel(b,());T=Float64)
    static_estimate=weak_pi_trajectory_steady_state(
        empty_plan,multisector_state;
        trajectories=2,settling_time=0.01,dt=0.005,return_info=true)
    @test static_estimate.state.data≈
          weak_pi_density(multisector_state).data atol=2e-15
    @test iszero(static_estimate.sample_spread)
    @test iszero(static_estimate.standard_error)
    @test iszero(static_estimate.residual)
    @test trace(static_estimate.state)≈1 atol=2e-15

    mixed=iid_state(b,ComplexF64[0.5 0;0 0.5])
    @test_throws ArgumentError weak_pi_pseudoket(mixed)
    @test_throws ArgumentError WeakPIPseudoKet(b,2 .* psi0.data)
    invalid=copy(psi0.data);invalid[1]=Complex(NaN,0)
    @test_throws ArgumentError WeakPIPseudoKet(b,invalid)

    model=PIModel(b,(LocalJump(sm;rate=0.7),
                     CollectiveJump(sm;rate=0.2)))
    plan=@inferred WeakPITrajectoryPlan(model)
    @test plan.model===model
    @test length(plan.branch_ranges)==2
    @test !isempty(plan.branch_ranges[1])
    @test !isempty(plan.branch_ranges[2])
    @test any(branch->branch.source_sector!=branch.target_sector,
              view(plan.branches,plan.branch_ranges[1]))
    @test all(branch->branch.source_sector==branch.target_sector,
              view(plan.branches,plan.branch_ranges[2]))
    @test _weak_pi_hot_allocations(plan,psi0)==(0,0)

    # The prepared local subduction Kraus maps close to K'K in every source
    # irrep. This is the representation-theory identity that makes the pure
    # pseudo-ket unraveling reproduce the density-valued PI gain map.
    for channel in eachindex(plan.jumps),source in eachindex(b.sectors)
        dimension=length(b.patterns[source])
        closure=zeros(ComplexF64,dimension,dimension)
        for branch_index in plan.branch_ranges[channel]
            branch=plan.branches[branch_index]
            branch.source_sector==source||continue
            closure .+= adjoint(branch.operator)*branch.operator
        end
        @test closure≈plan.jumps[channel].qblocks[source] atol=3e-12 rtol=3e-12
    end

    # Sum every unnormalized branch outer product and compare directly with
    # the existing density-valued channel gain, before Monte Carlo sampling.
    local_kernel=plan.jumps[1]
    density_plan=plan.density_plan
    density_work=TrajectoryWorkspace(density_plan,rho0)
    density_gain=similar(rho0.data)
    PermutationalInvariantDynamics._apply_gain!(density_gain,rho0.data,
        local_kernel,b,1.0,density_work.liouvillian_work)
    branch_gain=PIState(b;T=Float64)
    for branch_index in plan.branch_ranges[1]
        branch=plan.branches[branch_index]
        source=plan.offsets[branch.source_sector]:(
               plan.offsets[branch.source_sector+1]-1)
        target=plan.offsets[branch.target_sector]:(
               plan.offsets[branch.target_sector+1]-1)
        output=branch.operator*view(psi0.data,source)
        contribution=PermutationalInvariantDynamics.
            _divide_by_schur_multiplicity_scale(
                output*output',Float64,b.sectors[branch.target_sector])
        coefficient_block(branch_gain,b.sectors[branch.target_sector]).+=
            contribution
    end
    @test branch_gain.data≈density_gain atol=4e-12 rtol=4e-12

    @testset "single paths, batches, and statistics" begin
        workspace=WeakPITrajectoryWorkspace(plan,psi0)
        path=weak_pi_quantum_trajectory(plan,psi0,[0.0,0.15];dt=0.01,
            rng=MersenneTwister(41),workspace=workspace)
        @test path isa WeakPIQuantumTrajectory
        @test all(state->norm(state.data)≈1,path.states)
        @test length(path.jump_times)==length(path.jump_channels)==
              length(path.jump_records)
        @test all(record->record.channel in (1,2),path.jump_records)
        @test all(record->weight(record.source_partition)==b.N&&
                          weight(record.target_partition)==b.N,
                  path.jump_records)
        @test all(record->record.child_partition===nothing||
                          weight(record.child_partition)==b.N-1,
                  path.jump_records)

        fixed_workspace=WeakPITrajectoryWorkspace(plan,psi0;mode=:fixed)
        @test all(isempty,(fixed_workspace.k3,fixed_workspace.k4,
            fixed_workspace.k5,fixed_workspace.k6,fixed_workspace.k7,
            fixed_workspace.trial,fixed_workspace.embedded,
            fixed_workspace.start))
        @test sum(length,(fixed_workspace.tmp,fixed_workspace.k1,
            fixed_workspace.k2))==3length(psi0.data)
        @test Base.summarysize(fixed_workspace)<Base.summarysize(workspace)
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            plan,psi0,[0.0,0.05];dt=0.02,algorithm=:event,
            workspace=fixed_workspace,rng=MersenneTwister(1))

        # For one initially excited emitter the continuous-hazard trajectory
        # is exactly a survival process with p_e(t)=exp(-gamma*t).  This tests
        # the event root and normalized pseudo-ket, rather than merely
        # comparing two numerical integrators.
        decay_basis=PIBasis(1,2)
        decay_rho=iid_pure_state(decay_basis,ComplexF64[0,1])
        decay_psi=weak_pi_pseudoket(decay_rho)
        decay_rate=1.3
        decay_plan=WeakPITrajectoryPlan(PIModel(decay_basis,
            (LocalJump(sm;rate=decay_rate),)))

        weak_rate_calls=Ref(0)
        counted_plan=WeakPITrajectoryPlan(PIModel(decay_basis,
            (LocalJump(sm;rate=(t,p)->begin
                weak_rate_calls[]+=1
                decay_rate
            end),)))
        counted_work=WeakPITrajectoryWorkspace(counted_plan,decay_psi)
        hazard,error=PermutationalInvariantDynamics.
            _weak_conditional_dopri_trial!(counted_work,decay_psi.data,
                0.0,0.4,nothing,1e-11,1e-10)
        @test isfinite(hazard)&&isfinite(error)
        @test weak_rate_calls[]==7
        PermutationalInvariantDynamics._prepare_dopri_dense_output!(
            counted_work)
        root=PermutationalInvariantDynamics._dopri_dense_root(
            counted_work,0.4,hazard/2,hazard,1e-12)
        @test root!==nothing
        PermutationalInvariantDynamics._dopri_dense_state!(
            counted_work.tmp,decay_psi.data,counted_work,0.4,root/0.4)
        @test weak_rate_calls[]==7

        combined_plan=WeakPITrajectoryPlan(PIModel(decay_basis,
            (LocalJump(sm;rate=0.7),
             LocalJump(adjoint(sm);rate=0.2))))
        combined_work=WeakPITrajectoryWorkspace(combined_plan,decay_psi)
        combined_rhs=similar(decay_psi.data)
        combined_total=PermutationalInvariantDynamics.
            _weak_conditional_action_and_intensity!(combined_rhs,
                decay_psi.data,combined_work,0.0,nothing)
        reference_rhs=zeros(eltype(decay_psi.data),length(decay_psi.data))
        reference_total=PermutationalInvariantDynamics.
            _weak_channel_intensities_recursive!(combined_work,
                decay_psi.data,0.0,nothing,reference_rhs,Val(true),
                combined_plan.jumps,1)
        @. reference_rhs=reference_rhs+
            (reference_total/2)*decay_psi.data
        @test combined_total≈reference_total atol=2e-14
        @test combined_rhs≈reference_rhs atol=3e-14 rtol=3e-14

        decay_times=[0.0,0.7]
        decay_count=1200
        decay_paths=weak_pi_quantum_trajectories(
            decay_plan,decay_psi,decay_times,decay_count;
            dt=0.3,dtmax=0.5,algorithm=:event,seed=709,
            abstol=1e-10,reltol=1e-9)
        excitation=adjoint(sm)*sm
        sampled_population=sum(real(weak_pi_expectation(
            trajectory.states[end],excitation))
            for trajectory in decay_paths)/decay_count
        exact_population=exp(-decay_rate*decay_times[end])
        sampling_error=sqrt(exact_population*(1-exact_population)/decay_count)
        @test abs(sampled_population-exact_population)<5sampling_error+0.002
        event_times=reduce(vcat,(trajectory.jump_times
            for trajectory in decay_paths);init=Float64[])
        @test !isempty(event_times)
        @test any(time->abs(time/0.3-round(time/0.3))>1e-6,event_times)

        event_path=weak_pi_quantum_trajectory(decay_plan,decay_psi,
            decay_times;dt=0.3,algorithm=:event,
            rng=MersenneTwister(17))
        alias_path=weak_pi_quantum_trajectory(decay_plan,decay_psi,
            decay_times;dt=0.3,algorithm=:adaptive,
            rng=MersenneTwister(17))
        @test event_path.jump_times==alias_path.jump_times
        @test event_path.states[end].data==alias_path.states[end].data

        # Event localization must use the ulp of the absolute time only once.
        # A second factor of `abs(t)` would make this root snap to the trial
        # endpoint when an otherwise identical calculation starts at a large
        # time origin.
        large_origin=1.0e10
        large_seed=1
        expected_delay=randexp(MersenneTwister(large_seed))/decay_rate
        large_time_path=weak_pi_quantum_trajectory(
            decay_plan,decay_psi,
            [large_origin,large_origin+0.7];
            dt=0.3,algorithm=:event,rng=MersenneTwister(large_seed),
            abstol=1e-10,reltol=1e-9)
        @test length(large_time_path.jump_times)==1
        @test isapprox(large_time_path.jump_times[1]-large_origin,
                       expected_delay;atol=10eps(large_origin))

        # The root iteration count must follow the requested tolerance, not a
        # fixed cap.  This trial interval needs more than 60 bisections even
        # though its exactly solvable constant hazard places the event near
        # unit time.
        wide_seed=12
        wide_expected=randexp(MersenneTwister(wide_seed))/decay_rate
        wide_time_path=weak_pi_quantum_trajectory(
            decay_plan,decay_psi,[0.0,1.0e12];
            dt=1.0e12,dtmax=1.0e12,algorithm=:event,
            event_time_tolerance=1.0e-10,
            rng=MersenneTwister(wide_seed),
            abstol=1e-10,reltol=1e-9)
        @test length(wide_time_path.jump_times)==1
        @test isapprox(wide_time_path.jump_times[1],wide_expected;
                       atol=5.0e-10,rtol=0)

        @test_throws ArgumentError weak_pi_quantum_trajectories(
            decay_plan,decay_psi,decay_times,
            BigInt(typemax(Int))+1;dt=0.1)

        batch=WeakPITrajectoryBatchWorkspace(plan,psi0;
            workers=max(2,Threads.nthreads()))
        serial=weak_pi_quantum_trajectories(plan,psi0,[0.0,0.12],9;
            dt=0.01,seed=801,workspace=batch)
        repeated=weak_pi_quantum_trajectories(plan,psi0,[0.0,0.12],9;
            dt=0.01,seed=801,workspace=batch)
        threaded=weak_pi_quantum_trajectories(plan,psi0,[0.0,0.12],9;
            dt=0.01,seed=801,threaded=true,workspace=batch)
        @test all(serial[index].jump_times==repeated[index].jump_times==
                  threaded[index].jump_times&&
                  serial[index].jump_channels==repeated[index].jump_channels==
                  threaded[index].jump_channels&&
                  all(serial[index].states[t].data==repeated[index].states[t].data==
                      threaded[index].states[t].data
                      for t in eachindex(serial[index].states))
                  for index in eachindex(serial))
        @test all(worker->worker.plan===plan,batch.workers)
        @test batch.workers[1].current!==batch.workers[2].current

        # The allocation-free inner accumulation must remain identical to the
        # public equation-(7) scaling route, including its sector weights.
        reference=[PIState(b;T=Float64) for _ in eachindex(serial[1].times)]
        offsets=PermutationalInvariantDynamics._weak_pi_offsets(b)
        for path in serial,time_index in eachindex(path.times)
            for (sector,partition) in pairs(b.sectors)
                psi=view(path.states[time_index].data,
                    PermutationalInvariantDynamics._weak_sector_range(
                        offsets,sector))
                coefficient_block(reference[time_index],partition).+=
                    PermutationalInvariantDynamics.
                        _divide_by_schur_multiplicity_scale(
                            psi*psi',Float64,partition)
            end
        end
        for state in reference
            state.data./=length(serial)
        end
        optimized=weak_pi_trajectory_average(serial)
        @test all(isapprox(optimized[index].data,reference[index].data;
                           atol=2e-15)
                  for index in eachindex(reference))

        # Both unravelings converge to the same matrix-free master equation.
        # The tolerance is stochastic and intentionally much larger than the
        # deterministic integration tolerance.
        times=[0.0,0.35]
        weak_paths=weak_pi_quantum_trajectories(plan,psi0,times,800;
            dt=0.01,seed=99)
        density_paths=quantum_trajectories(model,rho0,times,500;
            dt=0.01,seed=199)
        weak_average=weak_pi_trajectory_average(weak_paths)
        density_average=trajectory_average(density_paths)
        master=time_evolve(liouvillian(model;representation=:matrixfree),
            rho0,(0.0,0.35);steps=500)
        @test norm(weak_average[end].data-master.data)<0.06
        @test norm(density_average[end].data-master.data)<0.06
        @test norm(weak_average[end].data-density_average[end].data)<0.08
        @test abs(trace(weak_average[end])-1)<3e-13

        statistics=weak_pi_trajectory_statistics(weak_paths;
            observables=(excitation=adjoint(sm)*sm,),nchannels=2)
        @test statistics.jumps.total_jumps==
              sum(length(path.jump_times) for path in weak_paths)
        @test sum(value for value in values(
            statistics.jumps.sector_transitions))==
            statistics.jumps.total_jumps
        @test statistics.observables.observables[:excitation].mean[end]≈
              real(collective_expectation(
                  statistics.average_states[end],excitation_plan)) atol=3e-13

        adaptive=adaptive_weak_pi_quantum_trajectories(
            decay_plan,decay_psi,decay_times;
            observables=(identity=Matrix{ComplexF64}(I,2,2),),
            dt=0.3,algorithm=:event,min_trajectories=4,
            max_trajectories=8,batch_size=2,atol=1.0,rtol=0.0,
            seed=81,jump_statistics=true)
        @test adaptive isa AdaptiveTrajectoryResult
        @test adaptive.backend==:weak_pi
        @test adaptive.converged
        @test adaptive.trajectory_count==4
        @test adaptive.metadata.effective_independent_samples==4
        @test adaptive.metadata.independence_unit==
              :separately_seeded_trajectory
        @test !adaptive.metadata.integration_bias_controlled
        @test adaptive.jumps.trajectories==4
    end

    @testset "streamed weak-PI steady state" begin
        steady_basis=PIBasis(1,2)
        steady_rho0=iid_pure_state(steady_basis,ComplexF64[0,1])
        steady_psi0=weak_pi_pseudoket(steady_rho0)
        steady_model=PIModel(steady_basis,
            (LocalJump(sm;rate=3.0),
             LocalJump(adjoint(sm);rate=2.0)))
        steady_plan=WeakPITrajectoryPlan(steady_model)
        steady_batch=WeakPITrajectoryBatchWorkspace(
            steady_plan,steady_psi0;workers=max(2,Threads.nthreads()))
        steady_times=[0.0,0.1,0.2,0.3]
        path_count=8

        # This retained-history reference deliberately performs the physical
        # outer product before averaging. Averaging pseudo-kets first would be
        # a different, incorrect estimator because sector phases are
        # unphysical and the density conversion is nonlinear.
        explicit_paths=weak_pi_quantum_trajectories(
            steady_plan,steady_psi0,steady_times,path_count;
            seed=418,dt=0.01,workspace=steady_batch)
        path_means=[reduce(+,
            (weak_pi_density(path.states[index]).data
             for index in 2:length(steady_times)))/3
            for path in explicit_paths]
        reference_mean=reduce(+,path_means)/path_count
        reference_m2=sum(norm(path_mean-reference_mean)^2
                         for path_mean in path_means)
        reference_spread=sqrt(reference_m2/(path_count-1))
        reference_standard_error=reference_spread/sqrt(path_count)
        excitation=ComplexF64[0 0;0 1]
        excitation_operator=collective_operator(steady_basis,excitation)
        path_excitations=[real(dot(excitation_operator.data,path_mean))
                          for path_mean in path_means]
        mean_excitation=sum(path_excitations)/path_count
        variance_excitation=sum(abs2(value-mean_excitation)
            for value in path_excitations)/(path_count-1)

        estimate=weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=path_count,settling_time=0.1,
            samples_per_trajectory=3,sampling_interval=0.1,
            seed=418,dt=0.01,workspace=steady_batch,
            observables=(excitation=excitation,),return_info=true)
        @test estimate isa TrajectorySteadyStateResult
        @test estimate.state.data≈reference_mean atol=3e-15
        @test estimate.trajectory_count==path_count
        @test estimate.samples_per_trajectory==3
        @test estimate.sampling_times≈steady_times[2:end] atol=eps(Float64)
        @test estimate.sample_spread≈reference_spread atol=3e-15
        @test estimate.standard_error≈reference_standard_error atol=3e-15
        @test estimate.trace_error≈abs(trace(estimate.state)-1) atol=1e-15
        sparse_steady=liouvillian(steady_model;representation=:sparse)
        @test estimate.residual≈
              norm(sparse_steady*estimate.state.data) atol=2e-14
        @test estimate.relative_residual≈
              estimate.residual/max(norm(estimate.state.data),1) atol=2e-14
        observable_estimate=estimate.observables.observables[:excitation]
        @test observable_estimate.mean≈mean_excitation atol=3e-15
        @test observable_estimate.variance≈variance_excitation atol=3e-15
        @test observable_estimate.standard_error≈
              sqrt(variance_excitation/path_count) atol=3e-15
        @test observable_estimate.lower<=observable_estimate.mean<=
              observable_estimate.upper
        @test estimate.metadata.worker_count==1
        @test estimate.metadata.sampling_interval==0.1
        @test Base.summarysize(estimate)<Base.summarysize(explicit_paths)

        state_only=weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=path_count,settling_time=0.1,
            samples_per_trajectory=3,sampling_interval=0.1,
            seed=418,dt=0.01,workspace=steady_batch)
        @test state_only isa PIState
        @test state_only.data≈estimate.state.data atol=3e-15

        batch_estimate=weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=path_count,settling_time=0.1,
            samples_per_trajectory=4,sampling_interval=0.1,
            batch_size=2,seed=419,dt=0.01,workspace=steady_batch,
            return_info=true)
        batch_report=batch_estimate.metadata.batch_means
        @test batch_report isa WeakPIBatchMeansDiagnostics
        @test batch_report.batch_size==2
        @test batch_report.batch_count==2path_count
        @test batch_report.effective_independent_samples==2path_count
        @test batch_report.sample_spread>=0
        @test batch_report.standard_error>=0
        @test batch_estimate.metadata.effective_independent_samples==path_count
        @test batch_report.assumptions.approximately_independent_batches
        @test !batch_report.assumptions.finite_window_bias_controlled
        @test !batch_estimate.metadata.finite_window_bias_controlled

        event_estimate=weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=4,settling_time=0.1,
            samples_per_trajectory=2,sampling_interval=0.1,
            seed=420,dt=0.03,algorithm=:event,return_info=true)
        @test event_estimate.metadata.algorithm==:event
        @test abs(trace(event_estimate.state)-1)<2e-13

        if Threads.nthreads()>1
            threaded_estimate=weak_pi_trajectory_steady_state(
                steady_plan,steady_psi0;
                trajectories=path_count,settling_time=0.1,
                samples_per_trajectory=3,sampling_interval=0.1,
                seed=418,dt=0.01,threaded=true,workspace=steady_batch,
                observables=(excitation=excitation,),return_info=true)
            @test threaded_estimate.state.data≈estimate.state.data atol=2e-14
            @test threaded_estimate.sample_spread≈
                  estimate.sample_spread atol=2e-14
            @test threaded_estimate.observables.observables[:excitation].mean≈
                  observable_estimate.mean atol=2e-14
            @test threaded_estimate.metadata.threaded
        end

        driven=PIModel(steady_basis,
            (LocalJump(sm;rate=(t,p)->one(t)),))
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            driven,steady_psi0;
            trajectories=2,settling_time=0.1,dt=0.01)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=1,settling_time=0.1,dt=0.01)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.0,dt=0.01)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.1,
            samples_per_trajectory=2,dt=0.01)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.1,
            samples_per_trajectory=2,sampling_interval=0.0,dt=0.01)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.1,
            samples_per_trajectory=3,sampling_interval=0.1,
            batch_size=2,dt=0.01)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.1,
            samples_per_trajectory=2,sampling_interval=0.1,
            batch_size=2,dt=0.01,return_info=false)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.1,dt=0.01,
            observables=(bad=sm,),return_info=true)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.1,dt=0.01,
            observables=(excitation=excitation,))
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.1,dt=0.01,
            confidence=1.0,return_info=true)
        single_workspace=WeakPITrajectoryWorkspace(
            steady_plan,steady_psi0)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            steady_plan,steady_psi0;
            trajectories=2,settling_time=0.1,dt=0.01,
            threaded=true,workspace=single_workspace)
    end

    @testset "direct jumps, qudits, and precision guards" begin
        direct_operator=collective_operator(b,sm)
        direct_model=PIModel(b,(DirectPIJump(direct_operator;rate=0.3),))
        direct_plan=WeakPITrajectoryPlan(direct_model)
        @test all(branch->branch.source_sector==branch.target_sector,
                  direct_plan.branches)
        direct_path=weak_pi_quantum_trajectory(
            direct_plan,psi0,[0.0,0.05];dt=0.005,rng=MersenneTwister(8))
        @test norm(direct_path.states[end].data)≈1

        qutrit_basis=PIBasis(3,3)
        transition=zeros(ComplexF64,3,3);transition[1,3]=1
        qutrit_rho=iid_pure_state(qutrit_basis,ComplexF64[0,0,1])
        qutrit_psi=weak_pi_pseudoket(qutrit_rho)
        qutrit_model=PIModel(qutrit_basis,(LocalJump(transition),))
        qutrit_plan=WeakPITrajectoryPlan(qutrit_model)
        @test any(branch->branch.source_sector!=branch.target_sector,
                  qutrit_plan.branches)
        qutrit_path=weak_pi_quantum_trajectory(qutrit_plan,qutrit_psi,
            [0.0,0.05];dt=0.005,rng=MersenneTwister(9))
        @test norm(qutrit_path.states[end].data)≈1
        @test weak_pi_dimension(qutrit_basis)<3^qutrit_basis.N

        b32=PIBasis(2,2);sm32=ComplexF32[0 1;0 0]
        rho32=iid_pure_state(b32,ComplexF32[0,1])
        psi32=weak_pi_pseudoket(rho32)
        model32=PIModel(b32,(LocalJump(sm32;rate=1f0),))
        plan32=WeakPITrajectoryPlan(model32)
        workspace32=WeakPITrajectoryWorkspace(plan32,psi32)
        path32=weak_pi_quantum_trajectory(plan32,psi32,Float32[0,0.05];
            dt=0.005f0,rng=MersenneTwister(10),workspace=workspace32)
        @test eltype(path32.times)===Float32
        @test eltype(path32.states[end].data)===ComplexF32
        @test eltype(workspace32.branch_intensities)===Float32
        event32=weak_pi_quantum_trajectory(plan32,psi32,
            Float32[0,0.05];dt=0.01f0,algorithm=:event,
            abstol=1f-5,reltol=1f-4,rng=MersenneTwister(110))
        @test eltype(event32.times)===Float32
        @test eltype(event32.states[end].data)===ComplexF32
        @test norm(event32.states[end].data)≈1f0 atol=2f-6
        steady32=weak_pi_trajectory_steady_state(
            plan32,psi32;
            trajectories=3,settling_time=0.02f0,
            samples_per_trajectory=2,sampling_interval=0.01f0,
            dt=0.005f0,seed=211,return_info=true)
        @test eltype(steady32.state.data)===ComplexF32
        @test eltype(steady32.sampling_times)===Float32
        @test steady32.sample_spread isa Float32
        @test steady32.standard_error isa Float32
        @test steady32.residual isa Float32
        @test abs(trace(steady32.state)-1)<2e-5
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            plan32,psi32,[0.0,0.05];dt=0.005f0)
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            plan32,psi32,Float32[0,0.05];dt=0.005)
        @test_throws ArgumentError weak_pi_trajectory_steady_state(
            plan32,psi32;
            trajectories=2,settling_time=0.02,dt=0.005f0)

        negative=WeakPITrajectoryPlan(
            PIModel(b32,(LocalJump(sm32;rate=-1f0),)))
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            negative,psi32,Float32[0,0.01];dt=0.005f0)
        nonfinite=WeakPITrajectoryPlan(
            PIModel(b32,(LocalJump(sm32;rate=Float32(Inf)),)))
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            nonfinite,psi32,Float32[0,0.01];dt=0.005f0)
        complex_hamiltonian=WeakPITrajectoryPlan(
            PIModel(b32,(LocalHamiltonian(ComplexF32[0 1;1 0];
                                            rate=1f0+0f0im),)))
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            complex_hamiltonian,psi32,Float32[0,0.01];dt=0.005f0)

        pair=kron(sm,sm)
        @test_throws ArgumentError WeakPITrajectoryPlan(
            PIModel(PIBasis(2,2),(LocalPBodyJump(pair,2),)))
        schedule=(t,p)->sm
        @test_throws ArgumentError WeakPITrajectoryPlan(
            PIModel(b,(LocalJump(schedule),)))
    end
end
