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
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            plan32,psi32,[0.0,0.05];dt=0.005f0)
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            plan32,psi32,Float32[0,0.05];dt=0.005)

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
