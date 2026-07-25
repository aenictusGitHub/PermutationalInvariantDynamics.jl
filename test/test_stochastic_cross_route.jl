@testset "Stochastic cross-route and failure-recovery invariants" begin
    basis = PIBasis(1, 2)
    lowering = ComplexF64[0 1; 0 0]
    raising = adjoint(lowering)

    @testset "N=1 density and weak-PI event paths" begin
        rate = 1.3
        model = PIModel(basis, (LocalJump(lowering; rate),))
        density_initial = iid_pure_state(basis, ComplexF64[0, 1])
        weak_initial = weak_pi_pseudoket(density_initial)
        times = [0.0, 0.7, 2.0]
        seed = 17

        density_path = quantum_trajectory(
            TrajectoryPlan(model), density_initial, times;
            algorithm=:event, dt=0.3, dtmax=0.5,
            abstol=1e-11, reltol=1e-10,
            rng=MersenneTwister(seed))
        weak_path = weak_pi_quantum_trajectory(
            WeakPITrajectoryPlan(model), weak_initial, times;
            algorithm=:event, dt=0.3, dtmax=0.5,
            abstol=1e-11, reltol=1e-10,
            rng=MersenneTwister(seed))

        @test density_path.jump_channels == weak_path.jump_channels == [1]
        @test density_path.jump_times ≈ weak_path.jump_times atol=2e-10 rtol=0
        @test only(density_path.jump_times) ≈
              randexp(MersenneTwister(seed)) / rate atol=2e-10 rtol=0
        @test all(
            isapprox(
                density_path.states[index].data,
                weak_pi_density(weak_path.states[index]).data;
                atol=2e-10, rtol=2e-10)
            for index in eachindex(times))
        @test all(state -> isapprox(trace(state), 1; atol=3e-13),
                  density_path.states)
        @test all(state -> isapprox(norm(state.data), 1; atol=3e-13),
                  weak_path.states)
    end

    @testset "rejected schedules do not poison reusable workspaces" begin
        superposition = ComplexF64[1, 1] / sqrt(2)
        density_initial = iid_pure_state(basis, superposition)
        weak_initial = weak_pi_pseudoket(density_initial)
        scheduled_model = PIModel(
            basis,
            (LocalJump(lowering; rate=(t, p) -> p.down),
             LocalJump(raising; rate=(t, p) -> p.up)))
        density_plan = TrajectoryPlan(scheduled_model)
        weak_plan = WeakPITrajectoryPlan(scheduled_model)
        density_workspace = TrajectoryWorkspace(density_plan, density_initial)
        weak_workspace = WeakPITrajectoryWorkspace(weak_plan, weak_initial)
        times = [0.0, 0.04]
        bad_parameters = (down=0.4, up=-0.2)
        good_parameters = (down=0.7, up=0.3)
        seed = 0x5eed

        @test_throws ArgumentError quantum_trajectory(
            density_plan, density_initial, times;
            dt=0.01, parameters=bad_parameters,
            rng=MersenneTwister(seed), workspace=density_workspace)
        @test_throws ArgumentError weak_pi_quantum_trajectory(
            weak_plan, weak_initial, times;
            dt=0.01, parameters=bad_parameters,
            rng=MersenneTwister(seed), workspace=weak_workspace)

        recovered_density = quantum_trajectory(
            density_plan, density_initial, times;
            dt=0.01, parameters=good_parameters,
            rng=MersenneTwister(seed), workspace=density_workspace)
        reference_density = quantum_trajectory(
            density_plan, density_initial, times;
            dt=0.01, parameters=good_parameters,
            rng=MersenneTwister(seed),
            workspace=TrajectoryWorkspace(density_plan, density_initial))
        recovered_weak = weak_pi_quantum_trajectory(
            weak_plan, weak_initial, times;
            dt=0.01, parameters=good_parameters,
            rng=MersenneTwister(seed), workspace=weak_workspace)
        reference_weak = weak_pi_quantum_trajectory(
            weak_plan, weak_initial, times;
            dt=0.01, parameters=good_parameters,
            rng=MersenneTwister(seed),
            workspace=WeakPITrajectoryWorkspace(weak_plan, weak_initial))

        @test recovered_density.jump_times == reference_density.jump_times
        @test recovered_density.jump_channels == reference_density.jump_channels
        @test all(
            recovered_density.states[index].data ==
            reference_density.states[index].data
            for index in eachindex(times))
        @test recovered_weak.jump_times == reference_weak.jump_times
        @test recovered_weak.jump_channels == reference_weak.jump_channels
        @test all(
            recovered_weak.states[index].data ==
            reference_weak.states[index].data
            for index in eachindex(times))
        @test density_initial.data ==
              iid_pure_state(basis, superposition).data
        @test weak_initial.data == weak_pi_pseudoket(density_initial).data
    end

    @testset "zero-coupling HOPS follows deterministic PI dynamics" begin
        many_body_basis = PIBasis(2, 2)
        sigma_x = ComplexF64[0 1; 1 0]
        hamiltonian = collective_operator(
            many_body_basis, 0.23 .* sigma_x)
        zero_coupling = PIOperator(many_body_basis; T=Float64)
        hops_plan = HOPSPlan(
            hamiltonian, HOPSBath(zero_coupling, 0.08, 0.7);
            max_depth=1, scaling=:scaled)
        density_initial = iid_pure_state(
            many_body_basis, ComplexF64[1, im] / sqrt(2))
        weak_initial = weak_pi_pseudoket(density_initial)
        times = [0.0, 0.04, 0.1]

        first_path = hops_trajectory(
            hops_plan, weak_initial, times;
            dt=0.002, rng=MersenneTwister(601))
        second_path = hops_trajectory(
            hops_plan, weak_initial, times;
            dt=0.002, rng=MersenneTwister(602))
        deterministic = time_evolve(
            PIModel(
                many_body_basis,
                (DirectPIHamiltonian(hamiltonian),)),
            density_initial, (first(times), last(times)); steps=50)

        @test first_path.noise != second_path.noise
        @test all(
            first_path.states[index].data == second_path.states[index].data
            for index in eachindex(times))
        @test hops_density(first_path, length(times)).data ≈
              deterministic.data atol=2e-12 rtol=2e-12
        @test trace(hops_density(first_path, length(times))) ≈
              1 atol=2e-13
    end
end
