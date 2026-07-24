const PIDHOPS = PermutationalInvariantDynamics

function _hops_dephasing_integral(coefficient, frequency, time)
    coefficient * (frequency * time - one(time) +
                   exp(-frequency * time)) / frequency^2
end

function _hops_physical_blocks(operator::PIOperator)
    basis = operator.basis
    [Matrix(physical_block(operator, partition))
     for partition in basis.sectors]
end

function _hops_blockdiag(blocks)
    dimensions = size.(blocks, 1)
    result = zeros(eltype(first(blocks)), sum(dimensions), sum(dimensions))
    first_index = 1
    for block in blocks
        last_index = first_index + size(block, 1) - 1
        result[first_index:last_index, first_index:last_index] .= block
        first_index = last_index + 1
    end
    result
end

@testset "PI hierarchy of pure states" begin
    spin = spin_matrices()

    @testset "construction, validation, and hierarchy size" begin
        basis = PIBasis(3, 2)
        hamiltonian = collective_operator(basis, 0.17 .* spin.jx)
        coupling = collective_operator(basis, spin.jz)
        bath = HOPSBath(coupling, [0.21, 0.04], [1.3, 2.1])
        plan = HOPSPlan(hamiltonian, bath;
                        max_depth=2, scaling=:unscaled)

        # There are binomial(K+D,D) triangular hierarchy labels, and every
        # auxiliary carries one direct-sum Schur pseudo-ket rather than a PI
        # density-operator vector.
        @test length(plan.multiindices) == 6
        @test plan.multiindices ==
              [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]
        @test size(plan) ==
              (6weak_pi_dimension(basis), 6weak_pi_dimension(basis))
        @test size(plan, 3) == 1
        @test size(plan, 1) < 6length(basis)
        @test hops_coordinate_scale(plan, 1) == 1
        @test hops_coordinate_scale(plan, [1, 0]) == 1
        @test length(hops_auxiliary_importances(plan)) == 6
        @test first(hops_auxiliary_importances(plan)) == 1
        selfadjoint_workspace = HOPSWorkspace(plan)
        @test selfadjoint_workspace.plan === plan
        @test sum(length, plan.bath_edges) == length(plan.topology.lower)
        @test_throws ArgumentError HOPSWorkspace(plan; memory_budget=1)
        @test_throws ArgumentError HOPSBatchWorkspace(
            plan; workers=1, memory_budget=1)

        @test_throws DimensionMismatch HOPSBath(
            coupling, [0.2], [1.0, 2.0])
        @test_throws ArgumentError HOPSBath(coupling, Float64[], Float64[])
        @test_throws ArgumentError HOPSBath(coupling, NaN, 1.0)
        @test_throws ArgumentError HOPSBath(coupling, 0.2, 0.0)
        @test_throws ArgumentError HOPSBath(coupling, 0.2, -1.0)
        @test_throws ArgumentError HOPSPlan(
            hamiltonian, bath; max_depth=-1)
        @test_throws ArgumentError HOPSPlan(
            hamiltonian, bath; max_depth=true)
        @test_throws ArgumentError HOPSPlan(
            hamiltonian, bath; max_depth=2, scaling=:invalid)
        @test_throws ArgumentError HOPSPlan(
            hamiltonian, bath; max_depth=2, memory_budget=1)

        other_basis = PIBasis(2, 2)
        @test_throws ArgumentError HOPSPlan(
            hamiltonian,
            HOPSBath(collective_operator(other_basis, spin.jz), 0.2, 1.0);
            max_depth=1)

        heom_bath = HEOMBath(coupling, 0.2, 1.1)
        converted = HOPSBath(heom_bath)
        @test converted.coefficients == heom_bath.coefficients
        @test converted.frequencies == heom_bath.frequencies
        @test HOPSPlan(
            hamiltonian, heom_bath; max_depth=1).coefficients ==
              converted.coefficients
        @test_throws ArgumentError HOPSBath(
            HEOMBath(coupling, 0.2, 1.1; residue=0.01))

        combined = HOPSPlan(
            hamiltonian,
            HOPSBath(coupling, [0.20, -0.05], [1.1, 1.1]);
            max_depth=2)
        @test combined.coefficients ≈ ComplexF64[0.15] atol=eps() rtol=0
        @test combined.frequencies == ComplexF64[1.1]
        cancelled = HOPSPlan(
            hamiltonian,
            HOPSBath(coupling, [0.20, -0.20], [1.1, 1.1]);
            max_depth=20)
        @test hops_number_auxiliaries(cancelled) == 1
        @test isempty(cancelled.coefficients)

        many_poles = HOPSBath(
            coupling, fill(0.001, 35), collect(1.0:35.0))
        pruned = HOPSPlan(
            hamiltonian, many_poles;
            max_depth=35, importance_cutoff=1.0)
        @test pruned.full_ado_count > typemax(Int)
        @test hops_number_auxiliaries(pruned) == 1
        @test hops_auxiliary_importances(pruned) == [1.0]
    end

    @testset "conditioned matrix-free hierarchy action" begin
        basis = PIBasis(3, 2)
        hamiltonian = collective_operator(basis, 0.19 .* spin.jx)
        # A lowering coupling exercises the distinct L and L' paths; HOPS
        # does not require Hermitian system-bath coupling operators.
        coupling = collective_operator(basis, spin.jm)
        coefficients = ComplexF64[0.21 + 0.06im, 0.04 - 0.02im]
        frequencies = ComplexF64[1.3 + 0.2im, 2.1 - 0.4im]
        bath = HOPSBath(coupling, coefficients, frequencies)
        plan = HOPSPlan(hamiltonian, bath;
                        max_depth=2, scaling=:unscaled)
        @test size(HOPSWorkspace(plan).coupling) ==
              (weak_pi_dimension(basis), length(plan.multiindices))
        stochastic_initial = weak_pi_pseudoket(
            iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2)))
        @test !hops_hierarchy_metadata(plan).internal_noise_supported
        @test_throws ArgumentError hops_trajectory(
            plan, stochastic_initial, [0.0, 0.01]; dt=0.01)
        signed_plan = HOPSPlan(
            hamiltonian, HOPSBath(coupling, -0.08, 0.7);
            max_depth=1)
        @test !hops_hierarchy_metadata(
            signed_plan).internal_noise_supported
        @test_throws ArgumentError hops_trajectory(
            signed_plan, stochastic_initial, [0.0, 0.01]; dt=0.01)

        workspace = HOPSWorkspace(plan)
        rng = MersenneTwister(0x484f5053)
        source = randn(rng, ComplexF64, size(plan, 1))
        destination = similar(source)
        noise = ComplexF64[0.13 - 0.07im]
        hops_rhs!(destination, plan, source, noise, workspace)

        H = _hops_blockdiag(_hops_physical_blocks(hamiltonian))
        L = _hops_blockdiag(_hops_physical_blocks(coupling))
        dimension = weak_pi_dimension(basis)
        reference = zeros(ComplexF64, length(source))
        lookup = Dict(Tuple(index) => position
                      for (position, index) in pairs(plan.multiindices))
        for (ado, occupations) in pairs(plan.multiindices)
            target = (ado - 1) * dimension + 1:ado * dimension
            x = view(source, target)
            view(reference, target) .=
                (-im .* H -
                 sum(occupations .* frequencies) .* I +
                 conj(only(noise)) .* L) * x
            for pole in eachindex(coefficients)
                if occupations[pole] > 0
                    lower_label = copy(occupations)
                    lower_label[pole] -= 1
                    lower = lookup[Tuple(lower_label)]
                    lower_range =
                        (lower - 1) * dimension + 1:lower * dimension
                    view(reference, target) .+=
                        occupations[pole] * coefficients[pole] .* L *
                        view(source, lower_range)
                end
                upper_label = copy(occupations)
                upper_label[pole] += 1
                upper = get(lookup, Tuple(upper_label), 0)
                if !iszero(upper)
                    upper_range =
                        (upper - 1) * dimension + 1:upper * dimension
                    view(reference, target) .-=
                        adjoint(L) * view(source, upper_range)
                end
            end
        end
        @test destination ≈ reference atol=8e-13 rtol=8e-13

        # Conditioned application is deterministic: an ODE solver may query
        # the same stochastic path and time repeatedly without consuming new
        # random numbers.
        repeated = similar(destination)
        hops_rhs!(repeated, plan, source, noise, workspace)
        @test repeated == destination
        @test_throws ArgumentError hops_rhs!(
            source, plan, source, noise, workspace)
        overlapping = randn(rng, ComplexF64, length(source) + 1)
        @test_throws ArgumentError hops_rhs!(
            view(overlapping, 2:length(overlapping)), plan,
            view(overlapping, 1:length(source)), noise, workspace)
        @test_throws ArgumentError hops_rhs!(
            vec(workspace.coupling), plan, source, noise, workspace)

        # Scaled and unscaled hierarchies are related by an exact diagonal
        # similarity. The scale is sqrt(prod(n_k! * abs(c_k)^n_k)).
        scaled = HOPSPlan(hamiltonian, bath;
                          max_depth=2, scaling=:scaled)
        scaled_workspace = HOPSWorkspace(scaled)
        scales = map(scaled.multiindices) do occupations
            sqrt(prod(factorial(occupations[k]) *
                      abs(coefficients[k])^occupations[k]
                      for k in eachindex(coefficients)))
        end
        coordinate_scales = repeat(scales; inner=dimension)
        scaled_source = source ./ coordinate_scales
        scaled_destination = similar(scaled_source)
        hops_rhs!(scaled_destination, scaled, scaled_source,
                  noise, scaled_workspace)
        @test scaled_destination ≈
              reference ./ coordinate_scales atol=2e-12 rtol=2e-12
        @test hops_coordinate_scale(scaled, [0, 0]) == 1
        @test hops_coordinate_scale(scaled, [1, 0]) ≈
              sqrt(abs(coefficients[1]))
        @test_throws ArgumentError hops_coordinate_scale(
            scaled, [3, 0])

        # All hierarchy-sized and block scratch must already be owned by the
        # explicit workspace on a warmed hot call.
        allocation = @allocated hops_rhs!(
            repeated, plan, source, noise, workspace)
        @test allocation <= 512
    end

    @testset "prescribed-noise dephasing and density extraction" begin
        basis = PIBasis(1, 2)
        zero_hamiltonian = PIOperator(basis; T=Float64)
        projector = collective_operator(
            basis, ComplexF64[0 0; 0 1])
        coefficient = 0.3
        frequency = 1.2
        final_time = 0.7
        bath = HOPSBath(projector, coefficient, frequency)
        initial = weak_pi_pseudoket(
            iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2)))
        zero_noise = (time::Real, bath_number::Integer) -> 0.0 + 0.0im

        errors = Float64[]
        trajectories = Any[]
        phi = _hops_dephasing_integral(
            coefficient, frequency, final_time)
        # Stored qubit GT order is m=+1/2,-1/2, so the local |e><e|
        # projector acts on the first direct-sum amplitude.
        expected_root = ComplexF64[exp(-phi) / sqrt(2),
                                   inv(sqrt(2))]
        for depth in (2, 4, 6)
            plan = HOPSPlan(zero_hamiltonian, bath;
                            max_depth=depth, scaling=:scaled)
            trajectory = hops_trajectory(
                plan, initial, [0.0, final_time];
                dt=final_time / 200, noise=zero_noise)
            push!(trajectories, trajectory)
            root = trajectory.states[end].data
            push!(errors, norm(root - expected_root))
        end
        @test errors[3] < 2e-9
        @test errors[3] < errors[2] < errors[1]

        final_density = hops_density(last(trajectories), 2)
        final_root = last(trajectories).states[end].data
        @test trace(final_density) ≈ norm(final_root)^2 atol=3e-13
        expected_density = final_root * final_root'
        @test physical_block(final_density, basis.sectors[1]) ≈
              expected_density atol=3e-13 rtol=3e-13

        # External noise is a deterministic function of time, including when
        # stage times are revisited by the integrator.
        calls = Dict{Float64,ComplexF64}()
        consistent_calls = Ref(true)
        deterministic_noise = (time::Real, bath_number::Integer) -> begin
            value = (0.11 + 0.07im) * exp(-(0.8 + 0.2im) * time)
            if haskey(calls, time)
                consistent_calls[] &= calls[time] == value
            else
                calls[time] = value
            end
            value
        end
        deterministic_plan = HOPSPlan(
            zero_hamiltonian, bath; max_depth=4)
        first_path = hops_trajectory(
            deterministic_plan, initial, [0.0, 0.2];
            dt=0.002, noise=deterministic_noise)
        second_path = hops_trajectory(
            deterministic_plan, initial, [0.0, 0.2];
            dt=0.002, noise=deterministic_noise)
        @test consistent_calls[]
        @test first_path.states[end].data == second_path.states[end].data
    end

    @testset "stationary OU noise, reproducibility, and Float32" begin
        basis = PIBasis(1, 2)
        hamiltonian = PIOperator(basis; T=Float64)
        coupling = collective_operator(basis, spin.jz)
        bath = HOPSBath(coupling, 0.08, 0.7 + 0.2im)
        plan = HOPSPlan(hamiltonian, bath;
                        max_depth=2, scaling=:scaled)
        initial = weak_pi_pseudoket(
            iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2)))
        times = [0.0, 0.05, 0.1]
        first_path = hops_trajectory(
            plan, initial, times; dt=0.005,
            rng=MersenneTwister(71))
        repeated_path = hops_trajectory(
            plan, initial, times; dt=0.005,
            rng=MersenneTwister(71))
        different_path = hops_trajectory(
            plan, initial, times; dt=0.005,
            rng=MersenneTwister(72))
        @test all(first_path.states[index].data ==
                  repeated_path.states[index].data
                  for index in eachindex(times))
        @test first_path.noise == repeated_path.noise
        @test first_path.noise != different_path.noise
        @test !iszero(first_path.noise[1, 1])

        serial_average = hops_average(
            plan, initial, times, 6;
            dt=0.005, seed=74, threaded=false)
        repeated_average = hops_average(
            plan, initial, times, 6;
            dt=0.005, seed=74, threaded=false)
        @test all(serial_average[index].data ==
                  repeated_average[index].data
                  for index in eachindex(times))
        if Threads.nthreads() > 1
            threaded_average = hops_average(
                plan, initial, times, 6;
                dt=0.005, seed=74, threaded=true,
                workspace=HOPSBatchWorkspace(plan; workers=2))
            @test all(isapprox(
                serial_average[index].data,
                threaded_average[index].data;
                atol=3e-15, rtol=3e-15)
                for index in eachindex(times))
        end

        basis32 = PIBasis(1, 2)
        hamiltonian32 = PIOperator(basis32; T=Float32)
        coupling32 = collective_operator(
            basis32, ComplexF32[-0.5 0; 0 0.5];
            cache=OneBodyGeometry(basis32, Float32))
        bath32 = HOPSBath(coupling32, 0.08f0, 0.7f0)
        plan32 = HOPSPlan(hamiltonian32, bath32;
                          max_depth=2, scaling=:scaled)
        initial32 = weak_pi_pseudoket(
            iid_pure_state(
                basis32, ComplexF32[1, 1] / sqrt(2.0f0)))
        path32 = hops_trajectory(
            plan32, initial32, Float32[0, 0.02];
            dt=0.002f0, rng=MersenneTwister(73))
        @test eltype(path32.times) === Float32
        @test eltype(path32.states[end].data) === ComplexF32
        @test eltype(path32.noise) === ComplexF32
        @test_throws ArgumentError hops_trajectory(
            plan32, initial32, [0.0, 0.02];
            dt=0.002f0, rng=MersenneTwister(73))
        @test_throws ArgumentError hops_trajectory(
            plan32, initial32, Float32[0, 0.02];
            dt=0.002, rng=MersenneTwister(73))
    end

    @testset "zero coupling, Schur multiplicities, and averaging" begin
        basis = PIBasis(3, 2)
        zero_operator = PIOperator(basis; T=Float64)
        bath = HOPSBath(zero_operator, 0.08, 0.7)
        plan = HOPSPlan(zero_operator, bath;
                        max_depth=2, scaling=:scaled)
        raw = ComplexF64.(1:weak_pi_dimension(basis))
        raw ./= norm(raw)
        initial = WeakPIPseudoKet(basis, raw)
        reference = weak_pi_density(initial)
        times = [0.0, 0.03]

        path = hops_trajectory(
            plan, initial, times; dt=0.01,
            rng=MersenneTwister(80))
        @test_throws ArgumentError hops_trajectory(
            plan, initial, times; dt=0.01,
            rng=MersenneTwister(80), memory_budget=1)
        @test hops_density(path, 2).data ≈
              reference.data atol=3e-14 rtol=3e-14
        averaged = hops_average(
            plan, initial, times, 4;
            dt=0.01, seed=81, threaded=false)
        @test length(averaged) == length(times)
        @test all(state -> isapprox(
            state.data, reference.data; atol=3e-14, rtol=3e-14), averaged)
        @test all(state -> isapprox(
            trace(state), 1; atol=3e-14), averaged)

        ensemble = hops_average(
            plan, initial, times, 4;
            dt=0.01, seed=81, threaded=false, return_info=true)
        @test ensemble isa HOPSEnsembleResult
        @test ensemble.trajectory_count == 4
        @test ensemble.times == times
        @test ensemble.metadata.equation === :linear
        @test all(iszero, ensemble.sample_spread)
        @test all(iszero, ensemble.standard_error)
        @test_throws ArgumentError hops_average(
            plan, initial, times, 4;
            dt=0.01, seed=81, memory_budget=1)
        @test_throws ArgumentError hops_average(
            plan, initial, times, true;
            dt=0.01, seed=81)
        @test_throws ArgumentError HOPSBatchWorkspace(plan; workers=true)

        # A general PI density operator is sampled through the eigensystems
        # of its multiplicity-weighted Schur blocks. Recombining those exact
        # component weights must recover a genuinely mixed, multi-sector
        # state without sampling any multiplicity tableaux.
        symmetric_partition, mixed_partition = basis.sectors
        symmetric_weights = Float64[0.0, 0.0, 0.1, 0.2]
        mixed_weights = Float64[0.3, 0.4]
        symmetric_multiplicity = Float64(
            symmetric_group_dimension(symmetric_partition))
        mixed_multiplicity = Float64(
            symmetric_group_dimension(mixed_partition))
        mixed_state = state_from_schur_blocks(
            basis,
            [symmetric_partition =>
                 Matrix(Diagonal(
                     symmetric_weights ./ symmetric_multiplicity)),
             mixed_partition =>
                 Matrix(Diagonal(mixed_weights ./ mixed_multiplicity))];
            validate=true)
        initial_ensemble = hops_initial_ensemble(plan, mixed_state)
        @test initial_ensemble.basis === basis
        @test sort(initial_ensemble.weights) ≈
              [0.1, 0.2, 0.3, 0.4] atol=3e-15 rtol=3e-15
        @test sort(unique(initial_ensemble.sectors)) == [1, 2]
        @test initial_ensemble.source_trace ≈ 1 atol=3e-15
        @test_throws ArgumentError hops_initial_ensemble(
            plan, mixed_state; memory_budget=1)

        weak_offsets = Int[1]
        for patterns in basis.patterns
            push!(weak_offsets, last(weak_offsets) + length(patterns))
        end
        reconstructed = zeros(ComplexF64, length(basis))
        for component in eachindex(initial_ensemble.weights)
            root_data = zeros(ComplexF64, weak_pi_dimension(basis))
            sector = initial_ensemble.sectors[component]
            copyto!(
                view(root_data,
                     weak_offsets[sector]:weak_offsets[sector + 1] - 1),
                initial_ensemble.amplitudes[component])
            contribution = hops_density(HOPSRootKet(basis, root_data))
            reconstructed .+=
                initial_ensemble.weights[component] .* contribution.data
        end
        @test reconstructed ≈ mixed_state.data atol=3e-15 rtol=3e-15

        sampled_mixed = hops_average(
            plan, mixed_state, [0.0], 16;
            dt=0.01, seed=82, threaded=false)
        @test trace(only(sampled_mixed)) ≈ 1 atol=3e-15

        invalid_state = state_from_schur_blocks(
            basis,
            [symmetric_partition =>
                 Matrix(Diagonal(Float64[-0.1, 0.0, 0.0, 1.1]))])
        @test trace(invalid_state) ≈ 1 atol=3e-15
        @test_throws ArgumentError HOPSInitialEnsemble(invalid_state)
    end

    @testset "BigFloat precision context" begin
        basis = PIBasis(1, 2)
        hamiltonian, bath, initial = setprecision(192) do
            H = PIOperator(basis; T=BigFloat)
            local_z = Complex{BigFloat}[
                BigFloat(-1) / 2 0
                0 BigFloat(1) / 2
            ]
            geometry = OneBodyGeometry(basis, BigFloat)
            coupling = collective_operator(
                basis, local_z; cache=geometry)
            prepared_bath = HOPSBath(
                coupling, BigFloat("0.08"), BigFloat("0.7"))
            psi = WeakPIPseudoKet(
                basis,
                Complex{BigFloat}[1, 1] / sqrt(BigFloat(2)))
            H, prepared_bath, psi
        end

        setprecision(64) do
            plan = HOPSPlan(
                hamiltonian, bath; max_depth=2, scaling=:scaled)
            @test plan.precision_bits >= 192
            @test minimum(precision(real(value))
                          for block in plan.hamiltonian_blocks
                          for value in block) >= 192
            workspace = HOPSWorkspace(plan)
            @test precision(real(workspace.current[1])) >= 192
            path = hops_trajectory(
                plan, initial, BigFloat[0, BigFloat("0.01")];
                dt=BigFloat("0.005"), rng=MersenneTwister(91),
                workspace)
            @test precision(real(path.states[end].data[1])) >= 192
            rho = hops_density(path, 2)
            @test precision(real(rho.data[1])) >= 192
            average = hops_average(
                plan, initial, BigFloat[0, BigFloat("0.01")], 2;
                dt=BigFloat("0.005"), seed=92)
            @test precision(real(average[end].data[1])) >= 192
        end
    end
end
