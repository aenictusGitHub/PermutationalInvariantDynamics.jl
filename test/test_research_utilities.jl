@testset "Research-oriented PI utilities" begin
    @testset "spectral traces and population iterators" begin
        b=PIBasis(4,2)
        rho=maximally_mixed_state(b)
        @test isapprox(spectral_trace(rho,identity),1;atol=2e-12)
        @test isapprox(spectral_trace(rho,x->x^2),purity(rho);atol=2e-12)
        @test isapprox(spectral_trace(rho,_ -> 1.0),2.0^b.N;atol=2e-12)
        @test_throws ArgumentError spectral_trace(rho,_ -> Inf)

        coordinates=each_population_coordinate(b)
        @test length(coordinates)==population_dimension(b)
        @test [coordinate.index for coordinate in coordinates]==collect(eachindex(coordinates))
        @test all(coordinate->coordinate.pi_index in
            (b.offsets[coordinate.sector_index]:b.offsets[coordinate.sector_index+1]-1),
            coordinates)
        @test all(coordinate->coordinate.multiplicity==
            symmetric_group_dimension(coordinate.sector),coordinates)

        sm=ComplexF64[0 1;0 0]
        plan=PopulationPlan(PIModel(b,(LocalJump(sm;rate=0.3),)))
        transitions=population_transitions(plan)
        generator=population_generator(plan;representation=:sparse)
        rows,columns,values=findnz(generator)
        @test length(transitions)==count(index->
            rows[index]!=columns[index]&&!iszero(values[index]),eachindex(values))
        @test all(transition->transition.source.index!=transition.destination.index,
                  transitions)
        @test all(transition->transition.rate==
            generator[transition.destination.index,transition.source.index],
            transitions)
    end

    @testset "composable PI channels" begin
        b=PIBasis(1,2)
        rho=iid_state(b,ComplexF64[0.3 0.1;0.1 0.7])
        identity_map=identity_channel(b)
        report=check_pi_channel(identity_map)
        @test report.completely_positive===true
        @test report.trace_preserving
        @test report.unital
        @test apply_channel(identity_map,rho).data==rho.data
        @test_throws ArgumentError apply_channel!(rho,identity_map,rho)

        probability=0.28
        K0=collective_operator(b,ComplexF64[1 0;0 sqrt(1-probability)])
        K1=collective_operator(b,ComplexF64[0 sqrt(probability);0 0])
        damping=kraus_channel((K0,K1);check=true)
        damping_report=check_pi_channel(damping)
        @test damping_report.completely_positive===true
        @test damping_report.trace_preserving
        @test !damping_report.unital
        damped=apply_channel(damping,rho)
        @test isapprox(trace(damped),1;atol=2e-12)
        @test ispositive(damped)

        twice=compose_channels(damping,damping)
        @test isapprox(apply_channel(twice,rho).data,
            apply_channel(damping,apply_channel(damping,rho)).data;atol=2e-13)
        x=randn(MersenneTwister(3),ComplexF64,length(b))
        y=randn(MersenneTwister(4),ComplexF64,length(b))
        @test isapprox(dot(y,damping*x),dot(channel_adjoint(damping)*y,x);
                       atol=2e-13)

        matrix=identity_map.matrix
        matrixfree=MatrixFreePIChannel(b,ComplexF64,
            (destination,source)->mul!(destination,matrix,source),
            (destination,source)->mul!(destination,adjoint(matrix),source))
        matrixfree_report=check_pi_channel(matrixfree)
        @test matrixfree_report.completely_positive===missing
        @test matrixfree_report.trace_preserving
        @test check_pi_channel(matrixfree;materialize=true).completely_positive===true
        @test_throws DimensionMismatch PIChannel(b,zeros(3,3))

        # A strictly positive Choi matrix must report its actual minimum,
        # rather than a zero introduced by a reduction initializer.
        identity_vector=identity_operator(b).data
        depolarizing=PIChannel(b,(identity_vector/2)*adjoint(identity_vector))
        depolarizing_report=check_pi_channel(depolarizing)
        @test depolarizing_report.completely_positive===true
        @test isapprox(depolarizing_report.minimum_choi_eigenvalue,0.5;
                       atol=2e-13)
    end

    @testset "PI POVM sampling and maximum likelihood" begin
        b=PIBasis(1,2)
        E0=collective_operator(b,ComplexF64[1 0;0 0])
        E1=collective_operator(b,ComplexF64[0 0;0 1])
        effects=(E0,E1)
        rho=iid_state(b,ComplexF64[0.72 0;0 0.28])
        probabilities=pi_povm_probabilities(rho,effects)
        @test probabilities≈[0.72,0.28]
        sample=sample_pi_povm(rho,effects,2_000;rng=MersenneTwister(9))
        @test sum(sample.counts)==2_000
        @test sample.probabilities==probabilities

        estimate=maximum_likelihood_tomography(b,effects,[720,280];
            maxiter=1_000,require_convergence=true)
        @test estimate.converged
        @test estimate.probabilities≈[0.72,0.28] atol=2e-7
        @test isphysical(estimate.state)
        @test isapprox(trace(estimate.state),1;atol=2e-12)

        restricted=PIBasis(3,2;sectors=[(2,1)])
        sector=restricted.sectors[1]
        restricted_effects=(
            operator_from_schur_blocks(restricted,
                (sector=>ComplexF32[1 0;0 0],);T=Float32),
            operator_from_schur_blocks(restricted,
                (sector=>ComplexF32[0 0;0 1],);T=Float32))
        restricted_estimate=maximum_likelihood_tomography(
            restricted,restricted_effects,[650,350];
            initial_state=maximally_mixed_state(restricted;T=Float32),
            maxiter=1_000,atol=1f-6,rtol=1f-5,dilution=0.5f0,
            require_convergence=true)
        @test eltype(restricted_estimate.state.data)===ComplexF32
        @test restricted_estimate.probabilities≈Float32[0.65,0.35] atol=2f-4
        @test isapprox(trace(restricted_estimate.state),1f0;atol=2f-5)

        bad_effect=collective_operator(b,ComplexF64[2 0;0 0])
        @test_throws ArgumentError pi_povm_probabilities(rho,(bad_effect,E1))
        @test_throws ArgumentError maximum_likelihood_tomography(
            b,effects,[0,0])
    end

    @testset "versioned checkpoints" begin
        b=PIBasis(3,2;sectors=[(3,0)])
        rho=computational_product_state(b,2;T=Float32)
        checkpoint=PIStateCheckpoint(rho;time=Float32(0.25),
                                     metadata=(model="test",seed=4))
        @test checkpoint.schema_version==PI_CHECKPOINT_VERSION
        mktempdir() do directory
            path=joinpath(directory,"state.pid")
            @test save_checkpoint(path,checkpoint)==path
            loaded=load_checkpoint(path)
            @test loaded.schema_version==checkpoint.schema_version
            @test loaded.state.basis.N==b.N
            @test loaded.state.basis.d==b.d
            @test loaded.state.basis.sectors==b.sectors
            @test loaded.state.data==rho.data
            @test loaded.time===Float32(0.25)
            @test loaded.metadata==Dict("model"=>"test","seed"=>"4")
            detached=checkpoint_state(loaded);detached.data[1]+=1
            @test detached.data!=loaded.state.data

            payload=PermutationalInvariantDynamics._checkpoint_payload(checkpoint)
            wrong_label=merge(payload,(scalar_type="Float64",))
            @test_throws ArgumentError begin
                PermutationalInvariantDynamics._checkpoint_from_payload(wrong_label)
            end
            mixed_precision=merge(payload,(imag=Float64.(payload.imag),))
            @test_throws ArgumentError begin
                PermutationalInvariantDynamics._checkpoint_from_payload(mixed_precision)
            end

            big_checkpoint=setprecision(BigFloat,160) do
                big_state=computational_product_state(
                    PIBasis(1,2;sectors=[(1,0)]),1;T=BigFloat)
                PIStateCheckpoint(big_state;time=BigFloat("0.125"))
            end
            big_path=joinpath(directory,"state-big.pid")
            save_checkpoint(big_path,big_checkpoint)
            loaded_big=load_checkpoint(big_path)
            @test all(value->precision(real(value))==160&&
                precision(imag(value))==160,loaded_big.state.data)
            @test precision(loaded_big.time)==160
            @test loaded_big.state.data==big_checkpoint.state.data

            mixed_big=checkpoint_state(big_checkpoint)
            mixed_big.data[1]=Complex(
                setprecision(BigFloat,128) do
                    BigFloat(real(mixed_big.data[1]))
                end,
                imag(mixed_big.data[1]))
            mixed_checkpoint=PIStateCheckpoint(mixed_big)
            @test_throws ArgumentError save_checkpoint(
                joinpath(directory,"mixed-big.pid"),mixed_checkpoint)

            corrupt=joinpath(directory,"corrupt.pid")
            open(corrupt,"w") do io;write(io,"not a checkpoint");end
            @test_throws ArgumentError load_checkpoint(corrupt)
        end
    end

    @testset "simultaneous weak symmetries" begin
        b=PIBasis(2,2)
        sx=ComplexF64[0 1;1 0]
        sz=ComplexF64[1 0;0 -1]
        projector=joint_symmetry_projector(b,(sx=>1,sz=>1))
        @test projector.range_dimension>0
        @test projector.charge==(1+0im,1+0im)
        source=randn(MersenneTwister(17),ComplexF64,length(b))
        once=projector*source;twice=projector*once
        @test isapprox(twice,once;atol=2e-11)
        work=JointSymmetryProjectorWorkspace(projector);destination=similar(once)
        apply!(destination,projector,source,work)
        @test destination≈once
        single=joint_symmetry_projector(b,(sx,1))
        @test isapprox(single*(single*source),single*source;atol=2e-11)

        hadamard=ComplexF64[1 1;1 -1]/sqrt(2)
        @test_throws ArgumentError joint_symmetry_projector(
            b,(sx=>1,hadamard=>1))
    end

    @testset "implicit and checkpointed adjoint gradients" begin
        b=PIBasis(1,2)
        sm=ComplexF64[0 1;0 0]
        sx=ComplexF64[0 1;1 0]
        base=PIModel(b,(CollectiveJump(sm;rate=1.0),))
        stationary=PIState(b,steady_state(base;method=:direct))
        derivative=PIModel(b,(LocalHamiltonian(sx;rate=1.0),))
        plan=SteadyStateGradientPlan(base,stationary)
        workspace=SteadyStateGradientWorkspace(plan;krylovdim=4)
        gradient=implicit_steady_state_gradient(plan,(derivative,);
            workspace,krylovdim=4,maxiter=100)
        epsilon=1e-5
        plus=PIModel(b,(CollectiveJump(sm;rate=1.0),
                        LocalHamiltonian(sx;rate=epsilon)))
        minus=PIModel(b,(CollectiveJump(sm;rate=1.0),
                         LocalHamiltonian(sx;rate=-epsilon)))
        finite_difference=(steady_state(plus;method=:direct)-
                           steady_state(minus;method=:direct))/(2epsilon)
        @test norm(gradient.tangents[1].data-finite_difference)<3e-8
        @test abs(trace(gradient.tangents[1]))<2e-12
        @test gradient.solver_info[1].converged

        objective=collective_operator(b,ComplexF64[0 0;0 1])
        control_derivative=liouvillian(derivative;representation=:sparse)
        base_matrix=spzeros(ComplexF64,length(b),length(b))
        times=collect(range(0.0,0.7;length=81))
        controls=fill(0.4,1,length(times)-1)
        adjoint_result=checkpointed_adjoint_gradient(base_matrix,
            (control_derivative,),computational_product_state(b,1),objective,
            times,controls;checkpoint_stride=11)
        objective_at=function (amplitude)
            final=time_evolve(base_matrix+amplitude*control_derivative,
                computational_product_state(b,1),(times[1],times[end]);steps=800)
            real(expectation(final,objective))
        end
        finite_control=(objective_at(0.4+epsilon)-objective_at(0.4-epsilon))/(2epsilon)
        @test isapprox(sum(adjoint_result.gradient),finite_control;
                       atol=2e-7,rtol=2e-6)
        @test isapprox(trace(adjoint_result.final_state),1;atol=2e-12)
        @test adjoint_result.metadata.retained_checkpoints<length(times)
        @test adjoint_result.metadata.recomputed_intervals==length(times)-1
        dense_checkpoints=checkpointed_adjoint_gradient(base_matrix,
            (control_derivative,),computational_product_state(b,1),objective,
            times,controls;checkpoint_stride=1)
        @test isapprox(dense_checkpoints.gradient,adjoint_result.gradient;
                       atol=2e-12)
        nonhermitian=collective_operator(b,sm)
        @test_throws ArgumentError checkpointed_adjoint_gradient(base_matrix,
            (control_derivative,),computational_product_state(b,1),
            nonhermitian,times,controls)
    end
end
