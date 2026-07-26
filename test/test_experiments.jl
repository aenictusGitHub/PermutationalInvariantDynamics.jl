@testset "reproducible experiment layer" begin
    PIDExperiments=PermutationalInvariantDynamics

    basis=PIBasis(2,2)
    lowering=ComplexF64[0 1;0 0]
    model=PIModel(basis,[LocalJump(lowering;rate=0.4)])
    rho0=iid_pure_state(basis,ComplexF64[0,1])
    local_z=ComplexF64[-1 0;0 1]

    verification=PIDExperiments.VerificationSpec(
        atol=1e-9,rtol=1e-7)
    experiment=PIDExperiments.PIExperiment(
        model;task=:steady_state,algorithm=DirectAlgorithm(),
        observables=(magnetization=local_z,),
        verification,metadata=(label="unit test",))
    plan=PIDExperiments.plan_experiment(experiment)
    @test plan.task===:steady_state
    @test plan.selected_algorithm===:direct
    @test plan.backend===:sparse
    @test plan.exactness.complete_pi_basis
    @test PIDExperiments.explain_experiment(experiment) isa
        PIDExperiments.ExperimentExecutionPlan

    result=PIDExperiments.verified_solve(experiment)
    @test result isa PIDExperiments.ExperimentResult
    @test result.solution isa SteadyStateResult
    @test result.report.verified
    @test result.report.solver_converged
    @test result.report.physical_valid
    @test result.report.refinement_converged===missing
    @test result.report.verification_level===:solver_and_physical
    @test result.observables[:magnetization]≈-2
    @test result.provenance.package_version==string(
        Base.pkgversion(PermutationalInvariantDynamics))
    @test length(result.provenance.structural_digest)==16
    @test result.provenance.metadata==("label"=>"unit test",)
    _,archive_states,_,_=
        PIDExperiments._experiment_archive_payload(result)
    @test only(archive_states)===result.solution.state

    steady_refinement=PIDExperiments.RefinementSpec(
        :krylov_dimension,(4,6,8);
        atol=1e-7,rtol=1e-5,consecutive=1)
    refined_steady=PIDExperiments.PIExperiment(
        model;algorithm=GMRESAlgorithm(maxiter=200),
        verification=PIDExperiments.VerificationSpec(
            atol=1e-8,rtol=1e-6,refinement=steady_refinement))
    refined_steady_result=PIDExperiments.verified_solve(refined_steady)
    @test refined_steady_result.report.refinement_converged
    @test refined_steady_result.report.evidence.refinement.parameter===
        :krylov_dimension

    mktempdir() do directory
        path=joinpath(directory,"steady.pidrun")
        @test PIDExperiments.save_experiment(path,result)==abspath(path)
        archive=PIDExperiments.load_experiment(path)
        @test archive isa PIDExperiments.ExperimentArchive
        @test archive.schema_version==
            PIDExperiments.PI_EXPERIMENT_ARCHIVE_VERSION
        @test archive.task===:steady_state
        @test isempty(archive.times)
        @test length(archive.states)==1
        @test archive.states[1].basis.sectors==basis.sectors
        @test archive.states[1].data==result.solution.state.data
        @test archive.observables["magnetization"]≈-2
        @test archive.metadata["verified"]=="true"
        @test archive.metadata["user.label"]=="unit test"
        @test archive.metadata["bigfloat_precision"]==
            string(result.provenance.bigfloat_precision)
        @test archive.metadata["bigfloat_rounding"]==
            result.provenance.bigfloat_rounding
        @test haskey(
            archive.metadata,"verification.evidence.solver.residual")
        @test haskey(
            archive.metadata,"verification.evidence.physical.trace_error")
        @test haskey(
            archive.metadata,"verification.evidence.specification.atol")
        @test archive.metadata["verification.solver_converged"]=="true"
        @test_throws ArgumentError PIDExperiments.load_experiment(
            path;memory_budget=1)
        @test_throws ArgumentError PIDExperiments.save_experiment(path,result)

        refined_path=joinpath(directory,"refined-steady.pidrun")
        PIDExperiments.save_experiment(
            refined_path,refined_steady_result)
        refined_archive=PIDExperiments.load_experiment(refined_path)
        @test haskey(refined_archive.metadata,
            "verification.evidence.refinement.pairwise_errors.2")
        @test refined_archive.metadata[
            "verification.evidence.refinement.converged"]=="true"
    end

    dynamics=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=rho0,tspan=(0.0,0.1),
        saveat=[0.0,0.05,0.1],steps_per_interval=16,
        observables=(magnetization=local_z,),verification)
    dynamics_result=PIDExperiments.verified_solve(dynamics)
    @test dynamics_result.solution isa DynamicsStreamResult
    @test dynamics_result.report.verified
    @test dynamics_result.report.physical_valid
    @test dynamics_result.report.refinement_converged===missing
    @test dynamics_result.report.verification_level===
        :single_resolution_physical
    @test length(dynamics_result.solution.states)==3
    @test length(dynamics_result.observables[:magnetization])==3
    archive_times,archive_states,archive_observables,_=
        PIDExperiments._experiment_archive_payload(dynamics_result)
    @test archive_times===dynamics_result.solution.times
    @test archive_states===dynamics_result.solution.states
    @test archive_observables["magnetization"]===
        dynamics_result.observables[:magnetization]

    refinement=PIDExperiments.RefinementSpec(
        :steps_per_interval,(8,16,32);
        atol=1e-7,rtol=1e-5,consecutive=1,
        require_convergence=true)
    refined_verification=PIDExperiments.VerificationSpec(
        atol=1e-8,rtol=1e-6,refinement=refinement)
    refined=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=rho0,tspan=(0.0,0.05),
        saveat=[0.0,0.05],observables=(magnetization=local_z,),
        verification=refined_verification)
    refined_result=PIDExperiments.verified_solve(refined)
    @test refined_result.report.refinement_converged
    @test refined_result.report.verification_level===
        :physical_and_timestep_refinement
    @test refined_result.report.evidence.refinement.parameter===
        :steps_per_interval

    state_free_verification=PIDExperiments.VerificationSpec(
        require_physical=false)
    state_free=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=rho0,tspan=(0.0,0.05),
        saveat=[0.0,0.05],observables=(magnetization=local_z,),
        save_states=false,verification=state_free_verification)
    state_free_result=PIDExperiments.verified_solve(state_free)
    @test state_free_result.report.verified
    @test state_free_result.report.physical_valid===missing
    @test state_free_result.report.verification_level===
        :single_resolution_observables

    restricted_basis=PIBasis(2,2;sectors=[(2,0)])
    restricted_model=PIModel(
        restricted_basis,[CollectiveJump(lowering;rate=0.4)])
    @test_throws ArgumentError PIDExperiments.PIExperiment(restricted_model)
    restricted=PIDExperiments.PIExperiment(
        restricted_model;representation=:declared_sectors)
    restricted_plan=PIDExperiments.plan_experiment(restricted)
    @test restricted_plan.exactness.physical_approximation===
        :user_declared_sector_restriction

    @test_throws ArgumentError PIDExperiments.RefinementSpec(
        :steps_per_interval,(8,8))
    @test_throws ArgumentError PIDExperiments.RefinementSpec(
        :steps_per_interval,(8,big(typemax(Int))+1))
    @test_throws ArgumentError PIDExperiments.PIExperiment(
        model;task=:dynamics,tspan=(0.0,0.1))
    @test_throws ArgumentError PIDExperiments.PIExperiment(
        model;task=:steady_state,solver_options=(return_info=true,))
    @test_throws ArgumentError PIDExperiments.PIExperiment(
        model;task=:steady_state,algorithm=GMRESAlgorithm(krylovdim=8),
        solver_options=(krylovdim=6,))
    @test_throws ArgumentError PIDExperiments.PIExperiment(
        model;task=:steady_state,
        verification=PIDExperiments.VerificationSpec(
            refinement=PIDExperiments.RefinementSpec(
                :steps_per_interval,(8,16))))

    mutable_initial=iid_pure_state(basis,ComplexF64[0,1])
    mutable_saveat=[0.0,0.05]
    mutable_tspan=[0.0,0.05]
    mutable_parameters=[2.0]
    mutable_observable=copy(local_z)
    detached=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=mutable_initial,
        tspan=mutable_tspan,saveat=mutable_saveat,
        parameters=mutable_parameters,
        observables=(z=mutable_observable,),verification)
    saved_initial=copy(detached.initial_state.data)
    saved_observable=copy(last(first(detached.observables)).data)
    mutable_initial.data[1]+=1
    mutable_saveat[2]=0.04
    mutable_tspan[2]=0.04
    mutable_parameters[1]=3.0
    mutable_observable[1,1]=9
    @test detached.initial_state.data==saved_initial
    @test detached.saveat==[0.0,0.05]
    @test detached.tspan==(0.0,0.05)
    @test detached.parameters==[2.0]
    @test last(first(detached.observables)).data==saved_observable
    range_saveat=range(0.0,0.05;length=3)
    range_experiment=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=rho0,tspan=(0.0,0.05),
        saveat=range_saveat,observables=(z=local_z,),verification)
    @test range_experiment.saveat===range_saveat

    base_refinement_experiment=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=rho0,tspan=(0.0,0.05),
        saveat=[0.0,0.05],observables=(magnetization=local_z,),
        verification=refined_verification,memory_budget=Inf)
    cumulative=PIDExperiments._experiment_refinement_peak_bytes(
        base_refinement_experiment)
    one_level_peak=PIDExperiments.plan_experiment(
        base_refinement_experiment).recommendation.known_peak_bytes
    @test cumulative.peak_bytes>one_level_peak
    guarded_budget=one_level_peak+
        (cumulative.peak_bytes-one_level_peak)÷2
    guarded_refinement=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=rho0,tspan=(0.0,0.05),
        saveat=[0.0,0.05],observables=(magnetization=local_z,),
        verification=refined_verification,memory_budget=guarded_budget)
    guarded_plan=PIDExperiments.plan_experiment(guarded_refinement)
    @test guarded_plan.recommendation.known_peak_bytes<=guarded_budget
    @test guarded_plan.recommendation.budget_status!==:exceeds
    @test_throws ArgumentError PIDExperiments.verified_solve(
        guarded_refinement)

    provenance_a=PIDExperiments._experiment_provenance(
        PIDExperiments.PIExperiment(model;memory_budget=10^7,
            verification=PIDExperiments.VerificationSpec(atol=1e-9)))
    provenance_b=PIDExperiments._experiment_provenance(
        PIDExperiments.PIExperiment(model;memory_budget=10^7+1,
            verification=PIDExperiments.VerificationSpec(atol=1e-9)))
    provenance_c=PIDExperiments._experiment_provenance(
        PIDExperiments.PIExperiment(model;memory_budget=10^7,
            verification=PIDExperiments.VerificationSpec(atol=2e-9)))
    @test provenance_a.structural_digest!=provenance_b.structural_digest
    @test provenance_a.structural_digest!=provenance_c.structural_digest

    # A raw experiment owns fixed built-in operator data rather than aliasing
    # the user's mutable one-site matrix.
    mutable_jump=ComplexF64[0 1;0 0]
    mutable_model=PIModel(basis,(LocalJump(mutable_jump;rate=0.3),))
    owned=PIDExperiments.PIExperiment(mutable_model)
    owned_jump=PIDExperiments.term_operator(only(owned.source.terms))
    @test owned_jump!==mutable_jump
    mutable_jump[1,2]=7
    @test owned_jump==ComplexF64[0 1;0 0]
    @test_throws ArgumentError PIDExperiments.PIExperiment(
        mutable_model;memory_budget=0)

    # Equal BigFloat values with different stored precisions describe
    # different numerical experiments even under one ambient context.
    low_precision=setprecision(BigFloat,64) do
        BigFloat(1)
    end
    high_precision=setprecision(BigFloat,128) do
        BigFloat(1)
    end
    low_experiment=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=rho0,tspan=(0.0,0.01),
        saveat=[0.0,0.01],parameters=[low_precision],
        observables=(z=local_z,),verification)
    high_experiment=PIDExperiments.PIExperiment(
        model;task=:dynamics,initial_state=rho0,tspan=(0.0,0.01),
        saveat=[0.0,0.01],parameters=[high_precision],
        observables=(z=local_z,),verification)
    low_provenance=PIDExperiments._experiment_provenance(low_experiment)
    high_provenance=PIDExperiments._experiment_provenance(high_experiment)
    @test low_provenance.structural_digest!=
        high_provenance.structural_digest
    @test low_provenance.bigfloat_precision==precision(BigFloat)
    @test low_provenance.bigfloat_rounding==string(rounding(BigFloat))

    # Archive loading rejects semantic inconsistencies after bounded header
    # inspection rather than returning a partially coherent result.
    mktempdir() do directory
        bad_kind=joinpath(directory,"bad-kind.pidrun")
        PIDExperiments.save_experiment(bad_kind,dynamics_result)
        manifest=PIDExperiments._experiment_read_manifest(
            joinpath(bad_kind,"manifest.pidexp"))
        manifest.metadata["observable.magnetization.kind"]="scalar"
        PIDExperiments._experiment_write_manifest(
            joinpath(bad_kind,"manifest.pidexp"),manifest.task,
            manifest.state_count,manifest.names,manifest.metadata)
        @test_throws ArgumentError PIDExperiments.load_experiment(bad_kind)

        bad_series=joinpath(directory,"bad-series.pidrun")
        PIDExperiments.save_experiment(bad_series,dynamics_result)
        PIDExperiments._experiment_save_vector(
            joinpath(bad_series,"observable_000001.pidvec"),[0.0])
        @test_throws DimensionMismatch PIDExperiments.load_experiment(
            bad_series)

        bad_basis=joinpath(directory,"bad-basis.pidrun")
        PIDExperiments.save_experiment(bad_basis,dynamics_result)
        other_basis=PIBasis(2,2;sectors=[(2,0)])
        other_state=PIState(other_basis;T=Float64)
        PIDExperiments.save_checkpoint(
            joinpath(bad_basis,"state_000002.pid"),other_state)
        @test_throws ArgumentError PIDExperiments.load_experiment(bad_basis)
    end
end
