const PIDINFER=PermutationalInvariantDynamics

@testset "least-squares inference and identifiability" begin
    design=Float64[
        1 2
        3 -1
        1 0
        0 1]
    target_parameters=[0.7,-0.25]
    observations=design*target_parameters
    problem=PIDINFER.LeastSquaresInferenceProblem(
        parameters->design*parameters,
        observations,[0.1,0.2];
        jacobian=parameters->design,
        standard_deviations=[0.2,0.3,0.4,0.5],
        parameter_names=(:drive,:detuning))
    result=PIDINFER.fit_parameters(
        problem;derivative_method=:auto,maxiter=20,
        gradient_tolerance=1e-12,step_tolerance=1e-12,
        objective_tolerance=1e-14,require_convergence=true)
    @test result.converged
    @test result.derivative_method===:analytic
    @test result.parameters≈target_parameters atol=2e-11 rtol=2e-11
    @test result.objective<1e-20
    @test result.identifiability.identifiable
    @test result.identifiability.numerical_rank==2
    @test result.identifiability.covariance!==nothing
    @test result.identifiability.parameter_names==(:drive,:detuning)
    @test occursin("ParameterInferenceResult",sprint(show,result))

    diagnostics=PIDINFER.parameter_identifiability(
        problem,target_parameters;derivative_method=:analytic)
    weighted_design=design./problem.standard_deviations
    @test diagnostics.fisher≈weighted_design'*weighted_design
    @test diagnostics.identifiable

    finite_problem=PIDINFER.LeastSquaresInferenceProblem(
        parameters->[
            parameters[1]^2+parameters[2],
            parameters[1]+parameters[2]^2,
            parameters[1]-parameters[2]],
        [0.8^2-0.3,0.8+(-0.3)^2,0.8-(-0.3)],
        [0.55,-0.05];
        lower_bounds=[0.0,-0.6],
        upper_bounds=[1.2,0.2])
    finite_result=PIDINFER.fit_parameters(
        finite_problem;maxiter=40,gradient_tolerance=2e-9,
        step_tolerance=2e-10,objective_tolerance=1e-13)
    @test finite_result.converged
    @test finite_result.derivative_method===:finite_difference
    @test finite_result.parameters≈[0.8,-0.3] atol=2e-7 rtol=2e-7
    derivative_metadata=
        finite_result.identifiability.metadata.derivative_metadata
    @test all(scheme->scheme in (:central,:forward,:backward),
              derivative_metadata.schemes)

    bounded=PIDINFER.LeastSquaresInferenceProblem(
        parameters->[parameters[1]],[2.0],[0.2];
        lower_bounds=[0.0],upper_bounds=[1.0])
    bounded_result=PIDINFER.fit_parameters(
        bounded;maxiter=20,gradient_tolerance=1e-12,
        step_tolerance=1e-12,objective_tolerance=1e-14)
    @test bounded_result.parameters[1]<=1
    @test bounded_result.converged
    @test bounded_result.termination===:gradient_tolerance

    deficient=PIDINFER.LeastSquaresInferenceProblem(
        parameters->[parameters[1]+parameters[2],
                     2(parameters[1]+parameters[2])],
        [1.0,2.0],[0.2,0.8];
        jacobian=parameters->[1.0 1.0;2.0 2.0])
    deficient_report=PIDINFER.parameter_identifiability(
        deficient;derivative_method=:analytic)
    @test !deficient_report.identifiable
    @test deficient_report.numerical_rank==1
    @test deficient_report.covariance===nothing
    @test all(isfinite,deficient_report.pseudocovariance)

    tall_design=reshape(
        Float64[isodd(index) ? 1 : -1 for index in 1:400],200,2)
    tall_problem=PIDINFER.LeastSquaresInferenceProblem(
        parameters->tall_design*parameters,zeros(200),zeros(2);
        jacobian=parameters->tall_design)
    tall_report=PIDINFER.parameter_identifiability(
        tall_problem;derivative_method=:analytic)
    @test size(tall_report.fisher)==(2,2)
    @test size(tall_report.pseudocovariance)==(2,2)

    wide_problem=PIDINFER.LeastSquaresInferenceProblem(
        parameters->[sum(parameters)],[0.0],zeros(3);
        jacobian=parameters->ones(1,3))
    wide_report=PIDINFER.parameter_identifiability(
        wide_problem;derivative_method=:analytic)
    @test !wide_report.identifiable
    @test size(wide_report.pseudocovariance)==(3,3)
    @test all(isfinite,wide_report.pseudocovariance)

    @test_throws ArgumentError PIDINFER.LeastSquaresInferenceProblem(
        parameters->[parameters[1]],[1.0],[0.2];
        standard_deviations=[0.0])
    @test_throws ArgumentError PIDINFER.LeastSquaresInferenceProblem(
        parameters->[parameters[1]],[1.0],[1.2];
        lower_bounds=[0.0],upper_bounds=[1.0])
    @test_throws ArgumentError PIDINFER.fit_parameters(
        problem;derivative_method=:implicit_steady_state)
    @test_throws ArgumentError PIDINFER.fit_parameters(
        problem;memory_budget=1)
    @test_throws ArgumentError PIDINFER.parameter_identifiability(
        problem;memory_budget=1)
end

@testset "inference conversion and solver guards" begin
    @test PIDINFER._inference_checked_real(1.0,"value",Float32)==1.0f0
    @test_throws ArgumentError PIDINFER._inference_checked_real(
        big"1e-1000","value",Float32)
    @test_throws ArgumentError PIDINFER._inference_checked_real(
        big"1e1000","value",Float32)
    @test PIDINFER._inference_checked_real(
        Inf,"bound",Float32;allow_infinite=true)===Float32(Inf)

    @test_throws ArgumentError PIDINFER.LeastSquaresInferenceProblem(
        parameters->Float32[parameters[1]],Float32[1],Float32[1];
        lower_bounds=[big"1e-1000"])
    @test_throws ArgumentError PIDINFER.LeastSquaresInferenceProblem(
        parameters->Float32[parameters[1]],Float32[1],Float32[1];
        upper_bounds=[big"1e1000"])

    prediction_underflow=PIDINFER.LeastSquaresInferenceProblem(
        parameters->[1e-100],Float32[0],Float32[1])
    @test_throws ArgumentError PIDINFER.parameter_identifiability(
        prediction_underflow)

    jacobian_underflow=PIDINFER.LeastSquaresInferenceProblem(
        parameters->Float32[parameters[1]],Float32[1],Float32[1];
        jacobian=parameters->reshape([1e-100],1,1))
    @test_throws ArgumentError PIDINFER.parameter_identifiability(
        jacobian_underflow;derivative_method=:analytic)

    weighted_overflow=PIDINFER.LeastSquaresInferenceProblem(
        parameters->Float32[parameters[1]],Float32[0],Float32[0];
        jacobian=parameters->reshape(Float32[floatmax(Float32)],1,1),
        standard_deviations=Float32[floatmin(Float32)])
    weighted_overflow_error=try
        PIDINFER.parameter_identifiability(
            weighted_overflow;derivative_method=:analytic)
        nothing
    catch error
        error
    end
    @test weighted_overflow_error isa ArgumentError
    @test occursin("wider inference data",
        sprint(showerror,weighted_overflow_error))

    weighted_underflow=PIDINFER.LeastSquaresInferenceProblem(
        parameters->Float32[parameters[1]],Float32[0],Float32[0];
        jacobian=parameters->reshape(
            Float32[nextfloat(0.0f0)],1,1),
        standard_deviations=Float32[floatmax(Float32)])
    weighted_underflow_error=try
        PIDINFER.parameter_identifiability(
            weighted_underflow;derivative_method=:analytic)
        nothing
    catch error
        error
    end
    @test weighted_underflow_error isa ArgumentError
    @test occursin("underflows to zero",
        sprint(showerror,weighted_underflow_error))

    objective_underflow=PIDINFER.LeastSquaresInferenceProblem(
        parameters->Float32[nextfloat(0.0f0)],Float32[0],Float32[0];
        jacobian=parameters->reshape(Float32[1],1,1))
    @test_throws ArgumentError PIDINFER.fit_parameters(
        objective_underflow;derivative_method=:analytic)

    residual_overflow=PIDINFER.LeastSquaresInferenceProblem(
        parameters->Float32[floatmax(Float32)],
        Float32[-floatmax(Float32)],Float32[0];
        jacobian=parameters->reshape(Float32[1],1,1))
    @test_throws ArgumentError PIDINFER.fit_parameters(
        residual_overflow;derivative_method=:analytic)

    @test PIDINFER._inference_require_stationary_convergence(
        (converged=true,))===nothing
    @test_throws ArgumentError PIDINFER._inference_require_stationary_convergence(
        (converged=false,))
    @test_throws ArgumentError PIDINFER._inference_require_stationary_convergence(
        (residual=0.0,))
end

@testset "implicit steady-state parameter inference" begin
    basis=PIBasis(1,2)
    spin=spin_matrices()
    lowering=ComplexF64[0 1;0 0]
    excited=collective_operator(
        basis,ComplexF64[0 0;0 1])
    model_builder=parameters->PIModel(basis,(
        CollectiveJump(lowering;rate=1.0),
        LocalHamiltonian(spin.jx;rate=parameters[1])))
    derivative_builder=(parameters,model)->(
        PIModel(basis,(LocalHamiltonian(spin.jx;rate=1.0),)),)
    true_parameter=[0.36]
    true_state=stationary_state(
        model_builder(true_parameter);algorithm=DirectAlgorithm())
    observations=[real(expectation(true_state,excited))]
    problem=PIDINFER.steady_state_inference_problem(
        model_builder,(excited,),observations,[0.12];
        derivative_builder,
        solver_options=(algorithm=DirectAlgorithm(),),
        gradient_options=(krylovdim=4,maxiter=100,),
        lower_bounds=[0.02],upper_bounds=[0.8],
        parameter_names=(:drive,))
    report=PIDINFER.parameter_identifiability(
        problem,true_parameter;derivative_method=:auto)
    @test report.metadata.derivative_method===:implicit_steady_state
    @test report.identifiable
    epsilon=1e-5
    plus=stationary_state(
        model_builder([true_parameter[1]+epsilon]);
        algorithm=DirectAlgorithm())
    minus=stationary_state(
        model_builder([true_parameter[1]-epsilon]);
        algorithm=DirectAlgorithm())
    finite=(real(expectation(plus,excited))-
            real(expectation(minus,excited)))/(2epsilon)
    @test isapprox(sqrt(report.fisher[1,1]),abs(finite);
                   atol=2e-7,rtol=2e-6)

    result=PIDINFER.fit_parameters(
        problem;maxiter=25,gradient_tolerance=1e-10,
        step_tolerance=1e-10,objective_tolerance=1e-14)
    @test result.converged
    @test result.derivative_method===:implicit_steady_state
    @test isapprox(result.parameters[1],true_parameter[1];
                   atol=3e-6,rtol=3e-6)
    @test result.metadata.predictor_metadata.workflow===
          :steady_state_observables

    finite_problem=PIDINFER.steady_state_inference_problem(
        model_builder,(excited,),observations,[0.12];
        solver_options=(algorithm=DirectAlgorithm(),),
        lower_bounds=[0.02],upper_bounds=[0.8])
    finite_report=PIDINFER.parameter_identifiability(
        finite_problem,true_parameter)
    @test finite_report.metadata.derivative_method===:finite_difference
end
