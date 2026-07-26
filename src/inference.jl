"""
    LeastSquaresInferenceProblem(predictor, observations, initial_parameters;
                                 jacobian=nothing,
                                 standard_deviations=nothing,
                                 parameter_names=nothing,
                                 lower_bounds=nothing, upper_bounds=nothing,
                                 metadata=(;))

Immutable description of a real weighted least-squares problem. `predictor`
maps a parameter vector to predicted observations. An optional `jacobian`
returns a matrix whose rows are observations and columns are parameters.
Without it, [`fit_parameters`](@ref) can use explicitly reported finite
differences. `standard_deviations` are known positive observation errors; the
objective is

```math
\\frac{1}{2}\\sum_j
\\left(\\frac{f_j(\\theta)-y_j}{\\sigma_j}\\right)^2.
```

Bounds are enforced exactly as supplied. The constructor copies all numerical
inputs but cannot make user callbacks immutable or task safe.
"""
struct LeastSquaresInferenceProblem{F,J,R,N,M}
    predictor::F
    jacobian::J
    observations::Vector{R}
    standard_deviations::Vector{R}
    initial_parameters::Vector{R}
    parameter_names::N
    lower_bounds::Vector{R}
    upper_bounds::Vector{R}
    metadata::M
end

function _inference_float_type(observations,parameters,standard_deviations)
    T=promote_type(
        mapreduce(typeof,promote_type,observations),
        mapreduce(typeof,promote_type,parameters),
        mapreduce(typeof,promote_type,standard_deviations))
    R=float(T)
    R in (Float32,Float64)||throw(ArgumentError(
        "least-squares inference currently requires Float32 or Float64 real " *
        "inputs because identifiability diagnostics use LAPACK"))
    R
end

function _inference_checked_real(value,label,::Type{R};
        allow_infinite::Bool=false) where R
    value isa Real||throw(ArgumentError("$label must be a real number"))
    if allow_infinite
        isnan(value)&&throw(ArgumentError("$label must not be NaN"))
    else
        isfinite(value)||throw(ArgumentError("$label must be finite"))
    end
    converted=try
        R(value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("$label is not representable in $R"))
    end
    if allow_infinite
        isnan(converted)&&throw(ArgumentError("$label is NaN in $R"))
        isfinite(value)&&!isfinite(converted)&&throw(ArgumentError(
            "$label overflows $R"))
    else
        isfinite(converted)||throw(ArgumentError(
            "$label is not finite in $R"))
    end
    !iszero(value)&&iszero(converted)&&throw(ArgumentError(
        "$label underflows to zero in $R; use wider inference data"))
    converted
end

function _inference_real_vector(values,label,::Type{R}) where R
    output=Vector{R}(undef,length(values))
    for index in eachindex(values)
        output[index]=_inference_checked_real(
            values[index],"$label $index",R)
    end
    output
end

function _inference_bound_vector(values,count,default,label,::Type{R}) where R
    values===nothing&&return fill(R(default),count)
    raw=values isa Real ? fill(values,count) : collect(values)
    length(raw)==count||throw(DimensionMismatch(
        "$label must have one entry per parameter"))
    output=Vector{R}(undef,count)
    for index in eachindex(raw)
        output[index]=_inference_checked_real(
            raw[index],"$label $index",R;allow_infinite=true)
    end
    output
end

function LeastSquaresInferenceProblem(predictor,observations,
        initial_parameters;jacobian=nothing,standard_deviations=nothing,
        parameter_names=nothing,lower_bounds=nothing,upper_bounds=nothing,
        metadata::NamedTuple=(;))
    observed=collect(observations)
    initial=collect(initial_parameters)
    isempty(observed)&&throw(ArgumentError(
        "at least one observation is required"))
    isempty(initial)&&throw(ArgumentError(
        "at least one inferred parameter is required"))
    all(value->value isa Real,observed)||throw(ArgumentError(
        "least-squares observations must be real"))
    all(value->value isa Real,initial)||throw(ArgumentError(
        "least-squares parameters must be real"))
    raw_sigmas=standard_deviations===nothing ?
        fill(one(float(first(observed))),length(observed)) :
        (standard_deviations isa Real ?
            fill(standard_deviations,length(observed)) :
            collect(standard_deviations))
    length(raw_sigmas)==length(observed)||throw(DimensionMismatch(
        "standard_deviations must have one entry per observation"))
    R=_inference_float_type(observed,initial,raw_sigmas)
    y=_inference_real_vector(observed,"observation",R)
    theta=_inference_real_vector(initial,"initial parameter",R)
    sigmas=_inference_real_vector(
        raw_sigmas,"standard deviation",R)
    all(>(zero(R)),sigmas)||throw(ArgumentError(
        "standard deviations must be strictly positive"))
    lower=_inference_bound_vector(
        lower_bounds,length(theta),-Inf,"lower bound",R)
    upper=_inference_bound_vector(
        upper_bounds,length(theta),Inf,"upper bound",R)
    all(lower.<=upper)||throw(ArgumentError(
        "every lower bound must be no greater than its upper bound"))
    all((lower.<=theta).&(theta.<=upper))||throw(ArgumentError(
        "initial parameters must satisfy their bounds"))
    names=if parameter_names===nothing
        Tuple(Symbol("theta_",index) for index in eachindex(theta))
    else
        raw=Tuple(Symbol(name) for name in parameter_names)
        length(raw)==length(theta)||throw(DimensionMismatch(
            "parameter_names must have one entry per parameter"))
        length(unique(raw))==length(raw)||throw(ArgumentError(
            "parameter names must be unique"))
        raw
    end
    applicable(predictor,theta)||throw(ArgumentError(
        "predictor must accept one parameter vector"))
    jacobian===nothing||applicable(jacobian,theta)||throw(ArgumentError(
        "jacobian must accept one parameter vector"))
    LeastSquaresInferenceProblem(
        predictor,jacobian,y,sigmas,theta,names,lower,upper,metadata)
end

"""
    ParameterIdentifiabilityReport

Local weighted-Jacobian diagnostics at one parameter point. `identifiable`
means the numerical Jacobian has full column rank at the explicit tolerance.
`covariance` is available only in that case. `pseudocovariance` is always the
truncated-SVD local generalized inverse and must not be presented as an
uncertainty certificate for a rank-deficient model.
"""
struct ParameterIdentifiabilityReport{R,M,N}
    fisher::Matrix{R}
    singular_values::Vector{R}
    numerical_rank::Int
    parameter_count::Int
    identifiable::Bool
    condition_number::R
    covariance::Union{Nothing,Matrix{R}}
    pseudocovariance::Matrix{R}
    correlation::Union{Nothing,Matrix{R}}
    tolerance::R
    parameter_names::N
    metadata::M
end

"""
    ParameterInferenceResult

Bounded Levenberg--Marquardt result. `converged=false` is retained rather than
repaired or relabelled. `derivative_method` is the method actually used:
`:analytic`, `:implicit_steady_state`, or `:finite_difference`.
"""
struct ParameterInferenceResult{R,I,H,M}
    parameters::Vector{R}
    predictions::Vector{R}
    residuals::Vector{R}
    standardized_residuals::Vector{R}
    objective::R
    iterations::Int
    converged::Bool
    termination::Symbol
    derivative_method::Symbol
    identifiability::I
    history::H
    metadata::M
end

function Base.show(io::IO,result::ParameterInferenceResult)
    print(io,"ParameterInferenceResult(objective=$(result.objective), " *
        "iterations=$(result.iterations), converged=$(result.converged), " *
        "method=$(result.derivative_method), " *
        "identifiable=$(result.identifiability.identifiable))")
end

struct _SteadyStateInferencePredictor{B,D,O,C,S,G}
    model_builder::B
    derivative_builder::D
    observables::O
    compile_options::C
    solver_options::S
    gradient_options::G
end

function _inference_call_builder(builder,parameters)
    if applicable(builder,parameters)
        builder(parameters)
    elseif applicable(builder,parameters...)
        builder(parameters...)
    else
        throw(ArgumentError(
            "model_builder must accept the parameter vector or its scalar entries"))
    end
end

function _inference_call_derivative_builder(builder,parameters,model)
    if applicable(builder,parameters,model)
        builder(parameters,model)
    elseif applicable(builder,parameters)
        builder(parameters)
    elseif applicable(builder,parameters...)
        builder(parameters...)
    else
        throw(ArgumentError(
            "derivative_builder must accept (parameters, model), the " *
            "parameter vector, or its scalar entries"))
    end
end

function _inference_forbid(options::NamedTuple,keys,label)
    found=Symbol[key for key in keys if haskey(options,key)]
    isempty(found)||throw(ArgumentError(
        "$label contains internally managed keyword(s): " *
        join(string.(found),", ")))
    options
end

"""
    steady_state_inference_problem(model_builder, observables, observations,
                                   initial_parameters;
                                   derivative_builder=nothing,
                                   compile_options=(backend=:matrixfree,),
                                   solver_options=(;), gradient_options=(;),
                                   kwargs...)

Construct a [`LeastSquaresInferenceProblem`](@ref) whose predictions are
steady-state PI expectation values. `model_builder` returns a `PIModel` or
prepared PI model for the current parameter vector. If `derivative_builder`
returns the corresponding generators `dL/dtheta_mu`, `derivative_method=:auto`
uses [`implicit_steady_state_gradient`](@ref) and reuses the already solved
stationary state. Otherwise the inference report explicitly records finite
differences.

The observables must be Hermitian `PIOperator`s on the exact model basis.
Compilation and solver choices remain explicit; no failed solve is converted
to an approximate prediction.
"""
function steady_state_inference_problem(model_builder,observables,
        observations,initial_parameters;derivative_builder=nothing,
        compile_options::NamedTuple=(backend=:matrixfree,),
        solver_options::NamedTuple=(;),
        gradient_options::NamedTuple=(;),metadata::NamedTuple=(;),kwargs...)
    ops=Tuple(observables)
    isempty(ops)&&throw(ArgumentError(
        "at least one steady-state observable is required"))
    all(operator->operator isa PIOperator&&ishermitian(operator),ops)||
        throw(ArgumentError(
        "steady-state inference observables must be Hermitian PIOperators"))
    _inference_forbid(
        solver_options,(:return_info,),"solver_options")
    _inference_forbid(
        gradient_options,(:observables,),"gradient_options")
    predictor=_SteadyStateInferencePredictor(
        model_builder,derivative_builder,ops,compile_options,
        solver_options,gradient_options)
    metadata=merge((
        workflow=:steady_state_observables,
        derivative_source=derivative_builder===nothing ?
            :finite_difference : :implicit_steady_state,
        observation_count=length(ops)),metadata)
    LeastSquaresInferenceProblem(
        predictor,observations,initial_parameters;
        kwargs...,metadata)
end

function _inference_validate_prediction(problem,values)
    raw=collect(values)
    length(raw)==length(problem.observations)||throw(DimensionMismatch(
        "predictor returned $(length(raw)) values for " *
        "$(length(problem.observations)) observations"))
    _inference_real_vector(
        raw,"predicted observation",eltype(problem.observations))
end

function _inference_steady_context(predictor,parameters)
    model=_inference_call_builder(predictor.model_builder,parameters)
    source=if model isa PIModel
        compile(model;predictor.compile_options...)
    elseif model isa Union{CompiledPIModel,SpecializedPIModel}
        model
    else
        throw(ArgumentError(
            "steady-state inference model_builder must return PIModel, " *
            "CompiledPIModel, or SpecializedPIModel"))
    end
    basis=_operator_basis(source)
    basis===nothing&&throw(ArgumentError(
        "steady-state inference source has no PI basis metadata"))
    all(operator->operator.basis===basis,predictor.observables)||
        throw(ArgumentError(
        "steady-state inference observables and model must use the exact same basis"))
    stationary=stationary_state(
        source;return_info=true,predictor.solver_options...)
    _inference_require_stationary_convergence(stationary.info)
    raw_values=[expectation(stationary.state,operator)
                for operator in predictor.observables]
    R=_real_float_type(eltype(stationary.state.data))
    tolerance=R(64)*eps(R)*max(one(R),
        maximum(abs,raw_values;init=zero(R)))
    all(value->abs(imag(value))<=tolerance,raw_values)||
        throw(ArgumentError(
        "a Hermitian steady-state observable has an appreciably complex " *
        "expectation value; validate the stationary state before inference"))
    values=real.(raw_values)
    values,(;model,source,stationary)
end

function _inference_require_stationary_convergence(info)
    status=hasproperty(info,:converged) ? getproperty(info,:converged) : missing
    status===true||throw(ArgumentError(
        "steady-state inference requires a stationary solve with " *
        "converged=true; received $(repr(status))"))
    nothing
end

function (predictor::_SteadyStateInferencePredictor)(parameters)
    first(_inference_steady_context(predictor,parameters))
end

function _inference_predict(problem,parameters)
    if problem.predictor isa _SteadyStateInferencePredictor
        values,context=_inference_steady_context(
            problem.predictor,parameters)
        return _inference_validate_prediction(problem,values),context,
            (solver_info=context.stationary.info,)
    end
    values=problem.predictor(parameters)
    _inference_validate_prediction(problem,values),nothing,(;)
end

function _inference_validate_jacobian(problem,jacobian)
    rows=length(problem.observations)
    columns=length(problem.initial_parameters)
    size(jacobian)==(rows,columns)||throw(DimensionMismatch(
        "Jacobian must have size ($rows, $columns)"))
    raw=Matrix(jacobian)
    all(value->value isa Real&&isfinite(value),raw)||throw(ArgumentError(
        "Jacobian entries must be finite and real"))
    R=eltype(problem.observations)
    converted=Matrix{R}(undef,rows,columns)
    for index in eachindex(raw)
        converted[index]=_inference_checked_real(
            raw[index],"Jacobian entry $index",R)
    end
    converted
end

function _inference_checked_ratio(numerator::R,denominator::R,label) where
        {R<:AbstractFloat}
    value=numerator/denominator
    isfinite(value)||throw(ArgumentError(
        "$label overflows $R; use wider inference data"))
    !iszero(numerator)&&iszero(value)&&throw(ArgumentError(
        "$label underflows to zero in $R; use wider inference data"))
    value
end

function _inference_weight_rows(matrix::AbstractMatrix{R},
        standard_deviations::AbstractVector{R},label) where
        {R<:AbstractFloat}
    size(matrix,1)==length(standard_deviations)||throw(DimensionMismatch(
        "$label row count must match the standard-deviation vector"))
    output=Matrix{R}(undef,size(matrix))
    for column in axes(matrix,2),row in axes(matrix,1)
        output[row,column]=_inference_checked_ratio(
            matrix[row,column],standard_deviations[row],
            "$label entry ($row, $column)")
    end
    output
end

function _inference_standardize(residual::AbstractVector{R},
        standard_deviations::AbstractVector{R}) where {R<:AbstractFloat}
    length(residual)==length(standard_deviations)||throw(DimensionMismatch(
        "residual and standard-deviation vectors have different lengths"))
    output=Vector{R}(undef,length(residual))
    for index in eachindex(residual,standard_deviations)
        output[index]=_inference_checked_ratio(
            residual[index],standard_deviations[index],
            "standardized residual $index")
    end
    output
end

function _inference_residual(prediction::AbstractVector{R},
        observations::AbstractVector{R}) where {R<:AbstractFloat}
    length(prediction)==length(observations)||throw(DimensionMismatch(
        "prediction and observation vectors have different lengths"))
    output=prediction-observations
    all(isfinite,output)||throw(ArgumentError(
        "prediction residuals overflow $R; use wider inference data"))
    output
end

function _inference_objective(standardized::AbstractVector{R}) where
        {R<:AbstractFloat}
    squared=sum(abs2,standardized)
    isfinite(squared)||throw(ArgumentError(
        "least-squares objective overflows $R; use wider inference data"))
    any(!iszero,standardized)&&iszero(squared)&&throw(ArgumentError(
        "least-squares objective underflows to zero in $R; use wider inference data"))
    objective=squared/R(2)
    !iszero(squared)&&iszero(objective)&&throw(ArgumentError(
        "least-squares objective underflows to zero in $R; use wider inference data"))
    objective
end

function _inference_checked_gram(weighted::AbstractMatrix{R}) where
        {R<:AbstractFloat}
    fisher=Matrix{R}(weighted'*weighted)
    all(isfinite,fisher)||throw(ArgumentError(
        "weighted Fisher matrix overflows $R; use wider inference data"))
    for column in axes(weighted,2)
        any(!iszero,@view weighted[:,column])&&
            iszero(fisher[column,column])&&throw(ArgumentError(
            "weighted Fisher diagonal $column underflows to zero in $R; " *
            "use wider inference data"))
    end
    fisher
end

function _inference_fd_steps(problem,parameters,finite_difference_step)
    R=eltype(parameters)
    count=length(parameters)
    raw=if finite_difference_step===nothing
        [cbrt(eps(R))*max(abs(value),one(R)) for value in parameters]
    elseif finite_difference_step isa Real
        fill(finite_difference_step,count)
    else
        collect(finite_difference_step)
    end
    length(raw)==count||throw(DimensionMismatch(
        "finite_difference_step must have one entry per parameter"))
    steps=_inference_real_vector(raw,"finite-difference step",R)
    all(>(zero(R)),steps)||throw(ArgumentError(
        "finite-difference steps must be strictly positive"))
    steps
end

function _inference_fd_jacobian(problem,parameters,prediction;
        finite_difference_step=nothing)
    rows=length(prediction);columns=length(parameters)
    R=eltype(parameters)
    jacobian=zeros(R,rows,columns)
    steps=_inference_fd_steps(
        problem,parameters,finite_difference_step)
    schemes=Vector{Symbol}(undef,columns)
    for column in 1:columns
        h=steps[column]
        plus=copy(parameters);minus=copy(parameters)
        plus[column]+=h;minus[column]-=h
        can_plus=plus[column]<=problem.upper_bounds[column]&&
                 plus[column]!=parameters[column]
        can_minus=minus[column]>=problem.lower_bounds[column]&&
                  minus[column]!=parameters[column]
        if can_plus&&can_minus
            plus_prediction,_,_=_inference_predict(problem,plus)
            minus_prediction,_,_=_inference_predict(problem,minus)
            @views jacobian[:,column].=
                (plus_prediction-minus_prediction)/
                (plus[column]-minus[column])
            schemes[column]=:central
        elseif can_plus
            plus_prediction,_,_=_inference_predict(problem,plus)
            @views jacobian[:,column].=(plus_prediction-prediction)/
                (plus[column]-parameters[column])
            schemes[column]=:forward
        elseif can_minus
            minus_prediction,_,_=_inference_predict(problem,minus)
            @views jacobian[:,column].=(prediction-minus_prediction)/
                (parameters[column]-minus[column])
            schemes[column]=:backward
        else
            throw(ArgumentError(
                "parameter $column has no representable finite-difference " *
                "step inside its bounds"))
        end
    end
    jacobian,(steps=copy(steps),schemes=Tuple(schemes))
end

function _inference_derivative_method(problem,method)
    method in (:auto,:analytic,:finite_difference,:implicit_steady_state)||
        throw(ArgumentError(
        "derivative_method must be :auto, :analytic, :finite_difference, " *
        "or :implicit_steady_state"))
    if method===:auto
        if problem.predictor isa _SteadyStateInferencePredictor &&
           problem.predictor.derivative_builder!==nothing
            return :implicit_steady_state
        end
        return problem.jacobian===nothing ?
            :finite_difference : :analytic
    end
    method===:analytic&&problem.jacobian===nothing&&throw(ArgumentError(
        "derivative_method=:analytic requires a jacobian callback"))
    if method===:implicit_steady_state
        problem.predictor isa _SteadyStateInferencePredictor||throw(ArgumentError(
            "implicit steady-state derivatives require a problem created by " *
            "steady_state_inference_problem"))
        problem.predictor.derivative_builder!==nothing||throw(ArgumentError(
            "implicit steady-state derivatives require derivative_builder"))
    end
    method
end

function _inference_jacobian(problem,parameters,prediction,context,method;
        finite_difference_step=nothing)
    if method===:finite_difference
        jacobian,metadata=_inference_fd_jacobian(
            problem,parameters,prediction;finite_difference_step)
        return _inference_validate_jacobian(problem,jacobian),metadata
    elseif method===:analytic
        return _inference_validate_jacobian(
            problem,problem.jacobian(parameters)),(;)
    end
    predictor=problem.predictor
    context===nothing&&throw(ArgumentError(
        "implicit steady-state gradient context is missing"))
    derivatives=Tuple(_inference_call_derivative_builder(
        predictor.derivative_builder,parameters,context.model))
    length(derivatives)==length(parameters)||throw(DimensionMismatch(
        "derivative_builder must return one generator derivative per parameter"))
    result=implicit_steady_state_gradient(
        context.source,context.stationary.state,derivatives;
        observables=predictor.observables,
        predictor.gradient_options...)
    jacobian=_inference_validate_jacobian(
        problem,result.observable_gradients)
    jacobian,(gradient_solver_info=result.solver_info,)
end

function _inference_rank_tolerance(value,::Type{R},rows,columns) where R
    if value===nothing
        return R(max(rows,columns))*sqrt(eps(R))
    end
    tolerance=_inference_real_vector(
        [value],"rank_rtol",R)[1]
    tolerance>=zero(R)||throw(ArgumentError(
        "rank_rtol must be nonnegative"))
    tolerance
end

function _inference_identifiability(problem,weighted_jacobian;
        rank_rtol=nothing,metadata=(;))
    R=eltype(weighted_jacobian)
    rows,columns=size(weighted_jacobian)
    all(isfinite,weighted_jacobian)||throw(ArgumentError(
        "weighted Jacobian entries must be finite; use wider inference data"))
    relative=_inference_rank_tolerance(
        rank_rtol,R,rows,columns)
    # Only the right singular vectors are used.  A full factorization would
    # retain an unused rows-by-rows U matrix for tall data sets and defeat the
    # O(rows*columns + columns^2) memory preflight.
    factorization=svd(weighted_jacobian;full=false)
    singular_values=Vector{R}(factorization.S)
    all(isfinite,singular_values)||throw(ArgumentError(
        "weighted-Jacobian singular values are not finite in $R; " *
        "use wider inference data"))
    largest=isempty(singular_values) ? zero(R) : first(singular_values)
    tolerance=relative*max(largest,one(R))
    isfinite(tolerance)||throw(ArgumentError(
        "identifiability tolerance overflows $R; use wider inference data"))
    numerical_rank=count(value->value>tolerance,singular_values)
    identifiable=numerical_rank==columns
    condition_number=identifiable ?
        _inference_checked_ratio(
            largest,singular_values[columns],"Jacobian condition number") :
        R(Inf)
    fisher=_inference_checked_gram(weighted_jacobian)
    inverse_squares=zeros(R,length(singular_values))
    for index in eachindex(singular_values)
        if singular_values[index]>tolerance
            inverse_value=inv(singular_values[index])
            isfinite(inverse_value)||throw(ArgumentError(
                "inverse singular value $index overflows $R; " *
                "use wider inference data"))
            inverse_square=inverse_value*inverse_value
            !iszero(inverse_value)&&iszero(inverse_square)&&throw(ArgumentError(
                "inverse squared singular value $index underflows to zero in $R; " *
                "use wider inference data"))
            isfinite(inverse_square)||throw(ArgumentError(
                "inverse squared singular value $index overflows $R; " *
                "use wider inference data"))
            inverse_squares[index]=inverse_square
        end
    end
    pseudocovariance=Matrix{R}(
        factorization.V*Diagonal(inverse_squares)*factorization.V')
    all(isfinite,pseudocovariance)||throw(ArgumentError(
        "pseudocovariance is not finite in $R; use wider inference data"))
    covariance=identifiable ? copy(pseudocovariance) : nothing
    correlation=if covariance===nothing
        nothing
    else
        output=similar(covariance)
        for column in 1:columns,row in 1:columns
            row_variance=covariance[row,row]
            column_variance=covariance[column,column]
            row_variance>=zero(R)&&column_variance>=zero(R)||throw(
                ArgumentError(
                "covariance has a negative diagonal entry in $R; " *
                "use wider inference data"))
            denominator=sqrt(row_variance)*sqrt(column_variance)
            isfinite(denominator)||throw(ArgumentError(
                "correlation normalization overflows $R; " *
                "use wider inference data"))
            output[row,column]=iszero(denominator) ?
                (row==column ? one(R) : zero(R)) :
                _inference_checked_ratio(
                    covariance[row,column],denominator,
                    "correlation entry ($row, $column)")
        end
        all(isfinite,output)||throw(ArgumentError(
            "correlation matrix is not finite in $R; use wider inference data"))
        output
    end
    ParameterIdentifiabilityReport(
        fisher,singular_values,numerical_rank,columns,identifiable,
        condition_number,covariance,pseudocovariance,correlation,tolerance,
        problem.parameter_names,metadata)
end

"""
    parameter_identifiability(problem, parameters=problem.initial_parameters;
                              derivative_method=:auto,
                              finite_difference_step=nothing,
                              rank_rtol=nothing,
                              memory_budget=512MiB)

Evaluate the local weighted Jacobian, Fisher matrix, numerical rank,
condition number, and covariance diagnostics without optimizing parameters.
Finite differences and implicit-gradient solver metadata are retained in the
report.
"""
function parameter_identifiability(problem::LeastSquaresInferenceProblem,
        parameters=problem.initial_parameters;derivative_method=:auto,
        finite_difference_step=nothing,rank_rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    values=_inference_real_vector(
        collect(parameters),"parameter",eltype(problem.observations))
    length(values)==length(problem.initial_parameters)||throw(DimensionMismatch(
        "parameter vector has the wrong length"))
    all((problem.lower_bounds.<=values).&
        (values.<=problem.upper_bounds))||
        throw(ArgumentError("parameters must satisfy the problem bounds"))
    rows=length(problem.observations)
    columns=length(problem.initial_parameters)
    entries=8BigInt(rows)*BigInt(columns)+
            14BigInt(columns)^2+24BigInt(rows+columns)
    estimated_bytes=_performance_entries_bytes(
        entries,eltype(problem.observations))
    _require_performance_budget(
        "parameter-identifiability workspace",estimated_bytes,memory_budget;
        guidance="Reduce the number of observations/parameters or use a larger explicit budget.")
    method=_inference_derivative_method(problem,derivative_method)
    prediction,context,prediction_metadata=_inference_predict(problem,values)
    jacobian,derivative_metadata=_inference_jacobian(
        problem,values,prediction,context,method;finite_difference_step)
    weighted=_inference_weight_rows(
        jacobian,problem.standard_deviations,"weighted Jacobian")
    report=_inference_identifiability(
        problem,weighted;rank_rtol,
        metadata=(derivative_method=method,prediction_metadata,
                  derivative_metadata,estimated_peak_bytes=estimated_bytes,
                  memory_budget=isfinite(memory_budget) ?
                      BigInt(floor(BigInt,memory_budget)) : nothing,
                  exclusions=(
                      "model-builder and predictor allocations",
                      "stationary-state compilation and solver workspaces",
                      "user callback state and external data")))
    report
end

function _inference_scalar(value,label,::Type{R};
        positive::Bool=false,nonnegative::Bool=false) where R
    converted=_inference_real_vector([value],label,R)[1]
    positive&&converted<=zero(R)&&throw(ArgumentError(
        "$label must be strictly positive"))
    nonnegative&&converted<zero(R)&&throw(ArgumentError(
        "$label must be nonnegative"))
    converted
end

"""
    fit_parameters(problem; derivative_method=:auto, maxiter=50,
                   damping=1e-3, max_trials=8,
                   gradient_tolerance=1e-8,
                   step_tolerance=1e-8,
                   objective_tolerance=1e-10,
                   finite_difference_step=nothing,
                   rank_rtol=nothing,
                   memory_budget=512MiB,
                   require_convergence=false)

Run a bounded Levenberg--Marquardt least-squares fit. Every iteration has at
most `max_trials` damping attempts, and the retained dense storage is
preflighted against `memory_budget`. Predictor/model construction and solver
storage are callback dependent and are listed as resource exclusions.

`:auto` uses an explicit Jacobian callback, implicit PI stationary-state
gradients, or finite differences in that order of applicability. The actual
method, finite-difference schemes, gradient solver reports, rejected steps,
and termination reason are returned. Nonconvergence is never relabelled;
`require_convergence=true` raises after producing the same bounded search.
"""
function fit_parameters(problem::LeastSquaresInferenceProblem;
        derivative_method=:auto,maxiter::Integer=50,
        damping::Real=1e-3,max_trials::Integer=8,
        gradient_tolerance::Real=1e-8,
        step_tolerance::Real=1e-8,
        objective_tolerance::Real=1e-10,
        finite_difference_step=nothing,rank_rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        require_convergence::Bool=false)
    maxiter isa Bool&&throw(ArgumentError(
        "maxiter must be an integer, not Bool"))
    max_trials isa Bool&&throw(ArgumentError(
        "max_trials must be an integer, not Bool"))
    maxiter>0&&max_trials>0||throw(ArgumentError(
        "maxiter and max_trials must be positive"))
    BigInt(maxiter)<=typemax(Int)&&BigInt(max_trials)<=typemax(Int)||
        throw(ArgumentError(
        "maxiter and max_trials must be representable as Int"))
    iterations_limit=Int(maxiter)
    trials_limit=Int(max_trials)
    R=eltype(problem.observations)
    damping_value=_inference_scalar(
        damping,"damping",R;positive=true)
    gradient_tol=_inference_scalar(
        gradient_tolerance,"gradient_tolerance",R;nonnegative=true)
    step_tol=_inference_scalar(
        step_tolerance,"step_tolerance",R;nonnegative=true)
    objective_tol=_inference_scalar(
        objective_tolerance,"objective_tolerance",R;nonnegative=true)
    method=_inference_derivative_method(problem,derivative_method)
    rows=length(problem.observations)
    columns=length(problem.initial_parameters)
    entries=8BigInt(rows)*BigInt(columns)+
            14BigInt(columns)^2+24BigInt(rows+columns)+
            12BigInt(iterations_limit)
    estimated_bytes=_performance_entries_bytes(entries,R)
    _require_performance_budget(
        "least-squares inference workspace",estimated_bytes,memory_budget;
        guidance="Reduce the number of observations/parameters or use a larger explicit budget.")

    parameters=copy(problem.initial_parameters)
    prediction,context,prediction_metadata=
        _inference_predict(problem,parameters)
    jacobian,derivative_metadata=_inference_jacobian(
        problem,parameters,prediction,context,method;
        finite_difference_step)
    residual=_inference_residual(prediction,problem.observations)
    standardized=_inference_standardize(
        residual,problem.standard_deviations)
    objective=_inference_objective(standardized)
    history=NamedTuple[]
    converged=false
    termination=:maximum_iterations
    completed_iterations=0

    for iteration in 1:iterations_limit
        completed_iterations=iteration
        weighted_jacobian=_inference_weight_rows(
            jacobian,problem.standard_deviations,"weighted Jacobian")
        gradient=weighted_jacobian'*standardized
        all(isfinite,gradient)||throw(ArgumentError(
            "least-squares gradient overflows $R; use wider inference data"))
        projected_gradient=copy(gradient)
        active=falses(columns)
        for index in eachindex(projected_gradient)
            at_lower=parameters[index]<=problem.lower_bounds[index]
            at_upper=parameters[index]>=problem.upper_bounds[index]
            if (at_lower&&gradient[index]>zero(R))||
               (at_upper&&gradient[index]<zero(R))
                projected_gradient[index]=zero(R)
                active[index]=true
            end
        end
        gradient_norm=norm(projected_gradient,Inf)
        if gradient_norm<=gradient_tol
            converged=true
            termination=:gradient_tolerance
            push!(history,(iteration,objective,damping=damping_value,
                gradient_norm,step_norm=zero(R),accepted=true,
                trials=0,reason=:gradient_tolerance))
            break
        end
        fisher=_inference_checked_gram(weighted_jacobian)
        diagonal=R[max(fisher[index,index],one(R))
                   for index in 1:columns]
        accepted=false
        step_norm=zero(R)
        candidate_objective=objective
        accepted_trials=0
        candidate_parameters=parameters
        candidate_prediction=prediction
        candidate_context=context
        candidate_prediction_metadata=prediction_metadata
        for trial in 1:trials_limit
            regularized=Matrix{R}(fisher)
            for index in 1:columns
                regularized[index,index]+=
                    damping_value*diagonal[index]
            end
            all(isfinite,regularized)||throw(ArgumentError(
                "regularized normal equations overflow $R; " *
                "use wider inference data"))
            for index in 1:columns
                active[index]||continue
                @views fill!(regularized[index,:],zero(R))
                @views fill!(regularized[:,index],zero(R))
                regularized[index,index]=one(R)
            end
            step=try
                -(regularized\projected_gradient)
            catch error
                error isa SingularException||rethrow()
                damping_value*=R(10)
                continue
            end
            all(isfinite,step)||throw(ArgumentError(
                "least-squares step is not finite"))
            raw_candidate=parameters+step
            all(isfinite,raw_candidate)||throw(ArgumentError(
                "least-squares candidate parameters overflow $R; " *
                "use wider inference data or finite bounds"))
            candidate=clamp.(
                raw_candidate,problem.lower_bounds,problem.upper_bounds)
            projected_step=candidate-parameters
            step_norm=norm(projected_step)
            if step_norm<=step_tol*(norm(parameters)+step_tol)
                termination=:step_tolerance
                break
            end
            trial_prediction,trial_context,trial_metadata=
                _inference_predict(problem,candidate)
            trial_residual=_inference_residual(
                trial_prediction,problem.observations)
            trial_standardized=_inference_standardize(
                trial_residual,problem.standard_deviations)
            trial_objective=_inference_objective(trial_standardized)
            if trial_objective<objective
                accepted=true
                accepted_trials=trial
                candidate_parameters=candidate
                candidate_prediction=trial_prediction
                candidate_context=trial_context
                candidate_prediction_metadata=trial_metadata
                candidate_objective=trial_objective
                damping_value=max(damping_value/R(3),eps(R))
                break
            end
            damping_value*=R(10)
            isfinite(damping_value)||throw(ArgumentError(
                "least-squares damping overflowed"))
        end
        if !accepted
            push!(history,(iteration,objective,damping=damping_value,
                gradient_norm,step_norm,accepted=false,
                trials=trials_limit,reason=termination))
            termination===:step_tolerance||
                (termination=:no_acceptable_step)
            break
        end
        previous_objective=objective
        parameters=copy(candidate_parameters)
        prediction=copy(candidate_prediction)
        context=candidate_context
        prediction_metadata=candidate_prediction_metadata
        residual=_inference_residual(prediction,problem.observations)
        standardized=_inference_standardize(
            residual,problem.standard_deviations)
        objective=candidate_objective
        jacobian,derivative_metadata=_inference_jacobian(
            problem,parameters,prediction,context,method;
            finite_difference_step)
        push!(history,(iteration,objective,damping=damping_value,
            gradient_norm,step_norm,accepted=true,
            trials=accepted_trials,reason=:accepted))
        if previous_objective-objective<=
           objective_tol*max(one(R),previous_objective)
            converged=true
            termination=:objective_tolerance
            break
        end
    end

    weighted_jacobian=_inference_weight_rows(
        jacobian,problem.standard_deviations,"weighted Jacobian")
    identifiability=_inference_identifiability(
        problem,weighted_jacobian;rank_rtol,
        metadata=(derivative_method=method,prediction_metadata,
                  derivative_metadata))
    degrees_of_freedom=max(0,rows-identifiability.numerical_rank)
    reduced_chisq=if degrees_of_freedom>0
        numerator=objective+objective
        isfinite(numerator)||throw(ArgumentError(
            "reduced chi-squared numerator overflows $R; use wider inference data"))
        _inference_checked_ratio(
            numerator,R(degrees_of_freedom),"reduced chi-squared")
    else
        nothing
    end
    metadata=(
        parameter_names=problem.parameter_names,
        estimated_peak_bytes=estimated_bytes,
        memory_budget=isfinite(memory_budget) ?
            BigInt(floor(BigInt,memory_budget)) : nothing,
        degrees_of_freedom,
        reduced_chisq,
        final_damping=damping_value,
        predictor_metadata=problem.metadata,
        derivative_metadata,
        exclusions=(
            "model-builder and predictor allocations",
            "stationary-state compilation and solver workspaces",
            "user callback state and external data"))
    result=ParameterInferenceResult(
        copy(parameters),copy(prediction),copy(residual),copy(standardized),
        objective,completed_iterations,converged,termination,method,
        identifiability,history,metadata)
    require_convergence&&!converged&&throw(ArgumentError(
        "least-squares inference did not converge within the bounded search: " *
        "termination=$termination, objective=$objective"))
    result
end
