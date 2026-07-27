# Guided, compatibility-preserving front door for ordinary high-level studies.
#
# This layer deliberately delegates all numerical work to the established
# high-level commands. It owns no mutable numerical workspace, changes no
# solver default, and never materializes a source merely to explain it.

"""
    PIDiagnosticIssue

One structured, machine-readable issue reported by [`check`](@ref),
[`explain_failure`](@ref), or [`doctor`](@ref). `code` is a stable symbolic
identifier, `severity` is `:error`, `:warning`, or `:info`, and `suggestion`
and `documentation` provide an actionable next step without parsing the
human-readable `message`.
"""
struct PIDiagnosticIssue
    code::Symbol
    severity::Symbol
    message::String
    suggestion::Union{Nothing,String}
    documentation::Union{Nothing,String}
    function PIDiagnosticIssue(code::Symbol,severity::Symbol,
            message::AbstractString;
            suggestion::Union{Nothing,AbstractString}=nothing,
            documentation::Union{Nothing,AbstractString}=nothing)
        severity in (:error,:warning,:info)||throw(ArgumentError(
            "diagnostic severity must be :error, :warning, or :info"))
        isempty(String(code))&&throw(ArgumentError(
            "diagnostic code cannot be empty"))
        isempty(message)&&throw(ArgumentError(
            "diagnostic message cannot be empty"))
        new(code,severity,String(message),
            suggestion===nothing ? nothing : String(suggestion),
            documentation===nothing ? nothing : String(documentation))
    end
end

function Base.show(io::IO,issue::PIDiagnosticIssue)
    print(io,String(issue.code)," [",issue.severity,"]: ",issue.message)
end

"""
    PIStudy(source; task=:steady_state, initial_state=nothing, tspan=nothing,
            saveat=nothing, observables=nothing, algorithm=:auto,
            memory_budget=512*1024^2, validate=true, solver_options...)

Describe one guided high-level PI calculation without running it. Supported
tasks are `:steady_state`, `:dynamics`, and `:spectrum`. The study delegates to
[`stationary_state`](@ref), [`solve_dynamics`](@ref), or
[`liouvillian_spectrum`](@ref), respectively, and therefore preserves their
algorithms, precision rules, and memory safeguards.

Call [`explain`](@ref) before an expensive calculation and `solve(study)` to
obtain one [`PIStudyResult`](@ref). `validate=true` records physical
diagnostics for the returned final `PIState`; it never repairs the state.
Composite states for which no generic physical validator exists are reported
as `missing`.

Extra keywords are forwarded unchanged to the selected solver. The guided
layer owns no numerical scratch and is safe to share between tasks as long as
the objects captured by `source`, `initial_state`, and callbacks are
themselves safe to share.
"""
struct PIStudy{S,I,TS,SA,O,A,M,K}
    source::S
    task::Symbol
    initial_state::I
    tspan::TS
    saveat::SA
    observables::O
    algorithm::A
    memory_budget::M
    validate::Bool
    solver_options::K
end

function PIStudy(source;task::Symbol=:steady_state,initial_state=nothing,
        tspan=nothing,saveat=nothing,observables=nothing,algorithm=:auto,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        validate::Bool=true,kwargs...)
    task in (:steady_state,:dynamics,:spectrum)||throw(ArgumentError(
        "PIStudy task must be :steady_state, :dynamics, or :spectrum"))
    _resource_memory_budget(memory_budget)
    options=(;kwargs...)
    haskey(options,:return_info)&&throw(ArgumentError(
        "PIStudy controls return_info so that it can construct a uniform result"))
    PIStudy(source,task,initial_state,tspan,saveat,observables,algorithm,
        memory_budget,validate,options)
end

function Base.show(io::IO,study::PIStudy)
    dimension=try
        pi_dimension(study.source)
    catch
        missing
    end
    print(io,"PIStudy(task=$(study.task), dimension=$dimension, ",
        "algorithm=$(study.algorithm))")
end

"""
    PIStudyReport

Read-only explanation returned by [`check`](@ref) and [`explain`](@ref).
`recommendation` is the ordinary [`recommend_solver`](@ref) report when
preflight succeeded. `runnable` is `true`, `false`, or `missing`: `missing`
means no known error was found but the resource estimate contains an unknown
component, never that execution was certified safe.
"""
struct PIStudyReport{S,R}
    task::Symbol
    dimension::Union{Nothing,Int}
    autonomous::Union{Missing,Bool}
    source_summary::S
    recommendation::R
    issues::Vector{PIDiagnosticIssue}
    runnable::Union{Missing,Bool}
end

function Base.show(io::IO,report::PIStudyReport)
    errors=count(issue->issue.severity===:error,report.issues)
    warnings=count(issue->issue.severity===:warning,report.issues)
    route=report.recommendation===nothing ? "unavailable" :
        "$(report.recommendation.backend)/$(report.recommendation.algorithm)"
    print(io,"PIStudyReport(task=$(report.task), dimension=$(report.dimension), ",
        "route=$route, runnable=$(report.runnable), errors=$errors, ",
        "warnings=$warnings)")
end

function Base.show(io::IO,::MIME"text/plain",report::PIStudyReport)
    println(io,"PI study explanation")
    println(io,"  task:       ",report.task)
    println(io,"  dimension:  ",report.dimension)
    println(io,"  autonomous: ",report.autonomous)
    println(io,"  runnable:   ",report.runnable)
    if report.recommendation!==nothing
        recommendation=report.recommendation
        println(io,"  route:      ",recommendation.backend," / ",
            recommendation.algorithm)
        println(io,"  reason:     ",recommendation.reason)
        println(io,"  peak bytes: ",recommendation.known_peak_bytes,
            " (",recommendation.peak_provenance,")")
    end
    if isempty(report.issues)
        print(io,"  issues:     none")
    else
        println(io,"  issues:")
        for issue in report.issues
            println(io,"    - ",issue)
            issue.suggestion===nothing||
                println(io,"      suggestion: ",issue.suggestion)
            issue.documentation===nothing||
                println(io,"      docs: ",issue.documentation)
        end
    end
end

@inline function _study_issue(code,severity,message;
        suggestion=nothing,documentation=nothing)
    PIDiagnosticIssue(Symbol(code),severity,message;
        suggestion,documentation)
end

"""
    explain_failure(error)

Convert an exception from a guided or ordinary high-level workflow into one
structured [`PIDiagnosticIssue`](@ref). This helper never suppresses or
rewrites the original exception; call it from a `catch` block when a
machine-readable explanation is useful.
"""
function explain_failure(error::Exception)
    message=sprint(showerror,error)
    lower=lowercase(message)
    if occursin("memory_budget",lower)||occursin("memory budget",lower)
        return _study_issue(
            "PID-E-MEMORY-BUDGET",:error,message;
            suggestion="Use recommend_solver to choose a bounded route, reduce retained output, or explicitly revise memory_budget.",
            documentation="api/solvers.md")
    elseif occursin("autonomous",lower)||occursin("time-dependent",lower)
        return _study_issue(
            "PID-E-TIME-DEPENDENT-STATIONARY",:error,message;
            suggestion="Freeze the generator at an explicit time or select a dynamics task.",
            documentation="api/solvers.md")
    elseif occursin("nonnegative",lower)&&
            (occursin("rate",lower)||occursin("trajectory",lower))
        return _study_issue(
            "PID-E-STOCHASTIC-RATE",:error,message;
            suggestion="Use finite real nonnegative rates for stochastic simulation.",
            documentation="api/dynamics.md")
    elseif error isa DimensionMismatch
        return _study_issue(
            "PID-E-DIMENSION-MISMATCH",:error,message;
            suggestion="Check basis ownership, local dimensions, and state/operator shapes.",
            documentation="framework.md")
    elseif error isa ArgumentError
        return _study_issue(
            "PID-E-INVALID-REQUEST",:error,message;
            suggestion="Inspect explain(study) and correct the reported input.",
            documentation="getting_started.md")
    end
    _study_issue(
        "PID-E-UNEXPECTED",:error,message;
        suggestion="Run doctor(), retain the original stack trace, and report a minimal reproducer.",
        documentation="CONTRIBUTING.md")
end

function _study_tspan_issue(tspan)
    tspan===nothing&&return _study_issue(
        "PID-E-MISSING-TSPAN",:error,
        "dynamics requires an explicit two-endpoint tspan";
        suggestion="Pass tspan=(t0, t1) with finite ordered endpoints.",
        documentation="getting_started.md")
    try
        length(tspan)==2||return _study_issue(
            "PID-E-INVALID-TSPAN",:error,
            "tspan must contain exactly two endpoints";
            suggestion="Use tspan=(t0, t1).",
            documentation="getting_started.md")
        t0,t1=tspan
        t0 isa Real&&t1 isa Real&&
            !(t0 isa Bool)&&!(t1 isa Bool)&&
            isfinite(t0)&&isfinite(t1)&&t1>=t0||return _study_issue(
                "PID-E-INVALID-TSPAN",:error,
                "tspan endpoints must be finite real numbers with t1 >= t0";
                suggestion="Correct or reorder the requested time interval.",
                documentation="getting_started.md")
    catch
        return _study_issue(
            "PID-E-INVALID-TSPAN",:error,
            "tspan must be an iterable pair of finite ordered real numbers";
            suggestion="Use tspan=(t0, t1).",
            documentation="getting_started.md")
    end
    nothing
end

function _study_saved_count(tspan,saveat)
    saveat===nothing&&return (2,
        promote_type(typeof(float(first(tspan))),typeof(float(last(tspan)))))
    if saveat isa Real
        saveat isa Bool&&throw(ArgumentError(
            "saveat must be a positive real number or an explicit time grid"))
        isfinite(saveat)&&saveat>0||throw(ArgumentError(
            "saveat must be finite and positive"))
        t0,t1=tspan
        grid=float(t0):float(saveat):float(t1)
        count=length(grid)
        append_endpoint=isempty(grid)||last(grid)<float(t1)
        return (Base.checked_add(count,append_endpoint ? 1 : 0),
            eltype(grid))
    end
    applicable(length,saveat)||throw(ArgumentError(
        "explicit saveat must have a finite length"))
    count=length(saveat)
    count>0||throw(ArgumentError("saveat cannot be empty"))
    T=try
        eltype(saveat)
    catch
        Float64
    end
    T isa Type&&T<:Real ? (count,_real_float_type(T)) : (count,Float64)
end

function _study_saveat_issue(tspan,saveat)
    saveat===nothing&&return nothing
    if saveat isa Real
        !(saveat isa Bool)&&isfinite(saveat)&&saveat>0&&return nothing
        return _study_issue(
            "PID-E-INVALID-SAVEAT",:error,
            "scalar saveat must be a finite positive real number";
            suggestion="Pass a positive sampling interval or an explicit ordered time grid.",
            documentation="getting_started.md")
    end
    applicable(length,saveat)||return _study_issue(
        "PID-E-INVALID-SAVEAT",:error,
        "explicit saveat must be a finite collection";
        suggestion="Pass a finite vector or range containing both tspan endpoints.",
        documentation="getting_started.md")
    count=try
        length(saveat)
    catch
        return _study_issue(
            "PID-E-INVALID-SAVEAT",:error,
            "explicit saveat must have a finite length";
            suggestion="Pass a finite vector or range containing both tspan endpoints.",
            documentation="getting_started.md")
    end
    count>0||return _study_issue(
        "PID-E-INVALID-SAVEAT",:error,
        "explicit saveat cannot be empty";
        suggestion="Include both endpoints of tspan.",
        documentation="getting_started.md")
    t0,t1=tspan
    first_time=nothing
    previous=nothing
    observed=0
    try
        for value in saveat
            value isa Real&&!(value isa Bool)&&isfinite(value)||
                return _study_issue(
                    "PID-E-INVALID-SAVEAT",:error,
                    "explicit saveat entries must be finite real numbers";
                    suggestion="Remove nonnumeric, Boolean, Inf, and NaN entries.",
                    documentation="getting_started.md")
            observed+=1
            first_time===nothing&&(first_time=value)
            previous===nothing||value>=previous||return _study_issue(
                "PID-E-INVALID-SAVEAT",:error,
                "explicit saveat times must be nondecreasing";
                suggestion="Sort the time grid without removing the tspan endpoints.",
                documentation="getting_started.md")
            previous=value
        end
    catch
        return _study_issue(
            "PID-E-INVALID-SAVEAT",:error,
            "explicit saveat must be an iterable collection of finite real times";
            suggestion="Pass a finite vector or range containing both tspan endpoints.",
            documentation="getting_started.md")
    end
    observed==count||return _study_issue(
        "PID-E-INVALID-SAVEAT",:error,
        "explicit saveat length changed while it was inspected";
        suggestion="Use an immutable or otherwise stable time collection.",
        documentation="getting_started.md")
    first_time==t0&&previous==t1||return _study_issue(
        "PID-E-INVALID-SAVEAT",:error,
        "explicit saveat times must include both endpoints of tspan";
        suggestion="Set the first saved time to tspan[1] and the last to tspan[2].",
        documentation="getting_started.md")
    nothing
end

function _study_source_summary(source,dimension,autonomous)
    if source isa PIModel
        return merge(model_summary(source),(;source_type=typeof(source)))
    end
    (;source_type=typeof(source),dimension,autonomous,
      retained_bytes=Base.summarysize(source))
end

function _study_algorithm_symbol(study::PIStudy)
    if study.task===:steady_state
        return first(_algorithm_options(study.algorithm))
    elseif study.task===:dynamics
        return first(_dynamics_algorithm_options(study.algorithm))
    end
    algorithm=study.algorithm
    if algorithm isa AutoAlgorithm||algorithm===:auto
        return get(study.solver_options,:target,:largest_real)===:near_zero ?
            :harmonic : :auto
    end
    algorithm isa HarmonicArnoldiAlgorithm&&return :harmonic
    algorithm isa Symbol&&return algorithm
    throw(ArgumentError(
        "spectrum studies require a Symbol, AutoAlgorithm(), or HarmonicArnoldiAlgorithm()"))
end

function _study_recommendation(study::PIStudy)
    options=study.solver_options
    algorithm=_study_algorithm_symbol(study)
    if study.task===:steady_state
        krylovdim=get(options,:krylovdim,
            study.algorithm isa Union{GMRESAlgorithm,RecycledGMRESAlgorithm} ?
                study.algorithm.krylovdim : 30)
        recycle_dim=get(options,:recycle_dim,
            study.algorithm isa RecycledGMRESAlgorithm ?
                study.algorithm.recycle_dim : 0)
        return recommend_solver(
            study.source;task=:steady_state,algorithm,
            memory_budget=study.memory_budget,krylovdim,recycle_dim,
            T=_resource_scalar_type(study.source,study.initial_state))
    elseif study.task===:spectrum
        requested=study.algorithm isa HarmonicArnoldiAlgorithm ?
            study.algorithm.nev : get(options,:nev,6)
        requested isa Integer&&!(requested isa Bool)&&requested>0||
            throw(ArgumentError("nev must be a positive integer"))
        dimension=pi_dimension(study.source)
        nev=Int(min(BigInt(dimension),BigInt(requested)))
        krylovdim=study.algorithm isa HarmonicArnoldiAlgorithm ?
            study.algorithm.krylovdim :
            get(options,:krylovdim,max(20,2nev+4))
        return recommend_solver(
            study.source;task=:spectrum,algorithm,
            memory_budget=study.memory_budget,krylovdim,nev,
            block_size=get(options,:block_size,min(nev,4)),
            maxrestarts=get(options,:maxrestarts,20),
            vectors=get(options,:vectors,false),
            T=_resource_scalar_type(study.source))
    end
    samples,time_type=_study_saved_count(study.tspan,study.saveat)
    save_states=get(options,:save_states,true)
    save_states isa Bool||throw(ArgumentError(
        "save_states must be true or false"))
    observable_series=study.observables===nothing ? 0 :
        length(_named_observables(study.observables))
    algorithm_options=last(_dynamics_algorithm_options(study.algorithm))
    recommend_solver(
        study.source;task=:dynamics,algorithm,
        memory_budget=study.memory_budget,
        krylovdim=get(algorithm_options,:krylovdim,
            get(options,:krylovdim,30)),
        T=_resource_scalar_type(study.source,study.initial_state),
        time_type,samples,
        saved_states=save_states ? samples : 0,
        observable_series)
end

"""
    check(study::PIStudy)
    check(source; task=:steady_state, kwargs...)

Validate the requested guided workflow and return a [`PIStudyReport`](@ref)
without solving it. The check uses [`recommend_solver`](@ref) for resource
selection and does not materialize a Liouvillian. Negative deterministic
rates are reported as a warning because they can describe time-local
non-CP-divisible dynamics; they are not silently rejected.
"""
function check(study::PIStudy)
    issues=PIDiagnosticIssue[]
    dimension=nothing
    autonomous=missing
    try
        dimension=pi_dimension(study.source)
        dimension>0||push!(issues,_study_issue(
            "PID-E-EMPTY-SOURCE",:error,
            "the source has zero retained coordinates";
            suggestion="Construct a nonempty PI basis or operator source.",
            documentation="framework.md"))
    catch error
        push!(issues,explain_failure(error))
    end
    if applicable(isautonomous,study.source)
        try
            autonomous=isautonomous(study.source)
        catch error
            push!(issues,explain_failure(error))
        end
    end
    if study.task===:dynamics
        study.initial_state===nothing&&push!(issues,_study_issue(
            "PID-E-MISSING-INITIAL-STATE",:error,
            "dynamics requires initial_state";
            suggestion="Pass a PIState compatible with the source basis.",
            documentation="getting_started.md"))
        study.initial_state===nothing||study.initial_state isa PIState||
            push!(issues,_study_issue(
                "PID-E-UNSUPPORTED-INITIAL-STATE",:error,
                "guided deterministic dynamics currently requires a PIState initial_state";
                suggestion="Use the dedicated composite or hierarchy evolution API for this state type.",
                documentation="api/dynamics.md"))
        tspan_issue=_study_tspan_issue(study.tspan)
        tspan_issue===nothing||push!(issues,tspan_issue)
        if tspan_issue===nothing
            saveat_issue=_study_saveat_issue(study.tspan,study.saveat)
            saveat_issue===nothing||push!(issues,saveat_issue)
        end
    else
        study.tspan===nothing||push!(issues,_study_issue(
            "PID-W-UNUSED-TSPAN",:warning,
            "tspan is ignored for stationary and spectral tasks";
            suggestion="Remove tspan or select task=:dynamics.",
            documentation="getting_started.md"))
        study.saveat===nothing||push!(issues,_study_issue(
            "PID-W-UNUSED-SAVEAT",:warning,
            "saveat is ignored for stationary and spectral tasks";
            suggestion="Remove saveat or select task=:dynamics.",
            documentation="getting_started.md"))
    end
    study.task===:spectrum&&study.observables!==nothing&&push!(issues,
        _study_issue(
            "PID-E-UNSUPPORTED-OBSERVABLES",:error,
            "a selected-spectrum study does not evaluate observables";
            suggestion="Remove observables or use a stationary/dynamics study.",
            documentation="api/solvers.md"))
    if study.task in (:steady_state,:spectrum)&&autonomous===false
        push!(issues,_study_issue(
            "PID-E-TIME-DEPENDENT-STATIONARY",:error,
            "$(study.task) requires an autonomous source";
            suggestion="Freeze the model at an explicit time or solve its dynamics.",
            documentation="api/solvers.md"))
    end
    if study.source isa PIModel&&any(
            term->term_rate(term) isa Number&&term_rate(term)<0,
            study.source.terms)
        push!(issues,_study_issue(
            "PID-W-NEGATIVE-DETERMINISTIC-RATE",:warning,
            "the model contains a negative constant rate; deterministic evolution is allowed but complete positivity is not guaranteed";
            suggestion="Confirm that a time-local non-CP-divisible generator is intended.",
            documentation="framework.md"))
    end
    source_basis=try
        _operator_basis(study.source)
    catch
        nothing
    end
    if study.initial_state isa PIState&&source_basis isa PIBasis&&
            study.initial_state.basis!==source_basis
        push!(issues,_study_issue(
            "PID-E-BASIS-MISMATCH",:error,
            "initial_state belongs to a different PIBasis object than the source";
            suggestion="Construct the state from the exact basis used by the model.",
            documentation="framework.md"))
    end
    if study.task===:steady_state&&study.observables!==nothing&&
            study.source isa GlobalPseudomodeModel
        push!(issues,_study_issue(
            "PID-E-UNSUPPORTED-OBSERVABLES",:error,
            "guided stationary observables are not inferred on a shared-pseudomode composite state";
            suggestion="Solve without observables, reduce the result explicitly to the desired factor, and evaluate the observable there.",
            documentation="global_pseudomodes.md"))
    end
    recommendation=nothing
    if !any(issue->issue.severity===:error,issues)
        try
            recommendation=_study_recommendation(study)
            if recommendation.safe_to_run===false
                push!(issues,_study_issue(
                    "PID-E-MEMORY-BUDGET",:error,
                    "the selected route exceeds the requested memory budget";
                    suggestion="Reduce output/Krylov dimensions, choose a bounded route, or deliberately revise memory_budget.",
                    documentation="api/solvers.md"))
            elseif recommendation.safe_to_run===missing
                push!(issues,_study_issue(
                    "PID-W-UNKNOWN-MEMORY",:warning,
                    "the resource preflight contains allocations that cannot be bounded from the supplied source";
                    suggestion="Prepare the source explicitly and benchmark a representative smaller calculation.",
                    documentation="architecture.md"))
            end
        catch error
            push!(issues,explain_failure(error))
        end
    end
    has_error=any(issue->issue.severity===:error,issues)
    runnable=has_error ? false :
        recommendation===nothing ? missing :
        recommendation.safe_to_run
    summary=_study_source_summary(
        study.source,dimension,autonomous)
    PIStudyReport(study.task,dimension,autonomous,summary,recommendation,
        issues,runnable)
end

check(source;kwargs...)=check(PIStudy(source;kwargs...))

"""
    explain(study::PIStudy)
    explain(source; task=:steady_state, kwargs...)

Return the same structured preflight as [`check`](@ref), with a detailed
plain-text display intended for notebooks and the REPL.
"""
explain(study::PIStudy)=check(study)
explain(source;kwargs...)=explain(PIStudy(source;kwargs...))

"""
    PIStudyResult

Uniform result of `solve(::PIStudy)`. The original high-level result is
retained in `raw`. Common task-independent fields are:

- `state`: final state when one is available;
- `states`: saved state history or `nothing`;
- `observables`: named observable output or an empty named tuple;
- `values`: selected spectral values or `nothing`;
- `converged`: `Bool` or `missing` when no convergence study was requested;
- `residual`, `selected_algorithm`, `stats`, and `diagnostics`.

Use the corresponding `result_*` accessors when code must also accept legacy
high-level result objects. Accessors return `nothing` when a result does not
contain that category of output; an explicitly retained empty collection is
returned unchanged.
"""
struct PIStudyResult{R,S,H,O,V,C,E,A,ST,D}
    task::Symbol
    raw::R
    state::S
    states::H
    observables::O
    values::V
    converged::C
    residual::E
    selected_algorithm::A
    stats::ST
    diagnostics::D
end

function Base.show(io::IO,result::PIStudyResult)
    print(io,"PIStudyResult(task=$(result.task), ",
        "converged=$(result.converged), ",
        "algorithm=$(result.selected_algorithm))")
end

"""Return the final state represented by a package result, or `nothing`."""
function result_state(result)
    hasproperty(result,:state)&&return getproperty(result,:state)
    if hasproperty(result,:states)
        states=getproperty(result,:states)
        (states===nothing||isempty(states))&&return nothing
        return last(states)
    end
    hasproperty(result,:solution)&&
        return result_state(getproperty(result,:solution))
    nothing
end
result_state(state::Union{PIState,CompositePIState})=state

"""
    result_final_state(result)

Explicit alias of [`result_state`](@ref), useful in code that also calls
[`result_states`](@ref) and wants the distinction between one final state and
a saved history to be visible.
"""
result_final_state(result)=result_state(result)

"""Return the saved physical-time grid retained by a result, or `nothing`."""
function result_times(result)
    hasproperty(result,:times)&&return getproperty(result,:times)
    hasproperty(result,:raw)&&return result_times(getproperty(result,:raw))
    hasproperty(result,:solution)&&return result_times(getproperty(result,:solution))
    nothing
end

"""
    result_states(result)

Return a saved state history exactly as retained by `result`, or `nothing`.
This accessor does not wrap a stationary state in a one-element vector and
does not reconstruct state-free streaming output.
"""
function result_states(result)
    hasproperty(result,:states)&&return getproperty(result,:states)
    hasproperty(result,:raw)&&return result_states(getproperty(result,:raw))
    hasproperty(result,:solution)&&
        return result_states(getproperty(result,:solution))
    nothing
end

"""
    result_values(result)

Return selected spectral or other explicitly stored result values, or
`nothing`. The accessor never computes missing eigenvalues or observables.
"""
function result_values(result)
    hasproperty(result,:values)&&return getproperty(result,:values)
    hasproperty(result,:raw)&&return result_values(getproperty(result,:raw))
    hasproperty(result,:solution)&&
        return result_values(getproperty(result,:solution))
    nothing
end

"""Return named observable output from a package result, or an empty named tuple."""
function result_observables(result)
    hasproperty(result,:observables)&&return getproperty(result,:observables)
    hasproperty(result,:solution)&&
        return result_observables(getproperty(result,:solution))
    NamedTuple()
end

function _result_bool_status(value)
    value===missing&&return missing
    value===nothing&&return missing
    value isa Bool&&return value
    if value isa AbstractArray||value isa Tuple
        isempty(value)&&return missing
        all(entry->entry isa Bool,value)||return missing
        return all(value)
    end
    missing
end

"""Return reported solver convergence as `Bool` or `missing`."""
function result_converged(result)
    if hasproperty(result,:converged)
        return _result_bool_status(getproperty(result,:converged))
    elseif hasproperty(result,:info)
        return result_converged(getproperty(result,:info))
    elseif hasproperty(result,:report)
        report=getproperty(result,:report)
        hasproperty(report,:solver_converged)&&
            return _result_bool_status(report.solver_converged)
    elseif hasproperty(result,:solution)
        return result_converged(getproperty(result,:solution))
    end
    missing
end

"""Return a reported residual (scalar or collection), or `missing`."""
function result_residual(result)
    hasproperty(result,:residual)&&return getproperty(result,:residual)
    hasproperty(result,:residuals)&&return getproperty(result,:residuals)
    hasproperty(result,:info)&&return result_residual(getproperty(result,:info))
    hasproperty(result,:solution)&&
        return result_residual(getproperty(result,:solution))
    missing
end

"""Return the algorithm selected by a package result, or `missing`."""
function result_selected_algorithm(result)
    if hasproperty(result,:selected_algorithm)
        return getproperty(result,:selected_algorithm)
    elseif hasproperty(result,:info)
        selected=result_selected_algorithm(getproperty(result,:info))
        selected===missing||return selected
    end
    hasproperty(result,:algorithm)&&return getproperty(result,:algorithm)
    hasproperty(result,:solution)&&
        return result_selected_algorithm(getproperty(result,:solution))
    missing
end

"""Return task-specific solver statistics or metadata, or an empty named tuple."""
function result_stats(result)
    hasproperty(result,:stats)&&return getproperty(result,:stats)
    hasproperty(result,:info)&&return getproperty(result,:info)
    hasproperty(result,:metadata)&&return getproperty(result,:metadata)
    hasproperty(result,:report)&&return getproperty(result,:report)
    hasproperty(result,:solution)&&return result_stats(getproperty(result,:solution))
    if hasproperty(result,:times)
        times=getproperty(result,:times)
        states=hasproperty(result,:states) ? getproperty(result,:states) : nothing
        observables=hasproperty(result,:observables) ?
            getproperty(result,:observables) : nothing
        return (
            algorithm=result_selected_algorithm(result),
            samples=length(times),
            saved_states=states===nothing ? 0 : length(states),
            observable_count=observables===nothing ? 0 : length(observables),
        )
    end
    NamedTuple()
end

"""Return structured diagnostics retained by a result, or an empty named tuple."""
function result_diagnostics(result)
    hasproperty(result,:diagnostics)&&return getproperty(result,:diagnostics)
    hasproperty(result,:info)&&return getproperty(result,:info)
    hasproperty(result,:report)&&return getproperty(result,:report)
    hasproperty(result,:solution)&&
        return result_diagnostics(getproperty(result,:solution))
    NamedTuple()
end

function _study_final_physical(state,validate::Bool)
    validate||return missing
    state isa PIState||return missing
    state_diagnostics(state)
end

function _study_stationary_observables(state,observables)
    observables===nothing&&return NamedTuple()
    state isa PIState||throw(ArgumentError(
        "guided stationary observables currently require a PIState output; " *
        "reduce a composite state to its PI factor explicitly first"))
    operations=_prepare_streaming_observables(
        state.basis,observables;require_hermitian=false)
    Dict(name=>dot(operator.data,state.data)
         for (name,operator) in operations)
end

function _study_blocking_error(report::PIStudyReport)
    blocking=filter(issue->issue.severity===:error,report.issues)
    isempty(blocking)&&return nothing
    join(("$(String(issue.code)): $(issue.message)" for issue in blocking),"; ")
end

"""
    solve(study::PIStudy)

Run a checked guided study through the existing high-level solver and return a
uniform [`PIStudyResult`](@ref). This is a method of `SciMLBase.solve`; it does
not replace [`stationary_state`](@ref), [`solve_dynamics`](@ref), or
[`liouvillian_spectrum`](@ref), and the original result remains available as
`result.raw`.
"""
function solve(study::PIStudy)
    report=check(study)
    problem=_study_blocking_error(report)
    problem===nothing||throw(ArgumentError(
        "PIStudy is not runnable: $problem"))
    options=study.solver_options
    if study.task===:steady_state
        solve_options=study.initial_state===nothing ? options :
            merge(options,(;initial_state=study.initial_state))
        raw=stationary_state(
            study.source;algorithm=study.algorithm,
            memory_budget=study.memory_budget,return_info=true,
            solve_options...)
        state=result_state(raw)
        observables=_study_stationary_observables(
            state,study.observables)
        physical=_study_final_physical(state,study.validate)
        stats=result_stats(raw)
        diagnostics=(preflight=report,solver=stats,physical)
        return PIStudyResult(
            :steady_state,raw,state,nothing,observables,nothing,
            result_converged(raw),result_residual(raw),
            result_selected_algorithm(raw),stats,diagnostics)
    elseif study.task===:dynamics
        raw=solve_dynamics(
            study.source,study.initial_state,study.tspan;
            algorithm=study.algorithm,saveat=study.saveat,
            observables=study.observables,
            memory_budget=study.memory_budget,options...)
        states=hasproperty(raw,:states) ? raw.states : nothing
        state=result_state(raw)
        physical=_study_final_physical(state,study.validate)
        stats=result_stats(raw)
        diagnostics=(preflight=report,solver=stats,physical)
        return PIStudyResult(
            :dynamics,raw,state,states,result_observables(raw),nothing,
            result_converged(raw),result_residual(raw),
            result_selected_algorithm(raw),stats,diagnostics)
    end
    raw=liouvillian_spectrum(
        study.source;algorithm=study.algorithm,
        memory_budget=study.memory_budget,return_info=true,options...)
    stats=result_stats(raw)
    diagnostics=(preflight=report,solver=stats,physical=missing)
    PIStudyResult(
        :spectrum,raw,nothing,nothing,NamedTuple(),raw.values,
        result_converged(raw),result_residual(raw),
        result_selected_algorithm(raw),stats,diagnostics)
end

"""
    PIDoctorReport

Environment and smoke-test report returned by [`doctor`](@ref). Optional
extensions are reported as active or inactive; inactive weak dependencies are
normal and do not make the report unhealthy.
"""
struct PIDoctorReport{E,X,S}
    healthy::Bool
    environment::E
    extensions::X
    smoke_test::S
    issues::Vector{PIDiagnosticIssue}
end

function Base.show(io::IO,report::PIDoctorReport)
    print(io,"PIDoctorReport(healthy=$(report.healthy), ",
        "Julia=$(report.environment.julia_version), ",
        "threads=$(report.environment.threads), ",
        "smoke_test=$(report.smoke_test.passed))")
end

function Base.show(io::IO,::MIME"text/plain",report::PIDoctorReport)
    println(io,"PermutationalInvariantDynamics doctor")
    println(io,"  healthy:        ",report.healthy)
    println(io,"  package:        ",report.environment.package_version)
    println(io,"  Julia:          ",report.environment.julia_version)
    println(io,"  platform:       ",report.environment.kernel," / ",
        report.environment.architecture)
    println(io,"  threads:        ",report.environment.threads)
    println(io,"  BLAS threads:   ",report.environment.blas_threads)
    println(io,"  active project: ",report.environment.active_project)
    println(io,"  smoke test:     ",report.smoke_test.passed)
    active=String[]
    for name in propertynames(report.extensions)
        getproperty(report.extensions,name)&&push!(active,String(name))
    end
    println(io,"  extensions:     ",
        isempty(active) ? "none active" : join(active,", "))
    if isempty(report.issues)
        print(io,"  issues:         none")
    else
        println(io,"  issues:")
        for issue in report.issues
            println(io,"    - ",issue)
        end
    end
end

function _doctor_smoke_test()
    basis=PIBasis(1,2)
    spin=spin_matrices(2)
    model=PIModel(basis,(LocalJump(spin.jm;rate=1.0),))
    result=stationary_state(
        model;algorithm=DirectAlgorithm(),return_info=true,
        memory_budget=16*1024^2)
    physical=state_diagnostics(result.state)
    passed=result.info.converged&&physical.valid
    (;ran=true,passed,residual=result.info.residual,
      trace_error=result.info.trace_error,error=nothing)
end

"""
    doctor(; smoke_test=true)

Return a dependency-free environment report containing the package and Julia
versions, active project, thread and BLAS configuration, optional-extension
activation, and (by default) a tiny one-qubit stationary-state smoke solve.
The command performs no network access and never loads an optional
dependency. Set `smoke_test=false` when only environment metadata is wanted.
"""
function doctor(;smoke_test::Bool=true)
    issues=PIDiagnosticIssue[]
    smoke=if smoke_test
        try
            _doctor_smoke_test()
        catch error
            issue=explain_failure(error)
            push!(issues,_study_issue(
                "PID-E-SMOKE-TEST",:error,
                "the built-in one-qubit smoke solve failed: $(issue.message)";
                suggestion=issue.suggestion,
                documentation=issue.documentation))
            (;ran=true,passed=false,residual=missing,
              trace_error=missing,error=sprint(showerror,error))
        end
    else
        (;ran=false,passed=missing,residual=missing,
          trace_error=missing,error=nothing)
    end
    active_project=try
        Base.active_project()
    catch
        nothing
    end
    blas_threads=try
        BLAS.get_num_threads()
    catch
        missing
    end
    environment=(
        package_version=Base.pkgversion(@__MODULE__),
        julia_version=VERSION,
        active_project,
        threads=Threads.nthreads(),
        thread_pools=Threads.nthreadpools(),
        kernel=Sys.KERNEL,
        architecture=Sys.ARCH,
        word_size=Sys.WORD_SIZE,
        blas_threads,
        blas_config=sprint(show,BLAS.get_config()),
    )
    extensions=(
        Clarabel=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsClarabelExt)!==nothing,
        Distributed=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsDistributedExt)!==nothing,
        HDF5=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsHDF5Ext)!==nothing,
        JLD2=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsJLD2Ext)!==nothing,
        Makie=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsMakieExt)!==nothing,
        QuantumCumulants=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsQuantumCumulantsExt)!==nothing,
        QuantumOptics=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsQuantumOpticsExt)!==nothing,
        QuantumToolbox=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsQuantumToolboxExt)!==nothing,
        Tables=Base.get_extension(
            @__MODULE__,:PermutationalInvariantDynamicsTablesExt)!==nothing,
    )
    healthy=isempty(issues)&&(!smoke_test||smoke.passed===true)
    PIDoctorReport(healthy,environment,extensions,smoke,issues)
end
