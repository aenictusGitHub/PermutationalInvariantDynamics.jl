"""
    ConvergenceStudyResult

Complete record of a numerical refinement study. `refinements`, `results`,
`estimates`, and `diagnostics` retain every evaluated level. The first entry
of `pairwise_errors`, `tolerances`, and `pairwise_converged` is `missing`
because no coarser comparison exists. Rates are also `missing` until three
estimates and a refinement scale are available. `solver_converged` is
`missing` when an evaluator did not report an inner solver status.

`converged` refers to the final requested refinement window, never merely to
an earlier pair or to an inner solver flag. Use [`convergence_estimate`](@ref)
to obtain the final estimate with an explicit convergence guard.
"""
struct ConvergenceStudyResult{L,O,E,D,T,M}
    parameter::Symbol
    refinements::L
    results::O
    estimates::E
    diagnostics::D
    pairwise_errors::Vector{Union{Missing,T}}
    tolerances::Vector{Union{Missing,T}}
    pairwise_converged::Vector{Union{Missing,Bool}}
    observed_rates::Vector{Union{Missing,T}}
    solver_converged::Vector{Union{Missing,Bool}}
    converged::Bool
    first_passing_index::Union{Nothing,Int}
    consecutive_required::Int
    reason::Symbol
    metadata::M
end

Base.length(report::ConvergenceStudyResult)=length(report.refinements)
Base.firstindex(::ConvergenceStudyResult)=1
Base.lastindex(report::ConvergenceStudyResult)=length(report)
Base.getindex(report::ConvergenceStudyResult,index::Integer)=report.results[index]
Base.iterate(report::ConvergenceStudyResult,args...)=iterate(report.results,args...)

function Base.show(io::IO,report::ConvergenceStudyResult)
    print(io,"ConvergenceStudyResult(parameter=",report.parameter,
        ", levels=",length(report),", converged=",report.converged,
        ", reason=",report.reason)
    isempty(report.pairwise_errors)||print(io,", final_error=",
        last(report.pairwise_errors))
    print(io,")")
end

function _convergence_default_estimate(result)
    for name in (:estimate,:value,:solution,:state,:values,:solutions)
        hasproperty(result,name)&&return getproperty(result,name)
    end
    result
end

function _convergence_default_diagnostics(result)
    hasproperty(result,:info)&&return getproperty(result,:info)
    result isa NamedTuple&&return result
    nothing
end

function _convergence_status(result)
    if hasproperty(result,:converged)
        status=getproperty(result,:converged)
        (status===nothing||status===missing)&&return missing
        status isa Bool&&return status
        try
            return all(status)
        catch
            throw(ArgumentError(
                "reported solver convergence must be Bool or an iterable of Bool"))
        end
    end
    if hasproperty(result,:info)
        return _convergence_status(getproperty(result,:info))
    end
    missing
end

@inline _convergence_payload(value::Union{PIState,PIOperator})=value.data
@inline function _convergence_payload(value)
    if hasproperty(value,:data)
        data=getproperty(value,:data)
        data isa AbstractArray&&return data
    end
    value
end

function _convergence_array_distance(a::AbstractArray,b::AbstractArray)
    axes(a)==axes(b)||throw(DimensionMismatch(
        "successive convergence estimates have different axes"))
    R=promote_type(_real_float_type(eltype(a)),_real_float_type(eltype(b)))
    scale=zero(R);sum_squares=one(R)
    @inbounds for index in eachindex(a,b)
        magnitude=R(abs(a[index]-b[index]))
        isfinite(magnitude)||throw(ArgumentError(
            "successive convergence estimates contain a nonfinite difference"))
        iszero(magnitude)&&continue
        if scale<magnitude
            ratio=scale/magnitude
            sum_squares=one(R)+sum_squares*ratio*ratio
            scale=magnitude
        else
            ratio=magnitude/scale
            sum_squares+=ratio*ratio
        end
    end
    iszero(scale) ? zero(R) : scale*sqrt(sum_squares)
end

function _convergence_default_distance(left::PIState,right::PIState)
    left.basis===right.basis||throw(ArgumentError(
        "successive PI states use different PI bases"))
    _convergence_array_distance(left.data,right.data)
end

function _convergence_default_distance(left::PIOperator,right::PIOperator)
    left.basis===right.basis||throw(ArgumentError(
        "successive PI operators use different PI bases"))
    _convergence_array_distance(left.data,right.data)
end

function _convergence_default_distance(left::HEOMState,right::HEOMState)
    left.plan===right.plan||throw(ArgumentError(
        "successive HEOM states use different HEOM plans"))
    _convergence_array_distance(left.data,right.data)
end

function _convergence_default_distance(left,right)
    (left isa AbstractPIOperator||right isa AbstractPIOperator)&&
        throw(ArgumentError(
            "successive convergence estimates mix incompatible PI state/operator representations"))
    (left isa HEOMState||right isa HEOMState)&&throw(ArgumentError(
        "successive convergence estimates mix incompatible HEOM representations"))
    a=_convergence_payload(left);b=_convergence_payload(right)
    if a isa Number&&b isa Number
        return abs(a-b)
    end
    if a isa AbstractArray&&b isa AbstractArray
        return _convergence_array_distance(a,b)
    end
    try
        norm(a-b)
    catch error
        error isa MethodError||rethrow()
        throw(ArgumentError(
            "successive estimates do not define a normed difference; pass distance=(a,b)->..."))
    end
end

function _convergence_default_norm(value)
    payload=_convergence_payload(value)
    payload isa Number ? abs(payload) : norm(payload)
end

function _validate_convergence_levels(levels)
    values=collect(levels)
    length(values)>=2||throw(ArgumentError(
        "a convergence study requires at least two refinement levels"))
    values
end

function _convergence_real_type(errors)
    isempty(errors)&&return Float64
    R=_real_float_type(typeof(first(errors)))
    for error in Iterators.drop(errors,1)
        R=promote_type(R,_real_float_type(typeof(error)))
    end
    R
end

function _raw_refinement_scales(levels,refinement_scale)
    refinement_scale===nothing&&return nothing
    scales=map(refinement_scale,levels)
    for raw in scales
        raw isa Real||throw(ArgumentError(
            "refinement_scale must return a real number"))
        isfinite(raw)&&raw>zero(raw)||throw(ArgumentError(
            "refinement scales must be finite and positive"))
    end
    all(scales[index+1]<scales[index] for index in 1:length(scales)-1)||
        throw(ArgumentError(
            "refinement_scale must strictly decrease along the supplied sequence"))
    scales
end

function _convert_refinement_scales(raw_scales,::Type{R}) where R
    raw_scales===nothing&&return nothing
    scales=Vector{R}(undef,length(raw_scales))
    for index in eachindex(raw_scales)
        scales[index]=try
            R(raw_scales[index])
        catch error
            error isa InexactError||error isa OverflowError||rethrow()
            throw(ArgumentError(
                "refinement scale is not representable in convergence precision $R"))
        end
        isfinite(scales[index])&&scales[index]>zero(R)||throw(ArgumentError(
            "refinement scales must remain finite and positive in convergence precision $R"))
    end
    all(scales[index+1]<scales[index] for index in 1:length(scales)-1)||
        throw(ArgumentError(
            "refinement scales cease to be strictly decreasing in convergence precision $R"))
    scales
end

function _first_passing_window(pairwise,solver_status,required)
    n=length(pairwise)
    required<n||return nothing
    for endpoint in required+1:n
        first_pair=endpoint-required+1
        pair_ok=all(pairwise[index]===true for index in first_pair:endpoint)
        solver_ok=all(solver_status[index]!==false for
                      index in endpoint-required:endpoint)
        pair_ok&&solver_ok&&return endpoint
    end
    nothing
end

"""
    convergence_study(evaluator, refinements; parameter=:refinement,
                      estimate=_convergence_default_estimate,
                      diagnostics=_convergence_default_diagnostics,
                      distance=_convergence_default_distance,
                      estimate_norm=_convergence_default_norm,
                      refinement_scale=nothing, atol=0, rtol=nothing,
                      consecutive=2, require_convergence=false)

Evaluate a deterministic refinement sequence and compare every estimate with
its immediate predecessor. A pair passes when

```math
d(x_i,x_{i-1}) \\leq a + r\\max(\\lVert x_i\\rVert,\\lVert x_{i-1}\\rVert).
```

The final `consecutive` pairs must all pass, and no inner solver in that
window may report `converged=false`, before the study reports convergence.
The default requires two consecutive pairwise agreements, hence at least
three levels. Set `consecutive=1` only when one comparison is scientifically
adequate for the problem.

`evaluator(level)` may return the estimate directly or a result carrying an
`estimate`, `value`, `solution`, `state`, `values`, or `solutions` property.
All raw results are retained. Supply extraction functions for other result
schemas. If
`refinement_scale(level)` is given, it must be positive and strictly decrease;
the report includes empirical rates from three successive estimates when the
three scales are geometrically spaced. Rates are `missing` on a non-geometric
grid rather than applying a formula that is not valid there.

The precision-aware default is `rtol=sqrt(eps(R))`, where `R` is the real
error type. Nonfinite estimates, errors, norms, scales, or tolerances raise.
Set `require_convergence=true` to raise when the final window fails.
"""
function convergence_study(evaluator,refinements;parameter::Symbol=:refinement,
        estimate=_convergence_default_estimate,
        diagnostics=_convergence_default_diagnostics,
        distance=_convergence_default_distance,
        estimate_norm=_convergence_default_norm,
        refinement_scale=nothing,atol::Real=0,rtol=nothing,
        consecutive::Integer=2,require_convergence::Bool=false)
    levels=_validate_convergence_levels(refinements)
    applicable(evaluator,first(levels))||throw(ArgumentError(
        "evaluator must accept one refinement level"))
    consecutive>0||throw(ArgumentError("consecutive must be positive"))
    BigInt(consecutive)<=typemax(Int)||throw(ArgumentError(
        "consecutive must be representable as an Int"))
    isfinite(atol)&&atol>=zero(atol)||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    rtol===nothing||rtol isa Real&&isfinite(rtol)&&rtol>=zero(rtol)||
        throw(ArgumentError("rtol must be nothing or a finite nonnegative real"))
    raw_scales=_raw_refinement_scales(levels,refinement_scale)
    raw_results=map(evaluator,levels)
    estimates=map(estimate,raw_results)
    extracted_diagnostics=map(diagnostics,raw_results)
    solver_status=Union{Missing,Bool}[_convergence_status(result) for result in raw_results]

    raw_errors=map(2:length(estimates)) do index
        value=distance(estimates[index],estimates[index-1])
        value isa Real||throw(ArgumentError("distance must return a real number"))
        value>=0&&isfinite(value)||throw(ArgumentError(
            "pairwise convergence errors must be finite and nonnegative"))
        value
    end
    R=_convergence_real_type(raw_errors)
    atolT=try R(atol) catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("atol is not representable in convergence precision $R"))
    end
    rtolT=rtol===nothing ? sqrt(eps(R)) : try R(rtol) catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("rtol is not representable in convergence precision $R"))
    end
    isfinite(atolT)&&atolT>=zero(R)&&isfinite(rtolT)&&rtolT>=zero(R)||
        throw(ArgumentError("convergence tolerances must be finite and nonnegative"))
    scales=_convert_refinement_scales(raw_scales,R)

    errors=Vector{Union{Missing,R}}(missing,length(levels))
    tolerances=Vector{Union{Missing,R}}(missing,length(levels))
    pairwise=Vector{Union{Missing,Bool}}(missing,length(levels))
    rates=Vector{Union{Missing,R}}(missing,length(levels))
    for index in 2:length(levels)
        error=R(raw_errors[index-1]);errors[index]=error
        left_norm=estimate_norm(estimates[index-1])
        right_norm=estimate_norm(estimates[index])
        left_norm isa Real&&right_norm isa Real||throw(ArgumentError(
            "estimate_norm must return a real number"))
        left=R(left_norm);right=R(right_norm)
        isfinite(left)&&left>=zero(R)&&isfinite(right)&&right>=zero(R)||
            throw(ArgumentError("estimate norms must be finite and nonnegative"))
        tolerance=atolT+rtolT*max(left,right);tolerances[index]=tolerance
        isfinite(tolerance)||throw(ArgumentError(
            "convergence tolerance overflowed in $R; rescale the estimate or use wider precision"))
        pairwise[index]=error<=tolerance
        if index>=3&&scales!==nothing
            previous=errors[index-1]
            if previous!==missing&&previous>zero(R)&&error>zero(R)
                previous_ratio=scales[index-2]/scales[index-1]
                current_ratio=scales[index-1]/scales[index]
                ratio_tolerance=sqrt(eps(R))
                if isfinite(previous_ratio)&&isfinite(current_ratio)&&
                        isapprox(previous_ratio,current_ratio;
                                 atol=zero(R),rtol=ratio_tolerance)
                    rates[index]=(log(previous)-log(error))/
                        (log(scales[index-1])-log(scales[index]))
                end
            end
        end
    end
    required=Int(consecutive)
    first_passing=_first_passing_window(pairwise,solver_status,required)
    enough=required<length(levels)
    final_pairs = if enough
        all(pairwise[index]===true for
            index in length(levels)-required+1:length(levels))
    else
        false
    end
    final_solvers = if enough
        all(solver_status[index]!==false for
            index in length(levels)-required:length(levels))
    else
        false
    end
    converged=final_pairs&&final_solvers
    reason = if !enough
        :insufficient_refinements
    elseif !final_solvers
        :inner_solver_not_converged
    elseif !final_pairs
        first_passing===nothing ? :tolerance_not_met : :refinement_not_stable
    else
        :converged
    end
    metadata=(atol=atolT,rtol=rtolT,refinement_scales=scales)
    report=ConvergenceStudyResult(parameter,levels,raw_results,estimates,
        extracted_diagnostics,errors,tolerances,pairwise,rates,solver_status,
        converged,first_passing,required,reason,metadata)
    require_convergence&&!converged&&throw(ArgumentError(
        "$(parameter) convergence was not established after $(length(levels)) levels; reason=$reason, final_error=$(last(errors))"))
    report
end

"""
    convergence_estimate(report; require_convergence=true)

Return the finest estimate stored in a convergence report. By default an
unconverged report raises; set `require_convergence=false` only when the caller
will propagate the report's status and error data explicitly.
"""
function convergence_estimate(report::ConvergenceStudyResult;
                              require_convergence::Bool=true)
    require_convergence&&!report.converged&&throw(ArgumentError(
        "cannot use an unconverged $(report.parameter) estimate without explicitly setting require_convergence=false"))
    last(report.estimates)
end

function _strictly_decreasing_positive(values,label)
    all(value->value isa Real&&isfinite(value)&&value>0,values)||
        throw(ArgumentError("$label values must be finite and positive"))
    all(values[index+1]<values[index] for index in 1:length(values)-1)||
        throw(ArgumentError("$label values must be strictly decreasing"))
    values
end

function _strictly_increasing_integers(values,label;minimum=0)
    all(value->value isa Integer&&value>=minimum,values)||throw(ArgumentError(
        "$label values must be integers at least $minimum"))
    all(values[index+1]>values[index] for index in 1:length(values)-1)||
        throw(ArgumentError("$label values must be strictly increasing"))
    values
end

"""
    timestep_convergence(evaluator, timesteps; kwargs...)

Run [`convergence_study`](@ref) for a strictly decreasing sequence of positive
time steps. Empirical rates use the time step itself as the decreasing
refinement scale.
"""
function timestep_convergence(evaluator,timesteps;kwargs...)
    levels=_validate_convergence_levels(timesteps)
    _strictly_decreasing_positive(levels,"time-step")
    convergence_study(evaluator,levels;parameter=:time_step,
        refinement_scale=identity,kwargs...)
end
"""
    krylov_dimension_convergence(evaluator, dimensions; kwargs...)

Run a convergence study for strictly increasing positive Krylov dimensions.
The nominal rate scale is `1/dimension`; residual diagnostics returned by each
inner solve are retained and an explicit `converged=false` blocks the report.
"""
function krylov_dimension_convergence(evaluator,dimensions;kwargs...)
    levels=_validate_convergence_levels(dimensions)
    _strictly_increasing_integers(levels,"Krylov dimension";minimum=1)
    convergence_study(evaluator,levels;parameter=:krylov_dimension,
        refinement_scale=dimension->inv(float(dimension)),kwargs...)
end
"""
    hierarchy_depth_convergence(evaluator, depths; kwargs...)

Run a convergence study for strictly increasing nonnegative hierarchy depths.
The nominal refinement scale is `1/(depth+1)`. This wrapper is suitable for
[`HEOMPlan`](@ref) depth studies but remains independent of a particular HEOM
solver or bath decomposition.
"""
function hierarchy_depth_convergence(evaluator,depths;kwargs...)
    levels=_validate_convergence_levels(depths)
    _strictly_increasing_integers(levels,"hierarchy depth";minimum=0)
    convergence_study(evaluator,levels;parameter=:hierarchy_depth,
        refinement_scale=depth->inv(float(depth+1)),kwargs...)
end
"""
    sector_cutoff_convergence(evaluator, cutoffs; kwargs...)

Run a convergence study for a strictly increasing positive retained-sector
cutoff. The nominal refinement scale is `1/cutoff`. If a model uses the
opposite cutoff convention, call [`convergence_study`](@ref) directly with an
appropriate `refinement_scale`.
"""
function sector_cutoff_convergence(evaluator,cutoffs;kwargs...)
    levels=_validate_convergence_levels(cutoffs)
    _strictly_increasing_integers(levels,"sector cutoff";minimum=1)
    convergence_study(evaluator,levels;parameter=:sector_cutoff,
        refinement_scale=cutoff->inv(float(cutoff)),kwargs...)
end
