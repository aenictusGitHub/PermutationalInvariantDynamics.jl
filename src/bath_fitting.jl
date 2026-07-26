"""
    BathCorrelationSamples

Finite samples of a stationary bath correlation function. `times` are
nonnegative and strictly increasing, `values[j]` is the correlation at
`times[j]`, and `metadata` records how the samples were prepared. The object
does not claim that a time/frequency quadrature or finite time window has
converged.
"""
struct BathCorrelationSamples{R,C,M}
    times::Vector{R}
    values::Vector{C}
    metadata::M
end

"""
    BathFitResult

Result of fitting a finite exponential correlation

```math
C(t)=\\sum_k c_k\\exp(-\\nu_k t).
```

The vectors are detached from the inputs. `report.converged` refers only to
the requested weighted sample residual; it is not a hierarchy-depth,
time-step, or spectral-quadrature certificate.
"""
struct BathFitResult{R,C,M}
    times::Vector{R}
    samples::Vector{C}
    coefficients::Vector{C}
    frequencies::Vector{C}
    fitted::Vector{C}
    residuals::Vector{C}
    report::M
end

Base.length(result::BathFitResult)=length(result.coefficients)

function Base.show(io::IO,result::BathFitResult)
    print(io,"BathFitResult(terms=$(length(result)), " *
        "relative_residual=$(result.report.relative_residual), " *
        "converged=$(result.report.converged), " *
        "identifiable=$(result.report.identifiable))")
end

function _bathfit_real_type(values...)
    T=nothing
    for collection in values
        for value in collection
            value isa Number||throw(ArgumentError(
                "bath fitting inputs must contain only numbers"))
            for component in (real(value),imag(value))
                component isa Union{Integer,Rational}&&continue
                S=typeof(float(component))
                T=T===nothing ? S : promote_type(T,S)
            end
        end
    end
    R=T===nothing ? Float64 : T
    R in (Float32,Float64)||throw(ArgumentError(
        "finite exponential fitting currently requires Float32 or Float64 " *
        "data because its rank diagnostics use LAPACK; convert explicitly " *
        "or supply a validated finite decomposition directly to HEOMBath/HOPSBath"))
    R
end

function _bathfit_checked_real(::Type{R},value,label) where R
    value isa Real&&isfinite(value)||throw(ArgumentError(
        "$label must be a finite real number"))
    value isa AbstractFloat&&promote_type(R,typeof(value))!==R&&
        throw(ArgumentError(
            "$label has scalar type $(typeof(value)), which would narrow in " *
            "$R; provide consistently typed bath data or use wider precision"))
    converted=try
        R(value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("$label is not representable in $R"))
    end
    isfinite(converted)||throw(ArgumentError(
        "$label overflows in $R; provide bath data and parameters at " *
        "wider precision"))
    !iszero(value)&&iszero(converted)&&throw(ArgumentError(
        "$label is nonzero but underflows in $R; provide bath data and " *
        "parameters at wider precision"))
    converted
end

function _bathfit_checked_complex(::Type{R},value,label) where R
    value isa Number&&_heom_isfinite(value)||throw(ArgumentError(
        "$label must be a finite number"))
    for component in (real(value),imag(value))
        component isa AbstractFloat&&
            promote_type(R,typeof(component))!==R&&throw(ArgumentError(
                "$label has a component of type $(typeof(component)), which " *
                "would narrow in Complex{$R}; provide consistently typed " *
                "bath data or use wider precision"))
    end
    converted=try
        Complex{R}(value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("$label is not representable in Complex{$R}"))
    end
    _heom_isfinite(converted)||throw(ArgumentError(
        "$label overflows in Complex{$R}; provide bath data and parameters " *
        "at wider precision"))
    (!iszero(real(value))&&iszero(real(converted))||
     !iszero(imag(value))&&iszero(imag(converted)))&&throw(ArgumentError(
        "$label has a nonzero component that underflows in Complex{$R}; " *
        "provide bath data and parameters at wider precision"))
    converted
end

_bathfit_collection_source(value)=
    value===nothing ? () : value isa Number ? (value,) : value

function _bathfit_known_length(values)
    iterator_size=Base.IteratorSize(typeof(values))
    iterator_size isa Union{Base.HasLength,Base.HasShape} ?
        length(values) : nothing
end

_bathfit_raw_collection_bytes(count::Integer)=128BigInt(count)

function _bathfit_guard_raw_collection(
        retained_count::Integer,memory_budget,operation::AbstractString)
    estimate=_bathfit_raw_collection_bytes(retained_count)
    _require_performance_budget(
        operation,estimate,memory_budget;
        guidance="Reduce the supplied sample/candidate arrays.")
end

function _bathfit_preflight_known_collections(
        collections,memory_budget,operation::AbstractString)
    counts=map(_bathfit_known_length,collections)
    all(count->count!==nothing,counts)||return nothing
    total=sum(BigInt(count) for count in counts)
    _bathfit_guard_raw_collection(total,memory_budget,operation)
    total
end

function _bathfit_collect_guarded(
        values,label::AbstractString,memory_budget,retained_count::Base.RefValue)
    known=_bathfit_known_length(values)
    if known!==nothing
        next_count=BigInt(retained_count[])+BigInt(known)
        _bathfit_guard_raw_collection(
            next_count,memory_budget,"$label input collection")
        result=collect(values)
        retained_count[]=next_count
        return result
    end
    result=Any[]
    for value in values
        next_count=BigInt(retained_count[])+BigInt(length(result))+1
        _bathfit_guard_raw_collection(
            next_count,memory_budget,"$label input collection")
        push!(result,value)
    end
    retained_count[]=BigInt(retained_count[])+BigInt(length(result))
    result
end

function _bathfit_times(times,::Type{R};minimum_count::Int=2) where R
    values=R[_bathfit_checked_real(R,value,"correlation time $index")
             for (index,value) in pairs(times)]
    length(values)>=minimum_count||throw(ArgumentError(
        "at least $minimum_count correlation sample" *
        (minimum_count==1 ? " is" : "s are") * " required"))
    first(values)>=zero(R)||throw(ArgumentError(
        "correlation times must be nonnegative"))
    all(diff(values).>zero(R))||throw(ArgumentError(
        "correlation times must be strictly increasing"))
    values
end

function _bathfit_positive_weights(weights,count,::Type{R}) where R
    weights===nothing&&return ones(R,count)
    values=weights isa Real ? fill(weights,count) : weights
    length(values)==count||throw(DimensionMismatch(
        "fit weights and correlation samples must have equal lengths"))
    result=Vector{R}(undef,count)
    for index in eachindex(values)
        result[index]=_bathfit_checked_real(
            R,values[index],"fit weight $index")
        result[index]>zero(R)||throw(ArgumentError(
            "fit weights must be strictly positive"))
    end
    result
end

function _bathfit_poles(poles,::Type{R};label="fit pole") where R
    values=Complex{R}[
        _bathfit_checked_complex(R,value,"$label $index")
        for (index,value) in pairs(poles)]
    isempty(values)&&throw(ArgumentError("at least one fit pole is required"))
    for (index,value) in pairs(values)
        real(value)>zero(R)||throw(ArgumentError(
            "$label $index must have a strictly positive real part"))
    end
    length(unique(values))==length(values)||throw(ArgumentError(
        "fit poles must be distinct; combine exact duplicate poles before fitting"))
    values
end

function _bathfit_auto_poles(times,nterms::Int,::Type{R}) where R
    span=last(times)-first(times)
    smallest_step=minimum(diff(times))
    slow=inv(R(10)*span)
    fast=R(10)/smallest_step
    count_big=max(big(16),4BigInt(nterms))
    count_big<=typemax(Int)||throw(ArgumentError(
        "the automatic bath-pole count exceeds the addressable Int range"))
    count=Int(count_big)
    logarithms=range(log(slow),log(fast);length=count)
    Complex{R}.(exp.(logarithms))
end

function _bathfit_design(times,poles)
    T=promote_type(eltype(poles),Complex{eltype(times)})
    design=Matrix{T}(undef,length(times),length(poles))
    for column in eachindex(poles),row in eachindex(times)
        value=exp(-poles[column]*times[row])
        _heom_isfinite(value)||throw(ArgumentError(
            "fit exponential is not finite at sample $row, pole $column"))
        design[row,column]=value
    end
    design
end

function _bathfit_solve(design,samples,sqrtweights,ridge)
    rows,columns=size(design)
    weighted=similar(design)
    target=similar(samples)
    for row in 1:rows
        weight=sqrtweights[row]
        target[row]=weight*samples[row]
        @views weighted[row,:].=weight.*design[row,:]
    end
    coefficients=if iszero(ridge)
        weighted\target
    else
        augmented=zeros(eltype(design),rows+columns,columns)
        augmented_target=zeros(eltype(samples),rows+columns)
        @views augmented[1:rows,:].=weighted
        @views augmented_target[1:rows].=target
        scale=sqrt(ridge)
        for column in 1:columns
            augmented[rows+column,column]=scale
        end
        augmented\augmented_target
    end
    coefficients,weighted,target
end

function _bathfit_rank_report(weighted,rank_rtol)
    singular_values=svdvals(weighted)
    R=_real_float_type(eltype(weighted))
    largest=isempty(singular_values) ? zero(R) :
        first(singular_values)
    tolerance=rank_rtol*max(largest,one(largest))
    numerical_rank=count(value->value>tolerance,singular_values)
    columns=size(weighted,2)
    condition_number=numerical_rank==columns ?
        largest/last(singular_values) : oftype(largest,Inf)
    (;singular_values,numerical_rank,
      identifiable=numerical_rank==columns,
      rank_tolerance=tolerance,condition_number)
end

function _bathfit_select_omp(design,samples,sqrtweights,nterms,ridge,
        atol,rtol)
    rows,candidates=size(design)
    weighted=similar(design)
    target=similar(samples)
    for row in 1:rows
        target[row]=sqrtweights[row]*samples[row]
        @views weighted[row,:].=sqrtweights[row].*design[row,:]
    end
    column_norms=[norm(view(weighted,:,column))
                  for column in 1:candidates]
    all(>(zero(eltype(column_norms))),column_norms)||throw(ArgumentError(
        "a candidate fit exponential has zero weighted norm"))
    selected=Int[]
    available=trues(candidates)
    residual=copy(target)
    scale=norm(target)
    tolerance=atol+rtol*scale
    coefficients=eltype(samples)[]
    for _ in 1:min(nterms,candidates)
        best=0
        best_residual=oftype(first(column_norms),Inf)
        best_coefficients=eltype(samples)[]
        for candidate in 1:candidates
            available[candidate]||continue
            trial=copy(selected)
            push!(trial,candidate)
            trial_coefficients,trial_design,trial_target=
                _bathfit_solve(
                    view(design,:,trial),samples,sqrtweights,ridge)
            score=norm(trial_design*trial_coefficients-trial_target)
            if score<best_residual
                best=candidate
                best_residual=score
                best_coefficients=trial_coefficients
            end
        end
        best>0||break
        push!(selected,best)
        available[best]=false
        coefficients=best_coefficients
        residual.=target-weighted[:,selected]*coefficients
        norm(residual)<=tolerance&&break
        iszero(best_residual)&&break
    end
    isempty(selected)&&throw(ArgumentError(
        "orthogonal matching pursuit could not select a bath pole"))
    selected,coefficients
end

"""
    fit_bath_correlation(times, values;
                         poles=nothing, candidate_poles=nothing,
                         nterms=4, weights=nothing, ridge=0,
                         atol=0, rtol=nothing, rank_rtol=nothing,
                         memory_budget=512MiB)

Fit sampled complex correlation data to a stable finite exponential sum.

With `poles`, only the coefficients are fitted by weighted least squares.
With `candidate_poles`, deterministic residual-minimizing forward selection
retains at most `nterms` poles before a final least-squares solve. If neither
is supplied, a bounded logarithmic grid of positive real poles is generated;
oscillatory data should normally provide a physically motivated complex
candidate grid.

The operation is finite and memory guarded. `ridge` is an explicit Tikhonov
coefficient and is never chosen automatically. The returned report exposes
the residual, numerical rank, condition number, selected candidates, and
whether the coefficients are compatible with the built-in stationary
Ornstein--Uhlenbeck HOPS noise (`real(c_k) >= 0` and no imaginary parts).
"""
function fit_bath_correlation(times,values;
        poles=nothing,candidate_poles=nothing,nterms::Integer=4,
        weights=nothing,ridge::Real=0,atol::Real=0,rtol=nothing,
        rank_rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    nterms isa Bool&&throw(ArgumentError("nterms must be an integer, not Bool"))
    nterms>0||throw(ArgumentError("nterms must be positive"))
    BigInt(nterms)<=typemax(Int)||throw(ArgumentError(
        "nterms must be representable as an Int"))
    terms=Int(nterms)
    ridge isa Bool&&throw(ArgumentError("ridge must be a real number, not Bool"))
    atol isa Bool&&throw(ArgumentError("atol must be a real number, not Bool"))
    rtol===nothing||rtol isa Real&&!(rtol isa Bool)||throw(ArgumentError(
        "rtol must be a real number or nothing, not Bool"))
    rank_rtol isa Bool&&throw(ArgumentError(
        "rank_rtol must be a real number or nothing, not Bool"))
    rank_rtol===nothing||rank_rtol isa Real||throw(ArgumentError(
        "rank_rtol must be a real number or nothing"))
    poles===nothing||candidate_poles===nothing||throw(ArgumentError(
        "pass either poles or candidate_poles, not both"))

    time_source=_bathfit_collection_source(times)
    value_source=_bathfit_collection_source(values)
    pole_source=_bathfit_collection_source(poles)
    candidate_source=_bathfit_collection_source(candidate_poles)
    weight_source=weights===nothing||weights isa Real ?
        () : _bathfit_collection_source(weights)
    collections=(
        time_source,value_source,pole_source,candidate_source,weight_source)
    _bathfit_preflight_known_collections(
        collections,memory_budget,"finite exponential bath-fit input collection")
    retained_count=Ref{BigInt}(0)
    raw_times=_bathfit_collect_guarded(
        time_source,"correlation-time",memory_budget,retained_count)
    raw_values=_bathfit_collect_guarded(
        value_source,"correlation-value",memory_budget,retained_count)
    raw_poles=_bathfit_collect_guarded(
        pole_source,"fixed-pole",memory_budget,retained_count)
    raw_candidates=_bathfit_collect_guarded(
        candidate_source,"candidate-pole",memory_budget,retained_count)
    raw_weights=weights===nothing||weights isa Real ? weights :
        _bathfit_collect_guarded(
            weight_source,"fit-weight",memory_budget,retained_count)
    length(raw_times)==length(raw_values)||throw(DimensionMismatch(
        "correlation times and values must have equal lengths"))

    scalar_hints=(
        ridge,atol,
        (rtol===nothing ? () : (rtol,))...,
        (rank_rtol===nothing ? () : (rank_rtol,))...,
        (weights isa Real ? (weights,) : ())...)
    R=_bathfit_real_type(
        raw_times,raw_values,raw_poles,raw_candidates,
        raw_weights isa AbstractVector ? raw_weights : (),scalar_hints)

    rows=length(raw_times)
    rows>=2||throw(ArgumentError(
        "at least two correlation samples are required"))
    raw_weights isa AbstractVector&&length(raw_weights)!=rows&&
        throw(DimensionMismatch(
            "fit weights and correlation samples must have equal lengths"))
    predicted_candidates=if poles!==nothing
        length(raw_poles)
    elseif candidate_poles!==nothing
        length(raw_candidates)
    else
        count=max(big(16),4BigInt(terms))
        count<=typemax(Int)||throw(ArgumentError(
            "the automatic bath-pole count exceeds the addressable Int range"))
        Int(count)
    end
    predicted_candidates>0||throw(ArgumentError(
        "at least one fit pole is required"))
    candidate_poles!==nothing&&terms>predicted_candidates&&
        throw(ArgumentError(
            "nterms cannot exceed the number of candidate poles"))
    entries=BigInt(rows)*BigInt(predicted_candidates)+
            6BigInt(rows)*max(BigInt(predicted_candidates),BigInt(terms))+
            4(BigInt(rows)+BigInt(predicted_candidates))^2+
            24(BigInt(rows)+BigInt(predicted_candidates))
    estimated_bytes=_bathfit_raw_collection_bytes(retained_count[])+
        _performance_entries_bytes(entries,Complex{R})
    _require_performance_budget(
        "finite exponential bath fitting",estimated_bytes,memory_budget;
        guidance="Reduce the sample/candidate grid or fit a supplied smaller pole set.")

    ts=_bathfit_times(raw_times,R)
    samples=Complex{R}[
        _bathfit_checked_complex(R,value,"correlation sample $index")
        for (index,value) in pairs(raw_values)]
    fitweights=_bathfit_positive_weights(raw_weights,length(ts),R)
    sqrtweights=sqrt.(fitweights)
    ridgeR=_bathfit_checked_real(R,ridge,"ridge")
    ridgeR>=zero(R)||throw(ArgumentError("ridge must be nonnegative"))
    atolR=_bathfit_checked_real(R,atol,"atol")
    rtolR=rtol===nothing ? R(1e-6) :
        _bathfit_checked_real(R,rtol,"rtol")
    atolR>=zero(R)&&rtolR>=zero(R)||throw(ArgumentError(
        "fit tolerances must be nonnegative"))
    rank_rtolR=if rank_rtol===nothing
        rank_scale=R(max(length(ts),terms))
        isfinite(rank_scale)||throw(ArgumentError(
            "the default rank tolerance scale is not representable in $R; " *
            "reduce nterms or use wider precision"))
        rank_scale*eps(R)
    else
        _bathfit_checked_real(R,rank_rtol,"rank_rtol")
    end
    rank_rtolR>=zero(R)||throw(ArgumentError(
        "rank_rtol must be nonnegative"))

    method=:unknown
    candidates=Complex{R}[]
    selected=Int[]
    selected_poles=Complex{R}[]
    if poles!==nothing
        candidates=_bathfit_poles(raw_poles,R)
        selected=collect(eachindex(candidates))
        selected_poles=copy(candidates)
        method=:fixed_poles
    else
        candidates=candidate_poles===nothing ?
            _bathfit_auto_poles(ts,terms,R) :
            _bathfit_poles(raw_candidates,R;label="candidate pole")
        terms<=length(candidates)||throw(ArgumentError(
            "nterms cannot exceed the number of candidate poles"))
        method=candidate_poles===nothing ?
            :automatic_real_grid_forward_selection :
            :candidate_grid_forward_selection
        selected=Int[]
        selected_poles=Complex{R}[]
    end

    candidate_count=length(candidates)
    candidate_count==predicted_candidates||throw(AssertionError(
        "prepared bath candidate count differs from its guarded estimate"))

    candidate_design=_bathfit_design(ts,candidates)
    if method!==:fixed_poles
        selected,_=_bathfit_select_omp(candidate_design,samples,sqrtweights,
            terms,ridgeR,atolR,rtolR)
        selected_poles=candidates[selected]
    end
    design=candidate_design[:,selected]
    coefficients,weighted_design,weighted_target=
        _bathfit_solve(design,samples,sqrtweights,ridgeR)
    fitted=design*coefficients
    residuals=fitted-samples
    weighted_residual=weighted_design*coefficients-weighted_target
    weighted_norm=norm(weighted_residual)
    sample_norm=norm(weighted_target)
    tolerance=atolR+rtolR*sample_norm
    relative=sample_norm>zero(R) ? weighted_norm/sample_norm :
        (iszero(weighted_norm) ? zero(R) : oftype(weighted_norm,Inf))
    rank_report=_bathfit_rank_report(weighted_design,rank_rtolR)
    ou_compatible=all(coefficient->
        iszero(imag(coefficient))&&real(coefficient)>=zero(R),
        coefficients)
    report=merge(rank_report,(
        method,
        converged=weighted_norm<=tolerance,
        stable=all(pole->real(pole)>zero(R),selected_poles),
        weighted_residual_norm=weighted_norm,
        relative_residual=relative,
        rms_residual=norm(residuals)/sqrt(R(rows)),
        maximum_absolute_residual=maximum(abs,residuals;init=zero(R)),
        residual_tolerance=tolerance,
        sample_count=rows,
        candidate_count,
        selected_candidates=copy(selected),
        term_count=length(selected_poles),
        ridge=ridgeR,
        hops_stationary_ou_compatible=ou_compatible,
        estimated_peak_bytes=estimated_bytes,
        memory_budget=isfinite(memory_budget) ?
            BigInt(floor(BigInt,memory_budget)) : nothing,
        exclusions=(
            "allocator/LAPACK implementation overhead",
            "HEOM/HOPS hierarchy storage and solver workspaces")))
    BathFitResult(copy(ts),copy(samples),copy(coefficients),
        copy(selected_poles),copy(fitted),copy(residuals),report)
end

fit_bath_correlation(samples::BathCorrelationSamples;kwargs...)=
    fit_bath_correlation(samples.times,samples.values;kwargs...)

@inline function _bathfit_coth(value)
    threshold=sqrt(eps(typeof(value)))
    abs(value)<=threshold ?
        inv(value)+value/3-value^3/45 : inv(tanh(value))
end

function _bathfit_trapezoid_weights(grid)
    count=length(grid)
    weights=similar(grid)
    weights[1]=(grid[2]-grid[1])/2
    for index in 2:count-1
        weights[index]=(grid[index+1]-grid[index-1])/2
    end
    weights[end]=(grid[end]-grid[end-1])/2
    weights
end

"""
    correlation_from_spectral_density(frequencies, spectral_density, times;
                                      inverse_temperature=Inf,
                                      normalization=inv(pi),
                                      memory_budget=512MiB)

Compute trapezoidal samples of the bosonic correlation convention

```math
C(t)=a\\int_0^\\infty J(\\omega)
 \\left[\\coth\\left(\\frac{\\beta\\omega}{2}\\right)\\cos(\\omega t)
       -i\\sin(\\omega t)\\right]d\\omega ,
```

where `a=normalization`. The supplied frequency grid must be finite, strictly
positive, and increasing, and the spectral density samples must be finite,
real, and nonnegative. Strict positivity avoids silently inventing the
model-dependent `omega -> 0` limit of `J(omega)coth(beta*omega/2)`.

The returned metadata records the finite integration interval and quadrature.
It is not a quadrature-convergence certificate; refine the frequency grid and
cutoff separately from the subsequent exponential fit.
"""
function correlation_from_spectral_density(frequencies,spectral_density,times;
        inverse_temperature=Inf,normalization=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    inverse_temperature isa Real&&!(inverse_temperature isa Bool)||
        throw(ArgumentError(
            "inverse_temperature must be a real number, not Bool"))
    frequency_source=_bathfit_collection_source(frequencies)
    density_source=_bathfit_collection_source(spectral_density)
    time_source=_bathfit_collection_source(times)
    collections=(frequency_source,density_source,time_source)
    _bathfit_preflight_known_collections(
        collections,memory_budget,
        "spectral-density correlation input collection")
    retained_count=Ref{BigInt}(0)
    raw_frequencies=_bathfit_collect_guarded(
        frequency_source,"frequency",memory_budget,retained_count)
    raw_density=_bathfit_collect_guarded(
        density_source,"spectral-density",memory_budget,retained_count)
    raw_times=_bathfit_collect_guarded(
        time_source,"correlation-time",memory_budget,retained_count)
    length(raw_frequencies)==length(raw_density)||throw(DimensionMismatch(
        "frequency and spectral-density arrays must have equal lengths"))
    length(raw_frequencies)>=2||throw(ArgumentError(
        "at least two spectral-density samples are required"))
    scalar_hints=(
        isfinite(inverse_temperature) ? inverse_temperature :
            first(raw_frequencies),
        normalization===nothing ? first(raw_frequencies) : normalization)
    R=_bathfit_real_type(raw_frequencies,raw_density,raw_times,scalar_hints)
    estimated_bytes=_bathfit_raw_collection_bytes(retained_count[])+
        _performance_entries_bytes(
            5BigInt(length(raw_frequencies))+4BigInt(length(raw_times)),
            Complex{R})
    _require_performance_budget(
        "spectral-density correlation quadrature",
        estimated_bytes,memory_budget;
        guidance="Reduce the frequency/time sample grids.")

    omega=R[_bathfit_checked_real(R,value,"frequency $index")
            for (index,value) in pairs(raw_frequencies)]
    all(>(zero(R)),omega)||throw(ArgumentError(
        "spectral-density frequencies must be strictly positive"))
    all(diff(omega).>zero(R))||throw(ArgumentError(
        "spectral-density frequencies must be strictly increasing"))
    density=R[_bathfit_checked_real(R,value,"spectral density $index")
              for (index,value) in pairs(raw_density)]
    all(>=(zero(R)),density)||throw(ArgumentError(
        "spectral-density samples must be nonnegative"))
    ts=_bathfit_times(raw_times,R;minimum_count=1)
    beta=if inverse_temperature isa Real&&isinf(inverse_temperature)&&
            inverse_temperature>0
        R(Inf)
    else
        converted=_bathfit_checked_real(
            R,inverse_temperature,"inverse_temperature")
        converted>zero(R)||throw(ArgumentError(
            "inverse_temperature must be strictly positive or Inf"))
        converted
    end
    scale=normalization===nothing ? inv(R(pi)) :
        _bathfit_checked_real(R,normalization,"normalization")
    scale>zero(R)||throw(ArgumentError(
        "normalization must be strictly positive"))
    quadrature_weights=_bathfit_trapezoid_weights(omega)
    thermal=Vector{R}(undef,length(omega))
    if isinf(beta)
        fill!(thermal,one(R))
    else
        for index in eachindex(omega)
            argument=(beta/2)*omega[index]
            isfinite(argument)||throw(ArgumentError(
                "thermal spectral argument $index overflows; rescale the " *
                "frequency grid or use wider precision"))
            !iszero(beta)&&!iszero(omega[index])&&iszero(argument)&&
                throw(ArgumentError(
                    "thermal spectral argument $index underflows; rescale " *
                    "the frequency grid or use wider precision"))
            thermal[index]=_bathfit_coth(argument)
            isfinite(thermal[index])||throw(ArgumentError(
                "thermal spectral factor $index is not finite"))
        end
    end
    values=Vector{Complex{R}}(undef,length(ts))
    for (time_index,time) in pairs(ts)
        accumulated=zero(Complex{R})
        for index in eachindex(omega)
            phase=omega[index]*time
            isfinite(phase)||throw(ArgumentError(
                "quadrature phase at time $time_index, frequency $index is " *
                "nonfinite; rescale the grid or use wider precision"))
            cosine=cos(phase)
            sine=sin(phase)
            thermal_cosine=thermal[index]*cosine
            isfinite(thermal_cosine)||throw(ArgumentError(
                "thermal quadrature kernel at time $time_index, frequency " *
                "$index overflows; rescale the grid or use wider precision"))
            kernel=Complex{R}(thermal_cosine,-sine)
            integrand=density[index]*kernel
            _heom_isfinite(integrand)||throw(ArgumentError(
                "spectral integrand at time $time_index, frequency $index " *
                "overflows; rescale the spectral density or use wider precision"))
            contribution=quadrature_weights[index]*integrand
            _heom_isfinite(contribution)||throw(ArgumentError(
                "quadrature contribution at time $time_index, frequency " *
                "$index overflows; refine/rescale the grid or use wider precision"))
            updated=accumulated+contribution
            _heom_isfinite(updated)||throw(ArgumentError(
                "quadrature accumulation at time $time_index overflows; " *
                "rescale the spectral density or use wider precision"))
            accumulated=updated
        end
        result=scale*accumulated
        _heom_isfinite(result)||throw(ArgumentError(
            "quadrature result at time $time_index is nonfinite; rescale " *
            "normalization/spectral density or use wider precision"))
        values[time_index]=result
    end
    metadata=(
        source=:spectral_density,
        quadrature=:trapezoid,
        inverse_temperature=beta,
        normalization=scale,
        frequency_points=length(omega),
        frequency_interval=(first(omega),last(omega)),
        estimated_peak_bytes=estimated_bytes,
        converged=nothing,
        convergence_note="refine frequency spacing and cutoff explicitly")
    BathCorrelationSamples(copy(ts),values,metadata)
end

"""
    fit_bath_from_spectral_density(frequencies, spectral_density, times;
                                   correlation_options=(;),
                                   fit_options=(;))

Prepare finite-temperature correlation samples with
[`correlation_from_spectral_density`](@ref), then fit them with
[`fit_bath_correlation`](@ref). Spectral quadrature and exponential-fit
options are deliberately separate so their convergence claims cannot be
confused.
"""
function fit_bath_from_spectral_density(frequencies,spectral_density,times;
        correlation_options::NamedTuple=(;),fit_options::NamedTuple=(;))
    samples=correlation_from_spectral_density(
        frequencies,spectral_density,times;correlation_options...)
    fit_bath_correlation(samples;fit_options...)
end

function _bathfit_validate_preparation(result::BathFitResult;
        accept_unconverged::Bool,accept_rank_deficient::Bool)
    result.report.stable||throw(ArgumentError(
        "bath fit contains a pole without strictly positive real decay"))
    result.report.converged||accept_unconverged||throw(ArgumentError(
        "bath fit residual did not meet its requested tolerance; refine the " *
        "pole model or pass accept_unconverged=true explicitly"))
    result.report.identifiable||accept_rank_deficient||throw(ArgumentError(
        "bath fit coefficients are numerically rank deficient; reduce the " *
        "pole set or pass accept_rank_deficient=true explicitly"))
    nothing
end

"""
    prepare_heom_bath(coupling, fit; right_coefficients=nothing, residue=0,
                      accept_unconverged=false,
                      accept_rank_deficient=false, metadata=(;), kwargs...)

Construct an [`HEOMBath`](@ref) from a finite exponential fit. By default an
unconverged or rank-deficient fit is rejected rather than silently promoted to
a dynamical model. Fit provenance and diagnostics are attached to the bath
metadata. Remaining `kwargs` are forwarded to `HEOMBath`.
"""
function prepare_heom_bath(coupling::PIOperator,result::BathFitResult;
        right_coefficients=nothing,residue=0,
        accept_unconverged::Bool=false,
        accept_rank_deficient::Bool=false,
        metadata::NamedTuple=(;),kwargs...)
    _bathfit_validate_preparation(result;
        accept_unconverged,accept_rank_deficient)
    fit_metadata=(
        source=:finite_exponential_fit,
        fit_method=result.report.method,
        fit_converged=result.report.converged,
        fit_identifiable=result.report.identifiable,
        fit_relative_residual=result.report.relative_residual,
        fit_condition_number=result.report.condition_number,
        fit_term_count=result.report.term_count)
    HEOMBath(coupling,result.coefficients,result.frequencies;
        right_coefficients,residue,
        metadata=merge(fit_metadata,metadata),kwargs...)
end

"""
    prepare_hops_bath(coupling, fit; label=:fitted_bath,
                      require_stationary_ou=false,
                      accept_unconverged=false,
                      accept_rank_deficient=false)

Construct a [`HOPSBath`](@ref) from a finite exponential fit. Set
`require_stationary_ou=true` when the built-in stationary
Ornstein--Uhlenbeck path will be used; complex or negative fitted residues are
then rejected. A general HOPS decomposition may still be used with an
explicit covariance-correct noise provider.
"""
function prepare_hops_bath(coupling::PIOperator,result::BathFitResult;
        label=:fitted_bath,require_stationary_ou::Bool=false,
        accept_unconverged::Bool=false,
        accept_rank_deficient::Bool=false)
    _bathfit_validate_preparation(result;
        accept_unconverged,accept_rank_deficient)
    require_stationary_ou&&
        !result.report.hops_stationary_ou_compatible&&throw(ArgumentError(
            "the fitted coefficients are not compatible with the built-in " *
            "stationary Ornstein--Uhlenbeck HOPS noise; provide an explicit " *
            "valid noise process or change the pole decomposition"))
    HOPSBath(coupling,result.coefficients,result.frequencies;label)
end
