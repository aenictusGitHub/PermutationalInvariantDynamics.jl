module ExampleContourFit

using LinearAlgebra

export quadratic_level_contour_fit, save_level_contour_data

function _required_contour_property(object,name::Symbol,context::AbstractString)
    hasproperty(object,name)||throw(ArgumentError(
        "$context does not provide the required `$name` field"))
    getproperty(object,name)
end

function _validated_contour_series(
        object,xname::Symbol,yname::Symbol,label::AbstractString;
        require_nonempty::Bool)
    x=_required_contour_property(object,xname,label)
    y=_required_contour_property(object,yname,label)
    x isa AbstractVector||throw(ArgumentError(
        "$label `$xname` data must be an AbstractVector"))
    y isa AbstractVector||throw(ArgumentError(
        "$label `$yname` data must be an AbstractVector"))
    length(x)==length(y)||throw(DimensionMismatch(
        "$label x/y data have lengths $(length(x)) and $(length(y))"))
    require_nonempty&&isempty(x)&&throw(ArgumentError(
        "$label data must contain at least one point"))
    for (coordinate,values) in (("x",x),("y",y))
        for (index,value) in pairs(values)
            value isa Real&&isfinite(value)||throw(ArgumentError(
                "$label $coordinate[$index] must be a finite real number; got $value"))
        end
    end
    (;label=String(label),x,y)
end

function _single_line_metadata(value)
    replace(repr(value),'\t'=>"\\t",'\r'=>"\\r",'\n'=>"\\n")
end

function _write_contour_metadata(io,key,value)
    metadata_key=replace(string(key),'\t'=>"\\t",'\r'=>"\\r",'\n'=>"\\n")
    println(io,"# ",metadata_key," = ",
            _single_line_metadata(value))
end

function _write_fit_diagnostics(io,prefix,fit)
    for name in (:coefficients,:alpha,:beta,:rms_residual,
                 :relative_l2_residual,:maximum_relative_residual,
                 :residual_degrees_of_freedom,:numerical_rank,
                 :plottable,:objective)
        hasproperty(fit,name)||continue
        _write_contour_metadata(io,"$prefix.$name",getproperty(fit,name))
    end
end

function _atomic_contour_write(writer,path::AbstractString)
    isempty(strip(path))&&throw(ArgumentError(
        "the contour-data path must not be empty or whitespace"))
    output=abspath(String(path))
    isdir(output)&&throw(ArgumentError(
        "the contour-data path names an existing directory: $output"))
    directory=dirname(output)
    mkpath(directory)
    temporary,io=mktemp(directory;cleanup=false)
    moved=false
    try
        writer(io)
        flush(io)
        close(io)
        # The temporary file is in the destination directory, so this rename
        # exposes only a complete file and remains on the same filesystem.
        Base.Filesystem.rename(temporary,output)
        moved=true
    finally
        isopen(io)&&close(io)
        !moved&&ispath(temporary)&&rm(temporary;force=true)
    end
    output
end

function _validated_contour_column_label(label::AbstractString,name::Symbol)
    value=String(label)
    !isempty(value)&&value==strip(value)||throw(ArgumentError(
        "$name must not be empty or have leading or trailing whitespace"))
    any(character->character in ('\t','\r','\n'),value)&&throw(ArgumentError(
        "$name must not contain tab or newline characters"))
    value in ("series","point_index")&&throw(ArgumentError(
        "$name conflicts with a reserved contour-data column"))
    value
end

"""
    save_level_contour_data(
        path, fit;
        metadata=NamedTuple(),
        x_label="x",
        y_label="y",
    )

Atomically write the extracted and fitted curves returned by
[`quadratic_level_contour_fit`](@ref) as a UTF-8, tab-delimited text file.
Comment lines record the fit diagnostics and caller-supplied named-tuple
`metadata`; the tabular columns are `series`, `point_index`, `x`, and `y` by
default. Set `x_label` and `y_label` to self-describing coordinate names such
as `"J_over_omega_c"` and `"kappa_over_omega_c"`.

The stable series names are `extracted_boundary`,
`origin_quadratic_fit`, `general_quadratic_fit`, and `power_law_fit`. The
origin-constrained series is always present. Empirical candidate series are
included whenever the corresponding candidate is available, even when it was
not selected for display. Empty, unplottable candidate curves contribute
metadata but no numerical rows. Coordinates are printed in their existing
numeric type and are never narrowed.

Parent directories are created as needed. The returned string is the absolute
path of the completed file.
"""
function save_level_contour_data(
        path::AbstractString,fit;metadata::NamedTuple=NamedTuple(),
        x_label::AbstractString="x",y_label::AbstractString="y")
    x_column=_validated_contour_column_label(x_label,:x_label)
    y_column=_validated_contour_column_label(y_label,:y_label)
    x_column==y_column&&throw(ArgumentError(
        "x_label and y_label must identify different columns"))
    extracted=_validated_contour_series(
        fit,:crossing_x,:crossing_y,"extracted_boundary";
        require_nonempty=true)
    origin=_validated_contour_series(
        fit,:fit_x,:fit_y,"origin_quadratic_fit";
        require_nonempty=true)
    series=Any[extracted,origin]

    general=_required_contour_property(
        fit,:general_quadratic,"contour fit")
    if general!==nothing
        push!(series,_validated_contour_series(
            general,:fit_x,:fit_y,"general_quadratic_fit";
            require_nonempty=false))
    end
    power=_required_contour_property(fit,:power_law,"contour fit")
    if power!==nothing
        push!(series,_validated_contour_series(
            power,:fit_x,:fit_y,"power_law_fit";
            require_nonempty=false))
    end

    display_model=_required_contour_property(
        fit,:display_model,"contour fit")
    level=_required_contour_property(fit,:level,"contour fit")
    alpha=_required_contour_property(fit,:alpha,"contour fit")
    relative_l2_residual=_required_contour_property(
        fit,:relative_l2_residual,"contour fit")
    maximum_relative_residual=_required_contour_property(
        fit,:maximum_relative_residual,"contour fit")

    _atomic_contour_write(path) do io
        println(io,"# PermutationalInvariantDynamics.jl level-contour data")
        _write_contour_metadata(io,"format_version",1)
        _write_contour_metadata(io,"level",level)
        _write_contour_metadata(io,"display_model",display_model)
        _write_contour_metadata(io,"interpolation",
            _required_contour_property(fit,:interpolation,"contour fit"))
        _write_contour_metadata(io,"point_count",length(extracted.x))
        _write_contour_metadata(io,"touches_boundary",
            _required_contour_property(fit,:touches_boundary,"contour fit"))
        _write_contour_metadata(io,"boundary_sides",
            _required_contour_property(fit,:boundary_sides,"contour fit"))
        _write_contour_metadata(io,"origin_quadratic.coefficients",(;alpha))
        _write_contour_metadata(io,"origin_quadratic.relative_l2_residual",
                                relative_l2_residual)
        _write_contour_metadata(io,"origin_quadratic.maximum_relative_residual",
                                maximum_relative_residual)
        for name in (:rms_residual,:leave_one_out_range,:objective)
            hasproperty(fit,name)||continue
            _write_contour_metadata(
                io,"origin_quadratic.$name",getproperty(fit,name))
        end
        for (name,key) in (
                (:fallback_model,"fallback.model"),
                (:fallback_relative_l2_threshold,
                 "fallback.relative_l2_threshold"),
                (:fallback_metric,"fallback.metric"),
                (:fallback_triggered,"fallback.triggered"),
                (:fallback_status,"fallback.status"))
            _write_contour_metadata(io,key,
                _required_contour_property(fit,name,"contour fit"))
        end
        _write_contour_metadata(io,"general_quadratic.available",
                                general!==nothing)
        general===nothing||_write_fit_diagnostics(
            io,"general_quadratic",general)
        _write_contour_metadata(io,"power_law.available",power!==nothing)
        power===nothing||_write_fit_diagnostics(io,"power_law",power)
        for name in keys(metadata)
            _write_contour_metadata(
                io,"metadata.$name",getproperty(metadata,name))
        end
        println(io,"series\tpoint_index\t",x_column,'\t',y_column)
        for curve in series
            for index in eachindex(curve.x,curve.y)
                print(io,curve.label,'\t',index,'\t',curve.x[index],
                      '\t',curve.y[index],'\n')
            end
        end
    end
end

function _general_quadratic_fit(
        crossing_x,crossing_y,yminimum,ymaximum,sample_count,::Type{R}) where R
    point_count=length(crossing_x)
    point_count>=4||return nothing,:insufficient_residual_degrees_of_freedom

    center=sum(crossing_x)/R(point_count)
    scale=maximum(abs.(crossing_x.-center))
    isfinite(scale)&&scale>zero(R)||return nothing,:insufficient_distinct_abscissae
    normalized_x=(crossing_x.-center)./scale
    design=hcat(normalized_x.^2,normalized_x,ones(R,point_count))
    factorization=qr(design,ColumnNorm())
    diagonal=abs.(diag(factorization.R))
    largest_diagonal=maximum(diagonal)
    rank_tolerance=R(max(size(design)...))*eps(R)*largest_diagonal
    numerical_rank=count(value->value>rank_tolerance,diagonal)
    numerical_rank==3||return nothing,:rank_deficient

    normalized_coefficients=factorization\crossing_y
    quadratic_normalized,linear_normalized,constant_normalized=
        normalized_coefficients
    inverse_scale=inv(scale)
    quadratic=quadratic_normalized*inverse_scale^2
    linear=linear_normalized*inverse_scale-
        R(2)*center*quadratic_normalized*inverse_scale^2
    constant=constant_normalized-center*linear_normalized*inverse_scale+
        center^2*quadratic_normalized*inverse_scale^2
    coefficients=(;quadratic,linear,constant)

    predicted_y=design*normalized_coefficients
    residuals=crossing_y.-predicted_y
    rms_residual=sqrt(sum(abs2,residuals)/R(point_count))
    relative_l2_residual=norm(residuals)/norm(crossing_y)
    maximum_relative_residual=maximum(abs.(residuals)./crossing_y)
    fit_x=collect(range(first(crossing_x),last(crossing_x);
                        length=sample_count))
    normalized_fit_x=(fit_x.-center)./scale
    fit_y=quadratic_normalized.*normalized_fit_x.^2 .+
        linear_normalized.*normalized_fit_x .+ constant_normalized
    domain_tolerance=R(64)*eps(R)*max(one(R),abs(yminimum),abs(ymaximum))
    plottable=all(value->isfinite(value)&&value>zero(R)&&
        yminimum-domain_tolerance<=value<=ymaximum+domain_tolerance,fit_y)
    smallest_diagonal=minimum(diagonal)
    qr_diagonal_condition_estimate=largest_diagonal/smallest_diagonal
    candidate=(;coefficients,
        normalized_coefficients=(
            quadratic=quadratic_normalized,
            linear=linear_normalized,
            constant=constant_normalized),
        center,scale,numerical_rank,rank_tolerance,
        qr_diagonal_condition_estimate,
        residual_degrees_of_freedom=point_count-3,
        predicted_y,residuals,rms_residual,relative_l2_residual,
        maximum_relative_residual,fit_x,fit_y,
        fit_x_span=(first(fit_x),last(fit_x)),plottable,
        objective=:general_quadratic_least_squares)
    candidate,plottable ? :available : :outside_sampled_vertical_domain
end

function _power_law_profile(log_x,scaled_y,beta,::Type{R}) where R
    anchor=maximum(log_x)
    exponents=beta.*(log_x.-anchor)
    all(isfinite,exponents)||return (objective=R(Inf),gamma=R(NaN),
        anchor,weights=R[],predicted=R[])
    weights=exp.(exponents)
    denominator=dot(weights,weights)
    isfinite(denominator)&&denominator>zero(R)||return (
        objective=R(Inf),gamma=R(NaN),anchor,weights,predicted=R[])
    gamma=dot(weights,scaled_y)/denominator
    isfinite(gamma)&&gamma>zero(R)||return (
        objective=R(Inf),gamma,anchor,weights,predicted=R[])
    predicted=gamma.*weights
    objective=sum(abs2,scaled_y.-predicted)
    (;objective,gamma,anchor,weights,predicted)
end

function _power_law_bracket(objective,beta0,beta_limit,::Type{R}) where R
    step=max(one(R),abs(beta0)/R(4))
    f0=objective(beta0)
    isfinite(f0)||return nothing,:nonfinite_objective
    left=max(zero(R),beta0-step)
    right=min(beta_limit,beta0+step)
    fleft=left==beta0 ? f0 : objective(left)
    fright=right==beta0 ? f0 : objective(right)
    if f0<=fleft&&f0<=fright
        left<right||return nothing,:endpoint_dominated
        return (left,right),:available
    end

    direction=fleft<fright ? -one(R) : one(R)
    previous=beta0
    current=direction<zero(R) ? left : right
    fcurrent=direction<zero(R) ? fleft : fright
    current==previous&&return nothing,:endpoint_dominated
    if current==zero(R)||current==beta_limit
        return nothing,:endpoint_dominated
    end
    for _ in 1:64
        step*=R(2)
        next_beta=clamp(current+direction*step,zero(R),beta_limit)
        next_beta==current&&return nothing,:endpoint_dominated
        fnext=objective(next_beta)
        if !isfinite(fnext)||fnext>=fcurrent
            return (min(previous,next_beta),max(previous,next_beta)),:available
        end
        previous=current
        current=next_beta
        fcurrent=fnext
        if current==zero(R)||current==beta_limit
            return nothing,:endpoint_dominated
        end
    end
    nothing,:bracket_failed
end

function _golden_section_minimum(objective,bracket,::Type{R}) where R
    lower,upper=bracket
    ratio=(sqrt(R(5))-one(R))/R(2)
    left=upper-ratio*(upper-lower)
    right=lower+ratio*(upper-lower)
    fleft=objective(left)
    fright=objective(right)
    beta_tolerance=R(8)*sqrt(eps(R))
    converged=false
    iterations=0
    for iteration in 1:256
        iterations=iteration
        width=upper-lower
        tolerance=beta_tolerance*max(one(R),abs(left),abs(right))
        if width<=tolerance
            converged=true
            break
        end
        if fleft<=fright
            upper=right
            right=left
            fright=fleft
            left=upper-ratio*(upper-lower)
            if left==lower||left==right
                converged=true
                break
            end
            fleft=objective(left)
        else
            lower=left
            left=right
            fleft=fright
            right=lower+ratio*(upper-lower)
            if right==left||right==upper
                converged=true
                break
            end
            fright=objective(right)
        end
    end
    candidates=(lower,left,right,upper)
    objectives=map(objective,candidates)
    best=argmin(objectives)
    (;beta=candidates[best],objective=objectives[best],converged,
      iterations,beta_tolerance,final_bracket=(lower,upper))
end

function _positive_power_law_fit(
        crossing_x,crossing_y,yminimum,ymaximum,sample_count,::Type{R}) where R
    point_count=length(crossing_x)
    point_count>=3||return nothing,:insufficient_residual_degrees_of_freedom
    all(value->isfinite(value)&&value>zero(R),crossing_x)&&
        all(value->isfinite(value)&&value>zero(R),crossing_y)||
        return nothing,:nonpositive_data

    log_x=log.(crossing_x)
    log_y=log.(crossing_y)
    log_center=sum(log_x)/R(point_count)
    log_y_center=sum(log_y)/R(point_count)
    centered_log_x=log_x.-log_center
    centered_log_y=log_y.-log_y_center
    log_span=maximum(log_x)-minimum(log_x)
    span_tolerance=R(64)*eps(R)*max(one(R),maximum(abs,log_x))
    log_span>span_tolerance||return nothing,:insufficient_distinct_abscissae
    denominator=dot(centered_log_x,centered_log_x)
    denominator>zero(R)||return nothing,:insufficient_distinct_abscissae
    initial_beta=dot(centered_log_x,centered_log_y)/denominator
    isfinite(initial_beta)||return nothing,:nonfinite_initial_exponent

    smallest=nextfloat(zero(R))
    beta_limit=-log(smallest)/log_span
    isfinite(beta_limit)&&beta_limit>zero(R)||
        return nothing,:unrepresentable_exponent_domain
    beta0=clamp(initial_beta,zero(R),beta_limit)
    yscale=maximum(crossing_y)
    scaled_y=crossing_y./yscale
    evaluations=Ref(0)
    objective=beta->begin
        evaluations[]+=1
        _power_law_profile(log_x,scaled_y,beta,R).objective
    end
    bracket,status=_power_law_bracket(objective,beta0,beta_limit,R)
    bracket===nothing&&return nothing,status
    optimized=_golden_section_minimum(objective,bracket,R)
    optimized.converged||return nothing,:optimizer_not_converged
    beta=optimized.beta
    endpoint_tolerance=optimized.beta_tolerance*max(one(R),abs(beta))
    beta>endpoint_tolerance||return nothing,:nonpositive_exponent
    beta<beta_limit-endpoint_tolerance||return nothing,:endpoint_dominated

    profile=_power_law_profile(log_x,scaled_y,beta,R)
    isfinite(profile.objective)||return nothing,:nonfinite_objective
    log_alpha=log(yscale)+log(profile.gamma)-beta*profile.anchor
    alpha=exp(log_alpha)
    isfinite(alpha)&&alpha>zero(R)||return nothing,:coefficient_unrepresentable
    predicted_y=yscale.*profile.predicted
    all(value->isfinite(value)&&value>zero(R),predicted_y)||
        return nothing,:nonfinite_prediction
    residuals=crossing_y.-predicted_y
    rms_residual=sqrt(sum(abs2,residuals)/R(point_count))
    relative_l2_residual=norm(residuals)/norm(crossing_y)
    maximum_relative_residual=maximum(abs.(residuals)./crossing_y)

    jacobian=hcat(predicted_y,
                  predicted_y.*centered_log_x)
    factorization=qr(jacobian,ColumnNorm())
    diagonal=abs.(diag(factorization.R))
    largest_diagonal=maximum(diagonal)
    rank_tolerance=R(max(size(jacobian)...))*eps(R)*largest_diagonal
    numerical_rank=count(value->value>rank_tolerance,diagonal)
    numerical_rank==2||return nothing,:rank_deficient
    qr_diagonal_condition_estimate=maximum(diagonal)/minimum(diagonal)

    log_fit_x_lower=max(log(first(crossing_x)),
        (log(yminimum)-log_alpha)/beta)
    log_fit_x_upper=min(log(last(crossing_x)),
        (log(ymaximum)-log_alpha)/beta)
    finite_fit_domain=isfinite(log_fit_x_lower)&&isfinite(log_fit_x_upper)&&
        log_fit_x_lower<=log_fit_x_upper
    fit_x=if finite_fit_domain
        collect(range(exp(log_fit_x_lower),exp(log_fit_x_upper);
                      length=sample_count))
    else
        R[]
    end
    fit_y=exp.(log_alpha .+ beta.*log.(fit_x))
    domain_tolerance=R(64)*eps(R)*max(one(R),abs(yminimum),abs(ymaximum))
    plottable=finite_fit_domain&&!isempty(fit_y)&&
        all(value->isfinite(value)&&value>zero(R)&&
        yminimum-domain_tolerance<=value<=ymaximum+domain_tolerance,fit_y)
    candidate=(;coefficients=(;alpha,beta),alpha,beta,log_alpha,
        initial_beta,predicted_y,residuals,rms_residual,
        relative_l2_residual,maximum_relative_residual,
        fit_x,fit_y,
        fit_x_span=isempty(fit_x) ? nothing : (first(fit_x),last(fit_x)),
        plottable,
        residual_degrees_of_freedom=point_count-2,
        numerical_rank,rank_tolerance,qr_diagonal_condition_estimate,
        objective=:positive_power_law_original_y_least_squares,
        optimizer=:profiled_golden_section,
        converged=optimized.converged,iterations=optimized.iterations,
        evaluations=evaluations[],beta_bracket=bracket,
        final_beta_bracket=optimized.final_bracket,
        beta_tolerance=optimized.beta_tolerance)
    candidate,plottable ? :available : :outside_sampled_vertical_domain
end

function _push_unique_root!(roots,value,::Type{R}) where R
    tolerance=R(64)*eps(R)
    for root in roots
        abs(root-value)<=tolerance*max(one(R),abs(root),abs(value))&&
            return roots
    end
    push!(roots,value)
end

function _level_differences(left_value,right_value,level,::Type{R}) where R
    left=left_value-level
    right=right_value-level
    isfinite(left)&&isfinite(right)&&return left,right
    # Finite values near opposite floating-point extremes can overflow when
    # the level is subtracted.  A common positive rescaling leaves the zero
    # and the linear crossing fraction unchanged.
    scale=max(abs(left_value),abs(right_value),abs(level))
    iszero(scale)&&return zero(R),zero(R)
    left=left_value/scale-level/scale
    right=right_value/scale-level/scale
    isfinite(left)&&isfinite(right)||throw(ArgumentError(
        "level subtraction produced nonfinite contour-edge values"))
    left,right
end

function _opposite_sign_fraction(left,right,::Type{R}) where R
    left_magnitude=abs(left)
    right_magnitude=abs(right)
    # Evaluate |left|/(|left|+|right|) without overflowing the sum.
    if left_magnitude>=right_magnitude
        return inv(one(R)+right_magnitude/left_magnitude)
    end
    ratio=left_magnitude/right_magnitude
    ratio/(one(R)+ratio)
end

function _slice_level_roots(coordinates,values,level,::Type{R},context) where R
    roots=R[]
    for index in 1:length(coordinates)-1
        left,right=_level_differences(
            values[index],values[index+1],level,R)
        if iszero(left)&&iszero(right)
            throw(ArgumentError(
                "$context contains an interval lying exactly on level $level"))
        elseif iszero(left)
            _push_unique_root!(roots,coordinates[index],R)
        elseif iszero(right)
            _push_unique_root!(roots,coordinates[index+1],R)
        elseif signbit(left)!=signbit(right)
            fraction=_opposite_sign_fraction(left,right,R)
            root=coordinates[index]+fraction*(coordinates[index+1]-
                                               coordinates[index])
            _push_unique_root!(roots,root,R)
        end
    end
    roots
end

function _push_unique_point!(crossing_x,crossing_y,x,y,::Type{R}) where R
    tolerance=R(64)*eps(R)
    for index in eachindex(crossing_x)
        same_x=abs(crossing_x[index]-x)<=
            tolerance*max(one(R),abs(crossing_x[index]),abs(x))
        same_y=abs(crossing_y[index]-y)<=
            tolerance*max(one(R),abs(crossing_y[index]),abs(y))
        same_x&&same_y&&return false
    end
    push!(crossing_x,x)
    push!(crossing_y,y)
    true
end

"""
    quadratic_level_contour_fit(
        x, y, values;
        level,
        samples=200,
        fallback_relative_l2_threshold=0.1,
        fallback_model=:general_quadratic,
    )

Extract a single-valued level contour from a rectangular grid and fit the
origin-constrained physical law `y = alpha*x^2`. `values[i,j]` corresponds to
`(x[i],y[j])`. Crossings are linearly interpolated in the physical `x` and
`y` coordinates along every grid line; duplicate grid-vertex crossings are
removed. More than one crossing on any sampled row or column is rejected
rather than silently selecting a contour branch.

The returned named tuple contains the crossing points, origin-constrained
`alpha`, fitted curve, ordinary least-squares residual diagnostics,
leave-one-out alpha range, and boundary-touch metadata. If the constrained
relative two-norm residual strictly exceeds
`fallback_relative_l2_threshold`, the function also attempts the empirical
model selected by `fallback_model`. The default `:general_quadratic` uses the
centered, scaled, pivoted-QR fit `y = a*x^2 + b*x + c`; at least four
crossings and numerical rank three are required. `:power_law` instead fits
the positive law `y = alpha*x^beta` by profiling out `alpha` and minimizing
the ordinary least-squares residual in the physical `y` coordinate; at least
three crossings and numerical rank two are required. The logarithmic fit is
used only to initialize the one-dimensional exponent search.

`general_quadratic` or `power_law` retains the corresponding candidate, and
`display_model` identifies whether the constrained guide, empirical
fallback, or raw contour alone should be plotted. Pass `nothing` as the
threshold to disable either fallback.

All candidates are finite-grid visualization guides, not statistical
confidence intervals or evidence for a scaling law. A fallback candidate
that leaves the positive sampled vertical domain is retained for diagnostics
but is not selected for logarithmic plotting.
"""
function quadratic_level_contour_fit(
        x::AbstractVector{<:Real},y::AbstractVector{<:Real},
        values::AbstractMatrix{<:Real};level::Real,samples::Integer=200,
        fallback_relative_l2_threshold::Union{Nothing,Real}=0.1,
        fallback_model::Symbol=:general_quadratic)
    length(x)>=2||throw(ArgumentError("the contour x grid needs at least two points"))
    length(y)>=2||throw(ArgumentError("the contour y grid needs at least two points"))
    size(values)==(length(x),length(y))||throw(DimensionMismatch(
        "contour values must have size (length(x), length(y))"))
    sample_count=try
        Int(samples)
    catch
        throw(ArgumentError("samples=$samples cannot be represented as an Int"))
    end
    sample_count>=2||throw(ArgumentError("samples must be at least two"))
    fallback_model in (:general_quadratic,:power_law)||throw(ArgumentError(
        "fallback_model must be :general_quadratic or :power_law"))

    R=float(promote_type(eltype(x),eltype(y),eltype(values),typeof(level)))
    isconcretetype(R)&&R<:AbstractFloat||throw(ArgumentError(
        "contour coordinates and values must promote to a concrete floating type"))
    xgrid=R.(x);ygrid=R.(y);field=Matrix{R}(values);target=R(level)
    all(isfinite,xgrid)&&all(isfinite,ygrid)&&all(isfinite,field)&&
        isfinite(target)||throw(ArgumentError(
            "contour coordinates, values, and level must be finite"))
    all(diff(xgrid).>zero(R))||throw(ArgumentError(
        "contour x coordinates must be strictly increasing"))
    all(diff(ygrid).>zero(R))||throw(ArgumentError(
        "contour y coordinates must be strictly increasing"))
    all(xgrid.>zero(R))||throw(ArgumentError(
        "the fit y=alpha*x^2 requires positive x coordinates"))
    all(ygrid.>zero(R))||throw(ArgumentError(
        "the fit and logarithmic plot require positive y coordinates"))
    fallback_threshold=if fallback_relative_l2_threshold===nothing
        nothing
    else
        value=R(fallback_relative_l2_threshold)
        isfinite(value)&&value>=zero(R)||throw(ArgumentError(
            "fallback_relative_l2_threshold must be finite and nonnegative"))
        value
    end

    crossing_x=R[];crossing_y=R[]
    horizontal_count=0
    for yindex in eachindex(ygrid)
        roots=_slice_level_roots(
            xgrid,view(field,:,yindex),target,R,
            "horizontal contour slice y=$(ygrid[yindex])")
        length(roots)<=1||throw(ArgumentError(
            "level $target has multiple crossings on horizontal slice y=$(ygrid[yindex]); one contour branch is ambiguous"))
        isempty(roots)&&continue
        horizontal_count+=1
        _push_unique_point!(crossing_x,crossing_y,
                            only(roots),ygrid[yindex],R)
    end
    vertical_count=0
    for xindex in eachindex(xgrid)
        roots=_slice_level_roots(
            ygrid,view(field,xindex,:),target,R,
            "vertical contour slice x=$(xgrid[xindex])")
        length(roots)<=1||throw(ArgumentError(
            "level $target has multiple crossings on vertical slice x=$(xgrid[xindex]); one contour branch is ambiguous"))
        isempty(roots)&&continue
        vertical_count+=1
        _push_unique_point!(crossing_x,crossing_y,
                            xgrid[xindex],only(roots),R)
    end
    length(crossing_x)>=2||throw(ArgumentError(
        "level $target has fewer than two distinct grid-edge crossings"))

    order=sortperm(crossing_x)
    crossing_x=crossing_x[order]
    crossing_y=crossing_y[order]
    squared_x=crossing_x.^2
    denominator=dot(squared_x,squared_x)
    isfinite(denominator)&&denominator>zero(R)||throw(ArgumentError(
        "quadratic contour fit is rank deficient"))
    alpha=dot(squared_x,crossing_y)/denominator
    isfinite(alpha)&&alpha>zero(R)||throw(ArgumentError(
        "quadratic contour fit produced nonpositive or nonfinite alpha=$alpha"))
    predicted_y=alpha.*squared_x
    residuals=crossing_y.-predicted_y
    point_count=length(crossing_x)
    rms_residual=sqrt(sum(abs2,residuals)/R(point_count))
    relative_l2_residual=norm(residuals)/norm(crossing_y)
    maximum_relative_residual=maximum(abs.(residuals)./crossing_y)
    pointwise_alpha=crossing_y./squared_x

    leave_one_out=R[]
    if point_count>2
        for omitted in eachindex(crossing_x)
            reduced_denominator=denominator-abs2(squared_x[omitted])
            reduced_denominator>zero(R)||continue
            reduced_numerator=dot(squared_x,crossing_y)-
                squared_x[omitted]*crossing_y[omitted]
            push!(leave_one_out,reduced_numerator/reduced_denominator)
        end
    end
    leave_one_out_range=isempty(leave_one_out) ? (alpha,alpha) :
        extrema(leave_one_out)

    # Do not let an imperfect least-squares guide enlarge the plotted domain.
    # This matters in particular when the fitted curve misses the lowest
    # crossing appreciably on a logarithmic y axis.
    fit_x_lower=max(first(crossing_x),sqrt(first(ygrid)/alpha))
    fit_x_upper=min(last(crossing_x),sqrt(last(ygrid)/alpha))
    fit_x_lower<=fit_x_upper||throw(ArgumentError(
        "the fitted quadratic does not intersect the sampled plotting domain"))
    fit_x=collect(range(fit_x_lower,fit_x_upper;length=sample_count))
    fit_y=alpha.*fit_x.^2

    fallback_triggered=fallback_threshold!==nothing&&
        relative_l2_residual>fallback_threshold
    general_quadratic=nothing
    power_law=nothing
    fallback_status=fallback_threshold===nothing ? :disabled : :not_triggered
    display_model=:origin_constrained
    if fallback_triggered
        if fallback_model===:general_quadratic
            general_quadratic,fallback_status=_general_quadratic_fit(
                crossing_x,crossing_y,first(ygrid),last(ygrid),sample_count,R)
            if general_quadratic===nothing
                display_model=:raw_contour_only
            elseif fallback_status===:available&&
                    general_quadratic.relative_l2_residual<=relative_l2_residual
                display_model=:general_quadratic
                fallback_status=:selected
            elseif fallback_status===:available
                display_model=:raw_contour_only
                fallback_status=:not_improved
            else
                display_model=:raw_contour_only
            end
        else
            power_law,fallback_status=_positive_power_law_fit(
                crossing_x,crossing_y,first(ygrid),last(ygrid),sample_count,R)
            if power_law===nothing
                display_model=:raw_contour_only
            elseif fallback_status===:available
                improvement=relative_l2_residual-
                    power_law.relative_l2_residual
                improvement_tolerance=R(64)*eps(R)*max(
                    one(R),relative_l2_residual,
                    power_law.relative_l2_residual)
                if improvement>improvement_tolerance
                    display_model=:power_law
                    fallback_status=:selected
                else
                    display_model=:raw_contour_only
                    fallback_status=:not_improved
                end
            else
                display_model=:raw_contour_only
            end
        end
    end
    tolerance=R(64)*eps(R)
    on_boundary(value,boundary)=abs(value-boundary)<=
        tolerance*max(one(R),abs(value),abs(boundary))
    boundary_sides=Symbol[]
    any(value->on_boundary(value,first(xgrid)),crossing_x)&&push!(boundary_sides,:left)
    any(value->on_boundary(value,last(xgrid)),crossing_x)&&push!(boundary_sides,:right)
    any(value->on_boundary(value,first(ygrid)),crossing_y)&&push!(boundary_sides,:bottom)
    any(value->on_boundary(value,last(ygrid)),crossing_y)&&push!(boundary_sides,:top)

    (;level=target,alpha,crossing_x,crossing_y,predicted_y,residuals,
      pointwise_alpha,fit_x,fit_y,point_count,horizontal_count,vertical_count,
      rms_residual,relative_l2_residual,maximum_relative_residual,
      leave_one_out_alpha=leave_one_out,
      leave_one_out_range,boundary_sides,
      fit_x_span=(fit_x_lower,fit_x_upper),
      fallback_relative_l2_threshold=fallback_threshold,
      fallback_metric=:relative_l2_residual,
      fallback_model,fallback_triggered,fallback_status,
      general_quadratic,power_law,display_model,
      touches_boundary=!isempty(boundary_sides),
      interpolation=:linear_physical_coordinates,
      objective=:origin_constrained_least_squares)
end

end
