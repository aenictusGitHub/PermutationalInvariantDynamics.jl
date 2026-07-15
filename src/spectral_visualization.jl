"""
    ComplexSpectrum

Reusable complex spectral data for Liouvillian eigenvalues or Floquet
multipliers/exponents. Values retain their input order and scalar type.
`classifications` and `tolerance` are presentation metadata only; no value is
clipped, shifted, or removed.
"""
struct ComplexSpectrum{V,R,C,Q,T,P,M}
    values::V
    kind::Symbol
    representation::Symbol
    classifications::Vector{Symbol}
    residuals::R
    converged::C
    complete::Q
    tolerance::T
    period::P
    metadata::M
end

"""Dependency-free complex-spectrum text/SVG rendering configuration."""
struct SpectrumVisualization{S,LX,LY}
    spectrum::S
    title::String
    width::Int
    height::Int
    xlimits::LX
    ylimits::LY
    show_legend::Bool
    show_indices::Bool
    marker_size::Float64
end

function _spectral_real_type(values)
    eltype(values)<:Number || throw(ArgumentError("spectrum values must have a numeric element type"))
    _real_float_type(eltype(values))
end

function _validate_spectral_tolerance_inputs(atol,rtol)
    atol isa Real || throw(ArgumentError("atol must be real"))
    atol>=0 && isfinite(atol) || throw(ArgumentError("atol must be nonnegative and finite"))
    rtol===nothing || rtol isa Real || throw(ArgumentError("rtol must be real or nothing"))
    rtol===nothing || (rtol>=0 && isfinite(rtol)) ||
        throw(ArgumentError("rtol must be nonnegative and finite"))
    nothing
end

function _spectral_tolerance(values,atol,rtol)
    R=_spectral_real_type(values)
    _validate_spectral_tolerance_inputs(atol,rtol)
    resolved_rtol=rtol===nothing ? sqrt(eps(R)) : rtol
    scale=maximum((R(abs(value)) for value in values);init=zero(R))
    isfinite(scale) || throw(ArgumentError("spectrum scale is not representable in $R"))
    scale=max(scale,one(R))
    converted_atol=convert(R,atol);converted_rtol=convert(R,resolved_rtol)
    isfinite(converted_atol) && isfinite(converted_rtol) ||
        throw(ArgumentError("classification tolerances are not representable in $R"))
    !iszero(atol)&&iszero(converted_atol) &&
        throw(ArgumentError("atol underflows in $R"))
    !iszero(resolved_rtol)&&iszero(converted_rtol) &&
        throw(ArgumentError("rtol underflows in $R"))
    tolerance=converted_atol+converted_rtol*scale
    isfinite(tolerance) || throw(ArgumentError("classification tolerance overflowed in $R"))
    tolerance
end

function _liouvillian_classification(value,tolerance)
    abs(value)<=tolerance && return :stationary
    real(value)>tolerance && return :unstable
    abs(real(value))<=tolerance && return :peripheral
    :decaying
end

function _floquet_classification(value,representation,tolerance)
    if representation===:multipliers
        abs(value-one(value))<=tolerance && return :fixed
        modulus=abs(value)
        modulus>one(modulus)+tolerance && return :unstable
        abs(modulus-one(modulus))<=tolerance && return :peripheral
        return :contracting
    end
    abs(value)<=tolerance && return :fixed
    real(value)>tolerance && return :unstable
    abs(real(value))<=tolerance && return :peripheral
    :contracting
end

function _normalize_spectral_residuals(residuals,count)
    residuals===nothing && return nothing
    residuals isa AbstractVector || throw(ArgumentError("residuals must be a vector or nothing"))
    length(residuals)==count || throw(DimensionMismatch("residuals and spectrum values have different lengths"))
    result=collect(residuals)
    for value in result
        value isa Real || throw(ArgumentError("spectral residuals must be real"))
        value>=0 && isfinite(value) ||
            throw(ArgumentError("spectral residuals must be nonnegative and finite"))
    end
    result
end

function _normalize_spectral_convergence(converged,count)
    converged===nothing && return nothing
    converged isa AbstractVector || throw(ArgumentError("converged must be a vector or nothing"))
    length(converged)==count || throw(DimensionMismatch("converged and spectrum values have different lengths"))
    all(value->value isa Bool,converged) || throw(ArgumentError("converged entries must be Boolean"))
    BitVector(converged)
end

function _resolve_spectral_scope(count,dimension,complete)
    resolved_dimension = if dimension===nothing
        nothing
    else
        dimension isa Integer || throw(ArgumentError("spectrum dimension must be an integer or nothing"))
        dimension>=count || throw(DimensionMismatch("spectrum dimension is smaller than the number of values"))
        Int(dimension)
    end
    resolved_complete = if complete===nothing
        resolved_dimension===nothing ? missing : count==resolved_dimension
    elseif complete===missing || complete isa Bool
        complete
    else
        throw(ArgumentError("complete must be true, false, missing, or nothing"))
    end
    resolved_complete===true && resolved_dimension!==nothing && count!=resolved_dimension &&
        throw(ArgumentError("a spectrum with fewer values than its dimension cannot be complete"))
    resolved_dimension,resolved_complete
end

function _complex_spectrum(values::AbstractVector,kind,representation;
                           residuals=nothing,converged=nothing,
                           complete=nothing,dimension=nothing,period=nothing,
                           atol=0,rtol=nothing,metadata=NamedTuple())
    isempty(values) && throw(ArgumentError("spectrum values cannot be empty"))
    eltype(values)<:Number || throw(ArgumentError("spectrum values must be numeric"))
    copied=collect(values)
    all(value->isfinite(real(value))&&isfinite(imag(value)),copied) ||
        throw(ArgumentError("spectrum values must be finite"))
    tolerance=_spectral_tolerance(copied,atol,rtol)
    R=typeof(tolerance);count=length(copied)
    normalized_residuals=_normalize_spectral_residuals(residuals,count)
    normalized_converged=_normalize_spectral_convergence(converged,count)
    resolved_dimension,resolved_complete=
        _resolve_spectral_scope(count,dimension,complete)
    classifications = kind===:liouvillian ?
        [_liouvillian_classification(value,tolerance) for value in copied] :
        [_floquet_classification(value,representation,tolerance) for value in copied]
    metadata isa NamedTuple || throw(ArgumentError("metadata must be a NamedTuple"))
    core=(dimension=resolved_dimension,partial=resolved_complete!==true,
          residuals=normalized_residuals,converged=normalized_converged)
    merged=merge(metadata,core)
    ComplexSpectrum(copied,kind,representation,classifications,
                    normalized_residuals,normalized_converged,
                    resolved_complete,tolerance,period,merged)
end

function _spectral_property(result,name)
    hasproperty(result,name) && return getproperty(result,name)
    if hasproperty(result,:info)
        info=getproperty(result,:info)
        hasproperty(info,name) && return getproperty(info,name)
    end
    nothing
end

function _liouvillian_result_data(result;complete=nothing,dimension=nothing,
                                  atol=0,rtol=nothing)
    hasproperty(result,:values) || throw(ArgumentError("spectral result must contain values"))
    values=getproperty(result,:values)
    residuals=_spectral_property(result,:residuals)
    converged=_spectral_property(result,:converged)
    inferred_dimension=dimension===nothing ? _spectral_property(result,:dimension) : dimension
    inferred_complete=complete===nothing ? _spectral_property(result,:complete) : complete
    metadata=(source_type=typeof(result),source=:precomputed_result)
    _complex_spectrum(values,:liouvillian,:eigenvalues;
        residuals,converged,complete=inferred_complete,
        dimension=inferred_dimension,atol,rtol,metadata)
end

"""
    liouvillian_spectrum_data(values_or_result; ...)
    liouvillian_spectrum_data(source; target=:largest_real, nev=nothing,
                              algorithm=:auto, classification_atol=0,
                              classification_rtol=nothing, ...)

Construct reusable Liouvillian eigenvalue data. Raw vectors and existing
spectral results are never recomputed. Source inputs delegate once to
`liouvillian_spectrum` without requesting eigenvectors. For source inputs,
`atol` and `rtol` are forwarded to the spectral solver; the distinct
`classification_atol` and `classification_rtol` keywords control only marker
classification.
"""
function liouvillian_spectrum_data(values::AbstractVector;
                                   residuals=nothing,converged=nothing,
                                   complete=nothing,dimension=nothing,
                                   atol=0,rtol=nothing)
    _complex_spectrum(values,:liouvillian,:eigenvalues;
        residuals,converged,complete,dimension,atol,rtol,
        metadata=(source_type=typeof(values),source=:values))
end

liouvillian_spectrum_data(result::SpectrumResult;kwargs...)=
    _liouvillian_result_data(result;kwargs...)
liouvillian_spectrum_data(result::NamedTuple;kwargs...)=
    _liouvillian_result_data(result;kwargs...)

function liouvillian_spectrum_data(source;target=:largest_real,
                                   nev::Union{Nothing,Integer}=nothing,
                                   algorithm=:auto,method=nothing,
                                   classification_atol=0,
                                   classification_rtol=nothing,
                                   kwargs...)
    _validate_spectral_tolerance_inputs(classification_atol,classification_rtol)
    n=pi_dimension(source);n>0||throw(ArgumentError("Liouvillian dimension must be positive"))
    requested=nev===nothing ? min(n,40) : Int(nev)
    0<requested<=n || throw(ArgumentError("nev must lie between 1 and the Liouvillian dimension"))
    if method!==nothing
        algorithm===:auto || algorithm isa AutoAlgorithm ||
            throw(ArgumentError("specify only one of algorithm and method"))
        algorithm=method
    end
    options=(;kwargs...)
    reserved=(:vectors,:return_info,:sortby,:rev,:spectrum_kwargs)
    conflict=findfirst(name->haskey(options,name),reserved)
    conflict===nothing || throw(ArgumentError(
        "$(reserved[conflict]) is controlled by liouvillian_spectrum_data and cannot be overridden"))
    haskey(options,:rtol) && options.rtol===nothing && throw(ArgumentError(
        "solver rtol must be a nonnegative finite real number"))
    _validate_spectral_tolerance_inputs(
        get(options,:atol,0),get(options,:rtol,nothing))
    values=liouvillian_spectrum(source;target,nev=requested,algorithm,
                                vectors=false,return_info=false,options...)
    _complex_spectrum(values,:liouvillian,:eigenvalues;
        complete=length(values)==n,dimension=n,
        atol=classification_atol,rtol=classification_rtol,
        metadata=(source_type=typeof(source),source=:computed,
                  target,algorithm,requested_modes=requested))
end

function _validated_floquet_period(period;required::Bool)
    period===nothing && (required ? throw(ArgumentError(
        "a positive period is required for Floquet exponent conversion")) : return nothing)
    period isa Real || throw(ArgumentError("period must be real"))
    period>0 && isfinite(period) || throw(ArgumentError("period must be positive and finite"))
    period
end

function _floquet_input_kind(input,input_representation)
    if input_representation!==nothing
        input===:multipliers || input===input_representation ||
            throw(ArgumentError("input and input_representation disagree"))
        input=input_representation
    end
    input in (:multipliers,:exponents) ||
        throw(ArgumentError("input must be :multipliers or :exponents"))
    input
end

function _convert_floquet_values(values,input,representation,period)
    validated_period=_validate_floquet_conversion(input,representation,period)
    if input===representation
        return collect(values),validated_period
    elseif input===:multipliers
        any(iszero,values) && throw(ArgumentError(
            "a zero Floquet multiplier has no finite exponent; use representation=:multipliers"))
        return log.(complex.(values))./validated_period,validated_period
    end
    exp.(validated_period.*values),validated_period
end

function _validate_floquet_conversion(input,representation,period)
    input in (:multipliers,:exponents) ||
        throw(ArgumentError("input must be :multipliers or :exponents"))
    representation in (:multipliers,:exponents) ||
        throw(ArgumentError("representation must be :multipliers or :exponents"))
    required=representation===:exponents || input===:exponents
    _validated_floquet_period(period;required)
end

"""
    floquet_spectrum_data(F; period=nothing, representation=:multipliers)
    floquet_spectrum_data(source, period; steps=256, ...)
    floquet_spectrum_data(values; input=:multipliers, ...)

Construct reusable Floquet multiplier or exponent data. Multiplier-to-exponent
conversion uses Julia's principal complex logarithm. Supplied exponents remain
unchanged and are not asserted to lie in the principal zone. A matrix is always
interpreted as an already computed one-period propagator. A model or matrix-free
source with positional `period` is integrated exactly once.
"""
function floquet_spectrum_data(values::AbstractVector;
                               input=:multipliers,input_representation=nothing,
                               representation=:multipliers,period=nothing,
                               residuals=nothing,converged=nothing,
                               complete=nothing,dimension=nothing,
                               atol=0,rtol=nothing)
    resolved_input=_floquet_input_kind(input,input_representation)
    _validate_spectral_tolerance_inputs(atol,rtol)
    converted,validated_period=
        _convert_floquet_values(values,resolved_input,representation,period)
    branch=representation===:exponents && resolved_input===:multipliers ?
           :principal : nothing
    _complex_spectrum(converted,:floquet,representation;
        residuals,converged,complete,dimension,period=validated_period,
        atol,rtol,metadata=(source_type=typeof(values),source=:values,
                            input_representation=resolved_input,branch,
                            residual_representation=resolved_input))
end

function floquet_spectrum_data(F::AbstractMatrix;
                               period=nothing,representation=:multipliers,
                               atol=0,rtol=nothing)
    Base.require_one_based_indexing(F)
    size(F,1)==size(F,2)||throw(DimensionMismatch("Floquet propagator must be square"))
    size(F,1)>0||throw(ArgumentError("Floquet propagator cannot be empty"))
    _validate_spectral_tolerance_inputs(atol,rtol)
    _validate_floquet_conversion(:multipliers,representation,period)
    values=floquet_multipliers(F)
    converted,validated_period=
        _convert_floquet_values(values,:multipliers,representation,period)
    branch=representation===:exponents ? :principal : nothing
    _complex_spectrum(converted,:floquet,representation;
        complete=true,dimension=size(F,1),period=validated_period,atol,rtol,
        metadata=(source_type=typeof(F),source=:propagator,
                  input_representation=:multipliers,branch,
                  residual_representation=nothing))
end

floquet_spectrum_data(F::AbstractMatrix,period::Real;kwargs...)=
    floquet_spectrum_data(F;period,kwargs...)

const _FloquetSpectrumSource=Union{PIModel,CompiledPIModel,LiouvillianPlan,
                                   MatrixFreeLiouvillian}

function floquet_spectrum_data(source::_FloquetSpectrumSource,period::Real;
                               steps::Integer=256,t0::Real=0,parameters=nothing,
                               representation=:multipliers,atol=0,rtol=nothing)
    _validate_spectral_tolerance_inputs(atol,rtol)
    _validate_floquet_conversion(:multipliers,representation,period)
    validated_period=_validated_floquet_period(period;required=true)
    F=floquet_propagator(source,validated_period;steps,t0,parameters)
    values=floquet_multipliers(F)
    converted,_=_convert_floquet_values(
        values,:multipliers,representation,validated_period)
    branch=representation===:exponents ? :principal : nothing
    _complex_spectrum(converted,:floquet,representation;
        complete=true,dimension=size(F,1),period=validated_period,atol,rtol,
        metadata=(source_type=typeof(source),source=:integrated,
                  input_representation=:multipliers,steps=Int(steps),t0,
                  parameters,branch,residual_representation=nothing))
end

function _spectrum_scope_label(spectrum::ComplexSpectrum)
    spectrum.complete===true && return "complete"
    spectrum.complete===false && return "partial"
    "scope unknown (treated as partial)"
end

function show(io::IO,spectrum::ComplexSpectrum)
    print(io,"ComplexSpectrum(kind=",spectrum.kind,
          ", representation=",spectrum.representation,
          ", modes=",length(spectrum.values),
          ", ",_spectrum_scope_label(spectrum),")")
end

function show(io::IO,::MIME"text/plain",spectrum::ComplexSpectrum)
    show(io,spectrum)
    print(io,"\n  tolerance: ",spectrum.tolerance)
    spectrum.period===nothing || print(io,"\n  period: ",spectrum.period)
    spectrum.residuals===nothing || print(io,"\n  residual diagnostics: available")
    spectrum.converged===nothing || print(io,"\n  convergence diagnostics: available")
end

function _validate_spectrum_limits(limits,name)
    limits===nothing && return nothing
    (limits isa Tuple || limits isa AbstractVector) && length(limits)==2 ||
        throw(ArgumentError("$name must be a two-element tuple/vector or nothing"))
    first_limit,last_limit=limits
    first_limit isa Real && last_limit isa Real ||
        throw(ArgumentError("$name entries must be real"))
    first_value=Float64(first_limit);last_value=Float64(last_limit)
    isfinite(first_value) && isfinite(last_value) ||
        throw(ArgumentError("$name entries must be finite and representable as Float64"))
    first_value<last_value || throw(ArgumentError("$name must be strictly increasing"))
    isfinite(last_value-first_value) ||
        throw(ArgumentError("$name span is too large for SVG rendering"))
    (first_value,last_value)
end

function _validate_marker_size(marker_size)
    marker_size isa Real || throw(ArgumentError("marker_size must be real"))
    value=Float64(marker_size)
    value>0 && isfinite(value) ||
        throw(ArgumentError("marker_size must be positive and finite"))
    value
end

function _validated_spectrum_visual_options(title,width,height,xlimits,ylimits,
                                            marker_size)
    width>=320 || throw(ArgumentError("width must be at least 320"))
    height>=280 || throw(ArgumentError("height must be at least 280"))
    (title=title===nothing ? nothing : String(title),
     width=Int(width),height=Int(height),
     xlimits=_validate_spectrum_limits(xlimits,"xlimits"),
     ylimits=_validate_spectrum_limits(ylimits,"ylimits"),
     marker_size=_validate_marker_size(marker_size))
end

function _default_spectrum_title(spectrum::ComplexSpectrum)
    spectrum.kind===:liouvillian && return "Liouvillian spectrum"
    spectrum.representation===:multipliers && return "Floquet multipliers"
    _spectrum_is_principal(spectrum) ? "Floquet exponents (principal branch)" :
                                       "Floquet exponents"
end

_spectrum_is_principal(spectrum::ComplexSpectrum)=
    hasproperty(spectrum.metadata,:branch) && spectrum.metadata.branch===:principal

function _validate_spectrum_render_data(spectrum::ComplexSpectrum)
    spectrum.values isa AbstractVector ||
        throw(ArgumentError("spectrum values must be a vector"))
    count=length(spectrum.values)
    count>0 || throw(ArgumentError("spectrum values cannot be empty"))
    length(spectrum.classifications)==count || throw(DimensionMismatch(
        "classifications and spectrum values have different lengths"))
    allowed=spectrum.kind===:liouvillian ?
        (:stationary,:peripheral,:decaying,:unstable) :
        (:fixed,:peripheral,:contracting,:unstable)
    all(classification->classification in allowed,spectrum.classifications) ||
        throw(ArgumentError("spectrum contains an invalid classification"))
    if spectrum.residuals!==nothing
        spectrum.residuals isa AbstractVector ||
            throw(ArgumentError("residuals must be a vector or nothing"))
        length(spectrum.residuals)==count || throw(DimensionMismatch(
            "residuals and spectrum values have different lengths"))
        all(value->value isa Real&&value>=0&&isfinite(value),spectrum.residuals) ||
            throw(ArgumentError("spectral residuals must be nonnegative finite real values"))
    end
    if spectrum.converged!==nothing
        spectrum.converged isa AbstractVector ||
            throw(ArgumentError("converged must be a vector or nothing"))
        length(spectrum.converged)==count || throw(DimensionMismatch(
            "converged and spectrum values have different lengths"))
        all(value->value isa Bool,spectrum.converged) ||
            throw(ArgumentError("converged entries must be Boolean"))
    end
    spectrum.representation===:exponents &&
        _validated_floquet_period(spectrum.period;required=true)
    nothing
end

"""
    visualize_spectrum(spectrum::ComplexSpectrum; title=nothing,
                       width=760, height=560, xlimits=nothing,
                       ylimits=nothing, show_legend=true,
                       show_indices=false, marker_size=5)

Create a dependency-free SVG visualization configuration. Rendering reuses
the stored numerical data and never invokes an eigensolver or propagator.
"""
function visualize_spectrum(spectrum::ComplexSpectrum;title=nothing,
                            width::Integer=760,height::Integer=560,
                            xlimits=nothing,ylimits=nothing,
                            show_legend::Bool=true,
                            show_indices::Bool=false,marker_size=5)
    options=_validated_spectrum_visual_options(
        title,width,height,xlimits,ylimits,marker_size)
    spectrum.kind in (:liouvillian,:floquet) ||
        throw(ArgumentError("unsupported spectrum kind $(spectrum.kind)"))
    spectrum.representation in (:eigenvalues,:multipliers,:exponents) ||
        throw(ArgumentError("unsupported spectrum representation $(spectrum.representation)"))
    spectrum.kind===:liouvillian && spectrum.representation!==:eigenvalues &&
        throw(ArgumentError("Liouvillian spectra must use :eigenvalues"))
    spectrum.kind===:floquet && spectrum.representation===:eigenvalues &&
        throw(ArgumentError("Floquet spectra must use :multipliers or :exponents"))
    _validate_spectrum_render_data(spectrum)
    resolved_title=options.title===nothing ? _default_spectrum_title(spectrum) :
                                            options.title
    SpectrumVisualization(spectrum,resolved_title,options.width,options.height,
        options.xlimits,options.ylimits,show_legend,show_indices,
        options.marker_size)
end

"""
    visualize_liouvillian_spectrum(spectrum::ComplexSpectrum; options...)
    visualize_liouvillian_spectrum(source; options..., solver_options...)

Render existing Liouvillian spectral data, or compute selected modes once and
render them. Presentation options are the same as for `visualize_spectrum`.
"""
function visualize_liouvillian_spectrum(spectrum::ComplexSpectrum;kwargs...)
    spectrum.kind===:liouvillian || throw(ArgumentError(
        "visualize_liouvillian_spectrum requires Liouvillian spectral data"))
    visualize_spectrum(spectrum;kwargs...)
end

function visualize_liouvillian_spectrum(source;title=nothing,
                                        width::Integer=760,
                                        height::Integer=560,xlimits=nothing,
                                        ylimits=nothing,
                                        show_legend::Bool=true,
                                        show_indices::Bool=false,
                                        marker_size=5,
                                        spectrum_kwargs=NamedTuple(),kwargs...)
    spectrum_kwargs isa NamedTuple ||
        throw(ArgumentError("spectrum_kwargs must be a NamedTuple"))
    visual_options=_validated_spectrum_visual_options(
        title,width,height,xlimits,ylimits,marker_size)
    data=liouvillian_spectrum_data(source;spectrum_kwargs...,kwargs...)
    visualize_spectrum(data;title=visual_options.title,width=visual_options.width,
        height=visual_options.height,xlimits=visual_options.xlimits,
        ylimits=visual_options.ylimits,show_legend,show_indices,
        marker_size=visual_options.marker_size)
end

"""
    visualize_floquet_spectrum(spectrum::ComplexSpectrum; options...)
    visualize_floquet_spectrum(F; period=nothing, options...)
    visualize_floquet_spectrum(source, period; options...)

Render existing Floquet data, an already computed one-period propagator, or a
periodic PI source. A source is propagated and diagonalized exactly once.
"""
function visualize_floquet_spectrum(spectrum::ComplexSpectrum;kwargs...)
    spectrum.kind===:floquet || throw(ArgumentError(
        "visualize_floquet_spectrum requires Floquet spectral data"))
    visualize_spectrum(spectrum;kwargs...)
end

function visualize_floquet_spectrum(source,args...;title=nothing,
                                    width::Integer=760,
                                    height::Integer=560,xlimits=nothing,
                                    ylimits=nothing,
                                    show_legend::Bool=true,
                                    show_indices::Bool=false,
                                    marker_size=5,
                                    spectrum_kwargs=NamedTuple(),kwargs...)
    spectrum_kwargs isa NamedTuple ||
        throw(ArgumentError("spectrum_kwargs must be a NamedTuple"))
    visual_options=_validated_spectrum_visual_options(
        title,width,height,xlimits,ylimits,marker_size)
    data=floquet_spectrum_data(source,args...;spectrum_kwargs...,kwargs...)
    visualize_spectrum(data;title=visual_options.title,width=visual_options.width,
        height=visual_options.height,xlimits=visual_options.xlimits,
        ylimits=visual_options.ylimits,show_legend,show_indices,
        marker_size=visual_options.marker_size)
end

function show(io::IO,visualization::SpectrumVisualization)
    print(io,"SpectrumVisualization(kind=",visualization.spectrum.kind,
          ", representation=",visualization.spectrum.representation,
          ", size=",visualization.width,"×",visualization.height,")")
end

function show(io::IO,::MIME"text/plain",visualization::SpectrumVisualization)
    show(io,visualization)
    print(io,"\n  title: ",repr(visualization.title),
          "\n  modes: ",length(visualization.spectrum.values),
          " (",_spectrum_scope_label(visualization.spectrum),")",
          "\n  use display(...) for SVG output")
end

function _spectrum_float_coordinates(values)
    count=length(values);xs=Vector{Float64}(undef,count)
    ys=Vector{Float64}(undef,count)
    for index in eachindex(values)
        x=Float64(real(values[index]));y=Float64(imag(values[index]))
        isfinite(x) && isfinite(y) || throw(ArgumentError(
            "spectrum values are not representable as finite SVG coordinates"))
        xs[index]=x;ys[index]=y
    end
    xs,ys
end

function _padded_spectrum_limits(lower,upper)
    span=upper-lower
    isfinite(span) || throw(ArgumentError(
        "spectrum extent is too large for SVG rendering; pass finite view limits"))
    if iszero(span)
        padding=max(abs(lower)*0.12,1.0)
        padded=(lower-padding,upper+padding)
        all(isfinite,padded) || throw(ArgumentError(
            "spectrum extent is too large for SVG rendering; pass finite view limits"))
        return padded
    end
    padding=0.08span
    padded=(lower-padding,upper+padding)
    all(isfinite,padded) || throw(ArgumentError(
        "spectrum extent is too large for SVG rendering; pass finite view limits"))
    padded
end

function _auto_spectrum_xlimits(spectrum,xs)
    xmin=min(minimum(xs),0.0);xmax=max(maximum(xs),0.0)
    if spectrum.representation===:multipliers
        xmin=min(xmin,-1.0);xmax=max(xmax,1.0)
    end
    _padded_spectrum_limits(xmin,xmax)
end

function _auto_spectrum_ylimits(spectrum,ys)
    ymin=min(minimum(ys),0.0);ymax=max(maximum(ys),0.0)
    if spectrum.representation===:multipliers
        ymin=min(ymin,-1.0);ymax=max(ymax,1.0)
    elseif spectrum.representation===:exponents
        branch=Float64(pi/spectrum.period)
        isfinite(branch) || throw(ArgumentError(
            "the Floquet branch boundary is not representable in SVG coordinates"))
        ymin=min(ymin,-branch);ymax=max(ymax,branch)
    end
    _padded_spectrum_limits(ymin,ymax)
end

function _spectrum_plot_geometry(visualization,xlimits,ylimits)
    left=78.0;right=visualization.show_legend ? 178.0 : 34.0
    top=70.0;bottom=76.0
    available_width=visualization.width-left-right
    available_height=visualization.height-top-bottom
    available_width>0 && available_height>0 ||
        throw(ArgumentError("visualization dimensions leave no plotting area"))
    xspan=xlimits[2]-xlimits[1];yspan=ylimits[2]-ylimits[1]
    if visualization.spectrum.representation===:multipliers
        scale=min(available_width/xspan,available_height/yspan)
        scale>0 && isfinite(scale) || throw(ArgumentError(
            "spectrum view limits are too narrow for SVG rendering"))
        plot_width=scale*xspan;plot_height=scale*yspan
        x0=left+(available_width-plot_width)/2
        y0=top+(available_height-plot_height)/2
        return (x0=x0,y0=y0,width=plot_width,height=plot_height,
                xscale=scale,yscale=scale)
    end
    xscale=available_width/xspan;yscale=available_height/yspan
    xscale>0 && yscale>0 && isfinite(xscale) && isfinite(yscale) ||
        throw(ArgumentError("spectrum view limits are too narrow for SVG rendering"))
    (x0=left,y0=top,width=available_width,height=available_height,
     xscale,yscale)
end

_spectrum_x_coordinate(geometry,xlimits,x)=
    geometry.x0+(x-xlimits[1])*geometry.xscale
_spectrum_y_coordinate(geometry,ylimits,y)=
    geometry.y0+geometry.height-(y-ylimits[1])*geometry.yscale

function _spectrum_number_label(value)
    normalized=iszero(value) ? 0.0 : value
    string(round(normalized;sigdigits=6))
end

function _spectrum_complex_label(value)
    real_part=_spectrum_number_label(Float64(real(value)))
    imaginary=Float64(imag(value));imaginary_part=_spectrum_number_label(abs(imaginary))
    real_part*(signbit(imaginary) ? " - " : " + ")*imaginary_part*"im"
end

function _spectrum_color(classification)
    classification in (:stationary,:fixed) && return "#7c3aed"
    classification===:peripheral && return "#0284c7"
    classification in (:decaying,:contracting) && return "#16a34a"
    classification===:unstable && return "#dc2626"
    "#475569"
end

function _spectrum_axis_labels(spectrum)
    spectrum.kind===:liouvillian && return ("Re(λ)","Im(λ)")
    spectrum.representation===:multipliers && return ("Re(μ)","Im(μ)")
    ("Re(ξ)","Im(ξ)")
end

function _spectrum_legend_entries(spectrum)
    spectrum.kind===:liouvillian && return (
        (:stationary,"stationary"),(:peripheral,"peripheral"),
        (:decaying,"decaying"),(:unstable,"unstable"))
    ((:fixed,"fixed"),(:peripheral,"peripheral"),
     (:contracting,"contracting"),(:unstable,"unstable"))
end

function _spectrum_point_tooltip(spectrum,index)
    text="mode $index: "*_spectrum_complex_label(spectrum.values[index])*
         " · "*String(spectrum.classifications[index])
    spectrum.residuals===nothing ||
        (text*=" · residual "*string(spectrum.residuals[index]))
    spectrum.converged===nothing ||
        (text*=" · "*(spectrum.converged[index] ? "converged" : "not converged"))
    text
end

function _draw_spectrum_grid(io,geometry,xlimits,ylimits)
    for value in range(xlimits[1],xlimits[2];length=5)
        x=_spectrum_x_coordinate(geometry,xlimits,value)
        print(io,"<line x1=\"",x,"\" y1=\"",geometry.y0,"\" x2=\"",x,
              "\" y2=\"",geometry.y0+geometry.height,
              "\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>")
        print(io,"<text x=\"",x,"\" y=\"",geometry.y0+geometry.height+19,
              "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#475569\">",
              _svg_escape(_spectrum_number_label(value)),"</text>")
    end
    for value in range(ylimits[1],ylimits[2];length=5)
        y=_spectrum_y_coordinate(geometry,ylimits,value)
        print(io,"<line x1=\"",geometry.x0,"\" y1=\"",y,"\" x2=\"",
              geometry.x0+geometry.width,"\" y2=\"",y,
              "\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>")
        print(io,"<text x=\"",geometry.x0-9,"\" y=\"",y+4,
              "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#475569\">",
              _svg_escape(_spectrum_number_label(value)),"</text>")
    end
end

function _draw_spectrum_references(io,visualization,geometry,xlimits,ylimits)
    spectrum=visualization.spectrum
    if ylimits[1]<=0<=ylimits[2]
        y=_spectrum_y_coordinate(geometry,ylimits,0.0)
        print(io,"<line class=\"real-axis\" x1=\"",geometry.x0,"\" y1=\"",y,
              "\" x2=\"",geometry.x0+geometry.width,"\" y2=\"",y,
              "\" stroke=\"#64748b\" stroke-width=\"1.2\"/>")
    end
    if xlimits[1]<=0<=xlimits[2]
        x=_spectrum_x_coordinate(geometry,xlimits,0.0)
        print(io,"<line class=\"stability-boundary\" x1=\"",x,"\" y1=\"",
              geometry.y0,"\" x2=\"",x,"\" y2=\"",geometry.y0+geometry.height,
              "\" stroke=\"#334155\" stroke-width=\"1.5\" stroke-dasharray=\"6 4\"><title>",
              spectrum.representation===:multipliers ? "imaginary axis" :
                                                       "stability boundary Re = 0",
              "</title></line>")
    end
    if spectrum.representation===:multipliers
        cx=_spectrum_x_coordinate(geometry,xlimits,0.0)
        cy=_spectrum_y_coordinate(geometry,ylimits,0.0)
        radius=geometry.xscale
        print(io,"<circle class=\"unit-circle\" cx=\"",cx,"\" cy=\"",cy,
              "\" r=\"",radius,"\" fill=\"none\" stroke=\"#0f172a\" stroke-width=\"1.5\" stroke-dasharray=\"7 4\"><title>unit circle: Floquet stability boundary</title></circle>")
        if xlimits[1]<=1<=xlimits[2] && ylimits[1]<=0<=ylimits[2]
            x=_spectrum_x_coordinate(geometry,xlimits,1.0)
            print(io,"<path class=\"fixed-point-reference\" d=\"M ",x-5," ",cy,
                  " L ",x+5," ",cy," M ",x," ",cy-5," L ",x," ",cy+5,
                  "\" stroke=\"#7c3aed\" stroke-width=\"1.5\"><title>fixed multiplier μ = 1</title></path>")
        end
    elseif spectrum.representation===:exponents
        branch=Float64(pi/spectrum.period)
        boundary_name=_spectrum_is_principal(spectrum) ?
            "principal branch" : "principal-zone boundary"
        for (value,label) in ((-branch,"-π/T"),(branch,"+π/T"))
            ylimits[1]<=value<=ylimits[2] || continue
            y=_spectrum_y_coordinate(geometry,ylimits,value)
            print(io,"<line class=\"floquet-zone-boundary\" x1=\"",geometry.x0,
                  "\" y1=\"",y,"\" x2=\"",geometry.x0+geometry.width,
                  "\" y2=\"",y,"\" stroke=\"#f59e0b\" stroke-width=\"1.2\" stroke-dasharray=\"4 4\"><title>",boundary_name," ",label,
                  "</title></line>")
        end
    end
end

function show(io::IO,::MIME"image/svg+xml",visualization::SpectrumVisualization)
    spectrum=visualization.spectrum;_validate_spectrum_render_data(spectrum)
    xs,ys=_spectrum_float_coordinates(spectrum.values)
    xlimits=visualization.xlimits===nothing ?
        _auto_spectrum_xlimits(spectrum,xs) : visualization.xlimits
    ylimits=visualization.ylimits===nothing ?
        _auto_spectrum_ylimits(spectrum,ys) : visualization.ylimits
    geometry=_spectrum_plot_geometry(visualization,xlimits,ylimits)
    width=visualization.width;height=visualization.height
    labels=_spectrum_axis_labels(spectrum)
    visible=[xlimits[1]<=xs[i]<=xlimits[2] && ylimits[1]<=ys[i]<=ylimits[2]
             for i in eachindex(xs)]
    shown=count(identity,visible);hidden=length(visible)-shown
    print(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"",width,
          "\" height=\"",height,"\" viewBox=\"0 0 ",width," ",height,
          "\" role=\"img\" aria-label=\"",_svg_escape(visualization.title),"\">")
    print(io,"<title>",_svg_escape(visualization.title),"</title>")
    print(io,"<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>")
    print(io,"<text x=\"",width/2,"\" y=\"29\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"18\" font-weight=\"600\">",
          _svg_escape(visualization.title),"</text>")
    subtitle=spectrum.kind===:liouvillian ?
        "continuous-time modes · Re < 0 is stable" :
        (spectrum.representation===:multipliers ?
         "one-period multipliers · |μ| < 1 is stable" :
         ((_spectrum_is_principal(spectrum) ? "principal " : "")*
          "Floquet exponents · Re < 0 is stable"))
    print(io,"<text x=\"",width/2,"\" y=\"49\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#475569\">",
          _svg_escape(subtitle),"</text>")
    _draw_spectrum_grid(io,geometry,xlimits,ylimits)
    print(io,"<rect x=\"",geometry.x0,"\" y=\"",geometry.y0,"\" width=\"",
          geometry.width,"\" height=\"",geometry.height,
          "\" fill=\"none\" stroke=\"#94a3b8\" stroke-width=\"1\"/>")
    print(io,"<svg class=\"spectrum-plot-clip\" x=\"",geometry.x0,"\" y=\"",
          geometry.y0,"\" width=\"",geometry.width,"\" height=\"",geometry.height,
          "\" viewBox=\"",geometry.x0," ",geometry.y0," ",geometry.width," ",
          geometry.height,"\" overflow=\"hidden\">")
    _draw_spectrum_references(io,visualization,geometry,xlimits,ylimits)
    for index in eachindex(xs)
        visible[index] || continue
        x=_spectrum_x_coordinate(geometry,xlimits,xs[index])
        y=_spectrum_y_coordinate(geometry,ylimits,ys[index])
        classification=spectrum.classifications[index]
        unconverged=spectrum.converged!==nothing && !spectrum.converged[index]
        print(io,"<circle class=\"spectrum-point ",_svg_escape(String(classification)),
              unconverged ? " unconverged" : "",
              "\" cx=\"",x,"\" cy=\"",y,"\" r=\"",visualization.marker_size,
              "\" fill=\"",_spectrum_color(classification),
              "\" fill-opacity=\"0.82\" stroke=\"#0f172a\" stroke-width=\"",
              unconverged ? "1.8" : "0.8","\"",
              unconverged ? " stroke-dasharray=\"2 2\"" : "","><title>",
              _svg_escape(_spectrum_point_tooltip(spectrum,index)),"</title></circle>")
        if visualization.show_indices
            print(io,"<text x=\"",x+visualization.marker_size+2,"\" y=\"",y-3,
                  "\" font-family=\"monospace\" font-size=\"9\" fill=\"#0f172a\">",
                  index,"</text>")
        end
    end
    print(io,"</svg>")
    print(io,"<text x=\"",geometry.x0+geometry.width/2,"\" y=\"",height-38,
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">",
          labels[1],"</text>")
    axis_x=18.0;axis_y=geometry.y0+geometry.height/2
    print(io,"<text x=\"",axis_x,"\" y=\"",axis_y,
          "\" text-anchor=\"middle\" transform=\"rotate(-90 ",axis_x," ",axis_y,
          ")\" font-family=\"sans-serif\" font-size=\"12\">",labels[2],"</text>")
    if visualization.show_legend
        legend_x=width-158.0;legend_y=geometry.y0+8
        print(io,"<text x=\"",legend_x,"\" y=\"",legend_y,
              "\" font-family=\"sans-serif\" font-size=\"12\" font-weight=\"600\">classification</text>")
        for (offset,(classification,label)) in enumerate(_spectrum_legend_entries(spectrum))
            y=legend_y+20offset
            print(io,"<circle cx=\"",legend_x+6,"\" cy=\"",y-4,
                  "\" r=\"4\" fill=\"",_spectrum_color(classification),"\"/>",
                  "<text x=\"",legend_x+17,"\" y=\"",y,
                  "\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#334155\">",
                  label,"</text>")
        end
        spectrum.converged===nothing || print(io,"<text x=\"",legend_x,
            "\" y=\"",legend_y+104,
            "\" font-family=\"sans-serif\" font-size=\"9\" fill=\"#475569\">dashed outline: unconverged</text>")
    end
    footer="$(_spectrum_scope_label(spectrum)) · shown $shown/$(length(visible)) modes"
    hidden==0 || (footer*=" · $hidden outside viewport")
    print(io,"<text x=\"",width/2,"\" y=\"",height-14,
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#475569\">",
          _svg_escape(footer),"</text></svg>")
end

"""Write a spectrum SVG and return the path as a `String`."""
function save_spectrum_visualization(path::AbstractString,
                                     visualization::SpectrumVisualization)
    svg=sprint(show,MIME"image/svg+xml"(),visualization)
    open(path,"w") do io
        write(io,svg)
    end
    String(path)
end

function save_spectrum_visualization(path::AbstractString,
                                     spectrum::ComplexSpectrum;kwargs...)
    save_spectrum_visualization(path,visualize_spectrum(spectrum;kwargs...))
end

"""
    DensitySpectrumVisualization

Dependency-free configuration for a multiplicity-compressed density-operator
spectrum plot. `spectrum` is the exact named tuple returned by
`pi_density_spectrum`; its values and exact `BigInt` multiplicities are never
expanded or reordered by the visualization layer.
"""
struct DensitySpectrumVisualization{S,Y,T}
    spectrum::S
    title::String
    width::Int
    height::Int
    ylimits::Y
    show_legend::Bool
    show_indices::Bool
    show_degeneracies::Bool
    marker_size::Float64
    presentation_tolerance::T
end

const _DENSITY_SPECTRUM_FIELDS=
    (:values,:degeneracies,:sectors,:sector_indices,:total_dimension)

function _validate_density_spectrum_data(spectrum::NamedTuple)
    for name in _DENSITY_SPECTRUM_FIELDS
        haskey(spectrum,name) || throw(ArgumentError(
            "compressed density spectrum is missing required field :$name"))
    end
    values=spectrum.values
    degeneracies=spectrum.degeneracies
    sectors=spectrum.sectors
    indices=spectrum.sector_indices
    values isa AbstractVector || throw(ArgumentError(
        "density-spectrum values must be an AbstractVector"))
    degeneracies isa AbstractVector || throw(ArgumentError(
        "density-spectrum degeneracies must be an AbstractVector"))
    sectors isa AbstractVector || throw(ArgumentError(
        "density-spectrum sectors must be an AbstractVector"))
    indices isa AbstractVector || throw(ArgumentError(
        "density-spectrum sector_indices must be an AbstractVector"))
    isempty(values) && throw(ArgumentError("density spectrum cannot be empty"))
    axes(degeneracies)==axes(values) || throw(DimensionMismatch(
        "degeneracies and density-spectrum values have different axes"))
    axes(sectors)==axes(values) || throw(DimensionMismatch(
        "sectors and density-spectrum values have different axes"))
    axes(indices)==axes(values) || throw(DimensionMismatch(
        "sector_indices and density-spectrum values have different axes"))

    eltype(values)<:AbstractFloat || throw(ArgumentError(
        "density-spectrum values must have an AbstractFloat element type"))
    all(isfinite,values) || throw(ArgumentError(
        "density-spectrum values must be finite"))
    eltype(degeneracies)<:Integer && eltype(degeneracies)!==Bool ||
        throw(ArgumentError("density-spectrum degeneracies must be exact integers"))
    all(g->g>0,degeneracies) || throw(ArgumentError(
        "density-spectrum degeneracies must be positive"))
    eltype(sectors)<:Partition || throw(ArgumentError(
        "density-spectrum sectors must contain Partition values"))
    eltype(indices)<:Integer && eltype(indices)!==Bool ||
        throw(ArgumentError("density-spectrum sector_indices must be integers"))
    all(i->i>0,indices) || throw(ArgumentError(
        "density-spectrum sector_indices must be positive"))
    spectrum.total_dimension isa BigInt || throw(ArgumentError(
        "density-spectrum total_dimension must be an exact BigInt"))
    spectrum.total_dimension>0 || throw(ArgumentError(
        "density-spectrum total_dimension must be positive"))

    first_sector=first(sectors)
    particle_number=weight(first_sector)
    local_dimension=length(first_sector)
    all(p->length(p)==local_dimension&&weight(p)==particle_number,sectors) ||
        throw(ArgumentError(
            "all density-spectrum sectors must partition the same N at the same local dimension"))

    # A compressed result contains one mode for every eigenvalue of each
    # retained physical Schur block.  Validate the representation-theoretic
    # metadata as well as its scalar container types so a tampered named tuple
    # cannot mislabel a point or its Hilbert-space multiplicity.
    seen=Set{Tuple{Partition,BigInt}}()
    counts=Dict{Partition,BigInt}()
    total=big(0)
    for k in eachindex(values)
        p=sectors[k]
        expected=symmetric_group_dimension(p)
        BigInt(degeneracies[k])==expected || throw(ArgumentError(
            "degeneracy for sector $p must equal its symmetric-group dimension $expected"))
        local_index=BigInt(indices[k])
        local_index<=unitary_group_dimension(p) || throw(ArgumentError(
            "sector index $(indices[k]) exceeds the Schur-block dimension for $p"))
        pair=(p,local_index)
        pair in seen && throw(ArgumentError(
            "duplicate density-spectrum sector/index pair ($p, $(indices[k]))"))
        push!(seen,pair)
        counts[p]=get(counts,p,big(0))+1
        total+=expected
    end
    for (p,count) in counts
        expected=unitary_group_dimension(p)
        count==expected || throw(ArgumentError(
            "sector $p has $count compressed modes but its Schur block has dimension $expected"))
    end
    total==spectrum.total_dimension || throw(ArgumentError(
        "total_dimension does not equal the exact sum of compressed degeneracies"))
    nothing
end

function _validate_density_render_options(title,width,height,ylimits,
                                          show_legend,show_indices,
                                          show_degeneracies,marker_size,scale,
                                          presentation_atol,
                                          presentation_rtol)
    width isa Integer && width>=320 && width<=typemax(Int) ||
        throw(ArgumentError("width must be an integer of at least 320"))
    height isa Integer && height>=280 && height<=typemax(Int) ||
        throw(ArgumentError("height must be an integer of at least 280"))
    show_legend isa Bool || throw(ArgumentError("show_legend must be Boolean"))
    show_indices isa Bool || throw(ArgumentError("show_indices must be Boolean"))
    show_degeneracies isa Bool || throw(ArgumentError(
        "show_degeneracies must be Boolean"))
    scale===:linear || throw(ArgumentError(
        "density spectra support only scale=:linear so raw eigenvalues remain visible"))
    resolved_title=title===nothing ? "Density-operator spectrum by Schur sector" :
                                    String(title)
    _svg_escape(resolved_title)
    _validate_spectral_tolerance_inputs(presentation_atol,presentation_rtol)
    (title=resolved_title,width=Int(width),height=Int(height),
     ylimits=_validate_spectrum_limits(ylimits,"ylimits"),
     show_legend,show_indices,show_degeneracies,
     marker_size=_validate_marker_size(marker_size),
     presentation_atol,presentation_rtol)
end

function _validate_density_spectrum_kwargs(options::NamedTuple)
    allowed=(:sortby,:rev,:atol)
    for name in keys(options)
        name in allowed || throw(ArgumentError(
            "unsupported spectrum_kwargs entry :$name; density visualization always uses the compressed spectrum"))
    end
    if haskey(options,:sortby)
        options.sortby in (:value,:magnitude,:none) || throw(ArgumentError(
            "spectrum_kwargs.sortby must be :value, :magnitude, or :none"))
    end
    haskey(options,:rev) && !(options.rev isa Bool) &&
        throw(ArgumentError("spectrum_kwargs.rev must be Boolean"))
    if haskey(options,:atol)
        value=options.atol
        value isa Real && value>=0 && isfinite(value) || throw(ArgumentError(
            "spectrum_kwargs.atol must be a nonnegative finite real number"))
    end
    nothing
end

function _density_spectrum_visualization(spectrum::NamedTuple,options)
    _validate_density_spectrum_data(spectrum)
    tolerance=_spectral_tolerance(
        spectrum.values,options.presentation_atol,options.presentation_rtol)
    DensitySpectrumVisualization(
        spectrum,options.title,options.width,options.height,options.ylimits,
        options.show_legend,options.show_indices,options.show_degeneracies,
        options.marker_size,tolerance)
end

"""
    visualize_density_spectrum(rho::PIState; spectrum_kwargs=NamedTuple(), options...)
    visualize_density_spectrum(spectrum::NamedTuple; options...)

Create a dependency-free, linear-scale SVG visualization of a density
operator's multiplicity-compressed Schur spectrum. The horizontal coordinate
is compressed eigenmode rank and the vertical coordinate is the raw physical
eigenvalue. Points are colored by Schur sector. Exact degeneracy and the
within-sector eigenvalue index are retained in each tooltip; the exponentially
large repeated eigenvalue list is never constructed.

`presentation_atol` and `presentation_rtol` only decide when a negative value
gets a red outline. They never clip or replace an eigenvalue. Presentation
options and `spectrum_kwargs` are validated before a `PIState` is diagonalized.
Pass an existing compressed named tuple to render it without recomputation.
"""
function visualize_density_spectrum(spectrum::NamedTuple;
                                    spectrum_kwargs=NamedTuple(),title=nothing,
                                    width=760,height=560,ylimits=nothing,
                                    show_legend=true,show_indices=false,
                                    show_degeneracies=true,
                                    marker_size=5,scale=:linear,
                                    presentation_atol=0,
                                    presentation_rtol=nothing)
    spectrum_kwargs isa NamedTuple || throw(ArgumentError(
        "spectrum_kwargs must be a NamedTuple"))
    isempty(spectrum_kwargs) || throw(ArgumentError(
        "spectrum_kwargs cannot be used with precomputed density-spectrum data"))
    options=_validate_density_render_options(
        title,width,height,ylimits,show_legend,show_indices,show_degeneracies,
        marker_size,scale,presentation_atol,presentation_rtol)
    _density_spectrum_visualization(spectrum,options)
end

function visualize_density_spectrum(rho::PIState;
                                    spectrum_kwargs=NamedTuple(),title=nothing,
                                    width=760,height=560,ylimits=nothing,
                                    show_legend=true,show_indices=false,
                                    show_degeneracies=true,
                                    marker_size=5,scale=:linear,
                                    presentation_atol=0,
                                    presentation_rtol=nothing)
    spectrum_kwargs isa NamedTuple || throw(ArgumentError(
        "spectrum_kwargs must be a NamedTuple"))
    options=_validate_density_render_options(
        title,width,height,ylimits,show_legend,show_indices,show_degeneracies,
        marker_size,scale,presentation_atol,presentation_rtol)
    _validate_density_spectrum_kwargs(spectrum_kwargs)
    spectrum=pi_density_spectrum(rho;expanded=false,spectrum_kwargs...)
    _density_spectrum_visualization(spectrum,options)
end

function show(io::IO,visualization::DensitySpectrumVisualization)
    print(io,"DensitySpectrumVisualization(modes=",
          length(visualization.spectrum.values),", retained_dimension=",
          visualization.spectrum.total_dimension,", size=",
          visualization.width,"×",visualization.height,")")
end

function show(io::IO,::MIME"text/plain",
              visualization::DensitySpectrumVisualization)
    show(io,visualization)
    print(io,"\n  title: ",repr(visualization.title),
          "\n  multiplicities: compressed",
          "\n  presentation tolerance: ",
          visualization.presentation_tolerance,
          "\n  use display(...) for SVG output")
end

const _DENSITY_SECTOR_PALETTE=(
    "#2563eb","#16a34a","#9333ea","#ea580c","#0891b2","#ca8a04",
    "#db2777","#4f46e5","#059669","#c2410c","#7c3aed","#0284c7",
    "#65a30d","#be123c","#0f766e","#a16207")

function _density_sector_order(sectors)
    result=Partition[]
    for sector in sectors
        sector in result || push!(result,sector)
    end
    result
end

_density_sector_color(index::Integer)=
    _DENSITY_SECTOR_PALETTE[mod1(index,length(_DENSITY_SECTOR_PALETTE))]

function _density_float_values(values)
    result=Vector{Float64}(undef,length(values))
    for (rank,index) in enumerate(eachindex(values))
        converted=try
            Float64(values[index])
        catch error
            error isa InterruptException && rethrow()
            throw(ArgumentError(
                "density-spectrum values are not representable as finite SVG coordinates"))
        end
        isfinite(converted) || throw(ArgumentError(
            "density-spectrum values are not representable as finite SVG coordinates"))
        result[rank]=converted
    end
    result
end

function _density_rank_ticks(count)
    count==1 && return [1]
    unique!(sort!(round.(Int,range(1,count;length=min(count,6)))))
end

function _density_plot_geometry(visualization,ylimits)
    left=82.0
    right=visualization.show_legend ? 190.0 : 34.0
    top=70.0
    bottom=86.0
    width=visualization.width-left-right
    height=visualization.height-top-bottom
    width>0 && height>0 || throw(ArgumentError(
        "visualization dimensions leave no density-spectrum plotting area"))
    count=length(visualization.spectrum.values)
    xlimits=count==1 ? (0.5,1.5) : (0.5,count+0.5)
    xscale=width/(xlimits[2]-xlimits[1])
    yscale=height/(ylimits[2]-ylimits[1])
    isfinite(xscale) && isfinite(yscale) && xscale>0 && yscale>0 ||
        throw(ArgumentError(
            "density-spectrum view limits are too narrow for SVG rendering"))
    (x0=left,y0=top,width,height,xlimits,xscale,yscale)
end

_density_x_coordinate(geometry,rank)=
    geometry.x0+(rank-geometry.xlimits[1])*geometry.xscale
_density_y_coordinate(geometry,ylimits,value)=
    geometry.y0+geometry.height-(value-ylimits[1])*geometry.yscale

function _density_point_tooltip(spectrum,index,rank)
    "compressed mode $rank: eigenvalue $(spectrum.values[index]) · Schur sector "*
    _schur_label(spectrum.sectors[index])*" · sector index "*
    string(spectrum.sector_indices[index])*" · exact degeneracy "*
    string(spectrum.degeneracies[index])
end

function _draw_density_grid(io,geometry,ylimits,count)
    for rank in _density_rank_ticks(count)
        x=_density_x_coordinate(geometry,rank)
        print(io,"<line x1=\"",x,"\" y1=\"",geometry.y0,"\" x2=\"",x,
              "\" y2=\"",geometry.y0+geometry.height,
              "\" stroke=\"#f1f5f9\" stroke-width=\"1\"/>",
              "<text x=\"",x,"\" y=\"",geometry.y0+geometry.height+19,
              "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#475569\">",
              rank,"</text>")
    end
    for value in range(ylimits[1],ylimits[2];length=5)
        y=_density_y_coordinate(geometry,ylimits,value)
        print(io,"<line x1=\"",geometry.x0,"\" y1=\"",y,"\" x2=\"",
              geometry.x0+geometry.width,"\" y2=\"",y,
              "\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>",
              "<text x=\"",geometry.x0-9,"\" y=\"",y+4,
              "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#475569\">",
              _svg_escape(_spectrum_number_label(value)),"</text>")
    end
end

function show(io::IO,::MIME"image/svg+xml",
              visualization::DensitySpectrumVisualization)
    spectrum=visualization.spectrum
    _validate_density_spectrum_data(spectrum)
    values=_density_float_values(spectrum.values)
    ylimits=visualization.ylimits===nothing ?
        _padded_spectrum_limits(min(minimum(values),0.0),
                                max(maximum(values),0.0)) :
        visualization.ylimits
    geometry=_density_plot_geometry(visualization,ylimits)
    sector_order=_density_sector_order(spectrum.sectors)
    sector_numbers=Dict{Partition,Int}(
        sector=>index for (index,sector) in enumerate(sector_order))
    visible=[ylimits[1]<=value<=ylimits[2] for value in values]
    shown=count(identity,visible)
    negative=[spectrum.values[index] < -visualization.presentation_tolerance
              for index in eachindex(spectrum.values)]
    print(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"",
          visualization.width,"\" height=\"",visualization.height,
          "\" viewBox=\"0 0 ",visualization.width," ",visualization.height,
          "\" role=\"img\" aria-label=\"",_svg_escape(visualization.title),"\">",
          "<title>",_svg_escape(visualization.title),"</title>",
          "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
          "<text x=\"",visualization.width/2,
          "\" y=\"29\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"18\" font-weight=\"600\">",
          _svg_escape(visualization.title),"</text>",
          "<text x=\"",visualization.width/2,
          "\" y=\"49\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#475569\">multiplicity-compressed Schur eigenmodes · linear scale</text>")
    _draw_density_grid(io,geometry,ylimits,length(values))
    print(io,"<rect x=\"",geometry.x0,"\" y=\"",geometry.y0,
          "\" width=\"",geometry.width,"\" height=\"",geometry.height,
          "\" fill=\"none\" stroke=\"#94a3b8\" stroke-width=\"1\"/>",
          "<svg class=\"density-spectrum-plot-clip\" x=\"",geometry.x0,
          "\" y=\"",geometry.y0,"\" width=\"",geometry.width,
          "\" height=\"",geometry.height,"\" viewBox=\"",geometry.x0," ",
          geometry.y0," ",geometry.width," ",geometry.height,
          "\" overflow=\"hidden\">")
    if ylimits[1]<=0<=ylimits[2]
        yzero=_density_y_coordinate(geometry,ylimits,0.0)
        print(io,"<line class=\"zero-reference\" x1=\"",geometry.x0,
              "\" y1=\"",yzero,"\" x2=\"",geometry.x0+geometry.width,
              "\" y2=\"",yzero,
              "\" stroke=\"#334155\" stroke-width=\"1.5\" stroke-dasharray=\"6 4\"><title>zero eigenvalue reference</title></line>")
    end
    for (rank,index) in enumerate(eachindex(spectrum.values))
        visible[rank] || continue
        x=_density_x_coordinate(geometry,rank)
        y=_density_y_coordinate(geometry,ylimits,values[rank])
        sector_number=sector_numbers[spectrum.sectors[index]]
        isnegative=negative[rank]
        print(io,"<circle class=\"density-spectrum-point sector-",sector_number,
              isnegative ? " negative" : "",
              "\" cx=\"",x,"\" cy=\"",y,"\" r=\"",
              visualization.marker_size,"\" fill=\"",
              _density_sector_color(sector_number),
              "\" fill-opacity=\"0.84\" stroke=\"",
              isnegative ? "#dc2626" : "#0f172a","\" stroke-width=\"",
              isnegative ? "2.2" : "0.8","\"><title>",
              _svg_escape(_density_point_tooltip(spectrum,index,rank)),
              "</title></circle>")
        visualization.show_indices && print(io,"<text x=\"",
            x+visualization.marker_size+2,"\" y=\"",y-3,
            "\" font-family=\"monospace\" font-size=\"9\" fill=\"#0f172a\">",
            rank,"</text>")
        if visualization.show_degeneracies && spectrum.degeneracies[index]>1
            print(io,"<text class=\"density-degeneracy-label\" x=\"",
                  x+visualization.marker_size+2,"\" y=\"",
                  y+(visualization.show_indices ? 10 : 3),
                  "\" font-family=\"monospace\" font-size=\"9\" fill=\"#475569\">×",
                  spectrum.degeneracies[index],"</text>")
        end
    end
    print(io,"</svg>",
          "<text x=\"",geometry.x0+geometry.width/2,"\" y=\"",
          visualization.height-43,
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">compressed eigenmode rank</text>")
    axis_x=18.0
    axis_y=geometry.y0+geometry.height/2
    print(io,"<text x=\"",axis_x,"\" y=\"",axis_y,
          "\" text-anchor=\"middle\" transform=\"rotate(-90 ",axis_x," ",
          axis_y,
          ")\" font-family=\"sans-serif\" font-size=\"12\">raw eigenvalue</text>")
    if visualization.show_legend
        legend_x=visualization.width-172.0
        legend_y=geometry.y0+8
        print(io,"<text x=\"",legend_x,"\" y=\"",legend_y,
              "\" font-family=\"sans-serif\" font-size=\"12\" font-weight=\"600\">Schur sector</text>")
        available=max(1,floor(Int,(geometry.height-54)/19))
        displayed=min(length(sector_order),available)
        for number in 1:displayed
            y=legend_y+19number
            print(io,"<circle cx=\"",legend_x+6,"\" cy=\"",y-4,
                  "\" r=\"4\" fill=\"",_density_sector_color(number),"\"/>",
                  "<text x=\"",legend_x+17,"\" y=\"",y,
                  "\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#334155\">",
                  _svg_escape(_schur_label(sector_order[number])),"</text>")
        end
        if displayed<length(sector_order)
            y=legend_y+19(displayed+1)
            print(io,"<text x=\"",legend_x,"\" y=\"",y,
                  "\" font-family=\"sans-serif\" font-size=\"9\" fill=\"#475569\">+",
                  length(sector_order)-displayed," more sectors</text>")
        end
        print(io,"<text x=\"",legend_x,"\" y=\"",
              geometry.y0+geometry.height-8,
              "\" font-family=\"sans-serif\" font-size=\"9\" fill=\"#dc2626\">red outline: value &lt; -",
              _svg_escape(string(visualization.presentation_tolerance)),
              "</text>")
    end
    footer="$(length(values)) compressed modes · retained Hilbert dimension: $(spectrum.total_dimension)"
    shown==length(values) || (footer*=" · $(length(values)-shown) outside viewport")
    print(io,"<text x=\"",visualization.width/2,"\" y=\"",
          visualization.height-14,
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\" fill=\"#475569\">",
          _svg_escape(footer),"</text></svg>")
end

"""Write a multiplicity-compressed density-spectrum SVG and return its path."""
function save_density_spectrum_visualization(
        path::AbstractString,visualization::DensitySpectrumVisualization)
    svg=sprint(show,MIME"image/svg+xml"(),visualization)
    open(path,"w") do io
        write(io,svg)
    end
    String(path)
end

function save_density_spectrum_visualization(path::AbstractString,
                                             spectrum::NamedTuple;kwargs...)
    save_density_spectrum_visualization(
        path,visualize_density_spectrum(spectrum;kwargs...))
end

function save_density_spectrum_visualization(path::AbstractString,
                                             rho::PIState;kwargs...)
    save_density_spectrum_visualization(
        path,visualize_density_spectrum(rho;kwargs...))
end
