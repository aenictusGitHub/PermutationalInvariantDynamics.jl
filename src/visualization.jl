"""
    SchurBlockStructure

Numerical Schur-sector structure of a PI operator or superoperator.  For an
operator only diagonal entries of `weights` are used.  For a superoperator,
row `i`, column `j` is the norm of the map from input sector `j` to output
sector `i`.  `active` applies the requested absolute threshold without
modifying `weights` or the source object.
"""
struct SchurBlockStructure{D,B<:PIBasis{D},R,M}
    basis::B
    kind::Symbol
    sectors::Tuple{Vararg{Partition{D}}}
    block_dimensions::Tuple{Vararg{Int}}
    coordinate_dimensions::Tuple{Vararg{Int}}
    weights::Matrix{R}
    active::BitMatrix
    metric::Symbol
    representation::Symbol
    threshold::R
    metadata::M
end

"""
    SchurBlockVisualization

Dependency-free text/SVG rendering configuration for a Schur structure.
`show_young_diagrams=true` displays the partition shape attached to every
Schur-sector label.  These are unfilled Young diagrams: PI operators sum over
the symmetric-group multiplicity labels and therefore do not select a unique
standard Young tableau.
"""
struct SchurBlockVisualization{S}
    structure::S
    scale::Symbol
    normalize::Symbol
    show_values::Bool
    show_young_diagrams::Bool
    title::String
    width::Int
    height::Int
end

function _schur_dimensions(basis::PIBasis)
    block=Tuple(length(patterns) for patterns in basis.patterns)
    block,Tuple(n*n for n in block)
end

function _validate_schur_threshold(threshold)
    threshold isa Real || throw(ArgumentError("threshold must be a real number"))
    threshold>=0 || throw(ArgumentError("threshold must be nonnegative"))
    isfinite(threshold) || throw(ArgumentError("threshold must be finite"))
    threshold
end

function _schur_threshold(threshold,::Type{R}) where R
    _validate_schur_threshold(threshold)
    value=convert(R,threshold)
    isfinite(value) || throw(ArgumentError("threshold is not representable in $R"))
    !iszero(threshold)&&iszero(value) && throw(ArgumentError(
        "threshold underflows and is not representable in $R"))
    value
end

function _schur_structure(basis,kind,weights,metric,representation,threshold,metadata)
    isempty(basis.sectors) && throw(ArgumentError(
        "Schur-block visualization requires a PI basis with at least one retained sector"))
    all(isfinite,weights) || throw(ArgumentError("Schur-block weights must be finite"))
    block,coordinate=_schur_dimensions(basis)
    R=eltype(weights);cutoff=_schur_threshold(threshold,R)
    SchurBlockStructure(basis,kind,Tuple(basis.sectors),block,coordinate,
        weights,BitMatrix(weights .> cutoff),metric,representation,cutoff,metadata)
end

function _operator_representation(representation)
    resolved=representation===nothing ? :physical : representation
    resolved in (:physical,:coefficient) ||
        throw(ArgumentError("operator representation must be :physical or :coefficient"))
    resolved
end

function _schur_trace_norm(C,sector)
    try
        sum(svdvals(Matrix(C)))
    catch error
        error isa MethodError || rethrow()
        throw(ArgumentError(":trace_norm in sector $sector requires a scalar " *
            "type supported by LinearAlgebra.svdvals; got $(eltype(C)). " *
            "Use :frobenius or convert the operator explicitly."))
    end
end

function _scaled_physical_schur_metric(C,sector,metric::Symbol,::Type{R}) where
        R<:AbstractFloat
    maximum_exponent=typemin(Int);nonzero_entry=false
    for value in C,component in (real(value),imag(value))
        converted=R(component)
        isfinite(converted)||throw(ArgumentError(
            "$metric metric in sector $sector requires finite coefficients"))
        iszero(converted)&&continue
        nonzero_entry=true
        _,exponent=frexp(converted)
        maximum_exponent=max(maximum_exponent,exponent)
    end
    nonzero_entry||return zero(R)

    scaled=Matrix{Complex{R}}(undef,size(C))
    @inbounds for index in eachindex(scaled,C)
        value=C[index]
        scaled[index]=complex(ldexp(R(real(value)),-maximum_exponent),
                              ldexp(R(imag(value)),-maximum_exponent))
    end
    aggregate=metric===:frobenius ? norm(scaled) :
        _schur_trace_norm(scaled,sector)
    isfinite(aggregate)||throw(ArgumentError(
        "$metric metric in sector $sector is outside the finite range of $R; " *
        "use a wider scalar type"))

    multiplicity=symmetric_group_dimension(sector)
    if maximum_exponent>=0
        numerator_value=big(1)<<(2*maximum_exponent)
        denominator_value=multiplicity
    else
        numerator_value=one(BigInt)
        denominator_value=multiplicity<<(-2*maximum_exponent)
    end
    scale=_prepare_exact_scale(R,numerator_value,denominator_value,Val(true);
        context="physical $metric metric in sector $sector")
    _apply_prepared_exact_scale(aggregate,scale;
        context="physical $metric metric in sector $sector")
end

function _physical_schur_metric(C,sector,metric::Symbol,::Type{R}) where
        R<:AbstractFloat
    aggregate=metric===:frobenius ? norm(C) : _schur_trace_norm(C,sector)
    try
        _divide_by_schur_multiplicity_scale(aggregate,R,sector)
    catch error
        error isa ArgumentError||rethrow()
        _scaled_physical_schur_metric(C,sector,metric,R)
    end
end

function _superoperator_representation(representation)
    representation===nothing && return :coefficient
    representation===:coefficient || throw(ArgumentError(
        "superoperator blocks are expressed in orthonormal PI coefficient coordinates; representation must be :coefficient"))
    :coefficient
end

"""
    schur_block_structure(A::Union{PIState,PIOperator};
                          metric=:frobenius, representation=nothing,
                          threshold=0)

Measure each retained Schur block without constructing a full `d^N` operator.
Supported metrics are `:frobenius` and `:trace_norm`; `PIState` additionally
supports `:population`.  The default representation is `:physical`.
"""
function schur_block_structure(A::AbstractPIOperator;basis=nothing,
                               metric=:frobenius,representation=nothing,
                               threshold=0,time=nothing,parameters=nothing)
    basis===nothing || basis===A.basis ||
        throw(ArgumentError("operator and supplied PI basis are incompatible"))
    time===nothing || throw(ArgumentError("time is not used for PI operator blocks"))
    parameters===nothing || throw(ArgumentError("parameters are not used for PI operator blocks"))
    metric in (:frobenius,:trace_norm,:population) ||
        throw(ArgumentError("operator metric must be :frobenius, :trace_norm, or :population"))
    metric===:population && !(A isa PIState) &&
        throw(ArgumentError(":population is defined only for PIState"))
    resolved=metric===:population ? :physical : _operator_representation(representation)
    metric===:population && representation!==nothing && representation!==:physical &&
        throw(ArgumentError(":population uses the physical Schur representation"))
    R=_real_float_type(eltype(A.data));cutoff=_schur_threshold(threshold,R)
    count=length(A.basis.sectors)
    weights=zeros(R,count,count)
    for (i,sector) in pairs(A.basis.sectors)
        C=coefficient_block(A,sector)
        value = if metric===:population
            population=sector_population(A,sector)
            tolerance=sqrt(eps(R))*max(abs(population),one(R))
            abs(imag(population))<=tolerance || throw(ArgumentError(
                "sector $sector population is not real within tolerance: $population"))
            real(population)>=0 || throw(ArgumentError(
                "sector $sector has a negative population: $(real(population))"))
            real(population)
        elseif metric===:frobenius
            resolved===:physical ?
                _physical_schur_metric(C,sector,:frobenius,R) : norm(C)
        else
            resolved===:physical ?
                _physical_schur_metric(C,sector,:trace_norm,R) :
                _schur_trace_norm(C,sector)
        end
        weights[i,i]=R(value)
    end
    kind=A isa PIState ? :state : :operator
    metadata=(N=A.basis.N,d=A.basis.d,pi_dimension=length(A.basis),
              orientation=:diagonal,source_type=typeof(A))
    _schur_structure(A.basis,kind,weights,metric,resolved,cutoff,metadata)
end

function _coordinate_sectors(basis::PIBasis)
    result=Vector{Int}(undef,length(basis))
    for sector in eachindex(basis.sectors)
        fill!(view(result,basis.offsets[sector]:basis.offsets[sector+1]-1),sector)
    end
    result
end

function _matrix_block_weights(matrix::AbstractMatrix,basis::PIBasis,metric)
    Base.require_one_based_indexing(matrix)
    R=_real_float_type(eltype(matrix));count=length(basis.sectors)
    weights=zeros(R,count,count);labels=_coordinate_sectors(basis)
    sparse_matrix=matrix isa SparseMatrixCSC ? matrix :
                  (issparse(matrix) ? sparse(matrix) : nothing)
    if sparse_matrix!==nothing
        rows=rowvals(sparse_matrix);values=nonzeros(sparse_matrix)
        for column in axes(sparse_matrix,2)
            input=labels[column]
            for pointer in nzrange(sparse_matrix,column)
                output=labels[rows[pointer]];value=R(abs(values[pointer]))
                if metric===:frobenius
                    weights[output,input]=hypot(weights[output,input],value)
                else
                    weights[output,input]=max(weights[output,input],value)
                end
            end
        end
    else
        @inbounds for column in axes(matrix,2),row in axes(matrix,1)
            value=R(abs(matrix[row,column]));output=labels[row];input=labels[column]
            if metric===:frobenius
                weights[output,input]=hypot(weights[output,input],value)
            else
                weights[output,input]=max(weights[output,input],value)
            end
        end
    end
    weights
end

_visualization_basis(matrix::AbstractMatrix)=nothing
_visualization_basis(model::PIModel)=model.basis
_visualization_basis(plan::LiouvillianPlan)=plan.basis
_visualization_basis(compiled::CompiledPIModel)=compiled.model.basis
_visualization_basis(L::MatrixFreeLiouvillian)=L.plan===nothing ? nothing : L.plan.basis
_visualization_basis(A::AdjointMatrixFreeLiouvillian)=_visualization_basis(A.parent)
_visualization_basis(P::MatrixFreeSymmetryProjector)=P.basis

function _resolve_visualization_basis(source,basis)
    owned=_visualization_basis(source)
    if basis===nothing
        owned===nothing && throw(ArgumentError(
            "this superoperator does not carry PI basis metadata; pass the matching PIBasis explicitly"))
        return owned
    end
    basis isa PIBasis || throw(ArgumentError("basis must be a PIBasis"))
    owned===nothing || owned===basis ||
        throw(ArgumentError("superoperator and supplied PI basis are incompatible"))
    basis
end

function _visualization_source(source::PIModel)
    LiouvillianPlan(source)
end
_visualization_source(source::LiouvillianPlan)=source
_visualization_source(source::CompiledPIModel)=
    source.backend===:sparse ? source.operator : source.plan
_visualization_source(source::MatrixFreeLiouvillian)=source
_visualization_source(source::AdjointMatrixFreeLiouvillian)=source
_visualization_source(source::MatrixFreeSymmetryProjector)=source
_visualization_source(source::AbstractMatrix)=source

_visualization_autonomous(source::AbstractMatrix)=true
_visualization_autonomous(::MatrixFreeSymmetryProjector)=true
_visualization_autonomous(source)=isautonomous(source)

function _visualization_at(source::LiouvillianPlan,time,parameters)
    source.fallback_model===nothing && return source
    frozen_model=PIModel(source.basis,
        (freeze_term(term,time,parameters) for term in source.fallback_model.terms))
    LiouvillianPlan(frozen_model)
end
function _visualization_at(source::MatrixFreeLiouvillian,time,parameters)
    source.plan===nothing ? source : _visualization_at(source.plan,time,parameters)
end
_visualization_at(source,time,parameters)=source

_visualization_workspace(::AbstractMatrix)=nothing
_visualization_workspace(plan::LiouvillianPlan)=LiouvillianWorkspace(plan)
_visualization_workspace(L::MatrixFreeLiouvillian)=
    L.plan===nothing ? nothing : LiouvillianWorkspace(L.plan)
_visualization_workspace(A::AdjointMatrixFreeLiouvillian)=
    A.parent.plan===nothing ? nothing : LiouvillianWorkspace(A.parent.plan)
_visualization_workspace(P::MatrixFreeSymmetryProjector)=
    SymmetryProjectorWorkspace(P)

function _visualization_apply!(output,source::LiouvillianPlan,input,time,parameters,work)
    apply!(output,source,input,time,parameters,work)
end
function _visualization_apply!(output,source::MatrixFreeLiouvillian,input,time,parameters,work)
    work===nothing ? source.action!(output,input,time,parameters) :
                     apply!(output,source,input,time,parameters,work)
end
function _visualization_apply!(output,source::AdjointMatrixFreeLiouvillian,input,time,parameters,work)
    work===nothing ? mul!(output,source,input) :
                     apply_adjoint!(output,source.parent,input,time,parameters,work)
end
function _visualization_apply!(output,source::MatrixFreeSymmetryProjector,
                               input,time,parameters,work)
    apply!(output,source,input,work)
end

function _matrixfree_block_weights(source,basis,metric,time,parameters)
    T=eltype(source);R=_real_float_type(T);count=length(basis.sectors)
    weights=zeros(R,count,count);labels=_coordinate_sectors(basis)
    input=zeros(T,length(basis));output=similar(input)
    work=_visualization_workspace(source)
    for column in eachindex(input)
        input[column]=one(T)
        _visualization_apply!(output,source,input,time,parameters,work)
        input[column]=zero(T)
        input_sector=labels[column]
        @inbounds for row in eachindex(output)
            value=R(abs(output[row]));output_sector=labels[row]
            if metric===:frobenius
                weights[output_sector,input_sector]=hypot(
                    weights[output_sector,input_sector],value)
            else
                weights[output_sector,input_sector]=max(
                    weights[output_sector,input_sector],value)
            end
        end
    end
    weights
end

"""
    schur_block_structure(L, basis=nothing; metric=:frobenius,
                          representation=nothing, threshold=0,
                          time=nothing, parameters=nothing)

Measure sector-to-sector PI-coordinate blocks of a superoperator.  Rows are
output sectors and columns are input sectors.  Sparse/dense matrices are
scanned directly.  A matrix-free source is probed exactly once per input PI
coordinate with reusable scratch and is never materialized.
"""
function schur_block_structure(source::Union{AbstractMatrix,PIModel,LiouvillianPlan,
                                              CompiledPIModel,MatrixFreeLiouvillian,
                                              AdjointMatrixFreeLiouvillian,
                                              MatrixFreeSymmetryProjector};
                               basis=nothing,metric=:frobenius,
                               representation=nothing,threshold=0,
                               time=nothing,parameters=nothing)
    _validate_schur_threshold(threshold)
    metric in (:frobenius,:maxabs) ||
        throw(ArgumentError("superoperator metric must be :frobenius or :maxabs"))
    resolved_representation=_superoperator_representation(representation)
    resolved_basis=_resolve_visualization_basis(source,basis)
    prepared=_visualization_source(source)
    n=length(resolved_basis);size(prepared)==(n,n) ||
        throw(DimensionMismatch("superoperator dimensions do not match the PI basis"))
    if !_visualization_autonomous(prepared) && time===nothing
        throw(ArgumentError("a time-dependent superoperator requires an explicit time"))
    end
    evaluation_time=time===nothing ? 0.0 : time
    # A general operator-valued drive uses the Liouvillian compatibility
    # fallback.  Freeze and lower it once here so coordinate probing does not
    # rebuild an instantaneous sparse matrix for every input coordinate.
    prepared=_visualization_at(prepared,evaluation_time,parameters)
    cutoff=_schur_threshold(threshold,_real_float_type(eltype(prepared)))
    weights = prepared isa AbstractMatrix ?
        _matrix_block_weights(prepared,resolved_basis,metric) :
        _matrixfree_block_weights(prepared,resolved_basis,metric,
                                  evaluation_time,parameters)
    applications=prepared isa AbstractMatrix ? 0 : n
    metadata=(N=resolved_basis.N,d=resolved_basis.d,pi_dimension=n,
              orientation=:output_by_input,source_type=typeof(source),
              matrixfree=!(prepared isa AbstractMatrix),applications,
              time=evaluation_time,parameters)
    _schur_structure(resolved_basis,:superoperator,weights,metric,
                     resolved_representation,cutoff,metadata)
end

schur_block_structure(source,basis::PIBasis;kwargs...)=
    schur_block_structure(source;basis=basis,kwargs...)

function show(io::IO,structure::SchurBlockStructure)
    print(io,"SchurBlockStructure(kind=",structure.kind,
          ", sectors=",length(structure.sectors),
          ", metric=",structure.metric,
          ", representation=",structure.representation,
          ", active_blocks=",count(structure.active),")")
end

function show(io::IO,::MIME"text/plain",structure::SchurBlockStructure)
    show(io,structure)
    orientation=structure.kind===:superoperator ?
        "weights (rows=output, columns=input):" : "diagonal sector weights:"
    print(io,"\n  block dimensions: ",structure.block_dimensions,
          "\n  coordinate dimensions: ",structure.coordinate_dimensions,
          "\n  ",orientation,"\n")
    show(io,MIME"text/plain"(),structure.weights)
end

"""
    visualize_schur_blocks(structure; scale=:linear, normalize=:global,
                           show_values=false, show_young_diagrams=true,
                           title=nothing,
                           width=720, height=560)
    visualize_schur_blocks(source, [basis]; visualization_keywords...,
                           extraction_keywords...)

Create a dependency-free text/SVG view of Schur-sector weights. `scale` is
`:linear` or `:log`; `normalize` is `:global`, `:row`, `:column`, or `:none`.
`show_values=true` prints the raw numerical weights inside SVG cells. When a
state, operator, or superoperator is supplied directly, remaining keywords are
forwarded to [`schur_block_structure`](@ref).  By default,
`show_young_diagrams=true`
adds the unfilled partition shape beside each sector label.  Superoperators
show shapes on both the output-row and input-column axes; states and operators
show each diagonal sector shape once.  The diagrams do not choose a standard
tableau filling because PI operators sum over that multiplicity label.
"""
function visualize_schur_blocks(structure::SchurBlockStructure;
                                scale=:linear,normalize=:global,
                                show_values::Bool=false,
                                show_young_diagrams::Bool=true,title=nothing,
                                width::Integer=720,height::Integer=560)
    scale in (:linear,:log) || throw(ArgumentError("scale must be :linear or :log"))
    normalize in (:global,:row,:column,:none) || throw(ArgumentError(
        "normalize must be :global, :row, :column, or :none"))
    width>=320 || throw(ArgumentError("width must be at least 320"))
    height>=280 || throw(ArgumentError("height must be at least 280"))
    default_title=structure.kind===:superoperator ? "Schur-sector couplings" :
                                                   "Schur-sector blocks"
    label=title===nothing ? default_title : String(title)
    SchurBlockVisualization(structure,scale,normalize,show_values,
                            show_young_diagrams,label,
                            Int(width),Int(height))
end

function visualize_schur_blocks(source,basis::PIBasis;scale=:linear,
                                normalize=:global,show_values::Bool=false,
                                show_young_diagrams::Bool=true,
                                title=nothing,width::Integer=720,
                                height::Integer=560,kwargs...)
    structure=schur_block_structure(source,basis;kwargs...)
    visualize_schur_blocks(structure;scale,normalize,show_values,
                           show_young_diagrams,title,width,height)
end

function visualize_schur_blocks(source;scale=:linear,normalize=:global,
                                show_values::Bool=false,title=nothing,
                                show_young_diagrams::Bool=true,
                                width::Integer=720,height::Integer=560,kwargs...)
    structure=schur_block_structure(source;kwargs...)
    visualize_schur_blocks(structure;scale,normalize,show_values,
                           show_young_diagrams,title,width,height)
end

function show(io::IO,visualization::SchurBlockVisualization)
    print(io,"SchurBlockVisualization(kind=",visualization.structure.kind,
          ", scale=",visualization.scale,
          ", normalize=",visualization.normalize,
          ", young_diagrams=",visualization.show_young_diagrams,
          ", size=",visualization.width,"×",visualization.height,")")
end

function show(io::IO,::MIME"text/plain",visualization::SchurBlockVisualization)
    show(io,visualization)
    print(io,"\n  title: ",repr(visualization.title),
          "\n  use display(...) for SVG output")
end

function _svg_escape(value)
    text=string(value)
    for character in text
        code=UInt32(character)
        valid=code in (0x09,0x0a,0x0d) || 0x20<=code<=0xd7ff ||
              0xe000<=code<=0xfffd || 0x10000<=code<=0x10ffff
        valid || throw(ArgumentError("SVG text contains a character forbidden by XML"))
    end
    text=replace(text,"&"=>"&amp;")
    text=replace(text,"<"=>"&lt;",">"=>"&gt;")
    replace(text,"\""=>"&quot;","'"=>"&apos;")
end

_schur_label(sector::Partition)="("*join(sector.parts,",")*")"

function _normalized_schur_weights(visualization)
    weights=visualization.structure.weights;rows,columns=size(weights)
    values=zeros(Float64,rows,columns)
    if visualization.normalize===:global
        denominator=maximum(weights;init=zero(eltype(weights)))
        if denominator>0
            @inbounds for i in eachindex(values);values[i]=Float64(weights[i]/denominator);end
        end
    elseif visualization.normalize===:row
        for row in 1:rows
            denominator=maximum(view(weights,row,:);init=zero(eltype(weights)))
            denominator>0 || continue
            @inbounds for column in 1:columns
                values[row,column]=Float64(weights[row,column]/denominator)
            end
        end
    elseif visualization.normalize===:column
        for column in 1:columns
            denominator=maximum(view(weights,:,column);init=zero(eltype(weights)))
            denominator>0 || continue
            @inbounds for row in 1:rows
                values[row,column]=Float64(weights[row,column]/denominator)
            end
        end
    else
        @inbounds for i in eachindex(values);values[i]=clamp(Float64(weights[i]),0.0,1.0);end
    end
    if visualization.scale===:log
        @. values=log1p(999values)/log(1000)
    end
    values
end

const _SCHUR_SVG_PALETTE=("#eff6ff","#dbeafe","#bfdbfe","#93c5fd",
                          "#60a5fa","#3b82f6","#2563eb","#1d4ed8",
                          "#1e40af","#1e3a8a")

function _schur_color(value)
    index=clamp(floor(Int,value*(length(_SCHUR_SVG_PALETTE)-1))+1,
                1,length(_SCHUR_SVG_PALETTE))
    _SCHUR_SVG_PALETTE[index]
end

function _schur_value_label(value)
    iszero(value) && return "0"
    string(round(value;sigdigits=4))
end

# Keep small diagrams literal (one SVG rectangle per Young-diagram box) while
# bounding the number of SVG nodes for research-scale N.  The compressed row
# bands still show the partition silhouette; they never change numerical data.
const _SCHUR_YOUNG_EXACT_BOX_LIMIT=64
const _SCHUR_YOUNG_MAX_ROW_BANDS=64

function _young_diagram_tooltip(structure,index,compressed)
    sector=structure.sectors[index]
    multiplicity=symmetric_group_dimension(sector)
    suffix=compressed ? "; compressed thumbnail" : ""
    "Young diagram for partition $(_schur_label(sector)); " *
    "U($(structure.basis.d)) irrep dimension=$(structure.block_dimensions[index]); " *
    "f^ν=$multiplicity standard Young tableaux/multiplicity labels; " *
    "PI operators sum over these labels, so no unique filling is selected$suffix"
end

function _show_young_diagram(io,structure,index,center_x,center_y,
                             max_width,max_height,axis_class)
    sector=structure.sectors[index]
    rows=Int[value for value in sector.parts if value>0]
    total=sum(rows;init=0)
    exact=total<=_SCHUR_YOUNG_EXACT_BOX_LIMIT
    compressed=!exact
    partition=_svg_escape(_schur_label(sector))
    print(io,"<g class=\"young-diagram ",axis_class,
          "\" data-partition=\"",partition,"\" data-rendering=\"",
          exact ? "boxes" : "compressed","\"><title>",
          _svg_escape(_young_diagram_tooltip(structure,index,compressed)),
          "</title>")
    if isempty(rows)
        # N=0 has the empty partition.  There are deliberately no box nodes.
        print(io,"<text class=\"young-empty-diagram\" x=\"",center_x,
              "\" y=\"",center_y+3,"\" text-anchor=\"middle\" ",
              "font-family=\"serif\" font-size=\"11\">∅</text></g>")
        return
    end
    longest=first(rows);row_count=length(rows)
    box=min(max_width/longest,max_height/row_count)
    diagram_width=box*longest;diagram_height=box*row_count
    origin_x=center_x-diagram_width/2
    origin_y=center_y-diagram_height/2
    if exact
        for (row,row_length) in pairs(rows),column in 1:row_length
            x=origin_x+(column-1)*box;y=origin_y+(row-1)*box
            print(io,"<rect class=\"young-box\" x=\"",x,"\" y=\"",y,
                  "\" width=\"",box,"\" height=\"",box,
                  "\" fill=\"#f8fafc\" stroke=\"#334155\" ",
                  "stroke-width=\"0.65\"/>")
        end
    else
        # For large N render at most `_SCHUR_YOUNG_MAX_ROW_BANDS` row bands.
        # This preserves a fixed thumbnail and bounded SVG-node count instead
        # of emitting one DOM element per particle.
        band_count=min(row_count,_SCHUR_YOUNG_MAX_ROW_BANDS)
        for band in 1:band_count
            first_row=fld((band-1)*row_count,band_count)+1
            last_row=fld(band*row_count,band_count)
            row_length=rows[last_row]
            x=origin_x;y=origin_y+(first_row-1)*box
            print(io,"<rect class=\"young-row-band\" x=\"",x,"\" y=\"",y,
                  "\" width=\"",box*row_length,"\" height=\"",
                  box*(last_row-first_row+1),
                  "\" fill=\"#e2e8f0\" stroke=\"#334155\" ",
                  "stroke-width=\"0.65\"/>")
        end
    end
    print(io,"</g>")
end

function show(io::IO,::MIME"image/svg+xml",visualization::SchurBlockVisualization)
    structure=visualization.structure;count_sectors=length(structure.sectors)
    width=visualization.width;height=visualization.height
    diagrams=visualization.show_young_diagrams
    left=diagrams&&structure.kind===:superoperator ? 156.0 : 112.0
    right=34.0;top=diagrams ? 122.0 : 88.0;bottom=88.0
    cell=min((width-left-right)/count_sectors,(height-top-bottom)/count_sectors)
    grid_width=cell*count_sectors;grid_height=grid_width
    x0=left+(width-left-right-grid_width)/2
    y0=top+(height-top-bottom-grid_height)/2
    normalized=_normalized_schur_weights(visualization)
    print(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"",width,
          "\" height=\"",height,"\" viewBox=\"0 0 ",width," ",height,
          "\" role=\"img\">")
    print(io,"<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>")
    print(io,"<text x=\"",width/2,"\" y=\"30\" text-anchor=\"middle\" ",
          "font-family=\"sans-serif\" font-size=\"18\" font-weight=\"600\">",
          _svg_escape(visualization.title),"</text>")
    subtitle=structure.kind===:superoperator ? "columns: input sector · rows: output sector" :
                                               "diagonal PI Schur sectors"
    print(io,"<text x=\"",width/2,"\" y=\"52\" text-anchor=\"middle\" ",
          "font-family=\"sans-serif\" font-size=\"12\" fill=\"#475569\">",
          subtitle,"</text>")
    for (index,sector) in pairs(structure.sectors)
        label=_svg_escape(_schur_label(sector));center=x0+(index-0.5)*cell
        print(io,"<text x=\"",center,"\" y=\"",y0-10,
              "\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"11\">",
              label,"</text>")
        center_y=y0+(index-0.5)*cell
        print(io,"<text x=\"",x0-10,"\" y=\"",center_y+4,
              "\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"11\">",
              label,"</text>")
        if diagrams
            thumbnail_width=min(34.0,0.78cell)
            _show_young_diagram(io,structure,index,center,y0-48,
                                thumbnail_width,30.0,"young-column-diagram")
            if structure.kind===:superoperator
                thumbnail_height=min(34.0,0.78cell)
                _show_young_diagram(io,structure,index,x0-60,center_y,
                                    32.0,thumbnail_height,"young-row-diagram")
            end
        end
    end
    for row in 1:count_sectors,column in 1:count_sectors
        x=x0+(column-1)*cell;y=y0+(row-1)*cell
        active=structure.active[row,column]
        fill=active ? _schur_color(normalized[row,column]) : "#f8fafc"
        opacity=active ? "1" : "0.55"
        label=_schur_value_label(structure.weights[row,column])
        tooltip = if structure.kind===:superoperator
            "output $(_schur_label(structure.sectors[row])), input " *
            "$(_schur_label(structure.sectors[column])): $label"
        elseif row==column
            "sector $(_schur_label(structure.sectors[row])): $label"
        else
            "not applicable: PI operators are Schur diagonal"
        end
        print(io,"<rect x=\"",x,"\" y=\"",y,"\" width=\"",cell,
              "\" height=\"",cell,"\" fill=\"",fill,"\" fill-opacity=\"",
              opacity,"\" stroke=\"#cbd5e1\" stroke-width=\"1\"><title>",
              _svg_escape(tooltip),"</title></rect>")
        if visualization.show_values
            text_color=active&&normalized[row,column]>0.55 ? "#ffffff" : "#0f172a"
            print(io,"<text x=\"",x+cell/2,"\" y=\"",y+cell/2+4,
                  "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"10\" fill=\"",
                  text_color,"\">",_svg_escape(label),"</text>")
        end
    end
    print(io,"<text x=\"",width/2,"\" y=\"",height-24,
          "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#475569\">",
          "metric: ",structure.metric," · representation: ",structure.representation,
          " · threshold: ",_svg_escape(structure.threshold),"</text></svg>")
end

"""Write a Schur-block SVG and return the path as a `String`."""
function save_schur_block_visualization(path::AbstractString,
                                        visualization::SchurBlockVisualization)
    svg=sprint(show,MIME"image/svg+xml"(),visualization)
    open(path,"w") do io
        write(io,svg)
    end
    String(path)
end

function save_schur_block_visualization(path::AbstractString,
                                        structure::SchurBlockStructure;kwargs...)
    save_schur_block_visualization(path,visualize_schur_blocks(structure;kwargs...))
end
