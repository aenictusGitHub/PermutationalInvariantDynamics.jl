module PermutationalInvariantDynamicsMakieExt

using PermutationalInvariantDynamics
import Makie

const PID=PermutationalInvariantDynamics

# These conversions deliberately expose existing numerical data only.  They
# never trigger a solver, a phase-space transform, or matrix-free probing.

# Convergence reports store a leading `missing` by construction, so the field
# vector necessarily has a `Union{Missing,T}` element type.  Once that leading
# entry is removed and the remainder validated, expose a typed lazy view:
# Makie's dimension conversion requires the concrete `T`, while collecting the
# full error history solely to narrow its element type is avoidable.
struct _NonMissingVector{T,V<:AbstractVector} <: AbstractVector{T}
    parent::V
end
Base.size(values::_NonMissingVector)=size(values.parent)
Base.axes(values::_NonMissingVector)=axes(values.parent)
Base.IndexStyle(::Type{<:_NonMissingVector{T,V}}) where {T,V}=
    Base.IndexStyle(V)
@inline function Base.getindex(values::_NonMissingVector{T},index::Int) where T
    value=values.parent[index]
    ismissing(value)&&throw(ArgumentError(
        "convergence report has incomplete pairwise error data"))
    value::T
end

function _nonmissing_vector(values::AbstractVector)
    all(!ismissing,values)||throw(ArgumentError(
        "convergence report has incomplete pairwise error data"))
    T=Base.nonmissingtype(eltype(values))
    _NonMissingVector{T,typeof(values)}(values)
end

Makie.plottype(::PID.ComplexSpectrum)=Makie.Scatter

function _spectrum_coordinates(values::Vector{Complex{R}}) where R<:Real
    # Julia stores ordinary isbits complex scalars as adjacent real/imaginary
    # components.  Reinterpret that storage for the common Float32/Float64
    # path; retain the allocating fallback for generic scalar layouts.
    if isbitstype(R)&&sizeof(Complex{R})==2sizeof(R)
        components=reinterpret(reshape,R,values)
        return view(components,1,:),view(components,2,:)
    end
    real.(values),imag.(values)
end

_spectrum_coordinates(values::Vector{R}) where R<:Real=
    (values,range(zero(R);step=zero(R),length=length(values)))
_spectrum_coordinates(values)=real.(values),imag.(values)

function Makie.convert_arguments(::Type{<:Makie.Scatter},
                                 spectrum::PID.ComplexSpectrum)
    _spectrum_coordinates(spectrum.values)
end

Makie.plottype(::PID.SpinPhaseSpaceData)=Makie.Heatmap
function Makie.convert_arguments(::Type{<:Makie.Heatmap},
                                 data::PID.SpinPhaseSpaceData)
    # Makie indexes z as z[x_index,y_index], whereas the numerical API keeps
    # values[phi_index,theta_index].
    (data.theta,data.phi,PermutedDimsArray(data.values,(2,1)))
end

Makie.plottype(::PID.SchurBlockStructure)=Makie.Heatmap
function Makie.convert_arguments(::Type{<:Makie.Heatmap},
                                 structure::PID.SchurBlockStructure)
    columns=Base.OneTo(size(structure.weights,2))
    rows=Base.OneTo(size(structure.weights,1))
    (columns,rows,PermutedDimsArray(structure.weights,(2,1)))
end

Makie.plottype(::PID.QuditHusimiData)=Makie.Lines
function Makie.convert_arguments(::Type{<:Makie.Lines},
                                 data::PID.QuditHusimiData)
    (Base.OneTo(length(data.values)),data.values)
end

Makie.plottype(::PID.ConvergenceStudyResult)=Makie.Lines
function Makie.convert_arguments(::Type{<:Makie.Lines},
                                 report::PID.ConvergenceStudyResult)
    indices=2:length(report.refinements)
    x=view(report.refinements,indices)
    y=_nonmissing_vector(view(report.pairwise_errors,indices))
    (x,y)
end

Makie.plottype(::PID.ResultTable)=Makie.Lines
function Makie.convert_arguments(::Type{<:Makie.Lines},
                                 table::PID.ResultTable)
    names=propertynames(table.columns)
    length(names)==2||throw(ArgumentError(
        "lines(result_table) requires exactly two columns; select the " *
        "desired x and y columns explicitly for a wider table"))
    x=getproperty(table.columns,names[1])
    y=getproperty(table.columns,names[2])
    eltype(x)<:Number&&eltype(y)<:Number||throw(ArgumentError(
        "lines(result_table) requires two numeric columns"))
    (x,y)
end

Makie.plottype(::PID.SpectrumResult)=Makie.Scatter
function Makie.convert_arguments(::Type{<:Makie.Scatter},
                                 spectrum::PID.SpectrumResult)
    _spectrum_coordinates(spectrum.values)
end

end
