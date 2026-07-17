module PermutationalInvariantDynamicsTablesExt

using PermutationalInvariantDynamics
import Tables

const PID=PermutationalInvariantDynamics
const _SCAN_TABLE_NAMES=(
    :index,:parameter,:status,:residual,:trace_error,:converged,
    :iterations,:compile_seconds,:solve_seconds,:elapsed_seconds,
    :warm_started,:workspace_reused,:error_type,:message,
)
const _SCAN_TABLE_TYPES=(
    Int,Any,Symbol,Any,Any,Bool,Any,Float64,Float64,Float64,
    Bool,Bool,Union{Nothing,String},Union{Nothing,String},
)

# The iterator holds only the result reference. Rows are produced lazily and
# deliberately follow the dependency-free, output-free public row schema so a
# table sink never expands saved PI states or spectral vectors implicitly.
struct ParameterScanTableRows{R}
    result::R
end

Base.length(rows::ParameterScanTableRows)=length(rows.result)
Base.IteratorSize(::Type{<:ParameterScanTableRows})=Base.HasLength()
Base.IteratorEltype(::Type{<:ParameterScanTableRows})=Base.EltypeUnknown()
function Base.iterate(rows::ParameterScanTableRows,index::Int=1)
    index>length(rows.result)&&return nothing
    PID._scan_row(rows.result[index],false),index+1
end

Tables.istable(::Type{<:PID.ParameterScanResult})=true
Tables.rowaccess(::Type{<:PID.ParameterScanResult})=true
Tables.rows(result::PID.ParameterScanResult)=ParameterScanTableRows(result)
Tables.schema(::ParameterScanTableRows)=
    Tables.Schema(_SCAN_TABLE_NAMES,_SCAN_TABLE_TYPES)

const _COLUMN_RESULTS=Union{
    PID.ComplexSpectrum,
    PID.QuditHusimiData,
    PID.ConvergenceStudyResult,
}

# A missing optional diagnostic is a logical column, not `count` pieces of
# retained data.  Keep it as a tiny read-only vector so requesting Tables
# columns from a large spectrum does not allocate two full arrays merely to
# represent absent residual/convergence metadata.
struct _MissingColumn <: AbstractVector{Missing}
    count::Int
end
Base.size(column::_MissingColumn)=(column.count,)
Base.IndexStyle(::Type{_MissingColumn})=IndexLinear()
@inline function Base.getindex(column::_MissingColumn,index::Int)
    @boundscheck checkbounds(column,index)
    missing
end

Tables.istable(::Type{<:_COLUMN_RESULTS})=true
Tables.columnaccess(::Type{<:_COLUMN_RESULTS})=true

function Tables.columns(spectrum::PID.ComplexSpectrum)
    count=length(spectrum.values)
    residual=spectrum.residuals===nothing ? _MissingColumn(count) :
        spectrum.residuals
    converged=spectrum.converged===nothing ? _MissingColumn(count) :
        spectrum.converged
    (index=Base.OneTo(count),value=spectrum.values,
     classification=spectrum.classifications,residual,converged)
end

function Tables.columns(data::PID.QuditHusimiData)
    (point=Base.OneTo(length(data.values)),value=data.values)
end

function Tables.columns(report::PID.ConvergenceStudyResult)
    (level=Base.OneTo(length(report.refinements)),
     refinement=report.refinements,estimate=report.estimates,
     pairwise_error=report.pairwise_errors,
     tolerance=report.tolerances,
     pairwise_converged=report.pairwise_converged,
     observed_rate=report.observed_rates,
     solver_converged=report.solver_converged)
end

end
