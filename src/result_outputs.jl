"""Current schema version written by [`save_result`](@ref) for `.pidrun` records."""
const PI_RESULT_ARCHIVE_VERSION=UInt16(1)

"""
    ResultTable

Dependency-free, column-oriented view returned by [`result_table`](@ref).
`columns` is a named tuple of equally sized vectors and `metadata` is the
compact [`summarize`](@ref) output. Indexing or iterating produces named-tuple
rows without copying the columns. Large states and eigenvectors are excluded
unless the caller explicitly requests `include_output=true`.

Loading Tables.jl makes this object a Tables-compatible column table.
"""
struct ResultTable{C<:NamedTuple,M}
    columns::C
    metadata::M
    row_count::Int
end

function ResultTable(columns::NamedTuple;metadata=NamedTuple())
    names=propertynames(columns)
    count=nothing
    for name in names
        column=getproperty(columns,name)
        column isa AbstractVector||throw(ArgumentError(
            "result-table column $name must be an AbstractVector"))
        count===nothing ? (count=length(column)) :
            length(column)==count||throw(DimensionMismatch(
                "result-table columns have different lengths"))
    end
    ResultTable(columns,metadata,count===nothing ? 0 : count)
end

Base.length(table::ResultTable)=table.row_count
Base.firstindex(::ResultTable)=1
Base.lastindex(table::ResultTable)=length(table)
Base.isempty(table::ResultTable)=iszero(length(table))
function Base.getindex(table::ResultTable,index::Integer)
    @boundscheck checkbounds(Base.OneTo(length(table)),index)
    names=propertynames(table.columns)
    NamedTuple{names}(Tuple(
        getproperty(table.columns,name)[index] for name in names))
end
Base.iterate(table::ResultTable,index::Int=1)=
    index>length(table) ? nothing : (table[index],index+1)
function Base.show(io::IO,table::ResultTable)
    print(io,"ResultTable($(length(table)) rows, columns=",
        propertynames(table.columns),")")
end

# O(1) logical columns for absent optional diagnostics.
struct _ResultConstantColumn{T} <: AbstractVector{T}
    value::T
    count::Int
end
Base.size(column::_ResultConstantColumn)=(column.count,)
Base.IndexStyle(::Type{<:_ResultConstantColumn})=Base.IndexLinear()
@inline function Base.getindex(column::_ResultConstantColumn,index::Int)
    @boundscheck checkbounds(column,index)
    column.value
end

_result_package_version()=try
    string(Base.pkgversion(@__MODULE__))
catch
    "unknown"
end

@inline function _result_property(value,name::Symbol,default=missing)
    hasproperty(value,name) ? getproperty(value,name) : default
end

function _result_info_property(value,name::Symbol,default=missing)
    direct=_result_property(value,name,default)
    direct===default||return direct
    hasproperty(value,:info)||return default
    _result_property(getproperty(value,:info),name,default)
end

function _result_converged(value)
    status=_result_info_property(value,:converged,missing)
    status isa Bool&&return status
    status===missing&&return missing
    status===nothing&&return missing
    try
        all(status)
    catch
        missing
    end
end

_result_algorithm_name(value)=begin
    algorithm=_result_property(value,:algorithm,missing)
    algorithm===missing ? missing :
        algorithm isa Symbol ? algorithm : Symbol(nameof(typeof(algorithm)))
end

_base_result_summary(result)=(
    result_type=string(nameof(typeof(result))),
    package_version=_result_package_version())

"""
    summarize(result)

Return a compact, allocation-bounded named tuple describing a package result.
The summary reports convergence evidence, selected algorithms, sample counts,
and representation dimensions when those facts are stored in the result. It
does not run a solver, reconstruct a full Hilbert-space object, validate
positivity, or compute missing diagnostics.

The exact fields depend on the result type; `result_type` and
`package_version` are always present.
"""
summarize(result)=_base_result_summary(result)

function summarize(state::PIState)
    merge(_base_result_summary(state),(
        result_type="PIState",N=state.basis.N,d=state.basis.d,
        pi_dimension=length(state.data),sector_count=length(state.basis.sectors),
        trace=trace(state)))
end

function summarize(result::SteadyStateResult)
    state=result.state
    merge(_base_result_summary(result),(
        result_type="SteadyStateResult",
        algorithm=_result_algorithm_name(result),
        method=_result_info_property(result,:method),
        converged=_result_converged(result),
        residual=_result_info_property(result,:residual),
        trace_error=_result_info_property(result,:trace_error),
        N=state.basis.N,d=state.basis.d,
        pi_dimension=length(state.data)))
end

function _result_time_summary(result,name::AbstractString)
    times=result.times
    (
        result_type=name,
        sample_count=length(times),
        first_time=isempty(times) ? missing : first(times),
        last_time=isempty(times) ? missing : last(times),
    )
end

function summarize(result::DynamicsResult)
    merge(_base_result_summary(result),_result_time_summary(result,"DynamicsResult"),(
        algorithm=_result_algorithm_name(result),
        saved_states=length(result.states)))
end

function summarize(result::DynamicsStreamResult)
    observable_count=result.observables===nothing ? 0 :
        length(result.observables)
    merge(_base_result_summary(result),
        _result_time_summary(result,"DynamicsStreamResult"),(
            algorithm=_result_algorithm_name(result),
            saved_states=result.states===nothing ? 0 : length(result.states),
            observable_count))
end

function summarize(result::SpectrumResult)
    merge(_base_result_summary(result),(
        result_type="SpectrumResult",mode_count=length(result.values),
        vectors_saved=result.vectors!==nothing,
        converged=_result_converged(result),
        residual=_result_info_property(result,:residual)))
end

function summarize(result::ComplexSpectrum)
    count=length(result.values)
    converged=result.converged===nothing ? missing : all(result.converged)
    merge(_base_result_summary(result),(
        result_type="ComplexSpectrum",kind=result.kind,
        representation=result.representation,mode_count=count,
        complete=result.complete,converged))
end

function summarize(result::ParameterScanResult)
    successes=count(point->point.status===:success,result.points)
    merge(_base_result_summary(result),(
        result_type="ParameterScanResult",task=result.task,
        requested_points=length(result.parameters),
        completed_points=length(result.points),successful=successes,
        failed=length(result.points)-successes,
        stopped=_result_property(result.metadata,:stopped,missing),
        cancelled=_result_property(result.metadata,:cancelled,missing)))
end

function summarize(result::ConvergenceStudyResult)
    merge(_base_result_summary(result),(
        result_type="ConvergenceStudyResult",parameter=result.parameter,
        levels=length(result),converged=result.converged,
        reason=result.reason,first_passing_index=result.first_passing_index,
        final_error=isempty(result.pairwise_errors) ? missing :
            last(result.pairwise_errors)))
end

function summarize(result::TrajectoryEnsembleResult)
    merge(_base_result_summary(result),
        _result_time_summary(result,"TrajectoryEnsembleResult"),(
            trajectory_count=result.trajectory_count,
            state_histories_saved=result.trajectories!==nothing,
            observable_count=result.observables===nothing ? 0 :
                length(result.observables)))
end

function summarize(result::TrajectorySteadyStateResult)
    merge(_base_result_summary(result),(
        result_type="TrajectorySteadyStateResult",
        trajectory_count=result.trajectory_count,
        samples_per_trajectory=result.samples_per_trajectory,
        residual=result.residual,relative_residual=result.relative_residual,
        trace_error=result.trace_error,
        standard_error=result.standard_error))
end

function summarize(result::HOPSEnsembleResult)
    merge(_base_result_summary(result),
        _result_time_summary(result,"HOPSEnsembleResult"),(
            trajectory_count=result.trajectory_count,
            saved_states=length(result.states),
            final_standard_error=isempty(result.standard_error) ? missing :
                last(result.standard_error)))
end

function summarize(result::MeanFieldResult)
    merge(_base_result_summary(result),_result_time_summary(result,"MeanFieldResult"),(
        limit=result.limit,algorithm=result.algorithm,
        saved_states=length(result.states)))
end

function summarize(result::PopulationSolution)
    merge(_base_result_summary(result),
        _result_time_summary(result,"PopulationSolution"),(
            population_dimension=size(result.plan,1),
            saved_vectors=length(result.populations)))
end

function summarize(result::AdaptiveTrajectoryResult)
    merge(_base_result_summary(result),
        _result_time_summary(result,"AdaptiveTrajectoryResult"),(
            backend=result.backend,trajectory_count=result.trajectory_count,
            converged=result.converged,
            stopping_reason=result.stopping_reason))
end

function summarize(result::BathFitResult)
    merge(_base_result_summary(result),(
        result_type="BathFitResult",sample_count=length(result.times),
        exponential_terms=length(result),
        converged=_result_property(result.report,:converged,missing),
        relative_residual=_result_property(
            result.report,:relative_residual,missing),
        identifiable=_result_property(result.report,:identifiable,missing)))
end

function summarize(result::ParameterInferenceResult)
    merge(_base_result_summary(result),(
        result_type="ParameterInferenceResult",
        parameter_count=length(result.parameters),
        observation_count=length(result.predictions),
        objective=result.objective,iterations=result.iterations,
        converged=result.converged,termination=result.termination,
        derivative_method=result.derivative_method))
end

function summarize(result::ExperimentResult)
    merge(_base_result_summary(result),(
        result_type="ExperimentResult",task=result.report.task,
        verified=result.report.verified,
        verification_level=result.report.verification_level,
        solver_converged=result.report.solver_converged,
        physical_valid=result.report.physical_valid,
        refinement_converged=result.report.refinement_converged,
        selected_algorithm=result.plan.selected_algorithm,
        structural_digest=result.provenance.structural_digest))
end

function summarize(result::PIStudyResult)
    state_count=result.states===nothing ? (result.state===nothing ? 0 : 1) :
        length(result.states)
    value_count=result.values===nothing ? 0 : length(result.values)
    observable_count=result.observables===nothing ? 0 :
        length(result.observables)
    merge(_base_result_summary(result),(
        result_type="PIStudyResult",task=result.task,
        converged=result.converged,residual=result.residual,
        selected_algorithm=result.selected_algorithm,
        state_count,observable_count,value_count))
end

"""
    result_table(result; include_output=false)

Return a dependency-free [`ResultTable`](@ref) for supported result objects.
Time-series observables, spectra, scans, convergence studies, and inference
residuals become ordinary columns. Potentially large states, population
vectors, and eigenvectors are omitted by default and appear only with
`include_output=true`.

This function never computes missing observables or diagnostics. Unsupported
objects raise instead of being stringified implicitly.
"""
function result_table(result;include_output::Bool=false)
    throw(ArgumentError(
        "result_table does not define a tabular representation for $(typeof(result))"))
end
result_table(table::ResultTable;include_output::Bool=false)=table

function _result_table(columns::NamedTuple,result)
    ResultTable(columns;metadata=summarize(result))
end

function result_table(state::PIState;include_output::Bool=false)
    summary=summarize(state)
    names=propertynames(summary)
    columns=NamedTuple{names}(Tuple([getproperty(summary,name)] for name in names))
    include_output&&(columns=merge(columns,(state=[state],)))
    _result_table(columns,state)
end

function result_table(result::SteadyStateResult;include_output::Bool=false)
    summary=summarize(result)
    # Package version and the type label remain table metadata rather than
    # being repeated as numerical columns.
    names=Tuple(name for name in propertynames(summary)
                if name ∉ (:result_type,:package_version))
    columns=NamedTuple{names}(Tuple([getproperty(summary,name)] for name in names))
    include_output&&(columns=merge(columns,(state=[result.state],)))
    _result_table(columns,result)
end

function _named_result_columns(observables,count::Int)
    observables===nothing&&return NamedTuple()
    entries=observables isa NamedTuple ? pairs(observables) :
        observables isa AbstractDict ? pairs(observables) :
        throw(ArgumentError(
            "result observables must be a named tuple or dictionary"))
    output=Pair{Symbol,Any}[]
    seen=Set{Symbol}()
    for (raw_name,value) in entries
        name=Symbol(raw_name)
        name in seen&&throw(ArgumentError(
            "duplicate result observable name $name"))
        push!(seen,name)
        if value isa AbstractVector
            length(value)==count||throw(DimensionMismatch(
                "observable $name has a different sampling length"))
            push!(output,name=>value)
        elseif hasproperty(value,:mean)&&getproperty(value,:mean) isa AbstractVector
            for field in (:mean,:variance,:standard_error,:lower,:upper)
                hasproperty(value,field)||continue
                column=getproperty(value,field)
                column isa AbstractVector||continue
                length(column)==count||throw(DimensionMismatch(
                    "observable $name field $field has a different sampling length"))
                column_name=Symbol(name,"_",field)
                column_name in seen&&throw(ArgumentError(
                    "duplicate result column $column_name"))
                push!(seen,column_name)
                push!(output,column_name=>column)
            end
        else
            throw(ArgumentError(
                "observable $name is not a sampled vector or statistic"))
        end
    end
    (;output...)
end

function result_table(result::DynamicsResult;include_output::Bool=false)
    columns=(time=result.times,)
    include_output&&(columns=merge(columns,(state=result.states,)))
    _result_table(columns,result)
end

function result_table(result::DynamicsStreamResult;include_output::Bool=false)
    observable_columns=_named_result_columns(
        result.observables,length(result.times))
    :time in propertynames(observable_columns)&&throw(ArgumentError(
        "observable name :time conflicts with the result-table time column"))
    :state in propertynames(observable_columns)&&throw(ArgumentError(
        "observable name :state conflicts with the optional state column"))
    columns=merge((time=result.times,),observable_columns)
    include_output&&result.states!==nothing&&
        (columns=merge(columns,(state=result.states,)))
    _result_table(columns,result)
end

function result_table(result::SpectrumResult;include_output::Bool=false)
    count=length(result.values)
    residual=_result_info_property(result,:residuals,nothing)
    residual===nothing&&
        (residual=_ResultConstantColumn(missing,count))
    converged=_result_info_property(result,:converged,nothing)
    if converged===nothing
        converged=_ResultConstantColumn(missing,count)
    elseif converged isa Bool
        converged=_ResultConstantColumn(converged,count)
    end
    columns=(index=Base.OneTo(count),value=result.values,
        residual,converged)
    include_output&&result.vectors!==nothing&&
        (columns=merge(columns,(vector=eachcol(result.vectors),)))
    _result_table(columns,result)
end

function result_table(result::ComplexSpectrum;include_output::Bool=false)
    count=length(result.values)
    residual=result.residuals===nothing ?
        _ResultConstantColumn(missing,count) : result.residuals
    converged=result.converged===nothing ?
        _ResultConstantColumn(missing,count) : result.converged
    _result_table((index=Base.OneTo(count),value=result.values,
        classification=result.classifications,residual,converged),result)
end

function result_table(result::ParameterScanResult;include_output::Bool=false)
    _result_table(parameter_scan_columns(result;include_output),result)
end

function result_table(result::ConvergenceStudyResult;
                      include_output::Bool=false)
    columns=(level=Base.OneTo(length(result.refinements)),
        refinement=result.refinements,estimate=result.estimates,
        pairwise_error=result.pairwise_errors,tolerance=result.tolerances,
        pairwise_converged=result.pairwise_converged,
        observed_rate=result.observed_rates,
        solver_converged=result.solver_converged)
    include_output&&(columns=merge(columns,(output=result.results,
        diagnostics=result.diagnostics)))
    _result_table(columns,result)
end

function result_table(result::QuditHusimiData;include_output::Bool=false)
    _result_table((point=Base.OneTo(length(result.values)),
        value=result.values),result)
end

function result_table(result::MeanFieldResult;include_output::Bool=false)
    columns=(time=result.times,)
    include_output&&(columns=merge(columns,(state=result.states,)))
    _result_table(columns,result)
end

function result_table(result::PopulationSolution;include_output::Bool=false)
    columns=(time=result.times,)
    include_output&&
        (columns=merge(columns,(populations=result.populations,)))
    _result_table(columns,result)
end

function result_table(result::HOPSEnsembleResult;include_output::Bool=false)
    columns=(time=result.times,sample_spread=result.sample_spread,
        standard_error=result.standard_error)
    include_output&&(columns=merge(columns,(state=result.states,)))
    _result_table(columns,result)
end

function result_table(result::TrajectoryEnsembleResult;
                      include_output::Bool=false)
    observable_columns=_named_result_columns(
        result.observables,length(result.times))
    :time in propertynames(observable_columns)&&throw(ArgumentError(
        "observable name :time conflicts with the result-table time column"))
    columns=merge((time=result.times,),observable_columns)
    include_output&&result.trajectories!==nothing&&
        throw(ArgumentError(
            "trajectory histories are nested paths, not table columns; " *
            "export their requested online statistics or save individual checkpoints"))
    _result_table(columns,result)
end

function result_table(result::AdaptiveTrajectoryResult;
                      include_output::Bool=false)
    observable_columns=_named_result_columns(
        result.observables,length(result.times))
    :time in propertynames(observable_columns)&&throw(ArgumentError(
        "observable name :time conflicts with the result-table time column"))
    columns=merge((time=result.times,),observable_columns)
    _result_table(columns,result)
end

function result_table(result::BathFitResult;include_output::Bool=false)
    _result_table((time=result.times,sample=result.samples,
        fitted=result.fitted,residual=result.residuals),result)
end

function result_table(result::ParameterInferenceResult;
                      include_output::Bool=false)
    count=length(result.predictions)
    length(result.residuals)==count==
        length(result.standardized_residuals)||throw(DimensionMismatch(
            "inference result observation arrays have different lengths"))
    _result_table((observation=Base.OneTo(count),
        prediction=result.predictions,residual=result.residuals,
        standardized_residual=result.standardized_residuals),result)
end

function result_table(result::ExperimentResult;include_output::Bool=false)
    table=result_table(result.solution;include_output)
    ResultTable(table.columns;metadata=summarize(result))
end

function result_table(result::PIStudyResult;include_output::Bool=false)
    table=result_table(result.raw;include_output)
    columns=table.columns
    if result.task===:steady_state&&result.observables!==nothing
        additions=Pair{Symbol,Any}[]
        for (raw_name,value) in pairs(result.observables)
            value isa Number||throw(ArgumentError(
                "stationary PIStudy observable $raw_name is not scalar"))
            name=Symbol(raw_name)
            name in propertynames(columns)&&throw(ArgumentError(
                "stationary PIStudy observable $name conflicts with an " *
                "existing result-table column"))
            push!(additions,name=>[value])
        end
        columns=merge(columns,(;additions...))
    end
    ResultTable(columns;metadata=summarize(result))
end

function _result_format(path,format)
    format===:auto||return Symbol(format)
    extension=lowercase(splitext(String(path))[2])
    extension==".csv"&&return :csv
    extension in (".tsv",".txt")&&return :tsv
    extension in (".jld2",".jld")&&return :jld2
    extension in (".h5",".hdf5")&&return :hdf5
    extension==".pidrun"&&return :pidrun
    throw(ArgumentError(
        "cannot infer result format from extension $extension; use " *
        "format=:csv, :tsv, :pidrun, :jld2, or :hdf5"))
end

function _result_text(value)
    value===missing&&return ""
    value===nothing&&return ""
    value isa AbstractString&&return String(value)
    value isa Symbol&&return String(value)
    value isa Union{Number,Bool,Char}&&return repr(value)
    repr(value)
end

function _write_delimited_field(io,value,delimiter::Char)
    text=_result_text(value)
    must_quote=occursin('"',text)||occursin('\n',text)||occursin('\r',text)||
        occursin(delimiter,text)
    if must_quote
        write(io,'"')
        write(io,replace(text,"\""=>"\"\""))
        write(io,'"')
    else
        write(io,text)
    end
end

function _write_result_table(path,table::ResultTable,delimiter::Char)
    open(path,"w") do io
        names=propertynames(table.columns)
        for (position,name) in enumerate(names)
            position==1||write(io,delimiter)
            _write_delimited_field(io,String(name),delimiter)
        end
        write(io,'\n')
        for row in table
            for (position,name) in enumerate(names)
                position==1||write(io,delimiter)
                _write_delimited_field(io,getproperty(row,name),delimiter)
            end
            write(io,'\n')
        end
    end
    String(path)
end

function _result_metadata(metadata)
    output=Dict{String,String}()
    for (key,value) in pairs(metadata)
        output[string(key)]=_result_text(value)
    end
    output
end

function _result_detached_columns(table::ResultTable)
    names=propertynames(table.columns)
    NamedTuple{names}(Tuple(collect(getproperty(table.columns,name))
                            for name in names))
end

function _result_state_records(result)
    NamedTuple[]
end
_result_state_records(state::PIState)=[
    (label="state",time=nothing,state=state)]
_result_state_records(result::SteadyStateResult)=[
    (label="steady_state",time=nothing,state=result.state)]
_result_state_records(result::TrajectorySteadyStateResult)=[
    (label="trajectory_steady_state",time=nothing,state=result.state)]
function _result_state_records(result::Union{
        DynamicsResult,DynamicsStreamResult,HOPSEnsembleResult})
    result.states===nothing&&return NamedTuple[]
    [(label="state_$index",time=result.times[index],
      state=result.states[index]) for index in eachindex(result.states)]
end
function _result_state_records(result::ParameterScanResult)
    records=NamedTuple[]
    for point in result.points
        output=point.output
        state=output isa PIState ? output :
            output isa SteadyStateResult ? output.state : nothing
        state===nothing&&continue
        push!(records,(label="scan_$(point.index)",time=nothing,state))
    end
    records
end
_result_state_records(result::ExperimentResult)=
    _result_state_records(result.solution)
_result_state_records(result::PIStudyResult)=
    _result_state_records(result.raw)

function _write_pidrun_contents(path,result,table,summary,metadata)
    combined=merge(Dict(
        "schema_version"=>string(PI_RESULT_ARCHIVE_VERSION),
        "result_type"=>string(typeof(result)),
        "package_version"=>_result_package_version(),
        "julia_version"=>string(VERSION),
        "created_unix"=>repr(time())),
        _result_metadata(summary),_result_metadata(metadata))
    open(joinpath(path,"metadata.tsv"),"w") do io
        write(io,"key\tvalue\n")
        for (key,value) in sort!(collect(combined);by=first)
            _write_delimited_field(io,key,'\t');write(io,'\t')
            _write_delimited_field(io,value,'\t');write(io,'\n')
        end
    end
    _write_result_table(joinpath(path,"table.tsv"),table,'\t')
    records=_result_state_records(result)
    if !isempty(records)
        state_directory=joinpath(path,"states")
        mkpath(state_directory)
        index_table=ResultTable((
            index=collect(eachindex(records)),
            label=[record.label for record in records],
            time=[record.time for record in records],
            file=["state_$(lpad(index,6,'0')).pid" for index in eachindex(records)],
        ))
        for (index,record) in enumerate(records)
            checkpoint_metadata=merge(metadata,Dict(
                "result_type"=>string(typeof(result)),
                "label"=>record.label))
            save_checkpoint(joinpath(state_directory,
                    index_table.columns.file[index]),record.state;
                time=record.time,metadata=checkpoint_metadata,format=:pid)
        end
        _write_result_table(
            joinpath(state_directory,"index.tsv"),index_table,'\t')
    end
    path
end

function _save_pidrun(path,result,table,summary,metadata)
    target=abspath(path)
    ispath(target)&&throw(ArgumentError(
        "result archive path already exists: $path"))
    parent=dirname(target)
    isdir(parent)||throw(ArgumentError(
        "result archive parent directory does not exist: $parent"))
    staging=mktempdir(parent)
    moved=false
    try
        _write_pidrun_contents(
            staging,result,table,summary,metadata)
        # The staging directory and destination share a filesystem. Rename
        # publishes the completed record only after every checkpoint succeeds.
        mv(staging,target)
        moved=true
    finally
        !moved&&ispath(staging)&&rm(staging;recursive=true,force=true)
    end
    String(path)
end

function _save_result(::Val{:csv},path,result,table,summary,metadata)
    _write_result_table(path,table,',')
end
function _save_result(::Val{:tsv},path,result,table,summary,metadata)
    _write_result_table(path,table,'\t')
end
function _save_result(::Val{:pidrun},path,result,table,summary,metadata)
    _save_pidrun(path,result,table,summary,metadata)
end
function _save_result(::Val{F},path,result,table,summary,metadata) where F
    F in (:jld2,:hdf5)&&throw(ArgumentError(
        "result format :$F requires loading the corresponding optional " *
        "JLD2 or HDF5 package"))
    throw(ArgumentError("unknown result format :$F"))
end

"""
    save_result(path, result; format=:auto, include_output=false,
                metadata=Dict())

Export a supported result through one consistent interface. `.csv` and
`.tsv` write the compact [`result_table`](@ref). `.pidrun` creates a new,
versioned directory containing `metadata.tsv`, `table.tsv`, and exact portable
PI-state checkpoints when the result directly owns PI states. Existing
`.pidrun` paths are never replaced.

Loading JLD2 or HDF5 activates their optional backends. JLD2 stores the Julia
result plus its normalized summary and table; HDF5 stores the normalized
columns, metadata, and any exact PI-state checkpoint payloads. These optional
formats are not substitutes for the dependency-free, reconstructing
[`save_checkpoint`](@ref) and [`save_experiment`](@ref) schemas.

Large states, population vectors, and eigenvectors are excluded from tabular
columns by default. Set `include_output=true` only when the selected table
format can use those nested entries.
"""
function save_result(path,result;format=:auto,include_output::Bool=false,
                     metadata=Dict())
    selected=_result_format(path,format)
    table=result_table(result;include_output)
    summary=summarize(result)
    normalized_metadata=_result_metadata(metadata)
    reserved=Set((
        "schema_version","result_type","package_version",
        "julia_version","created_unix",
        string.(propertynames(summary))...,
    ))
    conflicts=sort!(collect(intersect(
        Set(keys(normalized_metadata)),reserved)))
    isempty(conflicts)||throw(ArgumentError(
        "result metadata uses reserved keys: $(join(conflicts, ", "))"))
    _save_result(Val(selected),String(path),result,table,summary,
        normalized_metadata)
end
