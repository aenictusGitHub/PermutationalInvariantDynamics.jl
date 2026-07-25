module PIBenchmarkHarness

using BenchmarkTools
using Dates
using LinearAlgebra
using SHA

export benchmark_call, benchmark_metadata, metadata_path, write_metadata,
       write_tsv

"""
    benchmark_call(f; samples, seconds, warmups=2)

Benchmark the zero-argument callable `f` after explicit warm-up calls. Every
sample contains exactly one evaluation. The returned allocation count belongs
to the fastest sample; retained object storage must be measured separately.
"""
function benchmark_call(f; samples::Integer, seconds::Real,
                        warmups::Integer=2)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    seconds > 0 || throw(ArgumentError("seconds must be positive"))
    warmups >= 1 || throw(ArgumentError("warmups must be positive"))

    for _ in 1:warmups
        f()
    end
    GC.gc()

    trial = @benchmark $(f)() samples=samples evals=1 seconds=seconds
    estimate = minimum(trial)
    times = sort!(copy(trial.times))
    count = length(times)
    median_time = isodd(count) ? Float64(times[(count + 1) ÷ 2]) :
        (Float64(times[count ÷ 2]) + Float64(times[count ÷ 2 + 1])) / 2

    (; minimum_time_ns=Float64(estimate.time),
       median_time_ns=median_time,
       minimum_gc_time_ns=Float64(estimate.gctime),
       allocated_bytes=Int(estimate.memory),
       allocations=Int(estimate.allocs),
       samples=count,
       evals=1)
end

function _command_output(command)
    try
        strip(read(command, String))
    catch
        "unavailable"
    end
end

function _working_tree_sha256(root)
    try
        difference = read(`git -C $root diff --binary HEAD -- .`)
        untracked_text = read(
            `git -C $root ls-files --others --exclude-standard -z`, String)
        untracked = sort!(filter(!isempty, split(untracked_text, '\0')))
        buffer = IOBuffer()
        write(buffer, "tracked-diff\0", difference, "\0untracked\0")
        for relative in untracked
            path = joinpath(root, relative)
            isfile(path) || continue
            write(buffer, relative, '\0', string(filesize(path)), '\0')
            write(buffer, read(path))
        end
        bytes2hex(sha256(take!(buffer)))
    catch
        "unavailable"
    end
end

function _file_sha256(path)
    path === nothing && return "absent"
    isfile(path) || return "absent"
    try
        open(path, "r") do io
            bytes2hex(sha256(io))
        end
    catch
        "unavailable"
    end
end

function _active_environment_metadata()
    project = try
        Base.active_project()
    catch
        nothing
    end
    project_path = project === nothing ? nothing : abspath(project)
    manifest = if project_path === nothing
        nothing
    else
        try
            Base.project_file_manifest_path(project_path)
        catch
            nothing
        end
    end
    manifest_path = manifest === nothing ? nothing : abspath(manifest)
    (;
        project_path=project_path === nothing ? "absent" : project_path,
        project_sha256=_file_sha256(project_path),
        manifest_path=manifest_path === nothing ? "absent" : manifest_path,
        manifest_sha256=_file_sha256(manifest_path),
    )
end

function _startup_file_disabled()
    setting = try
        Base.JLOptions().startupfile
    catch
        return "unknown"
    end
    setting == 2 && return "true"
    setting in (0, 1) && return "false"
    "unknown"
end

"""Return ordered, machine-readable metadata for a benchmark run."""
function benchmark_metadata(; mode, requested_samples, seconds, warmups,
                            output, repository_root=pwd(),
                            extra=Pair{String,String}[])
    root = abspath(repository_root)
    status = _command_output(`git -C $root status --porcelain`)
    dirty = status == "unavailable" ? "unknown" : string(!isempty(status))
    environment = _active_environment_metadata()
    entries = Pair{String,String}[
        "schema_version" => "2",
        "generated_utc" => string(now(UTC)),
        "julia_version" => string(VERSION),
        "benchmarktools_version" => string(Base.pkgversion(BenchmarkTools)),
        "active_project_path" => environment.project_path,
        "active_project_sha256" => environment.project_sha256,
        "active_manifest_path" => environment.manifest_path,
        "active_manifest_sha256" => environment.manifest_sha256,
        "startup_file_disabled" => _startup_file_disabled(),
        "os" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH),
        "cpu_name" => string(Sys.CPU_NAME),
        "logical_cpu_threads" => string(Sys.CPU_THREADS),
        "julia_threads" => string(Threads.nthreads()),
        "blas_threads" => string(BLAS.get_num_threads()),
        "blas_config" => sprint(show, BLAS.get_config()),
        "git_commit" => _command_output(`git -C $root rev-parse HEAD`),
        "git_dirty" => dirty,
        "git_worktree_sha256" => _working_tree_sha256(root),
        "mode" => string(mode),
        "requested_samples" => string(requested_samples),
        "seconds_per_benchmark" => string(seconds),
        "warmups" => string(warmups),
        "evals_per_sample" => "1",
        "output" => abspath(output),
    ]
    append!(entries, extra)
    entries
end

function _tsv_value(value)
    value === missing && return "NA"
    value === nothing && return "NA"
    text = value isa Symbol ? String(value) : string(value)
    replace(text, '\t' => ' ', '\n' => ' ', '\r' => ' ')
end

"""Write named-tuple rows with the explicit `columns` order as TSV."""
function write_tsv(path::AbstractString, rows, columns)
    directory = dirname(abspath(path))
    mkpath(directory)
    open(path, "w") do io
        println(io, join(string.(columns), '\t'))
        for row in rows
            println(io, join((_tsv_value(getproperty(row, column))
                              for column in columns), '\t'))
        end
    end
    path
end

"""Write ordered key/value benchmark metadata as a two-column TSV file."""
function write_metadata(path::AbstractString, entries)
    directory = dirname(abspath(path))
    mkpath(directory)
    open(path, "w") do io
        println(io, "key\tvalue")
        for (key, value) in entries
            println(io, _tsv_value(key), '\t', _tsv_value(value))
        end
    end
    path
end

"""Return the sibling metadata filename for a benchmark TSV path."""
function metadata_path(path::AbstractString)
    endswith(lowercase(path), ".tsv") ? path[1:(end - 4)] * ".metadata.tsv" :
        path * ".metadata.tsv"
end

end # module
