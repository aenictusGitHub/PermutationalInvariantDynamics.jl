module ComparisonCommon

using BenchmarkTools
using Dates
using LinearAlgebra
using SHA
using SparseArrays

export benchmark_case, expected_emission_spectrum, parse_adapter_arguments,
       as_sparse_matrix, spectral_signature_error, write_tsv

# Sparse matrix-vector application is the hot path in this comparison, but
# package setup can still call BLAS internally. Pin it explicitly so a user's
# global BLAS preference cannot make otherwise single-threaded runs differ.
BLAS.set_num_threads(1)

const COLUMNS = (
    :schema_version,
    :generated_utc,
    :track,
    :workload,
    :package,
    :package_version,
    :benchmarktools_version,
    :julia_version,
    :os,
    :arch,
    :cpu_name,
    :logical_cpu_threads,
    :julia_threads,
    :blas_vendor,
    :blas_config,
    :blas_threads,
    :benchmark_git_commit,
    :benchmark_git_dirty,
    :benchmark_worktree_sha256,
    :active_manifest_sha256,
    :requested_samples,
    :seconds_per_benchmark,
    :setup_samples,
    :apply_samples,
    :representation,
    :backend,
    :action_kind,
    :N,
    :local_dimension,
    :physical_hilbert_dimension,
    :retained_operator_dimension,
    :generator_nnz,
    :retained_bytes,
    :setup_median_ns,
    :setup_min_ns,
    :setup_allocations,
    :setup_allocated_bytes,
    :apply_median_ns,
    :apply_min_ns,
    :apply_allocations,
    :apply_allocated_bytes,
    :trace_derivative_abs,
    :hermiticity_relative_error,
    :frobenius_norm,
    :expected_frobenius_norm,
    :relative_norm_error,
    :spectral_signature_relative_error,
    :validation_passed,
    :notes,
)

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const RUN_GENERATED_UTC = string(now(UTC))

"""Return an existing CSC matrix without copying, or convert another matrix once."""
as_sparse_matrix(matrix::SparseMatrixCSC) = matrix
as_sparse_matrix(matrix) = sparse(matrix)

function _command_output(command)
    try
        strip(read(command, String))
    catch
        "unavailable"
    end
end

function _git_metadata()
    status = _command_output(`git -C $REPOSITORY_ROOT status --porcelain`)
    dirty = status == "unavailable" ? "unknown" : string(!isempty(status))
    commit = _command_output(`git -C $REPOSITORY_ROOT rev-parse HEAD`)
    worktree_sha256 = _working_tree_sha256(REPOSITORY_ROOT)
    (; commit, dirty, worktree_sha256)
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

function _active_manifest_sha256()
    project = Base.active_project()
    project === nothing && return "unavailable"
    manifest = joinpath(dirname(project), "Manifest.toml")
    isfile(manifest) || return "unavailable"
    bytes2hex(sha256(read(manifest)))
end

function parse_adapter_arguments(arguments)
    output = nothing
    samples = 20
    seconds = 1.0
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        index == length(arguments) &&
            throw(ArgumentError("missing value after $argument"))
        value = arguments[index + 1]
        if argument == "--output"
            output = abspath(value)
        elseif argument == "--samples"
            samples = parse(Int, value)
            samples > 0 || throw(ArgumentError("--samples must be positive"))
        elseif argument == "--seconds"
            seconds = parse(Float64, value)
            isfinite(seconds) && seconds > 0 ||
                throw(ArgumentError("--seconds must be finite and positive"))
        else
            throw(ArgumentError("unknown argument $argument"))
        end
        index += 2
    end
    isnothing(output) && throw(ArgumentError("--output PATH is required"))
    return (; output, samples, seconds)
end

function _measurement(f; samples, seconds)
    f() # Compile and warm this exact callable before collecting the trial.
    trial = BenchmarkTools.@benchmark $(f)() samples=samples evals=1 seconds=seconds
    middle = BenchmarkTools.median(trial)
    fastest = BenchmarkTools.minimum(trial)
    return (;
        median_ns=middle.time,
        min_ns=fastest.time,
        allocations=middle.allocs,
        allocated_bytes=middle.memory,
        samples=length(trial),
    )
end

"""Return the exact eigenvalue/count signature of the tested initial derivative."""
function expected_emission_spectrum(N::Integer, gamma::Real;
                                    collective::Bool)
    N > 0 || throw(ArgumentError("N must be positive"))
    gamma > 0 || throw(ArgumentError("gamma must be positive"))
    if collective
        dimension = big(N + 1)
        positive = gamma * N
        return ((value=-positive, count=big(1)),
                (value=zero(positive), count=dimension - 2),
                (value=positive, count=big(1)))
    end
    dimension = big(2)^N
    return ((value=-gamma * N, count=big(1)),
            (value=zero(gamma), count=dimension - N - 1),
            (value=gamma, count=big(N)))
end

"""
Compare a compressed numerical eigenspectrum with exact target values and
multiplicities. Actual roots are assigned to the nearest target; the result is
the larger of the normalized eigenvalue mismatch and count mismatch.
"""
function spectral_signature_error(values, degeneracies, expected)
    length(values) == length(degeneracies) || throw(DimensionMismatch(
        "spectrum values and degeneracies must have equal length"))
    isempty(expected) && throw(ArgumentError("expected spectrum is empty"))
    targets = [Float64(item.value) for item in expected]
    expected_counts = [BigInt(item.count) for item in expected]
    actual_counts = fill(big(0), length(expected))
    scale = max(maximum(abs, targets), eps(Float64))
    value_error = 0.0
    for (value, degeneracy) in zip(values, degeneracies)
        distances = abs.(Float64(real(value)) .- targets)
        target = argmin(distances)
        value_error = max(value_error, distances[target] / scale,
                          abs(Float64(imag(value))) / scale)
        actual_counts[target] += BigInt(degeneracy)
    end
    total = sum(expected_counts; init=big(0))
    count_error = maximum(Float64(BigFloat(abs(actual - wanted)) /
                                  BigFloat(max(total, big(1))))
                          for (actual, wanted) in
                              zip(actual_counts, expected_counts))
    max(value_error, count_error)
end

"""
Benchmark one setup closure and its warmed sparse matrix-vector action.

`setup()` must return a named tuple with `generator`, `input`, `output`,
`representation`, `backend`, `action_kind`, `physical_hilbert_dimension`, and
`retained_operator_dimension`. `validate` receives the retained case after one
application and returns trace, Hermiticity, Frobenius-norm, and analytical
spectral-signature errors. The retained-byte metric is the recursive size of
`(generator,input,output)`, with shared objects counted once.
"""
function benchmark_case(setup, validate;
                        track,
                        workload,
                        package,
                        package_version,
                        N,
                        expected_frobenius_norm,
                        notes,
                        samples,
                        seconds)
    setup_measurement = _measurement(setup; samples, seconds)
    case = setup()
    L = case.generator
    x = case.input
    y = case.output
    size(L, 2) == length(x) || throw(DimensionMismatch("generator/input mismatch"))
    size(L, 1) == length(y) || throw(DimensionMismatch("generator/output mismatch"))

    action = () -> mul!(y, L, x)
    apply_measurement = _measurement(action; samples, seconds)
    action()
    validation = validate(case)
    trace_derivative_abs = validation.trace_derivative_abs
    hermiticity_error = validation.hermiticity_relative_error
    measured_norm = validation.frobenius_norm
    signature_error = validation.spectral_signature_relative_error
    expected = Float64(expected_frobenius_norm)
    relative_error = abs(Float64(measured_norm) - expected) / max(expected, eps(expected))
    tolerance = 5e-11
    passed = isfinite(trace_derivative_abs) && isfinite(measured_norm) &&
             isfinite(hermiticity_error) && isfinite(signature_error) &&
             trace_derivative_abs <= tolerance * max(1, N) &&
             hermiticity_error <= tolerance &&
             relative_error <= tolerance && signature_error <= tolerance
    passed || error(
        "$package validation failed for $workload at N=$N: " *
        "|tr(Lrho)|=$trace_derivative_abs, Hermiticity error=$hermiticity_error, " *
        "norm=$measured_norm, expected=$expected, signature error=$signature_error")

    git = _git_metadata()

    return (;
        schema_version=3,
        generated_utc=RUN_GENERATED_UTC,
        track=String(track),
        workload=String(workload),
        package=String(package),
        package_version=String(package_version),
        benchmarktools_version=string(Base.pkgversion(BenchmarkTools)),
        julia_version=string(VERSION),
        os=string(Sys.KERNEL),
        arch=string(Sys.ARCH),
        cpu_name=String(Sys.CPU_NAME),
        logical_cpu_threads=Sys.CPU_THREADS,
        julia_threads=Threads.nthreads(),
        blas_vendor=string(BLAS.vendor()),
        blas_config=replace(sprint(show, BLAS.get_config()),
                            '\t' => ' ', '\n' => ' ', '\r' => ' '),
        blas_threads=BLAS.get_num_threads(),
        benchmark_git_commit=git.commit,
        benchmark_git_dirty=git.dirty,
        benchmark_worktree_sha256=git.worktree_sha256,
        active_manifest_sha256=_active_manifest_sha256(),
        requested_samples=Int(samples),
        seconds_per_benchmark=Float64(seconds),
        setup_samples=setup_measurement.samples,
        apply_samples=apply_measurement.samples,
        representation=String(case.representation),
        backend=String(case.backend),
        action_kind=String(case.action_kind),
        N=Int(N),
        local_dimension=2,
        physical_hilbert_dimension=string(case.physical_hilbert_dimension),
        retained_operator_dimension=Int(case.retained_operator_dimension),
        generator_nnz=nnz(L),
        retained_bytes=Base.summarysize((L, x, y)),
        setup_median_ns=setup_measurement.median_ns,
        setup_min_ns=setup_measurement.min_ns,
        setup_allocations=setup_measurement.allocations,
        setup_allocated_bytes=setup_measurement.allocated_bytes,
        apply_median_ns=apply_measurement.median_ns,
        apply_min_ns=apply_measurement.min_ns,
        apply_allocations=apply_measurement.allocations,
        apply_allocated_bytes=apply_measurement.allocated_bytes,
        trace_derivative_abs=Float64(trace_derivative_abs),
        hermiticity_relative_error=Float64(hermiticity_error),
        frobenius_norm=Float64(measured_norm),
        expected_frobenius_norm=expected,
        relative_norm_error=relative_error,
        spectral_signature_relative_error=Float64(signature_error),
        validation_passed=passed,
        notes=String(notes),
    )
end

function _tsv_value(value)
    text = string(value)
    occursin('\t', text) && throw(ArgumentError("TSV field contains a tab"))
    occursin('\n', text) && throw(ArgumentError("TSV field contains a newline"))
    return text
end

function write_tsv(path, rows)
    isempty(rows) && throw(ArgumentError("cannot write an empty comparison table"))
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(COLUMNS), '\t'))
        for row in rows
            propertynames(row) == COLUMNS ||
                throw(ArgumentError("comparison row has an inconsistent schema"))
            println(io, join((_tsv_value(getproperty(row, name)) for name in COLUMNS), '\t'))
        end
    end
    return path
end

end # module ComparisonCommon
