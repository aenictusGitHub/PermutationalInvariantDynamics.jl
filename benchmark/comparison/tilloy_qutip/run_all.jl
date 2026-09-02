function parse_arguments(arguments)
    mode = "quick"
    output = normpath(joinpath(@__DIR__, "..", "results", "tilloy_qutip"))
    samples = 3
    warmups = 1
    sizes = nothing
    python = get(ENV, "PYTHON", "python3")
    method = "fixed_point"
    index = 1
    while index <= length(arguments)
        index == length(arguments) &&
            throw(ArgumentError("missing value after $(arguments[index])"))
        argument, value = arguments[index], arguments[index + 1]
        if argument == "--mode"
            mode = value
        elseif argument == "--output"
            output = abspath(value)
        elseif argument == "--samples"
            samples = parse(Int, value)
        elseif argument == "--warmups"
            warmups = parse(Int, value)
        elseif argument == "--sizes"
            sizes = value
        elseif argument == "--python"
            python = value
        elseif argument == "--pid-method"
            method = value
        else
            throw(ArgumentError("unknown argument $argument"))
        end
        index += 2
    end
    mode in ("quick", "full") ||
        throw(ArgumentError("--mode must be quick or full"))
    samples > 0 || throw(ArgumentError("--samples must be positive"))
    warmups >= 0 || throw(ArgumentError("--warmups must be nonnegative"))
    method in ("fixed_point", "gmres") ||
        throw(ArgumentError("--pid-method must be fixed_point or gmres"))
    isnothing(sizes) && (sizes = mode == "quick" ?
        "8,16,24,32,40" : "8,16,24,32,40,48")
    (; mode, output, samples, warmups, sizes, python, method)
end

function read_tsv(path)
    lines = readlines(path)
    length(lines) > 1 || error("empty benchmark result $path")
    header = split(first(lines), '\t'; keepempty=true)
    [Dict(zip(header, split(line, '\t'; keepempty=true)))
     for line in @view(lines[2:end])]
end

function median_value(values)
    ordered = sort!(collect(values))
    count = length(ordered)
    isodd(count) ? ordered[(count + 1) ÷ 2] :
        (ordered[count ÷ 2] + ordered[count ÷ 2 + 1]) / 2
end

function summary_rows(rows)
    sizes = sort!(unique(parse(Int, row["N"]) for row in rows))
    output = NamedTuple[]
    for N in sizes
        selected(package, phase) = [row for row in rows
            if parse(Int, row["N"]) == N && row["package"] == package &&
               row["phase"] == phase]
        pid_time_to_solution = selected(
            "PermutationalInvariantDynamics", "time_to_solution")
        pid_setup = selected("PermutationalInvariantDynamics", "setup")
        pid_solve = selected("PermutationalInvariantDynamics", "solve")
        qutip_time_to_solution = selected("QuTiP", "time_to_solution")
        qutip_setup = selected("QuTiP", "setup")
        qutip_solve = selected("QuTiP", "solve")
        qutip_prepared_setup = selected("QuTiP-prepared", "setup")
        qutip_prepared_solve = selected("QuTiP-prepared", "solve")
        qutip_public_solve = selected("QuTiP-public", "solve")
        all(group -> !isempty(group),
            (pid_time_to_solution, pid_setup, pid_solve,
             qutip_time_to_solution, qutip_setup, qutip_solve,
             qutip_prepared_setup, qutip_prepared_solve)) ||
            error("missing package/phase rows for N=$N")
        dimension = parse(Int, first(pid_solve)["retained_operator_dimension"])
        dimension == parse(Int, first(qutip_solve)["retained_operator_dimension"]) ||
            error("retained dimensions differ at N=$N")
        pid_observable = parse(Float64, first(pid_solve)["observable_value"])
        qutip_observable = parse(Float64, first(qutip_solve)["observable_value"])
        tolerance = max(parse(Float64, first(pid_solve)["validation_tolerance"]),
                        parse(Float64, first(qutip_solve)["validation_tolerance"]))
        abs(pid_observable - qutip_observable) <= tolerance || error(
            "cross-package observable mismatch at N=$N: " *
            "PID=$pid_observable, QuTiP=$qutip_observable")
        for field in ("generator_probe_norm",
                      "generator_probe_checksum_real",
                      "generator_probe_checksum_imag")
            pid_value = parse(Float64, first(pid_solve)[field])
            qutip_value = parse(Float64, first(qutip_solve)[field])
            fingerprint_tolerance = 5e-6 * max(1, abs(pid_value),
                                                abs(qutip_value))
            abs(pid_value - qutip_value) <= fingerprint_tolerance || error(
                "cross-package generator fingerprint mismatch at N=$N " *
                "for $field: PID=$pid_value, QuTiP=$qutip_value")
        end
        median_seconds(group) = median_value(
            parse(Float64, row["seconds"]) for row in group)
        pid_time_to_solution_median = median_seconds(pid_time_to_solution)
        qutip_time_to_solution_median = median_seconds(qutip_time_to_solution)
        pid_setup_median = median_seconds(pid_setup)
        qutip_setup_median = median_seconds(qutip_setup)
        qutip_prepared_setup_median = median_seconds(qutip_prepared_setup)
        pid_solve_median = median_seconds(pid_solve)
        qutip_solve_median = median_seconds(qutip_solve)
        qutip_prepared_solve_median = median_seconds(qutip_prepared_solve)
        qutip_public_solve_median = isempty(qutip_public_solve) ? missing :
            median_seconds(qutip_public_solve)
        push!(output, (;
            N,
            retained_operator_dimension=dimension,
            pid_time_to_solution_median_s=pid_time_to_solution_median,
            qutip_time_to_solution_median_s=qutip_time_to_solution_median,
            pid_time_to_solution_speedup=
                qutip_time_to_solution_median / pid_time_to_solution_median,
            pid_setup_median_s=pid_setup_median,
            qutip_generator_setup_median_s=qutip_setup_median,
            qutip_prepared_setup_median_s=qutip_prepared_setup_median,
            pid_solve_median_s=pid_solve_median,
            qutip_refactor_solve_median_s=qutip_solve_median,
            pid_refactor_solve_speedup=qutip_solve_median / pid_solve_median,
            qutip_prepared_solve_median_s=qutip_prepared_solve_median,
            pid_prepared_solve_speedup=
                qutip_prepared_solve_median / pid_solve_median,
            qutip_public_solve_median_s=qutip_public_solve_median,
            pid_public_solve_speedup=ismissing(qutip_public_solve_median) ?
                missing : qutip_public_solve_median / pid_solve_median,
            pid_physical_residual_inf=parse(
                Float64, first(pid_solve)["physical_residual_inf"]),
            qutip_physical_residual_inf=parse(
                Float64, first(qutip_solve)["physical_residual_inf"]),
            pid_observable,
            qutip_observable,
        ))
    end
    output
end

function write_summary(path, rows)
    columns = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(columns), '\t'))
        for row in rows
            println(io, join((getproperty(row, key) for key in columns), '\t'))
        end
    end
end

function combine_raw(path, sources)
    header = nothing
    body = String[]
    for source in sources
        lines = readlines(source)
        isempty(lines) && error("empty benchmark result $source")
        if isnothing(header)
            header = first(lines)
        elseif first(lines) != header
            error("raw benchmark schemas differ")
        end
        append!(body, @view(lines[2:end]))
    end
    open(path, "w") do io
        println(io, header)
        foreach(line -> println(io, line), body)
    end
end

function main(arguments=ARGS)
    options = parse_arguments(arguments)
    mkpath(options.output)
    pid_output = joinpath(options.output, "pid_raw.tsv")
    qutip_output = joinpath(options.output, "qutip_raw.tsv")
    root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    pid_script = joinpath(@__DIR__, "pid_run.jl")
    qutip_script = joinpath(@__DIR__, "qutip_run.py")

    common = ["--sizes", options.sizes,
              "--samples", string(options.samples),
              "--warmups", string(options.warmups)]
    julia_command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$root $pid_script --output $pid_output $common --method $(options.method)`
    python_command = `$(options.python) $qutip_script --output $qutip_output $common`
    environment = (
        "OMP_NUM_THREADS" => "1",
        "OPENBLAS_NUM_THREADS" => "1",
        "MKL_NUM_THREADS" => "1",
        "VECLIB_MAXIMUM_THREADS" => "1",
        "NUMEXPR_NUM_THREADS" => "1",
        # QuTiP imports Matplotlib even though this benchmark does not plot.
        # Keep one disposable cache across result directories so font-cache
        # construction is neither repeated nor accidentally timed as setup.
        "MPLCONFIGDIR" => joinpath(tempdir(), "pid-qutip-matplotlib"),
        "XDG_CACHE_HOME" => joinpath(tempdir(), "pid-qutip-cache"),
    )

    println("==> PermutationalInvariantDynamics Tilloy solver")
    run(addenv(julia_command, environment...))
    println("\n==> QuTiP PIQS direct baseline")
    run(addenv(python_command, environment...))

    combined = joinpath(options.output, "raw.tsv")
    combine_raw(combined, (pid_output, qutip_output))
    rows = summary_rows(read_tsv(combined))
    summary = joinpath(options.output, "summary.tsv")
    write_summary(summary, rows)

    println("\nMedian time to solution (fresh setup + first solve; " *
            "speedup = QuTiP time / PID time):")
    println("N\tdim\tPID total (s)\tQuTiP restricted total (s)\t" *
            "PID speedup")
    for row in rows
        println(row.N, '\t', row.retained_operator_dimension, '\t',
                row.pid_time_to_solution_median_s, '\t',
                row.qutip_time_to_solution_median_s, '\t',
                row.pid_time_to_solution_speedup)
    end
    println("\nMedian prepared solve-only results:")
    println("N\tPID Tilloy (s)\tQuTiP spsolve/refactor (s)\t" *
            "QuTiP prepared splu (s)\tQuTiP public (s)")
    for row in rows
        println(row.N, '\t', row.pid_solve_median_s, '\t',
                row.qutip_refactor_solve_median_s, '\t',
                row.qutip_prepared_solve_median_s, '\t',
                row.qutip_public_solve_median_s)
    end
    println("\nRaw samples: ", combined)
    println("Summary: ", summary)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
