using LinearAlgebra
using Random
using PermutationalInvariantDynamics

BLAS.set_num_threads(1)

const OMEGA = 0.70
const DETUNING = 0.23
const GAMMA_DOWN = 0.31
const GAMMA_UP = 0.09
const SOLVER_ATOL = 1e-9
const SOLVER_RTOL = 1e-7
const VALIDATION_TOLERANCE = 5e-7

const COLUMNS = (
    :schema_version, :generated_unix_time, :package, :package_version,
    :dependency_versions,
    :language, :language_version, :os, :arch, :cpu, :logical_cpu_threads,
    :numerical_threads, :git_commit, :git_dirty, :N, :local_dimension,
    :physical_hilbert_dimension, :dicke_hilbert_dimension,
    :retained_operator_dimension, :embedded_liouville_dimension,
    :generator_nnz, :representation, :solver, :phase, :sample, :seconds,
    :omega, :detuning, :gamma_down, :gamma_up, :solver_atol, :solver_rtol,
    :validation_tolerance, :physical_residual_inf, :trace_error,
    :observable, :observable_value, :expected_observable,
    :observable_abs_error, :minimum_eigenvalue, :hermiticity_error,
    :iterations, :invariant_subspace_leakage, :validation_passed, :notes,
    :generator_probe_norm, :generator_probe_checksum_real,
    :generator_probe_checksum_imag,
)

function parse_arguments(arguments)
    output = nothing
    sizes = [8, 16, 24, 32, 40]
    samples = 3
    warmups = 1
    method = :fixed_point
    index = 1
    while index <= length(arguments)
        index == length(arguments) &&
            throw(ArgumentError("missing value after $(arguments[index])"))
        argument, value = arguments[index], arguments[index + 1]
        if argument == "--output"
            output = abspath(value)
        elseif argument == "--sizes"
            sizes = parse.(Int, filter(!isempty, split(value, ',')))
        elseif argument == "--samples"
            samples = parse(Int, value)
        elseif argument == "--warmups"
            warmups = parse(Int, value)
        elseif argument == "--method"
            method = Symbol(value)
        else
            throw(ArgumentError("unknown argument $argument"))
        end
        index += 2
    end
    isnothing(output) && throw(ArgumentError("--output PATH is required"))
    !isempty(sizes) && all(>(0), sizes) ||
        throw(ArgumentError("--sizes must contain positive integers"))
    samples > 0 || throw(ArgumentError("--samples must be positive"))
    warmups >= 0 || throw(ArgumentError("--warmups must be nonnegative"))
    method in (:fixed_point, :gmres) ||
        throw(ArgumentError("--method must be fixed_point or gmres"))
    (; output, sizes, samples, warmups, method)
end

function git_output(arguments...)
    try
        strip(read(`git -C $(normpath(joinpath(@__DIR__, "..", "..", ".."))) $(arguments)`, String))
    catch
        "unavailable"
    end
end

function expected_jz_per_particle()
    gamma_one = GAMMA_DOWN + GAMMA_UP
    gamma_two = gamma_one / 2
    equilibrium_z = (GAMMA_UP - GAMMA_DOWN) / gamma_one
    denominator = gamma_one * (gamma_two^2 + DETUNING^2) +
                  OMEGA^2 * gamma_two
    equilibrium_z * gamma_one * (gamma_two^2 + DETUNING^2) /
        (2 * denominator)
end

function workspace_dimensions(method)
    method === :fixed_point && return (;
        workspace_krylovdim=1,
        workspace_recycle_dim=0,
        solver_krylovdim=60,
        solver_recycle_dim=0,
    )
    method === :gmres && return (;
        workspace_krylovdim=60,
        workspace_recycle_dim=8,
        solver_krylovdim=60,
        solver_recycle_dim=8,
    )
    throw(ArgumentError("unsupported no-jump-resolvent iterative method $method"))
end

function prepare_case(N, method)
    basis = PIBasis(N, 2)
    spin = spin_matrices(2)
    local_hamiltonian = OMEGA * spin.jx + DETUNING * spin.jz
    model = PIModel(basis, (
        LocalHamiltonian(local_hamiltonian),
        LocalJump(spin.jm; rate=GAMMA_DOWN),
        LocalJump(spin.jp; rate=GAMMA_UP),
    ))
    plan = NoJumpIterativePlan(model; backend=:schur, memory_budget=Inf)
    dimensions = workspace_dimensions(method)
    workspace = NoJumpIterativeWorkspace(
        plan;
        krylovdim=dimensions.workspace_krylovdim,
        recycle_dim=dimensions.workspace_recycle_dim,
        memory_budget=Inf,
    )
    (; basis, spin, plan, workspace, dimensions)
end

function solve_case(case, method)
    no_jump_iterative_steady_state(
        case.plan;
        method,
        workspace=case.workspace,
        krylovdim=case.dimensions.solver_krylovdim,
        recycle_dim=case.dimensions.solver_recycle_dim,
        maxiter=3000,
        maxrestarts=40,
        atol=SOLVER_ATOL,
        rtol=SOLVER_RTOL,
        return_info=true,
        memory_budget=Inf,
        rng=MersenneTwister(0x5eed),
    )
end

function validate_case(case, info)
    state = info.state
    diagnostics = info.state_diagnostics
    value = real(collective_expectation(state, case.spin.jz)) / case.basis.N
    expected = expected_jz_per_particle()
    observable_error = abs(value - expected)
    residual = info.physical_residual_inf
    trace_error = abs(info.trace_error)
    minimum_eigenvalue = diagnostics.minimum_eigenvalue
    hermiticity_error = diagnostics.hermiticity_error
    dimension = length(case.basis)
    probe = ComplexF64[(sin(0.37index) + im * cos(0.19index)) /
                       sqrt(dimension) for index in 1:dimension]
    weight = ComplexF64[(cos(0.11index) + im * sin(0.29index)) /
                        sqrt(dimension) for index in 1:dimension]
    image = similar(probe)
    apply!(image, case.plan.liouvillian, probe,
           case.workspace.liouvillian)
    generator_probe_norm = norm(image)
    generator_probe_checksum = dot(weight, image)
    passed = residual <= VALIDATION_TOLERANCE &&
             trace_error <= VALIDATION_TOLERANCE &&
             observable_error <= VALIDATION_TOLERANCE &&
             minimum_eigenvalue >= -VALIDATION_TOLERANCE &&
             hermiticity_error <= VALIDATION_TOLERANCE
    passed || error(
        "PID validation failed at N=$(case.basis.N): residual=$residual, " *
        "trace error=$trace_error, observable error=$observable_error, " *
        "minimum eigenvalue=$minimum_eigenvalue, " *
        "Hermiticity error=$hermiticity_error")
    iterations = hasproperty(info, :linear_solver) ?
        info.linear_solver.iterations : info.arnoldi.iterations
    (; residual, trace_error, value, expected, observable_error,
       minimum_eigenvalue, hermiticity_error, iterations, passed,
       generator_probe_norm, generator_probe_checksum)
end

function timed_setup(N, method)
    GC.gc()
    start = time_ns()
    case = prepare_case(N, method)
    elapsed = (time_ns() - start) / 1e9
    case, elapsed
end

function timed_time_to_solution(N, method)
    GC.gc()
    start = time_ns()
    case = prepare_case(N, method)
    info = solve_case(case, method)
    elapsed = (time_ns() - start) / 1e9
    case, info, elapsed
end

function timed_solve(case, method)
    GC.gc()
    start = time_ns()
    info = solve_case(case, method)
    elapsed = (time_ns() - start) / 1e9
    info, elapsed
end

function row(case, validation, method, phase, sample, seconds)
    N = case.basis.N
    dicke_dimension = sum(N + 1:-2:1)
    status = git_output("status", "--porcelain")
    (
        schema_version=1,
        generated_unix_time=time(),
        package="PermutationalInvariantDynamics",
        package_version=string(Base.pkgversion(PermutationalInvariantDynamics)),
        dependency_versions="BLAS=$(BLAS.vendor())",
        language="Julia",
        language_version=string(VERSION),
        os=string(Sys.KERNEL),
        arch=string(Sys.ARCH),
        cpu=String(Sys.CPU_NAME),
        logical_cpu_threads=Sys.CPU_THREADS,
        numerical_threads="Julia=$(Threads.nthreads()),BLAS=$(BLAS.get_num_threads())",
        git_commit=git_output("rev-parse", "HEAD"),
        git_dirty=status == "unavailable" ? "unknown" : string(!isempty(status)),
        N,
        local_dimension=2,
        physical_hilbert_dimension=string(big(2)^N),
        dicke_hilbert_dimension=dicke_dimension,
        retained_operator_dimension=length(case.basis),
        embedded_liouville_dimension=length(case.basis),
        generator_nnz="NA_matrix_free",
        representation="complete_all_sector_PI_equation_7",
        solver="NoJumpIterative_$(method)_Schur_no_jump",
        phase=String(phase),
        sample,
        seconds,
        omega=OMEGA,
        detuning=DETUNING,
        gamma_down=GAMMA_DOWN,
        gamma_up=GAMMA_UP,
        solver_atol=SOLVER_ATOL,
        solver_rtol=SOLVER_RTOL,
        validation_tolerance=VALIDATION_TOLERANCE,
        physical_residual_inf=validation.residual,
        trace_error=validation.trace_error,
        observable="real(<Jz>)/N",
        observable_value=validation.value,
        expected_observable=validation.expected,
        observable_abs_error=validation.observable_error,
        minimum_eigenvalue=validation.minimum_eigenvalue,
        hermiticity_error=validation.hermiticity_error,
        iterations=validation.iterations,
        invariant_subspace_leakage=0.0,
        validation_passed=validation.passed,
        notes="time_to_solution is fresh setup plus first solve; setup prepares PI geometry, exact no-jump Schur factors, and a method-specific fixed-capacity workspace; solve is prepared/warmed and reuses them; timed no_jump_iterative_steady_state includes its true-residual and state-diagnostic checks",
        generator_probe_norm=validation.generator_probe_norm,
        generator_probe_checksum_real=real(validation.generator_probe_checksum),
        generator_probe_checksum_imag=imag(validation.generator_probe_checksum),
    )
end

function tsv_value(value)
    text = string(value)
    (occursin('\t', text) || occursin('\n', text)) &&
        throw(ArgumentError("TSV value contains a tab or newline"))
    text
end

function write_rows(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(COLUMNS), '\t'))
        for item in rows
            println(io, join((tsv_value(getproperty(item, key)) for key in COLUMNS), '\t'))
        end
    end
end

function main(arguments=ARGS)
    options = parse_arguments(arguments)
    rows = NamedTuple[]

    # Discard compilation and first-call initialization before any sample.
    compilation_case = prepare_case(first(options.sizes), options.method)
    solve_case(compilation_case, options.method)

    for N in options.sizes
        time_to_solution_seconds = Float64[]
        time_to_solution_case = nothing
        time_to_solution_info = nothing
        for _ in 1:options.samples
            time_to_solution_case, time_to_solution_info, seconds =
                timed_time_to_solution(N, options.method)
            push!(time_to_solution_seconds, seconds)
        end
        time_to_solution_validation =
            validate_case(time_to_solution_case, time_to_solution_info)

        setup_seconds = Float64[]
        case = nothing
        for _ in 1:options.samples
            case, seconds = timed_setup(N, options.method)
            push!(setup_seconds, seconds)
        end
        for _ in 1:options.warmups
            solve_case(case, options.method)
        end
        solve_seconds = Float64[]
        info = nothing
        for _ in 1:options.samples
            info, seconds = timed_solve(case, options.method)
            push!(solve_seconds, seconds)
        end
        validation = validate_case(case, info)
        for (sample, seconds) in enumerate(time_to_solution_seconds)
            push!(rows, row(time_to_solution_case,
                            time_to_solution_validation, options.method,
                            "time_to_solution", sample, seconds))
        end
        for (sample, seconds) in enumerate(setup_seconds)
            push!(rows, row(case, validation, options.method,
                            "setup", sample, seconds))
        end
        for (sample, seconds) in enumerate(solve_seconds)
            push!(rows, row(case, validation, options.method,
                            "solve", sample, seconds))
        end
        println("PID N=$N retained=$(length(case.basis)) " *
                "residual=$(validation.residual) " *
                "observable=$(validation.value)")
    end
    write_rows(options.output, rows)
    println("wrote ", options.output)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
