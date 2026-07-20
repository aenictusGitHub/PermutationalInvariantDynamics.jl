using LinearAlgebra
using SparseArrays
using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "..", "common.jl"))
using .ComparisonCommon

const GAMMA = 0.37
const COLLECTIVE_SIZES = (4, 8, 16, 32, 64)
const LOCAL_SIZES = (2, 4, 6, 8, 16)
const SIGMA_MINUS = ComplexF64[0 1; 0 0]

function collective_setup(N)
    basis = PIBasis(N, 2; sectors=[(N, 0)])
    state = iid_pure_state(basis, ComplexF64[0, 1])
    model = PIModel(basis, (CollectiveJump(SIGMA_MINUS; rate=GAMMA),))
    generator = liouvillian(model; representation=:sparse)
    generator isa SparseMatrixCSC || error(
        "the explicit PID Liouvillian did not return SparseMatrixCSC")
    input = copy(state.data)
    output = similar(input)
    return (;
        generator,
        input,
        output,
        basis,
        expected_spectrum=expected_emission_spectrum(N, GAMMA;
                                                     collective=true),
        representation="symmetric_schur_sector_pi",
        backend="explicit_sparse",
        action_kind="SparseArrays.mul!",
        physical_hilbert_dimension=big(2)^N,
        retained_operator_dimension=length(basis),
    )
end

function local_setup(N)
    basis = PIBasis(N, 2)
    state = iid_pure_state(basis, ComplexF64[0, 1])
    model = PIModel(basis, (LocalJump(SIGMA_MINUS; rate=GAMMA),))
    generator = liouvillian(model; representation=:sparse)
    generator isa SparseMatrixCSC || error(
        "the explicit PID Liouvillian did not return SparseMatrixCSC")
    input = copy(state.data)
    output = similar(input)
    return (;
        generator,
        input,
        output,
        basis,
        expected_spectrum=expected_emission_spectrum(N, GAMMA;
                                                     collective=false),
        representation="all_schur_sectors_pi",
        backend="explicit_sparse",
        action_kind="SparseArrays.mul!",
        physical_hilbert_dimension=big(2)^N,
        retained_operator_dimension=length(basis),
    )
end

function pid_validation(case)
    derivative = PIState(case.basis, case.output)
    block_error = 0.0
    block_scale = 0.0
    for sector in case.basis.sectors
        block = coefficient_block(derivative, sector)
        block_error = max(block_error, norm(block - block'))
        block_scale = max(block_scale, norm(block))
    end
    hermiticity_error = block_error / max(block_scale, eps(Float64))
    spectrum = pi_density_spectrum(derivative; atol=5e-11)
    signature_error = spectral_signature_error(
        spectrum.values, spectrum.degeneracies, case.expected_spectrum)
    return (;
        trace_derivative_abs=abs(trace(derivative)),
        hermiticity_relative_error=hermiticity_error,
        frobenius_norm=norm(case.output),
        spectral_signature_relative_error=signature_error,
    )
end

function main(arguments=ARGS)
    options = parse_adapter_arguments(arguments)
    package_version = string(Base.pkgversion(PermutationalInvariantDynamics))
    rows = NamedTuple[]
    for N in COLLECTIVE_SIZES
        push!(rows, benchmark_case(
            () -> collective_setup(N), pid_validation;
            track="matched_collective",
            workload="collective_emission_fully_excited",
            package="PermutationalInvariantDynamics",
            package_version,
            N,
            expected_frobenius_norm=sqrt(2) * GAMMA * N,
            notes="single spin-N/2 Schur sector; symmetric occupation lowering; sparse-first assembly",
            samples=options.samples,
            seconds=options.seconds,
        ))
    end
    for N in LOCAL_SIZES
        push!(rows, benchmark_case(
            () -> local_setup(N), pid_validation;
            track="local_emission_scaling",
            workload="independent_local_emission_fully_excited",
            package="PermutationalInvariantDynamics",
            package_version,
            N,
            expected_frobenius_norm=GAMMA * sqrt(N * (N + 1)),
            notes="complete PI operator space spanning every Schur sector",
            samples=options.samples,
            seconds=options.seconds,
        ))
    end
    write_tsv(options.output, rows)
    println("wrote ", options.output)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
