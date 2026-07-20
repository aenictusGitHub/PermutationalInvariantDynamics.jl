using LinearAlgebra
using SparseArrays
using QuantumToolbox

include(joinpath(@__DIR__, "..", "common.jl"))
using .ComparisonCommon

const GAMMA = 0.37
const COLLECTIVE_SIZES = (4, 8, 16, 32, 64)
const LOCAL_SIZES = (2, 4, 6)

function collective_setup(N)
    lowering = jmat(N / 2, Val(:-))
    state = ket2dm(spin_state(N / 2, N / 2))
    generator_object = liouvillian(nothing, (sqrt(GAMMA) * lowering,))
    generator = as_sparse_matrix(generator_object.data)
    input = collect(vec(state.data))
    output = similar(input)
    return (;
        generator,
        input,
        output,
        expected_spectrum=expected_emission_spectrum(N, GAMMA;
                                                     collective=true),
        representation="fixed_spin_irrep",
        backend="explicit_sparse",
        action_kind="SparseArrays.mul!",
        physical_hilbert_dimension=big(2)^N,
        retained_operator_dimension=(N + 1)^2,
    )
end

function local_setup(N)
    lowering = sigmam()
    identity = qeye(2)
    jumps = ntuple(N) do particle
        tensor(ntuple(site -> site == particle ? lowering : identity, N)...)
    end
    # QuantumToolbox orders spin states as m=+1/2,-1/2, so its lowering
    # operator acts on the first (zero-based fock index 0) basis vector.
    excited_one = fock(2, 0; sparse=Val(true))
    excited = tensor(ntuple(_ -> excited_one, N)...)
    state = ket2dm(excited)
    scaled_jumps = ntuple(i -> sqrt(GAMMA) * jumps[i], N)
    generator_object = liouvillian(nothing, scaled_jumps)
    generator = as_sparse_matrix(generator_object.data)
    input = collect(vec(state.data))
    output = similar(input)
    dimension = 1 << N
    return (;
        generator,
        input,
        output,
        expected_spectrum=expected_emission_spectrum(N, GAMMA;
                                                     collective=false),
        representation="full_hilbert",
        backend="explicit_sparse",
        action_kind="SparseArrays.mul!",
        physical_hilbert_dimension=big(2)^N,
        retained_operator_dimension=dimension^2,
    )
end

function full_space_validation(case)
    dimension = isqrt(case.retained_operator_dimension)
    derivative = reshape(case.output, dimension, dimension)
    hermiticity_error = norm(derivative - derivative') /
        max(norm(derivative), eps(Float64))
    values = eigvals(Hermitian((derivative + derivative') / 2))
    signature_error = spectral_signature_error(
        values, fill(big(1), length(values)), case.expected_spectrum)
    return (;
        trace_derivative_abs=abs(tr(derivative)),
        hermiticity_relative_error=hermiticity_error,
        frobenius_norm=norm(derivative),
        spectral_signature_relative_error=signature_error,
    )
end

function main(arguments=ARGS)
    options = parse_adapter_arguments(arguments)
    package_version = string(Base.pkgversion(QuantumToolbox))
    rows = NamedTuple[]
    for N in COLLECTIVE_SIZES
        push!(rows, benchmark_case(
            () -> collective_setup(N), full_space_validation;
            track="matched_collective",
            workload="collective_emission_fully_excited",
            package="QuantumToolbox",
            package_version,
            N,
            expected_frobenius_norm=sqrt(2) * GAMMA * N,
            notes="jmat(N/2), a single fixed-spin irrep",
            samples=options.samples,
            seconds=options.seconds,
        ))
    end
    for N in LOCAL_SIZES
        push!(rows, benchmark_case(
            () -> local_setup(N), full_space_validation;
            track="local_emission_scaling",
            workload="independent_local_emission_fully_excited",
            package="QuantumToolbox",
            package_version,
            N,
            expected_frobenius_norm=GAMMA * sqrt(N * (N + 1)),
            notes="full-Hilbert baseline; intentionally limited to N<=6",
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
