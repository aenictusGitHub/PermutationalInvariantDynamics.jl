using Test
using LinearAlgebra
using SparseArrays
using Random
using PermutationalInvariantDynamics

const TEST_FILES=(
    (:representation,"test_combinatorics.jl"),
    (:representation,"test_cgc.jl"),
    (:representation,"test_dense_schur_reference.jl"),
    (:lowering,"test_algebra_liouvillian.jl"),
    (:lowering,"test_collective_fastpaths.jl"),
    (:lowering,"test_fixed_gain_fusion.jl"),
    (:lowering,"test_liouvillian_properties.jl"),
    (:lowering,"test_physical_metamorphic.jl"),
    (:lowering,"test_threaded_apply.jl"),
    (:representation,"test_schur_construction.jl"),
    (:representation,"test_spin_conveniences.jl"),
    (:representation,"test_validation.jl"),
    (:representation,"test_prepared_geometry.jl"),
    (:lowering,"test_vectorization.jl"),
    (:solvers,"test_spectra.jl"),
    (:solvers,"test_krylov.jl"),
    (:solvers,"test_krylov_extensions.jl"),
    (:solvers,"test_evans.jl"),
    (:solvers,"test_symmetries.jl"),
    (:solvers,"test_restricted_symmetries.jl"),
    (:solvers,"test_automatic_symmetries.jl"),
    (:lowering,"test_pbody.jl"),
    (:dynamics,"test_floquet.jl"),
    (:dynamics,"test_evolution.jl"),
    (:nonmarkovian,"test_heom.jl"),
    (:nonmarkovian,"test_hops.jl"),
    (:nonmarkovian,"test_bath_fitting.jl"),
    (:nonmarkovian,"test_hierarchy_pulses.jl"),
    (:nonmarkovian,"test_pseudomodes.jl"),
    (:nonmarkovian,"test_global_pseudomodes.jl"),
    (:nonmarkovian,"test_composite.jl"),
    (:analysis,"test_populations.jl"),
    (:lowering,"test_time_operators.jl"),
    (:lowering,"test_correlated_jumps.jl"),
    (:dynamics,"test_meanfield.jl"),
    (:stochastic,"test_trajectories.jl"),
    (:stochastic,"test_batched_trajectories.jl"),
    (:stochastic,"test_counting_statistics.jl"),
    (:stochastic,"test_composite_trajectories.jl"),
    (:stochastic,"test_weak_pi_trajectories.jl"),
    (:stochastic,"test_diffusive.jl"),
    (:stochastic,"test_adaptive_ensembles.jl"),
    (:stochastic,"test_stochastic_cross_route.jl"),
    (:dynamics,"test_streaming_output.jl"),
    (:nonmarkovian,"test_correlations.jl"),
    (:workflows,"test_research_utilities.jl"),
    (:workflows,"test_experiments.jl"),
    (:workflows,"test_inference.jl"),
    (:workflows,"test_examples.jl"),
    (:workflows,"test_examples_catalog.jl"),
    (:workflows,"test_model_code_generator.jl"),
    (:workflows,"test_model_code_generator_productization.jl"),
    (:workflows,"test_benchmark_sources.jl"),
    (:literature,"test_published_models.jl"),
    (:analysis,"test_local_factor_trace.jl"),
    (:analysis,"test_entanglement.jl"),
    (:analysis,"test_prepared_artifacts.jl"),
    (:analysis,"test_genuine_entanglement.jl"),
    (:analysis,"test_quantum_fisher.jl"),
    (:analysis,"test_collective_observables.jl"),
    (:analysis,"test_cumulants.jl"),
    (:analysis,"test_qfim.jl"),
    (:analysis,"test_information.jl"),
    (:analysis,"test_nonstabilizerness.jl"),
    (:analysis,"test_large_multiplicity_analysis.jl"),
    (:analysis,"test_analysis_dense_oracles.jl"),
    (:visualization,"test_phase_space.jl"),
    (:visualization,"test_qudit_phase_space.jl"),
    (:representation,"test_scalar_generic.jl"),
    (:solvers,"test_advanced_diagnostics.jl"),
    (:workflows,"test_highlevel.jl"),
    (:workflows,"test_performance_safeguards.jl"),
    (:workflows,"test_compiled_families.jl"),
    (:workflows,"test_productization_performance.jl"),
    (:workflows,"test_accelerators.jl"),
    (:workflows,"test_architecture.jl"),
    (:workflows,"test_scans.jl"),
    (:workflows,"test_scan_extensions.jl"),
    (:workflows,"test_convergence.jl"),
    (:visualization,"test_visualization.jl"),
    (:visualization,"test_spectral_visualization.jl"),
)

const LISTED_TEST_FILES=last.(TEST_FILES)
const DISCOVERED_TEST_FILES=sort!(filter(
    name->startswith(name,"test_")&&endswith(name,".jl"),
    readdir(@__DIR__)))
length(unique(LISTED_TEST_FILES))==length(LISTED_TEST_FILES)||
    error("TEST_FILES contains duplicate entries")
missing_test_files=setdiff(DISCOVERED_TEST_FILES,LISTED_TEST_FILES)
stale_test_files=setdiff(LISTED_TEST_FILES,DISCOVERED_TEST_FILES)
isempty(missing_test_files)||error(
    "test files are not registered in TEST_FILES: $missing_test_files")
isempty(stale_test_files)||error(
    "TEST_FILES contains missing files: $stale_test_files")

const AVAILABLE_TEST_GROUPS=sort!(unique(first.(TEST_FILES)))
const REQUESTED_TEST_GROUPS=let raw=strip(get(ENV,"PID_TEST_GROUPS",""))
    isempty(raw)||lowercase(raw)=="all" ? Set(AVAILABLE_TEST_GROUPS) :
        Set(Symbol(strip(lowercase(name))) for name in split(raw,','))
end
unknown=setdiff(REQUESTED_TEST_GROUPS,Set(AVAILABLE_TEST_GROUPS))
isempty(unknown)||error("unknown PID_TEST_GROUPS entries $(sort!(collect(unknown))); " *
    "available groups are $AVAILABLE_TEST_GROUPS")

for (group,file) in TEST_FILES
    group in REQUESTED_TEST_GROUPS||continue
    include(file)
end
