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
    (:lowering,"test_fixed_gain_fusion.jl"),
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
    (:lowering,"test_pbody.jl"),
    (:dynamics,"test_floquet.jl"),
    (:dynamics,"test_evolution.jl"),
    (:nonmarkovian,"test_heom.jl"),
    (:nonmarkovian,"test_composite.jl"),
    (:analysis,"test_populations.jl"),
    (:lowering,"test_time_operators.jl"),
    (:lowering,"test_correlated_jumps.jl"),
    (:dynamics,"test_meanfield.jl"),
    (:stochastic,"test_trajectories.jl"),
    (:stochastic,"test_composite_trajectories.jl"),
    (:stochastic,"test_weak_pi_trajectories.jl"),
    (:stochastic,"test_diffusive.jl"),
    (:stochastic,"test_adaptive_ensembles.jl"),
    (:dynamics,"test_streaming_output.jl"),
    (:nonmarkovian,"test_correlations.jl"),
    (:workflows,"test_research_utilities.jl"),
    (:literature,"test_published_models.jl"),
    (:analysis,"test_entanglement.jl"),
    (:analysis,"test_quantum_fisher.jl"),
    (:analysis,"test_collective_observables.jl"),
    (:analysis,"test_cumulants.jl"),
    (:analysis,"test_qfim.jl"),
    (:analysis,"test_information.jl"),
    (:analysis,"test_large_multiplicity_analysis.jl"),
    (:visualization,"test_phase_space.jl"),
    (:visualization,"test_qudit_phase_space.jl"),
    (:representation,"test_scalar_generic.jl"),
    (:solvers,"test_advanced_diagnostics.jl"),
    (:workflows,"test_highlevel.jl"),
    (:workflows,"test_performance_safeguards.jl"),
    (:workflows,"test_compiled_families.jl"),
    (:workflows,"test_architecture.jl"),
    (:workflows,"test_scans.jl"),
    (:workflows,"test_scan_extensions.jl"),
    (:workflows,"test_convergence.jl"),
    (:visualization,"test_visualization.jl"),
    (:visualization,"test_spectral_visualization.jl"),
)

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
