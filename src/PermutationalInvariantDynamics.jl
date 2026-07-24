"""
    PermutationalInvariantDynamics

Exact and matrix-free tools for time-local open-system dynamics in the
permutationally invariant operator subspace of `N` identical `d`-level
systems. The package represents PI states and operators in Schur--Weyl
blocks, supports local, collective, and symmetric `p`-body processes, and
provides dynamics, stationary-state, spectral, information-theoretic, and
visualization routines without constructing the full `d^N` Hilbert space in
production algorithms.

Start with [`PIBasis`](@ref), [`PIModel`](@ref), [`compile`](@ref), and the
high-level commands [`solve_dynamics`](@ref), [`stationary_state`](@ref), and
[`liouvillian_spectrum`](@ref).
"""
module PermutationalInvariantDynamics

using LinearAlgebra
using SparseArrays
using Random
import SciMLBase
import LinearAlgebra: mul!, ldiv!, ishermitian
import Base: *, +, -, adjoint, copy, eltype, getindex, length, size, show

# Representation and model lowering.
include("partitions.jl")
include("gtpatterns.jl")
include("cgc.jl")
include("basis.jl")
include("tensor_indices.jl")
include("geometry.jl")
include("pbody.jl")
include("operators.jl")
include("terms.jl")
include("correlated_jumps.jl")
include("spin.jl")
include("vectorization.jl")
include("liouvillian.jl")
include("threaded_apply.jl")
include("compiled_families.jl")
include("source_protocol.jl")
include("result_protocol.jl")
include("solver_algorithms.jl")

# Iterative solvers, spectra, symmetries, and state analysis.
include("krylov.jl")
include("krylov_extensions.jl")
include("symmetries.jl")
include("spectra.jl")
include("evans.jl")
include("local_factor_trace.jl")
include("pseudomodes.jl")
include("entanglement.jl")
include("genuine_entanglement.jl")
include("observables.jl")
include("cumulants.jl")
include("information.jl")
include("nonstabilizerness.jl")
include("symmetry_information.jl")
include("phase_space.jl")
include("qudit_phase_space.jl")

# Deterministic, composite, non-Markovian, and stochastic dynamics.
include("sciml.jl")
include("meanfield.jl")
include("composite.jl")
include("evolution.jl")
include("restricted_symmetries.jl")
include("heom.jl")
include("trajectories.jl")
include("composite_trajectories.jl")
include("global_pseudomodes.jl")
include("weak_pi_trajectories.jl")
include("hops.jl")
include("diffusive.jl")
include("adaptive_ensembles.jl")
include("distributed_api.jl")

# Research workflows, diagnostics, and persistence.
include("floquet.jl")
include("response.jl")
include("correlations.jl")
include("highlevel.jl")
include("scans.jl")
include("convergence.jl")
include("populations.jl")
include("research_utilities.jl")
include("channels.jl")
include("tomography.jl")
include("checkpoints.jl")
include("control.jl")

# Dependency-free visualization data and SVG renderers.
include("visualization.jl")
include("spectral_visualization.jl")
include("phase_space_visualization.jl")

export Partition, partitions, weight, length_nonzero, removable_corners,
       addable_corners, remove_corner, add_corner, minus_plus_neighbors,
       reachable_sectors, symmetric_group_dimension, unitary_group_dimension,
       commutant_dimension, exact_binomial, exact_multinomial,
       GTPattern, gt_patterns, isvalid, shape, content,
       gt_entry, triangular_shift, OneBoxCGCache, cgc, partition_triangle,
       three_nu_symbol,
       PIBasis, PIOperator, PIState, coefficient_block, physical_block,
       each_schur_block, operator_from_schur_blocks,
       state_from_schur_blocks, sector_metadata,
       sector_view, identity_operator, maximally_mixed_state, trace, purity,
       normalize!, ispositive, isphysical, positivity_diagnostics,
       state_diagnostics, validate_state,
       FiniteOperatorBasis, CompositePIBasis, CompositePIOperator,
       CompositePIState, composite_tensor_operator, composite_tensor_state,
       composite_identity_operator, composite_trace_vector,
       CompositeReductionPlan, composite_reduced_state,
       composite_reduced_state!,
       sector_population,
       sector_populations, basis_state, sector_density_matrix, iid_pure_state,
       iid_state, thermal_state, computational_product_state, dicke_state,
       dicke_operator,
       ghz_state, spin_coherent_state, spin_matrices, collective_spin,
       collective_block, collective_operator,
       mean_local_operator, local_kernel_element, local_kernel_operator,
       OneBodyGeometry,
       PBodyGeometry, pbody_collective_block, pbody_collective_operator,
       pbody_kernel_element, pbody_kernel_operator,
       AbstractPITerm, InPlaceTimeOperator,
       LocalHamiltonian, CollectiveHamiltonian, LocalJump,
       CollectiveJump, CorrelatedLocalJumps, CorrelatedCollectiveJumps,
       DirectPIHamiltonian, DirectPIJump, PIModel,
       PBodyHamiltonian, LocalPBodyJump, CollectivePBodyJump,
       qubit_ensemble_model,
       left_superoperator, right_superoperator, sandwich_superoperator,
       commutator_superoperator, dissipator_superoperator,
       is_pi_operator, is_pi_superoperator, is_permutationally_invariant,
       liouvillian, MatrixFreeLiouvillian, LiouvillianPlan,
       LiouvillianWorkspace, CompiledPIModel, compile, apply!, apply_adjoint!,
       ThreadedLiouvillianWorkspace, threaded_apply!,
       threaded_apply_adjoint!,
       CompiledPIModelFamily, SpecializedPIModel, compile_family, specialize,
       CompositeSuperoperatorTerm, factorized_superoperator_term,
       local_superoperator_term, CompositeSuperoperator,
       CompositeSuperoperatorWorkspace,
       CompositeSuperoperatorBatchWorkspace, factor_left_superoperator,
       factor_right_superoperator, factor_sandwich_superoperator,
       composite_hamiltonian_superoperator,
       composite_dissipator_superoperator, composite_matrixfree,
       isautonomous, freeze, dynamics_problem, PISolution, state, state_at,
       EvolutionWorkspace, evolve!, time_evolve, time_evolution,
       DynamicsStreamResult, QuantumTrajectory, TrajectoryEnsembleResult,
       TrajectorySteadyStateResult,
       TrajectoryPlan, TrajectoryWorkspace,
       TrajectoryBatchWorkspace, quantum_trajectory,
       quantum_trajectories, trajectory_steady_state, trajectory_average,
       jump_statistics,
       trajectory_observable_statistics, trajectory_statistics,
       CompositeJumpChannel, CompositeTrajectoryPlan,
       CompositeTrajectoryWorkspace, CompositeTrajectoryBatchWorkspace,
       CompositeQuantumTrajectory, composite_master_superoperator,
       GlobalPseudomodeModel, global_pseudomode_model,
       shared_pseudomode_model, global_pseudomode_workspace,
       global_pseudomode_matrixfree, global_pseudomode_state,
       global_pseudomode_state!,
       WeakPIPseudoKet, weak_pi_dimension, weak_pi_density,
       weak_pi_pseudoket, weak_pi_expectation,
       WeakPIKrausBranch, WeakPIJumpRecord, WeakPIQuantumTrajectory,
       WeakPITrajectoryPlan, WeakPITrajectoryWorkspace,
       WeakPITrajectoryBatchWorkspace, weak_pi_quantum_trajectory,
       weak_pi_quantum_trajectories, weak_pi_trajectory_average,
       weak_pi_trajectory_steady_state, weak_pi_trajectory_statistics,
       WeakPIBatchMeansDiagnostics,
       HOPSBath, HOPSPlan, HOPSWorkspace, HOPSBatchWorkspace,
       HOPSRootKet, HOPSTrajectory, HOPSEnsembleResult,
       HOPSInitialEnsemble, hops_initial_ensemble,
       hops_number_auxiliaries, hops_multiindices, hops_hierarchy_metadata,
       hops_auxiliary_importances, hops_coordinate_scale,
       hops_rhs!, hops_trajectory, hops_density, hops_average,
       DiffusiveMonitor, homodyne_monitor, heterodyne_monitor,
       DiffusivePlan, DiffusiveWorkspace, DiffusiveBatchPlan,
       DiffusiveBatchWorkspace, DiffusiveTrajectory,
       diffusive_trajectory, diffusive_trajectories, diffusive_average,
       AdaptiveTrajectoryResult, adaptive_quantum_trajectories,
       adaptive_weak_pi_quantum_trajectories,
       adaptive_diffusive_trajectories,
       distributed_quantum_trajectories,
       distributed_diffusive_trajectories,
       expectation, variance, covariance, collective_expectation,
       collective_variance, collective_moments, CollectiveObservablePlan,
       OneBodyRDMWorkspace, one_body_rdm, one_body_rdm!, trace_error,
       collective_covariance, collective_covariance_matrix,
       kitagawa_ueda_squeezing, wineland_squeezing, two_body_rdm,
       two_body_expectation, connected_two_body_correlation,
       normalized_second_order_correlation,
       OrderedLocalMoments, ordered_local_moment, ordered_local_moments,
       CumulantTermPayload, CumulantModelPayload, CumulantBridgePayload,
       CumulantComparison, cumulant_model_payload, cumulant_bridge_payload,
       compare_cumulant_closure, quantumcumulants_initial_values,
       quantumcumulants_model,
       qfi_entanglement_depth, spin_squeezing_entangled,
       quantum_fisher_information, qfi, quantum_fisher_information_matrix,
       qfim,
       symmetric_logarithmic_derivatives, sld_commutator_matrix,
       mean_uhlmann_curvature, multiparameter_compatible,
       MeanFieldPlan, MeanFieldWorkspace, MeanFieldResult,
       meanfield_rhs!, meanfield_rhs, meanfield_problem,
       meanfield_evolve!, solve_meanfield,
       meanfield_jacobian, meanfield_stability,
       meanfield_stationary_state,
       meanfield_expectation, meanfield_collective_moments,
       meanfield_pbody_expectation,
       von_neumann_entropy, renyi_entropy, reduced_entropy,
       mutual_information, conditional_entropy, trace_distance, fidelity,
       bures_distance, quantum_relative_entropy, hilbert_schmidt_distance,
       StabilizerRenyiPlan, StabilizerRenyiWorkspace,
       stabilizer_renyi_entropy,
       sector_resolved_entropy, entropy_decomposition,
       sector_resolved_coherence, relative_entropy_of_coherence,
       symmetry_twirl, relative_entropy_of_asymmetry,
       relative_entropy_of_symmetry, wigner_yanase_asymmetry,
       sector_resolved_qfi, relative_entropy_decomposition,
       qfim_sector_decomposition,
       SpinPhaseSpaceData, spin_husimi_q, spin_wigner,
       QuditHusimiPlan, QuditHusimiData, qudit_husimi_q,
       hermiticity_error, minimum_sector_eigenvalue, check_generator,
       LocalFactorTracePlan, LocalFactorTraceWorkspace,
       local_factor_trace, local_factor_trace!,
       PISupersite, supersite_tensor_operator, lift_supersite_operator,
       lift_system_operator, lift_system_pbody_operator, lift_system_term,
       supersite_iid_state, supersite_product_state,
       BosonicPseudomode, PseudomodeCoupling, pseudomode_supersite,
       lift_pseudomode_operator, pseudomode_operators,
       pseudomode_coupling_terms, pseudomode_damping_terms,
       pseudomode_model, pseudomode_product_state,
       pseudomode_trace_plan, trace_pseudomodes, trace_pseudomodes!,
       negativity, logarithmic_negativity, ReductionPlan, ReductionWorkspace,
       PPTMixturePlan, PPTMixtureResult, ppt_mixture_test,
       reduced_state, reduced_state!, reduced_purity,
       reduced_purities,
       partial_transpose_spectrum, minimum_partial_transpose_eigenvalue,
       bipartition_negativities,
       charge_resolved_negativity, number_resolved_negativity,
       subduction_intertwiners,
       littlewood_richardson_coefficient,
       steady_state, krylov_steady_state, KrylovWorkspace, ArnoldiWorkspace,
       JacobiDavidsonWorkspace,
       BlockArnoldiWorkspace, block_arnoldi_spectrum,
       BlockGMRESWorkspace, block_gmres!, block_gmres,
       MultiShiftGMRESWorkspace, multishift_gmres!, multishift_gmres,
       RecycledGMRESWorkspace, recycled_gmres!, recycled_gmres,
       KrylovExpvWorkspace, krylov_expv!, krylov_expv,
       SchurSectorPreconditioner, schur_sector_preconditioner,
       preconditioner_cost,
       krylov_liouvillian_spectrum, harmonic_arnoldi_spectrum,
       implicitly_restarted_arnoldi_spectrum, jacobi_davidson_spectrum,
       liouvillian_eigenvalues, liouvillian_gap,
       pi_liouvillian_spectrum, pi_density_spectrum,
       pi_liouvillian_gap,
       pi_density_operator_spectrum, density_operator_spectrum,
       evans_uniqueness, has_unique_steady_state_evans,
       check_liouvillian_symmetry, is_liouvillian_symmetric,
       usual_liouvillian_symmetries, MatrixFreeSymmetryProjector,
       SymmetryProjectorWorkspace, matrixfree_symmetry_projector,
       JointSymmetryProjector, JointSymmetryProjectorWorkspace,
       joint_symmetry_projector,
       SymmetryCoordinateRestriction, RestrictionInvarianceReport,
       RestrictedLiouvillian, RestrictedLiouvillianWorkspace,
       diagonal_symmetry_restriction, retained_indices,
       restricted_trace_vector, restrict!, embed!, restriction_invariance,
       restriction_full_residual, restricted_steady_state,
       FloquetMap, FloquetWorkspace, FloquetBatchWorkspace,
       floquet_map, restricted_floquet_map,
       selected_floquet_multipliers,
       floquet_propagator, floquet_multipliers, floquet_exponents,
       floquet_gap, floquet_steady_state, stroboscopic_evolution,
       floquet_evolve,
       ResponseWorkspace, liouvillian_modes, resolvent_norm, adjoint_evolve,
       sensitivity_problem, sensitivity_state, classical_fisher_information,
       observable_decay_modes, integrated_correlation_time,
       steady_state_susceptibility, pseudospectral_abscissa,
       CorrelationPlan, CorrelationWorkspace, two_time_correlation,
       two_time_correlation!, delayed_second_order_correlation,
       second_order_correlation, stationary_correlation_spectrum,
       optical_spectrum, correlation_spectrum_fft,
       qfim_from_derivatives,
       estimate_basis_size, estimate_memory, basis_summary, model_summary,
       value_at,
       AbstractPIAlgorithm, AutoAlgorithm, DirectAlgorithm, SVDAlgorithm,
       EigenAlgorithm, ShiftInvertAlgorithm, GMRESAlgorithm,
       RecycledGMRESAlgorithm, ExpvAlgorithm,
       HarmonicArnoldiAlgorithm, SteadyStateResult, DynamicsResult,
       SpectrumResult, stationary_state, solve_dynamics,
       liouvillian_spectrum, diagnostics, pi_dimension,
       estimate_state_bytes, estimate_basis_bytes,
       estimate_liouvillian_bytes, estimate_geometry_bytes,
       estimate_solver_bytes, recommend_solver,
       ParameterScanPlan, ParameterScanWorkspace, ParameterScanPoint,
       ParameterScanResult, clear_parameter_scan_workspace!, parameter_scan,
       resume_parameter_scan, merge_parameter_scan_results,
       distributed_parameter_scan, parameter_scan_rows,
       parameter_scan_columns,
       HEOMBath, drude_lorentz_bath, underdamped_brownian_bath,
       heom_bath_metadata, heom_bath_residue,
       independent_local_pseudomode_model,
       HEOMPlan, HEOMWorkspace, HEOMEvolutionWorkspace, HEOMState,
       HEOMBlockPreconditioner, heom_block_preconditioner,
       heom_number_ados, heom_multiindices, heom_ado_importances,
       heom_hierarchy_metadata, heom_coordinate_scale,
       heom_initial_state, heom_thermal_state, heom_ado,
       heom_reduced_state, heom_evolve!, heom_evolve,
       heom_time_evolution, heom_problem, heom_liouvillian, heom_steady_state,
       heom_depth_convergence,
       ConvergenceStudyResult, convergence_study, convergence_estimate,
       timestep_convergence, krylov_dimension_convergence,
       hierarchy_depth_convergence, sector_cutoff_convergence,
       PopulationInvarianceReport, PopulationPlan, PopulationWorkspace,
       PopulationSolution, population_dimension, population_invariance,
       diagonal_populations, diagonal_populations!, state_from_populations,
       population_generator, evolve_populations!, solve_populations,
       stationary_populations,
       spectral_trace, PopulationCoordinate, PopulationCoordinates,
       each_population_coordinate, PopulationTransition,
       population_transitions,
       AbstractPIChannel, PIChannel, MatrixFreePIChannel,
       apply_channel!, apply_channel, compose_channels, channel_adjoint,
       identity_channel, kraus_channel, PIChannelCheck, check_pi_channel,
       pi_povm_probabilities, PIPOVMSample, sample_pi_povm,
       PITomographyResult, maximum_likelihood_tomography,
       PI_CHECKPOINT_VERSION, PIStateCheckpoint, checkpoint_state,
       save_checkpoint, load_checkpoint,
       SteadyStateGradientPlan, SteadyStateGradientWorkspace,
       SteadyStateGradientResult, implicit_steady_state_gradient,
       AdjointControlResult, checkpointed_adjoint_gradient,
       SchurBlockStructure, SchurBlockVisualization,
       schur_block_structure, visualize_schur_blocks,
       save_schur_block_visualization,
       ComplexSpectrum, SpectrumVisualization, DensitySpectrumVisualization,
       liouvillian_spectrum_data, floquet_spectrum_data,
       visualize_spectrum, visualize_liouvillian_spectrum,
       visualize_floquet_spectrum, save_spectrum_visualization,
       visualize_density_spectrum, save_density_spectrum_visualization,
       SpinPhaseSpaceVisualization, visualize_spin_phase_space,
       save_spin_phase_space_visualization

end
