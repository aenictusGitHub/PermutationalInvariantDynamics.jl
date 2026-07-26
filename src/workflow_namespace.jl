"""
    Workflow

Curated, stable entry points for constructing, solving, validating, and
analyzing PI models. The parent module retains the complete expert API;
`Workflow` is intentionally smaller and is the recommended namespace for new
research scripts whose imports should remain easy to audit.

The namespace contains no wrappers or mutable state. Its bindings are the
same functions and types exported by `PermutationalInvariantDynamics`, so
objects move between the curated and expert interfaces without conversion.
"""
module Workflow

import ..PermutationalInvariantDynamics:
    PIBasis,PIModel,PIState,PIOperator,
    LocalHamiltonian,CollectiveHamiltonian,LocalJump,CollectiveJump,
    PBodyHamiltonian,LocalPBodyJump,CollectivePBodyJump,
    CorrelatedLocalJumps,CorrelatedCollectiveJumps,
    compile,compile_family,compile_affine_family,specialize,
    PreparedGeometryBundle,PreparationCache,prepare_geometry,
    validate_prepared_geometry,onebody_geometry,pbody_geometry,
    prepared_reductions,prepare_geometry!,evict_prepared_geometry!,
    clear_preparation_cache!,preparation_cache_summary,
    ReductionPlanSet,ReductionWorkspaceSet,reduction_plan,
    reduced_states,reduced_purities,bipartition_negativities,
    stationary_state,solve_dynamics,liouvillian_spectrum,diagnostics,
    recommend_solver,
    AutoAlgorithm,DirectAlgorithm,GMRESAlgorithm,RecycledGMRESAlgorithm,
    HarmonicArnoldiAlgorithm,
    PIExperiment,VerificationSpec,RefinementSpec,plan_experiment,
    explain_experiment,verified_solve,save_experiment,load_experiment,
    computational_product_state,iid_state,iid_pure_state,dicke_state,
    ghz_state,spin_coherent_state,maximally_mixed_state,
    expectation,variance,collective_expectation,collective_variance,
    one_body_rdm,reduced_state,reduced_purity,negativity,
    quantum_fisher_information,quantum_fisher_information_matrix,
    validate_state,state_diagnostics,
    TrajectoryPlan,TrajectoryBatchWorkspace,quantum_trajectories,
    trajectory_steady_state,adaptive_quantum_trajectories,
    TiltedLiouvillianPlan,counting_scgf,counting_cumulants,
    pseudomode_model,global_pseudomode_model,
    HEOMBath,HEOMPlan,heom_evolve,heom_steady_state,
    HOPSBath,HOPSPlan,hops_trajectory,hops_average,
    fit_bath_correlation,fit_bath_from_spectral_density,
    LeastSquaresInferenceProblem,steady_state_inference_problem,
    parameter_identifiability,fit_parameters,
    ParameterScanPlan,ParameterScanWorkspace,parameter_scan,
    Models

export PIBasis,PIModel,PIState,PIOperator,
       LocalHamiltonian,CollectiveHamiltonian,LocalJump,CollectiveJump,
       PBodyHamiltonian,LocalPBodyJump,CollectivePBodyJump,
       CorrelatedLocalJumps,CorrelatedCollectiveJumps,
       compile,compile_family,compile_affine_family,specialize,
       PreparedGeometryBundle,PreparationCache,prepare_geometry,
       validate_prepared_geometry,onebody_geometry,pbody_geometry,
       prepared_reductions,prepare_geometry!,evict_prepared_geometry!,
       clear_preparation_cache!,preparation_cache_summary,
       ReductionPlanSet,ReductionWorkspaceSet,reduction_plan,
       reduced_states,reduced_purities,bipartition_negativities,
       stationary_state,solve_dynamics,liouvillian_spectrum,diagnostics,
       recommend_solver,
       AutoAlgorithm,DirectAlgorithm,GMRESAlgorithm,RecycledGMRESAlgorithm,
       HarmonicArnoldiAlgorithm,
       PIExperiment,VerificationSpec,RefinementSpec,plan_experiment,
       explain_experiment,verified_solve,save_experiment,load_experiment,
       computational_product_state,iid_state,iid_pure_state,dicke_state,
       ghz_state,spin_coherent_state,maximally_mixed_state,
       expectation,variance,collective_expectation,collective_variance,
       one_body_rdm,reduced_state,reduced_purity,negativity,
       quantum_fisher_information,quantum_fisher_information_matrix,
       validate_state,state_diagnostics,
       TrajectoryPlan,TrajectoryBatchWorkspace,quantum_trajectories,
       trajectory_steady_state,adaptive_quantum_trajectories,
       TiltedLiouvillianPlan,counting_scgf,counting_cumulants,
       pseudomode_model,global_pseudomode_model,
       HEOMBath,HEOMPlan,heom_evolve,heom_steady_state,
       HOPSBath,HOPSPlan,hops_trajectory,hops_average,
       fit_bath_correlation,fit_bath_from_spectral_density,
       LeastSquaresInferenceProblem,steady_state_inference_problem,
       parameter_identifiability,fit_parameters,
       ParameterScanPlan,ParameterScanWorkspace,parameter_scan,
       Models

end
