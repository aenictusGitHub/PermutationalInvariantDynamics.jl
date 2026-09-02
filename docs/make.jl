using Documenter
using PermutationalInvariantDynamics

const REPOSITORY = "https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl"
const REMOTE = Documenter.Remotes.GitHub(
    "aenictusGitHub", "PermutationalInvariantDynamics.jl")
# GitHub Actions supplies the exact revision.  The branch fallback also lets a
# source checkout without Git metadata build documentation with working links.
const SOURCE_REVISION = get(ENV, "GITHUB_SHA", "main")
const PACKAGE_ROOT = normpath(pkgdir(PermutationalInvariantDynamics))
const DOCUMENTATION_CHANNEL =
    startswith(get(ENV, "GITHUB_REF", ""), "refs/tags/v") ? "stable" : "dev"
const DOCUMENTATION_CANONICAL =
    "https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/" *
    DOCUMENTATION_CHANNEL * "/"
# Tracked Markdown is also read directly on GitHub, so `$...$` is the
# canonical inline syntax. Keep that delimiter explicit instead of depending
# on Documenter's default KaTeX configuration.
const MATH_ENGINE = Documenter.KaTeX(Dict(
    :delimiters => [
        Dict(:left => raw"$", :right => raw"$", :display => false),
        Dict(:left => raw"$$", :right => raw"$$", :display => true),
        Dict(:left => raw"\[", :right => raw"\]", :display => true),
    ],
))

function undocumented_exports(mod::Module)
    if isdefined(Base.Docs, :undocumented_names)
        return getfield(Base.Docs, :undocumented_names)(mod; private=false)
    end
    filter(names(mod; all=false, imported=false)) do name
        Base.Docs.doc(Base.Docs.Binding(mod, name)) === nothing
    end
end

const DOCUMENTED_MODULES = (
    PermutationalInvariantDynamics,
    PermutationalInvariantDynamics.Models,
)
const undocumented = [
    "$(nameof(mod)).$(name)"
    for mod in DOCUMENTED_MODULES
    for name in undocumented_exports(mod)
]
isempty(undocumented) || error(
    "Every exported binding must have a docstring; missing: " *
    join(sort!(undocumented), ", "))

makedocs(sitename="PermutationalInvariantDynamics.jl",
         modules=collect(DOCUMENTED_MODULES),
         checkdocs=:exports,
         remotes=Dict(PACKAGE_ROOT => (REMOTE, SOURCE_REVISION)),
         format=Documenter.HTML(
             mathengine=MATH_ENGINE,
             canonical=DOCUMENTATION_CANONICAL,
             edit_link="main",
             repolink=REPOSITORY,
             assets=[
                 Documenter.asset(
                     "assets/model_code_generator.css";
                     class=:css,islocal=true),
                 Documenter.asset(
                     "assets/model_code_generator_core.js";
                     class=:js,islocal=true,
                     attributes=Dict(:defer=>"")),
                 Documenter.asset(
                     "assets/model_code_generator_ui.js";
                     class=:js,islocal=true,
                     attributes=Dict(:defer=>"")),
             ],
             size_threshold_warn=128*1024),
         pages=["Home"=>"index.md",
                "Start here"=>[
                    "90-second quickstart"=>"quickstart.md",
                    "Installation and environment checks"=>"installing.md",
                    "Choose the right workflow"=>"choosing_workflow.md",
                    "Build a model step by step"=>"getting_started.md",
                    "Model code generator"=>"model_code_generator.md",
                ],
                "Concepts and core workflow"=>[
                    "Framework and conventions"=>"framework.md",
                    "Architecture and efficient workflows"=>"architecture.md",
                    "API tiers and prepared analysis"=>"api_tiers.md",
                    "Results, tables, plots, and exports"=>"result_outputs.md",
                    "Reusable prepared geometry"=>"prepared_artifacts.md",
                ],
                "Dynamics workflows"=>[
                    "Streaming output"=>"streaming_output.md",
                    "Mean-field predictions"=>"meanfield.md",
                    "Diffusive monitoring"=>"diffusive_monitoring.md",
                    "Matrix-RHS trajectory cohorts"=>"batched_trajectories.md",
                    "Weak-PI pseudo-ket trajectories"=>"weak_pi_trajectories.md",
                    "Quantum regression and spectra"=>"correlations.md",
                    "Higher-order cumulant bridge"=>"cumulant_bridge.md",
                    "Composite systems"=>"composite_systems.md",
                    "Global pseudomodes and shared cavities"=>"global_pseudomodes.md",
                    "Local pseudomodes and PI supersites"=>"pseudomodes.md",
                    "PI--HEOM non-Markovian dynamics"=>"heom.md",
                    "PI--HOPS stochastic non-Markovian dynamics"=>"hops.md",
                    "Bath-correlation fitting"=>"bath_fitting.md",
                    "Counting statistics and rare events"=>"counting_statistics.md",
                ],
                "Large-scale numerical methods"=>[
                    "Matrix-free Krylov solvers"=>"matrix_free_krylov.md",
                    "No-jump-resolvent iterative solvers"=>"no_jump_iterative_solvers.md",
                    "Block, multi-shift, and recycled Krylov"=>"krylov_extensions.md",
                    "Optional accelerators"=>"accelerators.md",
                    "Prepared parameter scans"=>"parameter_scans.md",
                    "Progress and cancellation"=>"progress.md",
                    "Numerical convergence reports"=>"convergence.md",
                    "Reproducible verified experiments"=>"experiments.md",
                    "Startup latency and local sysimages"=>"startup_performance.md",
                ],
                "Analysis and visualization"=>[
                    "Symmetric pure kets and block entropy"=>"symmetric_kets_and_block_entropy.md",
                    "Nonstabilizerness"=>"nonstabilizerness.md",
                    "Genuine multipartite entanglement"=>"genuine_entanglement.md",
                    "Research utilities and control"=>"research_utilities.md",
                    "Parameter inference"=>"inference.md",
                    "Schur-block visualization"=>"schur_visualization.md",
                    "Spectral visualization"=>"spectral_visualization.md",
                    "Spin phase space"=>"spin_phase_space.md",
                    "Qudit Husimi phase space"=>"qudit_phase_space.md",
                ],
                "Examples and validation"=>[
                    "Searchable example gallery"=>"example_gallery.md",
                    "Published validation"=>"published_validation.md",
                    "Research examples"=>"research_examples.md",
                    "Performance benchmarks"=>"benchmarks.md",
                ],
                "Optional ecosystem integrations"=>"interoperability.md",
                "Mathematical reference"=>"mathematics.md",
                "Public API"=>[
                    "Complete index"=>"api_reference.md",
                    "Representations, states, and models"=>"api/representation.md",
                    "Dynamics and evolution"=>"api/dynamics.md",
                    "Observables and quantum information"=>"api/analysis.md",
                    "Stationary states, spectra, and solvers"=>"api/solvers.md",
                    "Visualization"=>"api/visualization.md",
                ],
                "Maintainer guide"=>[
                    "Release and General registration"=>"releasing.md",
                ]])

deploydocs(
    repo="github.com/aenictusGitHub/PermutationalInvariantDynamics.jl.git",
    devbranch="main",
    # Pull requests build in a read-only job; only main and version tags deploy.
    push_preview=false,
)
