using Documenter
using PermutationalInvariantDynamics

const REPOSITORY = "https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl"
const REMOTE = Documenter.Remotes.GitHub(
    "aenictusGitHub", "PermutationalInvariantDynamics.jl")
# GitHub Actions supplies the exact revision.  The branch fallback also lets a
# source checkout without Git metadata build documentation with working links.
const SOURCE_REVISION = get(ENV, "GITHUB_SHA", "main")
const PACKAGE_ROOT = normpath(pkgdir(PermutationalInvariantDynamics))

const undocumented = Base.Docs.undocumented_names(
    PermutationalInvariantDynamics; private=false)
isempty(undocumented) || error(
    "Every exported binding must have a docstring; missing: " *
    join(sort!(string.(undocumented)), ", "))

makedocs(sitename="PermutationalInvariantDynamics.jl", modules=[PermutationalInvariantDynamics],
         checkdocs=:exports,
         remotes=Dict(PACKAGE_ROOT => (REMOTE, SOURCE_REVISION)),
         format=Documenter.HTML(
             canonical="https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/stable/",
             edit_link="main",
             repolink=REPOSITORY,
             size_threshold_warn=128*1024),
         pages=["Home"=>"index.md",
                "Framework introduction"=>"framework.md",
                "Published validation"=>"published_validation.md",
                "Research examples"=>"research_examples.md",
                "Architecture and workflows"=>"architecture.md",
                "Schur-block visualization"=>"schur_visualization.md",
                "Spectral visualization"=>"spectral_visualization.md",
                "Spin phase space"=>"spin_phase_space.md",
                "API tiers and prepared analysis"=>"api_tiers.md",
                "Mean-field predictions"=>"meanfield.md",
                "Streaming output"=>"streaming_output.md",
                "Diffusive monitoring"=>"diffusive_monitoring.md",
                "Weak-PI pseudo-ket trajectories"=>"weak_pi_trajectories.md",
                "Quantum regression and spectra"=>"correlations.md",
                "Higher-order cumulant bridge"=>"cumulant_bridge.md",
                "Research utilities and control"=>"research_utilities.md",
                "Composite systems"=>"composite_systems.md",
                "Matrix-free Krylov solvers"=>"matrix_free_krylov.md",
                "Mathematics"=>"mathematics.md",
                "Public API"=>[
                    "Complete index"=>"api_reference.md",
                    "Representations, states, and models"=>"api/representation.md",
                    "Dynamics and evolution"=>"api/dynamics.md",
                    "Observables and quantum information"=>"api/analysis.md",
                    "Stationary states, spectra, and solvers"=>"api/solvers.md",
                    "Visualization"=>"api/visualization.md",
                ]])

deploydocs(
    repo="github.com/aenictusGitHub/PermutationalInvariantDynamics.jl.git",
    devbranch="main",
    # Pull requests build in a read-only job; only main and version tags deploy.
    push_preview=false,
)
