using Pkg

const COMPARISON_ROOT = @__DIR__
const REPOSITORY_ROOT = normpath(joinpath(COMPARISON_ROOT, "..", ".."))

function setup_environment(name; develop_repository=false)
    environment = joinpath(COMPARISON_ROOT, name)
    println("\n==> Preparing ", environment)
    Pkg.activate(environment)
    develop_repository && Pkg.develop(PackageSpec(path=REPOSITORY_ROOT))
    Pkg.resolve()
    Pkg.instantiate()
end

setup_environment("pid"; develop_repository=true)
setup_environment("quantumoptics")
setup_environment("quantumtoolbox")

println("\nComparison environments are ready. Run:")
println("  julia --startup-file=no benchmark/comparison/run_all.jl")
