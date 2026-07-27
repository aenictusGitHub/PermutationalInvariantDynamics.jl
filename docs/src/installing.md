# Installation and environment checks

Use a project environment for every calculation. This makes package versions
part of the research record and prevents an unrelated global environment from
changing a solver run.

## Current installation

Until the first registered release reaches Julia's General registry:

```julia
using Pkg
Pkg.activate("my-pi-study")
Pkg.add(url="https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl")
```

After registration, the last line becomes:

```julia
Pkg.add("PermutationalInvariantDynamics")
```

Then load and inspect the environment:

```julia
using PermutationalInvariantDynamics
using Pkg

Pkg.status()
doctor()
```

`doctor()` is read-only with respect to files and external services. It
reports the Julia and package versions, thread count, loaded optional
integrations, and runs a bounded in-memory smoke calculation. Use
`doctor(smoke_test=false)` when only the environment report is wanted.

## Run a repository example

From a cloned checkout:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/getting_started.jl
```

Run examples from the repository root. Do not use `--project=examples` unless
an example explicitly documents a separate environment; otherwise Julia may
select an old or unrelated `examples/Manifest.toml`.

The searchable [example gallery](example_gallery.md) prints the exact command
for every example. Rendering can require an optional Makie backend; numerical
assertions do not depend on a generated figure.

## Develop the package locally

```julia
using Pkg
Pkg.activate("my-pi-study")
Pkg.develop(path="/path/to/PermutationalInvariantDynamics.jl")
Pkg.instantiate()
```

`Pkg.develop` follows the working tree. Use `Pkg.status()` in saved run
metadata and commit the package revision used for published results.

## Optional integrations

Core functionality depends only on Julia standard libraries and SciMLBase.
Install an optional package in the same environment only when its feature is
needed, for example:

```julia
Pkg.add("CairoMakie")  # static plots through the Makie extension
Pkg.add("JLD2")        # native Julia result archives
Pkg.add("HDF5")        # interoperable HDF5 output
Pkg.add("Tables")      # Tables.jl views of compact result tables
```

Loading an optional dependency activates its extension. It must not change
the numerical behavior of core solvers.

## A useful problem report

Include:

```julia
versioninfo()
Pkg.status()
doctor()
```

and a minimal script, the complete error and stack trace, the selected
algorithm, memory budget, and whether the failure reproduces with one Julia
thread. Never include access tokens, private repository credentials, or
unpublished data.
