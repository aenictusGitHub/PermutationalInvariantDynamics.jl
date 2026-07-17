# Pluto notebooks

From the repository root, instantiate the dedicated notebook environment and
develop the checked-out package into it:

```sh
julia --project=notebooks -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=notebooks -e 'using Pluto; Pluto.run()'
```

Then open `notebooks/pi_research_workflow.jl` in Pluto. The notebook project
contains Pluto but intentionally has no committed manifest. Developing the
repository checkout keeps the notebook synchronized with the local library
without adding Pluto to the package's runtime dependencies.
