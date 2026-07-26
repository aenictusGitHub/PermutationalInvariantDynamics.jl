# Startup latency and local sysimages

Package loading and first-call compilation are separate from steady-state or
trajectory runtime. The benchmark suite measures them separately; warmed
kernel timings should never be presented as time-to-first-result.

For repeated interactive work on one machine, an optional PackageCompiler
sysimage can precompile the common PI workflow without adding PackageCompiler
to this package's dependencies:

```sh
mkdir -p /tmp/pid-sysimage
julia --project=/tmp/pid-sysimage -e \
  'using Pkg; Pkg.add("PackageCompiler"); Pkg.develop(path=pwd())'
julia --project=/tmp/pid-sysimage scripts/build_sysimage.jl \
  --output /tmp/PermutationalInvariantDynamics.dylib
```

On Linux, use a `.so` output; on Windows, use `.dll`. Start Julia with

```sh
julia -J /tmp/PermutationalInvariantDynamics.dylib --project=.
```

The workload covers representation construction, matrix-free forward and
adjoint application, deterministic dynamics, a stationary solve, collective
observables, reductions, and one fixed-step trajectory. It uses small objects
and writes no result. A sysimage is platform-, Julia-, CPU-target-, and
environment-specific: rebuild it after changing Julia, package versions, or
the local checkout. It does not change numerical algorithms, precision,
memory guards, or convergence requirements.

For a portable benchmark rather than a local latency optimization, use the
cold-start and end-to-end scripts described in [Performance
benchmarks](benchmarks.md).
