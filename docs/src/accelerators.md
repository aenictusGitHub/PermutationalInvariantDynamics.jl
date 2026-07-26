# Optional accelerators

The package treats accelerator support as an explicit prepared backend, not as
an automatic array conversion. This is important for PI calculations: a
matrix-free Schur-block action is often more memory-efficient on the CPU than
materializing its PI-coordinate matrix merely to move it to a GPU.

## Current support boundary

The stable core exposes capability and resource preflight:

```julia
compiled = compile(model; backend=:matrixfree)

capability = accelerator_capability(:cuda)
report = accelerator_preflight(
    compiled;
    rhs_columns=16,
    rhs_kind=:matrix,
    memory_budget=2 * 1024^3,
    device_memory_budget=1024^3,
)

capability.functional
report.ready
report.issues
report.combined_peak_bytes
report.device_peak_bytes
```

The core does **not** currently claim a functional CUDA backend. A CUDA
package being installed does not prove that a compatible device, driver,
sparse library, scalar precision, and tested multiplication path are
available. Consequently,

```julia
accelerate(compiled; backend=:cuda)
```

raises before materialization unless a tested optional extension reports a
functional backend.

This conservative boundary prevents two failure modes:

- silently constructing a large sparse PI Liouvillian on the host before
  discovering that the device cannot accept it;
- silently copying a vector between host and device for every matrix-vector
  product, which can make an apparently accelerated Krylov solve slower.

## Contract for a future CUDA extension

A CUDA implementation must remain deliberately narrow:

1. accept an autonomous, already prepared `LiouvillianPlan`,
   `CompiledPIModel`, or `SpecializedPIModel`;
2. materialize the same exact-support sparse PI matrix used by the CPU sparse
   backend, under the normal memory guard;
3. upload that matrix once, with checked 32-bit sparse indices;
4. support only `Float32`, `Float64`, `ComplexF32`, and `ComplexF64`;
5. multiply device-resident vector and matrix right-hand sides without an
   implicit transfer;
6. reject the general matrix-free Schur kernels rather than presenting a
   hidden CPU fallback as GPU execution; and
7. pass forward, adjoint, vector-RHS, and matrix-RHS comparisons on a GPU CI
   runner before `functional=true` is exposed.

## Understanding the preflight

`accelerator_preflight` is allocation-light and never initializes a device. It
retains the exact source basis and prepared scalar type in the report.

When the source already owns a sparse operator, the report uses its exact
nonzero count. For a matrix-free source it uses the same structural nonzero
upper bound as guarded sparse materialization. The combined peak includes:

- the already retained prepared source;
- simultaneous host sparse assembly and result storage, when materialization
  is required;
- device sparse values, row indices, and column pointers; and
- both input and output device arrays for the requested RHS width.

`memory_budget` guards that combined peak. `device_memory_budget` separately
guards device-resident storage. `Inf` is the only explicit opt-out.

`rhs_kind=:auto` selects `:vector` when `rhs_columns==1` and `:matrix`
otherwise. Use `rhs_kind=:matrix, rhs_columns=1` when the backend must receive
a one-column matrix rather than a vector. `rhs_kind=:vector` requires exactly
one column. The selected kind is recorded in `report.rhs_kind`.

An unavailable backend is returned as a report issue, rather than being
confused with a model incompatibility. Typical issue symbols include
`:backend_unavailable`, `:nonautonomous_source`,
`:sparse_materialization_unsupported`, `:unsupported_scalar_type`,
`:sparse_index_overflow`, `:rhs_kind_unsupported`,
`:unsupported_transfer_policy`,
`:memory_budget_exceeded`, and `:device_memory_budget_exceeded`.

A functional extension is accepted only when it declares
`transfer_policy=:explicit_once`. Any capability that would perform implicit
host/device transfers during multiplication is rejected at preflight.

## When acceleration is likely to help

Sparse GPU multiplication is most plausible when the same Liouvillian is
applied repeatedly to a wide matrix of right-hand sides. It is usually a poor
fit for a single small PI vector, a driven model that must be rebuilt
frequently, or a model whose exact sparse materialization is much larger than
its matrix-free Schur workspace. Always benchmark full time to solution,
including sparse construction and upload.

## API

```@docs
AcceleratorCapability
AcceleratorPreflight
accelerator_capability
accelerator_preflight
accelerate
```
