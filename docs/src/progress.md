# Progress, cancellation, and resumable work

Long calculations can report dependency-free structured events without
requiring a logging, terminal, or user-interface package. The same protocol
also provides cooperative cancellation at numerical boundaries where all
mutable scratch is in a consistent state.

## Text and structured events

Set `progress=true` for short textual updates on standard error, or pass an
`IO` to redirect them:

```julia
result = parameter_scan(plan; progress=true)

open("scan-progress.log", "w") do io
    parameter_scan(plan; progress=io)
end
```

For a notebook, GUI, or scheduler, use `on_event`:

```julia
events = ProgressEvent[]

result = parameter_scan(plan; on_event=event -> begin
    push!(events, event)
    println(event.operation, ": ", event.completed, "/", event.total)
    nothing
end)
```

A `ProgressEvent` reports the operation, stage, completed and total safe work
units, elapsed wall time, a message, and workflow-specific metadata. Event
callbacks must return `nothing` to continue or `:cancel` to request
cancellation. They should remain quick: the numerical workflow calls them
synchronously.

## Cancellation from another task

Use a shared `CancellationToken` when cancellation originates outside the
event callback:

```julia
token = CancellationToken()

task = @async parameter_scan(plan; cancellation_token=token)

# For example, a UI callback or supervisory task may do this later.
cancel!(token)
partial = fetch(task)
```

Cancellation is cooperative, not an asynchronous exception injected into a
kernel. It is observed only after a complete parameter point, integration
step, or saved-output interval. `cancel!` is idempotent and thread safe.
Call `reset_cancellation!(token)` before explicitly reusing it for a new
operation; never reset a token while an operation is still running.

## Resumable scans

`parameter_scan` is the workflow with a native partial-result contract.
Cancellation returns a `ParameterScanResult` whose completed records and
continuation seed remain valid:

```julia
token = CancellationToken()

prefix = parameter_scan(plan;
    cancellation_token=token,
    on_event=event ->
        event.stage === :advanced && event.completed == 25 ?
            :cancel : nothing)

@assert prefix.metadata.cancelled

reset_cancellation!(token)
result = resume_parameter_scan(plan, prefix;
    cancellation_token=token)
```

Serial cancellation is observed between points. Threaded cancellation stops
dispatch at the coordinator boundary, drains already active workers, and
returns only an ordered prefix. It never exposes an out-of-order partial
history.

## Workflows that raise on cancellation

Some calculations cannot return their ordinary result type until every output
slot has been initialized. They raise `OperationCancelled` instead:

| Workflow | Safe cancellation boundary | State after the exception |
|---|---|---|
| `evolve!` | complete RK4 step | caller-owned destination contains the last completed step |
| `solve_dynamics` | saved-time interval | no partial result is returned |
| `heom_evolve!` | complete nominal RK4 step | caller-owned hierarchy destination contains the last completed step, including endpoint pulses |
| `hops_trajectory` | saved-output boundary | no partially initialized trajectory is returned |

Catch `OperationCancelled` only when this is expected control flow:

```julia
try
    solve_dynamics(model, rho0, (0.0, 100.0);
        saveat=0.1, cancellation_token=token)
catch error
    error isa OperationCancelled || rethrow()
end
```

The low-level event protocol is intentionally not claimed for
`liouvillian_spectrum`, trajectory ensembles (`quantum_trajectories` and
`trajectory_steady_state`), HEOM saved-history wrappers, or HOPS ensembles.
Those algorithms require native restart-cycle or worker-coordinator event
plumbing before cancellation can preserve their output and ownership
contracts. Use their existing bounded solver options, scan point boundaries,
or an outer scheduler rather than asynchronously interrupting them.

## API

```@docs
ProgressEvent
CancellationToken
cancel!
iscancelled
reset_cancellation!
OperationCancelled
```
