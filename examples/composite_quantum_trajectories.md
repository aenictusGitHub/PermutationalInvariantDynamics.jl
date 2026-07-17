# Composite stochastic systems

The runnable source is
[`composite_quantum_trajectories.jl`](composite_quantum_trajectories.jl).
It demonstrates density-valued quantum-jump trajectories for a compressed PI
ensemble coupled to a finite auxiliary system.

## Model

Four identical qubits form one PI factor, while a distinguished two-level
ancilla is retained as a `FiniteOperatorBasis(2)`. The coherent background is

```math
H=g J_x\otimes\sigma_x,
```

and the monitored channel is

```math
L=J_-\otimes\sigma_-.
```

The complete ensemble generator is

```math
\mathcal L(\rho)=-i[H,\rho]+\gamma\mathcal D[L](\rho).
```

`CompositeTrajectoryPlan` receives the trace-preserving background separately
from `CompositeJumpChannel`. The background must not already contain the
monitored dissipator: the plan adds it when assembling
`composite_master_superoperator(plan)`. This explicit split prevents an
unverifiable double count of the gain channel.

## Matrix-free stochastic propagation

Every conditional state remains a `CompositePIState`. For a channel with
gain `G[rho]=L rho L'` and `Q=L'L`, the normalized no-jump drift is

```math
\dot\rho_c=\mathcal L_0(\rho_c)
-\frac{\gamma}{2}\{Q,\rho_c\}
+\lambda(\rho_c)\rho_c,
\qquad
\lambda(\rho_c)=\gamma\,\mathrm{tr}(Q\rho_c).
```

At a jump, the update is `G[rho]/tr(G[rho])`. The gain and both loss maps are
applied one tensor mode at a time. No global Kronecker superoperator and no
`2^N` ensemble Hilbert space are constructed. A workspace retains one shared
pair of full composite buffers for all channels; only small factor-fibre
scratch grows with the channel count.

Each channel hazard is integrated with the same RK4 stages as the conditional
state. If the resulting jump probability exceeds the configured cap, the
trial state is discarded and the step is retried at a smaller size. This is
important for driven rates that change appreciably inside one proposed step.

The example compares 1,024 stochastic paths with deterministic RK4 evolution
under `composite_master_superoperator(plan)`. It also demonstrates
observable-only online statistics and checks that serial and threaded batches
use identical trajectory-indexed random streams.

Run it from the repository root:

```sh
julia --project=. examples/composite_quantum_trajectories.jl
```

## Interpretation and limits

These are density-valued trajectories. Unmonitored local PI channels in the
background can make an individual path mixed, even when a fully resolved
microscopic trajectory would be pure. The backend currently supports fixed
tensor-product jump operators and scalar time-dependent rates. It does not
infer an unraveling from an arbitrary assembled superoperator.

Finite auxiliary modes require an explicit truncation. Composite weak-PI
pseudo-kets, diffusive monitoring, adaptive event-time integration,
confidence-controlled stopping, and Distributed batches are separate future
extensions. Converge the fixed time step, maximum jump probability, auxiliary
truncation, and trajectory count independently.
