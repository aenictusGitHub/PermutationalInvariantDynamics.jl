# PI–HEOM collective dephasing

This example demonstrates the hierarchy equations of motion backend on an
eight-qubit permutationally invariant ensemble coupled collectively to an
exponentially correlated bosonic environment. It focuses on two independent
convergence questions:

- the ordinary RK4 time step;
- the HEOM hierarchy depth.

The implementation follows the unscaled finite-exponential HEOM convention
documented in [Permutationally invariant HEOM](../docs/src/heom.md). For a
general review of HEOM and exponential bath decompositions, see Y. Tanimura,
*J. Chem. Phys.* **153**, 020901 (2020),
[doi:10.1063/5.0011599](https://doi.org/10.1063/5.0011599).

## Model

The system-bath interaction is

```math
H_\mathrm{int}=J_z\otimes B,
```

and the bath correlation used for the benchmark is real and exponential,

```math
C(t)=c e^{-\nu t},\qquad c=0.30,\quad \nu=1.20.
```

For this real-pole case, `HEOMBath` uses the same pole for the conjugate
correlation and sets its right coefficient to `conj(c)=c`. Complex-pole
decompositions are automatically completed on a common conjugate-pole list;
already prepared left/right decompositions can instead pass
`right_coefficients` explicitly. See the convention page before importing a
third-party bath fit.

The system has no additional Hamiltonian or Markovian dissipator. It starts
in the product state polarized along `+x`,

```math
\rho(0)=|+x\rangle\langle+x|^{\otimes N},\qquad N=8.
```

Because `Jz` commutes with the system dynamics and the correlation is real,
the exact normalized transverse coherence is

```math
\frac{2\langle J_x(t)\rangle}{N}=e^{-g(t)},\qquad
g(t)=\frac{c}{\nu^2}\left(\nu t-1+e^{-\nu t}\right).
```

This gives a direct reference without reconstructing the `2^N` density
matrix.

## PI hierarchy construction

The executable script is [`pi_heom.jl`](pi_heom.jl). Its central setup is:

```julia
basis = PIBasis(N, 2)
spin = spin_matrices()
Jz = collective_operator(basis, spin.jz)
system = PIModel(basis, ())
bath = HEOMBath(Jz, coefficient, frequency)
rho0 = iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2))

plan = HEOMPlan(system, bath; max_depth=6)
hierarchy = heom_time_evolution(
    plan, rho0, times; steps_per_interval=4)
rho_t = heom_reduced_state(last(hierarchy))
```

With one exponential term and maximum depth `D`, there are `D+1` ADOs. Each
ADO has `length(basis)` PI coordinates. The retained HEOM dimension is thus

```math
(D+1)n_\mathrm{PI},
```

not `4^N`. For this `N=8` example, `n_PI=165`; depths 2, 4, and 6 therefore
use 495, 825, and 1155 complex hierarchy coordinates, respectively.

## Results

The script evaluates depths 2, 4, and 6 on the same time grid and checks that
the pointwise error decreases monotonically. With the supplied settings,
typical maximum errors are approximately

| maximum depth | ADOs | maximum signal error |
|---:|---:|---:|
| 2 | 3 | `2.8e-4` |
| 4 | 5 | `9.7e-8` |
| 6 | 7 | `1.3e-11` |

The root-ADO trace remains within roundoff of one. These numbers validate this
particular decomposition, time interval, and observable; they are not a
universal depth prescription. Stronger coupling, slower bath decay, complex
low-temperature decompositions, and longer propagation generally require a
deeper hierarchy.

The script also runs the stronger state-level check

```julia
depth_report = heom_depth_convergence(
    system, bath, rho0, (0.0, final_time);
    depths=(2, 4, 6), steps=240, atol=1e-7, rtol=0)
```

Although the `Jx` signal is converged at depth 6, this report deliberately
does **not** declare the complete reduced state converged: its successive
Hilbert--Schmidt differences are much larger than the pointwise error of that
single observable. This is an important practical distinction. Use an
observable-specific study when only that observable is claimed, and use
`heom_depth_convergence` before claiming convergence of the entire PI state.

When CairoMakie is available in the examples environment, the script writes a
two-panel figure showing the analytic coherence and the pointwise depth
errors. Numerical validation runs even without Makie.

## Matrix-free and stationary use

`HEOMPlan` never stores the hierarchy generator. Use

```julia
Lheom = heom_liouvillian(plan)
```

to obtain a synchronized matrix-free adapter for Krylov algorithms. For an
autonomous model with a unique stationary hierarchy,

```julia
hierarchy_ss = heom_steady_state(
    plan; krylovdim=50, maxiter=1000, atol=1e-10, rtol=1e-8)
rho_ss = heom_reduced_state(hierarchy_ss)
```

performs a trace-fixed restarted-GMRES solve. Pure dephasing alone does not
have a unique stationary state, so the executable benchmark deliberately
uses time evolution rather than a steady-state solve.

## Running

From the package root:

```bash
julia --project=. examples/pi_heom.jl
```

For production calculations, repeat the run with at least one smaller time
step and one larger hierarchy depth. The two limits control different errors
and should not be conflated.
