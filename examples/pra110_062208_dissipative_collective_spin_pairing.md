# Dissipative collective-spin pairing model from PRA 110, 062208

Source: [`pra110_062208_dissipative_collective_spin_pairing.jl`](pra110_062208_dissipative_collective_spin_pairing.jl)

## Model

The example implements the finite-size qubit model studied by Pausch *et al.*:

```math
H=\frac{V}{Nj}(J_x^2-J_y^2),
```

together with individual and collective spin-lowering channels of respective
rates `gammaI/j` and `gammaC/(N*j)` in the package dissipator convention. It
uses `N = 8`, `V = 1`, `gammaC = 0.2`, and scans `gammaI` below, at, and above
the thermodynamic transition `gammaI + gammaC = 2abs(V)`.

The reusable constructor preserves the exact microscopic decomposition

```math
J_x^2-J_y^2=
\sum_i\left(j_{x,i}^2-j_{y,i}^2\right)
+2\sum_{i<k}\left(j_{x,i}j_{x,k}-j_{y,i}j_{y,k}\right).
```

It therefore uses one `LocalHamiltonian` and one `PBodyHamiltonian` instead of
a preassembled `DirectPIHamiltonian`. The finite PI generator is unchanged,
while the body-order information lets the same `PIModel` be passed directly
to `MeanFieldPlan`. The one-body contribution vanishes for qubits but must be
retained for the higher-spin version of the model.

## Three levels of prediction

For every parameter point, the script compares:

1. the exact correlated finite-`N` PI steady state;
2. `MeanFieldPlan(...; limit=:finite)`, the finite-`N` product-state closure;
3. `MeanFieldPlan(...; limit=:thermodynamic)` and the analytical fixed point
   following Eqs. (10) of the article.

For qubits, the thermodynamic normalized spin coordinates obey

```math
\begin{aligned}
\dot X&=-2VYZ-\gamma_I X+\gamma_CXZ,\\
\dot Y&=-2VXZ-\gamma_I Y+\gamma_CYZ,\\
\dot Z&=4VXY-2\gamma_I(Z+1)-\gamma_C(X^2+Y^2).
\end{aligned}
```

When `gammaI > 0` and `gammaI + gammaC < 2abs(V)`, the positive broken-symmetry
branch is

```math
X=\frac{\sqrt{\gamma_I(2|V|-\gamma_I-\gamma_C)}}{2|V|-\gamma_C},\qquad
Y=\mathrm{sgn}(V)X,\qquad
Z=-\frac{\gamma_I}{2|V|-\gamma_C}.
```

Above the transition it is replaced by `X = Y = 0`, `Z = -1`. These formulas
follow Eqs. (10); Eq. (11) instead describes the singular collective-only case
`gammaI = 0`, which this scan deliberately does not use.

`meanfield_stationary_state` relaxes a slightly mixed seed inside the selected
branch's basin. The script checks both convergence and the residual of the
analytical fixed point, rather than merely printing the formula. The reported
stability abscissa becomes marginal at the critical parameter, as expected.

## Comparing a symmetry-restored finite state

The unique finite-`N` steady state preserves the model's `Z2` symmetry, so its
one-body values satisfy `X = Y = 0` even below the mean-field transition. It
would therefore be misleading to compare either value directly with one of
the two mean-field branches. The script compares the branch-independent
polarization `Z` and the parity-even transverse pair order

```math
C_\perp=
\frac{\langle J_x^2\rangle+\langle J_y^2\rangle-2Nj^2}
     {2N(N-1)j^2}.
```

For a product state this reduces to `(X^2 + Y^2)/2`, and to `X^2` on the
article branch. The exact PI value is evaluated with two reusable
`CollectiveObservablePlan`s and `collective_moments`; no full `2^N` density
matrix is constructed.

The exact steady state is obtained from one sparse compilation with
`stationary_state(...; algorithm=DirectAlgorithm())`, and the finite-size gap
with `pi_liouvillian_gap`. The finite product closure is not exact finite PI
dynamics after correlations form; its difference from both the exact PI
answer and the thermodynamic closure is intentional and is printed explicitly.

For larger systems, let `compile` select its backend, use `GMRESAlgorithm()`
for the stationary state, and use a matrix-free Krylov gap method with a
convergence-tested Krylov dimension.

## Makie figure

The Makie output summarizes all three comparisons in aligned panels: the
finite-`N` Liouvillian gap, the longitudinal polarization `Z`, and the
parity-even transverse order `Cperp`. Exact PI, finite-product,
thermodynamic, and analytical predictions use consistent markers and colours;
the vertical dashed line marks the thermodynamic transition. PDF and PNG
copies are saved as `pra110_062208_dissipative_collective_spin_pairing_meanfield.*`.

## Run

```sh
julia --project=examples examples/pra110_062208_dissipative_collective_spin_pairing.jl
```

## Expected output

![Expected dissipative LMG finite-PI and mean-field comparison](../docs/src/assets/example_figures/pra110_062208_dissipative_collective_spin_pairing_meanfield.png)

The three panels compare the default finite PI calculation, finite-product
closure, thermodynamic prediction, and available analytical results.
