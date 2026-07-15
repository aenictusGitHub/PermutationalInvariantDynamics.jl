# Published-model validation

## Additional literature validations

Three further examples reproduce finite-size reference results:

- Kitagawa and Ueda, *Phys. Rev. A* **47**, 5138 (1993): one-axis twisting,
  checked against ``\langle J_x(t)\rangle=(N/2)\cos^{N-1}(\chi t)``.
- Huelga *et al.*, *Phys. Rev. Lett.* **79**, 3865 (1997): Ramsey coherence
  under independent dephasing, checked against
  ``\langle J_x(t)\rangle=(N/2)e^{-\gamma t}``.
- Shammah *et al.*, *Phys. Rev. A* **98**, 063815 (2018): local incoherent
  pumping and emission, checked against the exact tensor-power thermal state.

The Huelga and Kitagawa--Ueda scripts compile a matrix-free model once and use
`solve_dynamics` for typed saved states. Both reuse a
`CollectiveObservablePlan` for every sampled mean spin; Kitagawa--Ueda also
reuses one `ReductionPlan` for the final one-particle purity. The Shammah
example uses the typed `stationary_state(...; return_info=true)` result and
checks `diagnostics` before comparison with the exact product state.

## Cooperative spontaneous emission (2016)

`examples/pra94_033838_superradiance.jl` implements Eqs. (3)--(5) of
Damanet, Braun, and Martin, *Phys. Rev. A* **94**, 033838 (2016). The decay
matrix is decomposed exactly as

```math
\gamma\,\mathcal D[J_-] +(\gamma_0-\gamma)\sum_i\mathcal D[\sigma_-^{(i)}].
```

The example compiles the sparse PI generator once for each correlation strength
and recreates the three curves in Fig. 6. Dense exponentiation is retained as
the independent small-`N` reference route. Tests compare 31 points on each
curve to Eqs. (41)--(43); the measured maximum absolute error is `1.4e-15` on
Julia 1.12.6.

## Dissipative LMG model (2024)

`examples/pra110_062208_lmg.jl` implements Eqs. (1)--(6) of Pausch *et al.*,
*Phys. Rev. A* **110**, 062208 (2024):

```math
H=\frac{V}{Nj}(J_x^2-J_y^2),\qquad
\dot\rho=-i[H,\rho]+\frac{\gamma_I}{j}\sum_i\mathcal D[L^{(i)}]\rho
+\frac{\gamma_C}{Nj}\mathcal D[L_C]\rho.
```

Both the spin-ladder and equal-matrix-element dissipators are covered for
qubits and qutrits. The example reports the finite-size gap across the
mean-field transition `(gammaI+gammaC)/abs(V)=2` and compares the scaled
steady-state polarization with Eq. (11). It uses a compiled sparse model, a
typed `DirectAlgorithm` stationary result, and a prepared `Jz` observable.
Finite-size values are not expected to equal the thermodynamic mean-field curve
exactly.
