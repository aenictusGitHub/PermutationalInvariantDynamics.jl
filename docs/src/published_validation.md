# Published-model validation

## Additional literature validations

Three further examples reproduce finite-size reference results:

- Kitagawa and Ueda, *Phys. Rev. A* **47**, 5138 (1993): one-axis twisting,
  checked against $\langle J_x(t)\rangle=(N/2)\cos^{N-1}(\chi t)$.
- Huelga *et al.*, *Phys. Rev. Lett.* **79**, 3865 (1997): Ramsey coherence
  under independent dephasing, checked against
  $\langle J_x(t)\rangle=(N/2)e^{-\gamma t}$.
- Shammah *et al.*, *Phys. Rev. A* **98**, 063815 (2018): local incoherent
  pumping and emission, checked against the exact tensor-power thermal state.

The Huelga and Kitagawa--Ueda scripts compile a matrix-free model once and use
`solve_dynamics` for typed saved states. Both reuse a
`CollectiveObservablePlan` for every sampled mean spin; Kitagawa--Ueda also
reuses one `ReductionPlan` for the time-resolved one-particle purity. The
Shammah example uses the typed `stationary_state(...; return_info=true)`
result and checks `diagnostics` before comparison with the exact product state.

With the examples-only CairoMakie environment, these validations also produce
PDF/PNG summaries. Huelga overlays the Ramsey signal with its exponential law
and reports the pointwise error; Kitagawa--Ueda combines the exact mean-spin
curve with the generated one-spin mixedness; Shammah resolves the thermal
steady state by total-spin sector and physical Schur-block eigenvalue. Plotting
therefore reuses the same checked numerical arrays rather than introducing a
second calculation path.

## Non-Markovian dynamical decoupling

`examples/nonmarkovian_dynamical_decoupling.jl` follows Colin Read,
*Studying pure dephasing and dynamical decoupling: comparison of the HOPS
method with exact analytical solutions* (report, September 2023). The
supplied report gives no journal venue or DOI, so this example is described as
a literature-motivated validation rather than attributed to an unverified
publication.

The script targets the CPMG and UDD4 comparisons of Figs. 8--9 qualitatively.
It uses the report's Eq. (15) parameters

```math
\omega_c=10,\qquad
\kappa=1.5,\qquad
g=\frac{1.6\,\omega_c^2}{\kappa},\qquad
T_c=\frac{2}{7\omega_c},
```

and the associated one-pole correlation

```math
C_{\mathrm{full}}(t)=
\frac{g\kappa}{2}\exp[-(\kappa-i\omega_c)t].
```

This correlation is the whole-real-line transform of the Lorentzian. It is
not the exact positive-frequency transform of the physical zero-temperature
spectral density. Both PI--HEOM and PI--HOPS are therefore checked against
the exact ideal-pulse filter function for $C_{\mathrm{full}}$, not against the
positive-frequency curve. The latter is plotted from a separate half-line
Gauss--Legendre quadrature, with two quadrature orders compared as its
deterministic convergence check. Hierarchy depth, time-step, and trajectory
convergence cannot remove the model difference between these two curves.

`PIUnitaryPulse` applies each ideal $\pi_x$ rotation to every HEOM ADO or HOPS
auxiliary, and `HierarchyPulseSequence` defines the ordered event times.
Consequently the comparison retains hierarchy memory and one continuous
colored-noise realization across every pulse. The report's finite Gaussian
pulse cannot be reproduced literally because its pulse width and bare qubit
frequency are not specified; the example instead uses the instantaneous
pulses of the analytical derivation in Eqs. (47)--(52).

The checked default is the report's single qubit. A fully symmetric $N\gt 1$
extension remains exact in PI coordinates for one common bath coupled through
$Q=2J_z$ and one collective pulse $U_x^{\otimes N}$. It is not a validation
of independent local non-Markovian environments.

The example additionally validates the 24-edge TEDD word from Read,
Serrano-Ensástiga, and Martin, *Platonic dynamical decoupling sequences for
interacting spin systems*, *Quantum* **9**, 1661 (2025),
[DOI: 10.22331/q-2025-03-12-1661](https://doi.org/10.22331/q-2025-03-12-1661).
It checks exact timing, the uniform tetrahedral group average, and projective
closure before running repeated cycles against the nonzero one-pole bath.
Four-cycle PI--HEOM and PI--HOPS results are compared at closed-cycle outputs,
and an eight-cycle HEOM refinement checks that halving the ideal-kick edge
interval reduces the final infidelity. This finite-bath calculation is not a
scalar-filter or finite-width-pulse reproduction of the Platonic-sequence
paper.

## Quantum-trajectory literature benchmarks

`examples/quantum_trajectories.jl` turns independent spontaneous emission into
an analytical regression for the Monte Carlo wave-function method of
Dalibard, Castin, and Mølmer, *Phys. Rev. Lett.* **68**, 580 (1992), and
Mølmer, Castin, and Dalibard, *J. Opt. Soc. Am. B* **10**, 524 (1993). It
compares the event-driven PI ensemble with the exact tensor-power state
$p_e(t)=e^{-\gamma t}$, the binomial excitation and photon-count laws, and
the exact no-jump probability. Stochastic assertions are expressed in
analytical standard-error units rather than as a brittle absolute tolerance.

`examples/superradiant_quantum_trajectories.jl` implements the decay-only
specialization of Eq. (1) and the two rate ratios of Fig. 2(a,b) in Zhang,
Zhang, and Mølmer, *New J. Phys.* **20**, 112001 (2018). The default
$N=10$, 256-path run compares cavity and free-space radiation against a
deterministically exponentiated, certified 36-coordinate population generator.
It is a finite-size regression of the published model and observables; the
paper used $N=50$ and 512 paths.

The conditional records require a precise distinction. The default
density-valued backend combines the particle-unresolved local gain and gives a
generally mixed conditional PI state. The separate
`examples/weak_pi_trajectories.jl` backend resolves the same CP gain into
one-box Schur-sector Kraus branches and retains a pure direct-sum pseudo-ket.
Both reproduce the master equation and linear ensemble observables, but a
Kraus decomposition is not unique, so individual $(J,M)$ paths and
trajectory variances are not asserted to equal the article's record. Lloyd,
Ziolkowska, and Keeling's 2026 construction also includes a shared bosonic
cavity; density-valued composite jumps are available, but composite
pseudo-ket trajectory compilation remains outside the single-ensemble
backend. The weak-PI example uses the article's
$\gamma_l/\Gamma_c=1$ case and independently checks certified population
and general matrix-free PI master evolution before comparing equal-size,
equal-control stochastic batches.

When run with `--project=examples`, every standalone literature validation
produces a CairoMakie PDF/PNG figure. The trajectory panels show deterministic
curves, ensemble means, and documented uncertainty bands. The weak-PI
example additionally shows ensemble-state error, sampled $J\rightarrow J'$
events, exact representation scaling through $N=50$, and an illustrative
warmed per-path timing. Damanet and Pausch add radiated-pulse/Schur-population
and gap/mean-field summaries. Morrison--Parkins, Meiser--Holland, and the
metrology examples visualize their validated steady observables or analytical
comparisons. The boundary, interacting, PT, and Floquet time-crystal scripts
pair finite-size spectra or gaps with the relevant dynamics and retain their
finite-size caveats. Makie remains confined to the examples environment.

## Cooperative spontaneous emission (2016)

`examples/correlated_superradiance.jl` implements Eqs. (3)--(5) of
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

It then computes the altered-superradiance pulse for `N=30` and
$\Delta\gamma/\gamma_0=0.4$. Since the fully excited state remains diagonal
in the Schur/GT basis, a certified `PopulationPlan` evolves 256 physical
populations instead of all 5,456 PI coordinates. Equation (39) is evaluated
from prepared physical-block diagonals, and the density operator at the pulse
maximum is rendered as multiplicity-weighted Schur-sector populations with
Young-diagram labels. Neither calculation materializes a length-$2^{30}$
state vector or a $2^{30}\times2^{30}$ density matrix.

## Dissipative LMG model (2024)

`examples/dissipative_collective_spin_pairing.jl` implements Eqs. (1)--(6) of Pausch *et al.*,
*Phys. Rev. A* **110**, 062208 (2024):

```math
H=\frac{V}{Nj}(J_x^2-J_y^2),\qquad
\dot\rho=-i[H,\rho]+\frac{\gamma_I}{j}\sum_i\mathcal D[L^{(i)}]\rho
+\frac{\gamma_C}{Nj}\mathcal D[L_C]\rho.
```

Both the spin-ladder and equal-matrix-element dissipators are covered for
qubits and qutrits. The collective quadratic Hamiltonian is lowered exactly to
its one-body self term and symmetric two-body cross term. This produces the
same finite PI Liouvillian as the direct $J_x^2-J_y^2$ construction while
retaining the microscopic body order needed by `MeanFieldPlan`.

The qubit example reports a 21-point finite-size gap curve across the transition
`(gammaI+gammaC)/abs(V)=2` and compares three distinct predictions: the exact
correlated `N=10` PI steady state, the finite-`N` product closure, and the
thermodynamic mean-field fixed point following Eqs. (10). The last is also
checked against the analytical branch and its numerical fixed-point residual.
Equation (11) is not used: it concerns the singular collective-only case
$\gamma_I=0$.

The finite steady state restores the $\mathbb Z_2$ symmetry and consequently
has zero transverse one-body order. The comparison therefore uses the
branch-independent polarization $Z$ and the parity-even pair correlator

```math
C_\perp=\frac{\langle J_x^2\rangle+\langle J_y^2\rangle-2Nj^2}
              {2N(N-1)j^2},
```

whose product-state prediction is $(X^2+Y^2)/2$. The exact moments use
prepared collective-observable contractions. Finite PI, finite product
closure, and thermodynamic mean field are deliberately not identified with
one another after correlations develop.
