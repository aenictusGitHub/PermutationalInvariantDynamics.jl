# Boundary time-crystal Liouvillian modes

Source: [`iemini2018_boundary_time_crystal.jl`](iemini2018_boundary_time_crystal.jl)

## Model

The collective-spin model of Iemini *et al.* has

\[
H=\omega_0 J_x,\qquad
\dot\rho=-i[H,\rho]+\frac{2\kappa}{N}\mathcal D[J_-]\rho .
\]

Only the fully symmetric Schur sector is required because both the Hamiltonian
and jump operator are collective. The script studies `ω0/κ = 0.5` and `1.5`
for `N = 8, 12, 16`.

## Solution

Build the collective Hamiltonian and jump, restrict the PI basis to the
symmetric sector, and call `compile(...; backend=:sparse)`. The high-level
`liouvillian_spectrum` command then returns the complete reduced spectrum,
ordered by real part. The eigenvalue at zero is the steady mode, the largest
nonzero real part determines the gap, and the imaginary part of the slow mode
diagnoses persistent oscillations.

The library's `pi_liouvillian_gap` routine can be used when only the gap is
needed and can exploit identified weak-symmetry blocks. Requesting the complete
dense spectrum in this small example is intentional because the literature
comparison also tracks the oscillatory branch.

## Makie figure

The optional CairoMakie figure presents the finite-size scaling against
`1/N` in three aligned panels: the Liouvillian gap, the decay rate of the slow
oscillatory mode, and its oscillation frequency. The stationary and
time-crystalline parameter ratios use consistent colours and markers, making
the gap closing and long-lived oscillatory branch directly comparable. PDF
and PNG copies are saved as `iemini2018_boundary_time_crystal.*`.

## Run

```sh
julia --project=examples examples/iemini2018_boundary_time_crystal.jl
```

The ordered finite-size results show gap closing in the oscillatory regime;
repeat with increasing `N` before drawing a thermodynamic conclusion.
