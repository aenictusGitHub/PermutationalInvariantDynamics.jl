# Dissipative discrete-time-crystal precursor

Source: [`gambetta2019_dissipative_discrete_time_crystal.jl`](gambetta2019_dissipative_discrete_time_crystal.jl)

Primary reference: F. M. Gambetta, F. Carollo, M. Marcuzzi,
J. P. Garrahan, and I. Lesanovsky, *Phys. Rev. Lett.* **122**, 015701
(2019), [arXiv:1807.10161](https://arxiv.org/abs/1807.10161).

## Model and Floquet protocol

The paper considers driven two-level Rydberg atoms with number operator

\[
n_i=\frac{I+\sigma_i^z}{2}
\]

and identical spontaneous-emission channels. During segment \(a\in\{0,R\}\),

\[
\mathcal L_a(\rho)=-i[H_a,\rho]
+\Gamma\sum_i\left(\sigma_i^-\rho\sigma_i^+
-\frac12\{\sigma_i^+\sigma_i^-,\rho\}\right),
\]

\[
H_a=\sum_i\left(\Omega_x^a\sigma_i^x+
\Omega_y^a\sigma_i^y+\Delta^a n_i\right)
+\frac{V_0}{N}\sum_{i\ne j}n_i n_j.
\]

Each period starts with the short rotation segment and then relaxes under the
base generator:

\[
\mathcal F=exp[(T-t_R)\mathcal L_0]\exp[t_R\mathcal L_R].
\]

The example uses the parameters of Fig. 2(c), in units where \(\Gamma=1\):

\[
\Omega_x^0=0.7,\quad \Omega_y^0=0,\quad \Delta^0=-3.5,
\quad V_0=6,\quad T=2,\quad t_R=0.01.
\]

## Rotation axis and normalization

The paper constructs a \(\pi\) pulse around the normalized bisector of the two
stable mean-field fixed points. With the literal ordered-pair interaction in
the Hamiltonian, the mean-field coupling is \(V=2V_0=12\). Solving the paper's
stationary equations gives

\[
\mathbf M_1=(0.4228756113,0.0733088525,-0.8973676066),
\]

\[
\mathbf M_2=(-0.4683299751,0.4753156586,-0.3345580780).
\]

Normalizing each vector before forming their bisector gives the hard-coded
axis

\[
\mathbf d=(-0.1314267564,0.4615875842,-0.8773049127).
\]

The rotation parameters then follow directly from Eq. (4) of the paper:

\[
\Omega_x^R=\frac{\pi d_x}{2t_R},\qquad
\Omega_y^R=\frac{\pi d_y}{2t_R},\qquad
\Delta^R=\frac{\pi d_z}{t_R}.
\]

There is a normalization ambiguity worth preserving rather than hiding. The
article writes \(\sum_{i\ne j}\), which counts ordered pairs, but one caption
identifies the effective mean-field coupling with \(V_0\). Read literally,
the Hamiltonian instead gives \(V\to2V_0\); this is also the normalization
consistent with bistability at the displayed point. The package's
`PBodyHamiltonian` sums unordered subsets \(i<j\), so the exact mapping of the
written Hamiltonian is

```julia
PBodyHamiltonian(kron(NUMBER, NUMBER), 2; rate=2V0 / N)
```

The local basis in the script is `[ground, Rydberg]`. Relative to the paper's
spin labels this changes the signs of both \(\sigma^y\) and \(\sigma^z\), while
leaving \(\sigma^x\) unchanged. The matrices in the source make that conversion
explicit.

## Finite-size spectral validation

For exact period doubling, the Floquet channel has a stationary multiplier
\(\varepsilon_0=1\) and a subharmonic multiplier approaching \(-1\). At finite
size its magnitude is below one, giving the decay time

\[
\tau_{\mathrm{DTC}}=-\frac{T}{\log|\varepsilon_-|}.
\]

The script uses \(N=4\), whose complete PI coordinate dimension is only 35.
It constructs the short rotation and long relaxation maps separately, avoiding
a wasteful uniform grid over the discontinuity. Results using 120/48 RK4 steps
for the relaxation/rotation segments are compared with 240/96 steps.

Independent full-Hilbert-space exponentiation of the same two \(N=4\)
generators gives

\[
\varepsilon_-=-0.32402271.
\]

The script checks the fine PI result against this reference within
\(5\times10^{-7}\), requires the multiplier step-doubling difference below
\(10^{-6}\), and also validates the Floquet stationary state. It prints the
stroboscopic \(S^x/N\) signal from the all-ground initial state. Once faster
modes have decayed, the ratio between consecutive deviations from the Floquet
stationary value is checked against \(\varepsilon_-\); its negative sign is the
period-doubled alternation.

This small-system negative multiplier is only a **finite-size precursor**, not
evidence by itself for a discrete-time-crystal phase. Gambetta *et al.* study
the increasing lifetime at \(N=12,20,28\); thermodynamic claims require such a
size-scaling analysis. The present example is deliberately a fast regression
of the PI mapping, segment order, pulse convention, and Floquet spectrum.

## Makie figure

When CairoMakie is available, the left panel shows every finite-`N` Floquet
multiplier against the unit circle and highlights both the stationary root and
the negative subharmonic multiplier. The right panel shows the stroboscopic
`Sx/N` deviation from its Floquet stationary value after the initial period,
with alternating periods colored separately. Since the highlighted multiplier has
`abs(epsilon_-) < 1`, the alternation visibly decays; the panel is explicitly
a finite-size precursor and not a phase diagnosis.

PDF and PNG versions are written with the stem
`gambetta2019_dissipative_discrete_time_crystal` in the configured
example-figure directory.

## Run

```sh
julia --project=examples examples/gambetta2019_dissipative_discrete_time_crystal.jl
```

Use `--project=.` to run only against the core environment; all assertions
remain active when the optional CairoMakie figure is skipped.

Increase both segment step counts together when changing parameters. A single
uniform integration over the full period must resolve \(t_R=0.01\) and is much
less efficient for this piecewise-constant protocol.
