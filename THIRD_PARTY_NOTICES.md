# Third-party notices

PermutationalInvariantDynamics.jl as a combined work is distributed under
`GPL-3.0-only`. This file records third-party expression incorporated or
adapted in the repository and preserves the notices required by its compatible
upstream license. Scientific citations alone are recorded in `PROVENANCE.md`
and the relevant documentation.

Runtime dependencies and optional extensions declared in the root
`Project.toml` are loaded from their own packages; their source is not vendored
here. Their licenses still apply to those packages and to any binary
distribution that includes them. Documentation, quality, benchmark, example,
and optional-test environments have additional development-only dependencies;
audit their exact resolution as well if a distribution bundles those tools.

## QuTiP Drude--Lorentz Padé construction

The tridiagonal eigensystem and residue-weight construction in
`src/heom.jl::_drude_pade_parameters` was adapted, translated to Julia, and
modified for explicit scalar precision, checked indexing, memory budgets,
finiteness validation, and the package's HEOM conventions from QuTiP's
open-source HEOM bath implementation:

- project: QuTiP, Quantum Toolbox in Python;
- upstream file: `qutip/core/environment.py`;
- upstream routines: `_kappa_epsilon`, `_calc_eps`, and `_calc_chi`;
- audited revision:
  [`e5dbb0195fdbf37fb39d4e52e27c80594f8eb655`](https://github.com/qutip/qutip/blob/e5dbb0195fdbf37fb39d4e52e27c80594f8eb655/qutip/core/environment.py#L1203-L1294);
- upstream license:
  [`LICENSE.txt` at that revision](https://github.com/qutip/qutip/blob/e5dbb0195fdbf37fb39d4e52e27c80594f8eb655/LICENSE.txt).

The following BSD 3-Clause notice applies to that upstream material.

<!-- SPDX-SnippetBegin -->
<!-- SPDX-SnippetCopyrightText: 2011-2026 QuTiP developers and contributors -->
<!-- SPDX-License-Identifier: BSD-3-Clause -->

> Copyright (c) 2011 to 2026 inclusive, QuTiP developers and contributors.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its contributors
>    may be used to endorse or promote products derived from this software
>    without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

<!-- SPDX-SnippetEnd -->

## Platonic dynamical-decoupling sequence data

The tetrahedral, octahedral, and icosahedral pulse words and their two
axis--angle generators in the marked snippet of `src/hierarchy_pulses.jl` are
transcribed from:

- Colin Read, Eduardo Serrano-Ensástiga, and John Martin, “Platonic dynamical
  decoupling sequences for interacting spin systems,” *Quantum* **9**, 1661
  (2025), [DOI 10.22331/q-2025-03-12-1661](https://doi.org/10.22331/q-2025-03-12-1661);
- original licensed material:
  <https://quantum-journal.org/papers/q-2025-03-12-1661/>.

Quantum publishes the article under the Creative Commons Attribution 4.0
International license (`CC-BY-4.0`). The local adaptation encodes the
published pulse words as `a`/`b` strings, expresses the published axes and
angles in caller-selected floating precision, and integrates them with
package-authored validation, PI lowering, scheduling, HEOM/HOPS application,
and tests. The complete license is in `LICENSES/CC-BY-4.0.txt`. Copyright in
the licensed article remains with its authors or other recorded holders. No
endorsement is implied.

## Runtime and extension dependency license inventory

This table is a release-review aid, not a substitute for the exact notices
shipped by a concrete dependency resolution or native artifact.

| Dependency | Role | Upstream license |
|---|---|---|
| Julia standard libraries (`LinearAlgebra`, `Random`, `SparseArrays`, `Distributed`) | core/optional runtime | MIT as distributed with Julia |
| [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl) | core solver interface | MIT |
| [Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) | optional conic solver | Apache-2.0 |
| [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) | optional persistence bridge | MIT; native HDF5 has its own BSD-style terms |
| [JLD2.jl](https://github.com/JuliaIO/JLD2.jl) | optional persistence bridge | MIT |
| [Makie.jl](https://github.com/MakieOrg/Makie.jl) | optional visualization | MIT |
| [QuantumCumulants.jl](https://github.com/qojulia/QuantumCumulants.jl) | optional interoperability | MIT |
| [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl) | optional interoperability | MIT |
| [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl) | optional interoperability | BSD-3-Clause |
| [Tables.jl](https://github.com/JuliaData/Tables.jl) | optional tabular output | MIT |

Source-package releases do not bundle these dependencies. A distributor of a
sysimage, application bundle, container, or executable must audit the exact
resolved dependency and artifact graph, preserve every required notice, and
provide the corresponding source and installation information required by
GPL-3.0-only. This includes Julia's native libraries; do not infer binary
compliance from the source-package table above.
