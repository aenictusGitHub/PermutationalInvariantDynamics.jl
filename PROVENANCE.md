# Source and research provenance

This ledger distinguishes adapted source expression from independent
implementations of published mathematics, API inspiration, and locally
generated assets. It is part of the release review and must be updated when a
new algorithm, external implementation, manuscript, or visual asset enters the
repository.

“Independent implementation” means that the repository records the scholarly
or API source used to understand the method, while the package implementation
was written for its own representations and contracts. It is not a claim that
the mathematical method originated here. “Adapted source” means recognizable
upstream program expression was translated or modified and therefore requires
the notice in `THIRD_PARTY_NOTICES.md`.

## Implementation ledger

| Area | Classification | Source and local scope |
|---|---|---|
| PI Schur--Weyl representation, one-body and Appendix-D processes | Independent implementation of the package authors' published framework | T. Bastin and J. Martin, *J. Phys. A* **58**, 275301 (2025), [DOI 10.1088/1751-8121/addfc1](https://doi.org/10.1088/1751-8121/addfc1); representation core, geometry, terms, and model lowering |
| Six-rate qubit ensemble, Dicke/state conveniences, population and phase-space user workflows | API and model inspiration; independent implementation over the package's general Schur representation | QuTiP PIQS documentation and N. Shammah *et al.*, *Phys. Rev. A* **98**, 063815 (2018), [DOI 10.1103/PhysRevA.98.063815](https://doi.org/10.1103/PhysRevA.98.063815); `src/spin.jl`, `src/populations.jl`, and phase-space helpers. The repository audit found no source-level translation from `qutip.piqs`; a maintainer must reconfirm this when reviewing later changes |
| Drude--Lorentz Padé HEOM bath parameters | Adapted source | QuTiP `qutip/core/environment.py` at revision `e5dbb0195fdbf37fb39d4e52e27c80594f8eb655`; `src/heom.jl::_drude_pade_parameters`. Full attribution and BSD-3-Clause terms are in `THIRD_PARTY_NOTICES.md` |
| Remaining finite HEOM hierarchy, physical bath constructors, workspaces, PI lowering, and solvers | Independent implementation of published algorithms | Hu, Xu, and Yan, *J. Chem. Phys.* **133**, 101106 (2010) and **134**, 244106 (2011); Lambert *et al.*, *Phys. Rev. Research* **5**, 013181 (2023); citations and conventions are detailed in `docs/src/heom.md` |
| PI hierarchy of pure states (HOPS) | Independent implementation of published algorithms | Süß, Eisfeld, and Strunz, *Phys. Rev. Lett.* **113**, 150403 (2014), and Diósi, Gisin, and Strunz, *Phys. Rev. A* **58**, 1699 (1998); `src/hops.jl` and `docs/src/hops.md` |
| Tetrahedral, octahedral, and icosahedral Eulerian pulse words and generator axes | Adapted/transcribed published data under CC BY 4.0 | Read, Serrano-Ensástiga, and Martin, *Quantum* **9**, 1661 (2025), [DOI 10.22331/q-2025-03-12-1661](https://doi.org/10.22331/q-2025-03-12-1661); the marked snippet in `src/hierarchy_pulses.jl` encodes the published data as precision-generic Julia. Full attribution and terms are in `THIRD_PARTY_NOTICES.md`; the constructors, validation, PI lifting, and workspace integration are package-authored |
| PI PPT-mixture genuine multipartite-entanglement test | Independent implementation of a published SDP | Novo, Moroder, and Gühne, *Phys. Rev. A* **88**, 012305 (2013), [DOI 10.1103/PhysRevA.88.012305](https://doi.org/10.1103/PhysRevA.88.012305); `src/genuine_entanglement.jl` and Clarabel extension |
| Symmetric-state stabilizer Rényi entropy | Independent implementation of a published measure and PI reduction | Passarelli, Fazio, and Lucignano, *Phys. Rev. A* **110**, 022436 (2024), [DOI 10.1103/PhysRevA.110.022436](https://doi.org/10.1103/PhysRevA.110.022436); `src/nonstabilizerness.jl` |
| Optional bridges to Makie, Clarabel, Tables, storage, cumulants, QuantumOptics, and QuantumToolbox | Interoperability against public APIs | Methods live in matching `ext/` modules. No dependency source is vendored; dependency licenses remain upstream |
| Browser model-code generator | Package-authored typed parser and fixed templates | `docs/src/assets/model_code_generator_*`; generated Julia programs are explicitly GPL-3.0-only because they contain substantial templates |
| Example figures | Locally generated output | Generated from the paired scripts, reviewed, and documented in `docs/src/assets/example_figures/README.md`; no publisher artwork or digitized journal figure is included |

## Manuscripts and non-public research material

Some examples were developed from material supplied directly to the
maintainers:

- `examples/all_to_all_xx_spin_local_pseudomodes.*` discusses a 2026 Debecker
  *et al.* draft, but implements a distinct all-to-all PI specialization and
  does not reproduce manuscript figures.
- `examples/nonmarkovian_dynamical_decoupling.*` reconstructs calculations
  described in a September 2023 Colin Read report and clearly identifies
  assumptions needed where the report is incomplete.

Before a public release, a maintainer must confirm that the authors and any
relevant institution permit the public descriptions, equations, numerical
parameters, and comparisons; that no confidential draft text or figure is
reproduced; and that the provenance statements remain accurate. Repository
automation cannot supply that permission.

## AI-assisted development

OpenAI Codex has been used extensively. The disclosure in `README.md` is
mandatory. AI output is not accepted as provenance: the submitting human must
review and understand it, identify any third-party source or close
implementation used as input, run the relevant tests, and certify the
submission under the Developer Certificate of Origin.

## Release attestation

The release issue must record that a maintainer has:

1. reviewed new code and assets for copied or adapted third-party expression;
2. updated this ledger and `THIRD_PARTY_NOTICES.md` where necessary;
3. checked the resolved licenses for any new dependency or artifact;
4. confirmed that the copyright start year and named holders match the
   underlying creation and employment/institution records;
5. obtained required coauthor, employer, institution, and manuscript
   permissions; and
6. confirmed that generated-code and binary-distribution obligations are
   accurately documented.
