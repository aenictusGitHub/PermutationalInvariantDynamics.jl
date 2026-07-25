---
name: Release checklist
about: Audit a PermutationalInvariantDynamics.jl release before General registration
title: "Release vX.Y.Z"
labels: release
assignees: ""
---

- [ ] Public changes and mathematical conventions reviewed by a maintainer
- [ ] `Project.toml`, `CHANGELOG.md`, and `CITATION.cff` versions synchronized
- [ ] `julia --startup-file=no scripts/release_gate.jl --expect-version X.Y.Z --require-clean` passes
- [ ] Julia 1.10 and current-stable clean-checkout tests pass
- [ ] Threaded performance, Aqua/JET, docs, and both executable-example shards pass
- [ ] No root/quality Manifest, generated artifact, credential, or secret tracked
- [ ] GPL-3.0-only and repository/citation/documentation URLs verified
- [ ] Exact release commit has green GitHub checks
- [ ] The `vX.Y.Z` tag passes the dated, tag-matching release-metadata gate
- [ ] `@JuliaRegistrator register` posted only after all checks above
- [ ] Generated General pull request reviewed
- [ ] TagBot tag and versioned documentation verified after registry merge
