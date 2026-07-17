---
name: Release checklist
about: Audit a PermutationalInvariantDynamics.jl release before General registration
title: "Release vX.Y.Z"
labels: release
assignees: ""
---

- [ ] Public changes and mathematical conventions reviewed by a maintainer
- [ ] `Project.toml`, `CHANGELOG.md`, and `CITATION.cff` versions synchronized
- [ ] Julia 1.10 and current-stable clean-checkout tests pass
- [ ] Threaded performance, Aqua/JET, docs, and representative examples pass
- [ ] No root/quality Manifest, generated artifact, credential, or secret tracked
- [ ] GPL-3.0-only and repository/citation/documentation URLs verified
- [ ] Exact release commit has green GitHub checks
- [ ] `@JuliaRegistrator register` posted only after all checks above
- [ ] Generated General pull request reviewed
- [ ] TagBot tag and versioned documentation verified after registry merge
