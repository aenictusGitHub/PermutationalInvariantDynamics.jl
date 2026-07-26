---
name: Release checklist
about: Audit a PermutationalInvariantDynamics.jl release before General registration
title: "Release vX.Y.Z"
labels: release
assignees: ""
---

## Ownership, provenance, and licensing

- [ ] Every contributor and relevant employer/institution has confirmed the right to publish the contributed work under `GPL-3.0-only`
- [ ] The copyright start year and named holders were checked against the code's actual creation and employment/institution records
- [ ] `COPYRIGHT.md`, `PROVENANCE.md`, and `THIRD_PARTY_NOTICES.md` are current
- [ ] Adapted third-party code identifies its upstream project, exact revision, license, and local modifications
- [ ] Generated-code licensing and the license header emitted by the browser assistant were reviewed
- [ ] Every curated example figure has a recorded generating script and contains no unlicensed third-party artwork
- [ ] Dependency and bundled-artifact licenses were reviewed for the source package and every distributed binary/container

## Release candidate

- [ ] Public changes and mathematical conventions reviewed by a maintainer
- [ ] `Project.toml`, `CHANGELOG.md`, and `CITATION.cff` versions are synchronized
- [ ] Julia 1.10 and current-stable clean-checkout tests pass
- [ ] Threaded performance, Aqua/JET, docs, and both executable-example shards pass
- [ ] The pinned REUSE licensing job passes with no missing or invalid declarations
- [ ] No root/quality Manifest, generated artifact, credential, or secret tracked
- [ ] GPL-3.0-only and repository/citation/documentation URLs verified
- [ ] Every external GitHub Action is pinned to a reviewed 40-character commit SHA
- [ ] GitHub private vulnerability reporting is enabled and the `SECURITY.md` advisory link was verified

## Finalize before Registrator

- [ ] One release date is assigned to both the `CHANGELOG.md` heading and `CITATION.cff` (do not leave the release candidate marked `Unreleased`)
- [ ] `julia --startup-file=no scripts/release_gate.jl --expect-version X.Y.Z --require-clean --require-released` passes
- [ ] The **CI** workflow was dispatched on the exact release commit with `release_readiness=true`
- [ ] The exact dated release commit has green GitHub checks
- [ ] Repository tag rules prevent updating or deleting published `v*` tags
- [ ] No `vX.Y.Z` tag was created manually

## Registration and publication

- [ ] `@JuliaRegistrator register` was posted for the exact green, dated release commit
- [ ] Generated General pull request reviewed
- [ ] General registry pull request merged
- [ ] TagBot created `vX.Y.Z` from the registered commit
- [ ] The tag-triggered **release metadata** job verified the dated metadata and tag target
- [ ] Versioned documentation and the GitHub release were verified
