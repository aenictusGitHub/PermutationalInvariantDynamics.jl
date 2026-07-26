# Release and General registration

The repository contains the metadata and automation needed for registration,
but registration itself must be triggered by a maintainer after reviewing a
clean release commit. Do not register an unreviewed working tree.

The first-release sequence is deliberately one-way:

```text
clean candidate
    -> assign one matching CHANGELOG/CITATION release date
    -> run the finalized release gate and green CI
    -> ask Registrator to open the General pull request
    -> merge the General pull request
    -> let TagBot create the immutable version tag
    -> let tag-triggered CI verify the tag and build versioned docs
```

Do not create, move, or reuse the version tag by hand.

## Release gate

The dependency-free metadata gate checks the package/citation/changelog
version, release-date state, GPL identifier and canonical license, public
URLs, required copyright/provenance/third-party records, the tracked-manifest
policy, generated-output paths, and optionally the complete nonignored
worktree.

During ordinary development and candidate preparation, run:

```sh
julia --startup-file=no scripts/release_gate.jl \
  --expect-version X.Y.Z --require-clean
```

It deliberately accepts a synchronized release candidate whose changelog is
still marked `Unreleased`. Before asking Registrator to act, replace
`Unreleased` with the intended release date, add the same `date-released` to
`CITATION.cff`, and run the stricter gate:

```sh
julia --startup-file=no scripts/release_gate.jl \
  --expect-version X.Y.Z --require-clean --require-released
```

If publication must move to another date, update both files in a new candidate
commit and repeat every final check before registration. Do not repair release
metadata after a version tag has been published. The script only validates
local metadata: it never creates or moves a tag, publishes a GitHub release,
or contacts General.

The **CI** workflow has a manual `release_readiness` input. Dispatch it with
that input enabled on the exact dated release commit. This makes the hosted
metadata job use `--require-released`; an ordinary push or pull request keeps
using the candidate gate.

Before changing the public version:

1. Run the full test suite on Julia 1.10 and current stable Julia from a clean
   checkout, plus the threaded performance gates, Aqua/JET, documentation,
   and representative examples. The same dependency-free numerical examples
   used by CI can be run locally as two isolated shards:

   ```sh
   julia --startup-file=no --project=. test/run_quick_examples.jl --shard 1/2
   julia --startup-file=no --project=. test/run_quick_examples.jl --shard 2/2
   ```

   These shards execute the examples' default numerical assertions with
   rendering explicitly disabled; they do not substitute smaller tolerances
   or weaker physics checks.
2. Confirm that `docs/Manifest.toml` is the only tracked `Manifest.toml`.
   Root, quality, example, benchmark, comparison, optional-test, and notebook
   manifests are generated locally and must remain untracked. Also confirm
   that no generated figure, token, private key, or deploy credential is
   tracked.
3. Review every public API change, numerical convention, convergence claim,
   and literature comparison. Update `CHANGELOG.md`, `Project.toml`, and the
   version in `CITATION.cff` together.
4. Audit ownership and provenance before relying on the GPL declaration:
   confirm contributor and employer/institution rights; review
   `COPYRIGHT.md`, `PROVENANCE.md`, and `THIRD_PARTY_NOTICES.md`; retain the
   original notice, license, exact upstream revision, and modification record
   for adapted code; and verify the provenance ledger for curated figures.
   Review the browser assistant's generated-code license separately because
   its output contains reusable template text.
5. Confirm `LICENSE` is the canonical GPL-3.0-only text and that repository,
   documentation, and citation URLs agree. Re-audit licenses for dependencies
   and binary artifacts whenever distributing a sysimage, application,
   container, or offline bundle rather than only package source. Confirm every
   external GitHub Action is pinned to a reviewed full commit SHA; Dependabot's
   monthly GitHub-Actions pull requests are the update path for those pins.
6. Run `reuse lint` and require the pinned **REUSE licensing** CI job to pass.
   This verifies file-level declarations and every license text independently
   of the package's metadata gate.
7. Assign the intended release date to both the current `CHANGELOG.md` heading
   and `CITATION.cff`, then run the strict `--require-released` gate.
8. Require green GitHub CI and documentation checks on the exact dated commit
   that will be registered. In particular, confirm both `executable examples`
   shards and the manually dispatched `release metadata` job with
   `release_readiness=true`.

## Documentation deployment

Documenter publishes generated pages to the `gh-pages` branch. In repository
**Settings → Pages**, select **Deploy from a branch**, then choose
`gh-pages` and `/(root)`. The branch is generated deployment output: do not
merge it into `main` or edit it as package source.

The documentation workflow currently deploys with the job-scoped
`GITHUB_TOKEN`. The repository also uses an SSH deploy key for TagBot and
CompatHelper. Generate that key pair with DocumenterTools for this repository,
then install it as follows:

1. Add the public key under **Settings → Deploy keys**, enable write access,
   and give it a repository-specific name.
2. Add the corresponding private value under
   **Settings → Secrets and variables → Actions** as `DOCUMENTER_KEY`.
3. Never paste either private key material or the secret value into a tracked
   file, issue, workflow log, or pull request.

After the first deployment, verify that the `gh-pages` branch exists and that
the Pages source still points to its root. To recover a missing or stale site,
run the **Documentation** workflow manually from `main` using
`workflow_dispatch`. If the deployment cannot push, verify the workflow's
`contents: write` permission and the repository Actions policy. If TagBot or
CompatHelper reports SSH authentication failures, replace the deploy-key pair
and update `DOCUMENTER_KEY`, then rerun the failed workflow. A successful
manual deployment should recreate or update `gh-pages`; it must not require a
generated documentation commit on `main`.

## Registering

Install the JuliaRegistrator GitHub App for the repository. Configure a GitHub
tag ruleset for `v*` that prevents published tags from being updated or
deleted except by the narrowly authorized release automation.

Only after the exact dated commit passes the strict local gate and the manual
release-readiness CI run, comment

```text
@JuliaRegistrator register
```

on that commit or its issue. Review the generated General-registry pull
request. Once it merges, TagBot uses the repository's configured credentials
to create the version tag; do not create the tag in advance. The tag push runs
the release-metadata gate with
`--require-released --tag-ref refs/tags/vX.Y.Z`. That check verifies the tag
name, dated package metadata, and that the workflow checkout is the commit
targeted by the tag. The same tag then builds versioned documentation.

If General does not merge on the assigned date, stop the release, close or
supersede the pending registration, update both date fields, and rerun the
strict gate and hosted checks on the new commit. Never move a published tag.

For later releases, increase the semantic version, update the changelog, and
repeat the same clean-checkout gate. Never reuse or move an existing version
tag.
