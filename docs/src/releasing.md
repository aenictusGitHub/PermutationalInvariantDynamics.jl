# Release and General registration

The repository contains the metadata and automation needed for registration,
but registration itself must be triggered by a maintainer after reviewing a
clean release commit. Do not register an unreviewed working tree.

## Release gate

The dependency-free metadata gate checks the package/citation/changelog
version, release-date state, GPL identifier and canonical license, public
URLs, the tracked-manifest policy, generated-output paths, and optionally the
complete nonignored worktree:

```sh
julia --startup-file=no scripts/release_gate.jl \
  --expect-version X.Y.Z --require-clean
```

It deliberately accepts a synchronized release candidate whose changelog is
still marked `Unreleased`. After a release date actually exists in both
`CHANGELOG.md` and `CITATION.cff`, add `--require-released`. The script only
validates local metadata: it never creates or moves a tag, publishes a GitHub
release, or contacts General.

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
4. Confirm `LICENSE` is the canonical GPL-3.0-only text and that repository,
   documentation, and citation URLs agree.
5. Require green GitHub CI and documentation checks on the exact commit that
   will be registered. In particular, confirm both `executable examples`
   shards and the `release metadata` job. On a `vX.Y.Z` push, that job
   automatically switches to `--require-released --tag-ref refs/tags/vX.Y.Z`;
   an undated or version-mismatched release tag therefore fails.

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

Install the JuliaRegistrator GitHub App for the repository, then comment

```text
@JuliaRegistrator register
```

on a commit or issue after the release gate passes. Review the generated
General-registry pull request. Once it merges, TagBot uses the repository's
configured credentials to create the version tag, which in turn builds
versioned documentation. Add `date-released` to `CITATION.cff` only after that
release actually exists.

For later releases, increase the semantic version, update the changelog, and
repeat the same clean-checkout gate. Never reuse or move an existing version
tag.
