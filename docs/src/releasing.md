# Release and General registration

The repository contains the metadata and automation needed for registration,
but registration itself must be triggered by a maintainer after reviewing a
clean release commit. Do not register an unreviewed working tree.

## Release gate

Before changing the public version:

1. Run the full test suite on Julia 1.10 and current stable Julia from a clean
   checkout, plus the threaded performance gates, Aqua/JET, documentation,
   and representative examples.
2. Confirm that no root `Manifest.toml`, `quality/Manifest.toml`, generated
   figure, token, or deploy key is tracked. Keep `docs/Manifest.toml`.
3. Review every public API change, numerical convention, convergence claim,
   and literature comparison. Update `CHANGELOG.md`, `Project.toml`, and the
   version in `CITATION.cff` together.
4. Confirm `LICENSE` is the canonical GPL-3.0-only text and that repository,
   documentation, and citation URLs agree.
5. Require green GitHub CI and documentation checks on the exact commit that
   will be registered.

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
