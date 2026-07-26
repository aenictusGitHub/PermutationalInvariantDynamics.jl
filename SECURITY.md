# Security policy

## Supported versions

Until the first tagged release, security fixes target the current `main`
branch. After release, the latest minor release line and `main` are supported;
older lines may receive a fix only when a maintainer explicitly says so.
Numerical disagreement or nonconvergence without a security impact belongs in
a normal bug report.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, credential leak,
unsafe deserialization path, or denial-of-service problem that could put users
at risk. Use the repository's **Security → Report a vulnerability** private
advisory form:

<https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/security/advisories/new>

Include the affected revision and Julia version, a minimal reproduction,
impact, and any proposed mitigation. Do not include real credentials or
private research data. Maintainers will acknowledge a complete report through
the private advisory, coordinate a fix and disclosure date, and credit the
reporter unless anonymity is requested.

The project never asks reporters to weaken or bypass the GPL, third-party
notices, or provenance requirements when sharing a reproducer.

