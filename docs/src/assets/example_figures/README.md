# Curated example output

These PNG and SVG files are lightweight, reviewable snapshots of the output
produced by the paired scripts in `examples/`. They let GitHub render the
expected numerical result directly in each example guide without running a
solver during documentation builds.

The numerical assertions in the Julia scripts are the regression tests. The
images are explanatory snapshots, not pixel-level test or convergence
certificates. Stochastic snapshots use the seed recorded in their script;
timings and uncertainty bands remain specific to the documented default run.

To update a CairoMakie snapshot:

1. instantiate the optional `examples` environment;
2. run the source script with its default checked controls;
3. inspect both the console assertions and the generated
   `examples/figures/<stem>.png`;
4. copy only the reviewed PNG here; and
5. check that its paired guide embeds it.

Dependency-free SVG snapshots are saved from the already computed
visualization objects. Do not regenerate any of these assets in ordinary CI or
in a Documenter build: several literature, HEOM, and trajectory examples are
intentionally too costly or stochastic for that workflow.
