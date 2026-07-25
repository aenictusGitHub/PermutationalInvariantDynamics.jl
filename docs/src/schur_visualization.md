# Visualizing Schur blocks

The Schur-block visualizer turns a PI state, operator, or superoperator into a
compact sector diagram. It has no plotting dependency: `visualize_schur_blocks`
returns a small object with an SVG display, which notebooks can render inline,
and `save_schur_block_visualization` writes the same SVG to a file.

For a PI state, one tile is shown for every partition $\nu$ retained by the
`PIBasis`. For a superoperator the diagram is a block matrix: **rows are output
sectors and columns are input sectors**. Thus a nonzero tile at row $\lambda$
and column $\nu$ means that the map can send an operator in sector $\nu$
to sector $\lambda$. Collective Hamiltonians and collective jumps are
sector diagonal. Permutation-invariant sums of unresolved local channels can
couple different Schur sectors, and appear as off-diagonal tiles.

## Quick start

```julia
using PermutationalInvariantDynamics

basis = PIBasis(4, 2)
sigma = ComplexF64[0.7 0.1; 0.1 0.3]
rho = iid_state(basis, sigma)

state_structure = schur_block_structure(rho)
state_figure = visualize_schur_blocks(state_structure;
    title="Product state")
display(state_figure)

sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, [LocalJump(sm; rate=0.2)])
prepared = compile(model; backend=:matrixfree)

generator_structure = schur_block_structure(prepared.operator, basis)
generator_figure = visualize_schur_blocks(generator_structure;
    title="Local decay Liouvillian", show_young_diagrams=true)
display(generator_figure)
```

The analysis and rendering steps are separate deliberately. Reuse a structure
to change the title or colour scale without probing a matrix-free operator
again. The sector labels and activity threshold belong to the extracted
structure and remain fixed.

## Young diagrams and tableau multiplicities

Young diagrams are shown by default beside the sector labels. For a state or
operator, each top-axis diagram is the shape of its diagonal Schur sector. For
a superoperator, the left-axis diagrams label output sectors and the top-axis
diagrams label input sectors, matching the row/column convention of the block
matrix.

A partition shape does **not** select one standard Young tableau. In the PI
basis the standard-tableau label is summed over, and the number of such labels
is the exact symmetric-group multiplicity $f^\nu$. Hovering a diagram reports
the padded partition, its U(d) irrep dimension, $f^\nu$, and this convention.
Use `show_young_diagrams=false` when a compact text-only layout is preferred.

For diagrams with at most 64 boxes, the SVG draws every box. Larger shapes use
normalized row bands with at most 64 graphical nodes, so rendering cost and
file size remain bounded independently of a large particle number. This
changes only the thumbnail; the partition and exact multiplicity in the
tooltip remain unchanged.

## Coefficient blocks and physical blocks

In the package convention, a state is stored as coefficient blocks
$C_\nu$ in the orthonormal PI basis of equation (7). Its physical block in
the Schur--Weyl decomposition is

```math
\rho_\nu=\frac{C_\nu}{\sqrt{f^\nu}},
```

where $f^\nu$ is the symmetric-group multiplicity. The complete operator is
$\bigoplus_\nu \rho_\nu\otimes I_{f^\nu}$. Consequently,

```math
\lVert C_\nu\rVert_F^2
=f^\nu\lVert\rho_\nu\rVert_F^2.
```

Choose which matrix is measured with `representation=:coefficient` or
`representation=:physical`. With `metric=:frobenius`, the coefficient choice
makes the squared value the Hilbert--Schmidt contribution of that whole Schur
sector, including its multiplicity. The physical choice instead measures one
irrep copy and omits $f^\nu$. `metric=:trace_norm` applies the same choice to
the Schatten one-norm. For a density operator, `metric=:population` reports
the trace weight $f^\nu\mathrm{tr}(\rho_\nu)$. No metric changes or
renormalizes the state. Population mode requires each reported sector trace to
be real (up to roundoff) and nonnegative; it throws for invalid data instead
of taking an absolute value or clipping it. `physical_block(rho, nu)` remains
the appropriate API when the matrix acting on one irrep copy is needed
explicitly.

`metric=:trace_norm` uses `LinearAlgebra.svdvals` and therefore requires a
scalar type supported by Julia's SVD backend. Unsupported types such as
`BigFloat` produce a clear `ArgumentError`; `:frobenius` and `:population`
retain the state's scalar precision.

For a superoperator with `metric=:frobenius`, each value is the Frobenius norm
of the corresponding linear map between vectorized coefficient blocks. The
coefficient basis is orthonormal and uses Julia column-major vectorization.
The value therefore measures total coupling strength across all source
coordinates in that block; it is not a transition probability, rate,
eigenvalue, or spectral norm. `metric=:maxabs` instead reports the largest
absolute matrix element in the block. Comparisons are most informative
between blocks of the same model and with the same metric.

## Thresholds and colour scaling

The numerical structure retains the measured block norms. A visualization
threshold only controls which tiles are declared active and drawn as
couplings; it does not modify the state or superoperator. Use an absolute
threshold in the same units as the block norm, and increase it only to hide
roundoff-level leakage:

```julia
structure = schur_block_structure(
    prepared.operator, basis; threshold=1e-12)
figure = visualize_schur_blocks(structure; title="Resolved couplings")
```

Do not interpret a block hidden by an arbitrarily large threshold as a proven
selection rule. For a statement about exact structure, compare against a
scale-aware tolerance and the construction of the physical terms.

Colour normalization affects only the SVG. `normalize=:global` uses a common
scale, while `:row`, `:column`, and `:none` support more specialized
structural comparisons. `scale=:linear` preserves magnitude ratios;
`scale=:log` reveals weak couplings across a wider dynamic range. The original
norms remain in `structure.weights` and the thresholded Boolean mask in
`structure.active`; use those fields for assertions or quantitative
reporting. With `normalize=:none`, raw values are interpreted directly on the
unit colour interval, so values above one saturate at the darkest colour.

## Sparse and matrix-free superoperators

A sparse PI Liouvillian can be split directly into sector rectangles. For a
matrix-free Liouvillian, the exact Frobenius norm of every block is obtained
by applying the map to coordinate vectors, one input coordinate at a time,
and accumulating the squared output norm by sector. If the PI coordinate
dimension is

```math
n_{\mathrm{PI}}=\sum_\nu \dim(U_\nu)^2,
```

this requires $n_{\mathrm{PI}}$ applications of the Liouvillian. It avoids
storing the $n_{\mathrm{PI}}\times n_{\mathrm{PI}}$ matrix, but it is a
diagnostic setup cost rather than a cheap operation inside a time-stepping or
parameter-scan loop. Compute the structure once and reuse it.

The matrix-free probe uses PI vectors and the compiled Schur kernels. It never
constructs a $d^N$ Hilbert-space matrix or a $d^{2N}$ Liouville-space
matrix. A superoperator's linear dimensions alone do not determine its Schur
sector boundaries, so supply the matching `PIBasis`:

```julia
structure = schur_block_structure(prepared.operator, basis)
```

For a time-dependent matrix-free map, `time=` and `parameters=` define the
instantaneous generator at the supplied values. Fixed-operator scalar rates
are evaluated by the prepared kernels during each probe; general
operator-valued terms are evaluated and lowered once before probing:

```julia
structure = schur_block_structure(
    driven.operator, basis; time=t, parameters=parameters)
```

The same matrix-free path accepts `MatrixFreeSymmetryProjector` objects and
uses one reusable `SymmetryProjectorWorkspace`; their diagrams expose which
Schur sectors are retained by the selected weak-symmetry charge.

## Saving SVG without repository artifacts

```julia
mktempdir() do directory
    path = joinpath(directory, "schur-blocks.svg")
    save_schur_block_visualization(path, generator_figure)
    @assert isfile(path)
end
```

SVG is text and remains sharp in papers and presentations. The runnable
`examples/irrep_block_visualization.jl` example solves a pump--decay steady
state, renders its sector populations and multiplicity-compressed density
spectrum, compares collective and local dissipators, and writes all generated
figures only inside a temporary directory.
