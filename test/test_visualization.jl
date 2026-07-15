@testset "Schur-block structures and visualization" begin
    b=PIBasis(3,2)
    p1,p2=b.sectors
    f1=Float64(symmetric_group_dimension(p1))
    f2=Float64(symmetric_group_dimension(p2))

    # Set the two physical sector populations explicitly.  This also checks
    # that visual diagnostics use the documented coefficient/physical-block
    # conversion rather than treating stored PI coefficients as density
    # matrices.
    rho=PIState(b)
    coefficient_block(rho,p1)[1,1]=0.25/sqrt(f1)
    coefficient_block(rho,p2)[1,1]=0.75/sqrt(f2)
    @test trace(rho)≈1

    populations=schur_block_structure(rho;metric=:population)
    @test populations isa SchurBlockStructure
    @test populations.basis===b
    @test populations.kind===:state
    @test populations.sectors==Tuple(b.sectors)
    @test populations.block_dimensions==(4,2)
    @test populations.coordinate_dimensions==(16,4)
    @test populations.metric===:population
    @test populations.representation===:physical
    @test populations.threshold==0
    @test populations.weights≈Diagonal([0.25,0.75])
    @test populations.active==BitMatrix([1 0;0 1])

    huge_partition=Partition((1050,1050))
    huge_basis=PIBasis(2100,2;sectors=[huge_partition.parts])
    huge_state=basis_state(huge_basis,huge_partition,
                           only(only(huge_basis.patterns)))
    @test only(schur_block_structure(
        huge_state;metric=:population).weights)≈1 atol=2e-15
    @test_throws ArgumentError schur_block_structure(
        huge_state;metric=:frobenius,representation=:physical)

    # The coefficient-space aggregate can overflow before division by
    # sqrt(f), even though the physical identity block has ordinary norms.
    # N=2062,j=1 gives a finite stored identity entry but overflowing 3x3
    # coefficient Frobenius and trace-norm aggregates.
    metric_partition=Partition((1032,1030))
    metric_basis=PIBasis(2062,2;sectors=[metric_partition.parts])
    metric_identity=identity_operator(metric_basis)
    @test !isfinite(norm(coefficient_block(
        metric_identity,metric_partition)))
    metric_frobenius=schur_block_structure(
        metric_identity;metric=:frobenius,representation=:physical)
    metric_trace=schur_block_structure(
        metric_identity;metric=:trace_norm,representation=:physical)
    @test only(metric_frobenius.weights)≈sqrt(3.0) rtol=8eps(Float64)
    @test only(metric_trace.weights)≈3.0 rtol=8eps(Float64)

    physical=schur_block_structure(rho;metric=:frobenius,
                                   representation=:physical)
    coefficient=schur_block_structure(rho;metric=:frobenius,
                                      representation=:coefficient)
    @test diag(physical.weights)≈[0.25,0.75/f2]
    @test diag(coefficient.weights)≈[0.25,0.75/sqrt(f2)]
    @test physical.weights≈schur_block_structure(rho;metric=:trace_norm).weights
    @test physical.metadata!==nothing
    @test (@inferred schur_block_structure(rho)) isa SchurBlockStructure
    @test (@inferred visualize_schur_blocks(populations)) isa
          SchurBlockVisualization

    big_state=PIState(b;T=BigFloat)
    @test schur_block_structure(big_state;metric=:frobenius).weights isa
          Matrix{BigFloat}
    @test_throws ArgumentError schur_block_structure(
        big_state;metric=:trace_norm)

    empty_basis=PIBasis(0,2;sectors=Tuple{Int,Int}[])
    @test_throws ArgumentError schur_block_structure(PIState(empty_basis))

    invalid_negative=copy(rho)
    coefficient_block(invalid_negative,p1)[1,1]=-0.1
    @test_throws ArgumentError schur_block_structure(
        invalid_negative;metric=:population)
    invalid_complex=copy(rho)
    coefficient_block(invalid_complex,p1)[1,1]+=0.1im
    @test_throws ArgumentError schur_block_structure(
        invalid_complex;metric=:population)

    # Operators may have identically zero Schur blocks; these must remain
    # visible as inactive cells rather than being omitted from the layout.
    op=PIOperator(b)
    coefficient_block(op,p2)[1,1]=2sqrt(f2)
    operator_structure=schur_block_structure(op;metric=:frobenius,
                                             representation=:physical)
    @test operator_structure.kind===:operator
    @test operator_structure.weights≈Diagonal([0.0,2.0])
    @test operator_structure.active==BitMatrix([0 0;0 1])
    @test diag(schur_block_structure(op;metric=:trace_norm).weights)≈[0,2]

    # A superoperator cell (row sector, column sector) is the corresponding
    # PI-coordinate submatrix.  Choose entries whose Frobenius and maximum
    # norms differ, making orientation and metric errors unambiguous.
    n=length(b)
    r1=b.offsets[1]:(b.offsets[2]-1)
    r2=b.offsets[2]:(b.offsets[3]-1)
    M=zeros(ComplexF64,n,n)
    M[r1[1],r1[1]]=3
    M[r1[2],r1[2]]=4
    M[r1[3],r2[1]]=12
    M[r2[1],r1[3]]=5
    M[r2[2],r1[4]]=12

    dense_structure=schur_block_structure(M,b;metric=:frobenius)
    sparse_structure=schur_block_structure(sparse(M),b;metric=:frobenius)
    @test dense_structure.kind===:superoperator
    @test dense_structure.representation===:coefficient
    @test dense_structure.block_dimensions==(4,2)
    @test dense_structure.coordinate_dimensions==(16,4)
    @test dense_structure.weights≈[5.0 12.0;13.0 0.0]
    @test sparse_structure.weights≈dense_structure.weights
    @test sparse_structure.active==dense_structure.active
    @test schur_block_structure(M,b;metric=:maxabs).weights≈
          [4.0 12.0;12.0 0.0]

    extreme=zeros(Float64,n,n)
    extreme[r1[1],r1[1]]=1e200
    extreme[r2[1],r2[1]]=1e-200
    extreme_dense=schur_block_structure(extreme,b)
    extreme_sparse=schur_block_structure(sparse(extreme),b)
    @test extreme_dense.weights[1,1]==1e200
    @test extreme_dense.weights[2,2]==1e-200
    @test extreme_sparse.weights==extreme_dense.weights
    @test schur_block_structure(adjoint(sparse(M)),b).weights≈
          permutedims(dense_structure.weights)

    tracevec=zeros(ComplexF64,n)
    action! = (y,x,t,p)->mul!(y,M,x)
    matrixfree=MatrixFreeLiouvillian(n,action!,ComplexF64,tracevec)
    matrixfree_structure=schur_block_structure(matrixfree,b;metric=:frobenius)
    @test matrixfree_structure.weights≈dense_structure.weights atol=2e-14
    @test matrixfree_structure.active==dense_structure.active

    extreme_action! = (y,x,t,p)->mul!(y,extreme,x)
    extreme_matrixfree=MatrixFreeLiouvillian(
        n,extreme_action!,Float64,zeros(Float64,n))
    @test schur_block_structure(extreme_matrixfree,b).weights==
          extreme_dense.weights

    threshold_calls=Ref(0)
    threshold_action! = (y,x,t,p)->begin
        threshold_calls[]+=1
        mul!(y,M,x)
    end
    threshold_matrixfree=MatrixFreeLiouvillian(
        n,threshold_action!,ComplexF64,tracevec)
    @test_throws ArgumentError schur_block_structure(
        threshold_matrixfree,b;threshold=-1)
    @test threshold_calls[]==0
    @test_throws ArgumentError schur_block_structure(
        threshold_matrixfree,b;threshold=big"1e-10000")
    @test threshold_calls[]==0

    # Exercise the compiled matrix-free backend as well as a synthetic linear
    # map.  Exact coordinate probing must agree with sparse materialization.
    sm=ComplexF64[0 1;0 0]
    model=PIModel(b,[LocalJump(sm;rate=0.4),CollectiveJump(sm;rate=0.1)])
    Ls=liouvillian(model;representation=:sparse)
    Lm=liouvillian(model;representation=:matrixfree)
    compiled_sparse=schur_block_structure(Ls,b;metric=:frobenius,
                                          threshold=1e-14)
    compiled_matrixfree=schur_block_structure(Lm,b;metric=:frobenius,
                                              threshold=1e-14)
    @test compiled_matrixfree.weights≈compiled_sparse.weights atol=2e-12
    @test compiled_matrixfree.active==compiled_sparse.active

    prepared_sparse=compile(model;backend=:sparse)
    prepared_matrixfree=compile(model;backend=:matrixfree)
    @test schur_block_structure(model).weights≈compiled_sparse.weights atol=2e-12
    @test schur_block_structure(LiouvillianPlan(model)).weights≈
          compiled_sparse.weights atol=2e-12
    @test schur_block_structure(prepared_sparse).weights≈compiled_sparse.weights
    @test schur_block_structure(prepared_matrixfree).weights≈
          compiled_sparse.weights atol=2e-12
    @test_throws ArgumentError schur_block_structure(
        prepared_sparse;representation=:physical)

    driven_model=PIModel(b,[LocalJump(sm;rate=(t,p)->p.gamma*t)])
    driven=compile(driven_model;backend=:matrixfree)
    @test_throws ArgumentError schur_block_structure(driven)
    driven_structure=schur_block_structure(
        driven;time=0.5,parameters=(gamma=0.8,))
    frozen_structure=schur_block_structure(
        freeze(driven_model;time=0.5,parameters=(gamma=0.8,)))
    @test driven_structure.weights≈frozen_structure.weights atol=2e-12
    @test driven_structure.metadata.time==0.5

    operator_calls=Ref(0)
    sx=ComplexF64[0 1;1 0]
    operator_drive=(t,p)->begin
        operator_calls[]+=1
        p.amplitude*t*sx
    end
    operator_driven=PIModel(
        b,[LocalHamiltonian(operator_drive;rate=0.3)])
    operator_structure=schur_block_structure(
        operator_driven;time=0.4,parameters=(amplitude=0.7,))
    @test operator_calls[]==1
    operator_reference=schur_block_structure(freeze(
        operator_driven;time=0.4,parameters=(amplitude=0.7,),
        representation=:sparse),b)
    @test operator_structure.weights≈operator_reference.weights atol=2e-12

    adjoint_structure=schur_block_structure(adjoint(Lm),b)
    @test adjoint_structure.weights≈permutedims(compiled_sparse.weights) atol=2e-12

    unitary=ComplexF64[1 0;0 -1]
    projector=matrixfree_symmetry_projector(b,unitary;charge=1)
    projector_structure=schur_block_structure(projector)
    projector_matrix=zeros(ComplexF64,n,n)
    projector_input=zeros(ComplexF64,n)
    projector_work=SymmetryProjectorWorkspace(projector)
    for column in 1:n
        fill!(projector_input,0);projector_input[column]=1
        apply!(view(projector_matrix,:,column),projector,
               projector_input,projector_work)
    end
    @test projector_structure.weights≈
          schur_block_structure(projector_matrix,b).weights atol=2e-12
    @test !any(projector_structure.active[i,j]
               for i in axes(projector_structure.active,1)
               for j in axes(projector_structure.active,2) if i!=j)

    # Thresholding controls visibility, not the diagnostic values themselves.
    thresholded=schur_block_structure(M,b;metric=:frobenius,threshold=5.5)
    @test thresholded.weights≈dense_structure.weights
    @test thresholded.active==BitMatrix([0 1;1 0])

    # Restricted bases retain precisely their requested sector ordering.  A
    # large-N symmetric-only case is also a guard against reconstructing d^N
    # Hilbert-space objects merely to display one Schur block.
    br=PIBasis(8,2;sectors=[(8,0),(7,1)])
    restricted=schur_block_structure(identity_operator(br))
    @test restricted.sectors==Tuple(br.sectors)
    @test restricted.block_dimensions==(9,7)
    @test restricted.coordinate_dimensions==(81,49)
    @test size(restricted.weights)==(2,2)

    blarge=PIBasis(100,2;sectors=[(100,0)])
    large_structure=schur_block_structure(PIOperator(blarge))
    @test large_structure.block_dimensions==(101,)
    @test large_structure.coordinate_dimensions==(101^2,)
    @test size(large_structure.weights)==(1,1)
    @test !large_structure.active[1,1]

    @test_throws ArgumentError schur_block_structure(rho;metric=:unknown)
    @test_throws ArgumentError schur_block_structure(op;metric=:population)
    @test_throws ArgumentError schur_block_structure(M,b;metric=:trace_norm)
    @test_throws ArgumentError schur_block_structure(rho;representation=:unknown)
    @test_throws ArgumentError schur_block_structure(rho;threshold=-1)
    @test_throws ArgumentError schur_block_structure(rho;threshold=NaN)
    @test_throws DimensionMismatch schur_block_structure(zeros(n+1,n+1),b)

    title="Schur <blocks> & \"sectors\""
    visualization=visualize_schur_blocks(dense_structure;scale=:linear,
                                         normalize=:global,title=title)
    @test visualization isa SchurBlockVisualization
    @test visualization.show_young_diagrams
    # Julia versions wrap displayed docstrings at different columns. Test the
    # content after normalizing presentation-only whitespace.
    visualization_docs=replace(sprint(show,MIME("text/plain"),
        @doc visualize_schur_blocks),r"\s+"=>" ")
    @test occursin("show_young_diagrams=true",visualization_docs)
    @test occursin("PI operators sum over that multiplicity label",visualization_docs)
    svg=sprint(show,MIME("image/svg+xml"),visualization)
    @test startswith(lstrip(svg),"<svg")
    @test occursin("</svg>",svg)
    @test occursin("<rect",svg)
    @test occursin("Schur &lt;blocks&gt; &amp; &quot;sectors&quot;",svg)
    @test !occursin(title,svg)
    @test occursin("SchurBlockStructure",sprint(show,dense_structure))
    @test occursin("SchurBlockVisualization",sprint(show,visualization))
    @test occursin("young_diagrams=true",sprint(show,visualization))
    @test length(collect(eachmatch(r"class=\"young-diagram ",svg)))==4
    @test length(collect(eachmatch(r"class=\"young-box\"",svg)))==12
    @test length(collect(eachmatch(r"young-column-diagram",svg)))==2
    @test length(collect(eachmatch(r"young-row-diagram",svg)))==2
    @test occursin("U(2) irrep dimension=4",svg)
    @test occursin("f^ν=1 standard Young tableaux",svg)
    @test occursin("no unique filling is selected",svg)
    state_text=sprint(show,MIME("text/plain"),populations)
    state_svg=sprint(show,MIME("image/svg+xml"),
                     visualize_schur_blocks(populations))
    @test occursin("diagonal sector weights",state_text)
    @test !occursin("rows=output",state_text)
    @test !occursin("output ",state_svg)
    # A state/operator has one diagram per diagonal sector rather than
    # duplicating the same shape on both axes.
    @test length(collect(eachmatch(r"class=\"young-diagram ",state_svg)))==2
    @test length(collect(eachmatch(r"class=\"young-box\"",state_svg)))==6
    @test !occursin("young-row-diagram",state_svg)
    operator_svg=sprint(show,MIME("image/svg+xml"),
        visualize_schur_blocks(schur_block_structure(op)))
    @test length(collect(eachmatch(r"class=\"young-diagram ",operator_svg)))==2
    @test length(collect(eachmatch(r"class=\"young-box\"",operator_svg)))==6

    # Qudit partitions omit padded zero rows from the diagram while retaining
    # the complete partition label and exact representation metadata.
    b3=PIBasis(3,3)
    qutrit_structure=schur_block_structure(identity_operator(b3))
    qutrit_svg=sprint(show,MIME("image/svg+xml"),
        visualize_schur_blocks(qutrit_structure))
    @test length(collect(eachmatch(r"class=\"young-diagram ",qutrit_svg)))==3
    @test length(collect(eachmatch(r"class=\"young-box\"",qutrit_svg)))==9
    @test occursin("data-partition=\"(3,0,0)\"",qutrit_svg)
    @test occursin("U(3) irrep dimension=10",qutrit_svg)

    compact=visualize_schur_blocks(dense_structure;show_young_diagrams=false)
    compact_svg=sprint(show,MIME("image/svg+xml"),compact)
    @test !compact.show_young_diagrams
    @test compact.structure===visualization.structure
    @test !occursin("young-diagram",compact_svg)

    # Research-scale partitions use a bounded silhouette instead of one SVG
    # node per particle.
    large_svg=sprint(show,MIME("image/svg+xml"),
        visualize_schur_blocks(large_structure))
    @test occursin("data-rendering=\"compressed\"",large_svg)
    @test length(collect(eachmatch(r"class=\"young-row-band\"",large_svg)))==1
    @test !occursin("class=\"young-box\"",large_svg)

    # Exercise both the empty partition and genuinely multirow compressed
    # silhouettes.  Equal long rows keep the U(3) irrep one-dimensional, so
    # this renderer check does not allocate a large PI block.
    empty_basis=PIBasis(0,2)
    empty_svg=sprint(show,MIME("image/svg+xml"),visualize_schur_blocks(
        schur_block_structure(identity_operator(empty_basis))))
    @test occursin("class=\"young-empty-diagram\"",empty_svg)
    @test occursin("data-partition=\"(0,0)\"",empty_svg)
    multirow_basis=PIBasis(120,3;sectors=[(40,40,40)])
    multirow_svg=sprint(show,MIME("image/svg+xml"),visualize_schur_blocks(
        schur_block_structure(identity_operator(multirow_basis))))
    @test occursin("data-rendering=\"compressed\"",multirow_svg)
    @test length(collect(eachmatch(r"class=\"young-row-band\"",multirow_svg)))==3
    @test !occursin("class=\"young-box\"",multirow_svg)
    invalid_xml=visualize_schur_blocks(populations;title="bad\u0001title")
    @test_throws ArgumentError sprint(show,MIME("image/svg+xml"),invalid_xml)

    # All renderer scaling/normalization modes must handle inactive zero cells.
    for scale in (:linear,:log), normalization in (:global,:row,:column,:none)
        rendered=visualize_schur_blocks(dense_structure;scale=scale,
                                        normalize=normalization)
        rendered_svg=sprint(show,MIME("image/svg+xml"),rendered)
        @test occursin("</svg>",rendered_svg)
        @test !occursin(r"(?i)(nan|inf)",rendered_svg)
    end
    @test visualize_schur_blocks(rho;metric=:population) isa SchurBlockVisualization
    @test visualize_schur_blocks(Ls,b;metric=:maxabs) isa SchurBlockVisualization
    @test_throws TypeError visualize_schur_blocks(
        dense_structure;show_young_diagrams=:yes)
    calls_before_invalid_render=threshold_calls[]
    @test_throws TypeError visualize_schur_blocks(
        threshold_matrixfree,b;show_young_diagrams=:yes)
    @test threshold_calls[]==calls_before_invalid_render
    @test_throws ArgumentError visualize_schur_blocks(dense_structure;scale=:sqrt)
    @test_throws ArgumentError visualize_schur_blocks(dense_structure;normalize=:bad)

    mktempdir() do dir
        path=joinpath(dir,"schur-blocks.svg")
        returned=save_schur_block_visualization(path,visualization)
        @test returned==path
        @test isfile(path)
        @test read(path,String)==svg
    end
end
