@testset "PI PPT-mixture plan" begin
    PID=PermutationalInvariantDynamics
    basis=PIBasis(3,2)
    plan=PPTMixturePlan(basis)

    @test plan.basis===basis
    @test plan.cuts==[1]
    @test length(plan.blocks)==2
    @test size(plan.constraint_matrix)==(196,41)
    @test length(plan.equality_rows)==20
    @test plan.cone_dimensions==[12,12,4,4]
    @test plan.estimated_setup_bytes>0
    @test plan.estimated_solve_bytes>plan.estimated_setup_bytes
    @test occursin("variables=41",sprint(show,plan))

    # The paper requires only one representative of each cut size, not all
    # exponentially many labeled bipartitions.
    for N in 2:10
        shape=PID._ppt_plan_shape(N,Float64)
        expected=sum((fld(k,2)+1)*(fld(N-k,2)+1)
                     for k in 1:fld(N,2))
        @test shape.block_count==expected
        @test shape.max_block==maximum((k+1)*(N-k+1)
            for k in 1:fld(N,2))
    end

    @test_throws ArgumentError PPTMixturePlan(PIBasis(3,3))
    @test_throws ArgumentError PPTMixturePlan(PIBasis(1,2))
    @test_throws ArgumentError PPTMixturePlan(basis;T=BigFloat)
    @test_throws ArgumentError PPTMixturePlan(basis;memory_budget=1)
    @test PPTMixturePlan(basis;T=Float32).constraint_matrix isa
        SparseMatrixCSC{Float32,Int}
end

@testset "PPT-mixture numerical safeguards" begin
    PID=PermutationalInvariantDynamics
    basis=PIBasis(3,2)
    rho=ghz_state(basis)
    plan=PPTMixturePlan(basis)
    rhs=PID._ppt_mixture_rhs(plan,rho)

    # A failed optimizer may return correctly sized but nonfinite work
    # vectors.  Diagnostics must invalidate them without calling LAPACK on
    # NaNs, and classification must remain explicitly inconclusive.
    x=fill(NaN,size(plan.constraint_matrix,2))
    z=fill(NaN,size(plan.constraint_matrix,1))
    result=PID._ppt_classified_result(plan,rhs,x,z;
        solver_status=:numerical_error,solved=true,
        primal_objective=NaN,solver_dual_objective=NaN,
        iterations=1,solve_time=0.0,
        certificate_atol=1e-7,certificate_rtol=1e-7)
    @test result.classification==:inconclusive
    @test result.ppt_mixture===missing
    @test isinf(result.equality_residual)
    @test isinf(result.dual_stationarity_residual)

    @test PID._ppt_tolerance(Float32,1f-4,"test tolerance")===1f-4
    @test_throws ArgumentError PID._ppt_tolerance(
        Float32,1e-4,"test tolerance")
    @test_throws ArgumentError PID._ppt_tolerance(
        Float64,big"-1e-1000","test tolerance")
    @test_throws ArgumentError PID._ppt_tolerance(
        Float32,big(2)^200,"test tolerance")
    @test PID._ppt_time_limit(0.5f0)===0.5
    @test PID._ppt_time_limit(Inf)===Inf
    @test_throws ArgumentError PID._ppt_time_limit(0.0)
    @test_throws ArgumentError PID._ppt_time_limit(big"1.0")
    @test_throws ArgumentError PID._ppt_time_limit(big(2)^200+1)

    # Backend selection is checked before an absent plan would trigger its
    # polynomial-size setup or memory guard.
    unsupported=try
        ppt_mixture_test(rho;solver=:unknown,memory_budget=1)
        nothing
    catch error
        error
    end
    @test unsupported isa ArgumentError
    @test occursin("unsupported PPT-mixture solver",
        sprint(showerror,unsupported))
    if Base.get_extension(PID,
            :PermutationalInvariantDynamicsClarabelExt)===nothing
        missing_backend=try
            ppt_mixture_test(rho;memory_budget=1)
            nothing
        catch error
            error
        end
        @test missing_backend isa ArgumentError
        @test occursin("import Clarabel",sprint(showerror,missing_backend))
    end
end

@testset "PPT-mixture real conic coordinates" begin
    PID=PermutationalInvariantDynamics
    rng=MersenneTwister(713)
    basis=PIBasis(3,2)
    plan=PPTMixturePlan(basis)
    block=first(plan.blocks)
    n=block.dimension
    raw=randn(rng,ComplexF64,n,n)
    X=(raw+raw')/2
    x=zeros(Float64,size(plan.constraint_matrix,2))
    PID._ppt_pack_hermitian!(x,block.variables,X)
    @test PID._ppt_unpack_hermitian(x,block.variables,n)≈X atol=2e-15

    function real_embedding(A)
        [real.(A) -imag.(A);imag.(A) real.(A)]
    end
    margin=0.125
    x[1]=margin
    positive_slack=-plan.constraint_matrix[block.positive_rows,:]*x
    partial_slack=-plan.constraint_matrix[
        block.partial_transpose_rows,:]*x
    embedded_identity=Matrix{Float64}(I,2n,2n)
    @test PID._ppt_unpack_svec(positive_slack,
        eachindex(positive_slack),2n)≈
        real_embedding(X).-margin.*embedded_identity atol=2e-15
    pt=PID._ppt_product_partial_transpose(X,block.da,block.db)
    @test PID._ppt_unpack_svec(partial_slack,
        eachindex(partial_slack),2n)≈
        real_embedding(pt).-margin.*embedded_identity atol=2e-15
    @test dot(plan.objective,x)==-margin
    @test all(iszero,plan.constraint_matrix[plan.equality_rows,1])
    @test PID._ppt_product_partial_transpose(pt,block.da,block.db)≈X
    @test tr(pt)≈tr(X)
    @test ishermitian(pt)

    # Independent tensor-product oracle.  Product-Schur coordinates store the
    # first subsystem fastest, hence a product is kron(B,A) and T_A acts only
    # on the second Kronecker factor in Julia's ordering.
    A=randn(rng,ComplexF64,block.da,block.da)
    B=randn(rng,ComplexF64,block.db,block.db)
    product=kron(B,A)
    @test PID._ppt_product_partial_transpose(product,block.da,block.db)≈
        kron(B,transpose(A))
end

@testset "PPT-mixture Schur equality and restricted sectors" begin
    PID=PermutationalInvariantDynamics
    basis=PIBasis(3,2)
    rho=ghz_state(basis;phase=pi/2)
    plan=PPTMixturePlan(basis)
    rhs=PID._ppt_mixture_rhs(plan,rho)

    # Put the state itself into the k=1 product-Schur variables.  Its full PI
    # twirl is unchanged, so this is an exact oracle for Eq. (38), regardless
    # of whether the product block is PPT.
    x=zeros(Float64,size(plan.constraint_matrix,2))
    reduction=ReductionPlan(plan.full_basis,1)
    for coupling in reduction.couplings
        block=only(filter(candidate->candidate.cut==1&&
            candidate.alpha==coupling.alpha&&candidate.beta==coupling.beta,
            plan.blocks))
        product=PID._product_block(rho,coupling,
            coupling.product_multiplicity^2)
        PID._ppt_pack_hermitian!(x,block.variables,product)
    end
    @test plan.constraint_matrix[plan.equality_rows,:]*x≈
        rhs[plan.equality_rows] atol=2e-14

    # Exercise both inequivalent cuts, including the balanced 2|2 cut.  A
    # complete product-Schur expansion at either cut must twirl back to the
    # same four-qubit PI state.
    basis4=PIBasis(4,2)
    rho4=ghz_state(basis4;phase=pi/3)
    plan4=PPTMixturePlan(basis4)
    rhs4=PID._ppt_mixture_rhs(plan4,rho4)
    @test plan4.cuts==[1,2]
    for cut in plan4.cuts
        x4=zeros(Float64,size(plan4.constraint_matrix,2))
        reduction4=ReductionPlan(plan4.full_basis,cut)
        for coupling in reduction4.couplings
            block=only(filter(candidate->candidate.cut==cut&&
                candidate.alpha==coupling.alpha&&
                candidate.beta==coupling.beta,plan4.blocks))
            product4=PID._product_block(rho4,coupling,
                coupling.product_multiplicity^2)
            PID._ppt_pack_hermitian!(x4,block.variables,product4)
        end
        @test plan4.constraint_matrix[plan4.equality_rows,:]*x4≈
            rhs4[plan4.equality_rows] atol=5e-14
    end

    # A restricted state still produces the complete equality system, with
    # exact zeros for every omitted parent sector.
    restricted_basis=PIBasis(3,2;sectors=[(3,0)])
    restricted=ghz_state(restricted_basis;phase=pi/2)
    restricted_plan=PPTMixturePlan(restricted_basis)
    restricted_rhs=PID._ppt_mixture_rhs(restricted_plan,restricted)
    @test size(restricted_plan.constraint_matrix)==size(plan.constraint_matrix)
    @test restricted_rhs==rhs

    other_basis=PIBasis(3,2)
    @test_throws ArgumentError PID._ppt_mixture_rhs(plan,ghz_state(other_basis))
    unnormalized=PIState(basis,2rho.data)
    @test_throws ArgumentError ppt_mixture_test(unnormalized;plan)
    @test_throws ArgumentError ppt_mixture_test(rho;plan,solver=:unknown)
    # Clarabel is normally absent from the core test target, but keep this
    # file reusable in a session where the weak extension is already loaded.
    if Base.get_extension(PID,
            :PermutationalInvariantDynamicsClarabelExt)===nothing
        @test_throws ArgumentError ppt_mixture_test(rho;plan)
    end
end
