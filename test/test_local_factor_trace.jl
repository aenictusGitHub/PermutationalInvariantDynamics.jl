function _dense_local_factor_transform_oracle(
        basis,letters,counts,::Type{R}) where R<:AbstractFloat
    builder=PermutationalInvariantDynamics._LocalFactorColumnBuilder(
        basis,letters,R)
    matrix=zeros(Complex{R},length(basis),length(counts))
    for (column,occupation) in pairs(counts)
        PermutationalInvariantDynamics._local_factor_column!(
            view(matrix,:,column),builder,occupation)
    end
    matrix
end

@testset "prepared local-factor trace" begin
    # A local Bell state loses its spin coherence when its second factor is
    # traced. This is already nontrivial at N=1.
    one_basis=PIBasis(1,4)
    bell=ComplexF64[1,0,0,1]/sqrt(2)
    one_source=iid_pure_state(one_basis,bell)
    one_plan=LocalFactorTracePlan(one_source,(2,2);traced_factor=2)
    one_work=LocalFactorTraceWorkspace(one_plan)
    one_output=local_factor_trace(one_source,one_plan;workspace=one_work)
    @test one_output.basis===one_plan.output_basis
    @test isapprox(one_output.data,
        iid_state(one_plan.output_basis,
                  Matrix{ComplexF64}(I,2,2)/2).data;
        atol=8e-13,rtol=8e-13)
    @test one_plan.estimates.gram_residual<1e-12
    @test one_plan.estimates.trace_residual<1e-12

    # Both tuple factors follow Julia kron ordering: the second factor is the
    # fast local index.
    rho2=ComplexF64[0.7 0.1im;-0.1im 0.3]
    rho3=ComplexF64[0.5 0.08 0;0.08 0.3 0.04;0 0.04 0.2]
    composite=kron(rho2,rho3)
    source_basis=PIBasis(2,6)
    source=iid_state(source_basis,composite)

    keep_first_plan=LocalFactorTracePlan(
        source_basis,(2,3);traced_factor=2)
    keep_first=local_factor_trace(source,keep_first_plan)
    @test isapprox(keep_first.data,
        iid_state(keep_first_plan.output_basis,rho2).data;
        atol=2e-11,rtol=2e-11)

    keep_second_plan=LocalFactorTracePlan(
        source_basis,(2,3);traced_factor=1)
    keep_second=local_factor_trace(source,keep_second_plan)
    @test isapprox(keep_second.data,
        iid_state(keep_second_plan.output_basis,rho3).data;
        atol=3e-11,rtol=3e-11)

    # A source restricted to its fully symmetric supersite sector can reduce
    # to a spin state occupying all Young sectors.
    N=3
    restricted_basis=PIBasis(
        N,4;sectors=[ntuple(index->index==1 ? N : 0,4)])
    restricted_source=iid_pure_state(restricted_basis,bell)
    restricted_plan=LocalFactorTracePlan(
        restricted_basis,(2,2);traced_factor=2)
    restricted_output=local_factor_trace(restricted_source,restricted_plan)
    @test length(restricted_plan.output_basis.sectors)>1
    @test isapprox(restricted_output.data,
        maximally_mixed_state(restricted_plan.output_basis).data;
        atol=3e-11,rtol=3e-11)

    # An embedded spin GHZ state retains its bipartite spin negativity after
    # the vacuum mode is removed.
    ghz_source_basis=PIBasis(4,4)
    symmetric=Partition((4,0,0,0))
    ground=computational_product_state(ghz_source_basis,1)
    excited=computational_product_state(ghz_source_basis,3)
    ground_block=coefficient_block(ground,symmetric)
    excited_block=coefficient_block(excited,symmetric)
    ground_index=argmax(real.(diag(ground_block)))
    excited_index=argmax(real.(diag(excited_block)))
    amplitudes=zeros(ComplexF64,size(ground_block,1))
    amplitudes[ground_index]=inv(sqrt(2))
    amplitudes[excited_index]=inv(sqrt(2))
    ghz_source=sector_density_matrix(
        ghz_source_basis,symmetric,amplitudes*amplitudes')
    ghz_plan=LocalFactorTracePlan(
        ghz_source_basis,(2,2);traced_factor=2)
    ghz_spin=local_factor_trace(ghz_source,ghz_plan)
    @test isapprox(
        ghz_spin.data,ghz_state(ghz_plan.output_basis).data;
        atol=3e-11,rtol=3e-11)
    cut=ReductionPlan(ghz_plan.output_basis,2)
    cut_work=ReductionWorkspace(cut,ghz_spin;mode=:negativity)
    @test isapprox(
        negativity(ghz_spin,2;plan=cut,workspace=cut_work),0.5;
        atol=3e-11)
end

@testset "local-factor trace identities and ownership" begin
    basis=PIBasis(3,4)
    psi1=ComplexF64[1,0.2im,0.4,0.1]
    psi1./=norm(psi1)
    psi2=ComplexF64[0.1,0.7,-0.2im,0.5]
    psi2./=norm(psi2)
    first_state=iid_pure_state(basis,psi1)
    second_state=iid_pure_state(basis,psi2)
    source=PIState(
        basis,0.4 .* first_state.data .+ 0.6 .* second_state.data)

    plan=LocalFactorTracePlan(source,(2,2);traced_factor=2)
    workspace=LocalFactorTraceWorkspace(plan)
    output=PIState(plan.output_basis;T=Float64)
    @test plan.lifted_columns isa SparseMatrixCSC{ComplexF64,Int}
    @test plan.output_columns isa SparseMatrixCSC{ComplexF64,Int}
    @test all(!iszero,nonzeros(plan.lifted_columns))
    @test all(!iszero,nonzeros(plan.output_columns))
    @test plan.estimates.storage===:exact_support_sparse_csc
    @test plan.estimates.lifted_nonzeros==nnz(plan.lifted_columns)
    @test plan.estimates.output_nonzeros==nnz(plan.output_columns)
    @test plan.estimates.gram_validation===:streamed_sparse_columns
    @test plan.estimates.gram_workspace_bytes>0
    @test plan.estimates.gram_pair_products>=
          nnz(plan.output_columns)
    @test plan.estimates.retained_entries==
          nnz(plan.lifted_columns)+nnz(plan.output_columns)
    @test plan.estimates.retained_bytes>=
          Base.summarysize(plan.lifted_columns)+
          Base.summarysize(plan.output_columns)
    @test 10*plan.estimates.retained_entries<
          plan.estimates.dense_entries
    @test plan.estimates.peak_bytes>=plan.estimates.retained_bytes
    occupations=PermutationalInvariantDynamics._local_factor_compositions(
        basis.N,Val(4))
    dense_output=_dense_local_factor_transform_oracle(
        plan.output_basis,
        PermutationalInvariantDynamics._local_factor_matrix_unit_letters(2),
        occupations,Float64)
    dense_lifted=_dense_local_factor_transform_oracle(
        basis,
        PermutationalInvariantDynamics._local_factor_lifted_letters(
            (2,2),2),
        occupations,Float64)
    @test Matrix(plan.output_columns)==dense_output
    @test Matrix(plan.lifted_columns)==dense_lifted

    # Streamed sparse validation has the same infinity-norm semantics as an
    # explicitly materialized Gram matrix, including missing diagonal entries,
    # but does not retain that quadratic matrix.
    probe=sparse(ComplexF64[
        1.0 0.2im 0.0;
        0.0 0.7   0.0;
        0.0 0.0   0.0])
    streamed_residual,streamed_scale,streamed_statistics=
        PermutationalInvariantDynamics.
        _local_factor_streamed_gram_validation(
            probe;memory_budget=Inf)
    dense_gram=Matrix(adjoint(probe)*probe)-I
    @test streamed_residual≈opnorm(dense_gram,Inf) atol=1e-15 rtol=1e-15
    @test streamed_scale≈max(1.0,opnorm(probe,Inf)^2) atol=1e-15 rtol=1e-15
    identity_probe=spdiagm(0=>ones(ComplexF64,64))
    _,_,identity_statistics=PermutationalInvariantDynamics.
        _local_factor_streamed_gram_validation(
            identity_probe;memory_budget=Inf)
    @test identity_statistics.workspace_entries<
          length(identity_probe)^2
    @test_throws ArgumentError PermutationalInvariantDynamics.
        _local_factor_streamed_gram_validation(
            probe;memory_budget=1)
    @test_throws ErrorException PermutationalInvariantDynamics.
        _local_factor_streamed_gram_validation(
            sparse(reshape(ComplexF64[Inf],1,1));memory_budget=Inf)
    @test_throws ErrorException PermutationalInvariantDynamics.
        _local_factor_streamed_gram_validation(
            sparse(reshape(ComplexF64[1e308],1,1));memory_budget=Inf)
    @test local_factor_trace!(output,source,plan,workspace)===output

    # The adjoint of local partial trace inserts an identity on the removed
    # factor, including for non-Hermitian local observables.
    X=ComplexF64[0 1;0 0]
    Y=ComplexF64[0.2 0.3im;-0.1im 0.7]
    identity_mode=Matrix{ComplexF64}(I,2,2)
    @test isapprox(
        ordered_local_moment(output,(X,Y)),
        ordered_local_moment(
            source,(kron(X,identity_mode),kron(Y,identity_mode)));
        atol=3e-11,rtol=3e-11)

    # Tracing particles and tracing one factor inside every particle commute.
    reduced_after=reduced_state(output,2)
    reduced_source=reduced_state(source,2)
    reduced_plan=LocalFactorTracePlan(
        reduced_source,(2,2);traced_factor=2)
    factor_after=local_factor_trace(reduced_source,reduced_plan)
    @test reduced_after.data≈factor_after.data atol=4e-11 rtol=4e-11

    # The hot prepared path performs only the two planned matrix-vector
    # contractions and reuses its occupation scratch.
    local_factor_trace!(output,source,plan,workspace;check=false)
    @test @allocated(local_factor_trace!(
        output,source,plan,workspace;check=false))<1_024

    equivalent_basis=PIBasis(3,4)
    equivalent_state=PIState(equivalent_basis,copy(source.data))
    @test_throws ArgumentError local_factor_trace(
        equivalent_state,plan;workspace)
    wrong_output=PIState(PIBasis(3,2);T=Float64)
    @test_throws ArgumentError local_factor_trace!(
        wrong_output,source,plan,workspace)
    other_plan=LocalFactorTracePlan(source,(2,2);traced_factor=2)
    other_workspace=LocalFactorTraceWorkspace(other_plan)
    @test_throws ArgumentError local_factor_trace!(
        output,source,plan,other_workspace)
    @test_throws ArgumentError local_factor_trace!(
        output,source,plan,workspace;atol=Inf)
    @test_throws ArgumentError local_factor_trace!(
        output,source,plan,workspace;rtol=NaN)
    source32=PIState(basis,ComplexF32.(source.data))
    @test_throws ArgumentError local_factor_trace(
        source32,plan;workspace)
    @test_throws ArgumentError LocalFactorTracePlan(
        source32,(2,2);T=Float64)
    @test_throws ArgumentError LocalFactorTracePlan(
        source,(2,2);T=Float32)

    @test_throws DimensionMismatch LocalFactorTracePlan(basis,(2,3))
    @test_throws ArgumentError LocalFactorTracePlan(
        basis,(2,2);traced_factor=3)
    @test_throws ArgumentError LocalFactorTracePlan(
        basis,(2,2);memory_budget=1)
end

@testset "local-factor trace scalar and boundary cases" begin
    basis32=PIBasis(2,4)
    local32=ComplexF32[0.6 0 0 0.1;0 0.1 0 0;
                       0 0 0.2 0;0.1 0 0 0.1]
    source32=iid_state(basis32,local32)
    plan32=LocalFactorTracePlan(source32,(2,2);traced_factor=2)
    output32=local_factor_trace(source32,plan32)
    @test eltype(plan32.lifted_columns)===ComplexF32
    @test eltype(output32.data)===ComplexF32
    @test isphysical(output32)

    # Removing a one-dimensional factor is the identity map up to the
    # deliberately distinct output-basis object.
    identity_basis=PIBasis(3,2)
    identity_source=iid_state(
        identity_basis,ComplexF64[0.7 0.1;0.1 0.3])
    identity_plan=LocalFactorTracePlan(
        identity_basis,(2,1);traced_factor=2)
    identity_output=local_factor_trace(identity_source,identity_plan)
    @test identity_output.data≈identity_source.data atol=2e-12 rtol=2e-12

    # The empty tensor product has one coordinate for every local dimension.
    empty_basis=PIBasis(0,6)
    empty_source=PIState(empty_basis,ComplexF64[1])
    empty_plan=LocalFactorTracePlan(
        empty_basis,(2,3);traced_factor=2)
    empty_output=local_factor_trace(empty_source,empty_plan)
    @test empty_output.basis.N==0
    @test empty_output.basis.d==2
    @test empty_output.data==ComplexF64[1]

    # A state-owned constructor captures storage precision rather than the
    # caller's ambient BigFloat precision.
    wide_source=setprecision(BigFloat,128) do
        wide_basis=PIBasis(1,4)
        local_density=Matrix{Complex{BigFloat}}(I,4,4)/BigFloat(4)
        iid_state(wide_basis,local_density)
    end
    wide_plan=setprecision(BigFloat,256) do
        LocalFactorTracePlan(wide_source,(2,2);traced_factor=2)
    end
    @test wide_plan.estimates.precision_bits==128
    wide_output=setprecision(BigFloat,64) do
        local_factor_trace(wide_source,wide_plan)
    end
    @test all(value->precision(real(value))==128,
              wide_output.data)
end
