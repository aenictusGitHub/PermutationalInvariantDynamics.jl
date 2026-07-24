@testset "prepared observable and reduction geometry" begin
    sx=ComplexF64[0 1;1 0];sy=ComplexF64[0 -im;im 0];sz=ComplexF64[1 0;0 -1]
    b=PIBasis(4,2);rho=iid_pure_state(b,normalize(ComplexF64[1,1im]))
    cache=OneBodyGeometry(b)
    px=PermutationalInvariantDynamics.CollectiveObservablePlan(b,sx;cache=cache)
    py=PermutationalInvariantDynamics.CollectiveObservablePlan(b,sy;cache=cache)
    pz=PermutationalInvariantDynamics.CollectiveObservablePlan(b,sz;cache=cache)

    @test collective_operator(b,sx;cache=cache).data≈collective_operator(b,sx).data
    @test collective_operator(px).data≈collective_operator(b,sx;cache=cache).data
    @test Matrix(local_kernel_operator(b,sx,sy;cache=cache))≈Matrix(local_kernel_operator(b,sx,sy))
    @test collective_expectation(rho,px)≈collective_expectation(rho,sx;cache=cache)
    @test collective_moments(rho,px).mean≈collective_moments(rho,sx;plan=px).mean
    @test collective_variance(rho,px)≈collective_variance(rho,sx;plan=px)
    @test collective_covariance(rho,px,py)≈collective_covariance(rho,sx,sy;plans=(px,py))
    @test collective_covariance_matrix(rho,[px,py,pz])≈collective_covariance_matrix(rho,[sx,sy,sz];plans=(px,py,pz))
    @test qfi(rho,px)≈qfi(rho,sx;plan=px)
    @test qfim(rho,[px,py])≈qfim(rho,[sx,sy];plans=(px,py))

    wrong_basis=PIBasis(4,2)
    @test_throws ArgumentError collective_expectation(iid_pure_state(wrong_basis,ComplexF64[1,0]),px)
    @test_throws ArgumentError collective_expectation(rho,sz;plan=px)
    @test_throws ArgumentError collective_expectation(rho,sx;cache=cache,plan=px)
    @test_throws ArgumentError collective_operator(wrong_basis,sx;cache=cache)
    @test_throws ArgumentError local_kernel_operator(wrong_basis,sx,sy;cache=cache)
    @test_throws ArgumentError one_body_rdm(iid_pure_state(wrong_basis,ComplexF64[1,0]);cache=cache)

    # Squeezing reuses all three prepared spin blocks rather than constructing
    # four independent OneBodyGeometry objects.
    spinplans=(PermutationalInvariantDynamics.CollectiveObservablePlan(b,sx/2;cache=cache),
               PermutationalInvariantDynamics.CollectiveObservablePlan(b,sy/2;cache=cache),
               PermutationalInvariantDynamics.CollectiveObservablePlan(b,sz/2;cache=cache))
    @test kitagawa_ueda_squeezing(rho;plans=spinplans)≈kitagawa_ueda_squeezing(rho)
    kitagawa_ueda_squeezing(rho;plans=spinplans);kitagawa_ueda_squeezing(rho)
    @test @allocated(kitagawa_ueda_squeezing(rho;plans=spinplans)) < @allocated(kitagawa_ueda_squeezing(rho))

    collective_moments(rho,px);collective_moments(rho,sx;cache=cache)
    @test @allocated(collective_moments(rho,px)) < @allocated(collective_moments(rho,sx;cache=cache))

    # General state spanning every Schur sector exercises all cached qubit
    # recouplers, rather than only the symmetric sector.
    general=PIState(b;T=Float64)
    for (s,p) in pairs(b.sectors)
        n=length(b.patterns[s]);A=reshape(ComplexF64.(1:n^2),n,n)
        coefficient_block(general,p).=A*A'
    end
    PermutationalInvariantDynamics.normalize!(general)
    reduction=PermutationalInvariantDynamics.ReductionPlan(b,2)
    planned_state=reduced_state(general,2;plan=reduction)
    @test planned_state.basis===reduction.output_basis
    @test planned_state.data≈reduced_state(general,2).data atol=3e-10
    # Independent product-block oracle for the fused
    # tr_B(U*C*U†)=sum_q U_q*C*U_q† contraction.
    oracle_blocks=[
        zeros(ComplexF64,length(patterns),length(patterns))
        for patterns in reduction.output_basis.patterns]
    for coupling in reduction.couplings
        scale_squared=coupling.alpha_multiplicity*
            coupling.beta_multiplicity^2
        product=PermutationalInvariantDynamics._product_block(
            general,coupling,scale_squared)
        partial=PermutationalInvariantDynamics._partial_trace_b(
            product,coupling.da,coupling.db)
        oracle_blocks[reduction.output_basis.index[coupling.alpha]].+=partial
    end
    @test all(pairs(reduction.output_basis.sectors)) do (sector,partition)
        isapprox(coefficient_block(planned_state,partition),
                 oracle_blocks[sector];atol=3e-10)
    end
    @test reduced_purity(general,2;plan=reduction)≈reduced_purity(general,2) atol=3e-10
    @test negativity(general,2;plan=reduction)≈negativity(general,2) atol=3e-10
    planned_pt=partial_transpose_spectrum(general,2;plan=reduction)
    direct_pt=partial_transpose_spectrum(general,2)
    @test length(planned_pt)==length(direct_pt)
    @test all(x->x[1].alpha==x[2].alpha&&x[1].beta==x[2].beta&&
                 x[1].eigenvalues≈x[2].eigenvalues,zip(planned_pt,direct_pt))
    @test_throws ArgumentError reduced_state(general,1;plan=reduction)
    @test_throws ArgumentError reduced_state(PIState(wrong_basis,general.data),2;plan=reduction)

    # A caller-owned workspace reuses every product-block contraction buffer.
    # The in-place entry point additionally reuses the returned PI state.
    reduction_work=ReductionWorkspace(reduction,general)
    plan_recouplers=[U for c in reduction.couplings
                       for (_,intertwiners) in c.intertwiners
                       for U in intertwiners]
    workspace_recouplers=[U for connections in reduction_work.recoupling_intertwiners
                            for intertwiners in connections
                            for U in intertwiners]
    @test !isempty(plan_recouplers)
    @test length(workspace_recouplers)==length(plan_recouplers)
    # Keep the immutable qubit plan compact and real, but give every hot
    # contraction a homogeneous matrix eltype inside its task-local workspace.
    @test all(U->eltype(U)===Float64,plan_recouplers)
    @test all(U->eltype(U)===reduction_work.Ttype,workspace_recouplers)
    @test all(pair->pair[1]!==pair[2],zip(plan_recouplers,workspace_recouplers))
    # A Float32 request cannot narrow the plan's Float64 CG convention.
    reduction_work32=ReductionWorkspace(reduction;T=Float32)
    @test reduction_work32.Ttype===ComplexF64
    @test all(U->eltype(U)===ComplexF64,
        (U for connections in reduction_work32.recoupling_intertwiners
           for intertwiners in connections for U in intertwiners))
    workspace_state=reduced_state(general,2;plan=reduction,
                                  workspace=reduction_work)
    @test workspace_state.data≈planned_state.data atol=3e-10
    @test reduced_state(general,2;workspace=reduction_work).data≈
          planned_state.data atol=3e-10
    @test reduced_purity(general,2;plan=reduction,
                         workspace=reduction_work)≈purity(planned_state) atol=3e-10
    @test negativity(general,2;plan=reduction,
                     workspace=reduction_work)≈negativity(general,2;plan=reduction) atol=3e-10
    @test logarithmic_negativity(general,2;plan=reduction,
                                 workspace=reduction_work)≈
          logarithmic_negativity(general,2;plan=reduction) atol=3e-10

    # One-body scans reuse both the one-box geometry and the largest-sector
    # multiplicity-weighted block. The validate-once in-place path has no
    # state-sized transient allocation after warm-up.
    one_body_work=OneBodyRDMWorkspace(cache,general)
    one_body_output=zeros(ComplexF64,b.d,b.d)
    one_body_reference=one_body_rdm(
        general;plan=ReductionPlan(b,1))
    @test one_body_rdm!(
        one_body_output,general,one_body_work)===one_body_output
    @test one_body_output≈one_body_reference atol=3e-10
    @test one_body_rdm(general;workspace=one_body_work)≈
          one_body_reference atol=3e-10
    one_body_rdm!(
        one_body_output,general,one_body_work;check=false)
    one_body_alloc=@allocated one_body_rdm!(
        one_body_output,general,one_body_work;check=false)
    @test one_body_alloc<=1024
    @test occursin("OneBodyRDMWorkspace",sprint(show,one_body_work))
    @test_throws ArgumentError OneBodyRDMWorkspace(
        cache;memory_budget=1)
    one_body_numeric_payload=
        length(one_body_work.weighted_block)*
        sizeof(eltype(one_body_work.weighted_block))
    @test_throws ArgumentError OneBodyRDMWorkspace(
        cache;memory_budget=one_body_numeric_payload)
    @test_throws DimensionMismatch one_body_rdm!(
        zeros(ComplexF64,3,3),general,one_body_work)
    @test_throws ArgumentError one_body_rdm!(
        zeros(ComplexF32,2,2),general,one_body_work)
    @test_throws ArgumentError one_body_rdm!(
        view(one_body_work.weighted_block,1:2,1:2),
        general,one_body_work;check=false)

    reduction_only=@inferred ReductionWorkspace(
        reduction,general;mode=:reduction)
    negativity_only=ReductionWorkspace(reduction,general;mode=:negativity)
    max_product=maximum(c.da*c.db for c in reduction.couplings)
    max_output=maximum(c.da for c in reduction.couplings)
    @test isempty(reduction_only.product_block)
    @test size(reduction_only.product_tmp,1)==max_output
    @test size(reduction_work.product_tmp,1)==max_product
    @test isempty(reduction_only.partial_trace)
    @test isempty(reduction_work.partial_trace)
    @test isempty(reduction_only.partial_transpose)
    @test isempty(negativity_only.partial_trace)
    @test isempty(negativity_only.reduced_blocks)
    @test Base.summarysize(reduction_only)<Base.summarysize(reduction_work)
    @test Base.summarysize(negativity_only)<Base.summarysize(reduction_work)
    @test reduced_state(general,2;plan=reduction,
                        workspace=reduction_only).data≈planned_state.data atol=3e-10
    @test negativity(general,2;plan=reduction,
                     workspace=negativity_only)≈
          negativity(general,2;plan=reduction) atol=3e-10
    @test_throws ArgumentError negativity(
        general,2;plan=reduction,workspace=reduction_only)
    @test_throws ArgumentError reduced_state(
        general,2;plan=reduction,workspace=negativity_only)
    @test_throws ArgumentError ReductionWorkspace(reduction,general;mode=:invalid)
    inplace_state=PIState(reduction.output_basis)
    returned=reduced_state!(inplace_state,general,reduction,reduction_work)
    @test returned===inplace_state
    @test inplace_state.data≈planned_state.data atol=3e-10
    @test reduced_state!(inplace_state,general,2;
                         workspace=reduction_work)===inplace_state
    # Prepared callers may validate once and then reuse the allocation-light
    # contraction. Output normalization and resource checks remain enabled.
    @test (@inferred reduced_state!(
        inplace_state,general,reduction,reduction_only;check=false))===
        inplace_state
    unchecked_reduction_alloc=@allocated reduced_state!(
        inplace_state,general,reduction,reduction_only;check=false)
    checked_reduction_alloc=@allocated reduced_state!(
        inplace_state,general,reduction,reduction_only)
    @test unchecked_reduction_alloc<checked_reduction_alloc
    @test unchecked_reduction_alloc<=8*1024

    invalid=copy(iid_pure_state(b,ComplexF64[1,0]))
    invalid_block=coefficient_block(invalid,first(b.sectors))
    occupied=findfirst(!iszero,diag(invalid_block))
    empty=findfirst(iszero,diag(invalid_block))
    invalid_block[occupied,occupied]+=0.25
    invalid_block[empty,empty]-=0.25
    @test_throws ArgumentError reduced_state(
        invalid,2;plan=reduction,workspace=reduction_only)
    @test reduced_state(
        invalid,2;plan=reduction,workspace=reduction_only,
        check=false) isa PIState
    @test_throws ArgumentError reduced_purity(
        invalid,2;plan=reduction,workspace=reduction_only)
    @test reduced_purity(
        invalid,2;plan=reduction,workspace=reduction_only,
        check=false) isa Real
    @test_throws ArgumentError reduced_purities(invalid;ks=[0,2,4])
    @test length(reduced_purities(
        invalid;ks=[0,2,4],check=false))==3
    @test_throws ArgumentError one_body_rdm!(
        one_body_output,invalid,one_body_work)
    @test one_body_rdm!(
        one_body_output,invalid,one_body_work;check=false)===
          one_body_output
    @test_throws ArgumentError reduced_state!(
        inplace_state,general,reduction,reduction_only;
        check=false,atol=-1)
    @test_throws ArgumentError reduced_purity(
        general,2;check=false,rtol=-1)
    @test_throws ArgumentError reduced_purities(
        general;ks=[1],check=false,atol=-1)
    @test_throws ArgumentError reduced_state(
        general,2;check=false,atol=Inf)
    @test_throws ArgumentError reduced_purity(
        general,2;check=false,rtol=NaN)
    @test_throws ArgumentError one_body_rdm!(
        one_body_output,general,one_body_work;check=false,rtol=Inf)

    other_reduction=ReductionPlan(b,1)
    other_work=ReductionWorkspace(other_reduction,general)
    @test_throws ArgumentError reduced_state(
        general,2;plan=reduction,workspace=other_work)
    @test_throws ArgumentError reduced_state!(
        PIState(other_reduction.output_basis),general,reduction,reduction_work)

    reduced_state(general,2;plan=reduction,workspace=reduction_work)
    workspace_alloc=@allocated reduced_state(
        general,2;plan=reduction,workspace=reduction_work)
    planned_only_alloc=@allocated reduced_state(general,2;plan=reduction)
    @test workspace_alloc<planned_only_alloc
    @test reduced_state(general,0).basis.N==0
    @test reduced_state(general,b.N).basis===b
    @test reduced_state(general,b.N).data==general.data
    @test reduced_purity(general,0)==1
    @test reduced_purity(general,b.N)==purity(general)
    @test_throws ArgumentError reduced_state(
        general,b.N;plan=other_reduction)

    # Qudit LR nullspaces dominate setup; a fixed ReductionPlan amortizes that
    # cost across every subsequent state analysis.
    b3=PIBasis(2,3);rho3=maximally_mixed_state(b3)
    reduction3=PermutationalInvariantDynamics.ReductionPlan(b3,1)
    @test reduced_state(rho3,1;plan=reduction3).data≈reduced_state(rho3,1).data atol=3e-9
    @test negativity(rho3,1;plan=reduction3)≈negativity(rho3,1) atol=3e-9
    anti=Partition((1,1,0));entangled=basis_state(b3,anti,first(b3.patterns[b3.index[anti]]))
    @test negativity(entangled,1;plan=reduction3)≈0.5 atol=3e-10
    reduction_work3=ReductionWorkspace(reduction3,rho3)
    plan_recoupler3=first(first(first(reduction3.couplings).intertwiners)[2])
    workspace_recoupler3=first(first(reduction_work3.recoupling_intertwiners)[1])
    @test eltype(plan_recoupler3)===reduction_work3.Ttype===ComplexF64
    @test reduction3.estimates.storage===:weight_block_sparse_csc
    @test reduction3.estimates.retained_entries<
          reduction3.estimates.dense_entries
    # The LR plan already stores the required homogeneous type, so its packed
    # exact-support blocks are shared with the workspace rather than copied.
    @test workspace_recoupler3===plan_recoupler3
    @test reduced_state(rho3,1;plan=reduction3,
                        workspace=reduction_work3).data≈
          reduced_state(rho3,1;plan=reduction3).data atol=3e-9
    @test reduced_purity(rho3,1;plan=reduction3,
                         workspace=reduction_work3)≈1/3 atol=3e-9
    @test negativity(entangled,1;plan=reduction3,
                     workspace=reduction_work3)≈0.5 atol=3e-10
    @test first(first(reduction_work3.recoupling_intertwiners)[1])===
          workspace_recoupler3
    reduced_state(rho3,1;plan=reduction3);reduced_state(rho3,1)
    planned_alloc=@allocated reduced_state(rho3,1;plan=reduction3)
    direct_alloc=@allocated reduced_state(rho3,1)
    # Sparse one-body transition staging also makes the one-off path cheaper;
    # a retained LR plan must still reduce repeated-call allocation.
    @test planned_alloc<direct_alloc

    setprecision(BigFloat,128) do
        amplitude3=Complex{BigFloat}[
            BigFloat("0.6"),BigFloat("0.8")*im,zero(BigFloat)]
        big_rho3=iid_pure_state(b3,amplitude3)
        big_work3=ReductionWorkspace(
            reduction3,big_rho3;mode=:reduction)
        big_recoupler3=first(
            first(big_work3.recoupling_intertwiners)[1])
        @test eltype(big_recoupler3)===Complex{BigFloat}
        @test all(block->PermutationalInvariantDynamics.
                _reduction_precision_bounds(block.nzval)==(128,128),
            big_recoupler3.blocks)
        big_reduced3=reduced_state(
            big_rho3,1;plan=reduction3,workspace=big_work3)
        @test purity(big_reduced3)≈one(BigFloat) atol=big"3e-15"
    end

    setprecision(BigFloat,128) do
        big_basis=PIBasis(2,2)
        big_amplitude=Complex{BigFloat}[
            BigFloat("0.6"),BigFloat("0.8")*im]
        big_state=iid_pure_state(big_basis,big_amplitude)
        big_plan=ReductionPlan(big_basis,1)
        big_work=ReductionWorkspace(
            big_plan,big_state;mode=:reduction)
        big_reduced=reduced_state(
            big_state,1;plan=big_plan,workspace=big_work)
        @test eltype(big_reduced.data)===Complex{BigFloat}
        @test trace(big_reduced)≈one(BigFloat) atol=big"2e-15"
        @test purity(big_reduced)≈one(BigFloat) atol=big"2e-15"
        @test reduced_purity(
            big_state,1;plan=big_plan,workspace=big_work)≈
            one(BigFloat) atol=big"2e-15"
        big_geometry=OneBodyGeometry(big_basis,BigFloat)
        big_one_work=OneBodyRDMWorkspace(big_geometry,big_state)
        big_one=zeros(Complex{BigFloat},2,2)
        setprecision(BigFloat,64) do
            one_body_rdm!(big_one,big_state,big_one_work)
        end
        @test precision(real(big_one[1]))==128
        @test big_one≈big_amplitude*big_amplitude' atol=big"2e-15"
        ambient_big_one=setprecision(BigFloat,256) do
            one_body_rdm(big_state)
        end
        @test all(value->max(
            precision(real(value)),precision(imag(value)))==128,
            ambient_big_one)
        @test ambient_big_one≈big_one atol=big"2e-15"
        machine_geometry=OneBodyGeometry(big_basis,Float64)
        @test_throws ArgumentError OneBodyRDMWorkspace(
            machine_geometry,big_state)
    end

    # Reduction workspaces own one BigFloat context independently of the
    # ambient precision at construction or application.  This guards against
    # `fill!`, `mul!`, or output assignment replacing 192-bit entries by
    # newly-created ambient-precision values.
    context_basis=PIBasis(3,2)
    context_plan=ReductionPlan(context_basis,1)
    context_state=setprecision(BigFloat,192) do
        amplitude=Complex{BigFloat}[
            BigFloat("0.6"),BigFloat("0.8")*im]
        iid_pure_state(context_basis,amplitude)
    end
    context_work=setrounding(BigFloat,RoundDown) do
        setprecision(BigFloat,64) do
            ReductionWorkspace(
                context_plan,context_state;mode=:reduction)
        end
    end
    @test context_work.precision_bits==192
    @test context_work.rounding_mode==RoundDown
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        context_work.product_tmp)==(192,192)
    context_output=setprecision(BigFloat,192) do
        PIState(context_plan.output_basis;T=BigFloat)
    end
    setprecision(BigFloat,64) do
        @test reduced_state!(
            context_output,context_state,context_plan,context_work;
            check=false)===context_output
    end
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        context_output.data)==(192,192)
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        context_work.product_tmp)==(192,192)
    @test all(block->
        PermutationalInvariantDynamics._reduction_precision_bounds(block)==
            (192,192),context_work.reduced_blocks)

    reduced_low=setprecision(BigFloat,64) do
        reduced_state(
            context_state,1;plan=context_plan,
            workspace=context_work,check=false)
    end
    purity_low=setprecision(BigFloat,64) do
        reduced_purity(
            context_state,1;plan=context_plan,
            workspace=context_work,check=false)
    end
    reduced_high=setprecision(BigFloat,256) do
        reduced_state(
            context_state,1;plan=context_plan,
            workspace=context_work,check=false)
    end
    purity_high=setprecision(BigFloat,256) do
        reduced_purity(
            context_state,1;plan=context_plan,
            workspace=context_work,check=false)
    end
    @test reduced_low.data==reduced_high.data
    @test purity_low==purity_high
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        reduced_low.data)==(192,192)
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        reduced_high.data)==(192,192)
    @test precision(purity_low)==precision(purity_high)==192

    planned_low=setprecision(BigFloat,64) do
        reduced_state(
            context_state,1;plan=context_plan,check=false)
    end
    planned_purity_low=setprecision(BigFloat,64) do
        reduced_purity(
            context_state,1;plan=context_plan,check=false)
    end
    planned_high=setprecision(BigFloat,256) do
        reduced_state(
            context_state,1;plan=context_plan,check=false)
    end
    planned_purity_high=setprecision(BigFloat,256) do
        reduced_purity(
            context_state,1;plan=context_plan,check=false)
    end
    @test planned_low.data==planned_high.data
    @test planned_purity_low==planned_purity_high
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        planned_low.data)==(192,192)
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        planned_high.data)==(192,192)
    @test precision(planned_purity_low)==
          precision(planned_purity_high)==192
    purities_low=setprecision(BigFloat,64) do
        reduced_purities(
            context_state;ks=[0,1,3],check=false)
    end
    purities_high=setprecision(BigFloat,256) do
        reduced_purities(
            context_state;ks=[0,1,3],check=false)
    end
    @test purities_low==purities_high
    @test all(value->precision(value)==192,purities_low)
    planned_one_body=setprecision(BigFloat,64) do
        one_body_rdm(
            context_state;plan=context_plan,check=false)
    end
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        planned_one_body)==(192,192)
    workspace_one_body=setprecision(BigFloat,64) do
        one_body_rdm(
            context_state;plan=context_plan,
            workspace=context_work,check=false)
    end
    @test PermutationalInvariantDynamics._reduction_precision_bounds(
        workspace_one_body)==(192,192)

    endpoint_plan=ReductionPlan(context_basis,0)
    endpoint_work=ReductionWorkspace(
        endpoint_plan,context_state;mode=:negativity)
    log_negativity=setprecision(BigFloat,64) do
        logarithmic_negativity(
            context_state,0;plan=endpoint_plan,
            workspace=endpoint_work)
    end
    @test iszero(log_negativity)
    @test precision(log_negativity)==192
    setprecision(BigFloat,192) do
        half=BigFloat(1)/2
        converted_log=
            PermutationalInvariantDynamics._logarithmic_negativity_value(
                half,3)
        expected_log=log(BigFloat(2))/log(BigFloat(3))
        @test converted_log==expected_log
        @test precision(converted_log)==192
    end
    @test PermutationalInvariantDynamics._logarithmic_negativity_value(
        Float32(0.5),2) isa Float64

    state128=setprecision(BigFloat,128) do
        iid_pure_state(
            context_basis,Complex{BigFloat}[BigFloat(1),BigFloat(0)])
    end
    output64=setprecision(BigFloat,64) do
        PIState(context_plan.output_basis;T=BigFloat)
    end
    @test_throws ArgumentError reduced_purity(
        state128,1;plan=context_plan,workspace=context_work,check=false)
    @test_throws ArgumentError reduced_state!(
        output64,context_state,context_plan,context_work;check=false)
    mixed_state=copy(context_state)
    mixed_state.data[1]=setprecision(BigFloat,128) do
        Complex{BigFloat}(BigFloat(0),BigFloat(0))
    end
    @test_throws ArgumentError ReductionWorkspace(
        context_plan,mixed_state;mode=:reduction)
    @test_throws ArgumentError reduced_purity(
        mixed_state,1;plan=context_plan,
        workspace=context_work,check=false)

    psi3=normalize(ComplexF64[1,2im,-1]);pure3=iid_pure_state(PIBasis(3,3),psi3)
    @test one_body_rdm(pure3)≈psi3*psi3' atol=3e-10
    oneplan=PermutationalInvariantDynamics.ReductionPlan(pure3.basis,1)
    @test one_body_rdm(pure3;plan=oneplan)≈psi3*psi3' atol=3e-10

    H=collective_operator(b,sz;cache=cache);thermal=thermal_state(H,0.3)
    @test isphysical(thermal)
    badH=copy(H);C=coefficient_block(badH,first(b.sectors));C[1,2]+=1e-3
    @test_throws ArgumentError thermal_state(badH,0.3;atol=0,rtol=0)
end

@testset "typed one- and p-body geometry" begin
    b=PIBasis(2,2);sx32=ComplexF32[0 1;1 0]
    cache32=@inferred OneBodyGeometry(b,Float32)
    block32=@inferred collective_block(b,sx32,first(b.sectors);cache=cache32)
    element32=@inferred local_kernel_element(cache32,sx32,sx32,
        first(b.sectors),1,1,first(b.sectors),1,1)
    @test eltype(block32)===ComplexF32
    @test element32 isa ComplexF32
    @test eltype(local_kernel_operator(b,sx32,sx32;cache=cache32))===ComplexF32
    mean32=@inferred mean_local_operator(b,sx32;cache=cache32)
    @test eltype(mean32.data)===ComplexF32
    @test mean32.data≈(
        collective_operator(b,sx32;cache=cache32)*inv(Float32(b.N))).data
    @test fieldtype(typeof(cache32),:basis)===typeof(b)
    rho32=iid_pure_state(b,ComplexF32[1,im]/sqrt(2.0f0))
    one_body_work32=@inferred OneBodyRDMWorkspace(cache32,rho32)
    one_body_output32=zeros(ComplexF32,2,2)
    @test @inferred(one_body_rdm!(
        one_body_output32,rho32,one_body_work32))===one_body_output32
    @test one_body_output32≈
          ComplexF32[0.5 -0.5im;0.5im 0.5]
    zero_basis=PIBasis(0,1)
    @test_throws ArgumentError mean_local_operator(
        zero_basis,ones(ComplexF32,1,1);
        cache=OneBodyGeometry(zero_basis,Float32))

    pair32=kron(sx32,sx32);pcache32=@inferred PBodyGeometry(b,2,Float32)
    pblock32=@inferred pbody_collective_block(pcache32,pair32,first(b.sectors))
    @test eltype(pblock32)===ComplexF32
    @test eltype(pbody_kernel_operator(b,pair32,pair32,2;cache=pcache32))===ComplexF32
    @test fieldtype(typeof(pcache32),:basis)===typeof(b)

    sxbig=Complex{BigFloat}[0 1;1 0];cachebig=@inferred OneBodyGeometry(b,BigFloat)
    @test eltype(collective_block(b,sxbig,first(b.sectors);cache=cachebig))===Complex{BigFloat}
    @test eltype(local_kernel_operator(b,sxbig,sxbig;cache=cachebig))===Complex{BigFloat}
    pairbig=kron(sxbig,sxbig);pcachebig=@inferred PBodyGeometry(b,2,BigFloat)
    @test eltype(pbody_collective_block(pcachebig,pairbig,first(b.sectors)))===Complex{BigFloat}
    @test eltype(pbody_kernel_operator(b,pairbig,pairbig,2;cache=pcachebig))===Complex{BigFloat}

    plan32=@inferred LiouvillianPlan(PIModel(b,[LocalJump(sx32;rate=1f0)]))
    @test eltype(plan32)===ComplexF32
    hplan32=@inferred LiouvillianPlan(PIModel(b,[LocalHamiltonian(sx32)]))
    @test eltype(hplan32)===ComplexF32

    # Exact rates and hbar values are divided before either enormous operand
    # is converted. The ordinary floating path remains the direct fast divide.
    PID=PermutationalInvariantDynamics
    huge_rate=big(10)^400
    @test PID._scaled_rate(1.0,2.0,Float64)==0.5
    @test PID._scaled_rate(huge_rate,huge_rate,Float64)==1.0
    @test PID._scaled_rate(-huge_rate,huge_rate,Float32)==-1f0
    @test PID._scaled_rate(huge_rate//3,huge_rate//6,Float64)==2.0
    driven_scale=PID._scaled_rate((t,p)->huge_rate,huge_rate,Float64)
    @test driven_scale(0.0,nothing)==1.0
    @test_throws ArgumentError PID._scaled_rate(huge_rate,1,Float64)
    @test_throws ArgumentError PID._scaled_rate(1,0,Float64)
    @test_throws ArgumentError PID._scaled_rate(1.0,0.0,Float64)
    exact_plan=LiouvillianPlan(PIModel(b,[
        LocalHamiltonian(sx32;rate=huge_rate,hbar=huge_rate)]))
    @test eltype(exact_plan)===ComplexF32
    @test only(exact_plan.kernels).scale==1f0

    # In the N=18 singlet sector each multiplicity fits in Float16, but the
    # old product f_lambda*f_nu overflowed before its square root was taken.
    # The physical collective identity remains exactly N times the identity.
    singlet18=Partition((9,9))
    b18=PIBasis(18,2;sectors=[singlet18.parts])
    identity16=Matrix{Complex{Float16}}(I,2,2)
    cache16=OneBodyGeometry(b18,Float16)
    block16=collective_block(b18,identity16,singlet18;cache=cache16)
    @test all(isfinite,block16)
    @test block16[1,1]≈Float16(18) atol=Float16(0.125)

    # This restricted block is still one dimensional and cheap to prepare,
    # while f^(550,550) exceeds Float64.  It guards the complete cancellation
    # rather than merely the Float16 multiplication edge above.
    singlet1100=Partition((550,550))
    b1100=PIBasis(1100,2;sectors=[singlet1100.parts])
    cache1100=OneBodyGeometry(b1100,Float64)
    block1100=collective_block(b1100,ComplexF64[1 0;0 1],singlet1100;
                               cache=cache1100)
    @test all(isfinite,block1100)
    @test block1100[1,1]≈1100.0 atol=5e-11
    # At N=2100, sqrt(f^(1050,1050)) itself exceeds Float64.  A vanishing
    # physical operator remains a representable stored zero, while a genuinely
    # nonzero coefficient raises with wider-type guidance.
    singlet2100=Partition((1050,1050))
    b2100=PIBasis(2100,2;sectors=[singlet2100.parts])
    cache2100=OneBodyGeometry(b2100,Float64)
    zero_operator=collective_operator(
        b2100,zeros(ComplexF64,2,2);cache=cache2100)
    @test iszero(only(zero_operator.data))
    @test_throws ArgumentError collective_operator(
        b2100,ComplexF64[1 0;0 1];cache=cache2100)
end

@testset "shared one-box coefficients during mixed model compilation" begin
    PID=PermutationalInvariantDynamics
    basis=PIBasis(4,3)
    x=ComplexF64[0 1 0;1 0 1;0 1 0]
    pair=kron(x,x)
    model=PIModel(basis,(
        LocalHamiltonian(x;rate=0.17),
        PBodyHamiltonian(pair,2;rate=0.09)))

    context=PID.TermCompileContext(model)
    @test context.coefficient_cache isa OneBoxCGCache
    @test context.coefficient_cache.max_depth==2
    @test context.onebody isa OneBodyGeometry
    pgeometry=PID._pbody_geometry!(context,2)
    @test pgeometry isa PBodyGeometry
    @test context.coefficient_cache.coefficient_count>0

    explicit=OneBoxCGCache(basis;max_depth=2)
    cached_plan=LiouvillianPlan(model;coefficient_cache=explicit)
    automatic_plan=LiouvillianPlan(model)
    probe=ComplexF64.(1:length(basis))
    cached_output=similar(probe);automatic_output=similar(probe)
    apply!(cached_output,cached_plan,probe,0.0,nothing,
           LiouvillianWorkspace(cached_plan))
    apply!(automatic_output,automatic_plan,probe,0.0,nothing,
           LiouvillianWorkspace(automatic_plan))
    @test cached_output==automatic_output

    shallow=OneBoxCGCache(basis;max_depth=1)
    @test_throws ArgumentError LiouvillianPlan(
        model;coefficient_cache=shallow)
    other=PIBasis(4,3)
    @test_throws ArgumentError LiouvillianPlan(
        PIModel(other,(LocalHamiltonian(x),));coefficient_cache=explicit)
end


@testset "large-N collective geometry stability and setup scaling" begin
    PID=PermutationalInvariantDynamics

    # Normalize before storing equation-(7) coefficients. For d=1 this is a
    # one-coordinate oracle: the branching scale still fits Float16, while an
    # extensive local value of two overflows and its average remains exact.
    mean_basis=PIBasis(60_000,1)
    mean_cache=OneBodyGeometry(mean_basis,Float16)
    local_identity=fill(Complex{Float16}(2),1,1)
    @test_throws ArgumentError collective_operator(
        mean_basis,local_identity;cache=mean_cache)
    normalized_identity=mean_local_operator(
        mean_basis,local_identity;cache=mean_cache)
    @test eltype(normalized_identity.data)===Complex{Float16}
    @test only(normalized_identity.data)==Complex{Float16}(2)

    # A fixed spin-one irrep has an N-independent three-dimensional matrix
    # representation even when its Schur multiplicity is enormous.  The
    # individual one-box branches are O(N), however, so the traceless result
    # requires cancellation-safe accumulation at large N.
    for (N,T,tolerance) in ((60_000,Float16,Float16(0.02)),
                            (100_000_000,Float32,2f-6),
                            (1_000_000_000_000_000,Float32,2f-6),
                            (1_000_000_000_000,Float64,5e-14))
        sector=Partition((N÷2+1,N÷2-1))
        basis=PIBasis(N,2;sectors=[sector.parts])
        cache=OneBodyGeometry(basis,T)
        sx=Complex{T}[0 1;1 0]
        sy=Complex{T}[0 -im;im 0]
        sz=Complex{T}[1 0;0 -1]
        X=collective_block(basis,sx,sector;cache=cache)
        Y=collective_block(basis,sy,sector;cache=cache)
        Z=collective_block(basis,sz,sector;cache=cache)
        @test norm(X*Y-Y*X-2im*Z)<=tolerance
        @test norm(X*X+Y*Y+Z*Z-8I)<=tolerance
    end

    # The public local-map element uses the same large-N one-box geometry. For
    # an identity local operator it is exactly N times the identity map even in
    # a fixed-spin sector, so both a large diagonal sum and a cancellation zero
    # provide precision-independent oracles.
    Nlocal=10^12;local_sector=Partition((Nlocal÷2+1,Nlocal÷2-1))
    local_basis=PIBasis(Nlocal,2;sectors=[local_sector.parts])
    local_cache=OneBodyGeometry(local_basis,Float32)
    identity32=Matrix{ComplexF32}(I,2,2)
    local_diagonal=local_kernel_element(local_cache,identity32,identity32,
        local_sector,2,2,local_sector,2,2)
    local_zero=local_kernel_element(local_cache,identity32,identity32,
        local_sector,1,2,local_sector,2,1)
    @test local_diagonal==ComplexF32(Float32(Nlocal))
    @test iszero(local_zero)

    # Guarded-wide conversion must not round a mathematically out-of-range
    # nonzero to zero or to a finite IEEE endpoint.
    strict_sector=Partition((34,0))
    strict_basis=PIBasis(34,2;sectors=[strict_sector.parts])
    below_min=zeros(BigFloat,2,2)
    below_min[1,1]=BigFloat(nextfloat(zero(Float16)))/(2BigFloat(34))
    @test_throws ArgumentError PID._collective_block_wide(
        strict_basis,below_min,strict_sector,Float64,Complex{Float16})
    above_max=zeros(BigFloat,2,2)
    above_max[1,1]=BigFloat(floatmax(Float16))*(1+BigFloat(2)^(-20))/
                   BigFloat(34)
    @test_throws ArgumentError PID._collective_block_wide(
        strict_basis,above_max,strict_sector,Float64,Complex{Float16})

    # The estimator is purely structural and reflects the sparse transition
    # staging. Geometry construction for the symmetric N=100 block formerly
    # repeated millions of exact CG calculations and allocated gigabytes.
    symmetric=Partition((100,0))
    basis=PIBasis(100,2;sectors=[symmetric.parts])
    estimate=PID._estimate_onebody_geometry(basis,Float64)
    @test estimate.connection_count==1
    # Content staging evaluates only the structurally compatible parent
    # patterns, rather than the former parent-by-child Cartesian product.
    @test estimate.cgc_evaluations_upper==200
    @test estimate.setup_bytes<1_000_000
    OneBodyGeometry(basis,Float64) # compilation/warmup
    GC.gc()
    @test @allocated(OneBodyGeometry(basis,Float64))<5_000_000

    # Qudit weight spaces can have multiplicity greater than one.  The
    # structural estimate must bound the constructor's actual sparse tuples and
    # retained cache size there as well as for qubits.
    for (N,d) in ((8,2),(6,3),(4,4))
        candidate_basis=PIBasis(N,d)
        candidate_estimate=PID._estimate_onebody_geometry(candidate_basis,Float64)
        candidate_cache=OneBodyGeometry(candidate_basis,Float64)
        retained_terms=sum(length(terms) for contractions in
            values(candidate_cache.contractions) for terms in contractions)
        retained_excluding_basis=Base.summarysize(candidate_cache)-
            Base.summarysize(candidate_basis)
        @test candidate_estimate.contraction_terms_upper>=retained_terms
        @test candidate_estimate.retained_bytes>=retained_excluding_basis
        @test candidate_estimate.setup_bytes>=candidate_estimate.retained_bytes
        if d==3
            legacy_tables=Dict(key=>
                [collect(table[row,column]) for row in axes(table,1),
                                                   column in axes(table,2)]
                for (key,table) in candidate_cache.contractions)
            @test Base.summarysize(candidate_cache.contractions)<
                  Base.summarysize(legacy_tables)
            @test length(candidate_cache.connections.nonempty)<
                  length(candidate_cache.connections)
        end
    end
end
