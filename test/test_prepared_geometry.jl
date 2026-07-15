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
    inplace_state=PIState(reduction.output_basis)
    returned=reduced_state!(inplace_state,general,reduction,reduction_work)
    @test returned===inplace_state
    @test inplace_state.data≈planned_state.data atol=3e-10
    @test reduced_state!(inplace_state,general,2;
                         workspace=reduction_work)===inplace_state

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

    # Qudit LR nullspaces dominate setup; a fixed ReductionPlan amortizes that
    # cost across every subsequent state analysis.
    b3=PIBasis(2,3);rho3=maximally_mixed_state(b3)
    reduction3=PermutationalInvariantDynamics.ReductionPlan(b3,1)
    @test reduced_state(rho3,1;plan=reduction3).data≈reduced_state(rho3,1).data atol=3e-9
    @test negativity(rho3,1;plan=reduction3)≈negativity(rho3,1) atol=3e-9
    anti=Partition((1,1,0));entangled=basis_state(b3,anti,first(b3.patterns[b3.index[anti]]))
    @test negativity(entangled,1;plan=reduction3)≈0.5 atol=3e-10
    reduction_work3=ReductionWorkspace(reduction3,rho3)
    @test reduced_state(rho3,1;plan=reduction3,
                        workspace=reduction_work3).data≈
          reduced_state(rho3,1;plan=reduction3).data atol=3e-9
    @test reduced_purity(rho3,1;plan=reduction3,
                         workspace=reduction_work3)≈1/3 atol=3e-9
    @test negativity(entangled,1;plan=reduction3,
                     workspace=reduction_work3)≈0.5 atol=3e-10
    reduced_state(rho3,1;plan=reduction3);reduced_state(rho3,1)
    planned_alloc=@allocated reduced_state(rho3,1;plan=reduction3)
    direct_alloc=@allocated reduced_state(rho3,1)
    # Sparse one-body transition staging also makes the one-off path cheaper;
    # a retained LR plan must still reduce repeated-call allocation.
    @test planned_alloc<direct_alloc

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
    end
end
