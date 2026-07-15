@testset "Matrix-free Krylov solvers" begin
    b=PIBasis(3,2);sm=ComplexF64[0 1;0 0]
    model=PIModel(b,[LocalJump(sm;rate=0.7),CollectiveJump(sm;rate=0.2)])
    Lm=liouvillian(model;representation=:matrixfree)
    reference=steady_state(model;method=:direct,return_info=true)

    ws=KrylovWorkspace(Lm,10)
    ks=krylov_steady_state(Lm;workspace=ws,krylovdim=10,maxiter=200,
                           atol=1e-12,rtol=1e-10,return_info=true)
    @test ks.method===:krylov
    @test ks.converged
    @test ks.state≈reference.state atol=2e-9
    @test ks.residual<1e-9
    @test ks.trace_error<1e-10
    @test steady_state(model;method=:gmres,maxiter=200)≈reference.state atol=2e-8

    P=schur_sector_preconditioner(Lm,b;expected_reuses=10,warn_unamortized=false)
    @test size(P)==size(Lm)
    @test length(P.factors)==length(b.sectors)
    @test P.operator_scale>0
    @test P.metadata.setup_liouvillian_applications==length(b)+3
    @test P.metadata.apply_triangular_solves==length(b.sectors)
    @test P.metadata.amortization_expected
    @test PermutationalInvariantDynamics.preconditioner_cost(P)===P.metadata
    pks=steady_state(Lm;basis=b,method=:krylov,preconditioner=P,
                     workspace=ws,maxiter=200,atol=1e-12,rtol=1e-10,
                     return_info=true)
    @test pks.state≈reference.state atol=2e-9
    @test pks.preconditioned
    @test pks.preconditioner===typeof(P)
    @test steady_state(model;method=:krylov,preconditioner=:schur,
                       maxiter=200)≈reference.state atol=2e-8
    @test_throws ArgumentError steady_state(Lm;method=:krylov,preconditioner=:schur)
    @test_throws ArgumentError schur_sector_preconditioner(Lm,b;regularization=-1)

    dense=pi_liouvillian_spectrum(model)
    arn=krylov_liouvillian_spectrum(Lm;nev=4,krylovdim=length(b),vectors=true)
    @test arn.values≈dense[1:4] atol=2e-10
    @test maximum(arn.residuals)<2e-10
    @test norm(hcat((Lm*arn.vectors[:,j] for j in axes(arn.vectors,2))...)-
               arn.vectors*Diagonal(arn.values))<2e-9
    @test pi_liouvillian_spectrum(model;method=:krylov,nev=4,
                                  krylovdim=length(b))≈dense[1:4] atol=2e-10

    densegap=pi_liouvillian_gap(model)
    kgap=pi_liouvillian_gap(model;method=:krylov,nev=4,
                            krylovdim=length(b),return_info=true)
    @test kgap.gap≈densegap atol=2e-10
    @test kgap.method===:krylov
    @test kgap.stationary_multiplicity_certified
    @test_throws ArgumentError pi_liouvillian_gap(model;method=:krylov,symmetry=:auto)

    harmonic=harmonic_arnoldi_spectrum(Lm;nev=3,krylovdim=length(b),
        vectors=true,rng=MersenneTwister(11),atol=1e-9,rtol=1e-7)
    @test maximum(harmonic.residuals)<2e-7
    @test sort(abs.(harmonic.values))[1]<2e-7
    @test harmonic.algorithm===:harmonic
    restarted=harmonic_arnoldi_spectrum(Lm;nev=2,krylovdim=8,
        maxrestarts=2,require_convergence=false,atol=0,rtol=0,
        rng=MersenneTwister(111))
    @test restarted.restarts==2

    # Exact-shift implicit QR restarting preserves the matrix-free Arnoldi
    # factorization and reaches the spectral edge with a much smaller basis
    # than the PI dimension.
    advanced_seed=randn(MersenneTwister(112),ComplexF64,length(b))
    implicit=pi_liouvillian_spectrum(model;method=:iram,nev=2,krylovdim=10,
        retained_dimension=5,maxrestarts=30,
        initial_vector=advanced_seed,vectors=true,
        atol=1e-9,rtol=1e-7)
    @test implicit.algorithm===:implicit_qr_arnoldi
    @test implicit.method===:iram && implicit.selection===:LR
    @test implicit.restarts>0
    @test implicit.values≈dense[1:2] atol=2e-7
    @test norm(hcat((Lm*implicit.vectors[:,j] for j in axes(implicit.vectors,2))...)-
               implicit.vectors*Diagonal(implicit.values))<3e-7
    implicit_gap=pi_liouvillian_gap(model;method=:iram,nev=2,krylovdim=10,
        retained_dimension=5,maxrestarts=30,return_info=true,
        initial_vector=advanced_seed,atol=1e-9,rtol=1e-7)
    @test implicit_gap.gap≈densegap atol=2e-7
    @test implicit_gap.method===:iram && implicit_gap.selection===:largest_real
    @test_throws ArgumentError pi_liouvillian_gap(model;method=:jd)

    # Jacobi--Davidson hard-locks an invariant subspace, so it resolves the
    # twofold -0.65 mode that a scalar unrestarted Krylov sequence cannot span.
    # Its correction GMRES reuses the Schur-sector preconditioner constructed
    # above and never materializes the matrix-free Liouvillian.
    jdworkspace=JacobiDavidsonWorkspace(Lm,10,8)
    jd=jacobi_davidson_spectrum(Lm;nev=3,target=0,subspace_dim=10,
        maxiter=120,correction_krylovdim=8,correction_maxiter=40,
        initial_vector=advanced_seed,preconditioner=P,workspace=jdworkspace,
        rng=MersenneTwister(113),vectors=true,atol=1e-9,rtol=1e-7)
    @test jd.algorithm===:jacobi_davidson
    @test jd.preconditioned && jd.hard_locked==3
    @test jd.workspace_reused
    @test sort(jd.values;by=real,rev=true)≈dense[1:3] atol=3e-7
    @test norm(hcat((Lm*jd.vectors[:,j] for j in axes(jd.vectors,2))...)-
               jd.vectors*Diagonal(jd.values))<3e-7

    # A genuinely nonnormal triangular matrix checks explicit right-vector
    # residuals and the projected preconditioner contract independently of PI
    # model structure.
    jdvals=ComplexF64[0,-0.05,-0.1,-1,-2,-3,-4,-5,-6,-7]
    nonnormal=Matrix(Diagonal(jdvals))
    for i in 1:size(nonnormal,1)-1;nonnormal[i,i+1]=0.15;end
    shifted_lu=lu(nonnormal-0.02I)
    nonnormal_jd=pi_liouvillian_spectrum(nonnormal;method=:jd,nev=3,target=0,
        krylovdim=7,maxiter=120,correction_krylovdim=6,
        correction_maxiter=30,preconditioner=shifted_lu,
        initial_vector=randn(MersenneTwister(114),ComplexF64,10),
        rng=MersenneTwister(115),vectors=true,atol=1e-10,rtol=1e-8)
    @test nonnormal_jd.values≈jdvals[1:3] atol=2e-7
    @test maximum(nonnormal_jd.residuals)<8e-8
    @test nonnormal_jd.hard_locked==3
    @test nonnormal_jd.method===:jd && nonnormal_jd.selection===:near_target
    narrow_jd=JacobiDavidsonWorkspace(ComplexF32,10,7,6)
    @test_throws ArgumentError jacobi_davidson_spectrum(nonnormal;nev=2,
        target=0,subspace_dim=7,correction_krylovdim=6,workspace=narrow_jd,
        initial_vector=ones(ComplexF32,10),maxiter=2)
    zero_jd=jacobi_davidson_spectrum(zeros(ComplexF64,4,4);nev=2,
        vectors=true)
    @test zero_jd.values==zeros(ComplexF64,2)
    @test zero_jd.residuals==zeros(2)
    @test all(zero_jd.converged)
    @test zero_jd.vectors'*zero_jd.vectors≈I

    sz=ComplexF64[1 0;0 -1]
    projector=matrixfree_symmetry_projector(b,sz;charge=:trivial)
    x=randn(MersenneTwister(12),ComplexF64,length(b));px=projector*x
    @test projector*px≈px atol=2e-12
    @test dot(x,projector*x)≈dot(projector*x,projector*x) atol=2e-12
    @test Lm*px≈projector*(Lm*x) atol=2e-11
    projected=pi_liouvillian_spectrum(model;method=:harmonic,symmetry=sz,
        charge=1,nev=3,krylovdim=12,maxrestarts=12,vectors=true,
        rng=MersenneTwister(13),atol=1e-9,rtol=1e-7)
    @test projected.symmetry_used
    @test projected.symmetry_charge≈1
    @test maximum(projected.residuals)<2e-7
    hgap=pi_liouvillian_gap(model;method=:harmonic,symmetry=sz,charge=1,
        nev=3,krylovdim=length(b),maxrestarts=3,return_info=true,
        rng=MersenneTwister(14),atol=1e-9,rtol=1e-7)
    Pmat=hcat((projector*Matrix{ComplexF64}(I,length(b),length(b))[:,j] for j in 1:length(b))...)
    PE=eigen(Hermitian((Pmat+Pmat')/2));Q=PE.vectors[:,PE.values.>0.5]
    pvals=eigvals(Q'*Matrix(liouvillian(model;representation=:sparse))*Q)
    ptol=1e-9;pgap=-maximum(real(v) for v in pvals if abs(v)>ptol)
    @test hgap.gap≈pgap atol=2e-7
    @test hgap.method===:harmonic && hgap.symmetry_used
    @test hgap.scope===:charge_sector
    @test hgap.selection===:near_zero
    @test !hgap.gap_certified
    @test !hgap.stability_certified
    @test_throws ArgumentError pi_liouvillian_gap(model;method=:harmonic,
        symmetry=sz,charge=1,nev=3,krylovdim=length(b))

    # Harmonic extraction orders by distance to zero, not by real part.  The
    # highly oscillatory -0.1+100im mode controls the global gap but is not a
    # near-zero-in-modulus mode, so a harmonic global-gap request must fail
    # explicitly rather than silently report the -1 mode.
    counterexample=Diagonal(ComplexF64[0,-0.1+100im,-1])
    @test pi_liouvillian_gap(counterexample)≈0.1
    @test_throws ArgumentError pi_liouvillian_gap(counterexample;
        method=:harmonic,nev=2,krylovdim=3,return_info=true)

    # A nontrivial conjugation charge usually contains no stationary mode.
    # Its decay rate is the negative sector spectral abscissa and must not
    # require an artificial zero eigenvalue.
    b1=PIBasis(1,2)
    charged=Diagonal(ComplexF64[0,-0.2,-0.3,-1])
    qdecay=pi_liouvillian_gap(charged;basis=b1,method=:harmonic,
        symmetry=sz,charge=-1,nev=2,krylovdim=4,return_info=true,
        rng=MersenneTwister(15),atol=1e-10,rtol=1e-8)
    @test qdecay.gap≈0.2 atol=2e-8
    @test qdecay.stationary_multiplicity==0
    @test qdecay.symmetry_charge≈-1
    @test qdecay.scope===:charge_sector
    @test qdecay.gap_certified
    @test qdecay.stability_certified
    @test qdecay.sector_dimension==2
    @test pi_liouvillian_gap(charged;basis=b1,method=:harmonic,
        symmetry=sz,charge=-1,nev=2,krylovdim=4,
        rng=MersenneTwister(15),atol=1e-10,rtol=1e-8)≈0.2 atol=2e-8
    @test_throws ArgumentError matrixfree_symmetry_projector(b,sz;charge=im)

    @test_throws ArgumentError krylov_steady_state(Lm;trace_vector=zeros(length(b)))
    @test_throws ArgumentError krylov_liouvillian_spectrum(Lm;nev=0)

    # The small GMRES triangular solve uses caller-owned scratch.
    R=triu(randn(MersenneTwister(20),ComplexF64,8,8));rhs=randn(MersenneTwister(21),ComplexF64,8)
    ysmall=zeros(ComplexF64,8)
    PermutationalInvariantDynamics._upper_triangular_solve!(ysmall,R,rhs,8)
    @test R*ysmall≈rhs atol=2e-13
    @test (@allocated PermutationalInvariantDynamics._upper_triangular_solve!(ysmall,R,rhs,8))==0

    # Iterative storage follows the model precision.  A complete Float32
    # problem stays Float32 (halving the dominant basis matrices), while a
    # wider initial vector or non-integer target promotes the solve rather
    # than being silently assigned into narrow scratch.
    sm32=ComplexF32.(sm)
    model32=PIModel(b,[LocalJump(sm32;rate=0.7f0),
                       CollectiveJump(sm32;rate=0.2f0)])
    L32=liouvillian(model32;representation=:matrixfree)
    kws32=KrylovWorkspace(L32,10)
    aws32=ArnoldiWorkspace(L32,length(b))
    jdws32=JacobiDavidsonWorkspace(L32,10,8)
    @test eltype(kws32.V)===ComplexF32
    @test eltype(aws32.V)===ComplexF32
    @test eltype(jdws32.arnoldi.V)===ComplexF32
    aws64=ArnoldiWorkspace(ComplexF64,length(b),length(b))
    @test 5*Base.summarysize(aws32)<3*Base.summarysize(aws64)
    p32=schur_sector_preconditioner(L32,b;expected_reuses=10,
                                    warn_unamortized=false)
    @test eltype(p32)===ComplexF32
    ks32=krylov_steady_state(L32;basis=b,workspace=kws32,krylovdim=10,
        maxiter=200,atol=2f-6,rtol=2f-5,return_info=true)
    @test eltype(ks32.state)===ComplexF32
    @test ks32.residual isa Float32
    @test ks32.operator_scale isa Float32
    @test ks32.state≈ComplexF32.(reference.state) atol=2f-5
    seed32=ComplexF32.(advanced_seed)
    arn32=krylov_liouvillian_spectrum(L32;nev=4,krylovdim=length(b),
        initial_vector=seed32,workspace=aws32,atol=2f-5,rtol=2f-4)
    @test eltype(arn32.values)===ComplexF32
    @test eltype(arn32.residuals)===Float32
    @test arn32.residual_tolerance isa Float32
    diagonal32=Diagonal(ComplexF32[0,-0.2,-0.5,-1])
    diagonal_seed32=ones(ComplexF32,4)
    iram32=implicitly_restarted_arnoldi_spectrum(diagonal32;nev=2,
        krylovdim=4,initial_vector=diagonal_seed32,atol=1f-5,rtol=1f-4)
    harmonic32=harmonic_arnoldi_spectrum(diagonal32;nev=2,krylovdim=4,
        initial_vector=diagonal_seed32,atol=1f-5,rtol=1f-4)
    zerojd32=jacobi_davidson_spectrum(zeros(ComplexF32,4,4);nev=2)
    for result32 in (iram32,harmonic32,zerojd32)
        @test eltype(result32.values)===ComplexF32
        @test eltype(result32.residuals)===Float32
        @test result32.residual_tolerance isa Float32
    end
    wide_jd=jacobi_davidson_spectrum(diagonal32;nev=1,target=0,
        subspace_dim=4,correction_krylovdim=4,maxiter=40,
        initial_vector=diagonal_seed32,
        workspace=JacobiDavidsonWorkspace(ComplexF64,4,4,4),
        atol=1f-5,rtol=1f-4)
    @test eltype(wide_jd.values)===ComplexF64
    @test eltype(wide_jd.residuals)===Float64
    @test wide_jd.target isa ComplexF64
    mixed=krylov_liouvillian_spectrum(diagonal32;nev=2,krylovdim=4,
        initial_vector=ComplexF64.(diagonal_seed32),require_convergence=false)
    @test eltype(mixed.values)===ComplexF64
    @test PermutationalInvariantDynamics._promote_krylov_scalar_type(
        ComplexF32,16_777_217)===ComplexF64
    @test PermutationalInvariantDynamics._promote_krylov_scalar_type(
        ComplexF32,typemax(Int))===Complex{BigFloat}
    @test_throws ArgumentError krylov_liouvillian_spectrum(
        diagonal32;nev=2,krylovdim=4,
        initial_vector=Real[1,1,1,1],require_convergence=false)
    # Compiled PI matvec scratch has the model's fixed scalar type.  Wider
    # iterative vectors must be rejected instead of being copied into that
    # narrower block scratch on every application.
    @test_throws ArgumentError krylov_liouvillian_spectrum(L32;nev=2,
        krylovdim=length(b),initial_vector=ComplexF64.(seed32),
        require_convergence=false)
    @test_throws ArgumentError krylov_liouvillian_spectrum(L32;nev=2,
        krylovdim=length(b),initial_vector=seed32,
        workspace=ArnoldiWorkspace(ComplexF64,length(b),length(b)),
        require_convergence=false)
    @test_throws ArgumentError krylov_steady_state(L32;basis=b,
        initial_state=ComplexF64.(ks32.state),krylovdim=10,
        maxiter=10,atol=2f-6,rtol=2f-5)
    target64=implicitly_restarted_arnoldi_spectrum(diagonal32;nev=2,
        krylovdim=4,target=0.0,initial_vector=diagonal_seed32,
        require_convergence=false)
    @test eltype(target64.values)===ComplexF64
    @test_throws ArgumentError implicitly_restarted_arnoldi_spectrum(
        diagonal32;nev=2,krylovdim=4,target=0.0,
        initial_vector=diagonal_seed32,workspace=ArnoldiWorkspace(ComplexF32,4,4),
        require_convergence=false)
    @test_throws ArgumentError implicitly_restarted_arnoldi_spectrum(
        diagonal32;nev=2,krylovdim=4,target=NaN)
    @test_throws ArgumentError harmonic_arnoldi_spectrum(
        diagonal32;nev=2,krylovdim=4,target=Inf)
    @test_throws ArgumentError jacobi_davidson_spectrum(
        diagonal32;nev=1,target=complex(0,Inf),subspace_dim=4)
    @test_throws ArgumentError krylov_liouvillian_spectrum(
        Diagonal(Complex{BigFloat}[0,-1,-2]);nev=2,krylovdim=3,
        initial_vector=ones(Complex{BigFloat},3),require_convergence=false)

    # Reusable Arnoldi storage preserves results and removes the large basis
    # and pencil allocations on repeated calls.
    seed=randn(MersenneTwister(22),ComplexF64,length(b))
    aws=PermutationalInvariantDynamics.ArnoldiWorkspace(Lm,length(b))
    awarm=krylov_liouvillian_spectrum(Lm;nev=4,krylovdim=length(b),
        initial_vector=seed,workspace=aws)
    araw=krylov_liouvillian_spectrum(Lm;nev=4,krylovdim=length(b),initial_vector=seed)
    @test awarm.values≈araw.values atol=2e-11
    @test awarm.workspace_reused
    alloc_arnoldi=@allocated krylov_liouvillian_spectrum(Lm;nev=4,
        krylovdim=length(b),initial_vector=seed,workspace=aws)
    alloc_arnoldi_raw=@allocated krylov_liouvillian_spectrum(Lm;nev=4,
        krylovdim=length(b),initial_vector=seed)
    @test alloc_arnoldi<3*alloc_arnoldi_raw÷4

    hws=PermutationalInvariantDynamics.ArnoldiWorkspace(Lm,length(b))
    hwarm=harmonic_arnoldi_spectrum(Lm;nev=3,krylovdim=length(b),
        initial_vector=seed,workspace=hws,rng=MersenneTwister(23),
        atol=1e-9,rtol=1e-7)
    hraw=harmonic_arnoldi_spectrum(Lm;nev=3,krylovdim=length(b),
        initial_vector=seed,rng=MersenneTwister(23),atol=1e-9,rtol=1e-7)
    @test hwarm.values≈hraw.values atol=3e-7
    @test hwarm.workspace_reused
    @test length(hwarm.restart_history)==hwarm.restarts+1
    # Separate identically seeded RNGs make any Krylov-breakdown fallback
    # reproducible without including RNG construction in the allocation gate.
    hrng_workspace=MersenneTwister(24);hrng_raw=MersenneTwister(24)
    alloc_harmonic=@allocated harmonic_arnoldi_spectrum(Lm;nev=3,
        krylovdim=length(b),initial_vector=seed,workspace=hws,
        rng=hrng_workspace,atol=1e-9,rtol=1e-7)
    alloc_harmonic_raw=@allocated harmonic_arnoldi_spectrum(Lm;nev=3,
        krylovdim=length(b),initial_vector=seed,rng=hrng_raw,
        atol=1e-9,rtol=1e-7)
    @test alloc_harmonic<3*alloc_harmonic_raw÷4

    # Normalizing the trace-fixed system by an operator-scale estimate keeps
    # convergence invariant under a global rate rescaling.
    Ls=liouvillian(model;representation=:sparse)
    for c in (1e-8,1e8)
        sws=KrylovWorkspace(ComplexF64,length(b),10)
        scaled=krylov_steady_state(c*Ls;basis=b,workspace=sws,krylovdim=10,
            maxiter=200,atol=1e-11,rtol=1e-9,return_info=true)
        @test scaled.state≈reference.state atol=3e-9
        @test scaled.operator_scale≈c*opnorm(Ls,Inf) rtol=2e-14
        @test scaled.normalized_residual<1e-9
    end

    # Spectral residual scales no longer use an absolute floor of one.
    for c in (1e-8,1e8)
        sws=PermutationalInvariantDynamics.ArnoldiWorkspace(c*Ls,length(b))
        scaled=krylov_liouvillian_spectrum(c*Ls;nev=4,krylovdim=length(b),
            initial_vector=seed,workspace=sws,atol=0,rtol=1e-8)
        @test scaled.values./c≈dense[1:4] atol=3e-9
        @test scaled.residual_scale/c>0
        hscaled=harmonic_arnoldi_spectrum(c*Ls;nev=3,krylovdim=length(b),
            initial_vector=seed,workspace=sws,atol=0,rtol=1e-7)
        @test maximum(hscaled.residuals)/c<5e-7
        @test abs(hscaled.harmonic_shift/c)<1e-6
    end
end
