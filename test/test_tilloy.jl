using Test
using LinearAlgebra
using SparseArrays
using Random
using PermutationalInvariantDynamics

struct UnsupportedTilloyKernel <:
       PermutationalInvariantDynamics.AbstractStaticPIKernel end

function tilloy_nojump_matrix(plan)
    n=length(plan.basis)
    matrix=zeros(plan.Ttype,n,n)
    for (sector,G) in pairs(plan.generator_blocks)
        range=plan.basis.offsets[sector]:plan.basis.offsets[sector+1]-1
        matrix[range,range].=
            left_superoperator(G)+right_superoperator(adjoint(G))
    end
    matrix
end

@testset "Tilloy no-jump iterative methods" begin
    basis=PIBasis(3,2)
    spin=spin_matrices(2)
    model=PIModel(basis,(
        LocalHamiltonian(spin.jx;rate=0.31),
        LocalJump(spin.jm;rate=0.7),
        LocalJump(spin.jp;rate=0.2),
        CollectiveJump(spin.jm;rate=0.08),
    ))
    L=Matrix(liouvillian(model;representation=:sparse,memory_budget=Inf))
    rng=MersenneTwister(0x71110)
    source=randn(rng,ComplexF64,length(basis))

    for backend in (:schur,:eigen)
        plan=TilloyPlan(model;backend,memory_budget=Inf)
        @test plan.metadata.backend===backend
        @test plan.metadata.strictly_stable
        @test plan.metadata.jump_channels==3
        @test plan.metadata.unique_steady_state===:assumed_not_certified
        @test plan.no_jump.metadata.unique_steady_state===:not_applicable
        @test plan.metadata.generator_mode===:autonomous
        @test plan.metadata.deflation_normalization===:trace_one_identity
        @test plan.metadata.fixed_point_action===:direct_gain
        @test plan.metadata.zero_shift_fixed_point_available
        @test :cptp_fixed_point in plan.metadata.guarantees
        @test size(plan)==(length(basis),length(basis))
        @test eltype(plan)===ComplexF64
        work=NoJumpResolventWorkspace(plan.no_jump;memory_budget=Inf)
        S=tilloy_nojump_matrix(plan.no_jump)
        for shift in (0.0,0.37)
            result=zeros(ComplexF64,length(basis))
            no_jump_resolvent!(result,plan.no_jump,source,shift,work)
            @test norm((shift*I-S)*result-source)<2e-11
            allocating=no_jump_resolvent(plan.no_jump,source;
                shift,workspace=work)
            @test allocating≈result atol=2e-12 rtol=2e-12
        end
        if backend===:schur
            allocation_destination=similar(source)
            no_jump_resolvent!(allocation_destination,plan.no_jump,source,
                0.37,work)
            # The prepared numerical buffers are reused; only small view
            # wrappers may remain on supported Julia releases. Guard against
            # accidentally allocating sector matrices or coordinate vectors.
            @test @allocated(no_jump_resolvent!(allocation_destination,
                plan.no_jump,source,0.37,work))<=4096
        end
        @test_throws ArgumentError no_jump_resolvent!(source,
            plan.no_jump,source,0.2,work)
        @test_throws ArgumentError no_jump_resolvent(
            plan.no_jump,source;shift=-0.1,workspace=work)
    end

    plan=TilloyPlan(model;memory_budget=Inf)
    @test isautonomous(plan)
    @test isautonomous(plan.no_jump)
    @test plan.metadata.trace_preserving
    @test plan.metadata.trace_certification===:matrix_free_adjoint
    @test plan.metadata.trace_preservation_residual<=
        plan.metadata.trace_preservation_tolerance
    @test NoJumpResolventPlan(plan) === plan.no_jump
    @test_throws ArgumentError NoJumpResolventPlan(plan;backend=:eigen)
    work=TilloyWorkspace(plan;krylovdim=18,recycle_dim=4,
                         memory_budget=Inf)
    S=tilloy_nojump_matrix(plan.no_jump)
    tracevec=collect(plan.tracevec)
    rho=maximally_mixed_state(basis)

    # Prepared-source constructors and allocating source wrappers must retain
    # the same generator and solve, independently of the compiled backend.
    compiled=compile(model;backend=:matrixfree,memory_budget=Inf)
    compiled_plan=TilloyPlan(compiled;memory_budget=Inf)
    compiled_no_jump=NoJumpResolventPlan(compiled;memory_budget=Inf)
    @test tilloy_nojump_matrix(compiled_plan.no_jump)≈S atol=2e-13 rtol=2e-13
    @test tilloy_nojump_matrix(compiled_no_jump)≈S atol=2e-13 rtol=2e-13

    # A scalar-rate specialization keeps the family's prepared geometry but
    # binds its own rates in both the no-jump split and full-L actions.
    family=compile_family(model;rate_indices=(2,3,4))
    specialized=specialize(
        family,(0.9,0.11,0.035);backend=:matrixfree,memory_budget=Inf)
    specialized_plan=TilloyPlan(specialized;memory_budget=Inf)
    independently_prepared=TilloyPlan(
        specialized.model;memory_budget=Inf)
    @test specialized_plan.metadata.source===:specialized_pi_liouvillian
    @test specialized_plan.metadata.bound_rates===specialized.rates
    @test specialized_plan.metadata.trace_preserving
    @test tilloy_nojump_matrix(specialized_plan.no_jump)≈
        tilloy_nojump_matrix(independently_prepared.no_jump) atol=2e-13 rtol=2e-13
    specialized_state=tilloy_steady_state(
        specialized_plan;method=:gmres,krylovdim=18,recycle_dim=4,
        atol=1e-10,rtol=1e-8,memory_budget=Inf)
    specialized_direct=steady_state(
        specialized;method=:direct,memory_budget=Inf)
    @test specialized_state.data≈specialized_direct atol=4e-10 rtol=4e-9
    specialized_wrapped=tilloy_steady_state(specialized;method=:gmres,
        krylovdim=18,recycle_dim=4,atol=1e-10,rtol=1e-8,
        memory_budget=Inf)
    @test specialized_wrapped.data≈specialized_state.data atol=4e-10 rtol=4e-9
    specialized_work=TilloyWorkspace(specialized_plan;krylovdim=8,
        recycle_dim=2,memory_budget=Inf)
    specialized_r0=no_jump_resolvent(specialized_plan.no_jump,source;
        shift=0,workspace=specialized_work.no_jump)
    specialized_gain=zeros(ComplexF64,length(basis))
    PermutationalInvariantDynamics._apply_tilloy_gain!(specialized_gain,
        specialized_plan,specialized_r0,specialized_work.liouvillian)
    specialized_L=Matrix(liouvillian(specialized;
        representation=:sparse,memory_budget=Inf))
    @test specialized_gain≈source+specialized_L*specialized_r0 atol=3e-11 rtol=3e-11

    # The positive-shift no-jump resolvent is CP and K*R_lambda is a strict
    # trace contraction on a trace-one positive input.
    z=no_jump_resolvent(plan.no_jump,rho;shift=0.4,
                        workspace=work.no_jump)
    @test ispositive(z;atol=2e-11,rtol=2e-10)
    K=L-S
    recycled_trace=real(dot(tracevec,K*z.data))
    @test -2e-11<=recycled_trace<1-1e-8

    # Phi=I+L*R_0 is trace preserving, including local gains that connect
    # different Schur sectors.
    r0=no_jump_resolvent(
        plan.no_jump,rho.data;shift=0,workspace=work.no_jump)
    phi=rho.data+L*r0
    @test dot(tracevec,phi)≈1 atol=3e-11
    direct_phi=zeros(ComplexF64,length(basis))
    PermutationalInvariantDynamics._apply_tilloy_gain!(
        direct_phi,plan,r0,work.liouvillian)
    @test direct_phi≈phi atol=3e-11 rtol=3e-11
    @test direct_phi≈K*r0 atol=3e-11 rtol=3e-11

    # The unfused lowering exercises channel-resolved collective and local
    # gain kernels rather than the fused static aggregate.
    unfused=TilloyPlan(LiouvillianPlan(model;fuse_static=false);
        memory_budget=Inf)
    unfused_work=TilloyWorkspace(unfused;krylovdim=8,recycle_dim=2,
        memory_budget=Inf)
    unfused_r0=no_jump_resolvent(unfused.no_jump,source;shift=0,
        workspace=unfused_work.no_jump)
    unfused_gain=zeros(ComplexF64,length(basis))
    PermutationalInvariantDynamics._apply_tilloy_gain!(unfused_gain,
        unfused,unfused_r0,unfused_work.liouvillian)
    @test unfused_gain≈source+L*unfused_r0 atol=3e-11 rtol=3e-11

    # Fixed correlated local and collective channels lower to ordinary
    # channel kernels. Their combined gain must obey K*R0=I+L*R0 too.
    gamma_matrix=ComplexF64[0.8 0.1im;-0.1im 0.4]
    correlated_model=PIModel(basis,(
        LocalJump(spin.jm;rate=0.4),LocalJump(spin.jp;rate=0.2),
        CorrelatedLocalJumps((spin.jm,spin.jz),gamma_matrix;rate=0.17),
        CorrelatedCollectiveJumps((spin.jm,spin.jz),gamma_matrix;rate=0.03),))
    correlated=TilloyPlan(
        LiouvillianPlan(correlated_model;fuse_static=false);
        memory_budget=Inf)
    correlated_work=TilloyWorkspace(correlated;krylovdim=8,recycle_dim=2,
        memory_budget=Inf)
    correlated_r0=no_jump_resolvent(correlated.no_jump,source;shift=0,
        workspace=correlated_work.no_jump,memory_budget=Inf)
    correlated_gain=similar(source)
    PermutationalInvariantDynamics._apply_tilloy_gain!(correlated_gain,
        correlated,correlated_r0,correlated_work.liouvillian)
    correlated_L=Matrix(liouvillian(correlated_model;
        representation=:sparse,memory_budget=Inf))
    @test correlated_gain≈source+correlated_L*correlated_r0 atol=5e-11 rtol=5e-11

    direct=steady_state(model;method=:direct,memory_budget=Inf)
    gmres=tilloy_steady_state(plan;method=:gmres,workspace=work,
        return_info=true,atol=1e-10,rtol=1e-8,memory_budget=Inf)
    @test gmres.converged
    @test gmres.method===:tilloy_gmres
    @test gmres.state.data≈direct atol=3e-10 rtol=3e-9
    @test gmres.physical_residual_inf<2e-10
    @test gmres.trace_error<2e-10
    @test gmres.state_diagnostics.valid
    @test gmres.linear_solver.right_preconditioned
    @test gmres.linear_solver.solution===gmres.state.data
    @test gmres.linear_solver.recycle_extraction===:harmonic
    @test gmres.linear_solver.recycle_extraction_used in
        (:harmonic,:rayleigh_ritz,:none)

    compiled_state=tilloy_steady_state(compiled;method=:gmres,
        krylovdim=18,recycle_dim=4,atol=1e-10,rtol=1e-8,
        memory_budget=Inf)
    @test compiled_state.data≈gmres.state.data atol=3e-10 rtol=3e-9

    fixed=tilloy_steady_state(plan;method=:fixed_point,krylovdim=18,
        maxrestarts=10,return_info=true,atol=1e-9,rtol=1e-7,
        memory_budget=Inf,rng=MersenneTwister(11))
    @test fixed.converged
    @test fixed.method===:tilloy_fixed_point
    @test fixed.fixed_point_value≈1 atol=2e-8 rtol=2e-8
    @test fixed.state.data≈direct atol=2e-8 rtol=2e-7
    @test fixed.physical_residual_inf<2e-8

    # A no-jump map can have a low-dimensional invariant Krylov space from
    # the identity seed. Block Arnoldi must inject orthogonal complements at
    # exact breakdown rather than normalize and repeat roundoff directions.
    driven_qubit_model=PIModel(PIBasis(2,2),(
        LocalHamiltonian(0.70spin.jx+0.23spin.jz),
        LocalJump(spin.jm;rate=0.31),LocalJump(spin.jp;rate=0.09)))
    driven_fixed=tilloy_steady_state(driven_qubit_model;
        method=:fixed_point,krylovdim=60,maxrestarts=40,return_info=true,
        atol=1e-9,rtol=1e-7,memory_budget=Inf,rng=MersenneTwister(1))
    @test driven_fixed.fixed_point_value≈1 atol=2e-10 rtol=2e-10
    @test driven_fixed.physical_residual_inf<2e-10

    # The absolute coordinate tolerance must shrink with the physical trace
    # functional. Otherwise a residual near 1e-12 can still lose the requested
    # trace accuracy once ||tracevec||=sqrt(2^N) is appreciable.
    larger_basis=PIBasis(16,2)
    larger_model=PIModel(larger_basis,(
        LocalJump(spin.jm;rate=0.7),
        LocalJump(spin.jp;rate=0.2),))
    larger_plan=TilloyPlan(larger_model;memory_budget=Inf)
    larger=tilloy_steady_state(larger_plan;method=:gmres,
        krylovdim=20,recycle_dim=4,maxiter=500,
        atol=1e-12,rtol=1e-10,return_info=true,memory_budget=Inf)
    @test larger.trace_control.trace_norm≈sqrt(2.0^16)
    @test larger.trace_control.inner_atol<1e-12
    @test larger.trace_error<=larger.state_diagnostics.trace_tolerance

    # Rank-one Sherman--Morrison preconditioning is tested against the exact
    # small PI-coordinate deflated linear system.
    shift=0.23;deflation=0.8
    rhs=randn(rng,ComplexF64,length(basis))
    result=tilloy_resolvent(plan,rhs;shift,deflation,
        krylovdim=18,recycle_dim=4,maxiter=300,atol=1e-11,rtol=1e-9,
        memory_budget=Inf)
    A=shift*I-L-deflation*plan.deflation_vector*adjoint(tracevec)
    @test result.solution≈A\rhs atol=2e-9 rtol=2e-8
    @test norm(A*result.solution-rhs,Inf)≈result.residual_inf atol=2e-12

    # Public allocating routes compose their returned vector/state and
    # validation scratch with the already prepared workspace peak. A budget
    # one byte below the structural estimate must fail before allocating the
    # requested output, while the exact estimate remains usable.
    coordinate_bytes=PermutationalInvariantDynamics._performance_entries_bytes(
        BigInt(length(basis)),ComplexF64)
    nojump_work=NoJumpResolventWorkspace(plan.no_jump;memory_budget=Inf)
    nojump_budget=nojump_work.accounted_peak_bytes+2coordinate_bytes
    @test_throws ArgumentError no_jump_resolvent(plan.no_jump,source;
        shift=0.23,workspace=nojump_work,memory_budget=nojump_budget-1)
    bounded_nojump=no_jump_resolvent(plan.no_jump,source;
        shift=0.23,workspace=nojump_work,memory_budget=nojump_budget)
    @test norm((0.23I-S)*bounded_nojump-source)<2e-11

    resolvent_budget=work.accounted_peak_bytes+2coordinate_bytes
    @test_throws ArgumentError tilloy_resolvent(plan,rhs;shift,deflation,
        workspace=work,memory_budget=resolvent_budget-1)
    bounded_resolvent=tilloy_resolvent(plan,rhs;shift,deflation,
        workspace=work,maxiter=300,atol=1e-11,rtol=1e-9,
        memory_budget=resolvent_budget)
    @test bounded_resolvent.solution≈result.solution atol=2e-9 rtol=2e-8

    stationary_budget=work.accounted_peak_bytes+
        PermutationalInvariantDynamics._tilloy_stationary_output_bytes(plan)
    @test_throws ArgumentError tilloy_steady_state(plan;method=:gmres,
        workspace=work,memory_budget=stationary_budget-1)
    bounded_stationary=tilloy_steady_state(plan;method=:gmres,
        workspace=work,atol=1e-10,rtol=1e-8,
        memory_budget=stationary_budget)
    @test bounded_stationary.data≈direct atol=3e-10 rtol=3e-9

    # Slow modes are mapped back and certified against the original L.
    dense_values=eigvals(L)
    nonzero=sort(filter(value->abs(value)>1e-8,dense_values);by=abs)
    spectrum=tilloy_liouvillian_spectrum(plan;nev=2,krylovdim=18,
        maxrestarts=8,inner_krylovdim=18,inner_recycle_dim=4,
        inner_maxiter=300,inner_atol=1e-11,inner_rtol=1e-9,
        atol=1e-8,rtol=1e-6,vectors=true,memory_budget=Inf,
        rng=MersenneTwister(12))
    @test spectrum.converged
    @test spectrum.method===:tilloy_shift_invert
    # The second slow-mode shell is a complex-conjugate pair with equal
    # modulus.  LAPACK ordering differs across supported Julia releases, so
    # accepting either member is the mathematically invariant assertion.
    spectral_atol=2e-7
    spectral_rtol=2e-6
    cutoff=abs(nonzero[2])
    eligible=findall(value->abs(value)<=
        cutoff+spectral_atol+spectral_rtol*cutoff,dense_values)
    remaining=collect(eligible)
    for value in spectrum.values
        distances=abs.(dense_values[remaining].-value)
        position=argmin(distances)
        reference=dense_values[remaining[position]]
        @test value≈reference atol=spectral_atol rtol=spectral_rtol
        deleteat!(remaining,position)
    end
    @test maximum(spectrum.physical_residuals)<2e-8
    @test maximum(spectrum.trace_errors)<2e-8
    @test all(abs.(spectrum.values).>spectrum.zero_exclusion_tolerance)
    @test spectrum.candidate_count==4
    @test spectrum.candidate_oversampling==2
    @test norm(L*spectrum.vectors-
        spectrum.vectors*Diagonal(spectrum.values))<2e-7
    @test_throws ArgumentError tilloy_liouvillian_spectrum(plan;
        nev=2,krylovdim=18,candidate_oversampling=-1,memory_budget=Inf)
    @test_throws ArgumentError tilloy_liouvillian_spectrum(plan;
        nev=2,krylovdim=3,candidate_oversampling=4,memory_budget=Inf)

    # One implicit-Euler step and the saved-grid wrapper match a direct small
    # solve, without trace normalization in the implementation.
    dt=0.07
    destination=PIState(basis;T=Float64)
    step=tilloy_implicit_euler_step!(destination,plan,rho,dt,work;
        reuse=false,atol=1e-11,rtol=1e-9,memory_budget=Inf)
    @test step.converged
    @test destination.data≈(I-dt*L)\rho.data atol=2e-9 rtol=2e-8
    @test trace(destination)≈1 atol=2e-9
    evolution=tilloy_implicit_euler(plan,rho,[0.0,dt,2dt];
        krylovdim=18,recycle_dim=4,atol=1e-9,rtol=1e-7,
        memory_budget=Inf)
    @test evolution.converged
    @test length(evolution.states)==3
    @test evolution.states[2].data≈destination.data atol=3e-8 rtol=3e-7
    @test evolution.diagnostics[1].solution===evolution.states[2].data
    @test evolution.diagnostics[2].solution===evolution.states[3].data
    @test evolution.generator_mode===:autonomous
    @test evolution.unique_steady_state===:not_applicable
    @test_throws ArgumentError tilloy_implicit_euler(plan,rho,[0.0,dt];
        workspace=work,memory_budget=1)
    @test_throws ArgumentError tilloy_implicit_euler(
        plan,rho,[0.0,0.1,0.1];memory_budget=Inf)

    # The zero-shift singularity is the theorem's dark-state branch; positive
    # shifts still admit an exact no-jump resolvent.
    dark_basis=PIBasis(1,2)
    dark_model=PIModel(dark_basis,(LocalJump(spin.jm;rate=1.0),))
    dark=TilloyPlan(dark_model;memory_budget=Inf)
    @test !dark.metadata.strictly_stable
    @test !dark.metadata.zero_shift_fixed_point_available
    @test :cptp_fixed_point ∉ dark.metadata.guarantees
    @test :positive_shift_contraction in dark.metadata.guarantees
    dark_source=randn(rng,ComplexF64,length(dark_basis))
    @test_throws ArgumentError no_jump_resolvent(
        dark.no_jump,dark_source;shift=0)
    dark_positive=no_jump_resolvent(dark.no_jump,dark_source;shift=0.5)
    dark_S=tilloy_nojump_matrix(dark.no_jump)
    @test norm((0.5I-dark_S)*dark_positive-dark_source)<2e-11
    # A positive shift can be representable yet too small for its inverse to
    # fit the prepared scalar type. Both sector backends must report that
    # overflow instead of returning Inf/NaN coefficients.
    dark_density=maximally_mixed_state(dark_basis).data
    for backend in (:schur,:eigen)
        tiny_plan=NoJumpResolventPlan(
            dark_model;backend,memory_budget=Inf)
        @test_throws ArgumentError no_jump_resolvent(
            tiny_plan,dark_density;shift=nextfloat(0.0))
    end
    # The manual triangular recurrence must catch overflow at the product or
    # accumulation where it occurs, rather than allowing Inf/NaN to propagate
    # to a superficially completed sector solve.
    overflow_triangular=ComplexF64[-1 floatmax(Float64);0 -1]
    overflow_rhs=zeros(ComplexF64,2,2)
    overflow_rhs[2,1]=4
    @test_throws ArgumentError begin
        PermutationalInvariantDynamics._solve_nojump_sector!(
            overflow_rhs,overflow_triangular,0.0)
    end
    @test_throws ArgumentError tilloy_steady_state(
        dark;method=:gmres,memory_budget=Inf)

    # A dissipative but defective effective generator exercises the robust
    # Schur recurrence. The optional eigen backend must reject rather than
    # silently invert a singular eigenvector matrix.
    defective_basis=PIBasis(1,2)
    G=ComplexF64[-1 1;0 -1]
    Q=-(G+G')
    H=(1im/2)*(G-G')
    jump=cholesky(Hermitian(Q)).U
    defective_model=PIModel(defective_basis,(
        LocalHamiltonian(H),LocalJump(jump),))
    robust=TilloyPlan(defective_model;backend=:schur,memory_budget=Inf)
    robust_source=randn(rng,ComplexF64,length(defective_basis))
    robust_result=no_jump_resolvent(
        robust.no_jump,robust_source;shift=0)
    robust_S=tilloy_nojump_matrix(robust.no_jump)
    @test norm(-robust_S*robust_result-robust_source)<3e-11
    @test_throws ArgumentError TilloyPlan(defective_model;
        backend=:eigen,memory_budget=Inf)

    # Float32 remains Float32, and incompatible workspaces/budgets/rates are
    # rejected before a solve or hidden allocation.
    spin32=spin_matrices(2;T=Float32)
    model32=PIModel(PIBasis(1,2),(
        LocalJump(spin32.jm;rate=Float32(0.7)),
        LocalJump(spin32.jp;rate=Float32(0.2)),))
    plan32=TilloyPlan(model32;memory_budget=Inf)
    @test eltype(plan32)===ComplexF32
    value32=no_jump_resolvent(plan32.no_jump,
        ComplexF32[1,0,0,0];shift=Float32(0.4))
    @test eltype(value32)===ComplexF32
    @test_throws ArgumentError tilloy_resolvent(
        plan32,ComplexF64[1,0,0,0];shift=Float32(0.4),memory_budget=Inf)
    @test_throws ArgumentError tilloy_resolvent(
        plan32,ComplexF32[1,0,0,0];shift=Float32(0.4),atol=Inf,
        memory_budget=Inf)
    @test_throws ArgumentError tilloy_resolvent(
        plan32,ComplexF32[1,0,0,0];shift=Float32(0.4),
        atol=nextfloat(0.0),rtol=Float32(1e-4),memory_budget=Inf)
    @test_throws ArgumentError tilloy_steady_state(plan32;
        method=:fixed_point,krylovdim=4,maxrestarts=1,
        atol=nextfloat(0.0),rtol=Float32(1e-4),memory_budget=Inf)
    @test_throws ArgumentError TilloyPlan(model32;stability_atol=Inf,
        memory_budget=Inf)
    # BigFloat cannot be accepted opportunistically: the retained general
    # complex factorization would otherwise inherit an ambient precision and
    # rounding mode that neither plan nor workspace owns.
    big_liouvillian=setprecision(BigFloat,128) do
        spin_big=spin_matrices(2;T=BigFloat)
        model_big=PIModel(PIBasis(1,2),(
            LocalJump(spin_big.jm;rate=BigFloat("0.7")),
            LocalJump(spin_big.jp;rate=BigFloat("0.2")),))
        LiouvillianPlan(model_big)
    end
    for (bits,mode) in ((64,RoundDown),(256,RoundUp))
        setrounding(BigFloat,mode) do
            setprecision(BigFloat,bits) do
                error=try
                    NoJumpResolventPlan(big_liouvillian;memory_budget=Inf)
                    nothing
                catch caught
                    caught
                end
                @test error isa ArgumentError
                @test occursin("do not currently support BigFloat",
                    sprint(showerror,error))
                @test_throws ArgumentError TilloyPlan(
                    big_liouvillian;memory_budget=Inf)
            end
        end
    end
    @test_throws ArgumentError tilloy_implicit_euler(
        plan32,PIState(plan32.basis;T=Float64),Float32[0,0.1];
        memory_budget=Inf)
    work32=TilloyWorkspace(plan32;krylovdim=4,recycle_dim=1,
        memory_budget=Inf)
    fill!(work32.rhs,ComplexF32(3,1))
    saved_rhs32=copy(work32.rhs)
    @test_throws ArgumentError tilloy_implicit_euler_step!(
        PIState(plan32.basis;T=Float32),plan32,
        maximally_mixed_state(plan32.basis;T=Float32),nextfloat(0.0f0),
        work32;memory_budget=Inf)
    @test work32.rhs==saved_rhs32
    @test_throws ArgumentError TilloyPlan(model;memory_budget=1)
    @test_throws ArgumentError TilloyWorkspace(plan;memory_budget=1)
    @test_throws ArgumentError TilloyWorkspace(plan;krylovdim=true,
        memory_budget=Inf)
    @test_throws ArgumentError TilloyWorkspace(plan;recycle_dim=false,
        memory_budget=Inf)
    unsupported_plan=LiouvillianPlan(basis,(UnsupportedTilloyKernel(),),
        plan.tracevec,nothing,ComplexF64,true)
    unsupported_error=try
        NoJumpResolventPlan(unsupported_plan;memory_budget=Inf)
        nothing
    catch error
        error
    end
    @test unsupported_error isa ArgumentError
    @test occursin("does not expose the fixed GKSL jump/no-jump split",
        sprint(showerror,unsupported_error))
    other=TilloyPlan(PIModel(PIBasis(1,2),(
        LocalJump(spin.jm;rate=0.5),LocalJump(spin.jp;rate=0.1)));
        memory_budget=Inf)
    other_work=TilloyWorkspace(other;memory_budget=Inf)
    @test_throws ArgumentError tilloy_steady_state(
        plan;workspace=other_work,memory_budget=Inf)
    other_rhs=copy(other_work.rhs)
    @test_throws ArgumentError tilloy_implicit_euler_step!(
        PIState(basis;T=Float64),plan,rho,0.1,other_work;
        memory_budget=Inf)
    @test other_work.rhs==other_rhs
    @test_throws ArgumentError TilloyPlan(PIModel(basis,(
        LocalJump(spin.jm;rate=-0.1),));memory_budget=Inf)
    # Bypassing a term constructor's Hermiticity check must not let a
    # non-GKSL Hamiltonian acquire the Tilloy CPTP/contraction guarantees.
    bad_hamiltonian=ComplexF64[0 1;0 0]
    bad_hamiltonian_model=PIModel(basis,(
        LocalHamiltonian(bad_hamiltonian;check=false),
        LocalJump(spin.jm;rate=0.7),LocalJump(spin.jp;rate=0.2)))
    @test_throws ArgumentError NoJumpResolventPlan(
        bad_hamiltonian_model;memory_budget=Inf)
    @test_throws ArgumentError TilloyPlan(
        bad_hamiltonian_model;memory_budget=Inf)

    # Certification is applied to the total physical Hamiltonian. Two
    # unchecked pieces whose sum is Hermitian remain a valid advanced input,
    # including through the diagnostic unfused-kernel route.
    cancelling_model=PIModel(basis,(
        LocalHamiltonian(bad_hamiltonian;check=false),
        LocalHamiltonian(adjoint(bad_hamiltonian);check=false),
        LocalJump(spin.jm;rate=0.7),LocalJump(spin.jp;rate=0.2)))
    cancelling_plan=TilloyPlan(
        LiouvillianPlan(cancelling_model;fuse_static=false);
        memory_budget=Inf)
    @test cancelling_plan.metadata.trace_preserving

    # A driven PIModel has a native, explicit instantaneous route that first
    # freezes every physical term, preserving the jump/no-jump split.
    driven=PIModel(basis,(
        LocalHamiltonian(spin.jx;
            rate=(time,parameters)->time*parameters.drive),
        LocalJump(spin.jm;rate=(time,parameters)->parameters.down),
        LocalJump(spin.jp;rate=0.2)))
    instant_time=0.5
    instant_parameters=(drive=0.62,down=0.7)
    instant_reference_model=PIModel(basis,(
        LocalHamiltonian(spin.jx;rate=0.31),
        LocalJump(spin.jm;rate=0.7),LocalJump(spin.jp;rate=0.2)))
    instant_reference=TilloyPlan(
        instant_reference_model;memory_budget=Inf)
    instant=TilloyPlan(driven;time=instant_time,
        parameters=instant_parameters,memory_budget=Inf)
    @test instant.metadata.instantaneous
    @test instant.metadata.instantaneous_time==instant_time
    @test instant.metadata.instantaneous_parameters_supplied
    @test instant.no_jump.metadata.instantaneous
    @test tilloy_nojump_matrix(instant.no_jump)≈
        tilloy_nojump_matrix(instant_reference.no_jump) atol=2e-13 rtol=2e-13
    instant_no_jump=NoJumpResolventPlan(driven;time=instant_time,
        parameters=instant_parameters,memory_budget=Inf)
    @test instant_no_jump.metadata.instantaneous
    @test tilloy_nojump_matrix(instant_no_jump)≈
        tilloy_nojump_matrix(instant_reference.no_jump) atol=2e-13 rtol=2e-13
    instant_state=tilloy_steady_state(driven;time=instant_time,
        parameters=instant_parameters,method=:gmres,krylovdim=18,
        recycle_dim=4,atol=1e-10,rtol=1e-8,memory_budget=Inf)
    instant_direct=steady_state(
        instant_reference_model;method=:direct,memory_budget=Inf)
    @test instant_state.data≈instant_direct atol=4e-10 rtol=4e-9
    @test_throws ArgumentError TilloyPlan(driven;memory_budget=Inf)
    @test_throws ArgumentError TilloyPlan(driven;
        parameters=instant_parameters,memory_budget=Inf)
    @test_throws ArgumentError TilloyPlan(driven;time=Inf,
        parameters=instant_parameters,memory_budget=Inf)
    @test_throws ArgumentError TilloyPlan(model;time=instant_time,
        memory_budget=Inf)
    @test_throws ArgumentError NoJumpResolventPlan(model;
        parameters=instant_parameters,memory_budget=Inf)
    @test_throws ArgumentError tilloy_steady_state(specialized;
        time=instant_time,memory_budget=Inf)

    # With a nonunique dephasing stationary manifold, a traceless zero mode
    # passes the usual trace test. The advertised nonzero spectrum must still
    # exclude it explicitly at the requested numerical tolerance.
    dephasing=TilloyPlan(PIModel(PIBasis(1,2),(
        LocalJump(spin.jz;rate=1.0),));memory_budget=Inf)
    dephasing_modes=tilloy_liouvillian_spectrum(dephasing;nev=1,shift=0.05,
        krylovdim=4,maxrestarts=4,inner_krylovdim=4,
        inner_recycle_dim=2,inner_maxiter=80,atol=1e-9,rtol=1e-7,
        require_convergence=false,memory_budget=Inf,rng=MersenneTwister(41))
    @test !isempty(dephasing_modes.values)
    @test all(abs.(dephasing_modes.values).>
        dephasing_modes.zero_exclusion_tolerance)

    # A restricted Schur basis is valid only when every retained channel
    # preserves it. Collective dynamics preserves the fully symmetric sector;
    # independent local dephasing leaks population into omitted sectors and
    # is rejected by the matrix-free adjoint trace certificate.
    symmetric_basis=PIBasis(3,2;sectors=[(3,0)])
    invariant_restriction=TilloyPlan(PIModel(symmetric_basis,(
        CollectiveHamiltonian(spin.jx;rate=0.2),
        CollectiveJump(spin.jm;rate=0.3),));memory_budget=Inf)
    @test invariant_restriction.metadata.trace_preserving
    @test invariant_restriction.metadata.trace_preservation_residual<=
        invariant_restriction.metadata.trace_preservation_tolerance
    @test_throws ArgumentError TilloyPlan(PIModel(symmetric_basis,(
        LocalJump(spin.jz;rate=0.3),));memory_budget=Inf)

    # Appendix-D local p-body losses use the same exact no-jump split.
    pbody_basis=PIBasis(2,2)
    pbody_model=PIModel(pbody_basis,(
        LocalPBodyJump(kron(spin.jm,spin.jm),2;rate=0.15),
        LocalJump(spin.jm;rate=0.6),LocalJump(spin.jp;rate=0.2),))
    pbody=TilloyPlan(pbody_model;memory_budget=Inf)
    pbody_source=randn(rng,ComplexF64,length(pbody_basis))
    pbody_result=no_jump_resolvent(pbody.no_jump,pbody_source;shift=0.2)
    pbody_S=tilloy_nojump_matrix(pbody.no_jump)
    @test norm((0.2I-pbody_S)*pbody_result-pbody_source)<3e-10
    pbody_work=TilloyWorkspace(pbody;krylovdim=8,recycle_dim=2,
        memory_budget=Inf)
    pbody_r0=no_jump_resolvent(pbody.no_jump,pbody_source;shift=0,
        workspace=pbody_work.no_jump)
    pbody_gain=zeros(ComplexF64,length(pbody_basis))
    PermutationalInvariantDynamics._apply_tilloy_gain!(pbody_gain,pbody,
        pbody_r0,pbody_work.liouvillian)
    pbody_L=Matrix(liouvillian(pbody_model;
        representation=:sparse,memory_budget=Inf))
    @test pbody_gain≈pbody_source+pbody_L*pbody_r0 atol=3e-10 rtol=3e-10
end
