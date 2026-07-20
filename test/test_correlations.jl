function _correlation_shifted_allocation(workspace,plan,state,omega)
    PermutationalInvariantDynamics._correlation_shifted_solve!(
        workspace,plan,state,omega,1e-12,1e-10,50)
    @allocated PermutationalInvariantDynamics._correlation_shifted_solve!(
        workspace,plan,state,omega,1e-12,1e-10,50)
end

function _correlation_time_allocation(destination,plan,state,delays,steps,
                                      workspace)
    PermutationalInvariantDynamics.two_time_correlation!(
        destination,plan,state,delays;
        steps_per_interval=steps,workspace=workspace)
    @allocated PermutationalInvariantDynamics.two_time_correlation!(
        destination,plan,state,delays;
        steps_per_interval=steps,workspace=workspace)
end

@testset "PI quantum-regression correlations" begin
    b=PIBasis(1,2)
    spin=spin_matrices()
    c=collective_spin(b,:minus)
    cdag=adjoint(c)
    gamma=0.7
    omega=1.3
    model=PIModel(b,(
        LocalHamiltonian(spin.jz;rate=omega),
        LocalJump(spin.jm;rate=gamma),
    ))
    compiled=compile(model;backend=:matrixfree)
    @test compiled.backend===:matrixfree
    excited=computational_product_state(b,2)

    # This non-Hermitian pair detects an accidental use of
    # expectation(x,A)=Tr(A' x): QRT requires Tr(A*x), with A=c'.
    plan=CorrelationPlan(compiled,cdag,c)
    workspace=CorrelationWorkspace(plan;krylovdim=4)
    time_workspace=CorrelationWorkspace(plan;krylovdim=4,mode=:time)
    @test time_workspace.krylov===nothing
    @test isempty(time_workspace.rhs)
    @test isempty(time_workspace.solution)
    @test Base.summarysize(time_workspace)<Base.summarysize(workspace)
    @test_throws ArgumentError CorrelationWorkspace(plan;mode=:invalid)
    delays=[0.0,0.2,0.7]
    values=two_time_correlation(plan,excited,delays;
        steps_per_interval=100,workspace=workspace)
    expected=exp.((-gamma/2+im*omega).*delays)
    @test values≈expected atol=2e-10 rtol=2e-10
    @test values[1]≈1

    destination=similar(values)
    @test two_time_correlation!(destination,plan,excited,delays;
        steps_per_interval=100,workspace=workspace)===destination
    @test destination≈values atol=2e-10
    @test two_time_correlation(plan,excited,delays;
        steps_per_interval=100,workspace=time_workspace)≈values atol=2e-10
    @test_throws ArgumentError two_time_correlation(plan,excited,[0.1,0.0])
    @test_throws ArgumentError two_time_correlation(plan,excited,[-0.1])

    # Incoherently pumped two-level stationary state.  Both the QRT time
    # correlation and the one-sided resolvent have closed forms.
    gamma_down=0.8
    gamma_up=0.2
    Gamma=gamma_down+gamma_up
    pe=gamma_up/Gamma
    stationary_model=PIModel(b,(
        LocalHamiltonian(spin.jz;rate=omega),
        LocalJump(spin.jm;rate=gamma_down),
        LocalJump(spin.jp;rate=gamma_up),
    ))
    stationary_compiled=compile(stationary_model;backend=:matrixfree)
    rho_ss=iid_state(b,ComplexF64[1-pe 0;0 pe])
    stationary_plan=CorrelationPlan(stationary_compiled,cdag,c)
    # Supplying a non-default workspace must not require repeating its
    # Krylov dimension on every call.
    stationary_workspace=CorrelationWorkspace(stationary_plan;krylovdim=4)
    frequencies=[0.0,omega,2omega]
    spectrum=stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace)
    expected_spectrum=[pe/(Gamma/2+im*(w-omega)) for w in frequencies]
    @test spectrum.values≈expected_spectrum atol=2e-11 rtol=2e-11
    @test spectrum.convention===:one_sided_exp_minus_iomega_t
    @test spectrum.method===:matrixfree_shifted_gmres
    @test spectrum.shared_arnoldi
    @test spectrum.solver===:multishift
    @test spectrum.shared_batches==1
    @test isempty(spectrum.fallback_frequencies)
    @test spectrum.connected
    @test spectrum.stationary_residual<1e-13
    @test_throws ArgumentError stationary_correlation_spectrum(
        stationary_plan,rho_ss,[omega];connected=false,
        workspace=stationary_workspace)
    stationary_time_workspace=CorrelationWorkspace(
        stationary_plan;krylovdim=4,mode=:time)
    @test_throws ArgumentError stationary_correlation_spectrum(
        stationary_plan,rho_ss,[omega];workspace=stationary_time_workspace)

    sequential_spectrum=stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        solver=:sequential)
    forced_shared_spectrum=stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        solver=:multishift)
    @test sequential_spectrum.solver===:sequential
    @test !sequential_spectrum.shared_arnoldi
    @test forced_shared_spectrum.values≈sequential_spectrum.values atol=2e-11 rtol=2e-11
    @test forced_shared_spectrum.solver===:multishift
    supplied_multi=MultiShiftGMRESWorkspace(
        ComplexF64,length(stationary_plan.basis),2,4)
    supplied_result=stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        multishift_workspace=supplied_multi)
    @test supplied_result.values≈sequential_spectrum.values atol=2e-11 rtol=2e-11
    @test supplied_result.shared_batches==1
    @test supplied_result.fallback_frequencies==[3]
    @test_throws ArgumentError stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        multishift_workspace=supplied_multi,multishift_batchsize=3)
    @test_throws ArgumentError stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        solver=:multishift,multishift_workspace=supplied_multi)
    wrong_multi=MultiShiftGMRESWorkspace(ComplexF64,2,2,2)
    @test_throws DimensionMismatch stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        multishift_workspace=wrong_multi)
    memory_fallback=stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        shared_memory_budget=1)
    @test memory_fallback.solver===:sequential
    @test memory_fallback.values≈sequential_spectrum.values atol=2e-11 rtol=2e-11
    @test_throws ArgumentError stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        solver=:multishift,shared_memory_budget=1)
    @test_throws ArgumentError stationary_correlation_spectrum(
        stationary_plan,rho_ss,frequencies;workspace=stationary_workspace,
        solver=:unsupported)
    empty_spectrum=stationary_correlation_spectrum(
        stationary_plan,rho_ss,Float64[];workspace=stationary_workspace,
        solver=:multishift,multishift_workspace=supplied_multi)
    @test isempty(empty_spectrum.values)

    # The shifted solve itself reuses every large Krylov array.  Warm the
    # exact callable-operator specialization before checking the small scalar
    # bookkeeping allowance across supported Julia versions.
    PID=PermutationalInvariantDynamics
    PID._stationary_correlation_seed!(
        stationary_workspace,stationary_plan,rho_ss,true)
    _correlation_shifted_allocation(
        stationary_workspace,stationary_plan,rho_ss.data,omega)
    shifted_alloc=_correlation_shifted_allocation(
        stationary_workspace,stationary_plan,rho_ss.data,omega)
    @test shifted_alloc<=512

    emission=optical_spectrum(stationary_compiled,rho_ss,c,[omega])
    @test only(emission.values)≈2pe/Gamma atol=2e-11
    emission_reused=optical_spectrum(
        stationary_plan,rho_ss,[omega];workspace=stationary_workspace)
    @test only(emission_reused.values)≈2pe/Gamma atol=2e-11

    # Dependency-free radix-two transform: compare every returned signed bin
    # against the defining trapezoidal sum, not against another FFT.
    fft_delays=collect(range(0.0,1.0;length=5))
    fft_values=exp.((-0.6+0.4im).*fft_delays)
    fft_result=correlation_spectrum_fft(
        fft_delays,fft_values;nfft=8)
    weights=ones(length(fft_delays));weights[[1,end]].=0.5
    direct=[(fft_delays[2]-fft_delays[1])*sum(weights.*fft_values.*
            exp.(-im*w.*fft_delays)) for w in fft_result.frequencies]
    @test fft_result.values≈direct atol=2e-14 rtol=2e-14
    @test fft_result.method===:radix2_fft
    @test fft_result.convention===:finite_one_sided_exp_minus_iomega_t
    @test all(iszero,correlation_spectrum_fft(
        fft_delays,ones(5);offset=1,nfft=8).values)
    @test_throws ArgumentError correlation_spectrum_fft(
        [0.0,0.2,0.5],ones(3))

    sampled=correlation_spectrum_fft(stationary_plan,rho_ss,
        collect(range(0.0,2.0;length=17));steps_per_interval=8,
        workspace=stationary_workspace,nfft=32)
    sampled_time_only=correlation_spectrum_fft(stationary_plan,rho_ss,
        collect(range(0.0,2.0;length=17));steps_per_interval=8,
        workspace=stationary_time_workspace,nfft=32)
    @test sampled.connected
    @test sampled.offset≈0 atol=1e-14
    @test sampled_time_only.values≈sampled.values atol=1e-14 rtol=1e-14

    gdelays=[0.0,0.4,1.0]
    g2=delayed_second_order_correlation(
        stationary_compiled,rho_ss,c,gdelays;steps_per_interval=100)
    @test g2≈1 .- exp.(-Gamma.*gdelays) atol=2e-10 rtol=2e-10
    @test g2[1]≈0 atol=1e-14
    @test second_order_correlation(
        stationary_compiled,rho_ss,c,[0.0];steps_per_interval=8)==g2[1:1]

    ground=computational_product_state(b,1)
    decay_only=compile(PIModel(b,(LocalJump(spin.jm;rate=gamma),));
                       backend=:matrixfree)
    @test_throws DomainError delayed_second_order_correlation(
        decay_only,ground,c,[0.0])
    @test_throws ArgumentError delayed_second_order_correlation(
        compiled,excited,c,[0.0])
    @test delayed_second_order_correlation(
        compiled,excited,c,[0.0];normalized=false)==ComplexF64[0]

    # The time-domain backend and its result retain a fully Float32 model.
    spin32=spin_matrices(2;T=Float32)
    geometry32=OneBodyGeometry(b;T=Float32)
    c32=collective_spin(b,:minus;cache=geometry32)
    model32=PIModel(b,(
        LocalHamiltonian(spin32.jz;rate=Float32(omega)),
        LocalJump(spin32.jm;rate=Float32(gamma)),
    ))
    excited32=computational_product_state(b,2;T=Float32)
    plan32=CorrelationPlan(compile(model32;backend=:matrixfree),adjoint(c32),c32)
    values32=two_time_correlation(plan32,excited32,Float32[0,0.1];
                                  steps_per_interval=16)
    @test eltype(values32)===ComplexF32
    @test eltype(two_time_correlation(plan32,excited32,[0];
                                     steps_per_interval=1))===ComplexF32
    @test_throws ArgumentError two_time_correlation(
        plan32,excited32,Float64[0,0.1])
    @test_throws ArgumentError two_time_correlation(
        plan32,excited32,[16_777_217])

    stationary_model32=PIModel(b,(
        LocalHamiltonian(spin32.jz;rate=Float32(omega)),
        LocalJump(spin32.jm;rate=Float32(gamma_down)),
        LocalJump(spin32.jp;rate=Float32(gamma_up)),
    ))
    stationary_compiled32=compile(stationary_model32;backend=:matrixfree)
    rho_ss32=iid_state(b,ComplexF32[1-Float32(pe) 0;0 Float32(pe)])
    stationary_plan32=CorrelationPlan(
        stationary_compiled32,adjoint(c32),c32)
    stationary_workspace32=CorrelationWorkspace(stationary_plan32;krylovdim=4)
    spectrum32=stationary_correlation_spectrum(
        stationary_plan32,rho_ss32,Float32[omega];
        workspace=stationary_workspace32)
    @test eltype(spectrum32.values)===ComplexF32
    @test_throws ArgumentError stationary_correlation_spectrum(
        stationary_plan32,rho_ss32,Float64[omega];
        workspace=stationary_workspace32)
    @test_throws ArgumentError stationary_correlation_spectrum(
        stationary_plan32,rho_ss32,Float32[omega];atol=1e-8,
        workspace=stationary_workspace32)
    @test_throws ArgumentError delayed_second_order_correlation(
        stationary_compiled32,rho_ss32,c32,Float32[0];
        stationarity_atol=1e-6)
    @test_throws ArgumentError correlation_spectrum_fft(
        stationary_plan32,rho_ss32,Float64[0,0.1];nfft=2,
        workspace=stationary_workspace32)

    # A genuinely multi-sector N=2 state, including coherent one-body data,
    # agrees with dense evolution in the ten-dimensional PI coordinate space.
    b2=PIBasis(2,2)
    c2=collective_spin(b2,:minus)
    A2=adjoint(c2)
    model2=PIModel(b2,(
        LocalHamiltonian(spin.jz;rate=0.73),
        LocalJump(spin.jm;rate=0.41),
        LocalJump(spin.jp;rate=0.17),
    ))
    rho2=iid_state(b2,ComplexF64[
        0.61       0.12+0.03im
        0.12-0.03im 0.39
    ])
    populations2=sector_populations(rho2)
    @test length(populations2)==2
    @test all(p->real(p)>0,Base.values(populations2))
    prepared2=compile(model2;backend=:matrixfree)
    plan2=CorrelationPlan(prepared2,A2,c2)
    workspace2=CorrelationWorkspace(plan2;krylovdim=8)
    time_workspace2=CorrelationWorkspace(plan2;krylovdim=8,mode=:time)
    delays2=[0.0,0.17,0.43]
    values2=two_time_correlation(plan2,rho2,delays2;
        steps_per_interval=32,workspace=workspace2)
    denseL2=Matrix(liouvillian(model2;representation=:sparse))
    seed2=c2*PIOperator(b2,rho2.data)
    reference2=[dot(adjoint(A2).data,exp(delay*denseL2)*seed2.data)
                for delay in delays2]
    @test values2≈reference2 atol=5e-12 rtol=5e-12
    @test Base.summarysize(time_workspace2)<Base.summarysize(workspace2)

    # The warmed in-place path performs no history allocation and reuses the
    # plan-owned Liouvillian geometry plus caller-owned RK4 storage.
    _correlation_time_allocation(
        values2,plan2,rho2,delays2,2,workspace2)
    time_alloc=_correlation_time_allocation(
        values2,plan2,rho2,delays2,2,time_workspace2)
    @test time_alloc<=512
end
