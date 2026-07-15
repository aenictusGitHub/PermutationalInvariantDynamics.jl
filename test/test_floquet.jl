@testset "Floquet dynamics and preallocated time dependence" begin
    sx=ComplexF64[0 1;1 0];sm=ComplexF64[0 1;0 0];T=1.3
    b=PIBasis(2,2);rate=(t,p)->1+0.3cos(2pi*t/T)
    periodic=PIModel(b,[LocalJump(sm;rate=rate)])
    L=liouvillian(periodic;representation=:matrixfree)
    rho=iid_pure_state(b,ComplexF64[0,1]);y=similar(rho.data)
    L.action!(y,rho.data,0.1,nothing)
    @test (@allocated L.action!(y,rho.data,0.2,nothing))<=1024
    L0=Matrix(liouvillian(PIModel(b,[LocalJump(sm)]);representation=:sparse))
    F=floquet_propagator(periodic,T;steps=180)
    @test F≈exp(T*L0) atol=2e-9
    tau=PermutationalInvariantDynamics._trace_vector(b,ComplexF64)
    @test norm(adjoint(tau)*F-adjoint(tau))<2e-10
    vals=floquet_multipliers(F);@test minimum(abs.(vals.-1))<2e-10
    @test floquet_exponents(F,T)≈log.(complex.(vals))./T
    @test floquet_gap(F,T)>=0

    # The gap is defined only after identifying an actual fixed multiplier.
    # A merely closest root must not be discarded as if it were stationary.
    near_fixed=Diagonal(ComplexF64[1+5e-9,exp(-0.4)])
    @test floquet_gap(near_fixed,1.0;atol=1e-8)≈0.4 atol=2e-15
    @test_throws ArgumentError floquet_gap(near_fixed,1.0;atol=1e-10)
    @test_throws ArgumentError floquet_gap(
        Diagonal(ComplexF64[0.9,0.5]),1.0;atol=1e-6)
    @test_throws ArgumentError floquet_gap(
        Diagonal(ComplexF64[1,2]),1.0;atol=1e-10)

    gap32_map=Diagonal(ComplexF32[1,0.5])
    gap32=floquet_gap(gap32_map,2f0;atol=1f-6)
    @test gap32 isa Float32
    @test gap32≈-log(0.5f0)/2f0 atol=2f-7
    # A wider period promotes the physical rate, while a one-dimensional map
    # retains the natural scalar type of its (undefined) subleading gap.
    @test floquet_gap(gap32_map,2.0;atol=1f-6) isa Float64
    single_gap32=floquet_gap(reshape(ComplexF32[1],1,1),1f0;atol=0)
    @test single_gap32 isa Float32 && isnan(single_gap32)

    for bad_period in (0.0,-1.0,Inf,NaN)
        @test_throws ArgumentError floquet_gap(gap32_map,bad_period)
    end
    for bad_atol in (-1.0,Inf,NaN)
        @test_throws ArgumentError floquet_gap(gap32_map,1f0;atol=bad_atol)
    end

    # Propagator buffers preserve a fully Float32 model/time problem, while a
    # compatible Float64 model/time uses Float64 integration. Integer times
    # are accepted only when exactly representable in the selected precision.
    sm32=ComplexF32.(sm);constant32=PIModel(b,[LocalJump(sm32)])
    F32=floquet_propagator(constant32,0.2f0;steps=40)
    constant64=PIModel(b,[LocalJump(ComplexF64.(sm32))])
    F64=floquet_propagator(constant64,0.2;steps=40)
    reference32=exp(0.2*Matrix(liouvillian(constant64;representation=:sparse)))
    @test eltype(F32)===ComplexF32
    @test eltype(F64)===ComplexF64
    @test ComplexF64.(F32)≈reference32 atol=2e-6
    @test F64≈reference32 atol=2e-7
    @test Base.summarysize(F32)<Base.summarysize(F64)
    # A compiled F32 plan owns F32 matvec scratch, so a wider time input that
    # promotes RK storage is rejected rather than silently narrowed per stage.
    @test_throws ArgumentError floquet_propagator(constant32,0.2;steps=2)
    @test_throws ArgumentError floquet_propagator(
        constant32,0.2f0;steps=2,t0=0.0)
    @test_throws ArgumentError floquet_propagator(
        constant32,typemax(Int);steps=1)
    @test_throws ArgumentError floquet_propagator(constant32,Inf;steps=1)
    @test_throws ArgumentError floquet_propagator(constant32,0.2f0;
                                                  steps=1,t0=NaN)

    ss=floquet_steady_state(periodic,T;steps=120,return_info=true)
    ground=iid_pure_state(b,ComplexF64[1,0])
    @test ss.state.data≈ground.data atol=2e-9
    @test ss.residual<2e-9
    trajectory=stroboscopic_evolution(rho,F,3)
    @test length(trajectory)==4
    @test trajectory[end].data≈floquet_evolve(rho,F,3).data atol=2e-11

    # Constant Hamiltonian is a second, non-dissipative reference.
    unitary=PIModel(b,[LocalHamiltonian(sx;rate=(t,p)->0.4)])
    Fu=floquet_propagator(unitary,T;steps=200)
    Lu=Matrix(liouvillian(PIModel(b,[LocalHamiltonian(sx;rate=.4)]);representation=:sparse))
    @test Fu≈exp(T*Lu) atol=3e-9

    bp=PIBasis(3,2);pair=kron(sm,sm);rp=(t,p)->0.2+0.1sin(t)
    Lp=liouvillian(PIModel(bp,[LocalPBodyJump(pair,2;rate=rp)]);representation=:matrixfree)
    Lpc=liouvillian(PIModel(bp,[LocalPBodyJump(pair,2;rate=rp(0.37,nothing))]);representation=:matrixfree)
    xp=iid_pure_state(bp,ComplexF64[0,1]).data;yp=similar(xp);yref=similar(xp)
    Lp.action!(yp,xp,0.37,nothing);Lpc.action!(yref,xp,0.0,nothing)
    @test yp≈yref atol=2e-11
end
