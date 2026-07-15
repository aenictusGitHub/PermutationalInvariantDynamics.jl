@testset "preallocated density-matrix evolution" begin
    sm=ComplexF64[0 1;0 0];b=PIBasis(2,2);model=PIModel(b,[LocalJump(sm)])
    Ls=liouvillian(model;representation=:sparse);Lm=liouvillian(model;representation=:matrixfree)
    rho=iid_pure_state(b,ComplexF64[0,1]);rho0=copy(rho);t=0.7
    reference=PIState(b,exp(t*Matrix(Ls))*rho.data)
    rs=time_evolve(Ls,rho,(0.0,t);steps=400);rm=time_evolve(Lm,rho,(0.0,t);steps=400)
    @test rs.data≈reference.data atol=2e-12
    @test rm.data≈reference.data atol=2e-12
    @test trace(rm)≈1 atol=2e-12
    @test rho.data==rho0.data

    dest=copy(rho);w=EvolutionWorkspace(rho)
    evolve!(dest,Lm,rho,(0.0,t);steps=400,workspace=w)
    @test dest.data≈reference.data atol=2e-12
    evolve!(dest,Lm,rho,(0.0,0.001);steps=1,workspace=w) # warm the kernel
    @test (@allocated evolve!(dest,Lm,rho,(0.0,0.001);steps=1,workspace=w))<=2048
    evolve!(dest,Lm,rho,(0.0,t);steps=400,workspace=w)
    evolve!(dest,Lm,dest,(t,2t);steps=400,workspace=w)
    @test dest.data≈exp(2t*Matrix(Ls))*rho.data atol=3e-12

    times=range(0,t;length=5);trajectory=time_evolution(Lm,rho,times;steps_per_interval=100)
    @test length(trajectory)==length(times)
    @test trajectory[1].data==rho.data
    @test trajectory[end].data≈reference.data atol=2e-12

    rate=(u,p)->1+0.2u
    driven_model=PIModel(b,[LocalJump(sm;rate=rate)])
    Lt=liouvillian(driven_model;representation=:matrixfree)
    rt=time_evolve(Lt,rho,(0.0,t);steps=500);integrated_rate=t+0.1t^2
    @test rt.data≈exp(integrated_rate*Matrix(Ls))*rho.data atol=2e-12
    @test !PermutationalInvariantDynamics.isautonomous(driven_model)
    @test !PermutationalInvariantDynamics.isautonomous(Lt)
    @test_throws ArgumentError Lt*rho.data
    @test_throws ArgumentError steady_state(driven_model)
    @test_throws ArgumentError steady_state(Lt;method=:krylov)

    tf=0.4
    Lf=PermutationalInvariantDynamics.freeze(driven_model;time=tf)
    Lfs=PermutationalInvariantDynamics.freeze(driven_model;time=tf,representation=:sparse)
    Lf_from_operator=PermutationalInvariantDynamics.freeze(Lt;time=tf)
    @test PermutationalInvariantDynamics.isautonomous(Lf)
    @test Lf*rho.data≈(1+0.2tf)*(Ls*rho.data) atol=2e-12
    @test Lfs*rho.data≈Lf*rho.data atol=2e-12
    @test Lf_from_operator*rho.data≈Lf*rho.data atol=2e-12
    @test PermutationalInvariantDynamics.freeze(Lt;time=tf,representation=:sparse)*rho.data≈Lf*rho.data atol=2e-12

    parameterized=PIModel(b,[LocalJump(sm;rate=(u,p)->p.base+u)])
    Lfp=PermutationalInvariantDynamics.freeze(parameterized;time=0.25,
                                               parameters=(base=0.75,))
    @test Lfp*rho.data≈Ls*rho.data atol=2e-12
    operator_driven=PIModel(b,[LocalJump((u,p)->sm;rate=1)])
    @test !PermutationalInvariantDynamics.isautonomous(operator_driven)
    @test PermutationalInvariantDynamics.freeze(operator_driven;time=tf)*rho.data≈Ls*rho.data atol=2e-12

    parameter_problem=dynamics_problem(parameterized,rho,(0.0,t);
                                       parameters=(base=0.75,))
    parameter_du=similar(rho.data)
    parameter_problem.f(parameter_du,rho.data,parameter_problem.p,tf)
    @test parameter_du≈(0.75+tf)*(Ls*rho.data) atol=2e-12

    for source in (Ls,Lm,Lt,driven_model)
        prob=dynamics_problem(source,rho,(0.0,t))
        du=similar(rho.data)
        prob.f(du,rho.data,nothing,tf)
        expected=source===Ls ? Ls*rho.data :
                 source===Lm ? Lm*rho.data : (1+0.2tf)*(Ls*rho.data)
        @test du≈expected atol=2e-12
    end
    @test_throws DimensionMismatch dynamics_problem(zeros(ComplexF64,2,2),rho,(0.0,t))
    @test_throws ArgumentError time_evolution(Ls,rho,[0.0,1.0,0.5])
    @test_throws ArgumentError time_evolve(Ls,rho,(0.0,1.0);steps=0)
end

@testset "compiled Liouvillian plans and workspaces" begin
    PID=PermutationalInvariantDynamics
    b=PIBasis(3,2)
    sx=ComplexF64[0 1;1 0];sm=ComplexF64[0 1;0 0]
    model=PIModel(b,[LocalHamiltonian(sx;rate=0.17),
                     LocalJump(sm;rate=0.31),CollectiveJump(sm;rate=0.07)])
    plan=PID.LiouvillianPlan(model)
    @test !ismutabletype(typeof(plan))
    @test all(k->!hasfield(typeof(k),:work),plan.kernels)
    w1=PID.LiouvillianWorkspace(plan);w2=PID.LiouvillianWorkspace(plan)
    @test w1.blocks[1][1]!==w2.blocks[1][1]

    Ls=liouvillian(model;representation=:sparse)
    Lm=liouvillian(model;representation=:matrixfree)
    rng=MersenneTwister(71);x=randn(rng,ComplexF64,length(b));y=similar(x)
    PID.apply!(y,plan,x,w1)
    @test y≈Ls*x atol=2e-12
    PID.apply!(y,plan,x,w1) # warm
    @test (@allocated PID.apply!(y,plan,x,w1))<=512

    X=randn(rng,ComplexF64,length(b),4);Y=similar(X);Y2=similar(X)
    PID.apply!(Y,plan,X,0.0,nothing,w1)
    mul!(Y2,Lm,X)
    @test Y≈Ls*X atol=3e-12
    @test Y2≈Ls*X atol=3e-12
    PID.apply!(Y,plan,X,0.0,nothing,w1) # warm
    @test (@allocated PID.apply!(Y,plan,X,0.0,nothing,w1))<=2048

    z=similar(x);PID.apply_adjoint!(z,plan,x,w1)
    @test z≈adjoint(Ls)*x atol=3e-12
    @test adjoint(Lm)*x≈adjoint(Ls)*x atol=3e-12

    bp=PIBasis(2,2);pair=kron(sm,sm)
    pmodel=PIModel(bp,[LocalPBodyJump(pair,2;rate=0.13),
                       CollectivePBodyJump(pair,2;rate=0.04)])
    pp=PID.LiouvillianPlan(pmodel);pw=PID.LiouvillianWorkspace(pp)
    px=randn(rng,ComplexF64,length(bp));py=similar(px)
    pLs=liouvillian(pmodel;representation=:sparse)
    PID.apply_adjoint!(py,pp,px,pw)
    @test py≈adjoint(pLs)*px atol=3e-12

    cs=PID.compile(model;backend=:auto)
    cm=PID.compile(model;backend=:auto,memory_budget=0)
    @test cs.backend===:sparse
    @test cm.backend===:matrixfree
    @test cs.estimates.scalar_type===eltype(cs)
    @test cs*x≈Ls*x atol=2e-12
    @test cm*x≈Ls*x atol=2e-12

    # Allocating products follow ordinary matrix promotion.  A Float64 plan
    # can consume a narrower source without first allocating an invalid
    # Float32 destination, for sparse, matrix-free, compiled, and adjoint
    # matrix-free routes alike.
    x32=ComplexF32.(x);reference32=Ls*ComplexF64.(x32)
    for operator in (Lm,cs,cm)
        product=operator*x32
        @test eltype(product)===ComplexF64
        @test product≈reference32 atol=2e-12
    end
    adjoint_product=adjoint(Lm)*x32
    @test eltype(adjoint_product)===ComplexF64
    @test adjoint_product≈adjoint(Ls)*ComplexF64.(x32) atol=3e-12

    X32=ComplexF32.(X);matrix_reference32=Ls*ComplexF64.(X32)
    for operator in (Lm,cs,cm)
        product=operator*X32
        @test size(product)==size(X32)
        @test eltype(product)===ComplexF64
        @test product≈matrix_reference32 atol=3e-12
    end
    adjoint_matrix_product=adjoint(Lm)*X32
    @test size(adjoint_matrix_product)==size(X32)
    @test eltype(adjoint_matrix_product)===ComplexF64
    @test adjoint_matrix_product≈adjoint(Ls)*ComplexF64.(X32) atol=4e-12

    @test steady_state(cs)≈steady_state(model) atol=2e-10
    @test pi_liouvillian_spectrum(cs)≈pi_liouvillian_spectrum(model) atol=2e-10
    @test pi_liouvillian_gap(cs)≈pi_liouvillian_gap(model) atol=2e-10
    @test_throws ArgumentError PID.compile(model;backend=:unknown)
    @test_throws ArgumentError PID.compile(model;memory_budget=-1)

    bb=PIBasis(1,2);smb=Complex{BigFloat}[0 1;0 0]
    cbig=PID.compile(PIModel(bb,[LocalJump(smb)]);backend=:matrixfree)
    @test cbig.estimates.scalar_type===Complex{BigFloat}
    xbig=randn(rng,Complex{BigFloat},length(bb));ybig=similar(xbig)
    PID.apply!(ybig,cbig,xbig,0.0,nothing,PID.LiouvillianWorkspace(cbig))
    @test ybig≈liouvillian(cbig.model;representation=:sparse)*xbig atol=big"1e-30"

    driven=PIModel(b,[LocalJump(sm;rate=(t,p)->p.rate*(1+t))])
    cd=PID.compile(driven;backend=:auto,memory_budget=typemax(Int))
    @test cd.backend===:matrixfree
    @test !PID.isautonomous(cd)
    @test_throws ArgumentError cd*x
    @test_throws ArgumentError PID.compile(driven;backend=:sparse)
    wd=PID.LiouvillianWorkspace(cd);yd=similar(x)
    PID.apply!(yd,cd,x,0.2,(rate=0.4,),wd)
    Lf=PID.freeze(driven;time=0.2,parameters=(rate=0.4,),representation=:sparse)
    @test yd≈Lf*x atol=2e-12
    PID.apply_adjoint!(yd,cd,x,0.2,(rate=0.4,),wd)
    @test yd≈adjoint(Lf)*x atol=2e-12

    # Explicit workspaces have no shared mutable state and are suitable for
    # one-per-task execution.  This runs concurrently when tests use >1 thread.
    inputs=[randn(rng,ComplexF64,length(b)) for _ in 1:8]
    explicit_outputs=Vector{Vector{ComplexF64}}(undef,length(inputs))
    shared_outputs=similar(explicit_outputs)
    @sync for i in eachindex(inputs)
        Threads.@spawn begin
            wi=PID.LiouvillianWorkspace(plan);out=similar(inputs[i])
            PID.apply!(out,plan,inputs[i],0.0,nothing,wi)
            explicit_outputs[i]=out
            shared_outputs[i]=Lm*inputs[i]
        end
    end
    @test all(i->isapprox(explicit_outputs[i],Ls*inputs[i];atol=3e-12),eachindex(inputs))
    @test all(i->isapprox(shared_outputs[i],Ls*inputs[i];atol=3e-12),eachindex(inputs))

    rho=iid_pure_state(b,ComplexF64[0,1]);times=range(0,0.2;length=4)
    from_model=time_evolution(model,rho,times;steps_per_interval=20)
    from_compiled=time_evolution(cm,rho,times;steps_per_interval=20)
    @test from_model[end].data≈from_compiled[end].data atol=3e-12
    @test isempty(Test.detect_ambiguities(PID;recursive=true))
end
