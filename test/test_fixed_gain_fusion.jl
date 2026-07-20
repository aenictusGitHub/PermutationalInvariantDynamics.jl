const PIDFixedGain = PermutationalInvariantDynamics

@testset "factorized fixed local gain kernels" begin
    b=PIBasis(4,2)
    lowering=ComplexF64[0 1;0 0]
    onebody=PIModel(b,(LocalJump(lowering;rate=0.37),))
    onebody_plan=LiouvillianPlan(onebody)
    @test only(onebody_plan.kernels) isa PIDFixedGain.FactorizedLocalJumpPIKernel
    onebody_kernel=only(onebody_plan.kernels)
    @test all(c->c isa PIDFixedGain._StaticOneBodyContraction,
              onebody_kernel.contractions)
    @test any(c->c.use_support,onebody_kernel.contractions)
    @test all(c->length(c.values)<=length(c.matrix),
              onebody_kernel.contractions)

    geometry=OneBodyGeometry(b)
    qblocks=[collective_block(b,adjoint(lowering)*lowering,sector;
                              cache=geometry) for sector in b.sectors]
    triplets=PIDFixedGain._local_kernel_triplets(
        b,geometry,lowering,lowering)
    reference_kernel=PIDFixedGain.LocalJumpPIKernel(qblocks,triplets,0.37)
    reference_plan=LiouvillianPlan(b,(reference_kernel,),
        copy(onebody_plan.tracevec),nothing,onebody_plan.Ttype,true)

    rng=MersenneTwister(918)
    input=randn(rng,ComplexF64,length(b))
    factorized=zeros(ComplexF64,length(b));reference=similar(factorized)
    apply!(factorized,onebody_plan,input,LiouvillianWorkspace(onebody_plan))
    apply!(reference,reference_plan,input,LiouvillianWorkspace(reference_plan))
    @test factorized≈reference rtol=2e-14 atol=2e-14

    apply_adjoint!(factorized,onebody_plan,input,
                   LiouvillianWorkspace(onebody_plan))
    apply_adjoint!(reference,reference_plan,input,
                   LiouvillianWorkspace(reference_plan))
    @test factorized≈reference rtol=2e-14 atol=2e-14

    onebody_inputs=randn(rng,ComplexF64,length(b),3)
    onebody_outputs=zeros(ComplexF64,size(onebody_inputs))
    onebody_references=similar(onebody_outputs)
    apply!(onebody_outputs,onebody_plan,onebody_inputs,0.0,nothing,
           LiouvillianWorkspace(onebody_plan))
    apply!(onebody_references,reference_plan,onebody_inputs,0.0,nothing,
           LiouvillianWorkspace(reference_plan))
    @test onebody_outputs≈onebody_references rtol=2e-14 atol=2e-14
    apply_adjoint!(onebody_outputs,onebody_plan,onebody_inputs,0.0,nothing,
                   LiouvillianWorkspace(onebody_plan))
    apply_adjoint!(onebody_references,reference_plan,onebody_inputs,0.0,
                   nothing,LiouvillianWorkspace(reference_plan))
    @test onebody_outputs≈onebody_references rtol=2e-14 atol=2e-14
    @test PIDFixedGain._matrix_from_plan(onebody_plan)≈
          PIDFixedGain._matrix_from_plan(reference_plan) rtol=2e-14 atol=2e-14

    p=2;pair_operator=kron(lowering,lowering)
    pmodel=PIModel(b,(LocalPBodyJump(pair_operator,p;rate=0.19),))
    pplan=LiouvillianPlan(pmodel)
    @test only(pplan.kernels) isa
          PIDFixedGain.FactorizedLocalPBodyJumpPIKernel
    pgeometry=PBodyGeometry(b,p)
    pair_q=adjoint(pair_operator)*pair_operator
    pqblocks=[pbody_collective_block(pgeometry,pair_q,sector;check=false)
              for sector in b.sectors]
    ptriplets=PIDFixedGain.pbody_kernel_triplets(
        pgeometry,pair_operator,pair_operator)
    preference_kernel=PIDFixedGain.LocalJumpPIKernel(
        pqblocks,ptriplets,0.19)
    preference_plan=LiouvillianPlan(b,(preference_kernel,),
        copy(pplan.tracevec),nothing,pplan.Ttype,true)
    apply!(factorized,pplan,input,LiouvillianWorkspace(pplan))
    apply!(reference,preference_plan,input,LiouvillianWorkspace(preference_plan))
    @test factorized≈reference rtol=3e-13 atol=3e-13
    @test PIDFixedGain._matrix_from_plan(pplan)≈
          PIDFixedGain._matrix_from_plan(preference_plan) rtol=3e-13 atol=3e-13

    inputs=randn(rng,ComplexF64,length(b),3)
    outputs=zeros(ComplexF64,size(inputs));references=similar(outputs)
    apply!(outputs,pplan,inputs,0.0,nothing,LiouvillianWorkspace(pplan))
    apply!(references,preference_plan,inputs,0.0,nothing,
           LiouvillianWorkspace(preference_plan))
    @test outputs≈references rtol=3e-13 atol=3e-13

    # Exact supports preserve the prepared scalar type and complex phases;
    # coefficient products remain evaluated at application time in the same
    # scale-first order as the dense path.
    b32=PIBasis(5,2)
    lowering32=ComplexF32[0 1+0.25im;0 0]
    plan32=LiouvillianPlan(PIModel(b32,(LocalJump(lowering32;rate=0.3f0),)))
    kernel32=only(plan32.kernels)
    @test kernel32 isa PIDFixedGain.FactorizedLocalJumpPIKernel
    @test eltype(first(kernel32.contractions))===ComplexF32
    @test any(c->c.use_support,kernel32.contractions)
    matrix32=Matrix(PIDFixedGain._matrix_from_plan(plan32))
    input32=randn(rng,ComplexF32,length(b32),2)
    output32=zeros(ComplexF32,size(input32))
    apply!(output32,plan32,input32,0.0,nothing,
           LiouvillianWorkspace(plan32))
    @test output32≈matrix32*input32 rtol=8f-6 atol=8f-6
    apply_adjoint!(output32,plan32,input32,0.0,nothing,
                   LiouvillianWorkspace(plan32))
    @test output32≈adjoint(matrix32)*input32 rtol=8f-6 atol=8f-6
end

@testset "autonomous fixed-kernel fusion" begin
    b=PIBasis(4,2)
    sx=ComplexF64[0 1;1 0]
    sz=ComplexF64[1 0;0 -1]
    lowering=ComplexF64[0 1;0 0]
    pair_lowering=kron(lowering,lowering)
    model=PIModel(b,(
        LocalHamiltonian(sx;rate=0.31),
        CollectiveHamiltonian(sz;rate=-0.07),
        CollectiveJump(lowering;rate=0.11),
        LocalJump(sz;rate=0.13),
        LocalPBodyJump(pair_lowering,2;rate=0.17)))
    fused=LiouvillianPlan(model)
    reference=LiouvillianPlan(model;fuse_static=false)
    @test length(fused.kernels)==1
    kernel=only(fused.kernels)
    @test kernel isa PIDFixedGain.FusedStaticPIKernel
    @test kernel.hamiltonian_blocks!==nothing
    @test kernel.loss_blocks!==nothing
    @test length(kernel.collective_gains)==1
    @test length(kernel.onebody_gains)==1
    @test length(kernel.pbody_gains)==1

    raw=PIDFixedGain._static_kernels_unfused(model)
    reused=only(PIDFixedGain._fuse_fixed_kernels(raw,b))
    @test reused.collective_gains[1].blocks[1]===raw[3].blocks[1]
    @test reused.onebody_gains[1].contractions[1]===
          raw[4].contractions[1]
    @test reused.pbody_gains[1].contractions[1]===
          raw[5].contractions[1]

    rng=MersenneTwister(511)
    input=randn(rng,ComplexF64,length(b))
    y=zeros(ComplexF64,length(b));yref=similar(y)
    apply!(y,fused,input,LiouvillianWorkspace(fused))
    apply!(yref,reference,input,LiouvillianWorkspace(reference))
    @test y≈yref rtol=4e-13 atol=4e-13
    apply_adjoint!(y,fused,input,LiouvillianWorkspace(fused))
    apply_adjoint!(yref,reference,input,LiouvillianWorkspace(reference))
    @test y≈yref rtol=4e-13 atol=4e-13

    inputs=randn(rng,ComplexF64,length(b),4)
    output=zeros(ComplexF64,size(inputs));output_ref=similar(output)
    apply!(output,fused,inputs,0.0,nothing,LiouvillianWorkspace(fused))
    apply!(output_ref,reference,inputs,0.0,nothing,
           LiouvillianWorkspace(reference))
    @test output≈output_ref rtol=4e-13 atol=4e-13
    apply_adjoint!(output,fused,inputs,0.0,nothing,
                   LiouvillianWorkspace(fused))
    apply_adjoint!(output_ref,reference,inputs,0.0,nothing,
                   LiouvillianWorkspace(reference))
    @test output≈output_ref rtol=4e-13 atol=4e-13
    @test PIDFixedGain._matrix_from_plan(fused)≈
          PIDFixedGain._matrix_from_plan(reference) rtol=4e-13 atol=4e-13

    compiled=compile(model;backend=:matrixfree,memory_budget=Inf)
    trajectory_plan=TrajectoryPlan(compiled)
    @test length(trajectory_plan.hamiltonians)==2
    @test length(trajectory_plan.jumps)==3
    @test all(k->!(k isa PIDFixedGain.FusedStaticPIKernel),
              trajectory_plan.liouvillian.kernels)

    diagonal_model=PIModel(b,(
        LocalHamiltonian(sz;rate=0.09),LocalJump(sz;rate=0.14)))
    diagonal_fused=LiouvillianPlan(diagonal_model)
    @test only(diagonal_fused.kernels) isa PIDFixedGain.FusedStaticPIKernel
    population_from_fused=PopulationPlan(diagonal_fused)
    population_from_model=PopulationPlan(diagonal_model)
    @test population_generator(population_from_fused;representation=:sparse)≈
          population_generator(population_from_model;representation=:sparse)

    # Scalar schedules remain term-resolved: their rates cannot be absorbed
    # into fixed Hamiltonian/loss blocks at compile time.
    driven=PIModel(b,(
        LocalHamiltonian(sx;rate=(t,p)->p.hrate),
        LocalJump(lowering;rate=(t,p)->p.jrate)))
    driven_plan=LiouvillianPlan(driven)
    @test all(kernel->!(kernel isa PIDFixedGain.FusedStaticPIKernel),
              driven_plan.kernels)
    @test driven_plan.kernels[2] isa
          PIDFixedGain.FactorizedLocalJumpPIKernel

    # A partially parameterized family can contain one scheduled kernel and
    # one autonomous fused group. Both matrix-free and explicit
    # specialization must retain the embedded fixed scales.
    family=compile_family(model;rate_indices=(1,))
    @test any(k->k isa PIDFixedGain.FusedStaticPIKernel,family.plan.kernels)
    specialized=specialize(family,(0.29,);backend=:sparse)
    specialized_model=PIModel(b,(
        LocalHamiltonian(sx;rate=0.29),model.terms[2:end]...))
    @test Matrix(specialized.operator)≈Matrix(liouvillian(
        specialized_model;representation=:sparse,memory_budget=Inf))
end
