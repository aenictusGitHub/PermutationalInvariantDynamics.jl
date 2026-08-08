const _SYMMETRIC_KET_PID = PermutationalInvariantDynamics

# Keep this focused test runnable before the source file is wired into the
# central include list during integration.
isdefined(_SYMMETRIC_KET_PID,:SymmetricKet)||Base.include(
    _SYMMETRIC_KET_PID,
    joinpath(@__DIR__,"..","src","symmetric_kets.jl"))

@testset "physical fully symmetric kets" begin
    PID=_SYMMETRIC_KET_PID
    @test Workflow.SymmetricKet===PID.SymmetricKet
    @test Workflow.symmetric_ket_density===PID.symmetric_ket_density
    @test Workflow.symmetric_ket_expectation===PID.symmetric_ket_expectation

    basis=PIBasis(4,2;sectors=[(4,0)])
    @test PID.symmetric_ket_dimension(basis)==5
    @test_throws ArgumentError PID.symmetric_ket_dimension(PIBasis(4,2))
    @test_throws DimensionMismatch PID.SymmetricKet(
        basis,ComplexF64[1,0])
    @test_throws ArgumentError PID.SymmetricKet(
        basis,ComplexF64[1,1,0,0,0])
    @test_throws ArgumentError PID.SymmetricKet(
        basis,ComplexF64[NaN,0,0,0,0])
    @test_throws ArgumentError PID.SymmetricKet(
        basis,ComplexF32[2,0,0,0,0];atol=1e100,rtol=0)
    @test_throws ArgumentError PID.SymmetricKet(
        basis,ComplexF32[1,0,0,0,0];atol=1e-100,rtol=0)
    real_state=PID.SymmetricKet(basis,Float64[1,0,0,0,0])
    @test eltype(real_state.data)===ComplexF64

    occupation=PID.symmetric_occupation_ket(basis,(3,1))
    @test length(basis)==length(occupation)^2
    occupation_density=PID.symmetric_ket_density(occupation)
    @test occupation_density.data==
        symmetric_occupation_state(basis,(3,1)).data
    @test trace(occupation_density)==1
    @test PID.validate_symmetric_ket(occupation)===occupation
    @test copy(occupation).data==occupation.data
    @test copy(occupation).data!==occupation.data

    local_ket=ComplexF64[1+im,2-im]
    local_ket./=norm(local_ket)
    product=PID.symmetric_product_ket(basis,local_ket)
    product_density=PID.symmetric_ket_density(product)
    @test product_density.data≈iid_pure_state(basis,local_ket).data atol=2e-15
    preallocated_density=PIState(basis)
    @test PID.symmetric_ket_density!(
        preallocated_density,product;check=false,memory_budget=0)===
        preallocated_density
    @test preallocated_density.data≈product_density.data atol=2e-15
    @test_throws ArgumentError PID.symmetric_ket_density!(
        preallocated_density,product;check=false,memory_budget=-1)
    reconstructed=PID.symmetric_ket(product_density)
    @test PID.symmetric_ket_density(reconstructed).data≈
        product_density.data atol=2e-14

    endpoint_a=PID.symmetric_ket_density(
        PID.symmetric_occupation_ket(basis,(4,0)))
    endpoint_b=PID.symmetric_ket_density(
        PID.symmetric_occupation_ket(basis,(0,4)))
    mixed_state=PIState(basis,(endpoint_a.data+endpoint_b.data)./2)
    @test_throws ArgumentError PID.symmetric_ket(mixed_state)

    product32=PID.symmetric_product_ket(
        basis,ComplexF32[inv(sqrt(2f0)),im*inv(sqrt(2f0))])
    @test eltype(product32.data)===ComplexF32
    @test eltype(PID.symmetric_ket_density(product32).data)===ComplexF32
    @test_throws ArgumentError PID.symmetric_ket_density(
        occupation;memory_budget=1)
    @test_throws ArgumentError PID.symmetric_ket(
        occupation_density;memory_budget=1)
    invalid=copy(occupation)
    invalid.data .*= 2
    @test_throws ArgumentError PID.symmetric_ket_density(invalid)

    setprecision(BigFloat,192) do
        local_big=Complex{BigFloat}[sqrt(big"0.25"),sqrt(big"0.75")]
        product_big=PID.symmetric_product_ket(basis,local_big)
        @test eltype(product_big.data)===Complex{BigFloat}
        @test norm(product_big.data)≈one(BigFloat) atol=eps(BigFloat)*100
    end
    one128=setprecision(BigFloat,128) do
        complex(one(BigFloat),zero(BigFloat))
    end
    zero256=setprecision(BigFloat,256) do
        complex(zero(BigFloat),zero(BigFloat))
    end
    @test_throws ArgumentError PID.SymmetricKet(
        basis,[one128,zero256,zero256,zero256,zero256])
    mixed_component=complex(real(one128),imag(zero256))
    @test_throws ArgumentError PID.SymmetricKet(
        basis,[mixed_component,zero256,zero256,zero256,zero256])

    state160=setprecision(BigFloat,160) do
        amplitude=inv(sqrt(BigFloat(2)))
        PID.symmetric_product_ket(
            basis,Complex{BigFloat}[amplitude,im*amplitude])
    end
    setprecision(BigFloat,256) do
        density160=PID.symmetric_ket_density(state160)
        @test all(value->precision(real(value))==160&&
                         precision(imag(value))==160,density160.data)
    end

    tiny_basis=PIBasis(1,2;sectors=[(1,0)])
    component_tiny=PID.SymmetricKet(
        tiny_basis,ComplexF64[1+1e-200im,1e-200])
    @test_throws ArgumentError PID.symmetric_ket_density(component_tiny)
end

@testset "symmetric-ket Hamiltonian plans and evolution" begin
    PID=_SYMMETRIC_KET_PID
    basis=PIBasis(5,2;sectors=[(5,0)])
    spin=spin_matrices()
    initial=PID.symmetric_product_ket(
        basis,ComplexF64[inv(sqrt(2)),im*inv(sqrt(2))])
    plan=PID.SymmetricKetHamiltonianPlan(
        basis,spin.jx;rate=0.7,hbar=1.3)
    @test_throws ArgumentError PID.SymmetricKetHamiltonianPlan(
        basis,spin.jx;memory_budget=1)
    @test issparse(plan.hamiltonian)
    @test size(plan)==(length(initial),length(initial))
    expected_block=collective_block(
        basis,spin.jx,only(basis.sectors))
    @test Matrix(plan.hamiltonian)≈expected_block atol=2e-15

    image=similar(initial.data)
    PID.apply_symmetric_hamiltonian!(
        image,plan,initial.data,0.0,nothing)
    @test image≈(-im*(0.7/1.3)).*expected_block*initial.data atol=3e-15
    @test_throws ArgumentError PID.apply_symmetric_hamiltonian!(
        initial.data,plan,initial.data)

    observable=collective_spin(basis,:z)
    scratch=similar(initial.data)
    direct_expectation=PID.symmetric_ket_expectation(
        initial,observable;workspace=scratch)
    @test direct_expectation≈dot(
        initial.data,coefficient_block(
            observable,only(basis.sectors))*initial.data) atol=2e-15
    @test PID.symmetric_ket_expectation(initial,plan;workspace=scratch)≈
        dot(initial.data,expected_block*initial.data) atol=2e-15

    direct_plan=PID.SymmetricKetHamiltonianPlan(observable;rate=0.4)
    @test Matrix(direct_plan.hamiltonian)≈
        coefficient_block(observable,only(basis.sectors))
    block_plan=PID.SymmetricKetHamiltonianPlan(
        basis,sparse(expected_block);representation=:block)
    @test Matrix(block_plan.hamiltonian)≈expected_block
    diagonal_plan=PID.SymmetricKetHamiltonianPlan(
        basis,Diagonal(diag(expected_block));representation=:block)
    @test diagonal_plan.hamiltonian isa Diagonal
    @test_throws DimensionMismatch PID.SymmetricKetHamiltonianPlan(
        basis,zeros(3,3);representation=:block)
    @test_throws ArgumentError PID.SymmetricKetHamiltonianPlan(
        basis,ComplexF64[0 im;im 0.2])

    rk_destination=copy(initial)
    rk_workspace=PID.SymmetricKetWorkspace(plan)
    @test_throws ArgumentError PID.SymmetricKetWorkspace(
        plan;memory_budget=1)
    @test_throws ArgumentError PID.evolve_symmetric_ket!(
        copy(initial),plan,initial,(0.0,0.1);
        workspace=rk_workspace,memory_budget=1)
    @test rk_workspace!==PID.SymmetricKetWorkspace(plan)
    aliased_workspace=PID.SymmetricKetWorkspace(
        plan,initial.data,similar(initial.data),similar(initial.data))
    @test_throws ArgumentError PID.evolve_symmetric_ket!(
        copy(initial),plan,initial,(0.0,0.1);workspace=aliased_workspace)
    PID.evolve_symmetric_ket!(
        rk_destination,plan,initial,(0.0,0.2);
        steps=256,workspace=rk_workspace)
    exact=exp((-im*0.2*(0.7/1.3)).*expected_block)*initial.data
    @test rk_destination.data≈exact atol=3e-12 rtol=3e-12
    @test norm(rk_destination.data)≈1 atol=3e-12
    @test_throws ArgumentError PID.time_evolve_symmetric_ket(
        plan,initial,(0.0,0.2);memory_budget=1)
    @test_throws ArgumentError PID.time_evolve_symmetric_ket(
        plan,initial,(0.0,20.0);steps=1)

    krylov_destination=copy(initial)
    krylov_workspace=KrylovExpvWorkspace(
        ComplexF64,length(initial),length(initial))
    result=PID.krylov_evolve_symmetric_ket!(
        krylov_destination,plan,initial,0.2;
        workspace=krylov_workspace,atol=1e-13,rtol=1e-13)
    @test result.converged
    @test result.state===krylov_destination
    @test krylov_destination.data≈exact atol=2e-12 rtol=2e-12
    @test_throws ArgumentError PID.krylov_evolve_symmetric_ket!(
        copy(initial),plan,initial,0.2;memory_budget=1)
    @test_throws ArgumentError PID.krylov_time_evolve_symmetric_ket(
        plan,initial,0.2;memory_budget=1)
    @test_throws ArgumentError PID.symmetric_ket_expectation(
        initial,observable;memory_budget=1)

    driven=PID.SymmetricKetHamiltonianPlan(
        basis,spin.jz;rate=(time,parameters)->parameters*time)
    driven_output=PID.time_evolve_symmetric_ket(
        driven,initial,(0.0,0.1);steps=32,parameters=0.2)
    @test all(isfinite,driven_output.data)
    @test_throws ArgumentError PID.krylov_evolve_symmetric_ket!(
        copy(initial),driven,initial,0.1)

    basis32=PIBasis(3,2;sectors=[(3,0)])
    spin32=spin_matrices(2;T=Float32)
    plan32=PID.SymmetricKetHamiltonianPlan(
        basis32,spin32.jz;rate=0.3f0)
    state32=PID.symmetric_occupation_ket(
        basis32,(2,1);T=Float32)
    image32=similar(state32.data)
    PID.apply_symmetric_hamiltonian!(image32,plan32,state32.data,0f0,nothing)
    @test eltype(image32)===ComplexF32

    setprecision(BigFloat,160) do
        spin_big=spin_matrices(2;T=BigFloat)
        plan_big=PID.SymmetricKetHamiltonianPlan(
            basis32,spin_big.jz;rate=BigFloat("0.3"))
        state_big=PID.symmetric_occupation_ket(
            basis32,(2,1);T=BigFloat)
        image_big=similar(state_big.data)
        PID.apply_symmetric_hamiltonian!(
            image_big,plan_big,state_big.data,zero(BigFloat),nothing)
        @test eltype(image_big)===Complex{BigFloat}
        @test all(isfinite,image_big)
    end

    plan_big,state_big=setprecision(BigFloat,160) do
        spin_big=spin_matrices(2;T=BigFloat)
        (PID.SymmetricKetHamiltonianPlan(basis32,spin_big.jz),
         PID.symmetric_occupation_ket(basis32,(2,1);T=BigFloat))
    end
    setprecision(BigFloat,256) do
        image_big=similar(state_big.data)
        PID.apply_symmetric_hamiltonian!(
            image_big,plan_big,state_big.data,big"0",nothing)
        @test all(value->precision(real(value))==160&&
                         precision(imag(value))==160,image_big)
        expectation=PID.symmetric_ket_expectation(state_big,plan_big)
        @test expectation isa Complex{BigFloat}
        @test precision(real(expectation))==160
        @test precision(imag(expectation))==160
    end
    diagonal_big=setprecision(BigFloat,160) do
        Diagonal(Complex{BigFloat}.(0:length(state_big)-1))
    end
    diagonal_big_plan=setprecision(BigFloat,256) do
        PID.SymmetricKetHamiltonianPlan(
            basis32,diagonal_big;representation=:block)
    end
    @test diagonal_big_plan.hamiltonian isa Diagonal
    @test diagonal_big_plan.precision_bits==160

    wide_rate=setprecision(BigFloat,256) do
        BigFloat("0.3")
    end
    promoted_plan=setprecision(BigFloat,160) do
        spin_big=spin_matrices(2;T=BigFloat)
        PID.SymmetricKetHamiltonianPlan(
            basis32,spin_big.jz;rate=wide_rate)
    end
    @test promoted_plan.precision_bits==256
    @test all(value->precision(real(value))==256&&
                     precision(imag(value))==256,
              nonzeros(promoted_plan.hamiltonian))

    scheduled_plan,scheduled_state=setprecision(BigFloat,160) do
        spin_big=spin_matrices(2;T=BigFloat)
        (PID.SymmetricKetHamiltonianPlan(
             basis32,spin_big.jz;rate=(time,parameters)->wide_rate),
         PID.symmetric_occupation_ket(basis32,(2,1);T=BigFloat))
    end
    @test_throws ArgumentError PID.apply_symmetric_hamiltonian!(
        similar(scheduled_state.data),scheduled_plan,scheduled_state.data,
        big"0",nothing)
    wide_block=setprecision(BigFloat,256) do
        Matrix{Complex{BigFloat}}(I,length(scheduled_state),
                                  length(scheduled_state))
    end
    @test_throws ArgumentError PID.symmetric_ket_expectation(
        scheduled_state,wide_block)
end

@testset "ket-native local-factor trace" begin
    PID=_SYMMETRIC_KET_PID
    basis=PIBasis(3,4;sectors=[(3,0,0,0)])
    local_vector=ComplexF64[1+im,2-im,-1+2im,3]
    local_vector./=norm(local_vector)
    ket=PID.symmetric_product_ket(basis,local_vector)

    for traced_factor in (1,2)
        plan=LocalFactorTracePlan(
            basis,(2,2);traced_factor,memory_budget=Inf)
        workspace=LocalFactorTraceWorkspace(plan)
        direct=PID.local_factor_trace(
            ket,plan;workspace,check=true)
        density_reference=local_factor_trace(
            PID.symmetric_ket_density(ket),plan;check=true)
        @test direct.basis===plan.output_basis
        @test direct.data≈density_reference.data atol=3e-13 rtol=3e-13

        destination=PIState(plan.output_basis;T=Float64)
        PID.local_factor_trace!(
            destination,ket,plan,workspace;
            check=false,memory_budget=0)
        @test destination.data≈direct.data atol=3e-13 rtol=3e-13
    end

    other=PIBasis(3,4;sectors=[(3,0,0,0)])
    wrong_plan=LocalFactorTracePlan(
        other,(2,2);memory_budget=Inf)
    @test_throws ArgumentError PID.local_factor_trace(ket,wrong_plan)

    # N independent Bell pairs are globally permutation symmetric. Tracing
    # one internal qubit leaves the other N-qubit factor maximally mixed, so
    # its algebraic entanglement entropy is exactly N bits. This exercises the
    # paper's ket-native use case without constructing the g^2 source state.
    bell=ComplexF64[inv(sqrt(2)),0,0,inv(sqrt(2))]
    bell_ket=PID.symmetric_product_ket(basis,bell)
    bell_plan=LocalFactorTracePlan(
        basis,(2,2);traced_factor=2,memory_budget=Inf)
    bell_reduced=PID.local_factor_trace(bell_ket,bell_plan)
    @test von_neumann_entropy(bell_reduced)≈basis.N atol=5e-12

    # A required outer-product entry below the destination scalar range must
    # raise rather than disappear silently in the ket-native contraction.
    tiny_basis=PIBasis(1,4;sectors=[(1,0,0,0)])
    tiny=PID.SymmetricKet(
        tiny_basis,ComplexF64[1,1e-200,0,0])
    tiny_plan=LocalFactorTracePlan(
        tiny_basis,(2,2);traced_factor=2,memory_budget=Inf)
    @test_throws ArgumentError PID.local_factor_trace(tiny,tiny_plan)
    @test_throws ArgumentError PID._symmetric_ket_checked_triple_product(
        ComplexF64(1,1e-200),ComplexF64(1,0),ComplexF64(1e-200,0),
        "component-underflow regression")
    endpoint_factor=ComplexF64(ldexp(1.0,-537),0)
    @test_throws ArgumentError PID._symmetric_ket_checked_triple_product(
        endpoint_factor,endpoint_factor,ComplexF64(0.75,0),
        "rounded-endpoint regression")
    cancelled_left=ComplexF64(1.7870584837762855,0.9597871919475343)
    cancelled_right=ComplexF64(0.6094763586552499,1.134803534089545)
    cancelled_native=cancelled_left*cancelled_right
    @test PID._symmetric_ket_complex_product_requires_wide(
        cancelled_left,cancelled_right,cancelled_native)
    cancelled_checked=PID._symmetric_ket_checked_triple_product(
        cancelled_left,cancelled_right,ComplexF64(1,0),
        "cancellation-error regression")
    @test_throws ArgumentError PID._symmetric_ket_checked_triple_product(
        cancelled_left,cancelled_right,ComplexF64(1,0),
        "certification-budget regression",0)
    cancelled_expected=setprecision(BigFloat,256) do
        Float64(real(Complex{BigFloat}(cancelled_left)*
                     Complex{BigFloat}(cancelled_right)))
    end
    @test real(cancelled_checked)==cancelled_expected
    conjugate_value=ComplexF64(0.3,-0.7)
    @test PID._symmetric_ket_checked_triple_product(
        conjugate_value,conj(conjugate_value),ComplexF64(1,0),
        "exact-zero cancellation",0)==ComplexF64(abs2(conjugate_value),0)
    setprecision(BigFloat,256) do
        delta=ldexp(one(BigFloat),-255)
        value=PID._symmetric_ket_checked_triple_product(
            complex(one(BigFloat),one(BigFloat)-delta),
            complex(one(BigFloat),one(BigFloat)+delta),
            complex(one(BigFloat),zero(BigFloat)),
            "deep-cancellation regression")
        @test real(value)==delta^2
        @test imag(value)==BigFloat(2)
    end
end
