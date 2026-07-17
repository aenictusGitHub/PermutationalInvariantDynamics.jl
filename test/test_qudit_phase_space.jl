@testset "generalized qudit Husimi phase space" begin
    # One qutrit is the defining U(3) irrep, so Q(U)=3*|<psi|U|3>|^2.
    basis=PIBasis(1,3)
    rho=computational_product_state(basis,3)
    zero_generator=zeros(ComplexF64,3,3)
    swap_generator=zeros(ComplexF64,3,3)
    swap_generator[1,3]=swap_generator[3,1]=pi/2
    plan=QuditHusimiPlan(basis,(zero_generator,swap_generator);
                         representation=:generator)
    data=qudit_husimi_q(rho,plan;resolved=true)
    @test data.normalization===:normalized_haar
    @test data.metadata.measure===:normalized_haar
    @test data.values[1]≈3.0 atol=2e-13
    @test data.values[2]≈0.0 atol=2e-13
    @test data.sector_values[1,:]≈data.values
    @test data.populations≈[1.0]
    @test data.irrep_dimensions==[3]
    @test data.multiplicities==BigInt[1]
    @test occursin("QuditHusimiPlan",sprint(show,plan))
    @test occursin("QuditHusimiData",sprint(show,data))

    unitary_points=(Matrix{ComplexF64}(I,3,3),exp(-im*swap_generator))
    from_unitaries=qudit_husimi_q(rho,unitary_points;
        representation=:unitary,resolved=true)
    @test from_unitaries.values≈data.values atol=2e-13
    @test_throws ArgumentError QuditHusimiPlan(basis,2I;
        representation=:unitary)
    @test_throws ArgumentError QuditHusimiPlan(basis,[0.0 1 0;0 0 0;0 0 0];
        representation=:generator)
    @test_throws DimensionMismatch QuditHusimiPlan(basis,ones(2,2);
        representation=:generator)

    # Normalized-Haar qubit data differ from the sphere density only by 4pi.
    qubit_basis=PIBasis(1,2)
    north=computational_product_state(qubit_basis,2)
    spin=spin_matrices(2)
    theta=[0.0,0.4,1.2]
    phi=[0.0]
    rotations=[exp(-im*angle*spin.jy) for angle in theta]
    generalized=qudit_husimi_q(north,rotations;
        representation=:unitary)
    sphere=spin_husimi_q(north,theta,phi)
    @test generalized.values≈vec(sphere.values).*4pi atol=5e-12

    # General PI states are represented as a sum of independently normalized
    # sector densities, not as one effective irrep.
    multibasis=PIBasis(3,3)
    mixed=maximally_mixed_state(multibasis)
    multipoint=QuditHusimiPlan(multibasis,zero_generator;
        representation=:generator)
    resolved=qudit_husimi_q(mixed,multipoint;resolved=true)
    @test sum(resolved.populations)≈1.0 atol=2e-12
    @test resolved.values[1]≈sum(resolved.sector_values[:,1]) atol=2e-12
    @test all(isfinite,resolved.values)

    other_basis=PIBasis(1,3)
    @test_throws ArgumentError qudit_husimi_q(
        computational_product_state(other_basis,3),multipoint)

    # Float32 points and state keep Float32 numerical storage.
    basis32=PIBasis(1,3)
    rho32=computational_product_state(basis32,3;T=Float32)
    plan32=QuditHusimiPlan(basis32,zeros(ComplexF32,3,3);
        representation=:generator)
    data32=qudit_husimi_q(rho32,plan32)
    @test eltype(data32.values)===Float32

    # Explicit precision may widen floating coordinates, but must not narrow
    # them silently.  Exact coordinates do not impose a floating precision.
    @test_throws ArgumentError QuditHusimiPlan(
        basis32,zeros(Float64,3,3);representation=:generator,T=Float32)
    @test_throws ArgumentError QuditHusimiPlan(
        basis32,zeros(ComplexF64,3,3);representation=:generator,T=Float32)
    widened=QuditHusimiPlan(
        basis32,zeros(ComplexF32,3,3);representation=:generator,T=Float64)
    @test widened.real_type===Float64
    exact=QuditHusimiPlan(basis32,Diagonal([0//1,1//2,-1//2]);
        representation=:generator,T=Float32)
    @test exact.real_type===Float32
    @test eltype(qudit_husimi_q(rho32,exact).values)===Float32
end
