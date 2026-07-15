@testset "partitions, dimensions, GT patterns" begin
    for N in 0:8, d in 1:4
        ps=partitions(N,d)
        @test all(p->weight(p)==N,ps)
        @test sum(unitary_group_dimension(p)^2 for p in ps)==commutant_dimension(N,d)
        for p in ps
            @test BigInt(length(gt_patterns(p)))==unitary_group_dimension(p)
            @test all(PermutationalInvariantDynamics.isvalid,gt_patterns(p))
            @test all(r->add_corner(remove_corner(p,r),r)==p,removable_corners(p))
        end
    end
    @test symmetric_group_dimension(Partition((3,1)))==3
    @test unitary_group_dimension(Partition((3,1)))==3
end

@testset "exact and scaled combinatorial factors" begin
    PID=PermutationalInvariantDynamics

    @test exact_binomial(200,100)==binomial(big(200),big(100))
    @test exact_binomial(5,-1)==0
    @test exact_binomial(5,6)==0
    @test exact_binomial(200,100) isa BigInt
    @test_throws ArgumentError exact_binomial(-1,0)

    @test exact_multinomial(())==1
    @test exact_multinomial((2,1,1))==12
    @test exact_multinomial((100,100))==exact_binomial(200,100)
    @test exact_multinomial((count for count in (2,1,1)))==12
    @test exact_multinomial((100,100)) isa BigInt
    @test_throws ArgumentError exact_multinomial((2,-1,1))
    @test_throws ArgumentError exact_multinomial((2,1.5))

    # Small values exercise the direct exact-ratio path.
    @test PID._checked_exact_ratio(Float32,3,2)==1.5f0
    @test PID._checked_sqrt_exact_ratio(Float32,9,4)==1.5f0

    # Numerator and denominator may each exceed the floating range even when
    # their final ratio is ordinary.  Likewise, the exact integer below is too
    # large for Float16/Float32 while its square root remains representable.
    huge=big(10)^100
    @test PID._checked_exact_ratio(Float32,3huge,2huge)==1.5f0
    @test PID._checked_sqrt_exact_ratio(Float16,big(2)^30,1)==
          ldexp(one(Float16),15)
    @test PID._checked_sqrt_exact_ratio(Float32,big(2)^200,1)==
          ldexp(one(Float32),100)
    @test_throws ArgumentError PID._checked_exact_ratio(
        Float32,1,big(2)^200)
    @test_throws ArgumentError PID._checked_sqrt_exact_ratio(
        Float32,big(2)^300,1)
    @test_throws ArgumentError PID._checked_exact_ratio(Float16,65505,1)
    @test PID._checked_sqrt_exact_integer(Float16,65505)==sqrt(Float16(65504))

    # Exact factors are fused with their numerical value when neither factor
    # is representable separately. Arrays prepare that binary scale once.
    @test PID._checked_mul_exact_ratio(Float16(2)^-10,big(2)^20,1)==Float16(1024)
    @test PID._checked_mul_exact_ratio(
        Float16[Float16(2)^-10,Float16(2)^-11],big(2)^20,1)==
        Float16[1024,512]
    @test PID._checked_mul_sqrt_exact_ratio(
        Float32(2)^-100,big(2)^200,1)==1.0f0
    @test PID._checked_mul_exact_ratio(
        Float16(1024),1,big(2)^30)==Float16(2)^-20
    endpoint_scale=PID._prepare_exact_scale(
        Float16,3,big(2)^26,Val(false);context="endpoint oracle")
    @test_throws ArgumentError PID._apply_prepared_exact_scale_product(
        Float16(1),Float16(1),endpoint_scale;context="endpoint oracle")

    # One-box branching and complete removal paths must agree exactly with
    # their hook-dimension definitions.  These identities protect both the
    # small direct path and the large-multiplicity geometry fallback.
    for N in 1:7, d in 2:3, lambda in partitions(N,d)
        flambda=symmetric_group_dimension(lambda)
        for row in removable_corners(lambda)
            mu=remove_corner(lambda,row)
            expected=big(N)*symmetric_group_dimension(mu)//flambda
            @test PID._one_box_branch_weight(lambda,mu)==expected
        end
        for p in 1:min(N,3), path in PID._removal_paths(lambda,p)
            expected=exact_binomial(N,p)*
                     symmetric_group_dimension(first(path))//flambda
            @test PID._subset_path_weight(path)==expected
        end
    end
    extreme=Partition((typemax(Int),0))
    @test PID._one_box_branch_weight(
        extreme,Partition((typemax(Int)-1,0)))==big(typemax(Int))//1
end
