@testset "matrix-RHS conditional trajectory kernels" begin
    PID=PermutationalInvariantDynamics
    @test isdefined(PID,:BatchedConditionalPlan)

    sm=ComplexF64[0 1;0 0]
    sx=ComplexF64[0 1;1 0]
    basis=PIBasis(3,2)
    model=PIModel(basis,(
        LocalHamiltonian(sx;rate=0.23),
        LocalJump(sm;rate=0.71),
        CollectiveJump(sm;rate=0.08),
    ))
    trajectory_plan=TrajectoryPlan(model)
    plan=PID.BatchedConditionalPlan(trajectory_plan)
    states=(
        iid_pure_state(basis,ComplexF64[0,1]),
        iid_pure_state(basis,ComplexF64[1,0]),
        iid_pure_state(basis,normalize(ComplexF64[1,1])),
    )
    X=hcat((state.data for state in states)...)
    work=PID.BatchedConditionalWorkspace(plan,states[1],3)

    @test size(plan)==(length(basis),length(basis))
    @test work.capacity==3
    @test size(work.tmp)==(length(basis),3)
    @test work.liouvillian_work.batch.capacity==3
    @test all(matrix->size(matrix)==(length(basis),3),
        (work.tmp,work.k1,work.k2,work.gather,work.gain))
    @test length(unique(objectid.(
        (work.tmp,work.k1,work.k2,work.gather,work.gain))))==5
    exact_budget=PID._batched_conditional_workspace_bytes(
        plan,eltype(states[1].data),3)
    exact_budget_work=PID.BatchedConditionalWorkspace(
        plan,states[1],3;memory_budget=exact_budget)
    @test exact_budget_work.capacity==3
    @test_throws ArgumentError PID.BatchedConditionalWorkspace(
        plan,states[1],3;memory_budget=1)
    @test_throws ArgumentError PID.BatchedConditionalWorkspace(
        plan,states[1],true)
    @test_throws ArgumentError PID.BatchedConditionalWorkspace(
        plan,states[1],big(typemax(Int))+1;memory_budget=Inf)

    Y=similar(X)
    lambdas=zeros(3)
    PID.batched_conditional_action!(Y,lambdas,plan,X,0.17,nothing,work)
    expected=similar(X)
    expected_lambda=zeros(3)
    for column in axes(X,2)
        scalar=TrajectoryWorkspace(trajectory_plan,states[1];mode=:fixed)
        PID._reset_effective_jump_cache!(scalar)
        expected_lambda[column]=PID._conditional_action_and_intensity!(
            view(expected,:,column),view(X,:,column),scalar,basis,
            0.17,nothing,trajectory_plan.liouvillian.tracevec)
    end
    @test Y≈expected atol=3e-14 rtol=3e-14
    @test lambdas≈expected_lambda atol=3e-14 rtol=3e-14

    # The matrix-RHS RK4 path is mathematically identical to repeated scalar
    # fixed steps. It may differ by BLAS accumulation roundoff only.
    batched_state=copy(X)
    scalar_state=copy(X)
    PID.batched_conditional_rk4!(
        batched_state,plan,0.17,0.004,nothing,work)
    for column in axes(X,2)
        scalar=TrajectoryWorkspace(trajectory_plan,states[1];mode=:fixed)
        PID._reset_effective_jump_cache!(scalar)
        PID._conditional_rk4!(
            view(scalar_state,:,column),scalar,basis,0.17,0.004,nothing,
            trajectory_plan.liouvillian.tracevec)
    end
    @test batched_state≈scalar_state atol=2e-13 rtol=2e-13
    @test all(abs(dot(trajectory_plan.liouvillian.tracevec,
                      view(batched_state,:,column))-1)<3e-14
              for column in axes(batched_state,2))

    rates=zeros(length(trajectory_plan.jumps),size(X,2))
    PID.batched_channel_intensities!(
        rates,plan,batched_state,0.174,nothing,work)
    expected_rates=similar(rates)
    for column in axes(X,2)
        scalar=TrajectoryWorkspace(trajectory_plan,states[1];mode=:fixed)
        expected_rates[:,column].=PID._channel_intensities!(
            scalar,view(batched_state,:,column),basis,0.174,nothing)
    end
    @test rates≈expected_rates atol=3e-14 rtol=3e-14

    selected=[1,2,0]
    jumped=copy(batched_state)
    expected_jumped=copy(batched_state)
    PID.batched_apply_jumps!(
        jumped,selected,plan,0.174,nothing,work)
    for column in axes(expected_jumped,2)
        channel=selected[column]
        iszero(channel)&&continue
        scalar=TrajectoryWorkspace(trajectory_plan,states[1];mode=:fixed)
        PID._channel_intensities!(
            scalar,view(expected_jumped,:,column),basis,0.174,nothing)
        PID._apply_gain!(scalar.channel_gain,
            view(expected_jumped,:,column),trajectory_plan.jumps[channel],
            basis,scalar.jump_scales[channel],scalar.liouvillian_work)
        z=dot(trajectory_plan.liouvillian.tracevec,scalar.channel_gain)
        expected_jumped[:,column].=scalar.channel_gain./z
    end
    @test jumped≈expected_jumped atol=3e-14 rtol=3e-14
    @test all(abs(dot(trajectory_plan.liouvillian.tracevec,
                      view(jumped,:,column))-1)<3e-14
              for column in axes(jumped,2))

    # The helper creates index-stable streams and consumes exactly the scalar
    # fixed-step sequence: one decision draw and, only on a jump, one channel
    # draw.
    rngs=PID.batched_trajectory_rngs(91,3)
    reference_rngs=PID.batched_trajectory_rngs(91,3)
    decisions=zeros(Int,3)
    PID.batched_sample_jumps!(decisions,rates,0.004,rngs)
    expected_decisions=zeros(Int,3)
    for column in axes(rates,2)
        lambda=sum(view(rates,:,column))
        probability=-expm1(-lambda*0.004)
        if lambda>0&&rand(reference_rngs[column],Float64)<probability
            expected_decisions[column]=PID._select_jump_channel(
                view(rates,:,column),
                rand(reference_rngs[column],Float64)*lambda)
        end
    end
    @test decisions==expected_decisions
    @test [rand(rng,UInt64) for rng in rngs]==
          [rand(rng,UInt64) for rng in reference_rngs]
    @test_throws ArgumentError PID.batched_trajectory_rngs(91,true)
    @test_throws ArgumentError PID.batched_trajectory_rngs(
        91,big(typemax(Int))+1)
    @test_throws ArgumentError PID.batched_sample_jumps!(
        falses(3),rates,0.004,rngs)
    @test_throws ArgumentError PID.batched_sample_jumps!(
        zeros(Int,3),rates,true,rngs)
    overflowing_rates=fill(floatmax(Float64),2,3)
    @test_throws ArgumentError PID.batched_sample_jumps!(
        zeros(Int,3),overflowing_rates,0.004,rngs)

    @test_throws ArgumentError PID.batched_conditional_action!(
        similar(hcat(X,X)),zeros(6),plan,hcat(X,X),0.0,nothing,work)
    @test_throws ArgumentError PID.batched_apply_jumps!(
        copy(X),BitVector((true,false,false)),plan,0.0,nothing,work)
    @test_throws ArgumentError PID.batched_apply_jumps!(
        copy(X),[1,3,0],plan,0.0,nothing,work)
    @test_throws DimensionMismatch PID.batched_channel_intensities!(
        zeros(1,3),plan,X,0.0,nothing,work)

    # A driven rate reused at the same physical time with another parameter
    # must be reevaluated rather than served from a stale node cache.
    driven_model=PIModel(basis,(
        LocalJump(sm;rate=(time,parameters)->parameters.rate),))
    driven_plan=PID.BatchedConditionalPlan(TrajectoryPlan(driven_model))
    driven_work=PID.BatchedConditionalWorkspace(
        driven_plan,states[1],3)
    driven_Y=similar(X)
    driven_lambda=zeros(3)
    PID.batched_conditional_action!(
        driven_Y,driven_lambda,driven_plan,X,0.2,(rate=0.1,),driven_work)
    first_lambda=copy(driven_lambda)
    PID.batched_conditional_action!(
        driven_Y,driven_lambda,driven_plan,X,0.2,(rate=0.4,),driven_work)
    @test driven_lambda≈4first_lambda atol=3e-14 rtol=3e-14

    # Exercise the Appendix-D rectangular gain batch as well as the one-body
    # and collective gain variants above.
    pbody_basis=PIBasis(2,2)
    pbody_state=iid_pure_state(pbody_basis,ComplexF64[0,1])
    pbody_trajectory=TrajectoryPlan(PIModel(pbody_basis,(
        LocalPBodyJump(kron(sm,sm),2;rate=0.3),)))
    pbody_plan=PID.BatchedConditionalPlan(pbody_trajectory)
    pbody_work=PID.BatchedConditionalWorkspace(
        pbody_plan,pbody_state,2)
    pbody_states=repeat(reshape(pbody_state.data,:,1),1,2)
    pbody_expected=copy(pbody_states)
    PID.batched_apply_jumps!(
        pbody_states,[1,1],pbody_plan,0.0,nothing,pbody_work)
    for column in axes(pbody_expected,2)
        scalar=TrajectoryWorkspace(
            pbody_trajectory,pbody_state;mode=:fixed)
        PID._channel_intensities!(
            scalar,view(pbody_expected,:,column),pbody_basis,0.0,nothing)
        PID._apply_gain!(scalar.channel_gain,
            view(pbody_expected,:,column),only(pbody_trajectory.jumps),
            pbody_basis,only(scalar.jump_scales),
            scalar.liouvillian_work)
        z=dot(pbody_trajectory.liouvillian.tracevec,scalar.channel_gain)
        pbody_expected[:,column].=scalar.channel_gain./z
    end
    @test pbody_states≈pbody_expected atol=3e-14 rtol=3e-14

    # After warm-up, a full matrix conditional action must reuse all retained
    # storage. Keep a tiny allowance for version-dependent view wrappers.
    PID.batched_conditional_action!(Y,lambdas,plan,X,0.17,nothing,work)
    allocated=@allocated PID.batched_conditional_action!(
        Y,lambdas,plan,X,0.17,nothing,work)
    @test allocated<=1024
    allocation_state=copy(X)
    PID.batched_conditional_rk4!(
        allocation_state,plan,0.17,0.004,nothing,work)
    rk4_allocated=@allocated PID.batched_conditional_rk4!(
        allocation_state,plan,0.174,0.004,nothing,work)
    @test rk4_allocated<=1024
    PID.batched_channel_intensities!(
        rates,plan,allocation_state,0.178,nothing,work)
    rates_allocated=@allocated PID.batched_channel_intensities!(
        rates,plan,allocation_state,0.178,nothing,work)
    @test rates_allocated<=1024
    copyto!(allocation_state,X)
    PID.batched_apply_jumps!(
        allocation_state,[1,0,0],plan,0.17,nothing,work)
    copyto!(allocation_state,X)
    jump_allocated=@allocated PID.batched_apply_jumps!(
        allocation_state,[1,0,0],plan,0.17,nothing,work)
    @test jump_allocated<=1024
end
