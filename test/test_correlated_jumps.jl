@testset "correlated Kossakowski jump reservoirs" begin
    PID=PermutationalInvariantDynamics
    basis=PIBasis(3,2)
    sm=ComplexF64[0 1;0 0]
    sz=ComplexF64[1 0;0 -1]
    mixing=ComplexF64[1.0 0.2im;0.3 0.5]
    gamma=mixing*adjoint(mixing)
    rate=0.37

    for (correlated,independent) in (
            (CorrelatedLocalJumps,LocalJump),
            (CorrelatedCollectiveJumps,CollectiveJump))
        term=correlated((sm,sz),gamma;rate=rate)
        model=PIModel(basis,(term,))
        sparse_generator=liouvillian(model;representation=:sparse)
        effective=ntuple(channel->
            mixing[1,channel]*sm+mixing[2,channel]*sz,2)
        reference=liouvillian(PIModel(basis,
            map(operator->independent(operator;rate=rate),effective));
            representation=:sparse)
        @test Matrix(sparse_generator)≈Matrix(reference) atol=3e-12 rtol=3e-12

        matrixfree=liouvillian(model;representation=:matrixfree)
        rng=MersenneTwister(6301)
        input=randn(rng,ComplexF64,length(basis));output=similar(input)
        mul!(output,matrixfree,input)
        @test output≈sparse_generator*input atol=3e-12 rtol=3e-12
        apply_adjoint!(output,matrixfree,input)
        @test output≈adjoint(sparse_generator)*input atol=4e-12 rtol=4e-12
        @test length(TrajectoryPlan(model).jumps)==length(term.factor)
    end

    # The off-diagonal Kossakowski entries generate genuine interference;
    # replacing Gamma by its diagonal changes both local and collective maps.
    diagonal_gamma=Matrix(Diagonal(diag(gamma)))
    for correlated in (CorrelatedLocalJumps,CorrelatedCollectiveJumps)
        full=liouvillian(PIModel(basis,
            (correlated((sm,sz),gamma),));representation=:sparse)
        diagonal=liouvillian(PIModel(basis,
            (correlated((sm,sz),diagonal_gamma),));representation=:sparse)
        @test norm(full-diagonal)>1e-3
    end

    # Construction copies fixed input and retains one factorization. Mutating
    # either caller-owned input after construction cannot change the term.
    copied_gamma=copy(gamma);copied_sm=copy(sm)
    copied_term=CorrelatedLocalJumps((copied_sm,sz),copied_gamma)
    factor_matrix=hcat(copied_term.factor...)
    reconstructed=factor_matrix*adjoint(factor_matrix)
    @test reconstructed≈gamma atol=3e-14 rtol=3e-14
    fill!(copied_gamma,0);fill!(copied_sm,0)
    copied_model=PIModel(basis,(copied_term,))
    @test norm(liouvillian(copied_model;representation=:sparse))>0

    # A zero PSD matrix remains a typed zero channel rather than producing an
    # empty, Float64-defaulted plan.
    zero_term=CorrelatedLocalJumps((ComplexF32.(sm),ComplexF32.(sz)),
                                   zeros(ComplexF32,2,2);rate=1f0)
    zero_plan=LiouvillianPlan(PIModel(basis,(zero_term,)))
    @test eltype(zero_plan)===ComplexF32
    @test iszero(norm(liouvillian(PIModel(basis,(zero_term,));
                                  representation=:sparse)))

    gamma32=ComplexF32.(gamma)
    term32=CorrelatedCollectiveJumps((ComplexF32.(sm),ComplexF32.(sz)),
                                     gamma32;rate=0.4f0)
    plan32=@inferred LiouvillianPlan(PIModel(basis,(term32,)))
    @test eltype(plan32)===ComplexF32
    schedule32=InPlaceTimeOperator(gamma32,(destination,t,p)->nothing)
    dynamic_plan32=@inferred LiouvillianPlan(PIModel(basis,(
        CorrelatedLocalJumps((ComplexF32.(sm),ComplexF32.(sz)),schedule32;
                             rate=0.4f0),)))
    @test eltype(dynamic_plan32)===ComplexF32
    risk_basis=PIBasis(34,2)
    risk_schedule=InPlaceTimeOperator(Matrix{Float16}(I,2,2),
                                      (destination,t,p)->nothing)
    risk_error=try
        LiouvillianPlan(PIModel(risk_basis,(
            CorrelatedLocalJumps((Float16.(real.(sm)),Float16.(real.(sz))),
                                 risk_schedule),)))
        nothing
    catch caught
        caught
    end
    @test risk_error isa ArgumentError
    @test occursin("wider InPlaceTimeOperator prototype",
                   sprint(showerror,risk_error))
    wider_rate_plan=LiouvillianPlan(PIModel(basis,(
        CorrelatedCollectiveJumps((ComplexF32.(sm),ComplexF32.(sz)),gamma32;
                                  rate=(t,p)->0.4),)))
    input32=ComplexF32.(randn(MersenneTwister(6300),ComplexF64,length(basis)))
    @test_throws ArgumentError apply!(similar(input32),wider_rate_plan,input32,
        0.2f0,nothing,LiouvillianWorkspace(wider_rate_plan))

    # The generic residual Cholesky route supports wider scalar types without
    # requiring a LAPACK eigensolver.
    gamma_big=BigFloat[1 big"0.25";big"0.25" 1]
    big_term=CorrelatedCollectiveJumps((BigFloat.(real.(sm)),
                                        BigFloat.(real.(sz))),gamma_big)
    big_factor=hcat(big_term.factor...)
    @test big_factor*adjoint(big_factor)≈gamma_big

    @test_throws ArgumentError CorrelatedLocalJumps(sm,gamma)
    @test_throws ArgumentError CorrelatedLocalJumps((),zeros(0,0))
    @test_throws DimensionMismatch CorrelatedLocalJumps((sm,zeros(3,3)),gamma)
    @test_throws ArgumentError CorrelatedLocalJumps((sm,
        ComplexF64[Inf 0;0 1]),gamma)
    @test_throws DimensionMismatch CorrelatedLocalJumps((sm,sz),ones(3,3))
    @test_throws ArgumentError CorrelatedLocalJumps((sm,sz),
        ComplexF64[1 0.2;0.3 1])
    @test_throws ArgumentError CorrelatedCollectiveJumps((sm,sz),
        ComplexF64[1 2;2 1])
    @test_throws ArgumentError CorrelatedCollectiveJumps((sm,sz),
        1e-300*ComplexF64[1 2;2 1])
    @test_throws ArgumentError CorrelatedLocalJumps((sm,sz),
        ComplexF64[1 Inf;Inf 1])
    @test_throws ArgumentError CorrelatedLocalJumps((sm,sz),gamma;rate=1im)
    @test_throws ArgumentError CorrelatedLocalJumps((sm,sz),gamma;rate=Inf)
    @test_throws ArgumentError CorrelatedLocalJumps((sm,sz),gamma;atol=-1)
    @test_throws DimensionMismatch PIModel(PIBasis(2,3),
        (CorrelatedLocalJumps((sm,sz),gamma),))
    symmetric_basis=PIBasis(3,2;sectors=[(3,0)])
    @test PIModel(symmetric_basis,
        (CorrelatedCollectiveJumps((sm,sz),gamma),)) isa PIModel
    @test_throws ArgumentError PIModel(symmetric_basis,
        (CorrelatedLocalJumps((sm,sz),gamma),))

    time=0.23
    calls=Ref(0)
    schedule=InPlaceTimeOperator(gamma,(destination,t,p)->begin
        calls[]+=1
        scale=one(t)+t
        @inbounds for index in eachindex(destination)
            destination[index]*=scale
        end
        nothing
    end)
    rng=MersenneTwister(6302)
    input=randn(rng,ComplexF64,length(basis));output=similar(input)
    inputs=randn(rng,ComplexF64,length(basis),3);outputs=similar(inputs)
    for correlated in (CorrelatedLocalJumps,CorrelatedCollectiveJumps)
        model=PIModel(basis,(correlated((sm,sz),schedule;rate=rate),))
        plan=LiouvillianPlan(model);workspace=LiouvillianWorkspace(plan)
        @test plan.kernels!==nothing
        @test !isautonomous(plan)
        if correlated===CorrelatedLocalJumps
            kernel=only(plan.kernels)
            prepared=only(workspace.kernel_workspaces)
            @test !hasfield(typeof(kernel),:I)
            @test !hasfield(typeof(kernel),:J)
            @test !hasfield(typeof(prepared),:values)
            @test length(prepared.contractions)==
                  length(kernel.branches.entries)*length(kernel.operators)
        end
        before=calls[]
        apply!(output,plan,input,time,nothing,workspace)
        @test calls[]==before+1
        frozen=freeze(model;time=time,representation=:sparse)
        @test output≈frozen*input atol=4e-12 rtol=4e-12
        apply_adjoint!(output,plan,input,time,nothing,workspace)
        @test output≈adjoint(frozen)*input atol=5e-12 rtol=5e-12
        before=calls[]
        apply!(outputs,plan,inputs,time,nothing,workspace)
        @test calls[]==before+1
        @test outputs≈frozen*inputs atol=5e-12 rtol=5e-12
        apply_adjoint!(outputs,plan,inputs,time,nothing,workspace)
        @test outputs≈adjoint(frozen)*inputs atol=6e-12 rtol=6e-12
        apply!(output,plan,input,time,nothing,workspace) # warm allocation path
        @test (@allocated apply!(output,plan,input,time,nothing,workspace))<=512
        apply!(outputs,plan,inputs,time,nothing,workspace)
        @test (@allocated apply!(outputs,plan,inputs,time,nothing,
                                 workspace))<=2048
        apply_adjoint!(outputs,plan,inputs,time,nothing,workspace)
        @test (@allocated apply_adjoint!(outputs,plan,inputs,time,nothing,
                                         workspace))<=2048
    end

    raw_calls=Ref(0)
    raw_matrix=(t,p)->begin
        raw_calls[]+=1
        p.scale*gamma
    end
    raw_model=PIModel(basis,
        (CorrelatedLocalJumps((sm,sz),raw_matrix;rate=rate),))
    raw_plan=LiouvillianPlan(raw_model)
    @test raw_plan.kernels===nothing
    apply!(outputs,raw_plan,inputs,time,(scale=1.4,),
           LiouvillianWorkspace(raw_plan))
    @test raw_calls[]==1
    raw_frozen=freeze(raw_model;time=time,parameters=(scale=1.4,),
                      representation=:sparse)
    @test outputs≈raw_frozen*inputs atol=5e-12 rtol=5e-12

    invalid_schedule=InPlaceTimeOperator(gamma,(destination,t,p)->begin
        destination[1,2]=2
        destination[2,1]=2
        nothing
    end)
    invalid_plan=LiouvillianPlan(PIModel(basis,
        (CorrelatedCollectiveJumps((sm,sz),invalid_schedule),)))
    @test_throws ArgumentError apply!(output,invalid_plan,input,time,nothing,
                                      LiouvillianWorkspace(invalid_plan))
    nonfinite_rate=PIModel(basis,(CorrelatedLocalJumps((sm,sz),gamma;
        rate=(t,p)->Inf),))
    nonfinite_plan=LiouvillianPlan(nonfinite_rate)
    @test_throws ArgumentError apply!(output,nonfinite_plan,input,time,nothing,
                                      LiouvillianWorkspace(nonfinite_plan))
end
