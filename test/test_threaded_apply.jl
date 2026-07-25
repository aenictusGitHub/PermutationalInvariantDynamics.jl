struct _ThreadedReadCountingVector{T,V<:AbstractVector{T}} <:
        AbstractVector{T}
    data::V
    reads::Base.RefValue{Int}
end
Base.size(vector::_ThreadedReadCountingVector)=size(vector.data)
Base.IndexStyle(::Type{<:_ThreadedReadCountingVector})=IndexLinear()
@inline function Base.getindex(
        vector::_ThreadedReadCountingVector,index::Int)
    vector.reads[]+=1
    vector.data[index]
end

@testset "deterministic target-sector Liouvillian threading" begin
    PID=PermutationalInvariantDynamics
    rng=MersenneTwister(0x71ea)
    sx=ComplexF64[0 1;1 0]
    sz=ComplexF64[1 0;0 -1]
    sm=ComplexF64[0 1;0 0]

    # Exercise every autonomous fixed-gain family after compile-time fusion:
    # collective, one-body local, and Appendix-D local p-body channels.
    basis=PIBasis(4,2)
    pair_jump=kron(sm,sm)
    model=PIModel(basis,(
        LocalHamiltonian(sx;rate=0.17),
        CollectiveHamiltonian(sz;rate=-0.03),
        CollectiveJump(sm;rate=0.13),
        LocalJump(sm+0.2sz;rate=0.11),
        LocalPBodyJump(pair_jump,2;rate=0.07),
    ))
    plan=LiouvillianPlan(model)
    @test only(plan.kernels) isa PID.FusedStaticPIKernel
    serial_work=LiouvillianWorkspace(plan)
    one_task=ThreadedLiouvillianWorkspace(plan;tasks=1)
    many_tasks=ThreadedLiouvillianWorkspace(plan;tasks=typemax(Int))
    @test length(one_task.assignments)==1
    @test length(many_tasks.assignments)==length(basis.sectors)
    @test sort(reduce(vcat,many_tasks.assignments))==
          collect(eachindex(basis.sectors))
    @test isempty(intersect((Set(group) for group in many_tasks.assignments)...))

    source=randn(rng,ComplexF64,length(basis))
    serial=similar(source);threaded=similar(source);repeated=similar(source)
    apply!(serial,plan,source,serial_work)
    threaded_apply!(threaded,plan,source,one_task)
    @test threaded==serial # the one-task route is the ordinary fast path
    threaded_apply!(threaded,plan,source,many_tasks)
    threaded_apply!(repeated,plan,source,many_tasks)
    @test threaded==serial
    @test repeated==threaded

    apply_adjoint!(serial,plan,source,serial_work)
    threaded_apply_adjoint!(threaded,plan,source,one_task)
    @test threaded==serial
    threaded_apply_adjoint!(threaded,plan,source,many_tasks)
    threaded_apply_adjoint!(repeated,plan,source,many_tasks)
    @test threaded==serial
    @test repeated==threaded

    # Multi-worker kernels share one caller-packed source instead of copying
    # the same Schur block once per term and worker.
    scheduled_rates=ntuple(4) do index
        (time,parameters)->(0.01index)*(one(time)+parameters.shift)
    end
    packing_model=PIModel(basis,ntuple(index->CollectiveHamiltonian(
        isodd(index) ? sx : sz;rate=scheduled_rates[index]),4))
    packing_plan=LiouvillianPlan(packing_model)
    @test length(packing_plan.kernels)==4
    packing_work=ThreadedLiouvillianWorkspace(packing_plan;tasks=3)
    packing_source=randn(rng,ComplexF64,length(basis))
    packing_reads=Ref(0)
    counted_source=_ThreadedReadCountingVector(
        packing_source,packing_reads)
    packing_output=similar(packing_source)
    threaded_apply!(packing_output,packing_plan,counted_source,0.2,
                    (shift=0.1,),packing_work)
    @test packing_reads[]==length(packing_source)
    packing_reads[]=0
    threaded_apply_adjoint!(
        packing_output,packing_plan,counted_source,0.2,
        (shift=0.1,),packing_work)
    @test packing_reads[]==length(packing_source)

    # The normal compiled-model flow delegates to the exact same immutable
    # plan and rejects a workspace prepared for any other plan object.
    compiled=compile(model;backend=:matrixfree,memory_budget=Inf)
    compiled_work=ThreadedLiouvillianWorkspace(compiled;tasks=3)
    threaded_apply!(threaded,compiled,source,compiled_work)
    apply!(serial,compiled,source)
    @test threaded==serial
    @test_throws ArgumentError threaded_apply!(threaded,compiled,source,
                                                many_tasks)

    # Dynamic operator and scalar schedules are evaluated once on the caller,
    # never once per worker or per sector.
    dynamic_basis=PIBasis(4,2)
    operator_calls=[Ref(0) for _ in 1:6]
    rate_calls=[Ref(0) for _ in 1:6]
    function scheduled(operator,index)
        InPlaceTimeOperator(operator,(destination,time,parameters)->begin
            operator_calls[index][]+=1
            @inbounds for entry in eachindex(destination,operator)
                destination[entry]=(one(time)+time)*operator[entry]
            end
            nothing
        end)
    end
    function rate(index,value)
        (time,parameters)->begin
            rate_calls[index][]+=1
            value*(one(time)+parameters.rate_shift)
        end
    end
    gamma=ComplexF64[1.0 0.2im;-0.2im 0.6]
    gamma_schedule=InPlaceTimeOperator(gamma,
        (destination,time,parameters)->begin
            operator_calls[5][]+=1
            @inbounds for entry in eachindex(destination,gamma)
                destination[entry]=(one(time)+time)*gamma[entry]
            end
            nothing
        end)
    gamma_collective_schedule=InPlaceTimeOperator(gamma,
        (destination,time,parameters)->begin
            operator_calls[6][]+=1
            @inbounds for entry in eachindex(destination,gamma)
                destination[entry]=(one(time)+time)*gamma[entry]
            end
            nothing
        end)
    dynamic_model=PIModel(dynamic_basis,(
        LocalHamiltonian(scheduled(sx,1);rate=rate(1,0.17)),
        CollectiveJump(scheduled(sm,2);rate=rate(2,0.13)),
        LocalJump(scheduled(sm+0.2sz,3);rate=rate(3,0.11)),
        LocalPBodyJump(scheduled(pair_jump,4),2;rate=rate(4,0.07)),
        CorrelatedLocalJumps((sm,sz),gamma_schedule;rate=rate(5,0.05)),
        CorrelatedCollectiveJumps((sm,sz),gamma_collective_schedule;
                                  rate=rate(6,0.04)),
    ))
    dynamic_plan=LiouvillianPlan(dynamic_model)
    dynamic_serial_work=LiouvillianWorkspace(dynamic_plan)
    dynamic_threaded_work=ThreadedLiouvillianWorkspace(dynamic_plan;tasks=3)
    dynamic_source=randn(rng,ComplexF64,length(dynamic_basis))
    dynamic_serial=similar(dynamic_source)
    dynamic_threaded=similar(dynamic_source)
    time=0.23;parameters=(rate_shift=0.08,)

    apply!(dynamic_serial,dynamic_plan,dynamic_source,time,parameters,
           dynamic_serial_work)
    foreach(counter->counter[]=0,operator_calls)
    foreach(counter->counter[]=0,rate_calls)
    threaded_apply!(dynamic_threaded,dynamic_plan,dynamic_source,time,
                    parameters,dynamic_threaded_work)
    @test getindex.(operator_calls)==ones(Int,length(operator_calls))
    @test getindex.(rate_calls)==ones(Int,length(rate_calls))
    @test dynamic_threaded==dynamic_serial

    apply_adjoint!(dynamic_serial,dynamic_plan,dynamic_source,time,parameters,
                   dynamic_serial_work)
    foreach(counter->counter[]=0,operator_calls)
    foreach(counter->counter[]=0,rate_calls)
    threaded_apply_adjoint!(dynamic_threaded,dynamic_plan,dynamic_source,time,
                            parameters,dynamic_threaded_work)
    @test getindex.(operator_calls)==ones(Int,length(operator_calls))
    @test getindex.(rate_calls)==ones(Int,length(rate_calls))
    @test dynamic_threaded==dynamic_serial

    # Fixed operators with scalar schedules retain unfused static kernels;
    # their rectangular one- and p-body factors use the same target ownership.
    static_rate_calls=Ref(0)
    scalar_rate=(time,parameters)->begin
        static_rate_calls[]+=1
        0.09*(one(time)+time)
    end
    scheduled_static_model=PIModel(basis,(
        LocalJump(sm+0.1sz;rate=scalar_rate),
        LocalPBodyJump(pair_jump,2;rate=0.04),
    ))
    scheduled_static_plan=LiouvillianPlan(scheduled_static_model)
    @test any(kernel->kernel isa PID.FactorizedLocalJumpPIKernel,
              scheduled_static_plan.kernels)
    scheduled_serial_work=LiouvillianWorkspace(scheduled_static_plan)
    scheduled_threaded_work=ThreadedLiouvillianWorkspace(
        scheduled_static_plan;tasks=3)
    apply!(serial,scheduled_static_plan,source,time,nothing,
           scheduled_serial_work)
    static_rate_calls[]=0
    threaded_apply!(threaded,scheduled_static_plan,source,time,nothing,
                    scheduled_threaded_work)
    @test static_rate_calls[]==1
    @test threaded==serial
    apply_adjoint!(serial,scheduled_static_plan,source,time,nothing,
                   scheduled_serial_work)
    static_rate_calls[]=0
    threaded_apply_adjoint!(threaded,scheduled_static_plan,source,time,nothing,
                            scheduled_threaded_work)
    @test static_rate_calls[]==1
    @test threaded==serial

    # Float32 plans preserve their prepared precision. A wider destination is
    # accepted without narrowing because arithmetic remains in plan scratch.
    basis32=PIBasis(5,2)
    model32=PIModel(basis32,(
        LocalHamiltonian(ComplexF32.(sx);rate=0.2f0),
        LocalJump(ComplexF32.(sm);rate=0.3f0),
    ))
    plan32=LiouvillianPlan(model32)
    source32=randn(rng,ComplexF32,length(basis32))
    reference32=zeros(ComplexF64,length(basis32))
    output64=zeros(ComplexF64,length(basis32))
    apply!(reference32,plan32,source32,LiouvillianWorkspace(plan32))
    threaded_apply!(output64,plan32,source32,
                    ThreadedLiouvillianWorkspace(plan32;tasks=3))
    @test output64==reference32

    # Warm calls retain all private matrices. Task construction has a small
    # bounded allocation, but no Schur scratch grows after construction.
    retained_ids=[(objectid(worker.input),objectid(worker.left),
                   objectid(worker.right),objectid(worker.rectangular))
                  for worker in many_tasks.workers]
    retained_bytes=Base.summarysize(many_tasks)
    threaded_apply!(threaded,plan,source,many_tasks)
    warm_allocated=@allocated threaded_apply!(threaded,plan,source,many_tasks)
    @test warm_allocated<512*1024
    @test Base.summarysize(many_tasks)==retained_bytes
    @test [(objectid(worker.input),objectid(worker.left),
            objectid(worker.right),objectid(worker.rectangular))
           for worker in many_tasks.workers]==retained_ids

    @test_throws ArgumentError ThreadedLiouvillianWorkspace(plan;tasks=0)
    @test_throws ArgumentError threaded_apply!(source,plan,source,many_tasks)
    many_tasks.busy[]=1
    @test_throws ArgumentError threaded_apply!(threaded,plan,source,many_tasks)
    many_tasks.busy[]=0

    raw_model=PIModel(basis,(LocalJump((time,parameters)->sm),))
    raw_plan=LiouvillianPlan(raw_model)
    @test raw_plan.kernels===nothing
    @test_throws ArgumentError ThreadedLiouvillianWorkspace(raw_plan)
end
