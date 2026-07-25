struct _ReadCountingVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    data::V
    reads::Base.RefValue{Int}
end
Base.size(vector::_ReadCountingVector)=size(vector.data)
Base.IndexStyle(::Type{<:_ReadCountingVector})=IndexLinear()
@inline function Base.getindex(vector::_ReadCountingVector,index::Int)
    vector.reads[]+=1
    vector.data[index]
end

@testset "preallocated operator-valued time functions" begin
    PID=PermutationalInvariantDynamics
    basis=PIBasis(3,2)
    sx=ComplexF64[0 1;1 0]
    sz=ComplexF64[1 0;0 -1]
    sm=ComplexF64[0 1;0 0]
    sp=adjoint(sm)
    parameters=(mix=0.23,)
    time=0.37

    function matrix_schedule(prototype,A,B,counter)
        InPlaceTimeOperator(prototype,(destination,t,p)->begin
            counter[]+=1
            a=cos(t);b=p.mix*sin(t)
            @inbounds for index in eachindex(destination,A,B)
                destination[index]=a*A[index]+b*B[index]
            end
            nothing
        end)
    end
    function direct_schedule(prototype,A,B,counter)
        InPlaceTimeOperator(prototype,(destination,t,p)->begin
            counter[]+=1
            a=cos(t);b=p.mix*sin(t)
            @inbounds for index in eachindex(destination.data,A.data,B.data)
                destination.data[index]=a*A.data[index]+b*B.data[index]
            end
            destination
        end)
    end

    counters=[Ref(0) for _ in 1:6]
    direct_h_a=collective_operator(basis,sx)
    direct_h_b=collective_operator(basis,sz)
    direct_l_a=collective_operator(basis,sm)
    direct_l_b=collective_operator(basis,sp)
    schedules=(
        matrix_schedule(sx,sx,sz,counters[1]),
        matrix_schedule(sz,sz,sx,counters[2]),
        matrix_schedule(sm,sm,sx,counters[3]),
        matrix_schedule(sm,sm,sp,counters[4]),
        direct_schedule(direct_h_a,direct_h_a,direct_h_b,counters[5]),
        direct_schedule(direct_l_a,direct_l_a,direct_l_b,counters[6]),
    )
    model=PIModel(basis,(
        LocalHamiltonian(schedules[1];rate=0.13),
        CollectiveHamiltonian(schedules[2];rate=-0.07),
        LocalJump(schedules[3];rate=0.19),
        CollectiveJump(schedules[4];rate=0.031),
        DirectPIHamiltonian(schedules[5];rate=0.041),
        DirectPIJump(schedules[6];rate=0.017),
    ))
    plan=LiouvillianPlan(model)
    work=LiouvillianWorkspace(plan)
    other_work=LiouvillianWorkspace(plan)
    @test plan.kernels!==nothing
    @test !isautonomous(plan)
    @test all(counter->counter[]==0,counters)
    @test work.kernel_workspaces[1].operator !== other_work.kernel_workspaces[1].operator

    rng=MersenneTwister(8801)
    input=randn(rng,ComplexF64,length(basis));output=similar(input)
    apply!(output,plan,input,time,parameters,work)
    @test all(counter->counter[]==1,counters)
    frozen=freeze(model;time=time,parameters=parameters,representation=:sparse)
    @test output≈frozen*input atol=4e-12 rtol=4e-12

    before=getindex.(counters)
    apply_adjoint!(output,plan,input,time,parameters,work)
    @test getindex.(counters)==before.+1
    @test output≈adjoint(frozen)*input atol=5e-12 rtol=5e-12

    inputs=randn(rng,ComplexF64,length(basis),4);outputs=similar(inputs)
    before=getindex.(counters)
    apply!(outputs,plan,inputs,time,parameters,work)
    @test getindex.(counters)==before.+1
    @test outputs≈frozen*inputs atol=5e-12 rtol=5e-12

    before=getindex.(counters)
    apply_adjoint!(outputs,plan,inputs,time,parameters,work)
    @test getindex.(counters)==before.+1
    @test outputs≈adjoint(frozen)*inputs atol=6e-12 rtol=6e-12

    before=getindex.(counters)
    structure=schur_block_structure(plan;time=time,parameters=parameters)
    @test getindex.(counters)==before.+1
    @test structure.metadata.applications==length(basis)

    compiled=compile(model;backend=:auto,memory_budget=typemax(Int))
    @test compiled.backend===:matrixfree
    apply!(output,compiled,input,time,parameters,LiouvillianWorkspace(compiled))
    @test output≈frozen*input atol=4e-12 rtol=4e-12

    # With an allocation-free callback, evaluated matrices, dynamic Schur
    # blocks, local gain coefficients, and multiplication scratch all remain
    # in the explicit workspace after warm-up.
    allocation_schedule=InPlaceTimeOperator(sm,(destination,t,p)->begin
        scale=one(t)+t
        @inbounds for index in eachindex(destination,sm)
            destination[index]=scale*sm[index]
        end
        nothing
    end)
    allocation_plan=LiouvillianPlan(PIModel(basis,[LocalJump(allocation_schedule;rate=0.2)]))
    allocation_work=LiouvillianWorkspace(allocation_plan)
    apply!(output,allocation_plan,input,time,nothing,allocation_work)
    @test (@allocated apply!(output,allocation_plan,input,time,nothing,
                             allocation_work))<=512
    apply!(outputs,allocation_plan,inputs,time,nothing,allocation_work)
    @test (@allocated apply!(outputs,allocation_plan,inputs,time,nothing,
                             allocation_work))<=2048

    # Resource guards count each dynamic-kernel workspace instead of using a
    # fixed multiple of the PI vector size. This small model deliberately
    # exceeds the former 16-vector approximation.
    many_schedules=ntuple(20) do index
        InPlaceTimeOperator(sx,(destination,t,p)->begin
            @. destination=(one(t)+index*t)*sx
            nothing
        end)
    end
    many_terms=ntuple(index->LocalHamiltonian(
        many_schedules[index];rate=0.01index),20)
    many_plan=LiouvillianPlan(PIModel(PIBasis(4,2),many_terms))
    many_work=LiouvillianWorkspace(many_plan)
    estimated=PID._performance_liouvillian_workspace_bytes(many_plan)
    payload=sum(sum(sizeof(array) for array in block)
                for block in many_work.blocks)+
        sum(sizeof(kernel.operator)+
            sum(sizeof(array) for array in kernel.blocks)
            for kernel in many_work.kernel_workspaces)
    @test estimated==payload
    @test estimated>16length(many_plan.basis)*sizeof(ComplexF64)

    # A complete heterogeneous action packs the source Schur blocks once.
    # This guards against restoring one full coordinate copy per term.
    many_source=randn(rng,ComplexF64,length(many_plan.basis))
    reads=Ref(0)
    counted_source=_ReadCountingVector(many_source,reads)
    many_output=similar(many_source)
    apply!(many_output,many_plan,counted_source,time,nothing,many_work)
    @test reads[]==length(many_source)
    reads[]=0
    apply_adjoint!(
        many_output,many_plan,counted_source,time,nothing,many_work)
    @test reads[]==length(many_source)

    batch_columns=8
    batch_estimate=PID._performance_liouvillian_workspace_bytes(
        many_plan;batch_columns)
    largest=maximum(length,many_plan.basis.patterns)
    @test batch_estimate-estimated==
        3largest^2*batch_columns*sizeof(ComplexF64)

    # A driven one-body local gain is retained as one rectangular contraction
    # per common child sector.  The former coarse representation stored every
    # allowed `(output coordinate,input coordinate)` pair, even when the
    # evaluated local matrix made most entries zero.
    local_scaling_basis=PIBasis(20,2)
    dense_local=ComplexF64[0.2 1.0;0.4im -0.1]
    local_scaling_schedule=InPlaceTimeOperator(
        dense_local,(destination,t,p)->nothing)
    local_scaling_plan=LiouvillianPlan(PIModel(local_scaling_basis,[
        LocalJump(local_scaling_schedule;rate=0.2)]))
    local_scaling_work=LiouvillianWorkspace(local_scaling_plan)
    local_scaling_kernel=only(local_scaling_plan.kernels)
    local_scaling_prepared=only(local_scaling_work.kernel_workspaces)
    @test !hasfield(typeof(local_scaling_kernel),:I)
    @test !hasfield(typeof(local_scaling_kernel),:J)
    @test !hasfield(typeof(local_scaling_prepared),:values)
    @test length(local_scaling_prepared.contractions)==
          length(local_scaling_kernel.branches.entries)
    connected_pairs=Set((branch.output_sector,branch.input_sector)
        for branch in local_scaling_kernel.branches.entries)
    old_coordinates=sum(length(local_scaling_basis.patterns[li])^2*
                        length(local_scaling_basis.patterns[ni])^2
                        for (li,ni) in connected_pairs)
    old_coordinate_bytes=old_coordinates*(
        2*sizeof(Int)+sizeof(eltype(local_scaling_plan)))
    retained_bytes=Base.summarysize(local_scaling_plan)+
                   Base.summarysize(local_scaling_work)
    @test old_coordinate_bytes>32*1024^2
    @test retained_bytes<old_coordinate_bytes÷32
    local_scaling_input=randn(rng,ComplexF64,length(local_scaling_basis))
    local_scaling_output=similar(local_scaling_input)
    apply!(local_scaling_output,local_scaling_plan,local_scaling_input,
           time,nothing,local_scaling_work)
    @test (@allocated apply!(local_scaling_output,local_scaling_plan,
        local_scaling_input,time,nothing,local_scaling_work))<=4096
    local_scaling_inputs=hcat(local_scaling_input,0.3local_scaling_input)
    local_scaling_outputs=similar(local_scaling_inputs)
    apply!(local_scaling_outputs,local_scaling_plan,local_scaling_inputs,
           time,nothing,local_scaling_work)
    @test (@allocated apply!(local_scaling_outputs,local_scaling_plan,
        local_scaling_inputs,time,nothing,local_scaling_work))<=4096

    stage_calls=Ref(0)
    floquet_schedule=InPlaceTimeOperator(sx,(destination,t,p)->begin
        stage_calls[]+=1
        @inbounds for index in eachindex(destination,sx)
            destination[index]=cos(t)*sx[index]
        end
        nothing
    end)
    floquet_propagator(PIModel(basis,[LocalHamiltonian(floquet_schedule)]),
                       0.1;steps=2)
    @test stage_calls[]==8 # four RK stages, independent of PI dimension

    # A raw function remains a supported allocating compatibility path. Its
    # instantaneous operator is nevertheless lowered only once for a batch.
    raw_calls=Ref(0)
    raw_operator=(t,p)->begin
        raw_calls[]+=1
        cos(t)*sm+p.mix*sin(t)*sx
    end
    raw_plan=LiouvillianPlan(PIModel(basis,[LocalJump(raw_operator;rate=0.2)]))
    @test raw_plan.kernels===nothing
    apply!(outputs,raw_plan,inputs,time,parameters,LiouvillianWorkspace(raw_plan))
    @test raw_calls[]==1
    raw_frozen=freeze(raw_plan.fallback_model;time=time,parameters=parameters,
                      representation=:sparse)
    @test outputs≈raw_frozen*inputs atol=4e-12 rtol=4e-12

    bad_return=InPlaceTimeOperator(sm,(destination,t,p)->copy(destination))
    bad_plan=LiouvillianPlan(PIModel(basis,[LocalJump(bad_return)]))
    @test_throws ArgumentError apply!(output,bad_plan,input,time,nothing,
                                      LiouvillianWorkspace(bad_plan))
    real_prototype=zeros(Float64,2,2)
    bad_precision=InPlaceTimeOperator(real_prototype,(destination,t,p)->begin
        destination[1,1]=1im
        nothing
    end)
    precision_plan=LiouvillianPlan(PIModel(basis,[LocalJump(bad_precision)]))
    @test_throws InexactError apply!(output,precision_plan,input,time,nothing,
                                     LiouvillianWorkspace(precision_plan))
    nonhermitian_schedule=InPlaceTimeOperator(sx,(destination,t,p)->begin
        destination[1,2]=2
        nothing
    end)
    nonhermitian_plan=LiouvillianPlan(PIModel(basis,[
        LocalHamiltonian(nonhermitian_schedule)]))
    @test_throws ArgumentError apply!(output,nonhermitian_plan,input,time,nothing,
                                      LiouvillianWorkspace(nonhermitian_plan))

    nonfinite_jump=InPlaceTimeOperator(sm,(destination,t,p)->begin
        destination[1,1]=ComplexF64(NaN)
        nothing
    end)
    nonfinite_jump_plan=LiouvillianPlan(PIModel(basis,[
        LocalJump(nonfinite_jump)]))
    nonfinite_jump_workspace=LiouvillianWorkspace(nonfinite_jump_plan)
    @test_throws ArgumentError apply!(
        output,nonfinite_jump_plan,input,time,nothing,
        nonfinite_jump_workspace)
    @test_throws ArgumentError apply_adjoint!(
        output,nonfinite_jump_plan,input,time,nothing,
        nonfinite_jump_workspace)

    nonfinite_hamiltonian=InPlaceTimeOperator(sx,(destination,t,p)->begin
        destination[1,1]=ComplexF64(Inf)
        nothing
    end)
    nonfinite_hamiltonian_plan=LiouvillianPlan(PIModel(basis,[
        LocalHamiltonian(nonfinite_hamiltonian)]))
    @test_throws ArgumentError apply!(
        output,nonfinite_hamiltonian_plan,input,time,nothing,
        LiouvillianWorkspace(nonfinite_hamiltonian_plan))

    fixed_nonfinite=copy(sm)
    fixed_nonfinite[1,1]=ComplexF64(NaN)
    @test_throws ArgumentError LiouvillianPlan(PIModel(basis,[
        LocalJump(fixed_nonfinite)]))
    nonfinite_prototype=InPlaceTimeOperator(
        fixed_nonfinite,(destination,t,p)->nothing)
    @test_throws ArgumentError LiouvillianPlan(PIModel(basis,[
        LocalJump(nonfinite_prototype)]))
    direct_nonfinite=identity_operator(basis)
    direct_nonfinite.data[1]=ComplexF64(Inf)
    @test_throws ArgumentError LiouvillianPlan(PIModel(basis,[
        DirectPIJump(direct_nonfinite)]))

    complex_rate_model=PIModel(basis,[LocalJump(sm;rate=1im)])
    @test_throws ArgumentError liouvillian(complex_rate_model;representation=:sparse)
    @test_throws ArgumentError LiouvillianPlan(complex_rate_model)
    dynamic_complex=PIModel(basis,[CollectiveJump(schedules[4];rate=(t,p)->1im)])
    dynamic_complex_plan=LiouvillianPlan(dynamic_complex)
    @test_throws ArgumentError apply!(output,dynamic_complex_plan,input,time,parameters,
                                      LiouvillianWorkspace(dynamic_complex_plan))

    pair_basis=PIBasis(3,2);pair_h=kron(sz,sz);pair_l=kron(sm,sm)
    pair_input=randn(rng,ComplexF64,length(pair_basis));pair_output=similar(pair_input)
    pair_inputs=randn(rng,ComplexF64,length(pair_basis),3)
    pair_outputs=similar(pair_inputs)
    for (prototype,constructor) in (
        (pair_h,schedule->PBodyHamiltonian(schedule,2;rate=0.11)),
        (pair_l,schedule->LocalPBodyJump(schedule,2;rate=0.07)),
        (pair_l,schedule->CollectivePBodyJump(schedule,2;rate=0.03)),
    )
        calls=Ref(0)
        pair_schedule=InPlaceTimeOperator(prototype,(destination,t,p)->begin
            calls[]+=1
            @inbounds for index in eachindex(destination,prototype)
                destination[index]=(1+t)*prototype[index]
            end
            nothing
        end)
        pair_model=PIModel(pair_basis,[constructor(pair_schedule)])
        pair_plan=LiouvillianPlan(pair_model);pair_work=LiouvillianWorkspace(pair_plan)
        apply!(pair_output,pair_plan,pair_input,time,nothing,pair_work)
        pair_frozen=freeze(pair_model;time=time,representation=:sparse)
        @test pair_output≈pair_frozen*pair_input atol=5e-12 rtol=5e-12
        apply_adjoint!(pair_output,pair_plan,pair_input,time,nothing,pair_work)
        @test pair_output≈adjoint(pair_frozen)*pair_input atol=5e-12 rtol=5e-12
        @test (@allocated apply!(pair_output,pair_plan,pair_input,time,nothing,
                                 pair_work))<=1024
        apply!(pair_outputs,pair_plan,pair_inputs,time,nothing,pair_work)
        @test pair_outputs≈pair_frozen*pair_inputs atol=5e-12 rtol=5e-12
        apply_adjoint!(pair_outputs,pair_plan,pair_inputs,time,nothing,pair_work)
        @test pair_outputs≈adjoint(pair_frozen)*pair_inputs atol=5e-12 rtol=5e-12
        @test calls[]==6
    end

    # The dynamic Appendix-D local gain is retained as rectangular path
    # contractions, not as a dense PI-coordinate I/J/value table. At N=20 the
    # removed representation would already occupy tens of MiB, whereas the
    # complete plan and caller workspace stay well below that scaling.
    scaling_basis=PIBasis(20,2)
    scaling_schedule=InPlaceTimeOperator(pair_l,(destination,t,p)->nothing)
    scaling_plan=LiouvillianPlan(PIModel(scaling_basis,[
        LocalPBodyJump(scaling_schedule,2)]))
    scaling_work=LiouvillianWorkspace(scaling_plan)
    scaling_kernel=only(scaling_plan.kernels)
    scaling_prepared=only(scaling_work.kernel_workspaces)
    @test !hasfield(typeof(scaling_kernel),:I)
    @test !hasfield(typeof(scaling_kernel),:J)
    @test !hasfield(typeof(scaling_prepared),:values)
    old_coordinates=sum(length(scaling_basis.patterns[group[1]])^2*
                        length(scaling_basis.patterns[group[2]])^2
                        for group in scaling_kernel.groups)
    old_coordinate_bytes=old_coordinates*(2*sizeof(Int)+sizeof(eltype(scaling_plan)))
    retained_bytes=Base.summarysize(scaling_plan)+Base.summarysize(scaling_work)
    @test old_coordinate_bytes>40*1024^2
    @test retained_bytes<old_coordinate_bytes÷16
    scaling_input=randn(rng,ComplexF64,length(scaling_basis))
    scaling_output=similar(scaling_input)
    apply!(scaling_output,scaling_plan,scaling_input,time,nothing,scaling_work)
    apply_adjoint!(scaling_output,scaling_plan,scaling_input,time,nothing,scaling_work)
    @test (@allocated apply!(scaling_output,scaling_plan,scaling_input,time,
                             nothing,scaling_work))<=4096
    @test (@allocated apply_adjoint!(scaling_output,scaling_plan,scaling_input,
                                     time,nothing,scaling_work))<=4096

    nonsymmetric=kron(sm,Matrix{ComplexF64}(I,2,2))
    bad_pair_schedule=InPlaceTimeOperator(pair_l,(destination,t,p)->begin
        copyto!(destination,nonsymmetric);nothing
    end)
    bad_pair_plan=LiouvillianPlan(PIModel(pair_basis,[
        LocalPBodyJump(bad_pair_schedule,2)]))
    @test_throws ArgumentError apply!(pair_output,bad_pair_plan,pair_input,time,nothing,
                                      LiouvillianWorkspace(bad_pair_plan))

    schedule32=InPlaceTimeOperator(ComplexF32.(sm),(destination,t,p)->nothing)
    plan32=@inferred LiouvillianPlan(PIModel(PIBasis(2,2),[
        CollectiveJump(schedule32;rate=1f0)]))
    @test eltype(plan32)===ComplexF32

    pair_schedule32=InPlaceTimeOperator(ComplexF32.(pair_l),
                                        (destination,t,p)->nothing)
    pair_basis32=PIBasis(3,2)
    pair_plan32=@inferred LiouvillianPlan(PIModel(pair_basis32,[
        LocalPBodyJump(pair_schedule32,2;rate=0.25f0)]))
    pair_work32=LiouvillianWorkspace(pair_plan32)
    pair_work32_other=LiouvillianWorkspace(pair_plan32)
    @test eltype(pair_plan32)===ComplexF32
    @test only(pair_work32.kernel_workspaces).contractions[1] !==
          only(pair_work32_other.kernel_workspaces).contractions[1]
    @test only(pair_work32.kernel_workspaces).gain_scratch !==
          only(pair_work32_other.kernel_workspaces).gain_scratch
    input32=randn(rng,ComplexF32,length(pair_basis32));output32=similar(input32)
    apply!(output32,pair_plan32,input32,time,nothing,pair_work32)
    frozen32=freeze(pair_plan32.fallback_model;time=time,representation=:sparse)
    @test output32≈frozen32*input32 atol=3f-5 rtol=3f-5
    apply_adjoint!(output32,pair_plan32,input32,time,nothing,pair_work32)
    @test output32≈adjoint(frozen32)*input32 atol=3f-5 rtol=3f-5

    local_basis32=PIBasis(3,2)
    local_plan32=@inferred LiouvillianPlan(PIModel(local_basis32,[
        LocalJump(schedule32;rate=0.25f0)]))
    local_work32=LiouvillianWorkspace(local_plan32)
    local_input32=randn(rng,ComplexF32,length(local_basis32))
    local_output32=similar(local_input32)
    @test eltype(local_plan32)===ComplexF32
    apply!(local_output32,local_plan32,local_input32,time,nothing,local_work32)
    local_frozen32=freeze(local_plan32.fallback_model;time=time,
                          representation=:sparse)
    @test local_output32≈local_frozen32*local_input32 atol=3f-5 rtol=3f-5
    apply_adjoint!(local_output32,local_plan32,local_input32,time,nothing,
                   local_work32)
    @test isapprox(local_output32,adjoint(local_frozen32)*local_input32;
                   atol=3f-5,rtol=3f-5)

    # Plan-owned block scratch has the compiled scalar type.  A wider source
    # would otherwise be narrowed while it is copied into that scratch.
    input64=ComplexF64.(input32);output64=similar(input64)
    @test_throws ArgumentError apply!(output64,pair_plan32,input64,time,
                                      nothing,pair_work32)
    @test_throws ArgumentError apply_adjoint!(output64,pair_plan32,input64,time,
                                              nothing,pair_work32)
    batch64=hcat(input64,2input64);batch_output64=similar(batch64)
    @test_throws ArgumentError apply!(batch_output64,pair_plan32,batch64,time,
                                      nothing,pair_work32)
    @test_throws ArgumentError apply_adjoint!(batch_output64,pair_plan32,
                                              batch64,time,nothing,pair_work32)
    narrow_output=zeros(Complex{Float16},length(input32))
    @test_throws ArgumentError apply!(narrow_output,pair_plan32,input32,time,
                                      nothing,pair_work32)
    # A wider destination is safe because every computed plan value can be
    # represented without changing the source/application precision.
    apply!(output64,pair_plan32,input32,time,nothing,pair_work32)
    @test output64≈ComplexF64.(frozen32*input32) atol=3e-5 rtol=3e-5

    # One-body schedules own fixed-precision Schur scratch. Beyond the native
    # cancellation threshold they must request a wider prototype at compile
    # time; static collective_block can widen selectively, but this hot path
    # cannot silently do so.
    risk_basis=PIBasis(34,2;sectors=[(34,0)])
    risk_operator=Complex{Float16}[0 1;1 0]
    risk_schedule=InPlaceTimeOperator(
        risk_operator,(destination,t,p)->nothing)
    for term in (CollectiveHamiltonian(risk_schedule),
                 CollectiveJump(risk_schedule))
        error=try
            LiouvillianPlan(PIModel(risk_basis,[term]));nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("wider InPlaceTimeOperator prototype",sprint(showerror,error))
    end

    # A direct-PI prototype stores coefficient blocks. The physical block may
    # remain representable even when sqrt(f^nu) is not; the prepared inverse
    # scale must therefore be fused with each coefficient rather than formed as
    # a standalone divisor.
    direct_sector=Partition((20,20))
    direct_basis=PIBasis(40,2;sectors=[direct_sector.parts])
    direct_operator=PIOperator(direct_basis;T=Float16)
    direct_operator.data[1]=complex(floatmax(Float16),zero(Float16))
    direct_context=PID.TermCompileContext(direct_basis,nothing,
        Dict{Int,PBodyGeometry{Float16,2,3,typeof(direct_basis)}}(),Float16)
    direct_builder=PID._direct_pi_block_builder(direct_context,Float16)
    @test !only(direct_builder.inverse_scales).use_divisor
    direct_blocks=[zeros(Complex{Float16},1,1)]
    PID._fill_dynamic_blocks!(direct_blocks,direct_builder,direct_operator)
    direct_expected=Float16(BigFloat(floatmax(Float16))/sqrt(BigFloat(
        symmetric_group_dimension(direct_sector))))
    @test only(direct_blocks[1])≈direct_expected rtol=Float16(2e-3)
end
