function _floquet_evolve_allocation_comparison(rho,F,nperiods)
    floquet_evolve(rho,F,nperiods)
    PIState(rho.basis,F^nperiods*rho.data)
    efficient=@allocated floquet_evolve(rho,F,nperiods)
    matrix_power=@allocated PIState(rho.basis,F^nperiods*rho.data)
    efficient,matrix_power
end

@testset "Reusable matrix-free Floquet maps" begin
    b=PIBasis(1,2);sm=ComplexF64[0 1;0 0];sx=ComplexF64[0 1;1 0]
    period=1.1
    driven=PIModel(b,[
        LocalHamiltonian(sx;rate=(t,p)->0.23sin(2pi*t/period)),
        LocalJump(sm;rate=(t,p)->0.8+0.1cos(2pi*t/period))])
    map=floquet_map(driven,period;steps=56)
    work=FloquetWorkspace(map)
    dense=floquet_propagator(driven,period;steps=56)
    n=length(b);x=randn(MersenneTwister(401),ComplexF64,n)
    a=randn(MersenneTwister(402),ComplexF64,n)
    y=similar(x);adjoint_y=similar(a)
    apply!(y,map,x,work)
    apply_adjoint!(adjoint_y,map,a,work)
    @test y≈dense*x atol=3e-13
    @test adjoint_y≈adjoint(dense)*a atol=3e-13
    @test dot(a,y)≈dot(adjoint_y,x) atol=4e-13
    @test size(map,3)==1

    batch=hcat(x,a);batch_result=similar(batch)
    apply!(batch_result,map,batch,work)
    @test batch_result≈dense*batch atol=4e-13
    aliased=copy(batch)
    apply!(aliased,map,aliased,work)
    @test aliased≈dense*batch atol=4e-13

    # The explicit fixed-capacity route advances all columns through one RK4
    # graph. Forward mode owns only origin/stage/derivative matrices; full
    # mode adds the arrays required by the exact discrete adjoint.
    forward_batch_work=FloquetBatchWorkspace(map,2;mode=:forward)
    @test forward_batch_work.source_workspace isa LiouvillianWorkspace
    @test forward_batch_work.source_workspace.batch.capacity>=2
    true_batch=similar(batch)
    apply!(true_batch,map,batch,forward_batch_work)
    @test true_batch≈dense*batch atol=4e-13
    warm_batch_allocation=@allocated apply!(true_batch,map,batch,
                                             forward_batch_work)
    @test warm_batch_allocation<=4096
    @test forward_batch_work.k2===nothing
    @test_throws ArgumentError apply_adjoint!(similar(batch),map,batch,
        forward_batch_work)
    full_batch_work=FloquetBatchWorkspace(map,2;mode=:full)
    batch_adjoint=similar(batch)
    apply_adjoint!(batch_adjoint,map,batch,full_batch_work)
    @test batch_adjoint≈adjoint(dense)*batch atol=4e-13
    @test dot(batch,true_batch)≈dot(batch_adjoint,batch) atol=8e-13
    @test Base.summarysize(forward_batch_work)<Base.summarysize(full_batch_work)
    in_place_batch=copy(batch)
    apply!(in_place_batch,map,in_place_batch,forward_batch_work)
    @test in_place_batch≈dense*batch atol=4e-13
    @test_throws ArgumentError apply!(similar(batch),map,batch,
        FloquetBatchWorkspace(map,1))

    grown_batch_work=FloquetBatchWorkspace(map,2)
    owned_before=PermutationalInvariantDynamics._block_operator_workspace_bytes(
        grown_batch_work)
    PermutationalInvariantDynamics._ensure_batch_capacity!(
        grown_batch_work.source_workspace.batch,4)
    @test grown_batch_work.source_workspace.batch.capacity==4
    @test PermutationalInvariantDynamics._block_operator_workspace_bytes(
        grown_batch_work)>owned_before

    map32=floquet_map(ComplexF32[-0.2 0;0 -0.3],0.2f0;steps=4,
        trace_vector=ComplexF32[1,0])
    map32_work=FloquetBatchWorkspace(map32,1)
    input32=reshape(ComplexF32[0.4,0.6],2,1)
    @test_throws ArgumentError apply!(zeros(ComplexF64,2,1),map32,input32,
                                      map32_work)
    apply!(similar(input32),map32,input32,map32_work)

    static_compiled=compile(PIModel(b,[LocalJump(sm;rate=0.8)]);
                            backend=:sparse,memory_budget=Inf)
    sparse_map=floquet_map(static_compiled,0.2;steps=4)
    @test FloquetWorkspace(sparse_map).source_workspace===nothing
    @test FloquetBatchWorkspace(sparse_map,2).source_workspace===nothing
    @test iszero(PermutationalInvariantDynamics._performance_source_action_bytes(
        static_compiled,eltype(sparse_map)))
    @test iszero(PermutationalInvariantDynamics._floquet_batch_source_bytes(
        static_compiled,2,eltype(sparse_map)))

    # A bare family operator binds its rates inside synchronized callbacks and
    # exposes no reconstructible plan, but it still owns a known Liouvillian
    # batch workspace. Floquet's constructor must budget that lazy growth.
    scalar_family=compile_family(PIModel(b,(LocalJump(sm;rate=0.8),)))
    scalar_specialization=specialize(
        scalar_family,(0.6,);backend=:matrixfree)
    bare_family_operator=scalar_specialization.operator
    @test bare_family_operator.plan===nothing
    @test bare_family_operator.workspace.batch.capacity==0
    bare_family_map=floquet_map(bare_family_operator,0.2;steps=4,
        trace_vector=copy(static_compiled.plan.tracevec))
    family_growth=
        PermutationalInvariantDynamics._performance_batched_action_growth_bytes(
            bare_family_operator,2)
    family_batch_budget=
        PermutationalInvariantDynamics._floquet_batch_workspace_bytes(
            bare_family_map,2,:forward)
    @test family_growth>0
    @test_throws ArgumentError FloquetBatchWorkspace(
        bare_family_map,2;memory_budget=family_batch_budget-1)
    family_batch_work=FloquetBatchWorkspace(
        bare_family_map,2;memory_budget=family_batch_budget)
    family_batch_result=similar(batch)
    apply!(family_batch_result,bare_family_map,batch,family_batch_work)
    @test family_batch_result≈
        floquet_propagator(scalar_specialization,0.2;steps=4,
            memory_budget=Inf)*batch atol=4e-13
    @test bare_family_operator.workspace.batch.capacity==2

    vector_stage_calls=Ref(0);batch_stage_calls=Ref(0)
    batched_source=MatrixFreeLiouvillian(n,
        (destination,source,t,p)->begin
            vector_stage_calls[]+=1;fill!(destination,0)
        end,ComplexF64,ones(ComplexF64,n);
        batched_action! = (destination,source,t,p)->begin
            batch_stage_calls[]+=1;fill!(destination,0)
        end)
    batched_map=floquet_map(batched_source,0.2;steps=5)
    batched_map_work=FloquetBatchWorkspace(batched_map,2)
    zero_generator_result=similar(batch)
    apply!(zero_generator_result,batched_map,batch,batched_map_work)
    @test zero_generator_result==batch
    @test batch_stage_calls[]==4batched_map.steps
    @test iszero(vector_stage_calls[])

    # Preparation owns raw matrix data instead of retaining a caller's
    # mutable array as part of a supposedly fixed map.
    raw=ComplexF64[-0.4 0.1;0 -0.7];raw_reference=copy(raw)
    owned=floquet_map(raw,0.3;steps=12,trace_vector=ComplexF64[1,0])
    owned_before=owned*ComplexF64[0.2,0.8]
    raw.=9
    @test owned*ComplexF64[0.2,0.8]≈owned_before atol=0
    @test owned.source==raw_reference

    selected=selected_floquet_multipliers(map;nev=n,method=:arnoldi,
        krylovdim=n,vectors=true,rng=MersenneTwister(403))
    @test selected.values≈floquet_multipliers(dense) atol=2e-12
    @test maximum(selected.residuals)<1e-11
    @test !selected.partial_scope
    block_selected=selected_floquet_multipliers(map;nev=n,
        method=:block_arnoldi,block_size=min(2,n),krylovdim=n,
        maxrestarts=0,vectors=true,rng=MersenneTwister(0x626c6f63))
    @test sort(block_selected.values;by=abs,rev=true)≈
        sort(floquet_multipliers(dense);by=abs,rev=true) atol=3e-12
    @test maximum(block_selected.residuals)<2e-11
    @test block_selected.operator_batches<
        block_selected.operator_applications
    block_capacity=min(2,n)
    block_budget=PermutationalInvariantDynamics._selected_spectrum_workspace_bytes(
        map,:block_arnoldi,n,n;vectors=true,target=nothing,
        block_size=block_capacity,maxrestarts=0)+
        PermutationalInvariantDynamics._floquet_batch_workspace_bytes(
            map,block_capacity,:forward)
    budgeted_block=selected_floquet_multipliers(map;nev=n,
        method=:block_arnoldi,block_size=block_capacity,krylovdim=n,
        maxrestarts=0,vectors=true,memory_budget=block_budget,
        rng=MersenneTwister(0x62756467))
    @test maximum(budgeted_block.residuals)<2e-11
    @test_throws ArgumentError selected_floquet_multipliers(map;nev=n,
        method=:block_arnoldi,block_size=block_capacity,krylovdim=n,
        maxrestarts=0,vectors=true,memory_budget=block_budget-1,
        rng=MersenneTwister(0x62756467))
    gap=floquet_gap(map;nev=n,method=:arnoldi,krylovdim=n,
        rng=MersenneTwister(404),return_info=true)
    @test gap.global_gap_certified
    @test gap.gap≈floquet_gap(dense,period) atol=2e-11
    partial=floquet_gap(map;nev=2,method=:arnoldi,krylovdim=n,
        rng=MersenneTwister(405),return_info=true)
    @test partial.partial_scope&&!partial.global_gap_certified
    @test_throws ArgumentError floquet_gap(map;nev=2,method=:arnoldi,
        krylovdim=n,rng=MersenneTwister(405))

    steady_krylov=floquet_steady_state(map;krylovdim=n,return_info=true)
    steady_dense=floquet_steady_state(map;method=:dense,return_info=true)
    @test steady_krylov.state.data≈steady_dense.state.data atol=3e-10
    @test steady_krylov.residual<2e-10
    @test steady_krylov.propagator===nothing
    @test steady_dense.propagator≈dense atol=3e-13
    rho=iid_pure_state(b,ComplexF64[0,1])
    @test floquet_evolve(rho,map,3).data≈dense^3*rho.data atol=4e-12
    @test last(stroboscopic_evolution(rho,map,2)).data≈
        dense^2*rho.data atol=3e-12
    @test floquet_exponents(map,period;nev=n,method=:arnoldi,
        krylovdim=n,rng=MersenneTwister(406))≈
        floquet_exponents(map;nev=n,method=:arnoldi,krylovdim=n,
            rng=MersenneTwister(406))
    @test_throws ArgumentError floquet_multipliers(map,2period)

    one_dimensional=floquet_map(zeros(ComplexF64,1,1),1.0;
        steps=2,trace_vector=ComplexF64[1])
    one_gap=floquet_gap(one_dimensional;nev=1,method=:arnoldi,
        krylovdim=1,rng=MersenneTwister(407),return_info=true)
    @test isnan(one_gap.gap)&&!one_gap.global_gap_certified
    @test isnan(floquet_gap(one_dimensional;nev=1,method=:arnoldi,
        krylovdim=1,rng=MersenneTwister(407)))

    # Arnoldi sees only the distinct invariant Krylov directions of repeated
    # fixed multipliers. The selected gap can be inspected, but is never
    # mislabeled as globally certified.
    repeated=floquet_map(Diagonal(ComplexF64[0,0,-0.5]),1.0;
        steps=8,trace_vector=ComplexF64[1,1,0])
    repeated_gap=floquet_gap(repeated;nev=3,method=:arnoldi,krylovdim=3,
        rng=MersenneTwister(408),require_convergence=false,return_info=true)
    @test repeated_gap.partial_scope&&!repeated_gap.global_gap_certified
    fixed_only=floquet_map(zeros(ComplexF64,2,2),1.0;
        steps=2,trace_vector=ComplexF64[1,1])
    fixed_only_gap=floquet_gap(fixed_only;nev=2,method=:arnoldi,krylovdim=2,
        rng=MersenneTwister(409),require_convergence=false,return_info=true)
    @test isnan(fixed_only_gap.gap)&&fixed_only_gap.partial_scope

    # A diagonal strong symmetry stays compressed during every map action.
    b2=PIBasis(2,2);sz=ComplexF64[1 0;0 -1]
    dephasing=floquet_map(PIModel(b2,[LocalJump(sz)]),0.4;steps=8)
    restriction=diagonal_symmetry_restriction(
        b2,Diagonal(ComplexF64[1,-1]);charge=1)
    restricted=restricted_floquet_map(dephasing,restriction)
    restricted_state=floquet_steady_state(restricted;
        krylovdim=size(restricted,1),operator_scale=1.0,return_info=true)
    @test restricted_state.full_residual<1e-11
    @test restricted_state.leakage_residual<1e-11
    restricted_response=ResponseWorkspace(restricted;
        krylovdim=size(restricted,1),mode=:linear)
    restricted_resolvent=resolvent_norm(restricted,2+0.3im;
        workspace=restricted_response,rng=MersenneTwister(410),
        max_power_iterations=100)
    restricted_dense=PermutationalInvariantDynamics._materialize(restricted)
    @test restricted_resolvent≈resolvent_norm(restricted_dense,2+0.3im) rtol=3e-8

    custom=MatrixFreeLiouvillian(2,
        (destination,source,t,p)->mul!(destination,raw_reference,source),
        ComplexF64,ComplexF64[1,0])
    custom_map=floquet_map(custom,0.2;steps=4)
    custom_x=ComplexF64[0.3,0.7]
    @test_throws ArgumentError apply_adjoint!(similar(custom_x),custom_map,
        custom_x,FloquetWorkspace(custom_map))
    @test_throws ArgumentError resolvent_norm(custom_map,2+im)

    map32=floquet_map(zeros(ComplexF32,1,1),1f0;steps=2,
        trace_vector=ComplexF32[1])
    @test_throws ArgumentError floquet_map(zeros(ComplexF32,1,1),1f0;
        steps=16_777_217,trace_vector=ComplexF32[1])
    @test_throws ArgumentError floquet_map(zeros(ComplexF32,1,1),
        floatmax(Float32);steps=2,t0=floatmax(Float32),
        trace_vector=ComplexF32[1])
    @test_throws ArgumentError floquet_map(zeros(ComplexF32,1,1),1f0;
        steps=2,trace_vector=Int[16_777_217])
    exact_integer_trace=floquet_map(zeros(ComplexF32,1,1),1f0;
        steps=2,trace_vector=Int[16_777_216])
    @test exact_integer_trace.tracevec==ComplexF32[16_777_216]
    other32=floquet_map(zeros(ComplexF32,1,1),1f0;steps=2,
        trace_vector=ComplexF32[1])
    @test_throws ArgumentError apply!(zeros(ComplexF32,1),map32,
        ones(ComplexF32,1),FloquetWorkspace(other32))
end

function _legacy_stroboscopic_evolution(rho,F,nperiods)
    out=typeof(rho)[];x=copy(rho.data);y=similar(x)
    push!(out,PIState(rho.basis,copy(x)))
    for _ in 1:nperiods
        mul!(y,F,x);x,y=y,x;push!(out,PIState(rho.basis,copy(x)))
    end
    out
end

function _stroboscopic_allocation_comparison(rho,F,nperiods)
    stroboscopic_evolution(rho,F,nperiods)
    _legacy_stroboscopic_evolution(rho,F,nperiods)
    efficient=@allocated stroboscopic_evolution(rho,F,nperiods)
    double_copy=@allocated _legacy_stroboscopic_evolution(rho,F,nperiods)
    efficient,double_copy
end

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
    @test all(trajectory[i].data!==trajectory[j].data
              for i in eachindex(trajectory),j in eachindex(trajectory) if i!=j)
    @test isempty(stroboscopic_evolution(rho,F,0;include_initial=false))
    @test floquet_evolve(rho,F,0).data==rho.data
    @test floquet_evolve(rho,F,-1).data≈F^-1*rho.data atol=2e-12
    @test_throws DimensionMismatch floquet_evolve(rho,zeros(2,2),1)

    # Allocating products follow ordinary matrix promotion, including the
    # zero-period identity map, without narrowing a Float64 map into a Float32
    # state buffer.
    rho32=iid_pure_state(b,ComplexF32[0,1])
    promoted=floquet_evolve(rho32,F,0)
    @test eltype(promoted.data)===ComplexF64
    @test promoted.data==ComplexF64.(rho32.data)
    promoted_path=stroboscopic_evolution(rho32,F,1)
    @test eltype(first(promoted_path).data)===ComplexF64
    @test last(promoted_path).data≈F*ComplexF64.(rho32.data) atol=2e-12

    # A dense matrix power constructs PI-dimensional matrix temporaries even
    # when only its action on one state is needed. Repeated two-vector `mul!`
    # applications retain the same value with substantially less allocation.
    allocation_basis=PIBasis(9,2)
    allocation_state=iid_pure_state(allocation_basis,ComplexF64[0,1])
    allocation_dimension=length(allocation_basis)
    allocation_map=Matrix{ComplexF64}(I,allocation_dimension,
                                      allocation_dimension)
    allocation_map[1,2]=0.01
    periods=8
    @test floquet_evolve(allocation_state,allocation_map,periods).data≈
          allocation_map^periods*allocation_state.data atol=2e-13
    evolve_alloc,power_alloc=_floquet_evolve_allocation_comparison(
        allocation_state,allocation_map,periods)
    @test evolve_alloc<power_alloc
    saved_alloc,double_copy_alloc=_stroboscopic_allocation_comparison(
        allocation_state,allocation_map,periods)
    @test saved_alloc<double_copy_alloc

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
