@testset "certified Schur-diagonal population backend" begin
    PID=PermutationalInvariantDynamics
    b=PIBasis(4,2)
    jm=ComplexF64[0 1;0 0]
    jp=jm'
    jz=ComplexF64[-0.5 0;0 0.5]
    model=PIModel(b,[LocalHamiltonian(jz;rate=0.37),
                     LocalJump(jm;rate=0.41),
                     LocalJump(jp;rate=0.13),
                     CollectiveJump(jm;rate=0.07)])

    @test population_dimension(b)==sum(length,b.patterns)
    report=population_invariance(model)
    @test report.invariant===true
    @test report.reason===:certified
    plan=PopulationPlan(model)
    @test size(plan)==(population_dimension(b),population_dimension(b))
    @test isautonomous(plan)
    @test plan.invariance.invariant===true
    @test !(:coordinate_map in fieldnames(typeof(plan)))
    compact_map=PID._population_coordinate_map(b,Float64)
    @test compact_map.diagonal_lookup isa Dict{Int,Int}
    @test length(compact_map.diagonal_lookup)==population_dimension(b)

    M=population_generator(plan;representation=:sparse)
    @test M isa SparseMatrixCSC
    @test size(M)==size(plan)
    @test maximum(abs,vec(sum(M;dims=1)))<2e-12

    p=ComplexF64.(collect(1:population_dimension(b)))
    p./=sum(p)
    rho=state_from_populations(b,p;validate=true)
    @test diagonal_populations(rho)≈p atol=2e-14
    @test trace(rho)≈1 atol=2e-14
    xcopy=copy(p);rho_copy=state_from_populations(b,xcopy)
    xcopy[1]=99
    @test diagonal_populations(rho_copy)[1]!=99

    full_plan=PID.LiouvillianPlan(model)
    full_work=PID.LiouvillianWorkspace(full_plan)
    dy=similar(rho.data)
    PID.apply!(dy,full_plan,rho.data,0.0,nothing,full_work)
    projected=diagonal_populations(PIState(b,dy))
    @test M*p≈projected atol=3e-12
    pw=PopulationWorkspace(plan,p)
    @test sum(length,(pw.stage,pw.k1,pw.k2,pw.k3,pw.k4))==3length(p)
    @test isempty(pw.k3)&&isempty(pw.k4)
    dp=similar(p)
    PID.apply!(dp,plan,p,0.0,nothing,pw)
    @test dp≈M*p atol=2e-12
    PID.apply!(dp,plan,p,0.0,nothing,pw)
    @test (@allocated PID.apply!(dp,plan,p,0.0,nothing,pw))<=512
    shared_population=similar(p)
    aliased_population=PopulationWorkspace(
        shared_population,shared_population,similar(p),similar(p,0),
        similar(p,0),similar(p))
    @test_throws ArgumentError evolve_populations!(
        shared_population,plan,p,(0.0,0.01);
        steps=1,workspace=aliased_population)

    rho0=iid_pure_state(b,ComplexF64[0,1])
    p0=diagonal_populations(rho0)
    times=range(0,0.25;length=5)
    psol=solve_populations(plan,p0,(0.0,0.25);saveat=times,
                           steps_per_interval=80)
    rsol=solve_dynamics(model,rho0,(0.0,0.25);saveat=times,
                        steps_per_interval=80)
    @test length(psol)==length(times)
    @test psol[1]≈p0 atol=1e-14
    @test psol[end]≈diagonal_populations(rsol[end]) atol=3e-12
    @test state(psol,lastindex(psol)).data≈rsol[end].data atol=3e-12
    @test state_at(psol,last(times)).data≈rsol[end].data atol=3e-12

    stationary=stationary_populations(plan;method=:direct)
    @test norm(M*stationary)<2e-10
    @test sum(stationary)≈1 atol=2e-12

    jx=ComplexF64[0 0.5;0.5 0]
    mixing=PIModel(b,[LocalHamiltonian(jx)])
    bad=population_invariance(mixing)
    @test bad.invariant===false
    @test bad.maximum_leakage>bad.tolerance
    @test_throws ArgumentError PopulationPlan(mixing)
    coherent=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
    @test_throws ArgumentError diagonal_populations(coherent)
    @test diagonal_populations(coherent;check=false) isa Vector

    # Certification is structural by default: a nonzero mixing term cannot
    # disappear merely because its rate is small or another channel is large.
    weak_mixing=PIModel(b,[LocalHamiltonian(jx;rate=1e-13)])
    weak_driven=PIModel(b,[LocalHamiltonian(
        jx;rate=(time,parameters)->1e-13)])
    mixed_scale=PIModel(b,[LocalJump(jm;rate=1),
                           LocalHamiltonian(jx;rate=1e-10)])
    @test population_invariance(weak_mixing).invariant===false
    @test population_invariance(weak_driven).invariant===false
    @test population_invariance(mixed_scale).invariant===false
    @test_throws ArgumentError PopulationPlan(weak_mixing)
    approximate=population_invariance(weak_mixing;atol=1e-12)
    @test approximate.invariant===true
    @test approximate.reason===:within_tolerance
    exact_cancellation=PIModel(b,[LocalHamiltonian(jx),
                                  LocalHamiltonian(-jx)])
    @test population_invariance(exact_cancellation).invariant===true
    @test all(iszero,nonzeros(population_generator(
        PopulationPlan(exact_cancellation);representation=:sparse)))
    @test_throws ArgumentError population_invariance(plan;atol=1e-12)
    @test_throws ArgumentError population_invariance(model;atol=Inf)
    @test_throws ArgumentError population_invariance(model;rtol=NaN)
    @test_throws ArgumentError population_invariance(
        PIModel(b,[LocalJump(jm;rate=Inf)]))

    driven=PIModel(b,[LocalJump(jm;rate=(t,parameters)->parameters.gamma*(1+t))])
    driven_plan=PopulationPlan(driven)
    @test !isautonomous(driven_plan)
    @test population_invariance(driven).invariant===true
    Md=population_generator(driven_plan;representation=:sparse,time=0.2,
                            parameters=(gamma=0.4,))
    base=population_generator(PopulationPlan(PIModel(b,[LocalJump(jm)]));
                              representation=:sparse)
    @test Md≈0.48base atol=2e-12
    @test_throws ArgumentError population_generator(driven_plan;representation=:sparse)
    pd=solve_populations(driven_plan,p0,(0.0,0.2);saveat=[0.0,0.2],
                         parameters=(gamma=0.4,),steps_per_interval=160)
    rd=solve_dynamics(driven,rho0,(0.0,0.2);saveat=[0.0,0.2],
                      parameters=(gamma=0.4,),steps_per_interval=160)
    @test pd[end]≈diagonal_populations(rd[end]) atol=4e-12

    schedule=InPlaceTimeOperator(jm,(destination,t,parameters)->begin
        destination .*= 1+t
    end)
    operator_driven=PIModel(b,[LocalJump(schedule)])
    uncertain=population_invariance(operator_driven)
    @test uncertain.invariant===missing
    @test uncertain.reason===:operator_time_dependence
    @test_throws ArgumentError PopulationPlan(operator_driven)
    frozen=PID.freeze(operator_driven;time=0.3,representation=:sparse)
    @test PopulationPlan(frozen,b).invariance.invariant===true

    prepared_matrixfree=compile(model;backend=:matrixfree).operator
    distinct_basis=PIBasis(b.N,b.d)
    @test_throws ArgumentError PopulationPlan(prepared_matrixfree,distinct_basis)
    @test_throws ArgumentError population_invariance(
        prepared_matrixfree,distinct_basis)

    zero_action=(destination,source,time,parameters)->begin
        fill!(destination,zero(eltype(destination)))
    end
    custom_matrixfree=MatrixFreeLiouvillian(
        length(b),zero_action,ComplexF64,ones(ComplexF64,length(b)))
    @test PopulationPlan(custom_matrixfree,b).invariance.invariant===true

    offdiagonal_row=findfirst(index->
        !haskey(compact_map.diagonal_lookup,index),1:length(b))
    nonfinite=zeros(ComplexF64,length(b),length(b))
    nonfinite[offdiagonal_row,first(compact_map.diagonal_indices)]=Inf
    @test_throws ArgumentError population_invariance(nonfinite,b)
    @test_throws ArgumentError PopulationPlan(nonfinite,b)

    integer_zero=zeros(Int,length(b),length(b))
    rational_zero=zeros(Rational{Int},length(b),length(b))
    @test eltype(PopulationPlan(integer_zero,b))===ComplexF64
    @test eltype(PopulationPlan(rational_zero,b))===ComplexF64

    b3=PIBasis(2,3)
    l10=zeros(ComplexF64,3,3);l10[1,2]=1
    qutrit_model=PIModel(b3,[LocalJump(l10;rate=0.2)])
    qutrit_plan=PopulationPlan(qutrit_model)
    @test qutrit_plan.invariance.invariant===true
    qutrit_rho=iid_pure_state(b3,ComplexF64[0,1,0])
    qutrit_p=diagonal_populations(qutrit_rho)
    qutrit_full=liouvillian(qutrit_model;representation=:sparse)*qutrit_rho.data
    @test qutrit_plan*qutrit_p≈diagonal_populations(PIState(b3,qutrit_full)) atol=3e-12

    # Direct and Appendix-D kernels use the same certified population scaling.
    pair_loss=kron(jm,jm)
    pair_zz=kron(jz,jz)
    direct_z=collective_spin(b,:z)
    direct_minus=collective_spin(b,:minus)
    diagonal_test_state=state_from_populations(b,p)
    for term in (DirectPIHamiltonian(direct_z),DirectPIJump(direct_minus),
                 PBodyHamiltonian(pair_zz,2),
                 LocalPBodyJump(pair_loss,2),
                 CollectivePBodyJump(pair_loss,2))
        term_model=PIModel(b,[term])
        term_plan=PopulationPlan(term_model)
        full_derivative=liouvillian(
            term_model;representation=:sparse)*diagonal_test_state.data
        @test term_plan*p≈diagonal_populations(
            PIState(b,full_derivative)) atol=4e-12 rtol=4e-12
    end

    b32=PIBasis(2,2)
    jm32=ComplexF32[0 1;0 0]
    plan32=PopulationPlan(PIModel(b32,[LocalJump(jm32;rate=Float32(0.2))]))
    @test eltype(plan32)===ComplexF32
    p32=diagonal_populations(iid_pure_state(b32,ComplexF32[0,1]))
    @test eltype(plan32*p32)===ComplexF32
    destination32=similar(p32)
    work32=PopulationWorkspace(plan32,p32)
    @test_throws ArgumentError evolve_populations!(
        destination32,plan32,p32,(0.0,0.2);steps=2,workspace=work32)
    evolve_populations!(destination32,plan32,p32,
        (Float32(0),Float32(0.2));steps=2,workspace=work32)
    promoted_solution=solve_populations(
        plan32,p32,(0.0,0.2);saveat=[0.0,0.2],steps_per_interval=2)
    @test eltype(promoted_solution[end])===ComplexF64

    huge_basis=PIBasis(200,2;sectors=[(100,100)])
    huge_state32=state_from_populations(huge_basis,Float32[1])
    @test all(isfinite,huge_state32.data)
    @test diagonal_populations(huge_state32)≈ComplexF32[1] rtol=8eps(Float32)
    fused_basis=PIBasis(2100,2;sectors=[(1050,1050)])
    fused_state=state_from_populations(fused_basis,Float64[1])
    @test trace(fused_state)≈1 atol=2e-15
    @test diagonal_populations(fused_state)≈ComplexF64[1] atol=2e-15
    unrepresentable_basis=PIBasis(400,2;sectors=[(200,200)])
    @test_throws ArgumentError state_from_populations(
        unrepresentable_basis,Float32[1])
    setprecision(192) do
        huge_state=state_from_populations(huge_basis,BigFloat[1])
        @test trace(huge_state)≈one(BigFloat) rtol=32eps(BigFloat)
        @test diagonal_populations(huge_state)≈BigFloat[1] rtol=32eps(BigFloat)

        big_jm=Complex{BigFloat}[0 1;0 0]
        big_plan=PopulationPlan(PIModel(PIBasis(2,2),[
            LocalJump(big_jm;rate=big"0.2")]))
        big_initial=diagonal_populations(iid_pure_state(
            big_plan.basis,Complex{BigFloat}[0,1]))
        big_destination=similar(big_initial)
        evolve_populations!(big_destination,big_plan,big_initial,
            (zero(BigFloat),BigFloat("0.1"));steps=4)
        @test eltype(big_destination)===Complex{BigFloat}
        @test sum(big_destination)≈one(BigFloat) rtol=128eps(BigFloat)
    end

    @test_throws DimensionMismatch state_from_populations(b,zeros(2))
    @test_throws ArgumentError state_from_populations(b,fill(-1.0,population_dimension(b));validate=true)
    @test_throws ArgumentError population_generator(plan;representation=:dense)
end
