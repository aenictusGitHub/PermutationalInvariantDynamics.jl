@testset "PI full counting statistics" begin
    basis=PIBasis(1,2)
    spin=spin_matrices()
    gamma_down=0.7
    gamma_up=0.3
    model=PIModel(basis,(
        LocalJump(spin.jm;rate=gamma_down),
        LocalJump(spin.jp;rate=gamma_up),
    ))
    compiled=compile(model;backend=:matrixfree)
    plan=TiltedLiouvillianPlan(compiled;channels=1)
    workspace=TiltedLiouvillianWorkspace(plan)

    @test size(plan)==(length(basis),length(basis))
    @test plan.channels==(1,)
    @test plan.increments==(1.0,)
    @test PermutationalInvariantDynamics._operator_basis(plan)===basis
    @test PermutationalInvariantDynamics._operator_trace_functional(plan)===
          plan.trajectory.liouvillian.tracevec

    source=ComplexF64[0.3+0.1im,0.2-0.4im,0.5+0.3im,-0.2+0.1im]
    zero_field=similar(source)
    reference=similar(source)
    apply_tilted!(zero_field,plan,source,0.0,workspace)
    apply!(reference,plan.trajectory.liouvillian,source,
           workspace.liouvillian)
    @test zero_field≈reference atol=2e-14 rtol=2e-14

    field=0.23
    operator=tilted_liouvillian(plan,field)
    image=operator*source
    adjoint_probe=ComplexF64[-0.1+0.2im,0.4+0.1im,0.3-0.2im,0.7]
    @test dot(adjoint_probe,image)≈
          dot(adjoint(operator)*adjoint_probe,source) atol=3e-13 rtol=3e-13

    # One-site pumping and emission reduce on populations to a tilted
    # two-state Markov generator. Its dominant eigenvalue is analytic.
    analytic(s)=(-(gamma_down+gamma_up)+sqrt(
        (gamma_down-gamma_up)^2+
        4gamma_down*gamma_up*exp(s)))/2
    scgf=counting_scgf(
        plan,field;krylovdim=length(basis),atol=1e-13,rtol=1e-13)
    @test real(scgf)≈analytic(field) atol=2e-12 rtol=2e-11
    @test abs(imag(scgf))<2e-12
    @test counting_scgf(plan,0.0)==0
    arnoldi=ArnoldiWorkspace(
        ComplexF64,length(basis),length(basis);mode=:ordinary)
    reused_scgf=counting_scgf(
        plan,field;krylovdim=length(basis),workspace=arnoldi,
        atol=1e-13,rtol=1e-13,return_info=true)
    @test reused_scgf.value≈scgf
    @test reused_scgf.spectrum.workspace_reused

    cumulants=counting_cumulants(
        plan;step=2e-3,krylovdim=length(basis),
        atol=1e-13,rtol=1e-13)
    expected_current=gamma_down*gamma_up/(gamma_down+gamma_up)
    expected_noise=expected_current-
        2(gamma_down*gamma_up)^2/(gamma_down+gamma_up)^3
    @test cumulants.current≈expected_current atol=2e-8 rtol=2e-8
    @test cumulants.noise≈expected_noise atol=3e-7 rtol=3e-7

    curve=counting_scgf_curve(
        plan,-0.3:0.1:0.3;krylovdim=length(basis),
        atol=1e-13,rtol=1e-13)
    raw_curve=counting_scgf_curve(
        model,-0.1:0.1:0.1;channels=1,krylovdim=length(basis),
        atol=1e-13,rtol=1e-13,memory_budget=Inf)
    @test raw_curve.values≈
        counting_scgf_curve(
            plan,-0.1:0.1:0.1;krylovdim=length(basis),
            atol=1e-13,rtol=1e-13,memory_budget=Inf).values
    rate=large_deviation_rate_function(
        curve,[expected_current])
    @test rate.rates[1]>=-2e-12
    @test !rate.boundary_maxima[1]

    # Pure decay permits at most one emitted quantum:
    # M(s,t)=exp(-gamma*t)+(1-exp(-gamma*t))*exp(s).
    decay_model=PIModel(
        basis,(LocalJump(spin.jm;rate=gamma_down),))
    decay_plan=TiltedLiouvillianPlan(decay_model)
    excited=computational_product_state(basis,2)
    duration=0.8
    finite=finite_time_mgf(
        decay_plan,excited,duration,field;
        krylovdim=length(basis),atol=1e-13,rtol=1e-13)
    exact=exp(-gamma_down*duration)+
          (1-exp(-gamma_down*duration))*exp(field)
    @test real(finite)≈exact atol=3e-11 rtol=3e-11
    @test abs(imag(finite))<3e-11
    tilted_work=TiltedLiouvillianWorkspace(decay_plan)
    expv_work=KrylovExpvWorkspace(
        ComplexF64,length(basis),length(basis))
    finite_reused=finite_time_mgf(
        decay_plan,excited,duration,field;
        workspace=tilted_work,expv_workspace=expv_work,
        atol=1e-13,rtol=1e-13,return_info=true)
    @test finite_reused.mgf≈finite
    @test finite_reused.krylov.workspace_reused
    exact_work=TiltedLiouvillianWorkspace(decay_plan)
    exact_expv=KrylovExpvWorkspace(
        ComplexF64,length(basis),length(basis))
    scalar_budget=
        PermutationalInvariantDynamics._counting_finite_time_mgf_peak_bytes(
            decay_plan,exact_work,length(basis),ComplexF64,
            length(basis),false)
    info_budget=
        PermutationalInvariantDynamics._counting_finite_time_mgf_peak_bytes(
            decay_plan,exact_work,length(basis),ComplexF64,
            length(basis),true)
    @test info_budget-scalar_budget==
        PermutationalInvariantDynamics._performance_entries_bytes(
            length(basis),ComplexF64)
    @test finite_time_mgf(
        decay_plan,excited,duration,field;
        workspace=exact_work,expv_workspace=exact_expv,
        memory_budget=scalar_budget,atol=1e-13,rtol=1e-13)≈finite
    @test_throws ArgumentError finite_time_mgf(
        decay_plan,excited,duration,field;
        workspace=exact_work,expv_workspace=exact_expv,
        return_info=true,memory_budget=scalar_budget,
        atol=1e-13,rtol=1e-13)
    exact_info=finite_time_mgf(
        decay_plan,excited,duration,field;
        workspace=exact_work,expv_workspace=exact_expv,
        return_info=true,memory_budget=info_budget,
        atol=1e-13,rtol=1e-13)
    @test exact_info.state.data !== exact_info.krylov.value

    PermutationalInvariantDynamics._ensure_batch_capacity!(
        tilted_work.liouvillian.batch,32)
    base_estimate=
        PermutationalInvariantDynamics._performance_krylov_expv_workspace_bytes(
            length(basis),ComplexF64,length(basis))+
        PermutationalInvariantDynamics._performance_tilted_workspace_bytes(
            decay_plan)+
        PermutationalInvariantDynamics._performance_entries_bytes(
            length(basis),ComplexF64)
    budget_error=try
        finite_time_mgf(
            decay_plan,excited,duration,field;
            workspace=tilted_work,expv_workspace=expv_work,
            memory_budget=base_estimate)
        nothing
    catch error
        error
    end
    @test budget_error isa ArgumentError
    @test occursin("memory_budget",sprint(showerror,budget_error))

    @test_throws ArgumentError TiltedLiouvillianPlan(
        compiled;channels=Int[])
    @test_throws ArgumentError TiltedLiouvillianPlan(
        compiled;channels=[1,1])
    @test_throws ArgumentError TiltedLiouvillianPlan(
        compiled;channels=[big(typemax(Int))+1])
    @test_throws DimensionMismatch TiltedLiouvillianPlan(
        compiled;channels=[1,2],increments=[1])
    @test_throws ArgumentError counting_scgf(plan,0.1im)
    @test_throws ArgumentError counting_scgf(plan,field;atol=true)
    @test_throws ArgumentError counting_scgf(plan,field;atol=-1.0)
    @test_throws ArgumentError counting_scgf(
        plan,field;atol=big"1e-1000")
    @test_throws ArgumentError large_deviation_rate_function(
        curve,[expected_current];rtol=true)
    @test_throws ArgumentError finite_time_mgf(
        decay_plan,excited,-0.1,field)
    @test_throws ArgumentError counting_scgf(
        plan,field;krylovdim=length(basis),memory_budget=0)
    raw_guard=try
        counting_scgf(
            model,field;krylovdim=length(basis),memory_budget=0)
        nothing
    catch error
        error
    end
    @test raw_guard isa ArgumentError
    @test occursin("model preparation",sprint(showerror,raw_guard))
    consumed=Ref(0)
    guarded_fields=(
        (consumed[]+=1; value) for value in range(-0.2,0.2;length=5))
    @test_throws ArgumentError counting_scgf_curve(
        plan,guarded_fields;krylovdim=length(basis),memory_budget=0)
    @test consumed[]==0
    @test_throws ArgumentError finite_time_mgf(
        decay_plan,excited,duration,field;memory_budget=0)
    @test_throws ArgumentError getfield(
        PermutationalInvariantDynamics,:_counting_tilt_factor)(
            0.5f0,nextfloat(0.0f0),ComplexF32)
    spin32=spin_matrices(2;T=Float32)
    model32=PIModel(
        basis,(LocalJump(spin32.jm;rate=0.2f0),))
    plan32=TiltedLiouvillianPlan(
        model32;increments=nextfloat(0.0f0))
    input32=ComplexF32.(source)
    output32=similar(input32)
    @test_throws ArgumentError apply_tilted!(
        output32,plan32,input32,0.5f0,TiltedLiouvillianWorkspace(plan32))
    driven=PIModel(basis,(
        LocalJump(spin.jm;rate=(time,parameters)->parameters),))
    @test_throws ArgumentError TiltedLiouvillianPlan(driven)
    negative=PIModel(
        basis,(LocalJump(spin.jm;rate=-0.1),))
    @test_throws ArgumentError TiltedLiouvillianPlan(negative)
end
