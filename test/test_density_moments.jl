# These deliberately exponential helpers are test-only.  The largest dense
# matrices constructed below are 2^3-by-2^3 and 3^2-by-3^2; production moment
# routines must stay in Schur coordinates.
function _density_moment_oracle_schur_reference(N::Int,d::Int)
    PID=PermutationalInvariantDynamics
    rows=Matrix{Float64}[]
    labels=NamedTuple[]
    offset=1
    for sector in partitions(N,d)
        paths=PID._removal_paths(sector,N)
        @assert length(paths)==symmetric_group_dimension(sector)
        for path in paths
            tensor=PID._path_isometry(path)
            block=reshape(tensor,size(tensor,1),d^N)
            range=offset:offset+size(block,1)-1
            push!(rows,block)
            push!(labels,(;sector,range))
            offset=last(range)+1
        end
    end
    (;transform=reduce(vcat,rows),labels)
end

function _density_moment_oracle_dense(A)
    reference=_density_moment_oracle_schur_reference(A.basis.N,A.basis.d)
    T=promote_type(eltype(A.data),Float64)
    schur=zeros(T,size(reference.transform))
    for label in reference.labels
        schur[label.range,label.range].=
            Matrix(physical_block(A,label.sector))
    end
    reference.transform'*schur*reference.transform
end

function _density_moment_oracle_partial_trace(
        rho::AbstractMatrix,d::Int,N::Int,k::Int)
    da=d^k
    db=d^(N-k)
    reduced=zeros(eltype(rho),da,da)
    for a in 0:da-1,b in 0:da-1,q in 0:db-1
        reduced[a+1,b+1]+=rho[a*db+q+1,b*db+q+1]
    end
    reduced
end

function _density_moment_oracle_random_state(basis,rng)
    PID=PermutationalInvariantDynamics
    rho=PIState(basis;T=Float64)
    for partition in basis.sectors
        n=length(basis.patterns[basis.index[partition]])
        A=randn(rng,ComplexF64,n,n)
        coefficient_block(rho,partition).=
            A*A'+0.25*Matrix{ComplexF64}(I,n,n)
    end
    PID.normalize!(rho)
end

function _density_moment_workspace_arrays(work)
    AbstractArray[
        value for name in fieldnames(typeof(work))
        for value in (getfield(work,name),) if value isa AbstractArray]
end

@testset "density trace powers against bounded dense oracles" begin
    @test PermutationalInvariantDynamics.Workflow.trace_power===trace_power
    @test PermutationalInvariantDynamics.Workflow.reduced_trace_power===
          reduced_trace_power
    @test PermutationalInvariantDynamics.Workflow.reduced_trace_powers===
          reduced_trace_powers
    @test PermutationalInvariantDynamics.Workflow.DensityPowerWorkspace===
          DensityPowerWorkspace
    rng=MersenneTwister(0xd31517)
    for (N,d) in ((3,2),(2,3))
        basis=PIBasis(N,d)
        rho=_density_moment_oracle_random_state(basis,rng)
        source_before=copy(rho.data)
        dense=_density_moment_oracle_dense(rho)
        state_work=DensityPowerWorkspace(rho;memory_budget=Inf)
        basis_work=DensityPowerWorkspace(
            basis;T=Float64,memory_budget=Inf)

        for order in 1:6
            expected=real(tr(dense^order))
            @test trace_power(rho,order)≈expected atol=8e-11 rtol=8e-11
            @test trace_power(
                rho,order;workspace=state_work,check=false,
                memory_budget=Inf)≈expected atol=8e-11 rtol=8e-11
            @test trace_power(
                rho,order;workspace=basis_work,check=false,
                memory_budget=Inf)≈expected atol=8e-11 rtol=8e-11
        end
        @test trace_power(rho,1)==real(trace(rho))
        @test trace_power(rho,2)==purity(rho)
        @test trace_power(rho,3)≈
              2.0^(-2*renyi_entropy(rho,3)) atol=8e-11 rtol=8e-11

        for k in 1:N-1
            plan=ReductionPlan(basis,k)
            reduction_work=ReductionWorkspace(plan,rho;mode=:reduction)
            power_work=DensityPowerWorkspace(
                reduction_work;memory_budget=Inf)
            dense_reduced=_density_moment_oracle_partial_trace(
                dense,d,N,k)
            for order in 1:6
                expected=real(tr(dense_reduced^order))
                @test reduced_trace_power(rho,k,order;plan)≈
                    expected atol=1e-10 rtol=1e-10
                @test reduced_trace_power(
                    rho,k,order;plan,workspace=reduction_work,
                    power_workspace=power_work,check=false,
                    memory_budget=Inf)≈expected atol=1e-10 rtol=1e-10
            end
            @test reduced_trace_power(
                rho,k,2;plan,workspace=reduction_work,
                power_workspace=power_work)==
                reduced_purity(rho,k;plan,workspace=reduction_work)
        end
        @test rho.data==source_before
    end
end

@testset "analytic full and reduced density moments" begin
    for (N,d) in ((0,1),(0,2),(4,1))
        trivial=maximally_mixed_state(PIBasis(N,d))
        for order in 1:8
            @test trace_power(trivial,order)==1
            @test all(==(1),reduced_trace_powers(
                trivial,order;ks=0:N))
        end
    end

    for d in (2,3),N in 1:4
        basis=PIBasis(N,d)
        mixed=maximally_mixed_state(basis)
        plans=map(k->ReductionPlan(basis,k),0:N)
        reduction_workspaces=map(
            plan->ReductionWorkspace(plan,mixed;mode=:reduction),plans)
        power_workspaces=map(
            plan->DensityPowerWorkspace(
                plan.output_basis;T=Float64,memory_budget=Inf),plans)
        for order in 1:6
            @test trace_power(mixed,order)≈
                float(d)^(N*(1-order)) atol=8e-10 rtol=8e-10
            for (k,plan,reduction_work,power_work) in zip(
                    0:N,plans,reduction_workspaces,power_workspaces)
                @test reduced_trace_power(
                    mixed,k,order;plan,workspace=reduction_work,
                    power_workspace=power_work,check=false,
                    memory_budget=Inf)≈
                    float(d)^(k*(1-order)) atol=2e-9 rtol=2e-9
            end
        end
    end

    N=5
    basis=PIBasis(N,2)
    symmetric=Partition((N,0))
    patterns=basis.patterns[basis.index[symmetric]]
    vector=zeros(ComplexF64,length(patterns))
    vector[findfirst(pattern->content(pattern)==(N,0),patterns)]=inv(sqrt(2))
    vector[findfirst(pattern->content(pattern)==(0,N),patterns)]=inv(sqrt(2))
    ghz=sector_density_matrix(basis,symmetric,vector*vector')
    ghz_plans=map(k->ReductionPlan(basis,k),0:N)
    ghz_reduction_workspaces=map(
        plan->ReductionWorkspace(plan,ghz;mode=:reduction),ghz_plans)
    ghz_power_workspaces=map(
        plan->DensityPowerWorkspace(
            plan.output_basis;T=Float64,memory_budget=Inf),ghz_plans)
    for order in 1:8
        @test trace_power(ghz,order)≈1 atol=3e-11
        for (index,k) in pairs(0:N)
            expected=k==0||k==N ? 1.0 : 2.0^(1-order)
            @test reduced_trace_power(
                ghz,k,order;plan=ghz_plans[index],
                workspace=ghz_reduction_workspaces[index],
                power_workspace=ghz_power_workspaces[index],
                check=false,memory_budget=Inf)≈
                expected atol=3e-10 rtol=3e-10
        end
    end

    # A basis state in a Schur sector of multiplicity f represents the equal
    # mixture of f orthogonal physical copies.  This catches a missing or
    # inverted f^(1-order) factor in the stored-coefficient convention.
    mixed_sector=Partition((2,1))
    multiplicity_basis=PIBasis(3,2)
    multiplicity_state=basis_state(
        multiplicity_basis,mixed_sector,
        first(multiplicity_basis.patterns[
            multiplicity_basis.index[mixed_sector]]))
    f=float(symmetric_group_dimension(mixed_sector))
    for order in 1:8
        @test trace_power(multiplicity_state,order)≈
            f^(1-order) atol=4e-12 rtol=4e-12
    end

    # Exercise the binary-power index schedule well beyond the common q=3/4
    # cases in a nontrivial multiplicity sector and a dense eigenbasis.
    long_basis=PIBasis(4,2;sectors=[(3,1)])
    long_sector=only(long_basis.sectors)
    dimension=length(only(long_basis.patterns))
    unitary=Matrix(qr(randn(MersenneTwister(0x91d3),
                            ComplexF64,dimension,dimension)).Q)
    probabilities=Float64[0.6,0.3,0.1]
    long_f=Float64(symmetric_group_dimension(long_sector))
    physical=unitary*Diagonal(probabilities/long_f)*unitary'
    long_state=sector_density_matrix(long_basis,long_sector,physical)
    for order in 1:32
        expected=long_f^(1-order)*sum(probabilities.^order)
        @test isapprox(trace_power(long_state,order),expected;
                       atol=0,rtol=2e-10)
    end
end

@testset "prepared reduced trace-power batches" begin
    basis=PIBasis(3,2)
    rho=_density_moment_oracle_random_state(
        basis,MersenneTwister(0x3d0c7))
    ks=(2,0,1,3)
    plans=map(k->ReductionPlan(basis,k),ks)
    workspaces=map(
        plan->ReductionWorkspace(plan,rho;mode=:reduction),plans)
    power_workspaces=map(
        plan->DensityPowerWorkspace(
            plan.output_basis;T=Float64,memory_budget=Inf),plans)
    expected=[reduced_trace_power(rho,k,3;plan,workspace=work,
                                  power_workspace=power,check=false)
              for (k,plan,work,power) in
                  zip(ks,plans,workspaces,power_workspaces)]

    @test isempty(reduced_trace_powers(rho,3;ks=Int[]))
    @test reduced_trace_powers(rho,3;ks)≈expected atol=4e-11
    @test reduced_trace_powers(rho,2;ks)≈
        reduced_purities(rho;ks) atol=4e-11
    @test_throws ArgumentError reduced_trace_powers(
        rho,2;ks,plans,workspaces,memory_budget=1)
    @test reduced_trace_powers(
        rho,3;ks,plans,workspaces,power_workspaces,
        memory_budget=Inf)≈expected atol=4e-11

    PID=PermutationalInvariantDynamics
    retained=sum(
        BigInt(Base.summarysize(work))+power.retained_bytes
        for (work,power) in zip(workspaces,power_workspaces))
    extra=maximum(
        PID._density_power_peak_bytes(power,3)-power.retained_bytes
        for power in power_workspaces)
    individual=maximum(
        BigInt(Base.summarysize(work))+
            PID._density_power_peak_bytes(power,3)
        for (work,power) in zip(workspaces,power_workspaces))
    @test individual<retained+extra
    @test_throws ArgumentError reduced_trace_powers(
        rho,3;ks,plans,workspaces,power_workspaces,
        memory_budget=individual)
    @test reduced_trace_powers(
        rho,3;ks,plans,workspaces,power_workspaces,
        memory_budget=retained+extra)≈expected atol=4e-11

    plan_set=ReductionPlanSet(basis,ks)
    workspace_set=ReductionWorkspaceSet(
        plan_set,rho;mode=:reduction)
    set_power_workspaces=map(
        plan->DensityPowerWorkspace(
            plan.output_basis;T=Float64,memory_budget=Inf),
        plan_set.plans)
    @test collect(reduced_trace_powers(rho,plan_set,3))≈
        expected atol=4e-11
    @test collect(reduced_trace_powers(
        rho,plan_set,3;workspace=workspace_set,
        power_workspaces=set_power_workspaces,
        memory_budget=Inf))≈expected atol=4e-11
end

@testset "density-power resource and validation contracts" begin
    basis=PIBasis(3,2)
    rho=_density_moment_oracle_random_state(
        basis,MersenneTwister(0x711ce))
    first_work=DensityPowerWorkspace(rho;memory_budget=Inf)
    second_work=DensityPowerWorkspace(rho;memory_budget=Inf)
    first_arrays=_density_moment_workspace_arrays(first_work)
    second_arrays=_density_moment_workspace_arrays(second_work)
    @test !isempty(first_arrays)
    @test length(first_arrays)==length(second_arrays)
    @test all(!Base.mightalias(a,b)
              for (a,b) in zip(first_arrays,second_arrays))
    aliased_work=DensityPowerWorkspace(rho;memory_budget=Inf)
    aliased_work.second=aliased_work.first
    @test_throws ArgumentError trace_power(
        rho,3;workspace=aliased_work,check=false,memory_budget=Inf)
    trace_power(
        rho,3;workspace=first_work,check=false,memory_budget=Inf)
    trace_power(rho,3;check=false,memory_budget=Inf)
    prepared_bytes=@allocated trace_power(
        rho,3;workspace=first_work,check=false,memory_budget=Inf)
    automatic_bytes=@allocated trace_power(
        rho,3;check=false,memory_budget=Inf)
    @test prepared_bytes<automatic_bytes

    equivalent_basis=PIBasis(3,2)
    equivalent_state=PIState(equivalent_basis,copy(rho.data))
    wrong_basis_work=DensityPowerWorkspace(
        equivalent_basis;T=Float64,memory_budget=Inf)
    @test_throws ArgumentError trace_power(
        rho,3;workspace=wrong_basis_work,check=false)
    @test_throws ArgumentError trace_power(
        equivalent_state,3;workspace=first_work,check=false)

    narrow_work=DensityPowerWorkspace(
        basis;T=Float32,memory_budget=Inf)
    @test_throws ArgumentError trace_power(
        rho,3;workspace=narrow_work,check=false)
    @test_throws ArgumentError DensityPowerWorkspace(
        basis;T=Float64,memory_budget=1)
    huge_guard_basis=PIBasis(
        10_000_000,2;sectors=[(5_000_000,5_000_000)])
    @test_throws ArgumentError DensityPowerWorkspace(
        huge_guard_basis;T=Float64,memory_budget=1024)
    @test_throws ArgumentError trace_power(rho,3;memory_budget=1)
    @test_throws ArgumentError reduced_trace_power(
        rho,0,1;memory_budget=-1)
    @test_throws ArgumentError reduced_trace_powers(
        rho,1;ks=Int[],memory_budget=NaN)

    for invalid_order in (0,-1,true)
        @test_throws ArgumentError trace_power(rho,invalid_order)
        @test_throws ArgumentError reduced_trace_power(
            rho,1,invalid_order)
        @test_throws ArgumentError reduced_trace_powers(
            rho,invalid_order;ks=(0,1))
    end
    @test_throws ArgumentError trace_power(
        rho,big(typemax(Int))+1)

    bad=copy(rho)
    bad.data .*= 2
    @test_throws ArgumentError trace_power(bad,3)
    @test_throws ArgumentError reduced_trace_power(bad,1,3)
    @test trace_power(bad,3;check=false)≈
        8trace_power(rho,3;check=false) atol=4e-11

    plan=ReductionPlan(basis,1)
    reduction_work=ReductionWorkspace(plan,rho;mode=:reduction)
    both_work=ReductionWorkspace(plan,rho;mode=:both)
    negativity_work=ReductionWorkspace(plan,rho;mode=:negativity)
    output_power=DensityPowerWorkspace(
        plan.output_basis;T=Float64,memory_budget=Inf)
    PID=PermutationalInvariantDynamics
    reduction_bytes=BigInt(Base.summarysize(reduction_work))
    power_peak=PID._density_power_peak_bytes(output_power,3)
    separate_budget=max(reduction_bytes,power_peak)
    combined_budget=reduction_bytes+power_peak
    @test separate_budget<combined_budget
    @test_throws ArgumentError reduced_trace_power(
        rho,1,3;plan,workspace=reduction_work,
        power_workspace=output_power,check=false,
        memory_budget=separate_budget)
    @test reduced_trace_power(
        rho,1,3;plan,workspace=reduction_work,
        power_workspace=output_power,check=false,
        memory_budget=combined_budget)≈
        reduced_trace_power(rho,1,3;plan) atol=4e-11
    @test reduced_trace_power(
        rho,1,3;plan,workspace=reduction_work,
        power_workspace=output_power)≈
        reduced_trace_power(rho,1,3;plan,workspace=both_work) atol=4e-11
    @test_throws ArgumentError reduced_trace_power(
        rho,1,3;plan,workspace=negativity_work,
        power_workspace=output_power)
    @test_throws ArgumentError reduced_trace_power(
        rho,1,2;plan,power_workspace=output_power,memory_budget=1)
    @test_throws ArgumentError DensityPowerWorkspace(
        negativity_work;memory_budget=Inf)

    other_plan=ReductionPlan(basis,1)
    other_reduction=ReductionWorkspace(other_plan,rho;mode=:reduction)
    @test_throws ArgumentError reduced_trace_power(
        rho,1,3;plan,workspace=other_reduction,
        power_workspace=output_power)
    @test_throws ArgumentError reduced_trace_power(
        rho,1,3;plan,
        power_workspace=DensityPowerWorkspace(
            basis;T=Float64,memory_budget=Inf))

    # Endpoint shortcuts must still validate every explicitly supplied
    # resource instead of bypassing ownership checks.
    zero_plan=ReductionPlan(basis,0)
    full_plan=ReductionPlan(basis,basis.N)
    @test_throws ArgumentError DensityPowerWorkspace(
        ReductionWorkspace(full_plan,rho;mode=:reduction);
        memory_budget=Inf)
    @test_throws ArgumentError reduced_trace_power(
        rho,0,3;plan=plan)
    @test_throws ArgumentError reduced_trace_power(
        rho,basis.N,3;plan=zero_plan)
    @test_throws ArgumentError reduced_trace_power(
        rho,0,3;plan=zero_plan,power_workspace=output_power)
    @test reduced_trace_power(
        rho,basis.N,3;plan=full_plan,
        power_workspace=first_work)≈trace_power(rho,3) atol=4e-11

    ks=(0,1,3)
    plans=(zero_plan,plan,full_plan)
    @test_throws DimensionMismatch reduced_trace_powers(
        rho,3;ks,plans=plans[1:2])
    @test_throws DimensionMismatch reduced_trace_powers(
        rho,3;ks,plans,workspaces=(nothing,))
    @test_throws DimensionMismatch reduced_trace_powers(
        rho,3;ks,plans,power_workspaces=(nothing,))
    @test_throws ArgumentError reduced_trace_powers(
        rho,3;ks=(-1,1))

    set=ReductionPlanSet(basis,ks)
    set_work=ReductionWorkspaceSet(set,rho;mode=:reduction)
    other_set=ReductionPlanSet(basis,ks)
    other_set_work=ReductionWorkspaceSet(
        other_set,rho;mode=:reduction)
    @test_throws ArgumentError reduced_trace_powers(
        rho,set,3;workspace=other_set_work)
    @test_throws DimensionMismatch reduced_trace_powers(
        rho,set,3;workspace=set_work,power_workspaces=(nothing,))

    # Nonzero values outside the requested scalar range raise rather than
    # silently becoming zero or Inf.  A meaningful imaginary trace is also an
    # invalid real density moment when validation was explicitly skipped.
    one_basis=PIBasis(1,2)
    tiny=PIState(one_basis;T=Float64)
    coefficient_block(tiny,only(one_basis.sectors))[1,1]=1e-200
    @test_throws ArgumentError trace_power(tiny,3;check=false)
    huge=PIState(one_basis;T=Float64)
    coefficient_block(huge,only(one_basis.sectors))[1,1]=1e200
    @test_throws ArgumentError trace_power(huge,3;check=false)
    complex_moment=PIState(one_basis;T=Float64)
    coefficient_block(
        complex_moment,only(one_basis.sectors))[1,1]=im
    @test_throws ArgumentError trace_power(complex_moment,3;check=false)

    # Exponentiating an exact sector multiplicity is also memory-guarded.
    large_basis=PIBasis(10,2;sectors=[(5,5)])
    large_state=basis_state(
        large_basis,only(large_basis.sectors),
        first(only(large_basis.patterns)))
    @test_throws ArgumentError trace_power(
        large_state,1_000_000_000;memory_budget=1024)

    # Huge f with a one-dimensional retained irrep: q=3 is still
    # representable even though direct combinatorial conversion is not.
    huge_f_basis=PIBasis(500,2;sectors=[(250,250)])
    huge_f_sector=only(huge_f_basis.sectors)
    huge_f_state=basis_state(
        huge_f_basis,huge_f_sector,only(only(huge_f_basis.patterns)))
    huge_f=symmetric_group_dimension(huge_f_sector)
    expected_huge=setprecision(BigFloat,256) do
        Float64(inv(BigFloat(huge_f)^2))
    end
    @test trace_power(huge_f_state,3)==expected_huge
    @test_throws ArgumentError trace_power(huge_f_state,4)
end

@testset "density-power scalar precision" begin
    basis16=PIBasis(2,2)
    rho16=iid_pure_state(
        basis16,normalize(Complex{Float16}[1,im]))
    moment16=trace_power(rho16,3;memory_budget=Inf)
    @test moment16 isa Float16
    @test moment16≈one(Float16) atol=8eps(Float16)

    basis32=PIBasis(2,2)
    rho32=iid_pure_state(
        basis32,normalize(ComplexF32[1,2im]))
    work32=DensityPowerWorkspace(rho32;memory_budget=Inf)
    moment32=trace_power(
        rho32,5;workspace=work32,memory_budget=Inf)
    @test moment32 isa Float32
    @test moment32≈one(Float32) atol=8eps(Float32)
    @test reduced_trace_power(rho32,1,1) isa Float32
    @test eltype(reduced_trace_powers(rho32,1;ks=0:2))===Float32

    basis=PIBasis(3,2)
    rho128=setprecision(BigFloat,128) do
        iid_pure_state(
            basis,Complex{BigFloat}[
                BigFloat("0.6"),BigFloat("0.8")*im])
    end
    work128=setrounding(BigFloat,RoundDown) do
        setprecision(BigFloat,64) do
            DensityPowerWorkspace(rho128;memory_budget=Inf)
        end
    end
    low=setprecision(BigFloat,64) do
        trace_power(
            rho128,5;workspace=work128,check=false,
            memory_budget=Inf)
    end
    high=setprecision(BigFloat,256) do
        trace_power(
            rho128,5;workspace=work128,check=false,
            memory_budget=Inf)
    end
    @test low==high
    @test precision(low)==precision(high)==128
    @test low≈one(BigFloat) atol=big"2e-35"

    rho192=setprecision(BigFloat,192) do
        iid_pure_state(
            basis,Complex{BigFloat}[
                BigFloat("0.6"),BigFloat("0.8")*im])
    end
    @test_throws ArgumentError trace_power(
        rho192,3;workspace=work128,check=false)

    plan=ReductionPlan(basis,1)
    reduction_work=ReductionWorkspace(
        plan,rho128;mode=:reduction)
    power128=setprecision(BigFloat,128) do
        DensityPowerWorkspace(
            plan.output_basis;T=BigFloat,memory_budget=Inf)
    end
    reduced_low=setprecision(BigFloat,64) do
        reduced_trace_power(
            rho128,1,5;plan,workspace=reduction_work,
            power_workspace=power128,check=false,
            memory_budget=Inf)
    end
    reduced_high=setprecision(BigFloat,256) do
        reduced_trace_power(
            rho128,1,5;plan,workspace=reduction_work,
            power_workspace=power128,check=false,
            memory_budget=Inf)
    end
    @test reduced_low==reduced_high
    @test precision(reduced_low)==precision(reduced_high)==128

    power64=setprecision(BigFloat,64) do
        DensityPowerWorkspace(
            plan.output_basis;T=BigFloat,memory_budget=Inf)
    end
    @test_throws ArgumentError reduced_trace_power(
        rho128,1,3;plan,workspace=reduction_work,
        power_workspace=power64,check=false)
end
