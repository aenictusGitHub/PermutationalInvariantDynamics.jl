function _pseudomode_pbody_lift_oracle(operator,ds::Int,da::Int)
    dsite=ds*da
    oracle=zeros(promote_type(ComplexF64,eltype(operator)),dsite^2,dsite^2)
    for system_row_1 in 1:ds,system_row_2 in 1:ds,
            system_column_1 in 1:ds,system_column_2 in 1:ds
        system_row=system_row_2+ds*(system_row_1-1)
        system_column=system_column_2+ds*(system_column_1-1)
        value=operator[system_row,system_column]
        iszero(value)&&continue
        for auxiliary_1 in 1:da,auxiliary_2 in 1:da
            local_row_1=auxiliary_1+da*(system_row_1-1)
            local_row_2=auxiliary_2+da*(system_row_2-1)
            local_column_1=auxiliary_1+da*(system_column_1-1)
            local_column_2=auxiliary_2+da*(system_column_2-1)
            row=local_row_2+dsite*(local_row_1-1)
            column=local_column_2+dsite*(local_column_1-1)
            oracle[row,column]=value
        end
    end
    oracle
end

@testset "generic PI supersites and local ordering" begin
    site=PISupersite(
        2,(2,3);labels=(:system,:auxiliary),T=Float32)
    @test site.factor_dimensions==(2,3)
    @test site.factor_labels==(:system,:auxiliary)
    @test site.factor_specifications==(nothing,nothing)
    @test site.basis.N==2
    @test site.basis.d==6
    @test length(site.basis)==commutant_dimension(2,6)
    @test eltype(site)===ComplexF32
    @test all(identity->eltype(identity)===ComplexF32,site.identities)
    @test all(issparse,site.identities)

    existing_basis=PIBasis(1,6)
    existing_site=PISupersite(
        existing_basis,(2,3);
        labels=(:system,:auxiliary),T=Float32)
    @test existing_site.basis===existing_basis

    A=ComplexF32[1 2im;-2im 3]
    B=ComplexF32[0 1 0;1 0 2;0 2 1]
    @test supersite_tensor_operator(site,A,B)==kron(A,B)
    @test supersite_tensor_operator(site,(A,B))==kron(A,B)
    @test lift_supersite_operator(site,A;factor=1)==
          kron(A,Matrix{ComplexF32}(I,3,3))
    @test lift_supersite_operator(site,B;factor=:auxiliary)==
          kron(Matrix{ComplexF32}(I,2,2),B)
    @test lift_system_operator(site,A)==
          lift_supersite_operator(site,A;factor=:system)
    sparse_wrapper=adjoint(sparse(A))
    wrapped=supersite_tensor_operator(site,sparse_wrapper,sparse(B))
    @test issparse(wrapped)
    @test Matrix(wrapped)==kron(Matrix(sparse_wrapper),B)

    @test_throws ArgumentError PISupersite(true,(2,2))
    @test_throws ArgumentError PISupersite(1,(2,false))
    @test_throws ArgumentError PISupersite(1,())
    @test_throws DimensionMismatch PISupersite(
        1,(2,2);labels=(:only_one,))
    @test_throws ArgumentError PISupersite(
        1,(2,2);labels=(:same,:same))
    @test_throws DimensionMismatch supersite_tensor_operator(site,A)
    @test_throws DimensionMismatch supersite_tensor_operator(
        site,A,ones(ComplexF32,2,2))
    @test_throws ArgumentError supersite_tensor_operator(
        site,A,fill(ComplexF32(NaN),3,3))
    @test_throws ArgumentError lift_supersite_operator(
        site,A;factor=:missing)
    @test_throws ArgumentError PISupersite(
        1,(2,2);memory_budget=1)
    @test_throws ArgumentError PISupersite(
        100,(2,2);memory_budget=512*1024^2)
    @test_throws ArgumentError supersite_tensor_operator(
        site,A,B;memory_budget=1)
end

@testset "bosonic pseudomodes and multiple local modes" begin
    mode=BosonicPseudomode(
        2;frequency=-1.25f0,damping=0.4f0,
        thermal_occupation=0.2f0,label=:cavity)
    @test eltype(mode)===ComplexF32
    @test (mode.nmax,mode.levels)==(2,3)
    @test mode.frequency===-1.25f0
    @test mode.annihilation≈ComplexF32[
        0 1 0
        0 0 sqrt(2f0)
        0 0 0]
    @test mode.creation==mode.annihilation'
    @test mode.number_operator≈Diagonal(ComplexF32[0,1,2])
    @test mode.parity==Diagonal(ComplexF32[1,-1,1])
    @test mode.top_projector==Diagonal(ComplexF32[0,0,1])
    @test mode.vacuum==ComplexF32[1,0,0]
    @test all(issparse,(mode.identity,mode.annihilation,mode.creation,
                        mode.number_operator,mode.parity,
                        mode.top_projector))

    # An explicit scalar type must control integer-valued default parameters
    # without an unintended Float64 promotion.
    @test try
        eltype(BosonicPseudomode(1;T=Float32))===ComplexF32
    catch
        false
    end

    second=BosonicPseudomode(
        1;frequency=0.7f0,damping=0.1f0,label=:reaction)
    site=pseudomode_supersite(1,2,mode,second)
    @test site.factor_dimensions==(2,3,2)
    @test site.factor_labels==(:system,:cavity,:reaction)
    @test site.factor_specifications[2]===mode
    @test site.factor_specifications[3]===second
    @test site.basis.d==12

    first_operators=pseudomode_operators(site,:cavity)
    second_operators=pseudomode_operators(site,2)
    I2=Matrix{ComplexF32}(I,2,2)
    I3=Matrix{ComplexF32}(I,3,3)
    @test first_operators.annihilation==
          kron(I2,kron(mode.annihilation,I2))
    @test second_operators.number_operator==
          kron(I2,kron(I3,second.number_operator))
    @test lift_pseudomode_operator(
        site,mode.parity;mode=:cavity)==first_operators.parity

    tuple_site=pseudomode_supersite(1,2,(mode,second))
    @test tuple_site.factor_dimensions==site.factor_dimensions
    existing=pseudomode_supersite(
        site.basis,2,mode,second)
    @test existing.basis===site.basis
    @test_throws ArgumentError pseudomode_supersite(
        1,2,mode,BosonicPseudomode(1;label=:cavity))
    @test_throws ArgumentError pseudomode_supersite(
        1,2,BosonicPseudomode(1;label=:system))
    @test_throws BoundsError pseudomode_operators(site,3)
    @test_throws ArgumentError pseudomode_operators(site,:system)

    zero_mode=BosonicPseudomode(0)
    @test zero_mode.levels==1
    @test iszero(zero_mode.annihilation)
    @test zero_mode.vacuum==ComplexF64[1]
    @test_throws ArgumentError BosonicPseudomode(-1)
    @test_throws ArgumentError BosonicPseudomode(
        1;damping=-0.1)
    @test_throws ArgumentError BosonicPseudomode(
        1;thermal_occupation=-0.1)
    @test_throws ArgumentError BosonicPseudomode(
        1;frequency=NaN)
    @test_throws ArgumentError BosonicPseudomode(
        1;frequency=big"0.1",T=Float32)
    tiny=1//(big(10)^1000)
    huge=big(10)^1000
    @test_throws ArgumentError BosonicPseudomode(
        1;frequency=tiny,T=Float64)
    @test_throws ArgumentError BosonicPseudomode(
        1;frequency=huge,T=Float64)
end

@testset "complex pseudomode couplings" begin
    mode=BosonicPseudomode(1;label=:mode)
    site=pseudomode_supersite(1,2,mode)
    lowering=ComplexF64[0 1;0 0]
    g=0.31+0.17im
    h=-0.09+0.23im
    coupling=PseudomodeCoupling(
        lowering;mode=:mode,strength=g,
        counterrotating_strength=h)
    stored=copy(coupling.operator)
    lowering[1,2]=2
    @test coupling.operator==stored
    terms=pseudomode_coupling_terms(site,coupling)
    @test length(terms)==4
    @test all(term->term isa LocalHamiltonian,terms)
    @test all(term->term.rate isa Real,terms)
    @test all(term->ishermitian(term.operator),terms)

    L=stored
    a=mode.annihilation
    expected=g*kron(L,mode.creation)+conj(g)*kron(L',a)+
             h*kron(L,a)+conj(h)*kron(L',mode.creation)
    assembled=sum(
        (term.rate*term.operator for term in terms);
        init=zeros(ComplexF64,4,4))
    @test assembled≈expected atol=2e-15 rtol=2e-15

    zero_coupling=PseudomodeCoupling(
        stored;strength=0,counterrotating_strength=0)
    @test isempty(pseudomode_coupling_terms(site,zero_coupling))
    @test_throws DimensionMismatch pseudomode_coupling_terms(
        site,PseudomodeCoupling(ones(ComplexF64,3,3)))
    @test_throws ArgumentError PseudomodeCoupling(
        stored;strength=Inf)
    @test_throws ArgumentError PseudomodeCoupling(
        stored;strength=true)
    @test_throws ArgumentError PseudomodeCoupling(
        stored;strength=1//(big(10)^1000))

    # Integer-valued omitted coefficients must not widen a Float32 coupling.
    @test try
        coupling32=PseudomodeCoupling(
            ComplexF32[0 1;0 0];strength=0.2f0)
        eltype(coupling32)===ComplexF32
    catch
        false
    end
end

@testset "interleaved system p-body lifting and terms" begin
    mode=BosonicPseudomode(1)
    site=pseudomode_supersite(2,2,mode)
    X=ComplexF64[0 1;1 0]
    Z=ComplexF64[1 0;0 -1]
    lowering=ComplexF64[0 1;0 0]
    pair=kron(X,Z)+kron(Z,X)+0.2kron(X,X)
    lifted=lift_system_pbody_operator(site,pair,2)
    oracle=_pseudomode_pbody_lift_oracle(pair,2,2)
    @test issparse(lifted)
    @test Matrix(lifted)==oracle
    @test nnz(lifted)==nnz(sparse(pair))*2^2
    @test lifted!=kron(pair,Matrix{ComplexF64}(I,4,4))
    @test_throws ArgumentError lift_system_pbody_operator(
        site,pair,0)
    @test_throws DimensionMismatch lift_system_pbody_operator(
        site,ones(3,3),2)
    @test_throws ArgumentError lift_system_pbody_operator(
        site,pair,2;memory_budget=1)
    wrapped_pair=adjoint(sparse(pair))
    @test Matrix(lift_system_pbody_operator(
        site,wrapped_pair,2))==
        _pseudomode_pbody_lift_oracle(Matrix(wrapped_pair),2,2)

    local_hamiltonian=LocalHamiltonian(X;rate=0.3,hbar=2)
    lifted_hamiltonian=lift_system_term(site,local_hamiltonian)
    @test lifted_hamiltonian isa LocalHamiltonian
    @test lifted_hamiltonian.operator==
          kron(X,Matrix{ComplexF64}(I,2,2))
    @test lifted_hamiltonian.rate===local_hamiltonian.rate
    @test lifted_hamiltonian.hbar===local_hamiltonian.hbar

    collective_jump=CollectiveJump(lowering;rate=0.4)
    lifted_jump=lift_system_term(site,collective_jump)
    @test lifted_jump isa CollectiveJump
    @test lifted_jump.operator==
          kron(lowering,Matrix{ComplexF64}(I,2,2))
    @test lifted_jump.rate===collective_jump.rate

    pair_hamiltonian=PBodyHamiltonian(
        pair,2;rate=-0.7,hbar=3)
    lifted_pair=lift_system_term(site,pair_hamiltonian)
    @test lifted_pair isa PBodyHamiltonian
    @test lifted_pair.p==2
    @test lifted_pair.operator==lifted
    @test lifted_pair.rate===pair_hamiltonian.rate
    @test lifted_pair.hbar===pair_hamiltonian.hbar

    pair_jump=LocalPBodyJump(kron(lowering,lowering),2;rate=0.12)
    lifted_pair_jump=lift_system_term(site,pair_jump)
    @test lifted_pair_jump isa LocalPBodyJump
    @test lifted_pair_jump.p==2
    @test lifted_pair_jump.rate===pair_jump.rate
    @test Matrix(lifted_pair_jump.operator)==
          _pseudomode_pbody_lift_oracle(
              kron(lowering,lowering),2,2)

    direct=DirectPIHamiltonian(identity_operator(PIBasis(2,2)))
    @test_throws ArgumentError lift_system_term(site,direct)
    @test_throws ArgumentError lift_system_term(
        site,LocalJump((time,parameters)->lowering))
end

@testset "supersite product states and pseudomode trace" begin
    mode_a=BosonicPseudomode(1;label=:a)
    mode_b=BosonicPseudomode(1;label=:b)
    site=pseudomode_supersite(2,2,mode_a,mode_b)
    psi=ComplexF64[inv(sqrt(2)),im*inv(sqrt(2))]
    excited=ComplexF64[0,1]

    pure=supersite_product_state(
        site,psi,mode_a.vacuum,excited)
    local_ket=kron(psi,kron(mode_a.vacuum,excited))
    @test pure.data≈iid_pure_state(site.basis,local_ket).data

    rho_system=ComplexF64[0.7 0.1im;-0.1im 0.3]
    mixed=pseudomode_product_state(
        site,rho_system;mode_states=(mode_a.vacuum,excited))
    local_density=kron(
        rho_system,kron(
            mode_a.vacuum*mode_a.vacuum',
            excited*excited'))
    @test mixed.data≈iid_state(site.basis,local_density).data
    @test trace(mixed)≈1

    vacuum_product=pseudomode_product_state(site,psi)
    vacuum_ket=kron(psi,kron(mode_a.vacuum,mode_b.vacuum))
    @test vacuum_product.data≈
          iid_pure_state(site.basis,vacuum_ket).data

    plan=pseudomode_trace_plan(site)
    workspace=LocalFactorTraceWorkspace(plan)
    traced=trace_pseudomodes(
        mixed,site;plan,workspace)
    expected=iid_state(plan.output_basis,rho_system)
    @test traced.basis===plan.output_basis
    @test traced.data≈expected.data atol=4e-11 rtol=4e-11
    @test plan.local_dimensions==(2,4)
    @test plan.traced_factor==2
    traced_inplace=PIState(plan.output_basis)
    @test trace_pseudomodes!(
        traced_inplace,mixed,site,plan,workspace)===traced_inplace
    @test traced_inplace.data≈traced.data

    # This local ket is entangled between the system and mode `a`, so it
    # cannot be expressed through the factor-product convenience constructor.
    correlated_ket=zeros(ComplexF64,site.basis.d)
    correlated_ket[1]=inv(sqrt(2))
    correlated_ket[7]=inv(sqrt(2)) # |system=2, a=2, b=1>
    correlated=supersite_iid_state(site,correlated_ket)
    @test correlated.data≈
          iid_pure_state(site.basis,correlated_ket).data
    correlated_system=trace_pseudomodes(
        correlated,site;plan,workspace)
    @test correlated_system.data≈
          iid_state(
              plan.output_basis,
              Matrix{ComplexF64}(I,2,2)/2).data atol=4e-11 rtol=4e-11

    other_site=pseudomode_supersite(2,2,mode_a,mode_b)
    @test_throws ArgumentError trace_pseudomodes(mixed,other_site)
    @test_throws DimensionMismatch supersite_product_state(
        site,psi,mode_a.vacuum)
    @test_throws ArgumentError supersite_product_state(
        site,psi,mode_a.vacuum,excited;memory_budget=1)
    @test_throws ArgumentError pseudomode_trace_plan(
        site;memory_budget=1)
    @test_throws DimensionMismatch supersite_iid_state(
        site,ones(ComplexF64,site.basis.d-1))
    @test_throws ArgumentError supersite_iid_state(
        site,correlated_ket;memory_budget=1)

    # A caller-provided plan must implement this site's system trace, not an
    # arbitrary factorization of the same local dimension.
    wrong_plan=LocalFactorTracePlan(
        site.basis,(4,2);traced_factor=2)
    @test_throws ArgumentError trace_pseudomodes(
        mixed,site;plan=wrong_plan)
end

@testset "pseudomode precision and checked scales" begin
    lowering=ComplexF64[0 1;0 0]
    overflow_mode=BosonicPseudomode(
        1;damping=floatmax(Float64),thermal_occupation=1.0)
    overflow_site=pseudomode_supersite(1,2,overflow_mode)
    @test_throws ArgumentError pseudomode_damping_terms(
        overflow_site)

    underflow_mode=BosonicPseudomode(
        1;damping=nextfloat(0.0),thermal_occupation=0.5)
    underflow_site=pseudomode_supersite(1,2,underflow_mode)
    @test_throws ArgumentError pseudomode_damping_terms(
        underflow_site)

    prepared=setprecision(BigFloat,256) do
        mode=BosonicPseudomode(
            1;frequency=big"0.7",damping=big"0.2",
            label=:wide,T=BigFloat)
        coupling=PseudomodeCoupling(
            Complex{BigFloat}.([0 1;0 0]);
            mode=:wide,strength=big"0.125")
        site=pseudomode_supersite(1,2,mode)
        rho=pseudomode_product_state(
            site,Complex{BigFloat}[1,0])
        (;mode,coupling,site,rho)
    end
    @test prepared.site.estimates.precision_bits==256

    lifted=setprecision(BigFloat,64) do
        lift_pseudomode_operator(
            prepared.site,prepared.mode.annihilation;mode=:wide)
    end
    @test minimum(value->max(
            precision(real(value)),precision(imag(value))),
        nonzeros(lifted))==256

    embedded=setprecision(BigFloat,64) do
        pseudomode_model(
            prepared.site,
            Complex{BigFloat}[0 0;0 1];
            couplings=prepared.coupling)
    end
    @test embedded.metadata.precision_bits==256
    @test eltype(embedded.site_hamiltonian)===
          Complex{BigFloat}

    plan=setprecision(BigFloat,64) do
        pseudomode_trace_plan(prepared.site)
    end
    workspace=setprecision(BigFloat,64) do
        LocalFactorTraceWorkspace(plan)
    end
    @test plan.estimates.precision_bits==256
    @test minimum(value->max(
            precision(real(value)),precision(imag(value))),
        plan.lifted_columns)==256
    @test minimum(value->max(
            precision(real(value)),precision(imag(value))),
        workspace.occupation_coordinates)==256
    traced=setprecision(BigFloat,64) do
        trace_pseudomodes(
            prepared.rho,prepared.site;plan,workspace)
    end
    @test minimum(value->max(
            precision(real(value)),precision(imag(value))),
        traced.data)==256

    output=setprecision(BigFloat,256) do
        PIState(plan.output_basis;T=BigFloat)
    end
    setprecision(BigFloat,64) do
        trace_pseudomodes!(
            output,prepared.rho,prepared.site,plan,workspace)
    end
    @test output.data≈traced.data

    ordinary_mode=BosonicPseudomode(1)
    @test_throws ArgumentError pseudomode_model(
        1,zeros(ComplexF64,2,2),ordinary_mode;
        system_rate=1//(big(10)^1000))
end

@testset "generalized pseudomode models and legacy compatibility" begin
    X=ComplexF64[0 1;1 0]
    Z=ComplexF64[1 0;0 -1]
    lowering=ComplexF64[0 1;0 0]
    Hsystem=0.17Z
    mode_a=BosonicPseudomode(
        1;frequency=0.8,damping=0.4,
        thermal_occupation=0.0,label=:a)
    mode_b=BosonicPseudomode(
        1;frequency=-0.3,damping=0.2,
        thermal_occupation=0.25,label=:b)
    coupling_a=PseudomodeCoupling(
        lowering;mode=:a,strength=0.21)
    coupling_b=PseudomodeCoupling(
        Z;mode=:b,strength=0.0+0.13im,
        counterrotating_strength=0.07)
    pair=kron(X,X)
    system_terms=(
        PBodyHamiltonian(pair,2;rate=-0.11),
        LocalJump(lowering;rate=0.04),
    )
    embedding=pseudomode_model(
        2,Hsystem,(mode_a,mode_b);
        couplings=(coupling_a,coupling_b),
        system_terms)
    @test embedding.supersite.basis===embedding.basis
    @test embedding.model.basis===embedding.basis
    @test embedding.basis.d==8
    @test embedding.metadata.mode_count==2
    @test embedding.metadata.oscillator_cutoffs==(1,1)
    @test embedding.metadata.ordering===:system_then_local_modes
    @test embedding.metadata.exact_permutation_symmetry
    @test embedding.base_site_hamiltonian===
          embedding.site_hamiltonian
    @test embedding.supersite_terms==()
    @test embedding.resource_estimates.setup_peak_bytes>0
    @test ishermitian(embedding.site_hamiltonian)
    @test length(embedding.mode_operators)==2
    @test length(embedding.damping_terms)==3
    @test embedding.damping_terms[1].rate≈0.4
    @test embedding.damping_terms[2].rate≈0.25
    @test embedding.damping_terms[3].rate≈0.05
    @test length(embedding.lifted_system_terms)==2
    @test embedding.lifted_system_terms[1] isa PBodyHamiltonian
    @test embedding.lifted_system_terms[2] isa LocalJump
    assembled=sum(
        (term.rate*term.operator for term in
         (embedding.local_hamiltonian_terms...,
          embedding.coupling_terms...));
        init=zeros(ComplexF64,8,8))
    @test assembled≈embedding.site_hamiltonian atol=3e-15 rtol=3e-15

    # Reusing one prepared supersite preserves exact basis identity across a
    # parameter scan while allowing all physical mode rates to be rebound.
    prepared_site=pseudomode_supersite(2,2,mode_a,mode_b)
    overridden=pseudomode_model(
        prepared_site,Hsystem;
        frequencies=(1.25,-0.75),
        dampings=(0.6,0.0),
        thermal_occupations=(0.5,0.9))
    @test overridden.supersite===prepared_site
    @test overridden.basis===prepared_site.basis
    @test overridden.model.basis===prepared_site.basis
    @test overridden.metadata.frequencies==(1.25,-0.75)
    @test overridden.metadata.dampings==(0.6,0.0)
    @test overridden.metadata.thermal_occupations==(0.5,0.9)
    @test length(overridden.damping_terms)==2
    @test overridden.damping_terms[1].rate≈0.9
    @test overridden.damping_terms[2].rate≈0.3
    @test mode_a.frequency==0.8
    @test mode_a.damping==0.4
    @test mode_a.thermal_occupation==0.0

    family_site=pseudomode_supersite(1,2,mode_a)
    zero_family=pseudomode_model(
        family_site,Hsystem;
        couplings=PseudomodeCoupling(
            lowering;mode=:a,strength=0.0),
        retain_zero_terms=true)
    @test !isempty(zero_family.coupling_terms)
    @test !isempty(zero_family.damping_terms)
    @test compile_family(zero_family.model) isa CompiledPIModelFamily
    generated_terms=(
        term for term in (LocalJump(lowering;rate=0.03),))
    generated_model=pseudomode_model(
        family_site,Hsystem;system_terms=generated_terms)
    @test length(generated_model.lifted_system_terms)==1
    @test generated_model.lifted_system_terms[1] isa LocalJump

    alias=independent_local_pseudomode_model(
        1,Hsystem,(mode_a,mode_b);couplings=(coupling_a,))
    @test alias.metadata.embedding===:identical_local_pseudomodes

    @test_throws ArgumentError pseudomode_model(
        1,ComplexF64[0 1;0 0],mode_a)
    @test_throws DimensionMismatch pseudomode_model(
        1,Hsystem,mode_a;
        couplings=PseudomodeCoupling(ones(3,3)))
    restricted_mode=BosonicPseudomode(1;damping=0.2)
    @test_throws ArgumentError pseudomode_model(
        2,Hsystem,restricted_mode;
        sectors=[(2,0,0,0)])
    @test_throws ArgumentError pseudomode_model(
        1,Hsystem,mode_a;memory_budget=1)

    frequency=1.1
    coupling_strength=0.23
    damping=0.4
    thermal_occupation=0.3
    legacy=independent_local_pseudomode_model(
        1,Hsystem,lowering;
        nmax=1,frequency,coupling_strength,damping,
        thermal_occupation)
    generic_mode=BosonicPseudomode(
        1;frequency,damping,thermal_occupation)
    generic=pseudomode_model(
        1,Hsystem,generic_mode;
        couplings=PseudomodeCoupling(
            lowering;strength=coupling_strength))
    @test legacy.basis.N==generic.basis.N
    @test legacy.basis.d==generic.basis.d
    @test length(legacy.basis)==length(generic.basis)
    @test legacy.site_hamiltonian≈generic.site_hamiltonian
    @test legacy.lifted_system_hamiltonian≈
          generic.lifted_system_hamiltonian
    @test legacy.lifted_coupling≈
          lift_system_operator(generic.supersite,lowering)
    @test legacy.annihilation≈
          generic.mode_operators[1].annihilation
    @test legacy.number_operator≈
          generic.mode_operators[1].number_operator
    legacy_liouvillian=liouvillian(
        legacy.model;representation=:sparse)
    generic_liouvillian=liouvillian(
        generic.model;representation=:sparse)
    @test Matrix(legacy_liouvillian)≈
          Matrix(generic_liouvillian) atol=3e-12 rtol=3e-12
    generic_matrixfree=liouvillian(
        generic.model;representation=:matrixfree)
    probe=randn(ComplexF64,length(generic.basis))
    @test generic_matrixfree*probe≈
          generic_liouvillian*probe atol=3e-12 rtol=3e-12
    @test adjoint(generic_matrixfree)*probe≈
          adjoint(generic_liouvillian)*probe atol=3e-12 rtol=3e-12

    mode32=BosonicPseudomode(
        1;frequency=0.5f0,damping=0.2f0,
        thermal_occupation=0f0)
    model32=try
        pseudomode_model(
            1,ComplexF32[1 0;0 -1],mode32)
    catch
        nothing
    end
    @test model32!==nothing&&
          eltype(model32.site_hamiltonian)===ComplexF32
    @test model32!==nothing&&
          all(term->eltype(term.operator)===ComplexF32,
              model32.local_hamiltonian_terms)
end
