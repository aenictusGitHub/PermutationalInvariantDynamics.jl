@testset "hierarchy-preserving unitary pulses" begin
    basis=PIBasis(2,2)
    spin=spin_matrices()
    local_x=exp(-im*pi*spin.jx)
    pulse_x=PIUnitaryPulse(basis,local_x)

    @test pulse_x.basis===basis
    @test eltype(pulse_x)===ComplexF64
    @test length(pulse_x.blocks)==length(basis.sectors)
    @test all(block->block' * block≈I,pulse_x.blocks)
    @test_throws ArgumentError PIUnitaryPulse(
        basis,ComplexF64[1 1;0 1])
    @test_throws ArgumentError PIUnitaryPulse(
        basis,local_x;memory_budget=1)
    @test_throws ArgumentError HierarchyPulseSequence(
        [0.2,0.1],pulse_x)

    duplicate_sequence=HierarchyPulseSequence(
        [0.25,0.25],pulse_x)
    @test duplicate_sequence.pulses[1]===pulse_x
    @test duplicate_sequence.pulses[2]===pulse_x

    Jz=collective_operator(basis,spin.jz)
    heom_plan=HEOMPlan(
        PIModel(basis,()),HEOMBath(Jz,0.08,1.1);
        max_depth=2,scaling=:scaled)
    heom_workspace=HEOMEvolutionWorkspace(heom_plan)
    rng=MersenneTwister(0x51a7)
    raw=randn(rng,ComplexF64,size(heom_plan,1))
    hierarchy=HEOMState(heom_plan,raw)
    before=copy(hierarchy.data)
    apply_hierarchy_pulse!(hierarchy,pulse_x,heom_workspace)

    npi=heom_plan.npi
    for ado in 1:heom_number_ados(heom_plan)
        base=(ado-1)*npi
        for sector in eachindex(basis.sectors)
            dimension=length(basis.patterns[sector])
            first_coordinate=base+basis.offsets[sector]
            range=first_coordinate:first_coordinate+dimension^2-1
            source=reshape(view(before,range),dimension,dimension)
            expected=pulse_x.blocks[sector]*source*
                     pulse_x.blocks[sector]'
            actual=reshape(
                view(hierarchy.data,range),dimension,dimension)
            @test actual≈expected atol=2e-14 rtol=2e-14
        end
    end

    zero_hamiltonian=PIOperator(basis;T=Float64)
    hops_plan=HOPSPlan(
        zero_hamiltonian,HOPSBath(Jz,0.08,1.1);
        max_depth=2,scaling=:scaled)
    hops_workspace=HOPSWorkspace(hops_plan)
    hops_workspace.current .= randn(
        rng,ComplexF64,size(hops_workspace.current))
    hops_workspace.ou_current .= randn(
        rng,ComplexF64,length(hops_workspace.ou_current))
    hops_before=copy(hops_workspace.current)
    noise_before=copy(hops_workspace.ou_current)
    apply_hierarchy_pulse!(hops_workspace,pulse_x)
    weak_offset=1
    for sector in eachindex(basis.sectors)
        dimension=length(basis.patterns[sector])
        range=weak_offset:weak_offset+dimension-1
        expected=pulse_x.blocks[sector]*
                 view(hops_before,range,:)
        @test view(hops_workspace.current,range,:)≈expected
        weak_offset+=dimension
    end
    @test hops_workspace.ou_current==noise_before

    other_basis=PIBasis(2,2)
    other_pulse=PIUnitaryPulse(other_basis,local_x)
    @test_throws ArgumentError apply_hierarchy_pulse!(
        hierarchy,other_pulse,heom_workspace)
    pulse32=PIUnitaryPulse(basis,ComplexF32.(local_x))
    @test_throws ArgumentError apply_hierarchy_pulse!(
        hierarchy,pulse32,heom_workspace)
    @test_throws ArgumentError apply_hierarchy_pulse!(
        hops_workspace,other_pulse)
    @test_throws ArgumentError apply_hierarchy_pulse!(
        hops_workspace,pulse32)

    # A pulse at a saved time is applied before saving, even when it does not
    # coincide with the nominal RK step grid.
    one_basis=PIBasis(1,2)
    one_spin=spin_matrices()
    one_Jz=collective_operator(one_basis,one_spin.jz)
    one_pulse=PIUnitaryPulse(
        one_basis,exp(-im*pi*one_spin.jx))
    pulse_time=0.35
    sequence=HierarchyPulseSequence([pulse_time],one_pulse)
    initial=computational_product_state(one_basis,1)

    depth_zero_heom=HEOMPlan(
        PIModel(one_basis,()),HEOMBath(one_Jz,0.0,1.0);
        max_depth=0)
    heom_states=heom_time_evolution(
        depth_zero_heom,initial,[0.0,pulse_time,1.0];
        steps_per_interval=2,pulses=sequence)
    @test real(expectation(
        heom_reduced_state(heom_states[1]),one_Jz))≈-0.5
    @test real(expectation(
        heom_reduced_state(heom_states[2]),one_Jz))≈0.5 atol=2e-14
    @test real(expectation(
        heom_reduced_state(heom_states[3]),one_Jz))≈0.5 atol=2e-14

    depth_zero_hops=HOPSPlan(
        PIOperator(one_basis;T=Float64),
        HOPSBath(one_Jz,0.0,1.0);max_depth=0)
    initial_ket=weak_pi_pseudoket(initial)
    hops_path=hops_trajectory(
        depth_zero_hops,initial_ket,[0.0,pulse_time,1.0];
        dt=0.2,noise=(time,bath)->0.0+0.0im,
        rng=MersenneTwister(11),pulses=sequence)
    @test real(expectation(
        hops_density(hops_path,1),one_Jz))≈-0.5
    @test real(expectation(
        hops_density(hops_path,2),one_Jz))≈0.5 atol=2e-14
    @test real(expectation(
        hops_density(hops_path,3),one_Jz))≈0.5 atol=2e-14

    # Equal-time events retain input order. Use noncommuting rotations so
    # reversing that order would change the resulting ket and density.
    pulse_half_x=PIUnitaryPulse(
        one_basis,exp(-im*(pi/2)*one_spin.jx))
    pulse_third_z=PIUnitaryPulse(
        one_basis,exp(-im*(pi/3)*one_spin.jz))
    ordered=HierarchyPulseSequence(
        [pulse_time,pulse_time],[pulse_half_x,pulse_third_z])
    ordered_heom=heom_time_evolution(
        depth_zero_heom,initial,[0.0,pulse_time];
        steps_per_interval=2,pulses=ordered)
    expected_block=pulse_third_z.blocks[1]*
                   pulse_half_x.blocks[1]*
                   Matrix(physical_block(initial,one_basis.sectors[1]))*
                   pulse_half_x.blocks[1]'*
                   pulse_third_z.blocks[1]'
    actual_block=Matrix(physical_block(
        heom_reduced_state(ordered_heom[end]),one_basis.sectors[1]))
    @test actual_block≈expected_block atol=2e-14 rtol=2e-14

    ordered_hops=hops_trajectory(
        depth_zero_hops,initial_ket,[0.0,pulse_time];
        dt=0.2,noise=(time,bath)->0.0+0.0im,
        rng=MersenneTwister(12),pulses=ordered)
    expected_ket=pulse_third_z.blocks[1]*
                 pulse_half_x.blocks[1]*initial_ket.data
    @test ordered_hops.states[end].data≈expected_ket

    # An empty schedule exercises the pulse-aware path without changing the
    # historical no-pulse result or random stream.
    empty_sequence=HierarchyPulseSequence(Float64[],one_pulse)
    heom_reference=heom_time_evolution(
        depth_zero_heom,initial,[0.0,0.2,0.4];
        steps_per_interval=2)
    heom_empty=heom_time_evolution(
        depth_zero_heom,initial,[0.0,0.2,0.4];
        steps_per_interval=2,pulses=empty_sequence)
    @test all(left.data==right.data
              for (left,right) in zip(heom_reference,heom_empty))

    stochastic_plan=HOPSPlan(
        PIOperator(one_basis;T=Float64),
        HOPSBath(one_Jz,0.04,1.2);max_depth=1)
    path_reference=hops_trajectory(
        stochastic_plan,initial_ket,[0.0,0.2,0.4];
        dt=0.05,rng=MersenneTwister(0x771))
    path_empty=hops_trajectory(
        stochastic_plan,initial_ket,[0.0,0.2,0.4];
        dt=0.05,rng=MersenneTwister(0x771),
        pulses=empty_sequence)
    @test path_reference.noise==path_empty.noise
    @test all(left.data==right.data
              for (left,right) in
              zip(path_reference.states,path_empty.states))

    threaded_workspace=HOPSBatchWorkspace(
        stochastic_plan;workers=max(1,min(2,Threads.nthreads())))
    threaded_sequence=HierarchyPulseSequence([0.2],one_pulse)
    threaded_first=hops_average(
        stochastic_plan,initial_ket,[0.0,0.2,0.4],8;
        dt=0.05,seed=0x891,threaded=true,
        workspace=threaded_workspace,pulses=threaded_sequence)
    threaded_second=hops_average(
        stochastic_plan,initial_ket,[0.0,0.2,0.4],8;
        dt=0.05,seed=0x891,threaded=true,
        workspace=threaded_workspace,pulses=threaded_sequence)
    @test all(left.data==right.data
              for (left,right) in
              zip(threaded_first,threaded_second))

    # Events outside the requested interval and at its initial endpoint are
    # harmless under the documented (start, stop] convention.
    outside=HierarchyPulseSequence([-1.0,0.0,2.0],one_pulse)
    outside_states=heom_time_evolution(
        depth_zero_heom,initial,[0.0,1.0];
        steps_per_interval=2,pulses=outside)
    @test outside_states[end].data==heom_states[1].data

    setprecision(BigFloat,128) do
        identity_pulse=PIUnitaryPulse(
            one_basis,identity_operator(one_basis;T=BigFloat))
        @test eltype(identity_pulse)===Complex{BigFloat}
        @test identity_pulse.precision_bits==128
    end
end

@testset "Platonic Eulerian pulse constructors" begin
    published_words=(
        tetrahedral="abaababbbaababbbaababbaa",
        octahedral=
            "abaaabbbabaabbbaababbaaa" *
            "ababbbabaabbaaaababbbabb",
        icosahedral=
            "baaabbaabaaaaabbaaab" *
            "abbbabaabbaabbabbabb" *
            "abbbaaaababbbaaababb" *
            "baaababbbaababbaabba" *
            "abbaabbbabbbaababbba" *
            "ababbbaababbbabaaaaa",
    )
    constructors=(
        tetrahedral=tetrahedral_pulse_sequence,
        octahedral=octahedral_pulse_sequence,
        icosahedral=icosahedral_pulse_sequence,
    )
    universal_dimensions=(
        tetrahedral=3,
        octahedral=4,
        icosahedral=6,
    )

    function event_generators(sequence,word)
        first_a=findfirst(==('a'),word)
        first_b=findfirst(==('b'),word)
        pulse_a=sequence.pulses[first_a]
        pulse_b=sequence.pulses[first_b]
        @test pulse_a!==pulse_b
        @test all(sequence.pulses[index]===
                  (letter=='a' ? pulse_a : pulse_b)
                  for (index,letter) in enumerate(word))
        pulse_a,pulse_b
    end

    function check_complete_twirl(sequence,dimension)
        identity_matrix=Matrix{ComplexF64}(I,dimension,dimension)
        frames=Matrix{ComplexF64}[]
        propagator=copy(identity_matrix)
        for pulse in sequence.pulses
            push!(frames,copy(propagator))
            propagator=pulse.blocks[1]*propagator
        end

        phase=tr(propagator)/dimension
        @test abs(abs(phase)-1)<=2e-12
        @test propagator≈phase*identity_matrix atol=3e-12 rtol=3e-12

        matrix_unit=zeros(ComplexF64,dimension,dimension)
        average=zeros(ComplexF64,dimension,dimension)
        for column in 1:dimension,row in 1:dimension
            fill!(matrix_unit,0)
            matrix_unit[row,column]=1
            fill!(average,0)
            for frame in frames
                mul!(average,frame',matrix_unit*frame,1,1)
            end
            average ./= length(frames)
            target=row==column ? identity_matrix/dimension :
                zero(identity_matrix)
            @test average≈target atol=4e-12 rtol=4e-12
        end
    end

    for group in keys(published_words)
        word=getproperty(published_words,group)
        constructor=getproperty(constructors,group)
        dimension=getproperty(universal_dimensions,group)
        basis=PIBasis(1,dimension)
        sequence=constructor(basis,0.125)

        @test length(sequence.times)==length(word)
        @test count(==('a'),word)==count(==('b'),word)==length(word)÷2
        @test first(sequence.times)==0.125
        @test last(sequence.times)==0.125length(word)
        @test all(==(0.125),diff(sequence.times))
        event_generators(sequence,word)
        check_complete_twirl(sequence,dimension)

        generic=platonic_pulse_sequence(basis,group,0.125)
        @test generic.times==sequence.times
        @test all(left.blocks==right.blocks
                  for (left,right) in
                  zip(generic.pulses,sequence.pulses))
    end

    # Check the published axis--angle generators independently in the
    # fundamental spin-1/2 representation.
    qubit_basis=PIBasis(1,2)
    spin=spin_matrices()
    phi=(1+sqrt(5.0))/2
    generator_data=(
        tetrahedral=(
            ((0.0,0.0,1.0),2pi/3),
            ((sqrt(2.0)/3,sqrt(2/3),1/3),2pi/3)),
        octahedral=(
            ((0.0,0.0,1.0),pi/2),
            ((1/sqrt(3.0),1/sqrt(3.0),1/sqrt(3.0)),2pi/3)),
        icosahedral=(
            ((0.0,-1/sqrt(phi+2),phi/sqrt(phi+2)),2pi/5),
            (((1-phi)/sqrt(3.0),0.0,phi/sqrt(3.0)),2pi/3)),
    )
    for group in keys(generator_data)
        word=getproperty(published_words,group)
        sequence=getproperty(constructors,group)(qubit_basis,0.25)
        pulse_a,pulse_b=event_generators(sequence,word)
        generators=getproperty(generator_data,group)
        for (pulse,(axis,angle)) in
                zip((pulse_a,pulse_b),generators)
            local_generator=axis[1]*spin.jx+axis[2]*spin.jy+
                            axis[3]*spin.jz
            # N=1 Schur blocks use the package's stored GT order
            # |+j>,...,|-j>, the reverse of spin_matrices' local order.
            expected=reverse(
                exp(-im*angle*local_generator);dims=(1,2))
            @test pulse.blocks[1]≈expected atol=2e-14 rtol=2e-14
        end
    end

    repeated=tetrahedral_pulse_sequence(
        qubit_basis,0.125;cycles=2,start_time=0.5)
    @test length(repeated.times)==48
    @test repeated.times==0.5 .+ 0.125 .* collect(1:48)
    @test all(repeated.pulses[index]===repeated.pulses[index+24]
              for index in 1:24)

    empty=tetrahedral_pulse_sequence(qubit_basis,0.125;cycles=0)
    @test isempty(empty.times)
    @test isempty(empty.pulses)
    @test empty.basis===qubit_basis

    float32_sequence=octahedral_pulse_sequence(
        qubit_basis,Float32(0.125);T=Float32)
    @test eltype(float32_sequence.times)===Float32
    @test eltype(first(float32_sequence.pulses))===ComplexF32

    @test_throws ArgumentError platonic_pulse_sequence(
        qubit_basis,:cubic,0.1)
    @test_throws ArgumentError platonic_pulse_sequence(
        qubit_basis,"tetrahedral",0.1)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,0.0)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,-0.1)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,NaN)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,true)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,0.1;start_time=Inf)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,0.1;cycles=true)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,0.1;cycles=-1)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,0.1;cycles=1.0)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,0.1;cycles=big(typemax(Int))+1)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,0.1;T=AbstractFloat)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,BigFloat("0.1");T=BigFloat)
    @test_throws ArgumentError tetrahedral_pulse_sequence(
        qubit_basis,0.1;memory_budget=1)

    # A complete Eulerian word closes projectively, so an otherwise static
    # HEOM root density returns to its initial value after the final event.
    final_time=last(tetrahedral_pulse_sequence(
        qubit_basis,0.01).times)
    sequence=tetrahedral_pulse_sequence(qubit_basis,0.01)
    Jz=collective_spin(qubit_basis,:z)
    plan=HEOMPlan(
        PIModel(qubit_basis,()),HEOMBath(Jz,0.0,1.0);
        max_depth=0)
    initial=computational_product_state(qubit_basis,1)
    evolved=heom_time_evolution(
        plan,initial,[0.0,final_time];
        steps_per_interval=2,pulses=sequence)
    @test evolved[end].data≈initial.data atol=5e-13 rtol=5e-13
end
