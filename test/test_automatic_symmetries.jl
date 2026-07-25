@testset "automatic strong-symmetry reduction" begin
    # A phase-covariant jump is a weak Z symmetry, but it does not commute
    # with Z and is therefore not a strong symmetry.
    qubit_basis=PIBasis(1,2)
    lowering=ComplexF64[0 1;0 0]
    parity_z=Diagonal(ComplexF64[1,-1])
    weak_only=PIModel(qubit_basis,(LocalJump(lowering),))
    @test check_liouvillian_symmetry(
        weak_only,parity_z;basis=qubit_basis).symmetric
    weak_report=strong_symmetry_report(
        weak_only;candidates=(parity_z=parity_z,))
    @test weak_report isa StrongSymmetryReport
    @test only(weak_report.candidates).status===false
    @test only(weak_report.candidates).terms[1].status===false
    @test_throws ArgumentError strong_symmetry_reduction(
        weak_only;report=weak_report,candidate=:parity_z)

    # The exact GF(2) support nullspace discovers diag(1,1,-1) without being
    # told its form.  It also correctly rejects the qutrit clock, whose first
    # two levels have different phases.
    basis=PIBasis(1,3)
    jump=zeros(ComplexF64,3,3)
    jump[1,2]=1
    model=PIModel(basis,(LocalJump(jump),))
    report=strong_symmetry_report(model)
    @test report.discovery.complete_within===:binary_sign
    @test report.discovery.microscopic_support_complete
    @test any(candidate->candidate.name===:clock_phase&&
                         candidate.status===false,report.candidates)
    certified=filter(candidate->candidate.status===true,report.candidates)
    @test length(certified)==1
    @test certified[1].name===:binary_parity_1
    @test length(certified[1].charges)==2
    @test Set(round.(Int,real.(certified[1].charges)))==Set((-1,1))

    reduction=strong_symmetry_reduction(model;report)
    @test reduction isa StrongSymmetryReduction
    @test reduction.candidate.name===:binary_parity_1
    @test length(reduction.sectors)==2
    @test sum(sector.dimension for sector in reduction.sectors)<
          length(basis)
    @test all(sector->sector.operator.backend===:lowered,reduction.sectors)
    @test all(sector->sector.leakage_certificate.invariant,
              reduction.sectors)
    @test reduction.resources.retained_bytes==
          reduction.resources.retained_source_bytes+
          reduction.resources.retained_sector_bytes
    @test reduction.resources.accounting===
          :source_plus_all_charge_restrictions
    @test reduction.resources.predicted_backends==(:lowered,:lowered)
    @test reduction.resources.preflight_peak_upper_bound>=
          reduction.resources.retained_bytes
    @test_throws ArgumentError strong_symmetry_reduction(
        model;report,memory_budget=1)

    stationary=strong_symmetry_steady_states(
        reduction;atol=1e-12,rtol=1e-10,krylovdim=4,maxiter=40)
    @test stationary.complete_trace_bearing_charge_enumeration
    @test !stationary.offdiagonal_charge_blocks_included
    @test !stationary.global_stationary_manifold_certified
    @test stationary.selected_stationary_sector===nothing
    @test length(stationary.sectors)==length(reduction.sectors)
    @test all(result->abs(trace(result.state)-1)<2e-11,
              stationary.sectors)
    @test all(result->result.full_residual<2e-10,
              stationary.sectors)
    @test all(result->result.leakage_residual<2e-12,
              stationary.sectors)
    @test stationary.resources.aggregate_peak_upper_bound>=
          reduction.resources.retained_bytes
    @test stationary.resources.result_metadata_bytes>0
    reduced_stationary=strong_symmetry_steady_states(
        reduction;embed_states=false,atol=1e-12,rtol=1e-10,
        krylovdim=4,maxiter=40)
    @test all(result->result.state===nothing,reduced_stationary.sectors)
    @test all(result->length(result.reduced_state)==result.dimension,
              reduced_stationary.sectors)
    @test reduced_stationary.resources.ambient_output_bytes==0
    @test_throws ArgumentError strong_symmetry_steady_states(
        reduction;memory_budget=1)

    spectra=strong_symmetry_spectra(
        reduction;method=:dense,vectors=false,
        atol=1e-12,rtol=1e-10)
    @test spectra.complete_trace_bearing_charge_enumeration
    @test !spectra.offdiagonal_charge_blocks_included
    @test !spectra.global_spectrum
    @test length(spectra.sectors)==length(reduction.sectors)
    for (sector,result) in zip(reduction.sectors,spectra.sectors)
        @test result.scope===:complete_trace_bearing_charge_sector
        @test result.validated_full
        @test result.spectrum.vectors===nothing
        @test length(result.spectrum.values)==sector.dimension
        @test length(result.full_residuals)==sector.dimension
        @test maximum(report.relative_residual
                      for report in result.full_residuals)<2e-12
    end
    @test spectra.resources.aggregate_peak_upper_bound>=
          reduction.resources.retained_bytes
    @test spectra.resources.result_metadata_bytes>0
    @test_throws ArgumentError strong_symmetry_spectra(
        reduction;method=:dense,memory_budget=1)

    # A scalar operator schedule is not inferred at an arbitrary time.  The
    # report remains explicitly inconclusive until the model is frozen.
    scheduled=PIModel(qubit_basis,(
        LocalHamiltonian((time,parameters)->
            Matrix(Diagonal(ComplexF64[time,-time]))),))
    scheduled_report=strong_symmetry_report(
        scheduled;candidates=(parity_z=parity_z,))
    @test ismissing(only(scheduled_report.candidates).status)
    @test only(scheduled_report.candidates).terms[1].reason===
          :operator_schedule_requires_freeze
    @test_throws ArgumentError strong_symmetry_reduction(
        scheduled;report=scheduled_report)

    # A fixed correlated channel is checked after its Kossakowski
    # factorization.  An inactive, noncommuting seed must not create a false
    # negative strong-symmetry report.
    correlated=PIModel(qubit_basis,(
        CorrelatedLocalJumps(
            (lowering,Matrix(parity_z)),Diagonal(ComplexF64[0,1])),))
    correlated_report=strong_symmetry_report(
        correlated;candidates=(parity_z=parity_z,))
    @test only(correlated_report.candidates).status===true
    @test only(correlated_report.candidates).terms[1].validation===
          :effective_channel_support_scan

    # Appendix-D support contributes exact parity equations: X⊗X preserves
    # total Z parity even though one X does not.
    x=ComplexF64[0 1;1 0]
    pair_model=PIModel(PIBasis(2,2),(
        PBodyHamiltonian(kron(x,x),2),
        LocalJump(Matrix(parity_z)),))
    pair_report=strong_symmetry_report(pair_model)
    @test any(candidate->candidate.status===true&&
                         length(candidate.charges)==2,
              pair_report.candidates)

    # The built-in qubit candidate is exactly ±1 rather than exp(πim), so its
    # charges remain exactly clustered even after a large tensor power.
    large_basis=PIBasis(200,2;sectors=[(200,0)])
    large_model=PIModel(large_basis,(
        CollectiveJump(Matrix(parity_z)),))
    large_report=strong_symmetry_report(large_model)
    large_parity=only(filter(
        candidate->candidate.name===:parity_z,
        large_report.candidates))
    @test Set(large_parity.charges)==Set(ComplexF64[-1,1])

    # With no support equations, the GF(2) nullspace modulo the physically
    # irrelevant all-ones local sign has dimension d-1.
    unconstrained=strong_symmetry_report(PIModel(PIBasis(1,4),()))
    @test count(candidate->startswith(
        string(candidate.name),"binary_parity_"),
        unconstrained.candidates)==3
end
