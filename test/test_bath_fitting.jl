const PIDBATHFIT=PermutationalInvariantDynamics

@testset "finite exponential bath preparation" begin
    times=collect(range(0.0,3.0;length=81))
    poles=ComplexF64[0.7,1.2+0.8im]
    coefficients=ComplexF64[0.31,0.17-0.09im]
    samples=[
        sum(coefficients.*exp.(-poles.*time))
        for time in times]

    fixed=PIDBATHFIT.fit_bath_correlation(
        times,samples;poles,rtol=2e-12)
    @test fixed.report.converged
    @test fixed.report.identifiable
    @test fixed.report.stable
    @test fixed.report.method===:fixed_poles
    @test fixed.coefficients≈coefficients atol=2e-13 rtol=2e-12
    @test fixed.frequencies==poles
    @test fixed.report.relative_residual<2e-14
    @test !fixed.report.hops_stationary_ou_compatible
    @test occursin("BathFitResult",sprint(show,fixed))

    selected=PIDBATHFIT.fit_bath_correlation(
        times,samples;
        candidate_poles=ComplexF64[0.35,0.7,1.2+0.8im,2.4],
        nterms=2,rtol=2e-12)
    @test selected.report.converged
    @test Set(selected.frequencies)==Set(poles)
    @test selected.report.selected_candidates==[2,3]
    @test selected.fitted≈samples atol=2e-13 rtol=2e-12

    @test_throws ArgumentError PIDBATHFIT.fit_bath_correlation(
        times,samples;poles=[0.7,-0.2])
    @test_throws ArgumentError PIDBATHFIT.fit_bath_correlation(
        times,samples;poles=[0.7,0.7])
    @test_throws DimensionMismatch PIDBATHFIT.fit_bath_correlation(
        times,samples[1:end-1];poles)
    @test_throws ArgumentError PIDBATHFIT.fit_bath_correlation(
        times,samples;poles,memory_budget=1)
    below_float32=Float64(nextfloat(0.0f0))/2
    above_float32=2Float64(floatmax(Float32))
    @test_throws ArgumentError PIDBATHFIT._bathfit_checked_real(
        Float32,below_float32,"test scalar")
    @test_throws ArgumentError PIDBATHFIT._bathfit_checked_real(
        Float32,above_float32,"test scalar")
    @test_throws ArgumentError PIDBATHFIT._bathfit_checked_complex(
        Float32,complex(1.0,below_float32),"test coefficient")
    @test_throws ArgumentError PIDBATHFIT._bathfit_checked_complex(
        Float32,complex(above_float32,0.0),"test coefficient")
    @test_throws ArgumentError PIDBATHFIT._bathfit_checked_real(
        Float32,0.1,"test narrowing")
    float32_times=Float32[0,1]
    float32_samples=ComplexF32[1,0.5]
    float32_fit=PIDBATHFIT.fit_bath_correlation(
        float32_times,float32_samples;poles=ComplexF32[0.7],
        memory_budget=Inf)
    @test eltype(float32_fit.coefficients)===ComplexF32
    promoted_fit=PIDBATHFIT.fit_bath_correlation(
        float32_times,float32_samples;poles=ComplexF32[0.7],
        ridge=0.1,memory_budget=Inf)
    @test eltype(promoted_fit.coefficients)===ComplexF64
    @test promoted_fit.report.ridge===0.1
    promoted_weights=PIDBATHFIT.fit_bath_correlation(
        float32_times,float32_samples;poles=ComplexF32[0.7],
        weights=Float64[1,1],memory_budget=Inf)
    @test eltype(promoted_weights.coefficients)===ComplexF64

    consumed=Ref(0)
    guarded_times=((consumed[]+=1; time) for time in times)
    @test_throws ArgumentError PIDBATHFIT.fit_bath_correlation(
        guarded_times,samples;poles,memory_budget=0)
    @test consumed[]==0

    basis=PIBasis(1,2)
    coupling=collective_operator(basis,spin_matrices().jz)
    heom=PIDBATHFIT.prepare_heom_bath(coupling,fixed)
    @test heom.coefficients[1:length(fixed)]≈fixed.coefficients
    @test heom.frequencies[1:length(fixed)]==fixed.frequencies
    @test last(heom.frequencies)==conj(last(fixed.frequencies))
    @test iszero(last(heom.coefficients))
    @test heom.metadata.source===:finite_exponential_fit
    @test heom.metadata.fit_converged
    hops=PIDBATHFIT.prepare_hops_bath(coupling,fixed)
    @test hops.coefficients≈fixed.coefficients
    @test_throws ArgumentError PIDBATHFIT.prepare_hops_bath(
        coupling,fixed;require_stationary_ou=true)

    positive_coefficients=ComplexF64[0.31,0.17]
    positive_poles=ComplexF64[0.7,1.2]
    positive_samples=[
        sum(positive_coefficients.*exp.(-positive_poles.*time))
        for time in times]
    positive=PIDBATHFIT.fit_bath_correlation(
        times,positive_samples;poles=positive_poles,rtol=2e-12)
    @test positive.report.hops_stationary_ou_compatible
    @test PIDBATHFIT.prepare_hops_bath(
        coupling,positive;require_stationary_ou=true) isa HOPSBath

    inaccurate=PIDBATHFIT.fit_bath_correlation(
        times,samples;poles=[0.4],rtol=1e-12)
    @test !inaccurate.report.converged
    @test_throws ArgumentError PIDBATHFIT.prepare_heom_bath(
        coupling,inaccurate)
    @test PIDBATHFIT.prepare_heom_bath(
        coupling,inaccurate;accept_unconverged=true) isa HEOMBath

    rank_deficient=PIDBATHFIT.fit_bath_correlation(
        [0.0,1.0],[1.0,0.5];poles=[0.4,0.9,1.4],rtol=1)
    @test !rank_deficient.report.identifiable
    @test_throws ArgumentError PIDBATHFIT.prepare_heom_bath(
        coupling,rank_deficient)
    @test PIDBATHFIT.prepare_heom_bath(
        coupling,rank_deficient;
        accept_rank_deficient=true) isa HEOMBath
end

@testset "spectral-density correlation quadrature" begin
    omega=Float64[0.4,0.8,1.3,2.0]
    density=0.2 .* omega .* exp.(-omega)
    times=Float64[0.0,0.2,0.7,1.4]
    samples=PIDBATHFIT.correlation_from_spectral_density(
        omega,density,times;inverse_temperature=2.3)
    @test samples.times==times
    @test length(samples.values)==length(times)
    @test samples.metadata.quadrature===:trapezoid
    @test samples.metadata.converged===nothing
    @test isreal(samples.values[1])
    @test real(samples.values[1])>0
    @test imag(samples.values[2])<0

    weights=Float64[
        (omega[2]-omega[1])/2,
        (omega[3]-omega[1])/2,
        (omega[4]-omega[2])/2,
        (omega[4]-omega[3])/2]
    expected=sum(weights.*density.*
        coth.(2.3 .* omega ./ 2))/pi
    @test isapprox(real(samples.values[1]),expected;rtol=2e-15)

    zero_temperature=PIDBATHFIT.correlation_from_spectral_density(
        omega,density,times;inverse_temperature=Inf)
    expected_zero=sum(weights.*density)/pi
    @test isapprox(real(zero_temperature.values[1]),expected_zero;rtol=2e-15)
    @test_throws ArgumentError PIDBATHFIT.correlation_from_spectral_density(
        [0.0,0.4],density[1:2],times)
    @test_throws ArgumentError PIDBATHFIT.correlation_from_spectral_density(
        omega,-density,times)
    @test_throws ArgumentError PIDBATHFIT.correlation_from_spectral_density(
        omega,density,times;memory_budget=1)
    @test_throws ArgumentError PIDBATHFIT.correlation_from_spectral_density(
        [1.0,3.0],[floatmax(Float64),floatmax(Float64)],[0.0];
        normalization=1.0,memory_budget=Inf)
    consumed=Ref(0)
    guarded_omega=((consumed[]+=1; value) for value in omega)
    @test_throws ArgumentError PIDBATHFIT.correlation_from_spectral_density(
        guarded_omega,density,times;memory_budget=0)
    @test consumed[]==0

    fit=PIDBATHFIT.fit_bath_from_spectral_density(
        omega,density,times;
        correlation_options=(inverse_temperature=Inf,),
        fit_options=(poles=[0.4],rtol=1.0))
    @test fit isa PIDBATHFIT.BathFitResult
    @test fit.report.converged
end
