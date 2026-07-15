@testset "compiled plans and high-level commands" begin
    b=PIBasis(3,2)
    @test isconcretetype(typeof(b))
    @test isconcretetype(typeof(b.sectors))
    @test isconcretetype(typeof(b.patterns))

    sm=ComplexF64[0 1;0 0]
    model=PIModel(b,[LocalJump(sm;rate=0.4)])
    @test model.terms isa Tuple
    @test isconcretetype(typeof(model))

    prepared=compile(model;backend=:matrixfree)
    @test prepared isa CompiledPIModel
    @test prepared.backend===:matrixfree
    @test prepared.estimates.scalar_type===eltype(prepared)
    @test prepared.estimates.scalar_retained_bytes==sizeof(eltype(prepared))
    @test prepared.estimates.scalar_storage_estimate===:exact_inline
    @test prepared.estimates.bigfloat_precision_assumption===nothing
    @test diagnostics(prepared).dimension==length(b)

    rho0=iid_pure_state(b,ComplexF64[0,1])
    low=steady_state(model;method=:direct)
    rho=stationary_state(model;algorithm=DirectAlgorithm())
    @test rho isa PIState
    @test rho.data≈low atol=2e-11
    report=stationary_state(prepared;algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),return_info=true)
    @test report isa SteadyStateResult
    @test report.state.data≈low atol=2e-9
    @test report.info.converged

    sol=solve_dynamics(prepared,rho0,(0.0,0.2);saveat=0.05,steps_per_interval=40)
    @test sol isa DynamicsResult
    @test length(sol)==5
    @test collect(sol)==sol.states
    @test state(sol,0.1)===sol[3]
    @test sol[end].data≈time_evolve(prepared,rho0,(0.0,0.2);steps=160).data atol=2e-11

    dense=liouvillian_spectrum(model;target=:largest_real,nev=3,
                               algorithm=:dense)
    reference=pi_liouvillian_spectrum(model)[1:3]
    @test dense≈reference atol=2e-11
    spectrum_report=liouvillian_spectrum(model;target=:largest_real,nev=3,
                                         algorithm=:dense,return_info=true)
    @test spectrum_report isa SpectrumResult
    @test spectrum_report.values≈reference atol=2e-11
    iram_high=liouvillian_spectrum(model;target=:largest_real,nev=2,
        algorithm=:iram,krylovdim=10,retained_dimension=5,maxrestarts=30,
        initial_vector=randn(MersenneTwister(91),ComplexF64,length(b)),
        atol=1e-9,rtol=1e-7)
    @test iram_high≈reference[1:2] atol=2e-7
    @test_throws ArgumentError liouvillian_spectrum(model;
        target=:largest_real,nev=2,algorithm=:jd)

    @test pi_dimension(model)==length(b)
    @test estimate_state_bytes(b)==length(b)*sizeof(ComplexF64)
    @test estimate_basis_bytes(b)>=estimate_state_bytes(b)
    @test estimate_liouvillian_bytes(prepared)>0
    geometry_estimate=estimate_geometry_bytes(b)
    @test geometry_estimate.setup_bytes>=geometry_estimate.retained_bytes>0
    @test geometry_estimate.scalar_type===Float64
    @test estimate_solver_bytes(model;krylovdim=8)>0
    @test estimate_solver_bytes(model;algorithm=:iram,krylovdim=8)>0
    @test estimate_solver_bytes(model;algorithm=:jd,krylovdim=8)>
          estimate_solver_bytes(model;algorithm=:iram,krylovdim=8)
    @test estimate_solver_bytes(model;algorithm=:rk4)==
          5*estimate_state_bytes(model)

    # Machine scalars retain the old inline-storage formulas, including the
    # real residual/history vector in a Float32 GMRES workspace.
    n32=big(length(b));m32=big(min(length(b),8))
    expected_gmres32=sizeof(ComplexF32)*(
        n32*(m32+6)+(m32+1)*m32+2m32+1)+sizeof(Float32)*m32
    @test estimate_solver_bytes(model;algorithm=:gmres,krylovdim=8,
                                T=ComplexF32)==expected_gmres32

    # BigFloat is heap backed: estimates must grow with the explicitly stated
    # working precision instead of treating an array slot as the whole value.
    state128=estimate_state_bytes(b;T=Complex{BigFloat},
                                  bigfloat_precision=128)
    state512=estimate_state_bytes(b;T=Complex{BigFloat},
                                  bigfloat_precision=512)
    @test state512>state128>0
    @test basis_summary(b;T=Complex{BigFloat},
                        bigfloat_precision=128).state_bytes==state128
    @test estimate_memory(3,2;T=Complex{BigFloat},
                          bigfloat_precision=128)==state128
    @test estimate_solver_bytes(model;algorithm=:gmres,krylovdim=8,
                                T=Complex{BigFloat},bigfloat_precision=512)>
          estimate_solver_bytes(model;algorithm=:gmres,krylovdim=8,
                                T=Complex{BigFloat},bigfloat_precision=128)
    @test_throws ArgumentError estimate_state_bytes(
        b;T=Complex{BigFloat},bigfloat_precision=1)
    big_basis=PIBasis(1,2)
    geometry128=estimate_geometry_bytes(big_basis;T=BigFloat,
                                        bigfloat_precision=128)
    geometry512=estimate_geometry_bytes(big_basis;T=BigFloat,
                                        bigfloat_precision=512)
    @test geometry512.setup_bytes>geometry128.setup_bytes
    @test geometry512.retained_bytes>geometry128.retained_bytes
    @test geometry128.scalar_storage_estimate===
          :conservative_retained_bound
    @test geometry128.bigfloat_precision_assumption==128
    big_model=PIModel(big_basis,[LocalHamiltonian(
        Complex{BigFloat}[0 1;1 0])])
    big_prepared=compile(big_model;backend=:matrixfree,
                         bigfloat_precision=192)
    @test big_prepared.estimates.scalar_storage_estimate===
          :conservative_retained_bound
    @test big_prepared.estimates.bigfloat_precision_assumption==192
    @test big_prepared.estimates.scalar_retained_bytes>0
    big_model_recommendation128=recommend_solver(big_model;
        T=Complex{BigFloat},bigfloat_precision=128)
    big_model_recommendation512=recommend_solver(big_model;
        T=Complex{BigFloat},bigfloat_precision=512)
    @test big_model_recommendation128.geometry_requirement===:required
    @test big_model_recommendation128.geometry_setup_upper_bytes==
          geometry128.setup_bytes
    @test big_model_recommendation512.geometry_setup_upper_bytes==
          geometry512.setup_bytes
    @test big_model_recommendation512.estimated_peak_bytes>
          big_model_recommendation128.estimated_peak_bytes
    restricted=PIBasis(100,2;sectors=[(100,0)])
    @test basis_summary(restricted).state_bytes==estimate_state_bytes(restricted)
    @test basis_summary(restricted).state_bytes<estimate_memory(100,2)
    recommendation=recommend_solver(model;task=:steady_state,memory_budget=1)
    @test recommendation.backend===:matrixfree
    @test recommendation.heuristic
    @test recommendation.geometry_setup_upper_bytes==geometry_estimate.setup_bytes
    @test recommendation.geometry_requirement===:required
    @test recommendation.geometry_assumption_source===:model_terms
    @test recommendation.scalar_storage_estimate===:exact_inline
    @test recommendation.estimated_peak_bytes>recommendation.selected_solver_bytes
    @test recommendation.fits_memory===false
    @test recommendation.selected_solver_bytes==recommendation.gmres_vector_bytes
    spectrum_recommendation=recommend_solver(model;task=:spectrum)
    @test spectrum_recommendation.selected_solver_bytes==
          spectrum_recommendation.arnoldi_vector_bytes

    # Models known not to use one-body lowering do not pay that setup bound.
    # A bare basis has no term provenance and therefore remains conservative.
    empty_model=PIModel(b,())
    empty_recommendation=recommend_solver(empty_model)
    @test empty_recommendation.geometry_requirement===:not_required
    @test empty_recommendation.geometry_setup_upper_bytes==0
    compiled_empty=compile(empty_model;backend=:matrixfree)
    compiled_recommendation=recommend_solver(compiled_empty)
    @test compiled_recommendation.geometry_requirement===:not_required
    @test compiled_recommendation.geometry_setup_upper_bytes==0
    basis_recommendation=recommend_solver(b)
    @test basis_recommendation.geometry_requirement===:conservative_unknown
    @test basis_recommendation.geometry_setup_upper_bytes==
          geometry_estimate.setup_bytes

    big_recommendation=recommend_solver(empty_model;T=Complex{BigFloat},
        bigfloat_precision=256)
    @test big_recommendation.scalar_storage_estimate===
          :conservative_retained_bound
    @test big_recommendation.bigfloat_precision_assumption==256

    state_report=diagnostics(rho)
    @test state_report.valid
    @test occursin("PIState",sprint(show,rho))
    @test occursin("PIModel",sprint(show,model))
    @test occursin("DynamicsResult",sprint(show,sol))

    badH=PIOperator(b)
    coefficient_block(badH,b.sectors[1])[1,2]=1
    @test_throws ArgumentError thermal_state(badH,1.0)
end
