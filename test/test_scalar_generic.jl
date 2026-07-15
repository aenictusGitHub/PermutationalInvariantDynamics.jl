@testset "scalar-generic analysis paths" begin
    basis=PIBasis(3,2)
    local32=ComplexF32[0.7 0.1;0.1 0.3]
    rho32=iid_state(basis,local32)
    sx32=ComplexF32[0 1;1 0]
    sz32=ComplexF32[1 0;0 -1]
    plan32=CollectiveObservablePlan(basis,sx32)

    @test isphysical(rho32)
    @test eltype(plan32.local_operator)===ComplexF32
    @test eltype(first(plan32.blocks))===ComplexF32
    @test collective_expectation(rho32,plan32) isa ComplexF32
    @test collective_variance(rho32,plan32) isa Float32
    @test collective_covariance_matrix(rho32,[sx32,sz32]) isa Matrix{Float32}
    @test two_body_expectation(rho32,sx32,sz32) isa ComplexF32
    @test qfi(rho32,plan32) isa Float32
    @test qfim(rho32,[plan32]) isa Matrix{Float32}
    @test von_neumann_entropy(rho32) isa Float32
    @test renyi_entropy(rho32,2) isa Float32
    @test trace_distance(rho32,rho32) isa Float32
    @test fidelity(rho32,rho32) isa Float32
    @test bures_distance(rho32,rho32) isa Float32
    @test quantum_relative_entropy(rho32,rho32) isa Float32
    @test one_body_rdm(rho32) isa Matrix{ComplexF32}

    zero_tangent32=PIState(basis;T=Float32)
    @test qfim_from_derivatives(rho32,[zero_tangent32]) isa Matrix{Float32}
    slds32=symmetric_logarithmic_derivatives(rho32,[sx32,sz32])
    @test all(L->eltype(L.data)===ComplexF32,slds32)
    @test sld_commutator_matrix(rho32,[sx32,sz32]) isa Matrix{Float32}
    @test mean_uhlmann_curvature(rho32,[sx32,sz32]) isa Matrix{Float32}
    @test kitagawa_ueda_squeezing(rho32) isa Float32

    sectors32=sector_resolved_entropy(rho32)
    @test all(x->x.probability isa Float32,sectors32)
    @test entropy_decomposition(rho32).total isa Float32
    @test relative_entropy_of_coherence(rho32) isa Float32
    @test eltype(symmetry_twirl(rho32,sz32).data)===ComplexF32
    @test relative_entropy_of_asymmetry(rho32,sz32) isa Float32
    @test wigner_yanase_asymmetry(rho32,sz32) isa Float32
    @test all(x->x.qfi isa Float32,sector_resolved_qfi(rho32,sz32))
    @test relative_entropy_decomposition(rho32,rho32).total isa Float32
    sector_qfim32=qfim_sector_decomposition(rho32,[zero_tangent32])
    @test sector_qfim32.classical isa Matrix{Float32}
    @test sector_qfim32.intra_sector isa Matrix{Float32}

    sm32=ComplexF32[0 1;0 0]
    L32=liouvillian(PIModel(basis,[LocalJump(sm32)]);representation=:sparse)
    A32=collective_operator(plan32)
    @test eltype(liouvillian_modes(L32;k=2).values)===ComplexF32
    @test resolvent_norm(L32,ComplexF32(0,1)) isa Float32
    @test pseudospectral_abscissa(L32,0.1f0;
        real_grid=Float32[-1,0],imag_grid=Float32[0.3]) isa Float32
    @test eltype(adjoint_evolve(L32,A32,0.1f0).data)===ComplexF32
    sensitivity=sensitivity_problem(L32,rho32,(0.0f0,0.1f0),[Matrix(L32)])
    @test eltype(sensitivity.u0)===ComplexF32
    @test classical_fisher_information(rho32,[zero_tangent32],
        [identity_operator(basis;T=Float32)]) isa Matrix{Float32}
    @test integrated_correlation_time(L32,rho32,A32) isa Float32
    @test eltype(steady_state_susceptibility(
        L32,rho32,zero(Matrix(L32))).data)===ComplexF32

    rho64=iid_state(basis,ComplexF64.(local32))
    @test Float64(qfi(rho32,plan32))≈qfi(rho64,ComplexF64.(sx32)) rtol=2e-5
    @test Float64(von_neumann_entropy(rho32))≈von_neumann_entropy(rho64) rtol=2e-5

    setprecision(128) do
        bigrho=iid_state(PIBasis(2,2),Complex{BigFloat}[big"0.7" 0;0 big"0.3"])
        @test positivity_diagnostics(bigrho).positive
        error=try
            von_neumann_entropy(bigrho)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("generic eigensolver",sprint(showerror,error))

        relative_error=try
            quantum_relative_entropy(bigrho,bigrho)
            nothing
        catch caught
            caught
        end
        @test relative_error isa ArgumentError
        @test occursin("generic eigensolver",sprint(showerror,relative_error))

        response_error=try
            liouvillian_modes(Complex{BigFloat}[0 1;-1 0];k=1)
            nothing
        catch caught
            caught
        end
        @test response_error isa ArgumentError
        @test occursin("generic linear-algebra backend",sprint(showerror,response_error))
    end
end
