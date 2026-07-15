include(joinpath(@__DIR__,"..","examples","paper_models.jl"))
using .PaperModels

@testset "PRA 94, 033838 (2016), Fig. 6 and Eqs. 41-43" begin
    for gamma in (1.0,0.75,0.0)
        m=damanet2016_model(2;gamma0=1.0,gamma=gamma);b=m.basis
        rho0=iid_pure_state(b,ComplexF64[0,1]); L=Matrix(liouvillian(m;representation=:sparse))
        Iop=damanet2016_intensity_operator(b;gamma0=1.0,gamma=gamma)
        for t in range(0,3;length=31)
            rho=PIState(b,exp(t*L)*rho0.data)
            @test real(expectation(rho,Iop)) ≈ damanet2016_intensity_exact(t;gamma0=1,gamma=gamma) atol=2e-11 rtol=2e-11
        end
    end
end


@testset "additional literature models" begin
    sx=ComplexF64[0 1;1 0]
    for N in (2,4,6), t in (0.0,0.17,0.41)
        b=PIBasis(N,2); rho=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
        L=Matrix(liouvillian(kitagawa1993_oat_model(N;chi=.7);representation=:sparse))
        rt=PIState(b,exp(t*L)*rho.data)
        @test collective_expectation(rt,sx/2) ≈ kitagawa1993_mean_spin_exact(N,t;chi=.7) atol=2e-10
    end
    for N in (1,3,5), t in (0.0,.2,.8)
        b=PIBasis(N,2); rho=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
        L=Matrix(liouvillian(huelga1997_dephasing_model(N;gamma=.4);representation=:sparse))
        rt=PIState(b,exp(t*L)*rho.data)
        @test collective_expectation(rt,sx/2) ≈ huelga1997_ramsey_exact(N,t;gamma=.4) atol=2e-10
    end
    m=shammah2018_thermal_model(4;down=1.0,up=.3)
    exact=shammah2018_thermal_state(m.basis;down=1.0,up=.3)
    @test steady_state(m) ≈ exact.data atol=2e-9

    for gamma in (0.12,0.3)
        m=morrison2008_model(4;Omega=.2,gamma=gamma)
        exact=morrison2008_exact_state(m.basis;Omega=.2,gamma=gamma)
        @test steady_state(m)≈exact.data atol=3e-10
    end
    ms=meiser2009_superradiance_model(4;GammaC=1,pump=2)
    rss=PIState(ms.basis,steady_state(ms));sm=ComplexF64[0 1;0 0];Jm=collective_operator(ms.basis,sm)
    @test real(expectation(rss,adjoint(Jm)*Jm))>0
    for N in (6,8)
        mb=iemini2018_btc_model(N;omega0=1.5,kappa=1)
        vals=eigvals(Matrix(liouvillian(mb;representation=:sparse)))
        @test any(abs(imag(z))>0.5 for z in vals)
    end
end

@testset "dissipative time-crystal literature models" begin
    # Nakanishi--Sasamoto Eq. (14) is an exact finite-N oracle. Match the
    # spectra as multisets because the dense eigensolver may permute exact
    # degeneracies arbitrarily.
    N=5;g=1.3;kappa=0.4
    model=nakanishi2023_pt_model(N;g=g,kappa=kappa,p=0)
    numerical=collect(eigvals(Matrix(liouvillian(model;representation=:sparse))))
    remaining=collect(nakanishi2023_pt_spectrum(N;g=g,kappa=kappa))
    maximum_error=0.0
    for z in numerical
        j=argmin(abs.(remaining.-z))
        maximum_error=max(maximum_error,abs(remaining[j]-z))
        deleteat!(remaining,j)
    end
    @test isempty(remaining)
    @test maximum_error<2e-10
    @test pi_liouvillian_gap(model)≈4kappa/N atol=2e-10 rtol=2e-10
    uniform=maximally_mixed_state(model.basis)
    @test norm(liouvillian(model;representation=:sparse)*uniform.data)<2e-12

    # Piccitto et al. use normalized Pauli magnetizations. Verify that the
    # p-body/collective implementation equals Eqs. (1), (2), and (5) written
    # directly in PI operator coordinates, including every factor of N.
    N=4;omega_z=0.7;omega_x=1.1;Gamma_up=0.23;Gamma_down=0.08
    interacting=piccitto2021_interacting_btc_model(N;omega_z=omega_z,
        omega_x=omega_x,Gamma_up=Gamma_up,Gamma_down=Gamma_down)
    b=interacting.basis;s=spin_matrices(2)
    Sx=collective_operator(b,2s.jx);Sz=collective_operator(b,2s.jz)
    Jp=collective_operator(b,s.jp)*(2/N)
    Jm=collective_operator(b,s.jm)*(2/N)
    H=-(omega_z/N)*(Sz*Sz)-omega_x*Sx
    direct=PIModel(b,[DirectPIHamiltonian(H),
        DirectPIJump(Jp;rate=N*Gamma_up),
        DirectPIJump(Jm;rate=N*Gamma_down)])
    @test Matrix(liouvillian(interacting;representation=:sparse))≈
          Matrix(liouvillian(direct;representation=:sparse)) atol=3e-12 rtol=3e-12
end

@testset "PRA 110, 062208 (2024), Eqs. 2-6" begin
    for d in (2,3), dissipator in (:spin,:equal)
        m=pausch2024_model(3,d;V=1,gammaI=.4,gammaC=.2,dissipator=dissipator)
        @test check_generator(m).trace_preservation_error < 2e-10
        L=liouvillian(m;representation=:sparse)
        @test size(L)==(commutant_dimension(3,d),commutant_dimension(3,d))
    end
    # At V=0, decay drives the exactly polarized |-j> tensor power to itself.
    m=pausch2024_model(4,2;V=0,gammaI=.7,gammaC=.3)
    ground=iid_pure_state(m.basis,ComplexF64[1,0])
    @test norm(liouvillian(m;representation=:sparse)*ground.data)<2e-12
end
