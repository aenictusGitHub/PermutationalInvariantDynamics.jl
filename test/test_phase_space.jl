function _phase_space_test_integral(values,theta,phi)
    dphi=2pi/length(phi)
    azimuthal=[sum(view(values,:,index))*dphi for index in eachindex(theta)]
    total=zero(eltype(values))
    for index in 1:length(theta)-1
        step=theta[index+1]-theta[index]
        total+=step*(azimuthal[index]*sin(theta[index])+
                     azimuthal[index+1]*sin(theta[index+1]))/2
    end
    total
end

@testset "sector-resolved spin phase space" begin
    @testset "single-qubit analytical conventions" begin
        basis=PIBasis(1,2)
        theta0=0.83
        phi0=0.47
        rho=spin_coherent_state(basis,theta0,phi0)
        theta=[0.21,1.07,2.42]
        phi=[-0.37,0.62,1.91]
        husimi=spin_husimi_q(rho,theta,phi;resolved=true)
        wigner=spin_wigner(rho,theta,phi;resolved=true)

        bloch=(sin(theta0)*cos(phi0),sin(theta0)*sin(phi0),cos(theta0))
        expected_q=similar(husimi.values)
        expected_w=similar(wigner.values)
        for theta_index in eachindex(theta),phi_index in eachindex(phi)
            direction=(sin(theta[theta_index])*cos(phi[phi_index]),
                       sin(theta[theta_index])*sin(phi[phi_index]),
                       cos(theta[theta_index]))
            overlap=sum(bloch[index]*direction[index] for index in 1:3)
            expected_q[phi_index,theta_index]=(1+overlap)/(4pi)
            expected_w[phi_index,theta_index]=(1+sqrt(3)*overlap)/(4pi)
        end
        @test husimi.values≈expected_q atol=4e-15 rtol=4e-15
        @test wigner.values≈expected_w atol=5e-15 rtol=5e-15
        @test only(husimi.sector_values)≈husimi.values atol=0 rtol=0
        @test only(wigner.sector_values)≈wigner.values atol=0 rtol=0
        @test husimi.normalization===:sphere_density
        @test wigner.normalization===:sphere_density
        @test husimi.twice_spins==[1]
        @test husimi.multiplicities==BigInt[1]
        @test husimi.populations≈[1.0] atol=2e-15
        @test size(husimi.values)==(length(phi),length(theta))

        # GT blocks are ordered m=+j,...,-j.  The north-pole coherent state
        # must therefore peak in the first Dicke coordinate, not the last.
        north=spin_husimi_q(spin_coherent_state(basis,0,0),[0.0,pi],[0.0])
        @test north.values[1,1]≈1/(2pi) atol=2e-15
        @test abs(north.values[1,2])<1e-30
        north_wigner=spin_wigner(spin_coherent_state(basis,0,0),
                                  [0.0,pi],[0.0])
        @test north_wigner.values[1,1]≈(1+sqrt(3))/(4pi) atol=3e-15
        @test north_wigner.values[1,2]≈(1-sqrt(3))/(4pi) atol=3e-15
        @test north_wigner.values[1,2]<0 # negativity is not clipped

        # A higher-spin coherent state checks the polarization-tensor phases,
        # not only the qubit closed form: covariance makes its Wigner value
        # depend solely on the angular separation from the coherent axis.
        spin_one_basis=PIBasis(2,2)
        rotated=spin_coherent_state(spin_one_basis,theta0,phi0)
        rotated_wigner=spin_wigner(rotated,theta,phi)
        north_spin_one=spin_coherent_state(spin_one_basis,0,0)
        covariance_error=0.0
        for theta_index in eachindex(theta),phi_index in eachindex(phi)
            cosine=cos(theta0)*cos(theta[theta_index])+
                   sin(theta0)*sin(theta[theta_index])*
                   cos(phi[phi_index]-phi0)
            separation=acos(clamp(cosine,-1,1))
            reference=spin_wigner(
                north_spin_one,[separation],[0.0]).values[1,1]
            covariance_error=max(covariance_error,abs(
                rotated_wigner.values[phi_index,theta_index]-reference))
        end
        @test covariance_error<8e-15
    end

    @testset "sector weights, normalization, and restricted marginals" begin
        N=3
        basis=PIBasis(N,2)
        weight=0.63
        symmetric=spin_coherent_state(basis,0.72,-0.31)
        lower=dicke_state(basis,1/2,-1/2)
        rho=PIState(basis,weight.*symmetric.data.+(1-weight).*lower.data)
        theta=collect(range(0,pi;length=321))
        phi=[2pi*(index-1)/48 for index in 1:48]
        husimi=spin_husimi_q(rho,theta,phi;resolved=true)
        wigner=spin_wigner(rho,theta,phi;resolved=true)

        @test husimi.sectors==basis.sectors
        @test husimi.twice_spins==[3,1]
        @test husimi.multiplicities==BigInt[1,2]
        @test husimi.populations≈[weight,1-weight] atol=3e-14 rtol=3e-14
        @test wigner.populations≈husimi.populations atol=3e-14 rtol=3e-14
        @test husimi.values≈husimi.sector_values[1].+husimi.sector_values[2]
        @test wigner.values≈wigner.sector_values[1].+wigner.sector_values[2]
        for index in eachindex(husimi.populations)
            @test isapprox(_phase_space_test_integral(
                husimi.sector_values[index],theta,phi),husimi.populations[index];
                atol=4e-5,rtol=4e-5)
            @test isapprox(_phase_space_test_integral(
                wigner.sector_values[index],theta,phi),wigner.populations[index];
                atol=8e-5,rtol=8e-5)
        end
        @test isapprox(_phase_space_test_integral(husimi.values,theta,phi),1;
                       atol=5e-5,rtol=5e-5)
        @test isapprox(_phase_space_test_integral(wigner.values,theta,phi),1;
                       atol=9e-5,rtol=9e-5)

        symmetric_partition=Partition((3,0))
        selected=spin_husimi_q(rho,theta[1:7],phi[1:5];
            sectors=symmetric_partition,resolved=true)
        @test selected.sectors==[symmetric_partition]
        @test selected.populations≈[weight] atol=3e-14
        @test selected.values==only(selected.sector_values)
        tuple_selected=spin_wigner(rho,theta[1:7],phi[1:5];sectors=(2,1))
        @test tuple_selected.sectors==[Partition((2,1))]
        @test tuple_selected.sector_values===nothing
        @test tuple_selected.metadata.selected_population≈1-weight atol=3e-14
    end

    @testset "uniform sectors and scalar precision" begin
        basis=PIBasis(4,2)
        mixed=maximally_mixed_state(basis;T=Float32)
        theta=Float32[0.1,0.8,2.4]
        phi=Float32[0,0.7,1.4]
        q=spin_husimi_q(mixed,theta,phi;resolved=true,
            atol=2f-6,rtol=2f-5)
        w=spin_wigner(mixed,theta,phi;resolved=true,
            atol=2f-6,rtol=2f-5)
        @test eltype(q.values)===Float32
        @test eltype(w.values)===Float32
        @test eltype(q.theta)===Float32
        @test q.values≈fill(Float32(1/(4pi)),size(q.values)) atol=2f-6 rtol=2f-6
        @test w.values≈fill(Float32(1/(4pi)),size(w.values)) atol=2f-6 rtol=2f-6
        for index in eachindex(q.populations)
            @test q.sector_values[index]≈fill(
                q.populations[index]/Float32(4pi),size(q.values)) atol=2f-6
            @test w.sector_values[index]≈fill(
                w.populations[index]/Float32(4pi),size(w.values)) atol=2f-6
        end

        singlet=dicke_state(basis,0,0)
        singlet_q=spin_husimi_q(singlet,Float64[0,1,pi],Float64[0,1])
        singlet_w=spin_wigner(singlet,Float64[0,1,pi],Float64[0,1])
        @test singlet_q.values≈fill(1/(4pi),size(singlet_q.values)) atol=2e-15
        @test singlet_w.values≈fill(1/(4pi),size(singlet_w.values)) atol=2e-15

        setprecision(128) do
            big_state=spin_coherent_state(PIBasis(1,2),big"0.4",big"-0.2";
                                          T=BigFloat)
            big_q=spin_husimi_q(big_state,BigFloat[0,1],BigFloat[0,1];
                                atol=big"1e-30",rtol=big"1e-28")
            big_w=spin_wigner(big_state,BigFloat[0,1],BigFloat[0,1];
                              atol=big"1e-30",rtol=big"1e-28")
            @test eltype(big_q.values)===BigFloat
            @test eltype(big_w.values)===BigFloat
        end
    end

    @testset "input validation" begin
        rho=spin_coherent_state(PIBasis(2,2),0.4,0.1)
        @test_throws ArgumentError spin_husimi_q(rho,Float64[],[0.0])
        @test_throws ArgumentError spin_husimi_q(rho,[0.0],Float64[])
        @test_throws ArgumentError spin_husimi_q(rho,[-0.1],[0.0])
        @test_throws ArgumentError spin_husimi_q(rho,[pi+0.1],[0.0])
        @test_throws ArgumentError spin_wigner(rho,[0.2],[Inf])
        @test_throws ArgumentError spin_husimi_q(
            rho,[0.2],[0.1];normalization=:overlap)
        @test_throws ArgumentError spin_wigner(
            rho,[0.2],[0.1];sectors=[(2,0),(2,0)])
        @test_throws ArgumentError spin_husimi_q(
            rho,[0.2],[0.1];sectors=Partition((1,1,0)))
        @test_throws ArgumentError spin_husimi_q(rho;ntheta=1)
        @test_throws ArgumentError spin_wigner(rho;nphi=1)
        @test_throws ArgumentError spin_husimi_q(
            maximally_mixed_state(PIBasis(2,3)),[0.2],[0.1])

        invalid=state_from_schur_blocks(PIBasis(1,2),[
            Partition((1,0))=>ComplexF64[1.1 0;0 -0.1]])
        @test_throws ArgumentError spin_husimi_q(invalid,[0.2],[0.1])
        nonhermitian=PIState(rho.basis,copy(rho.data))
        coefficient_block(nonhermitian,first(rho.basis.sectors))[1,2]+=0.1im
        @test_throws ArgumentError spin_wigner(
            nonhermitian,[0.2],[0.1];atol=1e-12,rtol=0)
    end

    @testset "dependency-free SVG rendering" begin
        rho=spin_coherent_state(PIBasis(3,2),0.8,0.4)
        q=spin_husimi_q(rho;ntheta=17,nphi=32,resolved=true)
        w=spin_wigner(rho;ntheta=17,nphi=32,resolved=true)
        q_figure=visualize_spin_phase_space(q;title="Q & <phase>")
        w_figure=visualize_spin_phase_space(w)
        @test q_figure isa SpinPhaseSpaceVisualization
        @test q_figure.palette===:sequential
        @test w_figure.palette===:diverging
        @test w_figure.colorlimits[1]≈-w_figure.colorlimits[2]
        @test occursin("SpinPhaseSpaceVisualization",sprint(show,q_figure))
        q_svg=sprint(show,MIME("image/svg+xml"),q_figure)
        w_svg=sprint(show,MIME("image/svg+xml"),w_figure)
        @test occursin("<svg",q_svg)
        @test occursin("Q &amp; &lt;phase&gt;",q_svg)
        @test occursin("equirectangular",q_svg)
        @test occursin("sphere-density normalization",q_svg)
        @test count("<path",q_svg)<=length(PermutationalInvariantDynamics._SPIN_Q_PALETTE)
        @test occursin("#67001f",w_svg)||occursin("#053061",w_svg)

        sector_figure=visualize_spin_phase_space(
            q;sector=first(q.sectors),show_colorbar=false)
        sector_svg=sprint(show,MIME("image/svg+xml"),sector_figure)
        @test occursin("sector",sector_svg)
        unresolved=spin_husimi_q(rho;ntheta=5,nphi=8)
        @test_throws ArgumentError visualize_spin_phase_space(
            unresolved;sector=first(unresolved.sectors))
        irregular=spin_husimi_q(rho,[0.0,0.2,0.7],[0.0,0.3,0.6])
        @test_throws ArgumentError visualize_spin_phase_space(irregular)
        @test_throws ArgumentError visualize_spin_phase_space(q;width=300)
        @test_throws ArgumentError visualize_spin_phase_space(q;height=200)
        @test_throws ArgumentError visualize_spin_phase_space(
            q;colorlimits=(1,1))
        invalid_xml=visualize_spin_phase_space(q;title="bad\u0001title")
        @test_throws ArgumentError sprint(show,MIME("image/svg+xml"),invalid_xml)

        mktempdir() do directory
            path=joinpath(directory,"spin_phase_space.svg")
            @test save_spin_phase_space_visualization(path,w_figure)==path
            @test isfile(path)
            @test occursin("<svg",read(path,String))
            data_path=joinpath(directory,"spin_husimi.svg")
            @test save_spin_phase_space_visualization(data_path,q)==data_path
        end
        @test (@doc spin_husimi_q)!==nothing
        @test (@doc spin_wigner)!==nothing
        @test (@doc visualize_spin_phase_space)!==nothing
    end
end
