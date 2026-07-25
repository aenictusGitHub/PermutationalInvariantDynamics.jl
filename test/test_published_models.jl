include(joinpath(@__DIR__,"..","examples","paper_models.jl"))
using .PaperModels
include(joinpath(@__DIR__,"..","examples","utils","contour_fit.jl"))
using .ExampleContourFit

@testset "finite-grid quadratic contour fits" begin
    x=[0.5,1.0,1.5]
    y=[0.1,1.0,3.0,6.0]
    alpha=2.0
    values=[yvalue-alpha*xvalue^2 for xvalue in x,yvalue in y]
    fit=quadratic_level_contour_fit(x,y,values;level=0.0,samples=31)
    # The physical contour is nonlinear between these deliberately sparse
    # x nodes.  The helper promises linear edge interpolation, not exact
    # recovery of the unsampled parabola.  Vertical roots are exact because
    # this field is linear in y; horizontal roots are the corresponding
    # secant intersections.
    expected_x=[0.5,2/3,1.0,1.2,1.5]
    expected_y=[0.5,1.0,2.0,3.0,4.5]
    expected_squared_x=expected_x.^2
    expected_alpha=dot(expected_squared_x,expected_y)/
        dot(expected_squared_x,expected_squared_x)
    @test fit.crossing_x≈expected_x atol=2e-14
    @test fit.crossing_y≈expected_y atol=2e-14
    @test fit.alpha≈expected_alpha atol=2e-14
    @test fit.residuals≈
          expected_y.-expected_alpha.*expected_squared_x atol=2e-14
    @test abs(fit.alpha-alpha)>1e-3
    @test fit.point_count==5
    @test fit.horizontal_count==2
    @test fit.vertical_count==3
    @test sort(fit.boundary_sides)==[:left,:right]
    @test length(fit.fit_x)==length(fit.fit_y)==31
    @test extrema(fit.fit_x)==extrema(fit.crossing_x)
    @test fit.interpolation===:linear_physical_coordinates
    @test fit.objective===:origin_constrained_least_squares
    @test length(fit.leave_one_out_alpha)==fit.point_count
    @test all(isfinite,fit.pointwise_alpha)
    @test fit.fallback_triggered===false
    @test fit.fallback_status===:not_triggered
    @test fit.fallback_model===:general_quadratic
    @test fit.general_quadratic===nothing
    @test fit.power_law===nothing
    @test fit.display_model===:origin_constrained

    shifted=0.5 .+ values
    half_fit=quadratic_level_contour_fit(x,y,shifted;level=0.5)
    @test half_fit.alpha≈fit.alpha atol=2e-14
    @test half_fit.crossing_x≈fit.crossing_x atol=2e-14
    @test half_fit.crossing_y≈fit.crossing_y atol=2e-14

    fine_x=collect(range(0.5,1.5;length=21))
    fine_y=collect(range(0.1,6.0;length=22))
    fine_values=[yvalue-alpha*xvalue^2
                 for xvalue in fine_x,yvalue in fine_y]
    fine_fit=quadratic_level_contour_fit(
        fine_x,fine_y,fine_values;level=0.0)
    @test abs(fine_fit.alpha-alpha)<abs(fit.alpha-alpha)
    @test fine_fit.relative_l2_residual<fit.relative_l2_residual

    values32=Float32.(values)
    fit32=quadratic_level_contour_fit(
        Float32.(x),Float32.(y),values32;level=0.0f0)
    @test fit32.alpha isa Float32
    @test eltype(fit32.crossing_x)===Float32
    @test eltype(fit32.fit_y)===Float32

    exact_x=[0.5,1.0,2.0]
    exact_y=[0.5,2.0,8.0]
    exact_values=[yvalue-2xvalue^2
                  for xvalue in exact_x,yvalue in exact_y]
    exact_fit=quadratic_level_contour_fit(
        exact_x,exact_y,exact_values;level=0.0)
    @test exact_fit.alpha≈2 atol=2e-14
    @test exact_fit.point_count==3
    @test exact_fit.horizontal_count==3
    @test exact_fit.vertical_count==3
    @test sort(exact_fit.boundary_sides)==[:bottom,:left,:right,:top]
    exact_zero_threshold=quadratic_level_contour_fit(
        exact_x,exact_y,exact_values;level=0.0,
        fallback_relative_l2_threshold=0.0)
    @test exact_zero_threshold.fallback_triggered===false

    general_quadratic(x)=1.25x^2+0.5x+1
    general_crossing_x=[0.5,1.0,1.5,2.0]
    general_x=[0.25;general_crossing_x;2.5]
    general_y=general_quadratic.(general_crossing_x)
    general_values=[yvalue-general_quadratic(xvalue)
                    for xvalue in general_x,yvalue in general_y]
    general_fit=quadratic_level_contour_fit(
        general_x,general_y,general_values;level=0.0,samples=17)
    @test general_fit.fallback_triggered===true
    @test general_fit.fallback_status===:selected
    @test general_fit.fallback_model===:general_quadratic
    @test general_fit.display_model===:general_quadratic
    @test general_fit.crossing_x≈general_crossing_x atol=2e-14
    @test general_fit.crossing_y≈general_y atol=2e-14
    @test general_fit.relative_l2_residual>0.1
    @test general_fit.general_quadratic!==nothing
    @test general_fit.power_law===nothing
    full_fit=general_fit.general_quadratic
    @test full_fit.coefficients.quadratic≈1.25 atol=2e-14
    @test full_fit.coefficients.linear≈0.5 atol=2e-14
    @test full_fit.coefficients.constant≈1.0 atol=2e-14
    @test full_fit.relative_l2_residual<2e-14
    @test full_fit.residual_degrees_of_freedom==1
    @test full_fit.numerical_rank==3
    @test full_fit.plottable===true
    @test full_fit.fit_x_span==(0.5,2.0)
    @test length(full_fit.fit_x)==length(full_fit.fit_y)==17
    @test full_fit.fit_y≈
          full_fit.coefficients.quadratic.*full_fit.fit_x.^2 .+
          full_fit.coefficients.linear.*full_fit.fit_x .+
          full_fit.coefficients.constant atol=3e-14
    fit_type=eltype(full_fit.fit_y)
    domain_tolerance=fit_type(64)*eps(fit_type)*max(
        one(fit_type),abs(first(general_y)),abs(last(general_y)))
    @test minimum(full_fit.fit_y)>=first(general_y)-domain_tolerance
    @test maximum(full_fit.fit_y)<=last(general_y)+domain_tolerance
    @test full_fit.objective===:general_quadratic_least_squares

    suppressed_fit=quadratic_level_contour_fit(
        general_x,general_y,general_values;level=0.0,
        fallback_relative_l2_threshold=0.2)
    @test suppressed_fit.fallback_triggered===false
    @test suppressed_fit.display_model===:origin_constrained
    threshold_fit=quadratic_level_contour_fit(
        general_x,general_y,general_values;level=0.0,
        fallback_relative_l2_threshold=general_fit.relative_l2_residual)
    @test threshold_fit.fallback_triggered===false
    below_threshold_fit=quadratic_level_contour_fit(
        general_x,general_y,general_values;level=0.0,
        fallback_relative_l2_threshold=
            prevfloat(general_fit.relative_l2_residual))
    @test below_threshold_fit.display_model===:general_quadratic

    general_fit32=quadratic_level_contour_fit(
        Float32.(general_x),Float32.(general_y),Float32.(general_values);
        level=0.0f0,fallback_relative_l2_threshold=0.1f0)
    @test general_fit32.display_model===:general_quadratic
    @test general_fit32.general_quadratic.coefficients.quadratic isa Float32
    @test eltype(general_fit32.general_quadratic.fit_x)===Float32
    @test general_fit32.general_quadratic.coefficients.quadratic≈1.25f0 atol=2e-5

    power_curve(x)=1.7*x^1.35
    power_crossing_x=[0.5,1.0,1.5,2.0]
    power_x=[0.25;power_crossing_x;2.5]
    power_y=power_curve.(power_crossing_x)
    power_values=[yvalue-power_curve(xvalue)
                  for xvalue in power_x,yvalue in power_y]
    power_fit=quadratic_level_contour_fit(
        power_x,power_y,power_values;level=0.0,samples=23,
        fallback_model=:power_law,
        fallback_relative_l2_threshold=0.0)
    @test power_fit.fallback_triggered===true
    @test power_fit.fallback_status===:selected
    @test power_fit.fallback_model===:power_law
    @test power_fit.display_model===:power_law
    @test power_fit.general_quadratic===nothing
    @test power_fit.power_law!==nothing
    @test power_fit.crossing_x≈power_crossing_x atol=2e-14
    @test power_fit.crossing_y≈power_y atol=2e-14
    @test power_fit.relative_l2_residual>
          power_fit.power_law.relative_l2_residual
    power_candidate=power_fit.power_law
    @test power_candidate.alpha≈1.7 atol=5e-7
    @test power_candidate.beta≈1.35 atol=3e-7
    @test power_candidate.coefficients.alpha===power_candidate.alpha
    @test power_candidate.coefficients.beta===power_candidate.beta
    @test power_candidate.objective===
          :positive_power_law_original_y_least_squares
    @test power_candidate.optimizer===:profiled_golden_section
    @test power_candidate.converged===true
    @test power_candidate.relative_l2_residual<2e-7
    @test power_candidate.residual_degrees_of_freedom==2
    @test power_candidate.numerical_rank==2
    @test power_candidate.plottable===true
    @test collect(power_candidate.fit_x_span)≈[0.5,2.0] atol=2e-7
    @test length(power_candidate.fit_x)==length(power_candidate.fit_y)==23
    @test power_candidate.fit_y≈(
        power_candidate.alpha.*power_candidate.fit_x.^power_candidate.beta) rtol=3e-14 atol=3e-14

    power_curve32(x)=1.7f0*x^1.35f0
    power_crossing_x32=Float32.(power_crossing_x)
    power_x32=Float32.([0.25;power_crossing_x;2.5])
    power_y32=power_curve32.(power_crossing_x32)
    power_values32=Float32[
        yvalue-power_curve32(xvalue)
        for xvalue in power_x32,yvalue in power_y32]
    power_fit32=quadratic_level_contour_fit(
        power_x32,power_y32,power_values32;level=0.0f0,samples=19,
        fallback_model=:power_law,
        fallback_relative_l2_threshold=0.0f0)
    @test power_fit32.display_model===:power_law
    @test power_fit32.general_quadratic===nothing
    @test power_fit32.power_law!==nothing
    power_candidate32=power_fit32.power_law
    @test power_candidate32.alpha isa Float32
    @test power_candidate32.beta isa Float32
    @test power_candidate32.alpha≈1.7f0 atol=8f-3
    @test power_candidate32.beta≈1.35f0 atol=5f-3
    @test power_candidate32.relative_l2_residual<5f-3
    @test power_candidate32.residual_degrees_of_freedom==2
    @test power_candidate32.numerical_rank==2
    @test eltype(power_candidate32.fit_x)===Float32
    @test eltype(power_candidate32.fit_y)===Float32

    sparse_x=[1.0,2.0]
    sparse_y=[2.0,3.0]
    sparse_values=[yvalue-(xvalue+1)
                   for xvalue in sparse_x,yvalue in sparse_y]
    sparse_fit=quadratic_level_contour_fit(
        sparse_x,sparse_y,sparse_values;level=0.0,
        fallback_relative_l2_threshold=0.1)
    @test sparse_fit.fallback_triggered===true
    @test sparse_fit.general_quadratic===nothing
    @test sparse_fit.fallback_status===
          :insufficient_residual_degrees_of_freedom
    @test sparse_fit.display_model===:raw_contour_only

    sparse_power_fit=quadratic_level_contour_fit(
        sparse_x,sparse_y,sparse_values;level=0.0,
        fallback_model=:power_law,
        fallback_relative_l2_threshold=0.1)
    @test sparse_power_fit.fallback_triggered===true
    @test sparse_power_fit.fallback_model===:power_law
    @test sparse_power_fit.general_quadratic===nothing
    @test sparse_power_fit.power_law===nothing
    @test sparse_power_fit.fallback_status===
          :insufficient_residual_degrees_of_freedom
    @test sparse_power_fit.display_model===:raw_contour_only

    @testset "contour text export" begin
        series_count(text,label)=count(
            line->startswith(line,"$label\t"),split(text,'\n'))
        function checked_export(path,fit;kwargs...)
            written=save_level_contour_data(path,fit;kwargs...)
            @test written==abspath(path)
            @test isfile(written)
            text=read(written,String)
            @test startswith(text,
                "# PermutationalInvariantDynamics.jl level-contour data\n")
            @test occursin("\nseries\tpoint_index\t",text)
            for line in split(text,'\n')
                isempty(line)||startswith(line,'#')||
                    startswith(line,"series\t")||
                    @test length(split(line,'\t'))==4
            end
            text
        end

        mktempdir() do directory
            origin_path=joinpath(directory,"nested","origin.txt")
            origin_text=checked_export(
                origin_path,fit;
                metadata=(;observable=:Cxx,N=3,note="first\nsecond\tfield"),
                x_label="J_over_omega_c",
                y_label="kappa_over_omega_c")
            @test occursin(
                "series\tpoint_index\tJ_over_omega_c\tkappa_over_omega_c",
                origin_text)
            @test occursin("# display_model = :origin_constrained",origin_text)
            @test occursin("# metadata.observable = :Cxx",origin_text)
            @test occursin("# metadata.N = 3",origin_text)
            @test occursin("first\\nsecond\\tfield",origin_text)
            @test series_count(origin_text,"extracted_boundary")==fit.point_count
            @test series_count(origin_text,"origin_quadratic_fit")==
                  length(fit.fit_x)
            @test series_count(origin_text,"general_quadratic_fit")==0
            @test series_count(origin_text,"power_law_fit")==0

            # A repeated write replaces the complete file, rather than
            # appending to it or exposing a partially written intermediate.
            replacement_text=checked_export(
                origin_path,fit;metadata=(;observable=:replacement))
            @test occursin("# metadata.observable = :replacement",
                           replacement_text)
            @test !occursin("# metadata.observable = :Cxx",replacement_text)
            @test count(==("origin.txt"),readdir(dirname(origin_path)))==1

            general_path=joinpath(directory,"general.txt")
            general_text=checked_export(general_path,general_fit)
            @test occursin("# display_model = :general_quadratic",general_text)
            @test occursin("# general_quadratic.available = true",general_text)
            @test occursin("# general_quadratic.coefficients = ",general_text)
            @test series_count(
                general_text,"general_quadratic_fit")==length(full_fit.fit_x)
            @test series_count(general_text,"power_law_fit")==0

            power_path=joinpath(directory,"power.txt")
            power_text=checked_export(power_path,power_fit)
            @test occursin("# display_model = :power_law",power_text)
            @test occursin("# power_law.available = true",power_text)
            @test occursin("# power_law.coefficients = ",power_text)
            @test series_count(power_text,"power_law_fit")==
                  length(power_candidate.fit_x)
            @test series_count(power_text,"general_quadratic_fit")==0

            raw_path=joinpath(directory,"raw.txt")
            raw_text=checked_export(raw_path,sparse_power_fit)
            @test occursin("# display_model = :raw_contour_only",raw_text)
            @test occursin("# fallback.status = " *
                           ":insufficient_residual_degrees_of_freedom",
                           raw_text)
            @test series_count(raw_text,"extracted_boundary")==
                  sparse_power_fit.point_count
            @test series_count(raw_text,"origin_quadratic_fit")==
                  length(sparse_power_fit.fit_x)
            @test series_count(raw_text,"general_quadratic_fit")==0
            @test series_count(raw_text,"power_law_fit")==0

            # Printing operates on the original coordinate values rather than
            # converting them through Float64.
            precise=parse(BigFloat,
                "0.50000000000000000000000000000000000000000000000001")
            precise_x=BigFloat.(fit.crossing_x)
            precise_x[1]=precise
            precise_fit=merge(fit,(
                crossing_x=precise_x,
                crossing_y=BigFloat.(fit.crossing_y),
                fit_x=BigFloat.(fit.fit_x),
                fit_y=BigFloat.(fit.fit_y)))
            precise_text=checked_export(
                joinpath(directory,"precise.txt"),precise_fit)
            @test occursin(string(precise),precise_text)

            mismatch=merge(fit,(crossing_y=fit.crossing_y[1:end-1],))
            @test_throws DimensionMismatch save_level_contour_data(
                joinpath(directory,"mismatch.txt"),mismatch)
            nonfinite_y=copy(fit.fit_y)
            nonfinite_y[1]=NaN
            nonfinite=merge(fit,(fit_y=nonfinite_y,))
            @test_throws ArgumentError save_level_contour_data(
                joinpath(directory,"nonfinite.txt"),nonfinite)
            nonreal_x=Any[fit.crossing_x...]
            nonreal_x[1]=1+im
            nonreal=merge(fit,(crossing_x=nonreal_x,))
            @test_throws ArgumentError save_level_contour_data(
                joinpath(directory,"nonreal.txt"),nonreal)
            @test_throws ArgumentError save_level_contour_data("",fit)
            existing_directory=mkpath(joinpath(directory,"not_a_file"))
            @test_throws ArgumentError save_level_contour_data(
                existing_directory,fit)
            for labels in ((x_label="",),
                           (x_label=" x",),
                           (x_label="x\ty",),
                           (x_label="series",),
                           (x_label="same",y_label="same"))
                @test_throws ArgumentError save_level_contour_data(
                    joinpath(directory,"bad_label.txt"),fit;labels...)
            end
            @test !ispath(joinpath(directory,"mismatch.txt"))
            @test !ispath(joinpath(directory,"nonfinite.txt"))
            @test !ispath(joinpath(directory,"nonreal.txt"))
        end
    end

    @test_throws DimensionMismatch quadratic_level_contour_fit(
        x,y,zeros(2,2);level=0.0)
    @test_throws ArgumentError quadratic_level_contour_fit(
        reverse(x),y,values;level=0.0)
    @test_throws ArgumentError quadratic_level_contour_fit(
        [0.5,1.0,1.0],y,values;level=0.0)
    @test_throws ArgumentError quadratic_level_contour_fit(
        [0.0,1.0,1.5],y,values;level=0.0)
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,[0.0,1.0,2.0,3.0],values;level=0.0)
    nonfinite=copy(values);nonfinite[1,1]=Inf
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,nonfinite;level=0.0)
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,values;level=NaN)
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,values;level=0.0,samples=1)
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,values;level=0.0,fallback_model=:not_a_fallback)
    for threshold in (-0.1,NaN,Inf)
        @test_throws ArgumentError quadratic_level_contour_fit(
            x,y,values;level=0.0,
            fallback_relative_l2_threshold=threshold)
    end
    disabled_fit=quadratic_level_contour_fit(
        general_x,general_y,general_values;level=0.0,
        fallback_relative_l2_threshold=nothing)
    @test disabled_fit.fallback_triggered===false
    @test disabled_fit.fallback_status===:disabled
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,ones(length(x),length(y));level=0.0)
    single=[(xvalue-0.5)+(yvalue-0.1)
            for xvalue in x,yvalue in y]
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,single;level=0.0)
    multiple=[(xvalue-0.75)*(xvalue-1.25)
              for xvalue in x,yvalue in y]
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,multiple;level=0.0)
    multiple_vertical=[(yvalue-0.75)*(yvalue-2.0)
                       for xvalue in x,yvalue in y]
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,multiple_vertical;level=0.0)
    @test_throws ArgumentError quadratic_level_contour_fit(
        x,y,zeros(length(x),length(y));level=0.0)

    # Opposite finite extrema must interpolate at the midpoint instead of
    # overflowing `right-left` and collapsing onto an endpoint.
    largest=floatmax(Float64)
    extreme=[xindex==1 ? largest : -largest
             for xindex in 1:2,yindex in 1:2]
    extreme_fit=quadratic_level_contour_fit(
        [1.0,2.0],[1.0,2.0],extreme;level=0.0)
    @test extreme_fit.crossing_x==[1.5,1.5]
    @test extreme_fit.crossing_y==[1.0,2.0]
    extreme_level=-0.75*largest
    shifted_extreme_fit=quadratic_level_contour_fit(
        [1.0,2.0],[1.0,2.0],extreme;level=extreme_level)
    @test shifted_extreme_fit.crossing_x≈[1.875,1.875] atol=4eps(Float64)
    @test shifted_extreme_fit.crossing_y==[1.0,2.0]
end

@testset "PRA 94, 033838 (2016), Fig. 6 and Eqs. 41-43" begin
    for gamma in (1.0,0.75,0.0)
        m=correlated_superradiance_model(2;gamma0=1.0,gamma=gamma);b=m.basis
        rho0=iid_pure_state(b,ComplexF64[0,1]); L=Matrix(liouvillian(m;representation=:sparse))
        Iop=correlated_superradiance_intensity_operator(b;gamma0=1.0,gamma=gamma)
        for t in range(0,3;length=31)
            rho=PIState(b,exp(t*L)*rho0.data)
            @test real(expectation(rho,Iop)) ≈ two_qubit_correlated_superradiance_intensity_exact(t;gamma0=1,gamma=gamma) atol=2e-11 rtol=2e-11
        end
    end
end


@testset "additional literature models" begin
    sx=ComplexF64[0 1;1 0]
    for N in (2,4,6), t in (0.0,0.17,0.41)
        b=PIBasis(N,2); rho=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
        L=Matrix(liouvillian(one_axis_twisting_model(N;chi=.7);representation=:sparse))
        rt=PIState(b,exp(t*L)*rho.data)
        @test collective_expectation(rt,sx/2) ≈ one_axis_twisting_mean_spin_exact(N,t;chi=.7) atol=2e-10
    end
    for N in (1,3,5), t in (0.0,.2,.8)
        b=PIBasis(N,2); rho=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
        L=Matrix(liouvillian(independent_dephasing_model(N;gamma=.4);representation=:sparse))
        rt=PIState(b,exp(t*L)*rho.data)
        @test collective_expectation(rt,sx/2) ≈ independent_dephasing_coherence_exact(N,t;gamma=.4) atol=2e-10
    end
    m=local_pump_decay_model(4;down=1.0,up=.3)
    exact=local_pump_decay_steady_state(m.basis;down=1.0,up=.3)
    @test steady_state(m) ≈ exact.data atol=2e-9

    # Zhang--Zhang--Mølmer Eq. (1), decay-only specialization.  These two
    # Dicke states fix both ladder conventions and every rate prefactor in
    # Ic=GammaC<J+J-> and Ifs=gammaL(N/2+<Jz>).
    zhang=collective_local_decay_model(4;GammaC=.3,gammaL=.7)
    radiation=collective_local_radiation_operators(
        zhang.basis;GammaC=.3,gammaL=.7)
    fully_excited=iid_pure_state(zhang.basis,ComplexF64[0,1])
    central=dicke_state(zhang.basis,2,0)
    ground=iid_pure_state(zhang.basis,ComplexF64[1,0])
    @test real(expectation(fully_excited,radiation.cavity))≈.3*4 atol=2e-13
    @test real(expectation(fully_excited,radiation.free_space))≈.7*4 atol=2e-13
    @test real(expectation(central,radiation.cavity))≈.3*6 atol=2e-13
    @test real(expectation(central,radiation.free_space))≈.7*2 atol=2e-13
    @test iszero(real(expectation(ground,radiation.cavity)))
    @test iszero(real(expectation(ground,radiation.free_space)))
    zhang_populations=PopulationPlan(zhang)
    @test zhang_populations.invariance.invariant===true
    @test size(population_generator(
        zhang_populations;representation=:sparse))==(9,9)
    zhang32=collective_local_decay_model(
        2;GammaC=Float32(.3),gammaL=Float32(.7))
    radiation32=collective_local_radiation_operators(
        zhang32.basis;GammaC=Float32(.3),gammaL=Float32(.7))
    @test eltype(liouvillian(zhang32;representation=:sparse))===ComplexF32
    @test eltype(radiation32.cavity.data)===ComplexF32
    @test eltype(radiation32.free_space.data)===ComplexF32
    @test_throws ArgumentError collective_local_decay_model(0)
    @test_throws ArgumentError collective_local_decay_model(2;gammaL=-1)

    for gamma in (0.12,0.3)
        m=cooperative_fluorescence_model(4;Omega=.2,gamma=gamma)
        exact=cooperative_fluorescence_exact_state(m.basis;Omega=.2,gamma=gamma)
        @test steady_state(m)≈exact.data atol=3e-10
    end
    ms=steady_superradiance_model(4;GammaC=1,pump=2)
    rss=PIState(ms.basis,steady_state(ms));sm=ComplexF64[0 1;0 0];Jm=collective_operator(ms.basis,sm)
    @test real(expectation(rss,adjoint(Jm)*Jm))>0
    for N in (6,8)
        mb=boundary_time_crystal_model(N;omega0=1.5,kappa=1)
        vals=eigvals(Matrix(liouvillian(mb;representation=:sparse)))
        @test any(abs(imag(z))>0.5 for z in vals)
    end
end

@testset "Debecker 2026 all-to-all Ising pseudomode specialization" begin
    # One retained boson is enough to fix the supersite ordering and scalar
    # contract without making this literature-model gate expensive.
    operators32=local_pseudomode_operators(1;T=Float32)
    @test (operators32.levels,operators32.dsite)==(2,4)
    @test all(matrix->eltype(matrix)===ComplexF32,
              (operators32.spin_paulis...,operators32.lifted_paulis...,
               operators32.mode_annihilation,operators32.mode_number,
               operators32.mode_top,operators32.exchange_minus,
               operators32.exchange_z))

    model32=all_to_all_xx_spin_local_pseudomode_model(
        2,1;Jpair=.23f0,omega_c=.9f0,gamma=.04f0,kappa=.5f0)
    @test model32.basis.d==4
    @test model32.terms[1] isa LocalHamiltonian
    @test model32.terms[2] isa LocalHamiltonian
    @test model32.terms[3] isa PBodyHamiltonian
    @test model32.terms[4] isa LocalJump
    @test model32.terms[1].rate===.9f0
    @test model32.terms[2].rate isa Float32
    @test model32.terms[2].rate≈Float32(sqrt(.04*.5)) rtol=eps(Float32)
    @test model32.terms[3].rate===-.23f0
    # The manuscript uses twice the package dissipator, so its kappa becomes
    # the package rate 2kappa.
    @test model32.terms[4].rate===1.0f0
    @test eltype(liouvillian(model32;representation=:sparse))===ComplexF32

    Jpair=.23;omega_c=.9;gamma=.04;kappa=.5
    operators=local_pseudomode_operators(1)
    basis=PIBasis(2,operators.dsite)
    model=all_to_all_xx_spin_local_pseudomode_model(
        basis,operators;Jpair,omega_c,gamma,kappa)
    sparse=liouvillian(model;representation=:sparse)
    matrixfree=liouvillian(model;representation=:matrixfree)
    local_vacuum=ComplexF64[1,0,0,0]
    rho0=iid_pure_state(basis,local_vacuum)
    for (particles,expected_pi,expected_weak) in
            ((3,816,44),(4,3876,116))
        size_basis=PIBasis(particles,operators.dsite)
        size_state=iid_pure_state(size_basis,local_vacuum)
        size_pseudoket=weak_pi_pseudoket(size_state)
        @test length(size_basis)==expected_pi
        @test weak_pi_dimension(size_basis)==expected_weak
        @test length(size_pseudoket)==weak_pi_dimension(size_basis)
        @test size_pseudoket.basis===size_basis
    end
    @test sparse*rho0.data≈matrixfree*rho0.data atol=2e-11 rtol=2e-11
    @test check_generator(model).trace_preservation_error<2e-10
    @test abs(trace(PIState(basis,sparse*rho0.data)))<2e-11

    # sum_(i<j) X_i X_j differs from Jx^2/2 only by the identity term N/2,
    # which drops out of the Hamiltonian commutator.  This independently fixes
    # the unordered-pair normalization of the Appendix-D term.
    Jx=collective_operator(basis,operators.x_site)
    direct_pair=(-(Jpair/2))*(Jx*Jx)
    direct_model=PIModel(basis,(
        model.terms[1],model.terms[2],DirectPIHamiltonian(direct_pair),
        model.terms[4]))
    @test Matrix(sparse)≈
          Matrix(liouvillian(direct_model;representation=:sparse)) atol=3e-11 rtol=3e-11

    z_model=all_to_all_xx_spin_local_pseudomode_model(
        basis,operators;Jpair,omega_c,gamma,kappa,coupling=:z)
    @test z_model.terms[2].operator==operators.exchange_z
    @test check_liouvillian_symmetry(
        z_model,operators.z_site;basis=basis).symmetric
    mode_parity=Diagonal(ComplexF64[1,-1])
    combined_spin_mode_parity=kron(
        operators.spin_paulis[2],mode_parity)
    combined_report=check_liouvillian_symmetry(
        z_model,combined_spin_mode_parity;basis=basis)
    @test combined_report.symmetric
    @test combined_report.relative_residual<2e-10
    strong_restriction=diagonal_symmetry_restriction(
        basis,operators.z_site;charge=1,
        label=:longitudinal_spin_parity)
    z_sparse=liouvillian(z_model;representation=:sparse)
    z_restricted=RestrictedLiouvillian(z_sparse,strong_restriction)
    strong_indices=retained_indices(strong_restriction)
    @test length(strong_restriction)<length(basis)
    @test z_restricted.backend===:compressed
    @test z_restricted.certificate.invariant
    @test z_restricted.certificate.validation===:exhaustive_matrix_scan
    strong_probe=ComplexF64.(1:length(strong_restriction))
    @test z_restricted*strong_probe≈
          z_sparse[strong_indices,strong_indices]*strong_probe atol=3e-11 rtol=3e-11
    @test_throws ArgumentError local_pseudomode_operators(0)
    @test_throws ArgumentError local_pseudomode_operators(typemax(Int))
    @test_throws ArgumentError local_pseudomode_operators(
        big(typemax(Int))+1)
    @test_throws ArgumentError all_to_all_xx_spin_local_pseudomode_model(
        1,1;Jpair)
    @test_throws ArgumentError all_to_all_xx_spin_local_pseudomode_model(
        2,1;Jpair,gamma=-gamma)
    @test_throws ArgumentError all_to_all_xx_spin_local_pseudomode_model(
        2,1;Jpair,kappa=0)
    @test_throws ArgumentError all_to_all_xx_spin_local_pseudomode_model(
        basis,operators;Jpair,coupling=:raising)
    @test_throws DimensionMismatch all_to_all_xx_spin_local_pseudomode_model(
        PIBasis(2,3),operators;Jpair)
    @test_throws ArgumentError all_to_all_xx_spin_local_pseudomode_model(
        PIBasis(2,4;sectors=[(2,0,0,0)]),operators;Jpair)
end

@testset "dissipative time-crystal literature models" begin
    # Nakanishi--Sasamoto Eq. (14) is an exact finite-N oracle. Match the
    # spectra as multisets because the dense eigensolver may permute exact
    # degeneracies arbitrarily.
    N=5;g=1.3;kappa=0.4
    model=balanced_gain_loss_time_crystal_model(N;g=g,kappa=kappa,p=0)
    numerical=collect(eigvals(Matrix(liouvillian(model;representation=:sparse))))
    remaining=collect(balanced_gain_loss_spectrum(N;g=g,kappa=kappa))
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
    interacting=interacting_boundary_time_crystal_model(N;omega_z=omega_z,
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
        m=dissipative_collective_spin_pairing_model(3,d;V=1,gammaI=.4,gammaC=.2,dissipator=dissipator)
        @test check_generator(m).trace_preservation_error < 2e-10
        L=liouvillian(m;representation=:sparse)
        @test size(L)==(commutant_dimension(3,d),commutant_dimension(3,d))

        # The body-resolved lowering must retain the factor of two in the
        # unordered-pair term and exactly reproduce V*(Jx^2-Jy^2)/(N*j).
        s=spin_matrices(d)
        Jx=collective_operator(m.basis,s.jx)
        Jy=collective_operator(m.basis,s.jy)
        H=(Jx*Jx-Jy*Jy)*(1/(3*s.j))
        direct=PIModel(m.basis,(DirectPIHamiltonian(H),m.terms[end-1:end]...))
        @test Matrix(L)≈Matrix(liouvillian(direct;representation=:sparse)) atol=3e-12 rtol=3e-12
    end


    # For qubits the thermodynamic lowering reproduces the article's Bloch
    # equations (10a-c), not the special gammaI=0 family in Eq. (11).
    N=8;V=1.0;gammaI=0.7;gammaC=0.2;s=spin_matrices(2)
    meanfield_model=dissipative_collective_spin_pairing_model(
        N,2;V=V,gammaI=gammaI,gammaC=gammaC)
    plan=MeanFieldPlan(meanfield_model;limit=:thermodynamic)
    X=0.2;Y=-0.3;Z=-0.4
    sigma=Matrix{ComplexF64}(I,2,2)/2+X*s.jx+Y*s.jy+Z*s.jz
    derivative=meanfield_rhs(plan,sigma)
    numerical=(real(meanfield_expectation(derivative,s.jx))/s.j,
               real(meanfield_expectation(derivative,s.jy))/s.j,
               real(meanfield_expectation(derivative,s.jz))/s.j)
    article=(-2V*Y*Z-gammaI*X+gammaC*X*Z,
             -2V*X*Z-gammaI*Y+gammaC*Y*Z,
             4V*X*Y-2gammaI*(Z+1)-gammaC*(X^2+Y^2))
    @test collect(numerical)≈collect(article) atol=3e-12 rtol=3e-12

    # At V=0, decay drives the exactly polarized |-j> tensor power to itself.
    m=dissipative_collective_spin_pairing_model(4,2;V=0,gammaI=.7,gammaC=.3)
    ground=iid_pure_state(m.basis,ComplexF64[1,0])
    @test norm(liouvillian(m;representation=:sparse)*ground.data)<2e-12

    # The N=1 edge case has no two-particle Hamiltonian term, but retains a
    # valid microscopic model and therefore remains mean-field compatible.
    single=dissipative_collective_spin_pairing_model(1,2;V=.6,gammaI=.4,gammaC=.1)
    @test length(single.terms)==3
    @test MeanFieldPlan(single;limit=:finite) isa MeanFieldPlan
    @test_throws ArgumentError dissipative_collective_spin_pairing_model(0,2)
    @test_throws ArgumentError dissipative_collective_spin_pairing_model(2,1)
end
