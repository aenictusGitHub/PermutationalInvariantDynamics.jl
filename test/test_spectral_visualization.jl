@testset "Liouvillian and Floquet spectrum visualization" begin
    # Compare spectra as multisets without relying on the ordering chosen by
    # dense eigensolvers for exactly degenerate roots.
    ordered(values)=sort(collect(values);by=z->(real(z),imag(z)))

    sm=ComplexF64[0 1;0 0]
    gamma=0.8
    period=0.9
    basis=PIBasis(1,2)
    model=PIModel(basis,[LocalJump(sm;rate=gamma)])
    sparse_L=liouvillian(model;representation=:sparse)
    matrixfree_L=liouvillian(model;representation=:matrixfree)
    compiled_sparse=compile(model;backend=:sparse)
    compiled_matrixfree=compile(model;backend=:matrixfree)

    # A one-qubit amplitude-damping generator has a closed-form spectrum.
    expected_L=ComplexF64[0,-gamma,-gamma/2,-gamma/2]
    sparse_data=liouvillian_spectrum_data(sparse_L;algorithm=:dense)
    matrixfree_data=liouvillian_spectrum_data(matrixfree_L;algorithm=:dense)
    model_data=liouvillian_spectrum_data(model;algorithm=:dense)
    compiled_sparse_data=liouvillian_spectrum_data(
        compiled_sparse;algorithm=:dense)
    compiled_matrixfree_data=liouvillian_spectrum_data(
        compiled_matrixfree;algorithm=:dense)

    @test sparse_data isa ComplexSpectrum
    @test sparse_data.kind===:liouvillian
    @test sparse_data.representation===:eigenvalues
    @test length(sparse_data.values)==4
    @test ordered(sparse_data.values)≈ordered(expected_L) atol=2e-13
    @test ordered(matrixfree_data.values)≈ordered(sparse_data.values) atol=2e-13
    @test ordered(model_data.values)≈ordered(sparse_data.values) atol=2e-13
    @test ordered(compiled_sparse_data.values)≈ordered(sparse_data.values) atol=2e-13
    @test ordered(compiled_matrixfree_data.values)≈ordered(sparse_data.values) atol=2e-13
    @test count(==(:stationary),sparse_data.classifications)==1
    @test count(==(:decaying),sparse_data.classifications)==3
    @test !any(==(:unstable),sparse_data.classifications)

    # Solver tolerances and display-classification tolerances are separate for
    # computed sources, so a Krylov tolerance is never consumed by styling.
    separated_tolerances=liouvillian_spectrum_data(
        model;algorithm=:dense,atol=1e-12,rtol=1e-10,
        classification_atol=1e-7,classification_rtol=0)
    @test separated_tolerances.tolerance==1e-7

    # Classification uses the complex-plane stability boundary and must not
    # discard oscillatory, peripheral, or unstable values.
    classified_values=ComplexF64[0,-0.2+0.7im,-0.2-0.7im,
                                 0+0.4im,0.1-0.3im]
    classified=liouvillian_spectrum_data(
        classified_values;atol=1e-12,rtol=0)
    @test classified.values==classified_values
    @test classified.classifications==
          [:stationary,:decaying,:decaying,:peripheral,:unstable]

    # High-level and lower-level partial results can be displayed directly.
    # Their solver diagnostics remain metadata; constructing a visualization
    # from them must not rerun an eigensolver.
    partial_result=liouvillian_spectrum(
        model;target=:largest_real,nev=2,algorithm=:dense,return_info=true)
    partial_data=liouvillian_spectrum_data(partial_result)
    @test partial_data.values==partial_result.values
    @test partial_data.metadata.partial
    @test partial_data.metadata.dimension==length(basis)

    named_result=(values=ComplexF64[0,-0.2+0.1im],
                  residuals=[0.0,2e-13],converged=Bool[true,true],dimension=4)
    named_data=liouvillian_spectrum_data(named_result)
    @test named_data.values==named_result.values
    @test named_data.metadata.partial
    @test named_data.metadata.dimension==4
    @test named_data.metadata.residuals==named_result.residuals
    @test named_data.metadata.converged==named_result.converged

    precise_residuals=BigFloat[big"1e-40",big"2e-40"]
    residual_precision=liouvillian_spectrum_data(
        ComplexF32[0,-1];residuals=precise_residuals)
    @test eltype(residual_precision.residuals)===BigFloat
    @test residual_precision.residuals==precise_residuals

    # Count applications of an otherwise opaque matrix-free source.  Exact
    # extraction may apply it, but rendering and saving already extracted data
    # must be purely presentational.
    calls=Ref(0)
    dense_matrix=Matrix(sparse_L)
    counted_action! = (y,x,t,p)->begin
        calls[]+=1
        mul!(y,dense_matrix,x)
    end
    counted=MatrixFreeLiouvillian(
        size(dense_matrix,1),counted_action!,ComplexF64,
        zeros(ComplexF64,size(dense_matrix,1)))
    counted_data=liouvillian_spectrum_data(counted;algorithm=:dense)
    calls_after_extraction=calls[]
    @test calls_after_extraction>0

    # Invalid presentation/conversion keywords fail before an expensive
    # matrix-free solve or one-period propagation starts.
    @test_throws ArgumentError liouvillian_spectrum_data(
        counted;algorithm=:dense,classification_atol=-1)
    @test calls[]==calls_after_extraction
    @test_throws ArgumentError floquet_spectrum_data(
        counted,period;steps=1,representation=:bad)
    @test calls[]==calls_after_extraction
    @test_throws ArgumentError floquet_spectrum_data(
        counted,period;steps=1,atol=-1)
    @test calls[]==calls_after_extraction
    @test_throws ArgumentError visualize_liouvillian_spectrum(
        counted;algorithm=:dense,width=319)
    @test calls[]==calls_after_extraction
    @test_throws ArgumentError visualize_floquet_spectrum(
        counted,period;steps=1,marker_size=0)
    @test calls[]==calls_after_extraction

    counted_figure=visualize_spectrum(counted_data)
    sprint(show,MIME("image/svg+xml"),counted_figure)
    sprint(show,MIME("image/svg+xml"),counted_figure)
    @test calls[]==calls_after_extraction

    # The exact one-period amplitude-damping map provides an analytical
    # multiplier oracle independent of the Floquet integrator.
    propagator=exp(period*Matrix(sparse_L))
    expected_multipliers=ComplexF64[
        1,exp(-gamma*period),exp(-gamma*period/2),exp(-gamma*period/2)]
    multiplier_data=floquet_spectrum_data(
        propagator;period=period,representation=:multipliers)
    positional_multiplier_data=floquet_spectrum_data(
        propagator,period;representation=:multipliers)
    @test multiplier_data isa ComplexSpectrum
    @test multiplier_data.kind===:floquet
    @test multiplier_data.representation===:multipliers
    @test multiplier_data.period==period
    @test ordered(multiplier_data.values)≈ordered(expected_multipliers) atol=2e-13
    @test ordered(positional_multiplier_data.values)≈
          ordered(multiplier_data.values) atol=2e-13
    @test count(==(:fixed),multiplier_data.classifications)==1
    @test count(==(:contracting),multiplier_data.classifications)==3
    @test !any(==(:unstable),multiplier_data.classifications)

    raw_multiplier_data=floquet_spectrum_data(
        expected_multipliers;period=period,input=:multipliers,
        representation=:multipliers)
    @test raw_multiplier_data.values==expected_multipliers
    @test raw_multiplier_data.classifications==
          [:fixed,:contracting,:contracting,:contracting]

    propagated_data=floquet_spectrum_data(
        model,period;steps=160,representation=:multipliers)
    @test ordered(propagated_data.values)≈ordered(expected_multipliers) atol=2e-10

    # Principal-branch exponent conversion is explicit about both the input
    # representation and the positive period.
    theta=3pi/4
    unitary_multipliers=ComplexF64[1,cis(theta),cis(-theta)]
    exponent_data=floquet_spectrum_data(
        unitary_multipliers;period=period,input=:multipliers,
        representation=:exponents,atol=1e-12,rtol=0)
    @test exponent_data.representation===:exponents
    @test exponent_data.values≈log.(complex.(unitary_multipliers))./period
    @test exponent_data.classifications==[:fixed,:peripheral,:peripheral]
    @test exponent_data.metadata.branch===:principal

    supplied_unfolded=floquet_spectrum_data(
        ComplexF64[10im];period=1.0,input=:exponents,
        representation=:exponents)
    @test supplied_unfolded.values==ComplexF64[10im]
    @test supplied_unfolded.metadata.branch===nothing
    @test visualize_spectrum(supplied_unfolded).title=="Floquet exponents"

    supplied_exponents=ComplexF64[0,-0.2+0.4im,0+0.3im,0.1]
    converted_multipliers=floquet_spectrum_data(
        supplied_exponents;period=period,input=:exponents,
        representation=:multipliers,atol=1e-12,rtol=0)
    @test converted_multipliers.values≈exp.(period.*supplied_exponents)
    @test converted_multipliers.classifications==
          [:fixed,:contracting,:peripheral,:unstable]

    multiplier_residuals=BigFloat[big"1e-30",big"2e-30"]
    converted_with_residuals=floquet_spectrum_data(
        ComplexF64[1,0.5];period=period,input=:multipliers,
        representation=:exponents,residuals=multiplier_residuals)
    @test converted_with_residuals.residuals==multiplier_residuals
    @test converted_with_residuals.metadata.residual_representation===:multipliers

    # BigFloat values remain BigFloat in the reusable numerical result.  SVG
    # conversion is allowed only at the final presentation layer.
    big_values=Complex{BigFloat}[
        complex(big"0",big"0"),complex(big"-0.125",big"0.25")]
    big_data=liouvillian_spectrum_data(big_values)
    @test eltype(big_data.values)===Complex{BigFloat}
    @test big_data.classifications==[:stationary,:decaying]
    big_svg=sprint(show,MIME("image/svg+xml"),visualize_spectrum(big_data))
    @test occursin("</svg>",big_svg)
    @test !occursin(r"(?i)(nan|inf)",big_svg)

    title="Spectrum <modes> & \"branches\""
    figure=visualize_spectrum(
        classified;title=title,show_indices=true,width=680,height=500)
    @test figure isa SpectrumVisualization
    @test figure.spectrum===classified
    svg=sprint(show,MIME("image/svg+xml"),figure)
    @test startswith(lstrip(svg),"<svg")
    @test occursin("</svg>",svg)
    @test occursin("<circle",svg)
    @test occursin("Re",svg)
    @test occursin("Im",svg)
    @test occursin("Spectrum &lt;modes&gt; &amp; &quot;branches&quot;",svg)
    @test !occursin(title,svg)
    @test !occursin(r"(?i)(nan|inf)",svg)
    @test occursin("ComplexSpectrum",sprint(show,classified))
    @test occursin("SpectrumVisualization",sprint(show,figure))
    @test occursin("partial",lowercase(sprint(show,MIME("text/plain"),
                                              partial_data)))

    # Mutable public diagnostics are revalidated at render time; malformed
    # classes cannot be interpolated into an SVG attribute.
    tampered=liouvillian_spectrum_data(ComplexF64[0,-1])
    tampered_figure=visualize_spectrum(tampered)
    tampered.classifications[1]=Symbol("x\" onmouseover=\"alert(1)")
    @test_throws ArgumentError sprint(
        show,MIME("image/svg+xml"),tampered_figure)

    floquet_figure=visualize_spectrum(
        multiplier_data;title="Floquet multipliers",show_indices=true)
    floquet_svg=sprint(show,MIME("image/svg+xml"),floquet_figure)
    @test occursin("unit circle",lowercase(floquet_svg))
    @test !occursin(r"(?i)(nan|inf)",floquet_svg)

    focused_figure=visualize_spectrum(
        classified;xlimits=(-0.05,0.05),ylimits=(-0.1,0.1))
    focused_svg=sprint(show,MIME("image/svg+xml"),focused_figure)
    @test occursin("outside viewport",focused_svg)
    @test focused_figure.spectrum.values==classified_values

    extreme_data=liouvillian_spectrum_data(
        ComplexF64[-floatmax(Float64),floatmax(Float64)])
    extreme_svg=sprint(show,MIME("image/svg+xml"),visualize_spectrum(
        extreme_data;xlimits=(-1.0,1.0),ylimits=(-1.0,1.0)))
    @test occursin("2 outside viewport",extreme_svg)
    @test !occursin(r"(?i)(nan|inf)",extreme_svg)
    @test visualize_liouvillian_spectrum(sparse_data) isa SpectrumVisualization
    @test visualize_liouvillian_spectrum(model;algorithm=:dense) isa
          SpectrumVisualization
    @test visualize_floquet_spectrum(multiplier_data) isa SpectrumVisualization
    @test visualize_floquet_spectrum(propagator;period=period) isa
          SpectrumVisualization
    @test visualize_floquet_spectrum(propagator,period) isa
          SpectrumVisualization
    @test (@doc visualize_liouvillian_spectrum)!==nothing
    @test (@doc visualize_floquet_spectrum)!==nothing

    invalid_xml=visualize_spectrum(classified;title="bad\u0001title")
    @test_throws ArgumentError sprint(
        show,MIME("image/svg+xml"),invalid_xml)

    mktempdir() do directory
        path=joinpath(directory,"liouvillian-spectrum.svg")
        returned=save_spectrum_visualization(path,figure)
        @test returned==path
        @test isfile(path)
        @test read(path,String)==svg
        counted_path=joinpath(directory,"counted-spectrum.svg")
        save_spectrum_visualization(counted_path,counted_figure)
        @test isfile(counted_path)
        @test calls[]==calls_after_extraction
    end

    # Invalid numerical data and ambiguous conversions are rejected rather
    # than silently dropping points or inventing a period.
    @test_throws ArgumentError liouvillian_spectrum_data(ComplexF64[])
    @test_throws ArgumentError liouvillian_spectrum_data(ComplexF64[NaN])
    @test_throws ArgumentError liouvillian_spectrum_data(ComplexF64[Inf*im])
    @test_throws DimensionMismatch liouvillian_spectrum_data(zeros(ComplexF64,2,3))
    @test_throws ArgumentError liouvillian_spectrum_data(classified_values;atol=-1)
    @test_throws ArgumentError liouvillian_spectrum_data(classified_values;rtol=-1)
    @test_throws ArgumentError liouvillian_spectrum_data(
        sparse_L;algorithm=:dense,spectrum_kwargs=(nev=1,))
    @test_throws ArgumentError liouvillian_spectrum_data(
        sparse_L;algorithm=:dense,vectors=true)
    @test_throws ArgumentError liouvillian_spectrum_data(
        sparse_L;algorithm=:dense,atol=-1)
    @test_throws ArgumentError liouvillian_spectrum_data(
        sparse_L;algorithm=:dense,rtol=-1)
    @test_throws ArgumentError liouvillian_spectrum_data(
        sparse_L;algorithm=:dense,rtol=nothing)

    driven=PIModel(basis,[LocalJump(sm;rate=(t,p)->1+t)])
    @test_throws ArgumentError liouvillian_spectrum_data(driven;algorithm=:dense)

    @test_throws ArgumentError floquet_spectrum_data(
        expected_multipliers;period=0,input=:multipliers)
    @test_throws ArgumentError floquet_spectrum_data(
        expected_multipliers;period=-1,input=:multipliers)
    @test_throws ArgumentError floquet_spectrum_data(
        expected_multipliers;period=Inf,input=:multipliers)
    @test_throws ArgumentError floquet_spectrum_data(
        expected_multipliers;period=NaN,input=:multipliers)
    @test_throws ArgumentError floquet_spectrum_data(
        expected_multipliers;input=:multipliers,representation=:exponents)
    @test_throws ArgumentError floquet_spectrum_data(
        ComplexF64[1,0];period=period,input=:multipliers,
        representation=:exponents)
    @test_throws ArgumentError floquet_spectrum_data(
        expected_multipliers;period=period,input=:bad)
    @test_throws ArgumentError floquet_spectrum_data(
        expected_multipliers;period=period,input=:multipliers,
        representation=:bad)
    @test_throws DimensionMismatch floquet_spectrum_data(
        zeros(ComplexF64,2,3);period=period)

    @test_throws ArgumentError visualize_spectrum(classified;width=319)
    @test_throws ArgumentError visualize_spectrum(classified;height=279)
    @test_throws ArgumentError visualize_spectrum(classified;marker_size=0)
    @test_throws ArgumentError visualize_spectrum(classified;marker_size=NaN)
    @test_throws ArgumentError visualize_spectrum(classified;xlimits=(1,1))
    @test_throws ArgumentError visualize_spectrum(classified;ylimits=(0,Inf))
end

@testset "Multiplicity-compressed density-spectrum visualization" begin
    # Infinite-temperature Gibbs data provide an exact spectrum while still
    # exercising the public thermal-state constructor.
    basis=PIBasis(3,2)
    sz=ComplexF64[1 0;0 -1]
    thermal=thermal_state(collective_operator(basis,sz),0.0)
    thermal_data=pi_density_spectrum(thermal;sortby=:none)
    thermal_figure=visualize_density_spectrum(
        thermal;spectrum_kwargs=(sortby=:none,),show_indices=true)
    @test thermal_figure isa DensitySpectrumVisualization
    @test thermal_figure.spectrum.values==thermal_data.values
    @test thermal_figure.spectrum.total_dimension==big(8)
    @test all(==(1/8),thermal_figure.spectrum.values)
    thermal_svg=sprint(show,MIME("image/svg+xml"),thermal_figure)
    @test occursin("multiplicity-compressed Schur eigenmodes",thermal_svg)
    @test occursin("compressed eigenmode rank",thermal_svg)
    @test occursin("raw eigenvalue",thermal_svg)
    @test occursin("zero-reference",thermal_svg)
    @test occursin("retained Hilbert dimension: 8",thermal_svg)
    @test occursin("exact degeneracy 2",thermal_svg)
    @test occursin("×2",thermal_svg)
    @test occursin("sector index",thermal_svg)
    @test occursin("sector-1",thermal_svg)
    @test occursin("sector-2",thermal_svg)
    @test !occursin(r"(?i)(nan|inf)",thermal_svg)

    # A restricted basis reports the dimension of the retained Hilbert space,
    # not d^N and not the number of compressed eigenmodes.
    restricted_basis=PIBasis(4,2;sectors=[(4,0)])
    restricted=maximally_mixed_state(restricted_basis)
    restricted_data=pi_density_spectrum(restricted;sortby=:none)
    restricted_figure=visualize_density_spectrum(restricted_data)
    restricted_svg=sprint(show,MIME("image/svg+xml"),restricted_figure)
    @test restricted_figure.spectrum===restricted_data
    @test restricted_figure.spectrum.total_dimension isa BigInt
    @test restricted_figure.spectrum.total_dimension==5
    @test occursin("5 compressed modes",restricted_svg)
    @test occursin("retained Hilbert dimension: 5",restricted_svg)
    @test !occursin("retained Hilbert dimension: 16",restricted_svg)

    # Precomputed values remain in their original precision and order. Tiny
    # negative values are preserved even when they lie inside the presentation
    # tolerance; a more negative value receives a red outline but is not
    # clipped to zero.
    symmetric=Partition((3,0))
    precise_values=BigFloat[big"0",-big"1e-40",-big"1e-5",big"0.75"]
    precise_data=(values=precise_values,
                  degeneracies=BigInt[1,1,1,1],
                  sectors=Partition[symmetric,symmetric,symmetric,symmetric],
                  sector_indices=[1,2,3,4],total_dimension=big(4))
    precise_figure=visualize_density_spectrum(
        precise_data;presentation_atol=big"1e-30",presentation_rtol=0,
        show_indices=true)
    @test precise_figure.spectrum===precise_data
    @test precise_figure.spectrum.values===precise_values
    @test eltype(precise_figure.spectrum.values)===BigFloat
    @test precise_figure.presentation_tolerance==big"1e-30"
    @test precise_figure.spectrum.values==precise_values
    precise_svg=sprint(show,MIME("image/svg+xml"),precise_figure)
    @test occursin("eigenvalue $(precise_values[2])",precise_svg)
    @test occursin("eigenvalue $(precise_values[3])",precise_svg)
    @test count(occursin(" negative",match.match) for match in
                eachmatch(r"<circle class=\"density-spectrum-point[^\"]*\"",
                          precise_svg))==1
    @test occursin("red outline",precise_svg)
    @test occursin("zero eigenvalue reference",precise_svg)

    repeated_values=Float32[0.25,0.25,-0.125]
    repeated_sector=Partition((2,0))
    repeated=(values=repeated_values,degeneracies=BigInt[1,1,1],
              sectors=Partition[repeated_sector,repeated_sector,repeated_sector],
              sector_indices=[2,1,3],total_dimension=big(3))
    repeated_figure=visualize_density_spectrum(repeated;presentation_rtol=0)
    @test repeated_figure.spectrum===repeated
    @test repeated_figure.spectrum.values===repeated_values
    @test repeated_figure.spectrum.values==Float32[0.25,0.25,-0.125]
    repeated_svg=sprint(show,MIME("image/svg+xml"),repeated_figure)
    first_rank=findfirst("compressed mode 1: eigenvalue 0.25",repeated_svg)
    second_rank=findfirst("compressed mode 2: eigenvalue 0.25",repeated_svg)
    third_rank=findfirst("compressed mode 3: eigenvalue -0.125",repeated_svg)
    @test first_rank!==nothing && second_rank!==nothing && third_rank!==nothing
    @test first(first_rank)<first(second_rank)<first(third_rank)

    escaped_title="Density <rho> & \"Schur\""
    escaped=visualize_density_spectrum(repeated;title=escaped_title)
    escaped_svg=sprint(show,MIME("image/svg+xml"),escaped)
    @test occursin("Density &lt;rho&gt; &amp; &quot;Schur&quot;",escaped_svg)
    @test !occursin(escaped_title,escaped_svg)
    @test occursin("DensitySpectrumVisualization",sprint(show,escaped))
    @test occursin("multiplicities: compressed",
                   sprint(show,MIME("text/plain"),escaped))
    @test (@doc visualize_density_spectrum)!==nothing
    @test (@doc save_density_spectrum_visualization)!==nothing

    mktempdir() do directory
        path=joinpath(directory,"density-spectrum.svg")
        returned=save_density_spectrum_visualization(path,precise_figure)
        @test returned==path
        @test isfile(path)
        @test read(path,String)==precise_svg

        data_path=joinpath(directory,"density-spectrum-data.svg")
        @test save_density_spectrum_visualization(data_path,repeated)==data_path
        @test isfile(data_path)

        state_path=joinpath(directory,"density-spectrum-state.svg")
        @test save_density_spectrum_visualization(
            state_path,restricted;spectrum_kwargs=(sortby=:none,))==state_path
        @test isfile(state_path)
    end

    valid=(values=Float64[0.5,0.25,0.0],
           degeneracies=BigInt[1,1,1],
           sectors=Partition[repeated_sector,repeated_sector,repeated_sector],
           sector_indices=[1,2,3],total_dimension=big(3))
    @test_throws ArgumentError visualize_density_spectrum((values=[1.0],))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(values=(0.5,0.25,0.0),)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(values=ComplexF64[0.5,0.25,0],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(values=Float64[0.5,NaN,0],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(values=Float64[0.5,Inf,0],)))
    @test_throws DimensionMismatch visualize_density_spectrum(
        merge(valid,(degeneracies=BigInt[1,1],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(degeneracies=Bool[true,true,true],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(degeneracies=BigInt[1,0,1],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(degeneracies=BigInt[2,1,1],total_dimension=big(4))))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(sectors=[1,2,3],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(sector_indices=Bool[true,true,true],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(sector_indices=[0,2,3],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(sector_indices=[1,1,3],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(sector_indices=[1,2,4],)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(total_dimension=3,)))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(total_dimension=big(4),)))
    wrong_weight=Partition((3,0))
    @test_throws ArgumentError visualize_density_spectrum(
        merge(valid,(sectors=Partition[repeated_sector,repeated_sector,wrong_weight],)))

    @test_throws ArgumentError visualize_density_spectrum(valid;width=319)
    @test_throws ArgumentError visualize_density_spectrum(valid;height=279)
    @test_throws ArgumentError visualize_density_spectrum(valid;marker_size=0)
    @test_throws ArgumentError visualize_density_spectrum(valid;scale=:log)
    @test_throws ArgumentError visualize_density_spectrum(
        valid;show_degeneracies=:yes)
    @test_throws ArgumentError visualize_density_spectrum(valid;ylimits=(1,1))
    @test_throws ArgumentError visualize_density_spectrum(valid;ylimits=(0,Inf))
    @test_throws ArgumentError visualize_density_spectrum(
        valid;presentation_atol=-1)
    @test_throws ArgumentError visualize_density_spectrum(
        valid;presentation_rtol=-1)
    @test_throws ArgumentError visualize_density_spectrum(
        valid;spectrum_kwargs=(sortby=:none,))
    @test_throws ArgumentError visualize_density_spectrum(
        valid;title="forbidden\u0001title")

    # All source-side options that can be checked without the state are
    # rejected before the expensive block eigendecompositions are attempted.
    nonhermitian=PIState(basis)
    coefficient_block(nonhermitian,basis.sectors[1])[1,2]=1
    width_error=try
        visualize_density_spectrum(nonhermitian;width=319)
        nothing
    catch error
        error
    end
    @test width_error isa ArgumentError
    @test occursin("width",sprint(showerror,width_error))
    sort_error=try
        visualize_density_spectrum(
            nonhermitian;spectrum_kwargs=(sortby=:not_a_sort,))
        nothing
    catch error
        error
    end
    @test sort_error isa ArgumentError
    @test occursin("sortby",sprint(showerror,sort_error))
    @test_throws ArgumentError visualize_density_spectrum(
        nonhermitian;spectrum_kwargs=(expanded=true,))

    # Public vectors are intentionally reusable, so render-time validation is
    # repeated and catches post-construction corruption.
    tampered_values=copy(valid.values)
    tampered=merge(valid,(values=tampered_values,))
    tampered_figure=visualize_density_spectrum(tampered)
    tampered_values[2]=NaN
    @test_throws ArgumentError sprint(
        show,MIME("image/svg+xml"),tampered_figure)

    huge=BigFloat[big"1e10000",big"0",big"-1"]
    huge_data=(values=huge,degeneracies=BigInt[1,1,1],
               sectors=Partition[repeated_sector,repeated_sector,repeated_sector],
               sector_indices=[1,2,3],total_dimension=big(3))
    huge_figure=visualize_density_spectrum(huge_data)
    @test huge_figure.spectrum.values===huge
    @test_throws ArgumentError sprint(
        show,MIME("image/svg+xml"),huge_figure)
end
