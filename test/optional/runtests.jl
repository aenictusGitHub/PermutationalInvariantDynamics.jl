using Test
using LinearAlgebra
using PermutationalInvariantDynamics
using Makie
using QuantumCumulants
using JLD2
using HDF5

@testset "optional package extensions" begin
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsMakieExt)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsQuantumCumulantsExt)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsJLD2Ext)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsHDF5Ext)!==nothing

    spectrum=liouvillian_spectrum_data(ComplexF64[0,-1+0.25im])
    x,y=Makie.convert_arguments(Makie.Scatter,spectrum)
    @test collect(x)==[0.0,-1.0]
    @test collect(y)==[0.0,0.25]

    report=convergence_study(h->1+h^2,[0.4,0.2,0.1];
        consecutive=1,atol=0.1,rtol=0)
    refinements,errors=Makie.convert_arguments(Makie.Lines,report)
    @test collect(refinements)==[0.2,0.1]
    @test all(isfinite,errors)

    basis=PIBasis(1,3)
    rho=computational_product_state(basis,3)
    data=qudit_husimi_q(rho,zeros(ComplexF64,3,3);
        representation=:generator)
    points,values=Makie.convert_arguments(Makie.Lines,data)
    @test collect(points)==[1]
    @test collect(values)==data.values

    checkpoint_basis=PIBasis(2,2)
    checkpoint_state=iid_pure_state(
        checkpoint_basis,ComplexF32[1,im]/sqrt(2.0f0))
    checkpoint=PIStateCheckpoint(checkpoint_state;time=0.25f0,
        metadata=Dict("source"=>"optional-extension smoke"))
    mktempdir() do directory
        for (format,extension) in ((:jld2,"jld2"),(:hdf5,"h5"))
            path=joinpath(directory,"state.$extension")
            @test save_checkpoint(path,checkpoint;format)==path
            loaded=load_checkpoint(path;format)
            @test loaded.schema_version==PI_CHECKPOINT_VERSION
            @test loaded.state.basis.N==checkpoint_basis.N
            @test loaded.state.basis.d==checkpoint_basis.d
            @test loaded.state.basis.sectors==checkpoint_basis.sectors
            @test loaded.state.data==checkpoint_state.data
            @test loaded.time===0.25f0
            @test loaded.metadata==checkpoint.metadata
        end
    end
end

# Reuse the package's exact normalization regressions after loading the weak
# dependency, which activates the conditional extension testset at its end.
include(joinpath(@__DIR__,"..","test_cumulants.jl"))
