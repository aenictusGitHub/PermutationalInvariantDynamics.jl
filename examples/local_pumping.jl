using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

N=10; down=1.0; up=0.3; model=local_pump_decay_model(N;down=down,up=up)
prepared=compile(model)
result=stationary_state(prepared;return_info=true)
numeric=result.state
exact=local_pump_decay_steady_state(model.basis;down=down,up=up)
@assert result.info.converged
@assert diagnostics(numeric).valid
state_error=norm(numeric.data-exact.data)

# Retain multiplicity-weighted sector populations and the compressed physical
# Schur-block spectra.  Neither diagnostic expands a 2^N density operator.
numeric_structure=schur_block_structure(numeric;metric=:population)
exact_structure=schur_block_structure(exact;metric=:population)
numeric_sector_populations=diag(numeric_structure.weights)
exact_sector_populations=diag(exact_structure.weights)
sector_spins=[(partition.parts[1]-partition.parts[2])/2
              for partition in model.basis.sectors]
sector_order=sortperm(sector_spins)
numeric_spectrum=pi_density_spectrum(numeric)
exact_spectrum=pi_density_spectrum(exact)

println("Local pump/emission steady-state error = ",state_error)

if makie_available()
    M=makie_module()
    figure=M.Figure(size=(1120,440),fontsize=17)
    sector_axis=M.Axis(
        figure[1,1];xlabel="total spin j",ylabel="sector population",
        title="Multiplicity-weighted Schur sectors")
    spectrum_axis=M.Axis(
        figure[1,2];xlabel="compressed Schur eigenvalue index",
        ylabel="physical-block eigenvalue",yscale=log10,
        title="Thermal steady-state spectrum")

    ordered_spins=sector_spins[sector_order]
    M.lines!(sector_axis,ordered_spins,
             exact_sector_populations[sector_order];
             color=:black,linewidth=2.7,label="exact product state")
    M.scatter!(sector_axis,ordered_spins,
               numeric_sector_populations[sector_order];
               color=:royalblue,markersize=10,label="PI steady state")
    M.axislegend(sector_axis;position=:lt,labelsize=13)

    spectrum_indices=collect(eachindex(exact_spectrum.values))
    M.lines!(spectrum_axis,spectrum_indices,exact_spectrum.values;
             color=:black,linewidth=2.7,label="exact product state")
    M.scatter!(spectrum_axis,spectrum_indices,numeric_spectrum.values;
               color=:firebrick,markersize=7,label="PI steady state")
    M.axislegend(spectrum_axis;position=:rt,labelsize=13)
    save_example_figure(figure, "local_pumping")
end
