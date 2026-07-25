using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

N=12; gamma=0.4; sx=ComplexF64[0 1;1 0]
model=independent_dephasing_model(N;gamma=gamma); b=model.basis
rho0=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
times=range(0,4;length=25)
prepared=compile(model;backend=:matrixfree)
solution=solve_dynamics(prepared,rho0,(first(times),last(times));
                        saveat=times,steps_per_interval=64)
Jx=CollectiveObservablePlan(b,sx/2)
numeric=[collective_expectation(rho,Jx) for rho in solution]
exact=[independent_dephasing_coherence_exact(N,t;gamma=gamma) for t in times]
@assert diagnostics(last(solution)).valid
@assert maximum(abs,imag.(numeric)) < 1e-10
errors=abs.(numeric.-exact)
maximum_error=maximum(errors)
println("Independent-dephasing coherence maximum error = ",maximum_error)

if makie_available()
    M=makie_module()
    scaled_times=gamma .* collect(times)
    figure=M.Figure(size=(1080,430),fontsize=17)
    signal_axis=M.Axis(
        figure[1,1];xlabel="γt",ylabel="⟨Jx⟩ / (N/2)",
        title="Coherence under local dephasing")
    error_axis=M.Axis(
        figure[1,2];xlabel="γt",ylabel="|⟨Jx⟩PI − ⟨Jx⟩exact|",
        title="Pointwise PI-space error")

    M.lines!(signal_axis,scaled_times,exact ./ (N/2);
             color=:black,linewidth=2.7,label="analytic exponential")
    M.scatter!(signal_axis,scaled_times,real.(numeric) ./ (N/2);
               color=:royalblue,markersize=8,label="PI dynamics")
    M.axislegend(signal_axis;position=:rt,labelsize=13)

    M.lines!(error_axis,scaled_times,errors;
             color=:firebrick,linewidth=2.2)
    M.scatter!(error_axis,scaled_times,errors;
               color=:firebrick,markersize=6)
    save_example_figure(figure, "independent_dephasing_coherence")
end
