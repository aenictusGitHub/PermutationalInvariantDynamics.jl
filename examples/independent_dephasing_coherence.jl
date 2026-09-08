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
@assert maximum_error < 1e-10
println("Independent-dephasing coherence maximum error = ",maximum_error)

if makie_available()
    M=makie_module()
    scaled_times=gamma .* collect(times)
    figure=example_figure(size=(1120,490))
    M.Label(figure[0,1:2], ExampleMakie.latex("Independent dephasing  •  \$N=$N,\\;\\gamma=$gamma\$");
            fontsize=22,font=:bold,halign=:left)
    signal_axis=M.Axis(
        figure[1,1];xlabel=ExampleMakie.latex(raw"$\gamma t$"),ylabel=ExampleMakie.latex(raw"$\langle J_x\rangle/(N/2)$"),
        title=ExampleMakie.latex("(a) Coherence decay"))
    error_axis=M.Axis(
        figure[1,2];xlabel=ExampleMakie.latex(raw"$\gamma t$"),ylabel=ExampleMakie.latex("absolute normalized coherence error"),
        title=ExampleMakie.latex("(b) Accuracy of the plotted observable"))

    M.lines!(signal_axis,scaled_times,exact ./ (N/2);
             color=:black,linewidth=2.7,label=ExampleMakie.latex("analytic exponential"))
    M.scatter!(signal_axis,scaled_times,real.(numeric) ./ (N/2);
               color=example_colors.blue,markersize=7,label=ExampleMakie.latex("PI dynamics"))
    M.axislegend(signal_axis;position=:rt,labelsize=13)

    M.lines!(error_axis,scaled_times,errors ./ (N/2);
             color=example_colors.red)
    M.scatter!(error_axis,scaled_times,errors ./ (N/2);
               color=example_colors.red,markersize=6)
    M.Label(figure[2,1:2], ExampleMakie.latex("RK4: 64 steps per interval  •  Linear error scale retains exact zeros; roundoff is visible.");
            fontsize=13,color=example_colors.gray)
    save_example_figure(figure, "independent_dephasing_coherence")
    save_example_data("independent_dephasing_coherence", (;
        time=collect(times), gamma_t=scaled_times,
        normalized_coherence=real.(numeric) ./ (N/2), exact_coherence=exact ./ (N/2),
        absolute_normalized_error=errors ./ (N/2));
        metadata=(; N,gamma,steps_per_interval=64))
end
