using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

N=8;chi=0.3;sx=ComplexF64[0 1;1 0]
model=one_axis_twisting_model(N;chi=chi);b=model.basis
rho0=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
times=range(0,1.5;length=31)
prepared=compile(model;backend=:matrixfree)
solution=solve_dynamics(prepared,rho0,(first(times),last(times));
                        saveat=times,steps_per_interval=64)
geometry=OneBodyGeometry(b)
Jx=CollectiveObservablePlan(b,sx/2;cache=geometry)
numeric=[collective_expectation(rho,Jx) for rho in solution]
exact=[one_axis_twisting_mean_spin_exact(N,t;chi=chi) for t in times]
@assert diagnostics(last(solution);atol=1e-10,rtol=1e-10).valid
@assert maximum(abs,imag.(numeric)) < 1e-10
errors=abs.(numeric.-exact)
@assert maximum(errors) < 1e-9
println("One-axis-twisting maximum |<Jx>-exact| = ",maximum(errors))

# Product-Schur data for the fixed one-particle marginal are also reusable.
one_body=ReductionPlan(b,1)
one_body_purities=[reduced_purity(rho,1;plan=one_body) for rho in solution]
exact_purities=(1 .+ (exact ./ (N/2)).^2) ./ 2
purity_errors=abs.(one_body_purities .- exact_purities)
@assert maximum(purity_errors) < 1e-9
println("final one-particle purity = ",last(one_body_purities))

if makie_available()
    M=makie_module()
    scaled_times=chi .* collect(times)
    figure=example_figure(size=(1200,500))
    M.Label(figure[0,1:3], ExampleMakie.latex("One-axis twisting  •  \$N=$N,\\;\\chi=$chi\$");
            fontsize=22,font=:bold,halign=:left)
    spin_axis=M.Axis(
        figure[1,1];xlabel=ExampleMakie.latex(raw"$\chi t$"),ylabel=ExampleMakie.latex(raw"$\langle J_x\rangle/(N/2)$"),
        title=ExampleMakie.latex("(a) Collective coherence"))
    purity_axis=M.Axis(
        figure[1,2];xlabel=ExampleMakie.latex(raw"$\chi t$"),ylabel=ExampleMakie.latex(raw"$\mathrm{tr}(\rho_1^2)$"),
        title=ExampleMakie.latex("(b) One-spin purity"))
    error_axis=M.Axis(figure[1,3];xlabel=ExampleMakie.latex(raw"$\chi t$"),ylabel=ExampleMakie.latex("absolute error"),
                      title=ExampleMakie.latex("(c) Analytic checks"))

    M.lines!(spin_axis,scaled_times,exact ./ (N/2);
             color=:black,linewidth=2.7,label=ExampleMakie.latex("Analytical formula"))
    M.scatter!(spin_axis,scaled_times,real.(numeric) ./ (N/2);
               color=example_colors.blue,markersize=7,label=ExampleMakie.latex("PI dynamics"))
    M.axislegend(spin_axis;position=:rt,labelsize=13)

    M.lines!(purity_axis,scaled_times,exact_purities;
             color=:black,label=ExampleMakie.latex("exact one-spin purity"))
    M.scatter!(purity_axis,scaled_times,one_body_purities;
               color=example_colors.orange,marker=:diamond,markersize=7,label=ExampleMakie.latex("reduced PI state"))
    M.hlines!(purity_axis,[1.0];color=:gray50,linestyle=:dash,
              label=ExampleMakie.latex("pure one-spin state"))
    M.hlines!(purity_axis,[0.5];color=:gray50,linestyle=:dot)
    M.ylims!(purity_axis,0.48,1.03)
    M.axislegend(purity_axis;position=:rb,labelsize=13)
    M.lines!(error_axis,scaled_times,errors ./ (N/2);
             color=example_colors.blue,label=ExampleMakie.latex("normalized mean spin"))
    M.lines!(error_axis,scaled_times,purity_errors;
             color=example_colors.orange,linestyle=:dash,label=ExampleMakie.latex("one-spin purity"))
    M.axislegend(error_axis;position=:lt,labelsize=12)
    M.Label(figure[2,1:3], ExampleMakie.latex("Unitary evolution: reduced purity decreases as spins become correlated.  •  RK4: 64 steps per interval.");
            fontsize=13,color=example_colors.gray)
    save_example_figure(figure, "one_axis_twisting")
    save_example_data("one_axis_twisting", (;
        time=collect(times), chi_t=scaled_times,
        normalized_spin=real.(numeric) ./ (N/2), exact_spin=exact ./ (N/2),
        spin_error=errors ./ (N/2), purity=one_body_purities,
        exact_purity=exact_purities, purity_error=purity_errors);
        metadata=(; N,chi,steps_per_interval=64))
end
