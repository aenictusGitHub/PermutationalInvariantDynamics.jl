using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

N=8;chi=0.3;sx=ComplexF64[0 1;1 0]
model=kitagawa1993_oat_model(N;chi=chi);b=model.basis
rho0=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
times=range(0,1.5;length=31)
prepared=compile(model;backend=:matrixfree)
solution=solve_dynamics(prepared,rho0,(first(times),last(times));
                        saveat=times,steps_per_interval=64)
geometry=OneBodyGeometry(b)
Jx=CollectiveObservablePlan(b,sx/2;cache=geometry)
numeric=[collective_expectation(rho,Jx) for rho in solution]
exact=[kitagawa1993_mean_spin_exact(N,t;chi=chi) for t in times]
@assert diagnostics(last(solution);atol=1e-10,rtol=1e-10).valid
@assert maximum(abs,imag.(numeric)) < 1e-10
errors=abs.(numeric.-exact)
println("Kitagawa--Ueda OAT maximum |<Jx>-exact| = ",maximum(errors))

# Product-Schur data for the fixed one-particle marginal are also reusable.
one_body=ReductionPlan(b,1)
one_body_purities=[reduced_purity(rho,1;plan=one_body) for rho in solution]
println("final one-particle purity = ",last(one_body_purities))

if makie_available()
    M=makie_module()
    scaled_times=chi .* collect(times)
    figure=M.Figure(size=(1080,430),fontsize=17)
    spin_axis=M.Axis(
        figure[1,1];xlabel="χt",ylabel="⟨Jx⟩ / (N/2)",
        title="One-axis-twisting mean spin")
    purity_axis=M.Axis(
        figure[1,2];xlabel="χt",ylabel="tr(ρ₁²)",
        title="One-spin purity")

    M.lines!(spin_axis,scaled_times,exact ./ (N/2);
             color=:black,linewidth=2.7,label="Kitagawa–Ueda formula")
    M.scatter!(spin_axis,scaled_times,real.(numeric) ./ (N/2);
               color=:royalblue,markersize=8,label="PI dynamics")
    M.axislegend(spin_axis;position=:rt,labelsize=13)

    M.lines!(purity_axis,scaled_times,one_body_purities;
             color=:darkorange,linewidth=2.7,label="reduced PI state")
    M.hlines!(purity_axis,[1.0];color=:gray50,linestyle=:dash,
              label="pure one-spin state")
    M.axislegend(purity_axis;position=:rb,labelsize=13)
    save_example_figure(figure, "kitagawa1993_one_axis_twisting")
end
