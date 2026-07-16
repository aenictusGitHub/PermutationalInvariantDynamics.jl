using LinearAlgebra
using PermutationalInvariantDynamics

include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A finite-size reproduction of the decay-only model and the two rate ratios
# in Fig. 2(a,b) of Zhang, Zhang, and Mølmer, NJP 20, 112001 (2018). The paper
# used N=50 and 512 sector-resolved pure pseudo-state trajectories. The smaller
# default below is a fast regression for this library's density-valued PI
# unraveling; set N and ntrajectories to the paper values for a production run.
N = 10
ntrajectories = 256
GammaC = 1.0

function physical_diagonal(operator)
    diagonal = reduce(vcat,
        (diag(block) for (_,block) in each_schur_block(operator)))
    maximum(abs,imag.(diagonal)) < 2e-12 ||
        error("observable has an unexpectedly complex Schur diagonal")
    real.(diagonal)
end

function run_decay_case(gammaL;times,seed,dt,dtmax)
    model = zhang2018_superradiance_model(N;GammaC,gammaL)
    basis = model.basis
    rho0 = iid_pure_state(basis,ComplexF64[0,1])
    radiation = zhang2018_radiation_operators(basis;GammaC,gammaL)

    # Decay from a fully excited state stays Schur/GT diagonal. The exact
    # master-equation reference can therefore be exponentiated in the
    # certified population space (36 rather than 286 coordinates for N=10).
    population_plan = PopulationPlan(model)
    @assert population_plan.invariance.invariant === true
    generator = Matrix(population_generator(
        population_plan;representation=:sparse))
    initial_populations = diagonal_populations(rho0)
    populations = [exp(t*generator)*initial_populations for t in times]
    cavity_weights = physical_diagonal(radiation.cavity)
    free_space_weights = physical_diagonal(radiation.free_space)
    reference_cavity = [real(dot(cavity_weights,p)) for p in populations]
    reference_free_space = [real(dot(free_space_weights,p)) for p in populations]

    trajectories = quantum_trajectories(
        model,rho0,times,ntrajectories;
        algorithm=:event,dt,dtmax,abstol=1e-9,reltol=1e-7,
        event_time_tolerance=1e-8,seed,
    )
    statistics = trajectory_statistics(
        trajectories;
        observables=(cavity=radiation.cavity,
                     free_space=radiation.free_space),
        nchannels=2,
    )
    cavity = statistics.observables.observables[:cavity]
    free_space = statistics.observables.observables[:free_space]

    # Sampling errors can vanish at the deterministic initial point. The
    # additive floor covers only floating-point/integration roundoff there;
    # everywhere else the six-standard-error statistical gate dominates.
    numerical_floor = 2e-7
    cavity_error = abs.(cavity.mean-reference_cavity)
    free_space_error = abs.(free_space.mean-reference_free_space)
    @assert all(cavity_error .<= 6 .* cavity.standard_error .+ numerical_floor)
    @assert all(free_space_error .<=
                6 .* free_space.standard_error .+ numerical_floor)
    maximum_cavity_z = maximum(cavity_error ./
        max.(cavity.standard_error,numerical_floor/6))
    maximum_free_space_z = maximum(free_space_error ./
        max.(free_space.standard_error,numerical_floor/6))

    @assert abs(reference_cavity[1]-N*GammaC) < 2e-11
    @assert abs(reference_free_space[1]-N*gammaL) < 2e-11
    @assert maximum(abs(sum(p)-1) for p in populations) < 2e-11

    println("gammaL/GammaC = ",gammaL/GammaC)
    println("  population / full PI coordinates = ",
            population_dimension(basis)," / ",length(basis))
    println("  max cavity pulse = ",maximum(reference_cavity),
            " at GammaC*t = ",times[argmax(reference_cavity)])
    println("  max free-space pulse = ",maximum(reference_free_space),
            " at GammaC*t = ",times[argmax(reference_free_space)])
    println("  max trajectory/master cavity standardized error = ",
            maximum_cavity_z)
    println("  max trajectory/master free-space standardized error = ",
            maximum_free_space_z)
    println("  mean jumps by channel (cavity, free space) = ",
            Tuple(channel.mean for channel in statistics.jumps.channels))

    (;times,reference_cavity,reference_free_space,cavity,free_space,
      maximum_cavity_z,maximum_free_space_z,jumps=statistics.jumps)
end

comparable_rates = run_decay_case(
    GammaC;
    times=collect(range(0.0,1.0;length=41)),
    seed=2018,dt=0.05,dtmax=0.1,
)
dominant_local_decay = run_decay_case(
    10GammaC;
    times=collect(range(0.0,0.6;length=31)),
    seed=2019,dt=0.02,dtmax=0.05,
)

# The qualitative contrast of Fig. 2 is already visible at this finite size:
# comparable rates permit a collective burst above its initial intensity,
# whereas strong local loss starts with a much larger free-space flux.
@assert maximum(comparable_rates.reference_cavity) > N*GammaC
@assert maximum(dominant_local_decay.reference_cavity) <= N*GammaC+2e-10
@assert maximum(dominant_local_decay.reference_cavity) <
        maximum(dominant_local_decay.reference_free_space)

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1200, 480), fontsize=17)
    cases = ((comparable_rates, 1), (dominant_local_decay, 10))
    for (column, (case, ratio)) in enumerate(cases)
        axis = M.Axis(
            figure[1, column]; xlabel="Γc t", ylabel="radiated intensity / Γc",
            title="γl / Γc = $ratio")
        M.band!(axis, case.times,
                case.cavity.mean .- case.cavity.standard_error,
                case.cavity.mean .+ case.cavity.standard_error;
                color=(:firebrick, 0.20))
        M.lines!(axis, case.times, case.reference_cavity;
                 color=:firebrick, linewidth=2.7,
                 label="cavity master equation")
        M.scatter!(axis, case.times, case.cavity.mean;
                   color=:firebrick, markersize=6,
                   label="cavity trajectories")
        M.band!(axis, case.times,
                case.free_space.mean .- case.free_space.standard_error,
                case.free_space.mean .+ case.free_space.standard_error;
                color=(:royalblue, 0.18))
        M.lines!(axis, case.times, case.reference_free_space;
                 color=:royalblue, linewidth=2.7,
                 label="free-space master equation")
        M.scatter!(axis, case.times, case.free_space.mean;
                   color=:royalblue, markersize=6,
                   label="free-space trajectories")
        M.axislegend(axis; position=:rt, labelsize=13)
    end
    save_example_figure(figure, "zhang2018_superradiant_trajectories")
end
