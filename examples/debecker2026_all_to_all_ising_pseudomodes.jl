using LinearAlgebra
using PermutationalInvariantDynamics

include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Permutation-invariant all-to-all specialization of the local-pseudomode
# embedding in Debecker et al., "Generating spatial correlations in Ising
# chains via bath engineering" (2026 draft).  The draft studies a
# nearest-neighbour ring.  Here every spin pair has the same interaction, so
# one spin plus its local truncated pseudomode is an exact PI supersite.
#
# Set PI_PSEUDOMODE_FULL_SCAN=1 for a denser parameter grid.  The default is a
# compact executable research example, not a digitization of the manuscript.
const FULL_SCAN = get(ENV, "PI_PSEUDOMODE_FULL_SCAN", "0") == "1"

function checked_real(value,name;atol=2e-9)
    abs(imag(value))<=atol*max(1,abs(real(value)))||error(
        "$name has an unexpectedly large imaginary part: $value")
    real(value)
end

"""Reconstruct the physical two-spin state after tracing both pseudomodes."""
function spin_pair_density_matrix(rho,operators,geometry)
    result=zeros(ComplexF64,4,4)
    for mu in 1:4,nu in 1:4
        moment=two_body_expectation(
            rho,operators.lifted_paulis[mu],operators.lifted_paulis[nu];
            cache=geometry)
        coefficient=checked_real(moment,"two-spin Pauli moment")
        result .+= (coefficient/4)*kron(
            operators.spin_paulis[mu],operators.spin_paulis[nu])
    end
    hermiticity=norm(result-adjoint(result))
    hermiticity<2e-9||error(
        "reconstructed spin-pair state is not Hermitian: residual=$hermiticity")
    abs(real(tr(result))-1)<2e-9||error(
        "reconstructed spin-pair state does not have unit trace")
    abs(imag(tr(result)))<2e-9||error(
        "reconstructed spin-pair trace is unexpectedly complex")
    minimum(eigvals(Hermitian(result)))>-2e-8||error(
        "reconstructed spin-pair state is not positive within numerical tolerance")
    result
end

function transpose_second_qubit(rho)
    size(rho)==(4,4)||throw(DimensionMismatch(
        "the spin-pair density matrix must be 4 by 4"))
    result=similar(rho)
    for first_row in 1:2,second_row in 1:2,
        first_column in 1:2,second_column in 1:2
        row=second_row+2(first_row-1)
        column=second_column+2(first_column-1)
        source_row=second_column+2(first_row-1)
        source_column=second_row+2(first_column-1)
        result[row,column]=rho[source_row,source_column]
    end
    result
end

"""Negativity of the two physical spins, not of two full supersites."""
function spin_pair_negativity(rho,operators,geometry)
    spin_pair=spin_pair_density_matrix(rho,operators,geometry)
    transposed=transpose_second_qubit(spin_pair)
    residual=norm(transposed-adjoint(transposed))
    residual<2e-9||error(
        "spin-pair partial transpose is not Hermitian: residual=$residual")
    eigenvalues=eigvals(Hermitian(transposed))
    sum((-value for value in eigenvalues if value<0);init=0.0)
end

function all_to_all_model(basis,operators;J,omega_c,gamma,kappa,
                          coupling=:minus)
    # J is the extensive Curie--Weiss scale.  The literal coefficient of each
    # unordered pair is written explicitly; the package inserts no Kac factor.
    Jpair=J/(basis.N-1)
    debecker2026_all_to_all_ising_pseudomode_model(
        basis,operators;Jpair,omega_c,gamma,kappa,coupling)
end

function vacuum_product_state(basis,operators)
    local_state=zeros(ComplexF64,operators.dsite)
    # Supersite order is spin tensor mode.  This is |g> tensor |0>, an
    # uncorrelated state with Cxx(0)=0.  The all-pair Hamiltonian and the
    # structured local reservoirs generate the plotted correlations.
    local_state[1]=1
    iid_pure_state(basis,local_state)
end

function excited_vacuum_product_state(basis,operators)
    local_state=zeros(ComplexF64,operators.dsite)
    # With spin tensor mode ordering, the first state of the excited-spin
    # block is |e> tensor |0>.  It has even global spin parity and therefore
    # selects the parity-even stationary component for coupling=:z.
    local_state[operators.levels+1]=1
    iid_pure_state(basis,local_state)
end

"""Maximum x-GHZ fidelity after tracing every pseudomode."""
function spin_x_ghz_fidelity(rho,operators,geometry)
    plus=ComplexF64[1,1]/sqrt(2)
    minus=ComplexF64[1,-1]/sqrt(2)
    identity_mode=Matrix{ComplexF64}(I,operators.levels,operators.levels)
    plus_projector=kron(plus*adjoint(plus),identity_mode)
    minus_projector=kron(minus*adjoint(minus),identity_mode)
    coherence=kron(plus*adjoint(minus),identity_mode)
    order=rho.basis.N
    plus_population=ordered_local_moment(
        rho,ntuple(_->plus_projector,order);cache=geometry)
    minus_population=ordered_local_moment(
        rho,ntuple(_->minus_projector,order);cache=geometry)
    offdiagonal=ordered_local_moment(
        rho,ntuple(_->coherence,order);cache=geometry)
    diagonal=checked_real(
        plus_population+minus_population,"x-GHZ diagonal population")
    fidelity=diagonal/2+abs(offdiagonal)
    -2e-8<=fidelity<=1+2e-8||error(
        "the optimized spin x-GHZ fidelity is outside [0,1]: $fidelity")
    fidelity
end

function parity_selected_ghz_point(
        basis,operators,rho0,ghz_geometry,expv_workspace;
        J,omega_c,gamma,kappa,settling_time)
    model=all_to_all_model(
        basis,operators;J,omega_c,gamma,kappa,coupling=:z)
    generator=liouvillian(model;representation=:matrixfree)
    propagated=krylov_expv(
        generator,rho0.data,settling_time;
        workspace=expv_workspace,atol=1e-10,rtol=1e-9,
        maximum_step=20.0)
    propagated.converged||error(
        "parity-selected exponential action did not reach t=$settling_time")
    state=PIState(basis,propagated.value)
    report=diagnostics(state)
    report.valid||error(
        "invalid parity-selected state at J=$J, kappa=$kappa")
    residual=norm(generator*state.data)
    fidelity=spin_x_ghz_fidelity(state,operators,ghz_geometry)
    (;fidelity,residual,trace_error=report.trace_error,
      estimated_error=propagated.estimated_error)
end

function pair_observable(basis,operators,pair_geometry)
    pbody_collective_operator(
        basis,kron(operators.x_site,operators.x_site),2;
        cache=pair_geometry)
end

function dynamics_case(basis,operators,pair_xx,top_level_sum,rho0;
                       J,omega_c,gamma,kappa,times,krylovdim,atol,rtol)
    model=all_to_all_model(
        basis,operators;J,omega_c,gamma,kappa,coupling=:minus)
    generator=liouvillian(model;representation=:matrixfree)
    workspace=KrylovExpvWorkspace(ComplexF64,length(basis),krylovdim)
    state=copy(rho0)
    next_data=similar(state.data)
    pair_count=binomial(basis.N,2)
    cxx=zeros(Float64,length(times))
    top=zeros(Float64,length(times))
    estimated_error=0.0
    operator_applications=0
    for index in eachindex(times)
        cxx[index]=checked_real(
            expectation(state,pair_xx),"pair correlation")/pair_count
        top[index]=checked_real(
            expectation(state,top_level_sum),"top-level population")/basis.N
        index==lastindex(times)&&continue
        interval=times[index+1]-times[index]
        interval>0||throw(ArgumentError(
            "dynamics output times must be strictly increasing"))
        result=krylov_expv!(
            next_data,generator,state.data,interval,workspace;
            atol,rtol,maximum_step=interval)
        result.converged||error(
            "dynamics exponential action did not reach t=$(times[index+1])")
        copyto!(state.data,next_data)
        estimated_error+=result.estimated_error
        operator_applications+=result.operator_applications
    end
    report=diagnostics(state)
    report.valid||error(
        "invalid final dynamics state for kappa=$kappa")
    (;kappa,cxx,top,estimated_error,operator_applications,
      trace_error=report.trace_error)
end

function stationary_point(basis,operators,geometry;J,omega_c,gamma,kappa)
    model=all_to_all_model(
        basis,operators;J,omega_c,gamma,kappa,coupling=:minus)
    # The default scan has only 816 PI coordinates.  Sparse direct solving is
    # both faster and more reliable here than making every grid point a Krylov
    # convergence problem; the central point is cross-checked matrix-free
    # below, and the guide gives the large-system replacement.
    prepared=compile(model;backend=:sparse)
    result=stationary_state(
        prepared;algorithm=DirectAlgorithm(),return_info=true)
    result.info.converged||error(
        "direct steady solve did not converge at J=$J, kappa=$kappa")
    result.info.residual<5e-8||error(
        "steady residual is too large at J=$J, kappa=$kappa")
    report=diagnostics(result.state)
    report.valid||error(
        "invalid stationary PI state at J=$J, kappa=$kappa")
    cxx=checked_real(two_body_expectation(
        result.state,operators.x_site,operators.x_site;cache=geometry),
        "stationary pair correlation")
    negativity=spin_pair_negativity(result.state,operators,geometry)
    top=checked_real(collective_expectation(
        result.state,operators.mode_top;cache=geometry),
        "stationary top-level population")/basis.N
    (;state=result.state,cxx,negativity,top,
      residual=result.info.residual,trace_error=report.trace_error)
end

function cutoff_curve(N,nmax;J,omega_c,gamma,kappa,times,
                      steps_per_interval)
    operators=debecker2026_pseudomode_operators(nmax)
    basis=PIBasis(N,operators.dsite)
    pair_geometry=PBodyGeometry(basis,2)
    pair_xx=pair_observable(basis,operators,pair_geometry)
    rho0=vacuum_product_state(basis,operators)
    model=all_to_all_model(
        basis,operators;J,omega_c,gamma,kappa,coupling=:minus)
    prepared=compile(model;backend=:matrixfree)
    output=solve_dynamics(
        prepared,rho0,(first(times),last(times));saveat=times,
        steps_per_interval,
        observables=(pair_sum=pair_xx,top_level=operators.mode_top),
        save_states=false)
    pair_count=binomial(N,2)
    cxx=checked_real.(output.observables[:pair_sum],"cutoff pair correlation") ./
        pair_count
    top=checked_real.(output.observables[:top_level],"cutoff top population") ./ N
    (;nmax,dimension=length(basis),cxx,top)
end

function main()
    N=3
    nmax=1
    omega_c=1.0
    gamma=0.05
    J_dynamic=0.25
    operators=debecker2026_pseudomode_operators(nmax)
    basis=PIBasis(N,operators.dsite)
    geometry=OneBodyGeometry(basis)
    pair_geometry=PBodyGeometry(basis,2)
    pair_xx=pair_observable(basis,operators,pair_geometry)
    top_level_sum=collective_operator(
        basis,operators.mode_top;cache=geometry)
    rho0=vacuum_product_state(basis,operators)

    println("All-to-all spin--pseudomode Ising specialization")
    println("N=$N, nmax=$nmax, local dimension=$(basis.d), " *
            "PI coordinates=$(length(basis))")
    println("The plotted J is Kac scaled: Jpair=J/(N-1).")

    # Independent lowering identities: the p-body pair Hamiltonian is the
    # collective square up to an irrelevant scalar, and sparse/matrix-free
    # plans must apply the same generator.
    reference_model=all_to_all_model(
        basis,operators;J=J_dynamic,omega_c,gamma,kappa=2.0,
        coupling=:minus)
    Jx=collective_operator(basis,operators.x_site)
    direct_terms=(reference_model.terms[1],reference_model.terms[2],
        DirectPIHamiltonian((-(J_dynamic/(N-1))/2)*(Jx*Jx)),
        reference_model.terms[4])
    direct_model=PIModel(basis,direct_terms)
    Ls=liouvillian(reference_model;representation=:sparse)
    Ldirect=liouvillian(direct_model;representation=:sparse)
    all_pair_error=norm(Ls-Ldirect)
    Lmf=liouvillian(reference_model;representation=:matrixfree)
    action_error=norm(Ls*rho0.data-Lmf*rho0.data)
    generator_report=check_generator(reference_model)
    @assert all_pair_error<2e-9
    @assert action_error<2e-10
    @assert generator_report.trace_preservation_error<2e-9

    # Manuscript-Fig.-2(b)-style dynamics, now using the only distinct PI pair
    # correlation rather than a distance-dependent nearest-neighbour value.
    dynamics_times=collect(range(0.0,100.0;length=201))
    kappas=(1.0,5.0,20.0)
    dynamics=[dynamics_case(
        basis,operators,pair_xx,top_level_sum,rho0;
        J=J_dynamic,omega_c,gamma,kappa,times=dynamics_times,
        krylovdim=30,atol=1e-11,rtol=1e-9) for kappa in kappas]
    refined=dynamics_case(
        basis,operators,pair_xx,top_level_sum,rho0;
        J=J_dynamic,omega_c,gamma,kappa=20.0,times=dynamics_times,
        krylovdim=40,atol=2e-12,rtol=2e-10)
    dynamics_krylov_error=maximum(abs.(dynamics[3].cxx-refined.cxx))
    @assert dynamics_krylov_error<2e-7
    @assert maximum(curve.estimated_error for curve in dynamics)<2e-8
    @assert maximum(maximum(curve.top) for curve in dynamics)<0.08

    # A coarse stationary map analogous in layout to Fig. 3.  Since a PI
    # state has no distance coordinate, the data are the common pair Cxx and
    # the negativity of the two physical spins after tracing both modes.
    J_values=FULL_SCAN ? collect(range(0.05,0.55;length=9)) :
                         [0.05,0.15,0.25,0.40,0.55]
    kappa_values=FULL_SCAN ? collect(10 .^ range(log10(0.5),log10(8.0);length=9)) :
                             [0.5,1.0,2.0,4.0,8.0]
    cxx_map=zeros(length(J_values),length(kappa_values))
    negativity_map=similar(cxx_map)
    top_map=similar(cxx_map)
    residual_map=similar(cxx_map)
    for (jindex,J) in pairs(J_values)
        for kindex in eachindex(kappa_values)
            point=stationary_point(
                basis,operators,geometry;
                J,omega_c,gamma,kappa=kappa_values[kindex])
            cxx_map[jindex,kindex]=point.cxx
            negativity_map[jindex,kindex]=point.negativity
            top_map[jindex,kindex]=point.top
            residual_map[jindex,kindex]=point.residual
        end
    end

    # At one reference point, compare independent sparse-direct and
    # matrix-free GMRES solutions.  Evaluate the direct reference explicitly
    # instead of assuming that an optional scan grid contains this point.
    direct_reference=stationary_point(
        basis,operators,geometry;
        J=J_dynamic,omega_c,gamma,kappa=2.0)
    krylov_prepared=compile(reference_model;backend=:matrixfree)
    krylov=stationary_state(
        krylov_prepared;
        algorithm=GMRESAlgorithm(krylovdim=50,maxiter=800),
        initial_state=rho0,atol=1e-10,rtol=1e-9,
        return_info=true)
    krylov.info.converged||error(
        "reference matrix-free GMRES steady solve did not converge")
    krylov_cxx=checked_real(two_body_expectation(
        krylov.state,operators.x_site,operators.x_site;cache=geometry),
        "Krylov stationary pair correlation")
    solver_agreement=abs(krylov_cxx-direct_reference.cxx)
    @assert solver_agreement<2e-6
    @assert maximum(residual_map)<5e-8
    @assert minimum(negativity_map)>=0
    @assert maximum(top_map)<0.12

    # Manuscript-Fig.-4-style strong-symmetry calculation.  For L=sigma_z,
    # the fully excited spin product has even global spin parity.  Evolving it
    # to a checked stationary limit selects that parity component without
    # asking an unrestricted trace-fixed solver to choose among steady states.
    ghz_geometry=PBodyGeometry(basis,N)
    ghz_initial=excited_vacuum_product_state(basis,operators)
    ghz_workspace=KrylovExpvWorkspace(ComplexF64,length(basis),35)
    ghz_settling_time=1600.0/omega_c
    ghz_map=zeros(length(J_values),length(kappa_values))
    ghz_residual_map=similar(ghz_map)
    ghz_error_map=similar(ghz_map)
    for (jindex,J) in pairs(J_values)
        for kindex in eachindex(kappa_values)
            point=parity_selected_ghz_point(
                basis,operators,ghz_initial,ghz_geometry,ghz_workspace;
                J,omega_c,gamma,kappa=kappa_values[kindex],
                settling_time=ghz_settling_time)
            ghz_map[jindex,kindex]=point.fidelity
            ghz_residual_map[jindex,kindex]=point.residual
            ghz_error_map[jindex,kindex]=point.estimated_error
        end
    end
    half_time_point=parity_selected_ghz_point(
        basis,operators,ghz_initial,ghz_geometry,ghz_workspace;
        J=first(J_values),omega_c,gamma,kappa=last(kappa_values),
        settling_time=ghz_settling_time/2)
    ghz_time_error=abs(half_time_point.fidelity-ghz_map[1,end])
    @assert minimum(ghz_map)>=-2e-8
    @assert maximum(ghz_map)<=1+2e-8
    @assert maximum(ghz_residual_map)<5e-8
    @assert maximum(ghz_error_map)<2e-8
    @assert ghz_time_error<2e-6

    # The draft does not state its pseudomode cutoff.  Compare nmax=1 and 2 at
    # one dynamical point and inspect the first omitted-boundary population in
    # the wider calculation.
    cutoff_times=collect(range(0.0,6.0;length=61))
    cutoff_1=cutoff_curve(
        N,1;J=J_dynamic,omega_c,gamma,kappa=2.0,
        times=cutoff_times,steps_per_interval=8)
    cutoff_2=cutoff_curve(
        N,2;J=J_dynamic,omega_c,gamma,kappa=2.0,
        times=cutoff_times,steps_per_interval=8)
    cutoff_error=maximum(abs.(cutoff_1.cxx-cutoff_2.cxx))
    @assert cutoff_error<5e-3
    @assert maximum(cutoff_2.top)<2e-6

    println("all-pair collective-square error = ",all_pair_error)
    println("sparse/matrix-free action error = ",action_error)
    println("Krylov tolerance-refinement Cxx error = ",dynamics_krylov_error)
    println("dynamics Liouvillian applications = ",
            Tuple(curve.operator_applications for curve in dynamics))
    println("direct/GMRES stationary Cxx error = ",solver_agreement)
    println("maximum scan residual = ",maximum(residual_map))
    println("maximum scan top-level population = ",maximum(top_map))
    println("spin-only negativity range = ",extrema(negativity_map))
    println("parity-selected x-GHZ fidelity range = ",extrema(ghz_map))
    println("maximum x-GHZ stationary residual = ",maximum(ghz_residual_map),
            "; half-time witness error = ",ghz_time_error)
    println("cutoff dimensions (nmax=1,2) = ",
            (cutoff_1.dimension,cutoff_2.dimension))
    println("cutoff Cxx difference = ",cutoff_error,
            "; wider top-level population = ",maximum(cutoff_2.top))

    (;N,nmax,omega_c,gamma,J_dynamic,dynamics_times,dynamics,
      J_values,kappa_values,cxx_map,negativity_map,top_map,residual_map,
      ghz_settling_time,ghz_map,ghz_residual_map,ghz_error_map,ghz_time_error,
      cutoff_times,cutoff_1,cutoff_2,dynamics_krylov_error,
      solver_agreement,cutoff_error)
end

results=main()

if makie_available()
    M=makie_module()
    figure=M.Figure(size=(1540,480),fontsize=17)
    dynamics_axis=M.Axis(
        figure[1,1];xlabel="ωc t",ylabel="Cxx(t)",
        title="Identical-pair dynamics, L = σ−")
    correlation_axis=M.Axis(
        figure[1,2];xlabel="J / ωc",ylabel="κ / ωc",
        yscale=log10,title="Steady identical-pair correlation")
    negativity_axis=M.Axis(
        figure[1,4];xlabel="J / ωc",ylabel="κ / ωc",
        yscale=log10,title="Spin-only two-spin negativity")

    colors=(:maroon,:tomato,:steelblue)
    for (curve,color) in zip(results.dynamics,colors)
        M.lines!(dynamics_axis,
            results.omega_c .* results.dynamics_times,curve.cxx;
            color,linewidth=2.6,
            label="κ / ωc = $(curve.kappa/results.omega_c)")
    end
    M.hlines!(dynamics_axis,[0.0];color=:gray65,linestyle=:dash)
    M.axislegend(dynamics_axis;position=:rt,labelsize=12)

    correlation_limit=max(maximum(abs,results.cxx_map),eps(Float64))
    correlation_plot=M.heatmap!(
        correlation_axis,results.J_values ./ results.omega_c,
        results.kappa_values ./ results.omega_c,results.cxx_map;
        colormap=:balance,colorrange=(-correlation_limit,correlation_limit))
    if minimum(results.cxx_map)<0<maximum(results.cxx_map)
        M.contour!(correlation_axis,
            results.J_values ./ results.omega_c,
            results.kappa_values ./ results.omega_c,results.cxx_map;
            levels=[0.0],color=:black,linewidth=2)
    end
    M.Colorbar(figure[1,3],correlation_plot;label="Cxx")

    negativity_limit=max(maximum(results.negativity_map),eps(Float64))
    negativity_plot=M.heatmap!(
        negativity_axis,results.J_values ./ results.omega_c,
        results.kappa_values ./ results.omega_c,results.negativity_map;
        colormap=:magma,colorrange=(0,negativity_limit))
    M.Colorbar(figure[1,5],negativity_plot;label="negativity")
    M.Label(
        figure[2,1:5],
        "All-to-all PI analogue: Jpair = J/(N−1), N=$(results.N), nmax=$(results.nmax). " *
        "No spatial correlation length is defined.";
        fontsize=14,color=:gray35,tellwidth=false)
    save_example_figure(
        figure,"debecker2026_all_to_all_ising_pseudomodes")

    cutoff_figure=M.Figure(size=(1050,430),fontsize=17)
    cutoff_axis=M.Axis(
        cutoff_figure[1,1];xlabel="ωc t",ylabel="Cxx(t)",
        title="Pseudomode-cutoff comparison")
    boundary_axis=M.Axis(
        cutoff_figure[1,2];xlabel="ωc t",
        ylabel="highest retained-level population",
        yscale=log10,title="Wider-cutoff boundary population")
    M.lines!(cutoff_axis,results.omega_c .* results.cutoff_times,
             results.cutoff_1.cxx;color=:black,linewidth=2.6,
             label="nmax = 1")
    M.lines!(cutoff_axis,results.omega_c .* results.cutoff_times,
             results.cutoff_2.cxx;color=:darkorange,linewidth=2.3,
             linestyle=:dash,label="nmax = 2")
    M.axislegend(cutoff_axis;position=:rb,labelsize=12)
    M.lines!(boundary_axis,results.omega_c .* results.cutoff_times,
             max.(results.cutoff_2.top,eps(Float64));
             color=:royalblue,linewidth=2.5)
    save_example_figure(
        cutoff_figure,"debecker2026_all_to_all_ising_pseudomodes_cutoff")

    ghz_figure=M.Figure(size=(700,520),fontsize=17)
    ghz_axis=M.Axis(
        ghz_figure[1,1];xlabel="J / ωc",ylabel="κ / ωc",
        yscale=log10,
        title="Parity-selected spin x-GHZ witness, L = σz")
    ghz_plot=M.heatmap!(
        ghz_axis,results.J_values ./ results.omega_c,
        results.kappa_values ./ results.omega_c,results.ghz_map;
        colormap=:RdBu,colorrange=(0,1))
    if minimum(results.ghz_map)<0.5<maximum(results.ghz_map)
        M.contour!(
            ghz_axis,results.J_values ./ results.omega_c,
            results.kappa_values ./ results.omega_c,results.ghz_map;
            levels=[0.5],color=:black,linewidth=2.2)
    end
    M.Colorbar(ghz_figure[1,2],ghz_plot;
               label="maximized spin x-GHZ fidelity")
    M.Label(
        ghz_figure[2,1:2],
        "Even-parity long-time state; max ‖ℒρ‖ = " *
        string(round(maximum(results.ghz_residual_map);sigdigits=3));
        fontsize=13,color=:gray35,tellwidth=false)
    save_example_figure(
        ghz_figure,"debecker2026_all_to_all_ising_pseudomodes_ghz")
end
