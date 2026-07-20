using LinearAlgebra
using PermutationalInvariantDynamics

include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie
include(joinpath(@__DIR__, "utils", "contour_fit.jl"))
using .ExampleContourFit

# Permutation-invariant all-to-all specialization of the local-pseudomode
# embedding in Debecker et al., "Generating spatial correlations in Ising
# chains via bath engineering" (2026 draft).  The draft studies a
# nearest-neighbour ring.  Here every spin pair has the same interaction, so
# one spin plus its local truncated pseudomode is an exact PI supersite.
#
# Set PI_PSEUDOMODE_FULL_SCAN=1 for a denser parameter grid.  The default is a
# compact executable research example, not a digitization of the manuscript.
const FULL_SCAN = get(ENV, "PI_PSEUDOMODE_FULL_SCAN", "0") == "1"
const CONTOUR_FIT_FALLBACK_RELATIVE_L2_THRESHOLD = 0.10

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

"""
Return the stored PI-coordinate indices of one Hilbert-space charge block of
`local_symmetry^tensor N`.  The fast content formula applies to a diagonal
local unitary and resolves ket and bra charges separately; unlike a weak
conjugation projector, it does not merge the `(+,+)` and `(-,-)` blocks.
"""
function diagonal_hilbert_charge_indices(
        basis,local_symmetry;charge,atol=1e-12,rtol=1e-10)
    restriction=diagonal_symmetry_restriction(
        basis,local_symmetry;charge,atol,rtol,
        label=:longitudinal_spin_parity)
    retained_indices(restriction)
end

"""Steady state in one certified diagonal strong-symmetry support block."""
function parity_selected_ghz_point(
        basis,operators,ghz_geometry,restriction;
        J,omega_c,gamma,kappa,invariance_atol=1e-12,
        invariance_rtol=1e-10)
    model=all_to_all_model(
        basis,operators;J,omega_c,gamma,kappa,coupling=:z)
    generator=liouvillian(model;representation=:sparse)
    reduced_generator=RestrictedLiouvillian(
        generator,restriction;atol=invariance_atol,
        rtol=invariance_rtol)
    reduced=steady_state(
        reduced_generator;trace_vector=reduced_generator.tracevec,
        method=:direct,
        diagnostics=:basic,return_info=true,atol=1e-11,rtol=1e-10)
    data=zeros(ComplexF64,length(basis))
    embed!(data,restriction,reduced.state)
    state=PIState(basis,data)
    report=diagnostics(state)
    report.valid||error(
        "invalid parity-selected state at J=$J, kappa=$kappa")
    residual=norm(generator*state.data)
    fidelity=spin_x_ghz_fidelity(state,operators,ghz_geometry)
    parity=checked_real(ordered_local_moment(
        state,ntuple(_->operators.z_site,basis.N);cache=ghz_geometry),
        "global spin parity")
    certificate=reduced_generator.certificate
    (;state,fidelity,residual,trace_error=report.trace_error,parity,
      invariance_leakage=certificate.leakage_norm,
      relative_invariance_leakage=certificate.relative_leakage,
      reduced_dimension=length(restriction))
end

"""Independent long-time propagation used only to validate the block solve."""
function parity_selected_ghz_propagation(
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
        "invalid propagated parity-selected state at J=$J, kappa=$kappa")
    residual=norm(generator*state.data)
    fidelity=spin_x_ghz_fidelity(state,operators,ghz_geometry)
    (;state,fidelity,residual,trace_error=report.trace_error,
      estimated_error=propagated.estimated_error,
      operator_applications=propagated.operator_applications)
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
    # For the default N=3 calculation there are only 816 PI coordinates.
    # Sparse direct solving is then faster and more reliable than making every
    # grid point a Krylov convergence problem; the central point is
    # cross-checked matrix-free below, and the guide gives the large-system
    # replacement.
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

function save_fitted_boundary_data(results)
    directory=figure_output_directory()
    scan_mode=FULL_SCAN ? :full : :compact
    run_stem="debecker2026_all_to_all_ising_pseudomodes_" *
        "N$(results.N)_nmax$(results.nmax)_$(scan_mode)"
    common_metadata=(
        model="all_to_all_ising_local_pseudomodes",
        particles=results.N,
        pseudomode_cutoff=results.nmax,
        scan_mode,
        x_coordinate="J_over_omega_c",
        y_coordinate="kappa_over_omega_c",
    )
    cxx_path=save_level_contour_data(
        joinpath(directory,"$(run_stem)_cxx_boundary.txt"),
        results.cxx_zero_fit;
        x_label="J_over_omega_c",y_label="kappa_over_omega_c",
        metadata=merge(common_metadata,(
            observable="Cxx",
            coupling="sigma_minus",
            boundary="Cxx_equals_0",
        )))
    witness_path=save_level_contour_data(
        joinpath(directory,"$(run_stem)_ghz_witness_boundary.txt"),
        results.ghz_half_fit;
        x_label="J_over_omega_c",y_label="kappa_over_omega_c",
        metadata=merge(common_metadata,(
            observable="phase_optimized_spin_x_GHZ_fidelity",
            coupling="sigma_z",
            boundary="F_GHZx_max_equals_0.5",
        )))
    println("Fitted boundary data written to $cxx_path and $witness_path")
    (;cxx=cxx_path,entanglement_witness=witness_path)
end

function main(;N::Int=parse(Int,get(ENV,"PI_PSEUDOMODE_N","3")),
              nmax::Int=1)
    N>=2||throw(ArgumentError(
        "the all-to-all example requires N>=2 for its Kac scaling"))
    nmax>=1||throw(ArgumentError("nmax must be positive"))
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

    dimensionless_J=J_values./omega_c
    dimensionless_kappa=kappa_values./omega_c
    cxx_zero_fit=quadratic_level_contour_fit(
        dimensionless_J,dimensionless_kappa,cxx_map;
        level=0.0,samples=240,
        fallback_relative_l2_threshold=
            CONTOUR_FIT_FALLBACK_RELATIVE_L2_THRESHOLD)
    @assert isfinite(cxx_zero_fit.alpha)&&cxx_zero_fit.alpha>0
    @assert cxx_zero_fit.point_count>=2

    # Estimate the same stationary state independently from density-valued
    # quantum-jump paths.  Samples along one path are correlated, so the
    # estimator first averages the post-settling time window of each path and
    # only then treats the path means as independent Monte Carlo samples.
    trajectory_plan=TrajectoryPlan(krylov_prepared)
    trajectory_count=FULL_SCAN ? 64 : 16
    trajectory_workers=min(trajectory_count,max(1,Threads.nthreads()))
    trajectory_workspace=TrajectoryBatchWorkspace(
        trajectory_plan,rho0;workers=trajectory_workers)
    trajectory_stationary=trajectory_steady_state(
        trajectory_plan,rho0;
        trajectories=trajectory_count,
        settling_time=100.0/omega_c,
        samples_per_trajectory=11,
        sampling_interval=10.0/omega_c,
        dt=0.02/omega_c,
        max_jump_probability=0.02,
        algorithm=:fixed,
        seed=2026,
        threaded=Threads.nthreads()>1,
        workspace=trajectory_workspace,
        return_info=true)
    trajectory_report=diagnostics(trajectory_stationary.state)
    trajectory_report.valid||error(
        "the trajectory stationary estimate is not a valid PI state")
    pair_count=binomial(N,2)
    trajectory_cxx=checked_real(
        expectation(trajectory_stationary.state,pair_xx),
        "trajectory stationary pair correlation")/pair_count
    trajectory_top=checked_real(
        expectation(trajectory_stationary.state,top_level_sum),
        "trajectory stationary top-level population")/N
    trajectory_negativity=spin_pair_negativity(
        trajectory_stationary.state,operators,geometry)
    trajectory_state_error=norm(
        trajectory_stationary.state.data-direct_reference.state.data)
    trajectory_cxx_error=abs(trajectory_cxx-direct_reference.cxx)
    trajectory_top_error=abs(trajectory_top-direct_reference.top)
    trajectory_negativity_error=abs(
        trajectory_negativity-direct_reference.negativity)
    @assert trajectory_state_error<=
        6*trajectory_stationary.standard_error+5e-3
    @assert trajectory_stationary.residual<0.1
    @assert trajectory_stationary.trace_error<2e-8

    # Resolve the same local channel into one-box Schur-sector Kraus branches.
    # Each conditional state is then a direct-sum weak-PI pseudo-ket, much
    # smaller here than the density-valued PI coordinate vector.  The
    # stationary estimator accumulates physical outer products directly and
    # retains no pseudo-ket histories.
    weak_initial=weak_pi_pseudoket(rho0)
    weak_trajectory_plan=WeakPITrajectoryPlan(krylov_prepared)
    weak_trajectory_count=2*trajectory_count
    weak_trajectory_workers=min(
        weak_trajectory_count,max(1,Threads.nthreads()))
    weak_trajectory_workspace=WeakPITrajectoryBatchWorkspace(
        weak_trajectory_plan,weak_initial;workers=weak_trajectory_workers)
    weak_trajectory_stationary=weak_pi_trajectory_steady_state(
        weak_trajectory_plan,weak_initial;
        trajectories=weak_trajectory_count,
        settling_time=100.0/omega_c,
        samples_per_trajectory=11,
        sampling_interval=10.0/omega_c,
        dt=0.02/omega_c,
        max_jump_probability=0.02,
        seed=22026,
        threaded=Threads.nthreads()>1,
        workspace=weak_trajectory_workspace,
        return_info=true)
    weak_trajectory_report=diagnostics(weak_trajectory_stationary.state)
    weak_trajectory_report.valid||error(
        "the weak-PI trajectory stationary estimate is not a valid PI state")
    weak_trajectory_cxx=checked_real(
        expectation(weak_trajectory_stationary.state,pair_xx),
        "weak-PI stationary pair correlation")/pair_count
    weak_trajectory_top=checked_real(
        expectation(weak_trajectory_stationary.state,top_level_sum),
        "weak-PI stationary top-level population")/N
    weak_trajectory_negativity=spin_pair_negativity(
        weak_trajectory_stationary.state,operators,geometry)
    weak_trajectory_state_error=norm(
        weak_trajectory_stationary.state.data-direct_reference.state.data)
    weak_trajectory_cxx_error=abs(
        weak_trajectory_cxx-direct_reference.cxx)
    weak_trajectory_top_error=abs(
        weak_trajectory_top-direct_reference.top)
    weak_trajectory_negativity_error=abs(
        weak_trajectory_negativity-direct_reference.negativity)
    trajectory_method_difference=norm(
        weak_trajectory_stationary.state.data-
        trajectory_stationary.state.data)
    weak_dimension=weak_pi_dimension(basis)
    @assert weak_dimension==length(weak_initial)
    @assert weak_initial.basis===basis
    @assert weak_dimension<length(basis)
    @assert BigInt(weak_dimension)<BigInt(basis.d)^N
    @assert weak_trajectory_state_error<=
        6*weak_trajectory_stationary.standard_error+5e-3
    @assert weak_trajectory_stationary.residual<0.1
    @assert weak_trajectory_stationary.trace_error<2e-8

    # Manuscript-Fig.-4-style strong-symmetry calculation.  For L=sigma_z,
    # the fully excited spin product has a definite global spin parity.  A
    # weak conjugation projector would retain both diagonal Hilbert-parity
    # blocks and leave the stationary solve nonunique.  Resolve the ket and
    # bra charge instead and solve directly in that exact support block.
    ghz_geometry=PBodyGeometry(basis,N)
    ghz_initial=excited_vacuum_product_state(basis,operators)
    ghz_initial_parity=checked_real(ordered_local_moment(
        ghz_initial,ntuple(_->operators.z_site,N);cache=ghz_geometry),
        "initial global spin parity")
    abs(abs(ghz_initial_parity)-1)<2e-10||error(
        "the longitudinal-coupling initial state has no definite spin parity")
    ghz_charge=ghz_initial_parity>0 ? 1.0 : -1.0
    ghz_restriction=diagonal_symmetry_restriction(
        basis,operators.z_site;charge=ghz_charge,
        label=:longitudinal_spin_parity)
    ghz_sector_indices=retained_indices(ghz_restriction)
    ghz_outside_indices=setdiff(1:length(basis),ghz_sector_indices)
    ghz_initial_leakage=norm(ghz_initial.data[ghz_outside_indices])
    ghz_initial_leakage<2e-12||error(
        "the longitudinal-coupling initial state leaks outside its selected parity block")
    ghz_reduced_dimension=length(ghz_sector_indices)
    ghz_reduced_dimension<length(basis)||error(
        "the selected spin-parity block did not reduce the PI coordinates")

    # There is also a global PI-compatible weak symmetry
    # (sigma_x tensor mode parity)^tensor N.  It flips the sign of every mode
    # jump, which leaves the dissipators invariant.  For odd N it exchanges
    # the two strong spin-parity blocks, so projecting onto only one of its
    # charges would change the requested parity-selected state.  For even N
    # it preserves those blocks; this example certifies that symmetry but
    # deliberately keeps the same strong-parity solve for both parities of N.
    mode_parity=Diagonal(ComplexF64[
        iseven(occupation) ? 1 : -1 for occupation in 0:operators.nmax])
    combined_weak_site=kron(operators.spin_paulis[2],mode_parity)
    longitudinal_reference_model=all_to_all_model(
        basis,operators;J=J_dynamic,omega_c,gamma,kappa=2.0,coupling=:z)
    spin_parity_symmetry=check_liouvillian_symmetry(
        longitudinal_reference_model,operators.z_site;basis)
    combined_weak_symmetry=check_liouvillian_symmetry(
        longitudinal_reference_model,combined_weak_site;basis)
    @assert spin_parity_symmetry.symmetric
    @assert combined_weak_symmetry.symmetric

    ghz_settling_time=1600.0/omega_c
    ghz_map=zeros(length(J_values),length(kappa_values))
    ghz_residual_map=similar(ghz_map)
    ghz_parity_map=similar(ghz_map)
    ghz_invariance_leakage_map=similar(ghz_map)
    for (jindex,J) in pairs(J_values)
        for kindex in eachindex(kappa_values)
            point=parity_selected_ghz_point(
                basis,operators,ghz_geometry,ghz_restriction;
                J,omega_c,gamma,kappa=kappa_values[kindex])
            ghz_map[jindex,kindex]=point.fidelity
            ghz_residual_map[jindex,kindex]=point.residual
            ghz_parity_map[jindex,kindex]=point.parity
            ghz_invariance_leakage_map[jindex,kindex]=
                point.invariance_leakage
        end
    end

    # Validate the reduced steady solve independently at one off-grid
    # reference point with the former long-time matrix-free propagation.
    ghz_reference=parity_selected_ghz_point(
        basis,operators,ghz_geometry,ghz_restriction;
        J=J_dynamic,omega_c,gamma,kappa=2.0)
    ghz_workspace=KrylovExpvWorkspace(ComplexF64,length(basis),35)
    propagated_reference=parity_selected_ghz_propagation(
        basis,operators,ghz_initial,ghz_geometry,ghz_workspace;
        J=J_dynamic,omega_c,gamma,kappa=2.0,
        settling_time=ghz_settling_time)
    half_time_reference=parity_selected_ghz_propagation(
        basis,operators,ghz_initial,ghz_geometry,ghz_workspace;
        J=J_dynamic,omega_c,gamma,kappa=2.0,
        settling_time=ghz_settling_time/2)
    ghz_time_error=abs(
        half_time_reference.fidelity-ghz_reference.fidelity)
    ghz_propagation_fidelity_error=abs(
        propagated_reference.fidelity-ghz_reference.fidelity)
    ghz_propagation_state_error=norm(
        propagated_reference.state.data-ghz_reference.state.data)
    @assert minimum(ghz_map)>=-2e-8
    @assert maximum(ghz_map)<=1+2e-8
    @assert maximum(ghz_residual_map)<5e-8
    @assert maximum(abs.(ghz_parity_map.-ghz_charge))<2e-8
    @assert maximum(ghz_invariance_leakage_map)<2e-10
    @assert propagated_reference.estimated_error<2e-8
    @assert ghz_propagation_fidelity_error<2e-8
    @assert ghz_propagation_state_error<2e-8
    @assert ghz_time_error<2e-6

    ghz_half_fit=quadratic_level_contour_fit(
        dimensionless_J,dimensionless_kappa,ghz_map;
        level=0.5,samples=240,
        fallback_model=:power_law,
        fallback_relative_l2_threshold=
            CONTOUR_FIT_FALLBACK_RELATIVE_L2_THRESHOLD)
    @assert isfinite(ghz_half_fit.alpha)&&ghz_half_fit.alpha>0
    @assert ghz_half_fit.point_count>=2

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
    println("trajectory stationary state error = ",trajectory_state_error,
            "; HS standard error = ",trajectory_stationary.standard_error)
    println("trajectory stationary Cxx = ",trajectory_cxx,
            "; direct-reference error = ",trajectory_cxx_error,
            "; spin negativity = ",trajectory_negativity)
    println("trajectory top-level population = ",trajectory_top,
            "; direct-reference error = ",trajectory_top_error,
            "; negativity error = ",trajectory_negativity_error)
    println("trajectory stationary residual = ",
            trajectory_stationary.residual,
            "; path means = ",trajectory_stationary.trajectory_count)
    println("weak-PI trajectory dimension = ",weak_dimension,
            "; density PI dimension = ",length(basis))
    println("weak-PI stationary state error = ",weak_trajectory_state_error,
            "; HS standard error = ",
            weak_trajectory_stationary.standard_error)
    println("weak-PI stationary Cxx = ",weak_trajectory_cxx,
            "; direct-reference error = ",weak_trajectory_cxx_error,
            "; spin negativity = ",weak_trajectory_negativity)
    println("weak-PI top-level population = ",weak_trajectory_top,
            "; direct-reference error = ",weak_trajectory_top_error,
            "; negativity error = ",weak_trajectory_negativity_error)
    println("weak-PI stationary residual = ",
            weak_trajectory_stationary.residual,
            "; weak/density estimate distance = ",
            trajectory_method_difference)
    println("maximum scan residual = ",maximum(residual_map))
    println("maximum scan top-level population = ",maximum(top_map))
    println("spin-only negativity range = ",extrema(negativity_map))
    println("Cxx=0 contour guide: kappa/omega_c = ",
            cxx_zero_fit.alpha," (J/omega_c)^2; crossings = ",
            cxx_zero_fit.point_count,"; relative L2 residual = ",
            cxx_zero_fit.relative_l2_residual,
            "; maximum relative residual = ",
            cxx_zero_fit.maximum_relative_residual,
            "; boundary sides = ",
            Tuple(cxx_zero_fit.boundary_sides))
    if cxx_zero_fit.fallback_triggered
        if cxx_zero_fit.general_quadratic===nothing
            println("Cxx=0 general-quadratic fallback unavailable: ",
                    cxx_zero_fit.fallback_status)
        else
            general=cxx_zero_fit.general_quadratic
            println("Cxx=0 general-quadratic fallback coefficients = ",
                    general.coefficients,"; relative L2 residual = ",
                    general.relative_l2_residual,"; status = ",
                    cxx_zero_fit.fallback_status)
        end
    end
    println("parity-selected x-GHZ fidelity range = ",extrema(ghz_map))
    println("longitudinal strong-parity reduction = ",
            ghz_reduced_dimension," / ",length(basis)," PI coordinates; charge = ",
            ghz_charge)
    println("combined spin-flip/mode-parity weak-symmetry residual = ",
            combined_weak_symmetry.residual)
    println("F_GHZ=0.5 contour guide: kappa/omega_c = ",
            ghz_half_fit.alpha," (J/omega_c)^2; crossings = ",
            ghz_half_fit.point_count,"; relative L2 residual = ",
            ghz_half_fit.relative_l2_residual,
            "; maximum relative residual = ",
            ghz_half_fit.maximum_relative_residual,
            "; boundary sides = ",
            Tuple(ghz_half_fit.boundary_sides))
    if ghz_half_fit.power_law!==nothing
        power=ghz_half_fit.power_law
        println("F_GHZ=0.5 power-law fallback: kappa/omega_c = " *
                "alpha_p*(J/omega_c)^beta; (alpha_p,beta) = ",
                (power.alpha,power.beta),
                "; relative L2 residual = ",power.relative_l2_residual,
                "; fallback status = ",ghz_half_fit.fallback_status)
    elseif ghz_half_fit.fallback_triggered
        println("F_GHZ=0.5 power-law fallback unavailable: ",
                ghz_half_fit.fallback_status)
    end
    println("maximum x-GHZ stationary residual = ",maximum(ghz_residual_map),
            "; maximum strong-block leakage = ",
            maximum(ghz_invariance_leakage_map))
    println("reduced steady/long-time propagation errors: state = ",
            ghz_propagation_state_error,", fidelity = ",
            ghz_propagation_fidelity_error,
            "; half-time witness error = ",ghz_time_error,
            "; validation propagation applications = ",
            propagated_reference.operator_applications)
    println("cutoff dimensions (nmax=1,2) = ",
            (cutoff_1.dimension,cutoff_2.dimension))
    println("cutoff Cxx difference = ",cutoff_error,
            "; wider top-level population = ",maximum(cutoff_2.top))

    (;N,nmax,pi_dimension=length(basis),weak_pi_dimension=weak_dimension,
      omega_c,gamma,J_dynamic,
      dynamics_times,dynamics,
      J_values,kappa_values,dimensionless_J,dimensionless_kappa,
      cxx_map,negativity_map,top_map,residual_map,cxx_zero_fit,
      cxx_zero_general_fit=cxx_zero_fit.general_quadratic,
      cxx_zero_display_model=cxx_zero_fit.display_model,
      ghz_settling_time,ghz_map,ghz_residual_map,ghz_parity_map,
      ghz_invariance_leakage_map,ghz_time_error,
      ghz_initial_parity,ghz_charge,ghz_initial_leakage,
      ghz_reduced_dimension,spin_parity_symmetry,combined_weak_symmetry,
      ghz_reference,propagated_reference,half_time_reference,
      ghz_propagation_fidelity_error,ghz_propagation_state_error,
      ghz_half_fit,ghz_half_power_law_fit=ghz_half_fit.power_law,
      ghz_half_display_model=ghz_half_fit.display_model,
      contour_fit_fallback_threshold=
          CONTOUR_FIT_FALLBACK_RELATIVE_L2_THRESHOLD,
      cutoff_times,cutoff_1,cutoff_2,dynamics_krylov_error,
      solver_agreement,cutoff_error,trajectory_stationary,
      trajectory_cxx,trajectory_top,trajectory_negativity,
      trajectory_state_error,trajectory_cxx_error,trajectory_top_error,
      trajectory_negativity_error,weak_trajectory_stationary,
      weak_trajectory_cxx,weak_trajectory_top,weak_trajectory_negativity,
      weak_trajectory_state_error,weak_trajectory_cxx_error,
      weak_trajectory_top_error,weak_trajectory_negativity_error,
      trajectory_method_difference)
end

results=main()
boundary_data_paths=save_fitted_boundary_data(results)
results=merge(results,(;boundary_data_paths))

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
        correlation_axis,results.dimensionless_J,
        results.dimensionless_kappa,results.cxx_map;
        colormap=:balance,colorrange=(-correlation_limit,correlation_limit))
    if minimum(results.cxx_map)<0<maximum(results.cxx_map)
        M.contour!(correlation_axis,
            results.dimensionless_J,
            results.dimensionless_kappa,results.cxx_map;
            levels=[0.0],color=:black,linewidth=2)
    end
    M.scatter!(correlation_axis,
        results.cxx_zero_fit.crossing_x,results.cxx_zero_fit.crossing_y;
        marker=:circle,markersize=9,color=:white,
        strokecolor=:black,strokewidth=1.5)
    if results.cxx_zero_display_model===:origin_constrained
        M.lines!(correlation_axis,
            results.cxx_zero_fit.fit_x,results.cxx_zero_fit.fit_y;
            color=:gold,linewidth=2.8,linestyle=:dash,
            label="κ/ωc = α(J/ωc)²: α=$(round(results.cxx_zero_fit.alpha;sigdigits=4)), " *
                  "rel. residual=$(round(results.cxx_zero_fit.relative_l2_residual;sigdigits=2))")
        M.axislegend(correlation_axis;position=:lt,labelsize=11)
    elseif results.cxx_zero_display_model===:general_quadratic
        fit=results.cxx_zero_general_fit
        coefficients=fit.coefficients
        M.lines!(correlation_axis,fit.fit_x,fit.fit_y;
            color=:darkorange,linewidth=2.8,linestyle=:dashdot,
            label="general quadratic fallback: " *
                  "a=$(round(coefficients.quadratic;sigdigits=3)), " *
                  "b=$(round(coefficients.linear;sigdigits=3)), " *
                  "c=$(round(coefficients.constant;sigdigits=3)), " *
                  "rel. residual=$(round(fit.relative_l2_residual;sigdigits=2))")
        M.axislegend(correlation_axis;position=:lt,labelsize=10)
    end
    M.Colorbar(figure[1,3],correlation_plot;label="Cxx")
    M.scatter!(correlation_axis,
        [results.J_dynamic/results.omega_c],[2.0/results.omega_c];
        marker=:star5,markersize=28,color=[results.weak_trajectory_cxx],
        colormap=:balance,
        colorrange=(-correlation_limit,correlation_limit),
        strokecolor=:black,strokewidth=2)

    negativity_limit=max(maximum(results.negativity_map),eps(Float64))
    negativity_plot=M.heatmap!(
        negativity_axis,results.J_values ./ results.omega_c,
        results.kappa_values ./ results.omega_c,results.negativity_map;
        colormap=:magma,colorrange=(0,negativity_limit))
    M.Colorbar(figure[1,5],negativity_plot;label="negativity")
    M.scatter!(negativity_axis,
        [results.J_dynamic/results.omega_c],[2.0/results.omega_c];
        marker=:star5,markersize=28,
        color=[results.weak_trajectory_negativity],
        colormap=:magma,colorrange=(0,negativity_limit),
        strokecolor=:black,strokewidth=2)
    M.Label(
        figure[2,1:5],
        "All-to-all PI analogue: Jpair = J/(N−1), N=$(results.N), nmax=$(results.nmax). " *
        "Stars: $(results.weak_trajectory_stationary.trajectory_count) weak-PI paths, " *
        "HS SE=$(round(results.weak_trajectory_stationary.standard_error;sigdigits=2)), " *
        "Cxx=$(round(results.weak_trajectory_cxx;sigdigits=3)), " *
        "negativity=$(round(results.weak_trajectory_negativity;sigdigits=3)); " *
        "no spatial correlation length is defined.";
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

    ghz_figure=M.Figure(size=(900,600),fontsize=17)
    ghz_axis=M.Axis(
        ghz_figure[1,1];xlabel="J / ωc",ylabel="κ / ωc",
        yscale=log10,
        title="Parity-selected spin x-GHZ witness, L = σz")
    ghz_plot=M.heatmap!(
        ghz_axis,results.dimensionless_J,
        results.dimensionless_kappa,results.ghz_map;
        colormap=:RdBu,colorrange=(0,1))
    if minimum(results.ghz_map)<0.5<maximum(results.ghz_map)
        M.contour!(
            ghz_axis,results.dimensionless_J,
            results.dimensionless_kappa,results.ghz_map;
            levels=[0.5],color=:black,linewidth=2.2)
    end
    M.scatter!(ghz_axis,
        results.ghz_half_fit.crossing_x,results.ghz_half_fit.crossing_y;
        marker=:circle,markersize=9,color=:white,
        strokecolor=:black,strokewidth=1.5)
    if results.ghz_half_display_model===:origin_constrained
        M.lines!(ghz_axis,
            results.ghz_half_fit.fit_x,results.ghz_half_fit.fit_y;
            color=:gold,linewidth=2.8,linestyle=:dash)
    elseif results.ghz_half_display_model===:power_law
        M.lines!(ghz_axis,
            results.ghz_half_power_law_fit.fit_x,
            results.ghz_half_power_law_fit.fit_y;
            color=:darkorange,linewidth=2.8,linestyle=:dashdot)
    end
    M.Colorbar(ghz_figure[1,2],ghz_plot;
               label="maximized spin x-GHZ fidelity")
    ghz_fit_note=if results.ghz_half_display_model===:power_law
        power=results.ghz_half_power_law_fit
        "Origin αx² residual = " *
        string(round(results.ghz_half_fit.relative_l2_residual;sigdigits=3)) *
        " > $(results.contour_fit_fallback_threshold).\n" *
        "Dash-dot empirical power law (no critical-exponent claim): " *
        "α=$(round(power.alpha;sigdigits=4)), " *
        "β=$(round(power.beta;sigdigits=4)), residual=" *
        string(round(power.relative_l2_residual;sigdigits=3))
    elseif results.ghz_half_display_model===:origin_constrained
        "Dashed origin-constrained guide: α = " *
        string(round(results.ghz_half_fit.alpha;sigdigits=4)) *
        ", relative L2 residual = " *
        string(round(results.ghz_half_fit.relative_l2_residual;sigdigits=3))
    else
        "No fitted guide displayed; power-law fallback status = " *
        string(results.ghz_half_fit.fallback_status)
    end
    M.Label(
        ghz_figure[2,1:2],
        "Spin-parity block $(results.ghz_reduced_dimension)/$(results.pi_dimension) PI coordinates; " *
        "max ‖ℒρ‖ = " *
        string(round(maximum(results.ghz_residual_map);sigdigits=3)) *
        ", propagation check ‖Δρ‖ = " *
        string(round(results.ghz_propagation_state_error;sigdigits=3)) *
        "\n" * ghz_fit_note;
        fontsize=12.5,color=:gray35,tellwidth=false)
    save_example_figure(
        ghz_figure,"debecker2026_all_to_all_ising_pseudomodes_ghz")
end
