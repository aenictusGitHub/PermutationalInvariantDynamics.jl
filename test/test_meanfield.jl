const _MF_PID = PermutationalInvariantDynamics

# Exact finite-N product-state closure evaluated through the independent PI
# Liouvillian. The full Schur basis is essential for a generic mixed iid input;
# `iid_state` rejects a restricted basis that omits occupied sectors rather
# than projecting and renormalizing it.
function _mf_pi_product_rhs(model::PIModel, sigma::AbstractMatrix;
                            time=0.0, parameters=nothing)
    basis = model.basis
    rho = iid_state(basis, sigma)
    L = liouvillian(model; representation=:matrixfree)
    drho_data = similar(rho.data)
    L.action!(drho_data, rho.data, time, parameters)
    drho = PIState(basis, drho_data)
    geometry = OneBodyGeometry(basis)
    out = zeros(promote_type(eltype(sigma), eltype(drho_data)), basis.d, basis.d)
    for i in 1:basis.d, j in 1:basis.d
        Eij = zeros(eltype(out), basis.d, basis.d)
        Eij[i, j] = 1
        out[j, i] = collective_expectation(drho, Eij; cache=geometry) / basis.N
    end
    out
end

function _mf_full_rank_state(d::Integer)
    A = ComplexF64[complex(1 + i + 2j, i - j) for i in 1:d, j in 1:d]
    rho = A * A' + Matrix{ComplexF64}(I, d, d)
    rho / tr(rho)
end

@testset "finite-N mean-field product closure" begin
    sx = ComplexF64[0 1; 1 0]
    sy = ComplexF64[0 -im; im 0]
    sz = ComplexF64[1 0; 0 -1]
    sm = ComplexF64[0 1; 0 0]
    sigma = ComplexF64[0.63 0.11-0.07im; 0.11+0.07im 0.37]

    # Exercise every optimized one- and two-particle lowering rule together.
    # Using a full basis makes the PI reference exactly sigma^otimes N.
    basis = PIBasis(4, 2)
    pair_jump = kron(sm, sm) + 0.13kron(sz, sz)
    terms = [
        LocalHamiltonian(sz; rate=0.17),
        CollectiveHamiltonian(sx; rate=-0.23),
        LocalJump(sm; rate=0.31),
        CollectiveJump(sm; rate=0.07),
        PBodyHamiltonian(kron(sz, sz), 2; rate=0.11),
        LocalPBodyJump(pair_jump, 2; rate=0.05),
    ]
    model = PIModel(basis, terms)
    plan = MeanFieldPlan(model; limit=:finite)
    direct_plan = MeanFieldPlan(4, 2, terms; limit=:finite)
    workspace = MeanFieldWorkspace(plan, sigma)
    du = similar(sigma)
    meanfield_rhs!(du, plan, sigma, 0.0, nothing, workspace)
    reference = _mf_pi_product_rhs(model, sigma)

    @test du ≈ reference atol=3e-10 rtol=3e-10
    @test meanfield_rhs(plan, sigma) ≈ reference atol=3e-10 rtol=3e-10
    @test meanfield_rhs(direct_plan, sigma) ≈ reference atol=3e-10 rtol=3e-10
    @test abs(tr(du)) < 2e-12
    @test norm(du - du') < 2e-12

    # The prepared in-place path is intended for ODE hot loops.  Keep the
    # threshold tolerant of Julia-version keyword/closure bookkeeping while
    # still rejecting tensor temporaries on every call.
    meanfield_rhs!(du, plan, sigma, 0.0, nothing, workspace)
    @test (@allocated meanfield_rhs!(du, plan, sigma, 0.0, nothing, workspace)) <= 4096
    @test_throws ArgumentError meanfield_rhs!(sigma, plan, sigma, 0.0, nothing,
                                               workspace)

    # A qutrit check catches accidental qubit/Bloch-vector assumptions and
    # tensor-index transpositions in the p-body contractions.
    basis3 = PIBasis(2, 3)
    sigma3 = _mf_full_rank_state(3)
    h3 = ComplexF64[0 1 0; 1 0 0.2im; 0 -0.2im 0.4]
    l3 = ComplexF64[0 1 0; 0 0 0.7; 0 0 0]
    d3 = Diagonal(ComplexF64[-0.4, 0.2, 0.8])
    model3 = PIModel(basis3, [
        LocalHamiltonian(h3; rate=0.19),
        LocalJump(l3; rate=0.27),
        CollectiveJump(l3; rate=0.04),
        PBodyHamiltonian(kron(d3, d3), 2; rate=-0.08),
        LocalPBodyJump(kron(l3, l3), 2; rate=0.03),
    ])
    plan3 = MeanFieldPlan(model3; limit=:finite)
    @test meanfield_rhs(plan3, sigma3) ≈
          _mf_pi_product_rhs(model3, sigma3) atol=8e-10 rtol=8e-10

    # Direct construction is independent of Schur geometry and must remain
    # usable at sizes for which constructing a full PIBasis would defeat the
    # purpose of a mean-field calculation.
    Nlarge = 100_000_000
    large_terms = [
        LocalJump(sm; rate=0.2),
        CollectiveJump(sm; rate=0.7/Nlarge),
        PBodyHamiltonian(kron(sz, sz), 2; rate=0.3/Nlarge),
    ]
    large_plan = MeanFieldPlan(Nlarge, 2, large_terms; limit=:finite)
    large_rhs = meanfield_rhs(large_plan, sigma)
    @test all(isfinite, large_rhs)
    @test abs(tr(large_rhs)) < 2e-12

    # Combinatorial factors are converted to the chosen working precision at
    # plan construction, rather than promoting Float32 hot loops to BigFloat.
    sigma32 = ComplexF32.(sigma)
    sz32 = ComplexF32.(sz)
    plan32 = MeanFieldPlan(Nlarge, 2,
        [PBodyHamiltonian(kron(sz32, sz32), 2; rate=1f0/Nlarge)];
        limit=:finite)
    @test eltype(meanfield_rhs(plan32, sigma32)) == ComplexF32
    work32 = MeanFieldWorkspace(plan32, sigma32)
    @test_throws ArgumentError meanfield_rhs!(similar(sigma), plan32, sigma,
                                               0.0, nothing, work32)

    driven32 = MeanFieldPlan(2, 2,
        [LocalJump(ComplexF32.(sm); rate=(t,p)->1.0f0)])
    driven32_work = MeanFieldWorkspace(driven32, sigma32)
    driven32_out = similar(sigma32)
    meanfield_rhs!(driven32_out, driven32, sigma32, 0.0, nothing, driven32_work)
    incompatible_rate = MeanFieldPlan(2, 2,
        [LocalJump(ComplexF32.(sm); rate=(t,p)->1.0)])
    @test_throws ArgumentError meanfield_rhs!(driven32_out, incompatible_rate,
        sigma32, 0.0, nothing, MeanFieldWorkspace(incompatible_rate, sigma32))

    sigma_big = Complex{BigFloat}.(sigma)
    sm_big = Complex{BigFloat}.(sm)
    plan_big = MeanFieldPlan(3, 2, [LocalJump(sm_big; rate=big"0.2")])
    @test eltype(meanfield_rhs(plan_big, sigma_big)) == Complex{BigFloat}
end

@testset "collective p-body mean-field overlap closure" begin
    sm = ComplexF64[0 1; 0 0]
    sz = ComplexF64[1 0; 0 -1]
    sigma = ComplexF64[0.59 0.13-0.08im; 0.13+0.08im 0.41]

    # The finite closure sums every ordered pair of subsets through its overlap
    # class.  Compare against the independent Appendix-D PI Liouvillian, which
    # does not use any mean-field overlap formulas.
    N = 4
    basis = PIBasis(N, 2)
    pair_jump = kron(sm, sm) + 0.17kron(sz, sz)
    pair_model = PIModel(basis,
        [CollectivePBodyJump(pair_jump, 2; rate=0.037)])
    pair_plan = MeanFieldPlan(pair_model; limit=:finite)
    pair_rhs = meanfield_rhs(pair_plan, sigma)
    @test pair_rhs ≈ _mf_pi_product_rhs(pair_model, sigma) atol=5e-10 rtol=5e-10
    @test abs(tr(pair_rhs)) < 3e-12
    @test norm(pair_rhs-pair_rhs') < 3e-12

    # At p=N the coherent subset sum contains one term, so collective and local
    # p-body dissipators have identical finite product derivatives.
    only_subset_collective = MeanFieldPlan(2, 2,
        [CollectivePBodyJump(kron(sm, sm), 2; rate=0.21)])
    only_subset_local = MeanFieldPlan(2, 2,
        [LocalPBodyJump(kron(sm, sm), 2; rate=0.21)])
    @test meanfield_rhs(only_subset_collective, sigma) ≈
          meanfield_rhs(only_subset_local, sigma) atol=2e-12 rtol=2e-12

    # N=6 permits every ordered three-subset overlap r=0,1,2,3.
    triple_jump = kron(sm, kron(sm, sm))
    triple_basis = PIBasis(6, 2)
    triple_model = PIModel(triple_basis,
        [CollectivePBodyJump(triple_jump, 3; rate=0.006)])
    @test meanfield_rhs(MeanFieldPlan(triple_model), sigma) ≈
          _mf_pi_product_rhs(triple_model, sigma) atol=8e-10 rtol=8e-10

    # The thermodynamic rule retains only the disjoint ordered-pair class.  For
    # p=2 its coefficient is N^3/2 and its field is determined by E_1 and
    # ell=tr(L sigma^tensor2).  Build both independently here.
    Nthermo = 23
    rate = 0.004
    thermo = MeanFieldPlan(Nthermo, 2,
        [CollectivePBodyJump(pair_jump, 2; rate=rate)];
        limit=:thermodynamic)
    effective = zeros(ComplexF64, 2, 2)
    @inbounds for col in 1:2,row in 1:2,environment_row in 1:2,
                  environment_col in 1:2
        effective[row,col] +=
            pair_jump[row+2(environment_row-1),col+2(environment_col-1)] *
            sigma[environment_col,environment_row]
    end
    ell = tr(pair_jump*kron(sigma,sigma))
    leading_count = Nthermo^3/2
    expected_thermo = rate*leading_count/2 .* (
        conj(ell).*(effective*sigma-sigma*effective) -
        ell.*(effective'*sigma-sigma*effective'))
    @test meanfield_rhs(thermo, sigma) ≈ expected_thermo atol=3e-11 rtol=3e-11

    # The same independent leading-disjoint-class formula for p=3 checks the
    # N^(2p-1)/((p-1)!p!) thermodynamic combinatorics.
    triple_thermo = MeanFieldPlan(Nthermo,2,
        [CollectivePBodyJump(triple_jump,3;rate=rate)];
        limit=:thermodynamic)
    triple_effective=zeros(ComplexF64,2,2)
    @inbounds for col in 1:2,row in 1:2,e1row in 1:2,e1col in 1:2,
                  e2row in 1:2,e2col in 1:2
        local_row=row+2(e1row-1)+4(e2row-1)
        local_col=col+2(e1col-1)+4(e2col-1)
        triple_effective[row,col]+=triple_jump[local_row,local_col]*
            sigma[e1col,e1row]*sigma[e2col,e2row]
    end
    triple_ell=tr(triple_jump*kron(sigma,kron(sigma,sigma)))
    triple_count=Nthermo^5/12
    expected_triple=rate*triple_count/2 .* (
        conj(triple_ell).*(triple_effective*sigma-sigma*triple_effective) -
        triple_ell.*(triple_effective'*sigma-sigma*triple_effective'))
    @test meanfield_rhs(triple_thermo,sigma)≈expected_triple atol=2e-9 rtol=2e-11

    # Effective operators and overlap contractions are caller-workspace owned;
    # no tensor-product or union-support matrices are rebuilt in the hot path.
    workspace = MeanFieldWorkspace(pair_plan, sigma)
    destination = similar(sigma)
    meanfield_rhs!(destination,pair_plan,sigma,0.0,nothing,workspace)
    @test (@allocated meanfield_rhs!(destination,pair_plan,sigma,0.0,nothing,
                                     workspace)) <= 4096

    # Small qudit sanity check: no qubit/Bloch identities enter the overlap
    # contraction or subset counts.
    basis3 = PIBasis(3, 3)
    sigma3 = _mf_full_rank_state(3)
    l3 = ComplexF64[0 1 0; 0 0 0.4; 0 0 0]
    qutrit_model = PIModel(basis3,
        [CollectivePBodyJump(kron(l3,l3), 2; rate=0.002)])
    @test meanfield_rhs(MeanFieldPlan(qutrit_model),sigma3) ≈
          _mf_pi_product_rhs(qutrit_model,sigma3) atol=2e-9 rtol=2e-9
end

@testset "mean-field time dependence and local analytical dynamics" begin
    sm = ComplexF64[0 1; 0 0]
    basis = PIBasis(3, 2)
    sigma0 = ComplexF64[0.27 0.08+0.04im; 0.08-0.04im 0.73]

    rate = (t, p) -> p.gamma * (1 + t)
    driven = PIModel(basis, [LocalJump(sm; rate=rate)])
    plan = MeanFieldPlan(driven; limit=:finite)
    parameters = (gamma=0.6,)
    time = 0.37
    reference_model = PIModel(basis, [LocalJump(sm; rate=rate(time, parameters))])
    @test meanfield_rhs(plan, sigma0; time=time, parameters=parameters) ≈
          _mf_pi_product_rhs(reference_model, sigma0) atol=2e-11 rtol=2e-11

    gamma = 0.7
    autonomous = MeanFieldPlan(PIModel(basis, [LocalJump(sm; rate=gamma)]);
                               limit=:finite)
    tf = 0.8
    excited = sigma0[2, 2] * exp(-gamma * tf)
    coherence = sigma0[1, 2] * exp(-gamma * tf / 2)
    exact = ComplexF64[1-excited coherence; conj(coherence) excited]

    destination = similar(sigma0)
    work = MeanFieldWorkspace(autonomous, sigma0)
    meanfield_evolve!(destination, autonomous, sigma0, (0.0, tf);
                      steps=800, workspace=work)
    @test destination ≈ exact atol=3e-11 rtol=3e-11
    @test_throws ArgumentError meanfield_evolve!(
        Matrix{Complex{BigFloat}}(sigma0), autonomous, sigma0, (0.0, tf);
        steps=2, workspace=work)

    solution = solve_meanfield(autonomous, sigma0, (0.0, tf);
                               saveat=range(0.0, tf; length=5),
                               steps_per_interval=200)
    @test first(solution.states) ≈ sigma0 atol=1e-14
    @test last(solution.states) ≈ exact atol=3e-11 rtol=3e-11

    problem = meanfield_problem(plan, sigma0, (0.0, tf); parameters=parameters)
    problem_du = similar(sigma0)
    problem.f(problem_du, sigma0, problem.p, time)
    @test problem_du ≈
          meanfield_rhs(plan, sigma0; time=time, parameters=parameters) atol=2e-12
end

@testset "mean-field limit, observables, stability, and validation" begin
    sx = ComplexF64[0 1; 1 0]
    sy = ComplexF64[0 -im; im 0]
    sz = ComplexF64[1 0; 0 -1]
    sm = ComplexF64[0 1; 0 0]
    sigma = ComplexF64[0.61 0.12-0.09im; 0.12+0.09im 0.39]
    N = 7
    kappa = 0.8
    finite_model = PIModel(PIBasis(N, 2), [CollectiveJump(sm; rate=kappa/N)])
    finite = MeanFieldPlan(finite_model; limit=:finite)
    thermodynamic = MeanFieldPlan(finite_model; limit=:thermodynamic)

    alpha = tr(sm * sigma)
    local_dissipator = sm*sigma*sm' - (sm'*sm*sigma + sigma*sm'*sm)/2
    cross = (conj(alpha)*(sm*sigma-sigma*sm) -
             alpha*(sm'*sigma-sigma*sm'))/2
    @test meanfield_rhs(finite, sigma) ≈
          (kappa/N) .* (local_dissipator + (N-1).*cross) atol=2e-12
    @test meanfield_rhs(thermodynamic, sigma) ≈ kappa .* cross atol=2e-12

    # Thermodynamic p-body mode changes only the subset combinatorics; it does
    # not infer a Kac factor from the numerical rate.
    pair_h = kron(sz, sz)
    finite_pair = MeanFieldPlan(N, 2, [PBodyHamiltonian(pair_h, 2; rate=1/N)];
                                limit=:finite)
    thermo_pair = MeanFieldPlan(N, 2, [PBodyHamiltonian(pair_h, 2; rate=1/N)];
                                limit=:thermodynamic)
    finite_pair_rhs = meanfield_rhs(finite_pair, sigma)
    thermo_pair_rhs = meanfield_rhs(thermo_pair, sigma)
    @test thermo_pair_rhs ≈ (N/(N-1)) .* finite_pair_rhs atol=2e-12

    triple_h = kron(sz, kron(sz, sz))
    finite_triple = MeanFieldPlan(N, 2,
        [PBodyHamiltonian(triple_h, 3; rate=1/N^2)]; limit=:finite)
    thermo_triple = MeanFieldPlan(N, 2,
        [PBodyHamiltonian(triple_h, 3; rate=1/N^2)]; limit=:thermodynamic)
    @test meanfield_rhs(thermo_triple, sigma) ≈
          (N^2/((N-1)*(N-2))) .* meanfield_rhs(finite_triple, sigma) atol=2e-12

    # Product-state observable identities, checked independently against PI
    # contractions for a full-rank iid state.
    rho = iid_state(finite_model.basis, sigma)
    one_mean = tr(sigma * sx)
    one_second = tr(sigma * (sx*sx))
    moments = meanfield_collective_moments(sigma, sx, N)
    @test meanfield_expectation(sigma, sx) ≈ one_mean atol=2e-13
    @test moments.mean ≈ N*one_mean atol=2e-13
    @test moments.second_moment ≈
          N*one_second + N*(N-1)*one_mean^2 atol=2e-12
    @test moments.mean ≈ collective_moments(rho, sx).mean atol=2e-10
    @test moments.second_moment ≈
          collective_moments(rho, sx).second_moment atol=2e-10

    huge_moments = meanfield_collective_moments(10^12,
        ComplexF64[0.75 0; 0 0.25], sz)
    @test huge_moments.variance ≈ 0.75e12 rtol=2e-15
    @test isfinite(real(huge_moments.second_moment))

    nonexact32_N=Int(2^24+1)
    nonexact32=meanfield_collective_moments(
        nonexact32_N,ComplexF32[1 0;0 0],ComplexF32[1 0;0 0])
    expected_pair32=Float32(
        big(nonexact32_N)*big(nonexact32_N-1))
    @test nonexact32.mean===ComplexF32(Float32(nonexact32_N))
    @test nonexact32.second_moment===
          ComplexF32(Float32(nonexact32_N)+expected_pair32)
    @test nonexact32.variance===0.0f0+0.0f0im

    pair = kron(sz, sz) + 0.2kron(sx, sx)
    pair_prediction = meanfield_pbody_expectation(sigma, pair, 2, N)
    pair_operator = pbody_collective_operator(finite_model.basis, pair, 2)
    @test pair_prediction ≈ binomial(N, 2)*tr(pair*kron(sigma, sigma)) atol=2e-12
    @test pair_prediction ≈ expectation(rho, pair_operator) atol=3e-10
    @test_throws ArgumentError meanfield_pbody_expectation(
        sigma, kron(sz, sx), 2, N)

    sigma16=ComplexF16[1 0;0 0]
    identity_pair16=Matrix{ComplexF16}(I,4,4)
    @test_throws ArgumentError meanfield_pbody_expectation(
        363,sigma16,identity_pair16,2)
    count1000=exact_binomial(1000,2)
    inverse1000=Float16(inv(BigFloat(count1000)))
    @test meanfield_pbody_expectation(
        1000,sigma16,inverse1000.*identity_pair16,2)≈
        Float16(BigFloat(count1000)*inverse1000) rtol=Float16(3e-2)
    normalized_identity=Matrix{ComplexF16}(I,2,2)./Float16(1000)
    normalized_moments=meanfield_collective_moments(
        1000,sigma16,normalized_identity)
    @test normalized_moments.mean≈Float16(1) rtol=Float16(3e-3)
    @test normalized_moments.second_moment≈Float16(1) rtol=Float16(4e-3)
    @test abs(normalized_moments.variance)<=Float16(2e-3)

    # Local pumping and loss have an interior analytical fixed point and a
    # nonsingular traceless-Hermitian Jacobian.
    down = 0.8
    up = 0.2
    thermal = MeanFieldPlan(PIModel(PIBasis(2, 2),
        [LocalJump(sm; rate=down), LocalJump(sm'; rate=up)]); limit=:finite)
    fixed = ComplexF64[down/(down+up) 0; 0 up/(down+up)]
    @test norm(meanfield_rhs(thermal, fixed)) < 2e-13
    jacobian = meanfield_jacobian(thermal, fixed)
    @test size(jacobian) == (3, 3)
    @test sort(real.(eigvals(jacobian))) ≈ [-1.0, -0.5, -0.5] atol=2e-6
    stability = meanfield_stability(thermal, fixed)
    @test stability.stable

    stationary = meanfield_stationary_state(thermal, Matrix{ComplexF64}(I, 2, 2)/2)
    stationary_state = hasproperty(stationary, :state) ? stationary.state : stationary
    @test stationary_state ≈ fixed atol=2e-8

    direct_h = DirectPIHamiltonian(collective_operator(finite_model.basis, sx))
    direct_l = DirectPIJump(collective_operator(finite_model.basis, sm))
    @test_throws ArgumentError MeanFieldPlan(PIModel(finite_model.basis, [direct_h]))
    @test_throws ArgumentError MeanFieldPlan(PIModel(finite_model.basis, [direct_l]))
    operator_driven = LocalJump((t, p) -> sm; rate=1.0)
    @test_throws ArgumentError MeanFieldPlan(PIModel(finite_model.basis,
                                                     [operator_driven]))
    @test !isautonomous(MeanFieldPlan(PIModel(finite_model.basis,
                                              [LocalJump(sm; rate=(t,p)->1.0)])))
    @test_throws ArgumentError meanfield_stationary_state(
        MeanFieldPlan(PIModel(finite_model.basis,
                              [LocalJump(sm; rate=(t,p)->1.0)])), sigma)
    @test_throws DimensionMismatch meanfield_expectation(zeros(2, 3), zeros(2, 3))
    @test_throws ArgumentError MeanFieldPlan(2, 2, [sm])
    @test_throws ArgumentError MeanFieldPlan(finite_model; limit=:invalid)
end
