module PaperModels

using LinearAlgebra
using PermutationalInvariantDynamics

export spin_matrices, correlated_superradiance_model, correlated_superradiance_intensity_operator,
       two_qubit_correlated_superradiance_intensity_exact, dissipative_collective_spin_pairing_model, one_axis_twisting_model,
       one_axis_twisting_mean_spin_exact, independent_dephasing_model,
       independent_dephasing_coherence_exact, local_pump_decay_model,
       local_pump_decay_steady_state, collective_local_decay_model,
       collective_local_radiation_operators
export cooperative_fluorescence_model, cooperative_fluorescence_exact_state,
       steady_superradiance_model, boundary_time_crystal_model,
       balanced_gain_loss_time_crystal_model, balanced_gain_loss_spectrum,
       interacting_boundary_time_crystal_model,
       local_pseudomode_operators,
       all_to_all_xx_spin_local_pseudomode_model

"""PRA 94, 033838 (2016), Eqs. (3)-(5), with `delta_gamma=gamma0-gamma`."""
function correlated_superradiance_model(N;gamma0=1.0,gamma=gamma0)
    0<=gamma<=gamma0 || throw(ArgumentError("the paper assumes 0 <= gamma <= gamma0"))
    b=PIBasis(N,2); sm=ComplexF64[0 1;0 0]
    PIModel(b,[CollectiveJump(sm;rate=gamma),LocalJump(sm;rate=gamma0-gamma)])
end

"""Operator whose expectation is the radiated rate, paper Eq. (39)."""
function correlated_superradiance_intensity_operator(b;gamma0=1.0,gamma=gamma0)
    sm=ComplexF64[0 1;0 0]; Jm=collective_operator(b,sm)
    gamma*(adjoint(Jm)*Jm)+(gamma0-gamma)*collective_operator(b,sm'*sm)
end

"""Analytical N=2 intensity, Eqs. (41)-(43)."""
function two_qubit_correlated_superradiance_intensity_exact(t;gamma0=1.0,gamma=gamma0)
    dg=gamma0-gamma
    iszero(dg) && return 2gamma0*exp(-2gamma0*t)*(1+2gamma0*t)
    iszero(gamma) && return 2gamma0*exp(-gamma0*t)
    exp(-2*(gamma+dg)*t)/((2gamma+dg)*dg)*((2gamma+dg)^2*dg+dg^2*(2gamma+dg)+
        (2gamma+dg)^3*(exp(dg*t)-1)+dg^3*(exp((2gamma+dg)*t)-1))
end

"""PRA 110, 062208 (2024), Eqs. (2)-(6), with microscopic body order retained."""
function dissipative_collective_spin_pairing_model(N,d;V=1.0,gammaI=0.0,gammaC=0.0,dissipator=:spin)
    N>=1||throw(ArgumentError(
        "the dissipative collective-spin pairing model requires N >= 1"))
    d>=2||throw(ArgumentError(
        "the dissipative collective-spin pairing model requires a nonzero spin (d >= 2)"))
    s=spin_matrices(d); b=PIBasis(N,d)

    # This is exactly V*(Jx^2-Jy^2)/(N*j).  Keeping the one- and two-body
    # pieces supplies MeanFieldPlan with the microscopic provenance that a
    # preassembled DirectPIHamiltonian cannot retain.  The local piece
    # vanishes for qubits, but is needed for higher spins.
    h1=s.jx*s.jx-s.jy*s.jy
    h2=kron(s.jx,s.jx)-kron(s.jy,s.jy)
    hamiltonian_terms=b.N==1 ?
        (LocalHamiltonian(h1;rate=V/(b.N*s.j)),) :
        (LocalHamiltonian(h1;rate=V/(b.N*s.j)),
         PBodyHamiltonian(h2,2;rate=2V/(b.N*s.j)))
    ell = dissipator===:spin ? s.jm : dissipator===:equal ?
        sqrt(2s.j)*diagm(1=>ones(ComplexF64,b.d-1)) : throw(ArgumentError("dissipator must be :spin or :equal"))
    PIModel(b,(hamiltonian_terms...,
               LocalJump(ell;rate=gammaI/s.j),
               CollectiveJump(ell;rate=gammaC/(b.N*s.j))))
end

"""Kitagawa--Ueda, PRA 47, 5138 (1993): `H=chi*Jz^2`."""
function one_axis_twisting_model(N;chi=1.0)
    b=PIBasis(N,2); sz=ComplexF64[1 0;0 -1]
    Jz=collective_operator(b,sz/2)
    PIModel(b,[DirectPIHamiltonian(chi*(Jz*Jz))])
end
one_axis_twisting_mean_spin_exact(N,t;chi=1.0)=N/2*cos(chi*t)^(N-1)

"""Huelga et al., PRL 79, 3865 (1997): independent Markovian dephasing."""
function independent_dephasing_model(N;gamma=1.0)
    b=PIBasis(N,2); sz=ComplexF64[1 0;0 -1]
    PIModel(b,[LocalJump(sz;rate=gamma/2)])
end
independent_dephasing_coherence_exact(N,t;gamma=1.0)=N/2*exp(-gamma*t)

"""Local pumping and emission benchmark used in Shammah et al., PRA 98, 063815 (2018)."""
function local_pump_decay_model(N;down=1.0,up=0.25)
    b=PIBasis(N,2); sm=ComplexF64[0 1;0 0]
    PIModel(b,[LocalJump(sm;rate=down),LocalJump(sm';rate=up)])
end
function local_pump_decay_steady_state(b;down=1.0,up=0.25)
    iid_state(b,ComplexF64[down 0;0 up]/(down+up))
end

"""
Zhang, Zhang, and Mølmer, NJP 20, 112001 (2018), Eq. (1), restricted to
collective cavity decay and individual free-space decay.

The package convention is `D[L]ρ=LρL†-{L†L,ρ}/2`.  Consequently the
positive rates below are the paper's `GammaC` and `gammaL` without an extra
factor of two.
"""
function collective_local_decay_model(N;GammaC=1.0,gammaL=zero(GammaC))
    N>=1||throw(ArgumentError(
        "the collective/local-decay model requires N >= 1"))
    GammaC>=0||throw(ArgumentError("GammaC must be nonnegative"))
    gammaL>=0||throw(ArgumentError("gammaL must be nonnegative"))
    R=promote_type(typeof(float(GammaC)),typeof(float(gammaL)))
    b=PIBasis(N,2);sm=spin_matrices(2;T=R).jm
    PIModel(b,(CollectiveJump(sm;rate=GammaC),
               LocalJump(sm;rate=gammaL)))
end

"""Cavity and free-space photon-flux operators for the collective/local-decay model."""
function collective_local_radiation_operators(b;GammaC=1.0,gammaL=zero(GammaC))
    b.d==2||throw(ArgumentError("the radiation operators require a qubit basis"))
    GammaC>=0||throw(ArgumentError("GammaC must be nonnegative"))
    gammaL>=0||throw(ArgumentError("gammaL must be nonnegative"))
    R=promote_type(typeof(float(GammaC)),typeof(float(gammaL)))
    sm=spin_matrices(2;T=R).jm
    cache=OneBodyGeometry(b;T=R)
    Jm=collective_operator(b,sm;cache)
    excitation=collective_operator(b,adjoint(sm)*sm;cache)
    (;cavity=GammaC*(adjoint(Jm)*Jm),free_space=gammaL*excitation)
end

"""Morrison--Parkins, PRA 77, 043810 (2008), Eq. (1)."""
function cooperative_fluorescence_model(N;Omega=0.2,gamma=0.3,restricted=true)
    b=restricted ? PIBasis(N,2;sectors=[(N,0)]) : PIBasis(N,2)
    sx=ComplexF64[0 1;1 0];sm=ComplexF64[0 1;0 0]
    PIModel(b,[CollectiveHamiltonian(sx/2;rate=Omega),
               CollectiveJump(sm;rate=2gamma/N)])
end

"""Exact symmetric-sector steady state, Morrison--Parkins Eq. (2)."""
function cooperative_fluorescence_exact_state(b;Omega=0.2,gamma=0.3)
    length(b.sectors)==1&&b.sectors[1]==Partition((b.N,0))||throw(ArgumentError("exact state requires the symmetric-sector basis"))
    sm=ComplexF64[0 1;0 0];Jm=collective_block(b,sm,b.sectors[1]);a=Omega*b.N/(2gamma)
    A=Jm+im*a*I;R=inv(A)*inv(A)';R=(R+R')/2;R./=tr(R)
    sector_density_matrix(b,b.sectors[1],R)
end

"""Meiser--Holland, PRA 81, 033847 (2010), Eq. (1)."""
function steady_superradiance_model(N;GammaC=1.0,pump=1.0)
    b=PIBasis(N,2);sm=ComplexF64[0 1;0 0]
    PIModel(b,[CollectiveJump(sm;rate=GammaC),LocalJump(sm';rate=pump)])
end

"""Iemini et al., PRL 121, 035301 (2018), Eq. (2)."""
function boundary_time_crystal_model(N;omega0=1.5,kappa=1.0)
    b=PIBasis(N,2;sectors=[(N,0)]);sx=ComplexF64[0 1;1 0];sm=ComplexF64[0 1;0 0]
    PIModel(b,[CollectiveHamiltonian(sx/2;rate=omega0),
               CollectiveJump(sm;rate=2kappa/N)])
end

"""
Nakanishi--Sasamoto, PRA 107, L010201 (2023), Eqs. (13)-(14).

The paper uses `D_paper[L] = 2D[L]`, `S=N/2`, and
`Sx^plus/minus = Sy plus/minus i*Sz`.  The returned model uses the package's
standard Lindblad dissipator and is restricted to the conserved spin-`S`
sector.  The dissipative time-crystal point is the balanced case `p=0`.
"""
function balanced_gain_loss_time_crystal_model(N;g=1.3,kappa=0.4,p=0.0)
    N>0||throw(ArgumentError("N must be positive"))
    kappa>=0||throw(ArgumentError("kappa must be nonnegative"))
    -1<=p<=1||throw(ArgumentError("p must lie in [-1,1]"))
    b=PIBasis(N,2;sectors=[(N,0)]);s=spin_matrices(2);S=N/2
    Sxp=s.jy+im*s.jz;Sxm=s.jy-im*s.jz
    PIModel(b,[CollectiveHamiltonian(s.jx;rate=g),
               CollectiveJump(Sxp;rate=2kappa*(1+p)/S),
               CollectiveJump(Sxm;rate=2kappa*(1-p)/S)])
end

"""Exact balanced (`p=0`) Liouvillian eigenvalue multiset, paper Eq. (14)."""
function balanced_gain_loss_spectrum(N;g=1.3,kappa=0.4)
    N>0||throw(ArgumentError("N must be positive"))
    kappa>=0||throw(ArgumentError("kappa must be nonnegative"))
    ComplexF64[im*g*q-(4kappa/N)*(abs(q)+l*(1+l+2abs(q)))
               for q in -N:N for l in 0:N-abs(q)]
end

"""
Piccitto *et al.*, PRB 104, 014307 (2021), Eqs. (1), (2), and (5).

This is the `p=2,q=1` interacting boundary-time-crystal model.  The paper's
magnetizations are `Jalpha=sum_i sigma_i^alpha/N`; the scalar part of
`-N*omega_z*Jz^2` is omitted because it drops out of the commutator.
"""
function interacting_boundary_time_crystal_model(N;omega_z=1.0,omega_x=3.0,
                                             Gamma_up=0.2,Gamma_down=0.0)
    N>=2||throw(ArgumentError("N must be at least two for the p=2 interaction"))
    Gamma_up>=0||throw(ArgumentError("Gamma_up must be nonnegative"))
    Gamma_down>=0||throw(ArgumentError("Gamma_down must be nonnegative"))
    b=PIBasis(N,2;sectors=[(N,0)]);s=spin_matrices(2)
    sx=2s.jx;sz=2s.jz
    PIModel(b,[CollectiveHamiltonian(sx;rate=-omega_x),
               PBodyHamiltonian(kron(sz,sz),2;rate=-2omega_z/N),
               CollectiveJump(s.jp;rate=4Gamma_up/N),
               CollectiveJump(s.jm;rate=4Gamma_down/N)])
end

"""
    local_pseudomode_operators(nmax; T=Float64)

Construct the local matrices for one spin coupled to one pseudomode truncated
to occupations `0:nmax`.  The supersite ordering is `spin tensor mode`, its
dimension is `2(nmax+1)`, and the returned named tuple contains Pauli matrices,
their supersite lifts, the mode annihilation/number/top-level operators, and
the two spin--mode exchange operators associated with `sigma_minus` and
`sigma_z` coupling.
"""
function local_pseudomode_operators(
        nmax::Integer;T::Type{<:AbstractFloat}=Float64)
    nmax>=1||throw(ArgumentError(
        "the pseudomode cutoff must retain at least occupations 0 and 1"))
    nmax_int=try
        Int(nmax)
    catch error
        (error isa InexactError||error isa OverflowError)||rethrow()
        throw(ArgumentError("nmax is too large for an explicit local cutoff"))
    end
    levels=try
        Base.checked_add(nmax_int,1)
    catch error
        error isa OverflowError||rethrow()
        throw(ArgumentError("nmax is too large for an explicit local cutoff"))
    end
    dsite=try
        Base.checked_mul(2,levels)
    catch error
        error isa OverflowError||rethrow()
        throw(ArgumentError("the spin--pseudomode dimension overflows Int"))
    end

    spin=spin_matrices(2;T)
    identity_spin=Matrix{Complex{T}}(I,2,2)
    identity_mode=Matrix{Complex{T}}(I,levels,levels)
    sigma_x=2spin.jx
    sigma_y=2spin.jy
    sigma_z=2spin.jz
    sigma_minus=spin.jm

    annihilation=zeros(Complex{T},levels,levels)
    @inbounds for occupation in 1:levels-1
        annihilation[occupation,occupation+1]=sqrt(T(occupation))
    end
    mode_annihilation=kron(identity_spin,annihilation)
    mode_number=adjoint(mode_annihilation)*mode_annihilation
    mode_top=zeros(Complex{T},levels,levels)
    mode_top[end,end]=one(T)

    spin_paulis=(identity_spin,sigma_x,sigma_y,sigma_z)
    lifted_paulis=map(operator->kron(operator,identity_mode),spin_paulis)
    lowering_site=kron(sigma_minus,identity_mode)
    z_site=lifted_paulis[4]
    exchange_minus=lowering_site*adjoint(mode_annihilation)+
                   adjoint(lowering_site)*mode_annihilation
    exchange_z=z_site*adjoint(mode_annihilation)+
               adjoint(z_site)*mode_annihilation

    (;nmax=nmax_int,levels,dsite,spin_paulis,lifted_paulis,
      x_site=lifted_paulis[2],z_site,lowering_site,
      mode_annihilation,mode_number,
      mode_top=kron(identity_spin,mode_top),
      exchange_minus,exchange_z)
end

"""
    all_to_all_xx_spin_local_pseudomode_model(
        basis, operators; Jpair, omega_c=1, gamma=0.05, kappa=1,
        coupling=:minus)
    all_to_all_xx_spin_local_pseudomode_model(
        N, nmax; Jpair, omega_c=1, gamma=0.05, kappa=1,
        coupling=:minus)

Construct the permutation-invariant all-to-all specialization of the local
pseudomode embedding in Debecker *et al.* (2026).  Each of the `N` identical
spin--pseudomode supersites has the matrices returned by
[`local_pseudomode_operators`](@ref).  The Hamiltonian is

`-Jpair * sum(i<j) X_i*X_j + omega_c * sum_i a_i' a_i +
 sqrt(gamma*kappa) * sum_i (L_i*a_i' + L_i'*a_i)`.

`coupling` is `:minus` or `:z`.  The local mode jump has package rate
`2kappa`, matching the manuscript convention
`D_paper[a] = 2D_package[a]`.  `Jpair` is the literal coefficient of every
unordered pair: no Kac scaling is inserted.  The basis-taking method supports
reuse across scans; the convenience method constructs a complete
`PIBasis(N,2(nmax+1))` with a non-narrowing scalar type inferred from the
parameters.
"""
function all_to_all_xx_spin_local_pseudomode_model(
        basis::PIBasis,operators;Jpair,omega_c=1,gamma=0.05,kappa=1,
        coupling::Symbol=:minus)
    basis.N>=2||throw(ArgumentError(
        "the all-to-all pair interaction requires N >= 2"))
    basis.d==operators.dsite||throw(DimensionMismatch(
        "the PI basis local dimension must equal the spin--pseudomode dimension $(operators.dsite)"))
    all(value->value isa Real&&isfinite(value),
        (Jpair,omega_c,gamma,kappa))||throw(ArgumentError(
        "Jpair, omega_c, gamma, and kappa must be finite real numbers"))
    gamma>=0||throw(ArgumentError("gamma must be nonnegative"))
    kappa>0||throw(ArgumentError("kappa must be positive"))
    exchange = coupling===:minus ? operators.exchange_minus :
        coupling===:z ? operators.exchange_z : throw(ArgumentError(
            "coupling must be :minus or :z"))
    # Taking the square roots before multiplying avoids a spurious overflow
    # when the representable geometric mean has very different factors.
    g=sqrt(gamma)*sqrt(kappa)
    decay_rate=2*float(kappa)
    isfinite(g)||throw(ArgumentError(
        "sqrt(gamma*kappa) is not representable in the parameter precision"))
    isfinite(decay_rate)||throw(ArgumentError(
        "2kappa is not representable in the parameter precision"))
    PIModel(basis,(
        LocalHamiltonian(operators.mode_number;rate=omega_c),
        LocalHamiltonian(exchange;rate=g),
        PBodyHamiltonian(kron(operators.x_site,operators.x_site),2;
                         rate=-Jpair),
        LocalJump(operators.mode_annihilation;rate=decay_rate),
    ))
end

function all_to_all_xx_spin_local_pseudomode_model(
        N::Integer,nmax::Integer;Jpair,omega_c=1,gamma=0.05,kappa=1,
        coupling::Symbol=:minus)
    N>=2||throw(ArgumentError(
        "the all-to-all pair interaction requires N >= 2"))
    all(value->value isa Real,(Jpair,omega_c,gamma,kappa))||throw(
        ArgumentError("Jpair, omega_c, gamma, and kappa must be real numbers"))
    R=promote_type(typeof(float(Jpair)),typeof(float(omega_c)),
                   typeof(float(gamma)),typeof(float(kappa)))
    operators=local_pseudomode_operators(nmax;T=R)
    basis=PIBasis(N,operators.dsite)
    all_to_all_xx_spin_local_pseudomode_model(
        basis,operators;Jpair,omega_c,gamma,kappa,coupling)
end

end
