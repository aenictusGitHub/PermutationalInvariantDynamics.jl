module PaperModels

using LinearAlgebra
using PermutationalInvariantDynamics

export spin_matrices, damanet2016_model, damanet2016_intensity_operator,
       damanet2016_intensity_exact, pausch2024_model, kitagawa1993_oat_model,
       kitagawa1993_mean_spin_exact, huelga1997_dephasing_model,
       huelga1997_ramsey_exact, shammah2018_thermal_model,
       shammah2018_thermal_state
export morrison2008_model, morrison2008_exact_state,
       meiser2009_superradiance_model, iemini2018_btc_model,
       nakanishi2023_pt_model, nakanishi2023_pt_spectrum,
       piccitto2021_interacting_btc_model

"""PRA 94, 033838 (2016), Eqs. (3)-(5), with `delta_gamma=gamma0-gamma`."""
function damanet2016_model(N;gamma0=1.0,gamma=gamma0)
    0<=gamma<=gamma0 || throw(ArgumentError("the paper assumes 0 <= gamma <= gamma0"))
    b=PIBasis(N,2); sm=ComplexF64[0 1;0 0]
    PIModel(b,[CollectiveJump(sm;rate=gamma),LocalJump(sm;rate=gamma0-gamma)])
end

"""Operator whose expectation is the radiated rate, paper Eq. (39)."""
function damanet2016_intensity_operator(b;gamma0=1.0,gamma=gamma0)
    sm=ComplexF64[0 1;0 0]; Jm=collective_operator(b,sm)
    gamma*(adjoint(Jm)*Jm)+(gamma0-gamma)*collective_operator(b,sm'*sm)
end

"""Analytical N=2 intensity, Eqs. (41)-(43)."""
function damanet2016_intensity_exact(t;gamma0=1.0,gamma=gamma0)
    dg=gamma0-gamma
    iszero(dg) && return 2gamma0*exp(-2gamma0*t)*(1+2gamma0*t)
    iszero(gamma) && return 2gamma0*exp(-gamma0*t)
    exp(-2*(gamma+dg)*t)/((2gamma+dg)*dg)*((2gamma+dg)^2*dg+dg^2*(2gamma+dg)+
        (2gamma+dg)^3*(exp(dg*t)-1)+dg^3*(exp((2gamma+dg)*t)-1))
end

"""PRA 110, 062208 (2024), Eqs. (2)-(6)."""
function pausch2024_model(N,d;V=1.0,gammaI=0.0,gammaC=0.0,dissipator=:spin)
    s=spin_matrices(d); b=PIBasis(N,d)
    Jx=collective_operator(b,s.jx); Jy=collective_operator(b,s.jy)
    H=(Jx*Jx-Jy*Jy)*(V/(N*s.j))
    ell = dissipator===:spin ? s.jm : dissipator===:equal ?
        sqrt(2s.j)*diagm(1=>ones(ComplexF64,d-1)) : throw(ArgumentError("dissipator must be :spin or :equal"))
    PIModel(b,[DirectPIHamiltonian(H),LocalJump(ell;rate=gammaI/s.j),
               CollectiveJump(ell;rate=gammaC/(N*s.j))])
end

"""Kitagawa--Ueda, PRA 47, 5138 (1993): `H=chi*Jz^2`."""
function kitagawa1993_oat_model(N;chi=1.0)
    b=PIBasis(N,2); sz=ComplexF64[1 0;0 -1]
    Jz=collective_operator(b,sz/2)
    PIModel(b,[DirectPIHamiltonian(chi*(Jz*Jz))])
end
kitagawa1993_mean_spin_exact(N,t;chi=1.0)=N/2*cos(chi*t)^(N-1)

"""Huelga et al., PRL 79, 3865 (1997): independent Markovian dephasing."""
function huelga1997_dephasing_model(N;gamma=1.0)
    b=PIBasis(N,2); sz=ComplexF64[1 0;0 -1]
    PIModel(b,[LocalJump(sz;rate=gamma/2)])
end
huelga1997_ramsey_exact(N,t;gamma=1.0)=N/2*exp(-gamma*t)

"""Local pumping and emission benchmark used in Shammah et al., PRA 98, 063815 (2018)."""
function shammah2018_thermal_model(N;down=1.0,up=0.25)
    b=PIBasis(N,2); sm=ComplexF64[0 1;0 0]
    PIModel(b,[LocalJump(sm;rate=down),LocalJump(sm';rate=up)])
end
function shammah2018_thermal_state(b;down=1.0,up=0.25)
    iid_state(b,ComplexF64[down 0;0 up]/(down+up))
end

"""Morrison--Parkins, PRA 77, 043810 (2008), Eq. (1)."""
function morrison2008_model(N;Omega=0.2,gamma=0.3,restricted=true)
    b=restricted ? PIBasis(N,2;sectors=[(N,0)]) : PIBasis(N,2)
    sx=ComplexF64[0 1;1 0];sm=ComplexF64[0 1;0 0]
    PIModel(b,[CollectiveHamiltonian(sx/2;rate=Omega),
               CollectiveJump(sm;rate=2gamma/N)])
end

"""Exact symmetric-sector steady state, Morrison--Parkins Eq. (2)."""
function morrison2008_exact_state(b;Omega=0.2,gamma=0.3)
    length(b.sectors)==1&&b.sectors[1]==Partition((b.N,0))||throw(ArgumentError("exact state requires the symmetric-sector basis"))
    sm=ComplexF64[0 1;0 0];Jm=collective_block(b,sm,b.sectors[1]);a=Omega*b.N/(2gamma)
    A=Jm+im*a*I;R=inv(A)*inv(A)';R=(R+R')/2;R./=tr(R)
    sector_density_matrix(b,b.sectors[1],R)
end

"""Meiser--Holland, PRA 81, 033847 (2010), Eq. (1)."""
function meiser2009_superradiance_model(N;GammaC=1.0,pump=1.0)
    b=PIBasis(N,2);sm=ComplexF64[0 1;0 0]
    PIModel(b,[CollectiveJump(sm;rate=GammaC),LocalJump(sm';rate=pump)])
end

"""Iemini et al., PRL 121, 035301 (2018), Eq. (2)."""
function iemini2018_btc_model(N;omega0=1.5,kappa=1.0)
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
function nakanishi2023_pt_model(N;g=1.3,kappa=0.4,p=0.0)
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
function nakanishi2023_pt_spectrum(N;g=1.3,kappa=0.4)
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
function piccitto2021_interacting_btc_model(N;omega_z=1.0,omega_x=3.0,
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

end
