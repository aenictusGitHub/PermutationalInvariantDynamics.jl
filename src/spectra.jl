function _spectrum_order(values,sortby,rev)
    sortby===:none&&return collect(eachindex(values))
    by=sortby===:real ? real : sortby===:imaginary ? imag : sortby===:magnitude ? abs :
       throw(ArgumentError("sortby must be :real, :imaginary, :magnitude, or :none"))
    sortperm(values;by=by,rev=rev)
end

function _require_autonomous_spectral_input(x)
    if isdefined(@__MODULE__,:isautonomous)
        f=getfield(@__MODULE__,:isautonomous)
        applicable(f,x)&&!f(x)&&throw(ArgumentError("stationary spectra and gaps require an autonomous Liouvillian; call freeze(...; time=..., parameters=...) explicitly"))
    end
    nothing
end

function _selected_spectrum_workspace_bytes(L,method,krylovdim,nev;
        vectors::Bool=false,projected::Bool=false,kwargs...)
    n=size(L,1);workspace=get(kwargs,:workspace,nothing)
    T=workspace isa ArnoldiWorkspace ? eltype(workspace.V) :
      workspace isa BlockArnoldiWorkspace ? eltype(workspace.V) :
      workspace isa JacobiDavidsonWorkspace ? eltype(workspace.arnoldi.V) :
      _complex_float_type(eltype(L))
    for key in (:target,:operator_scale,:initial_vector,:initial_subspace,
                :preconditioner)
        value=get(kwargs,key,nothing);value===nothing&&continue
        if value isa Number
            T=_advanced_promote_scalar_type(T,value)
        else
            S=try eltype(value) catch; Any end
            S isa Type&&S<:Number&&(T=promote_type(T,_complex_float_type(S)))
        end
    end
    effective_krylovdim=workspace isa ArnoldiWorkspace ?
        size(workspace.V,2)-1 : workspace isa JacobiDavidsonWorkspace ?
        size(workspace.arnoldi.V,2)-1 : workspace isa BlockArnoldiWorkspace ?
        size(workspace.V,2) : krylovdim
    estimate=if method in (:krylov,:arnoldi)
        _performance_arnoldi_bytes(n,T,effective_krylovdim;mode=:ordinary)
    elseif method in (:block_arnoldi,:block)
        effective_block=workspace isa BlockArnoldiWorkspace ?
            size(workspace.W,2) : Int(get(kwargs,:block_size,min(nev,4)))
        workspace isa BlockArnoldiWorkspace ?
            _block_arnoldi_workspace_owned_bytes(workspace)+
                _performance_block_arnoldi_projected_bytes(
                    effective_krylovdim,T) :
            _performance_block_arnoldi_bytes(n,T,effective_krylovdim,
                                             effective_block)
    elseif method in (:harmonic,:iram,:implicit_qr)
        _performance_arnoldi_bytes(n,T,effective_krylovdim;mode=:full)
    else
        correction_dim=workspace isa JacobiDavidsonWorkspace ?
            size(workspace.correction.V,2)-1 :
            get(kwargs,:correction_krylovdim,min(n,20))
        _performance_arnoldi_bytes(n,T,effective_krylovdim;mode=:full)+
            _performance_gmres_bytes(n,T,correction_dim)+
            _performance_array_bytes(n,T,0;linear_arrays=8)
    end
    if method in (:block_arnoldi,:block)
        estimate+=_performance_block_arnoldi_output_bytes(n,T,nev,
            get(kwargs,:maxrestarts,20);vectors)
    end
    estimate+=_performance_source_action_bytes(L,T)
    operator_workspace=get(kwargs,:operator_workspace,nothing)
    estimate+=_block_operator_workspace_bytes(operator_workspace)
    if method in (:block_arnoldi,:block)
        effective_block=workspace isa BlockArnoldiWorkspace ?
            size(workspace.W,2) : Int(get(kwargs,:block_size,min(nev,4)))
        active_block=min(n,effective_krylovdim,effective_block)
        estimate+=operator_workspace===nothing ?
            _performance_batched_action_growth_bytes(L,active_block) :
            _performance_batched_workspace_growth_bytes(
                operator_workspace,active_block)
    end
    # Requested output modes and a matrix-free projector's explicit action
    # buffers are live alongside the dominant solver workspace.
    vectors&&!(method in (:block_arnoldi,:block))&&
        (estimate+=_performance_entries_bytes(
        BigInt(n)*min(BigInt(n),BigInt(nev)),T))
    projected&&(estimate+=_performance_array_bytes(n,T,0;linear_arrays=4))
    estimate
end

function _guard_selected_spectrum_workspace(L,method,krylovdim,nev,
        vectors,memory_budget;projected::Bool=false,kwargs...)
    estimate=_selected_spectrum_workspace_bytes(L,method,krylovdim,nev;
        vectors,projected,kwargs...)
    _require_performance_budget("selected Liouvillian spectral workspace",
        estimate,memory_budget;guidance=
        "Reduce krylovdim/nev or choose a larger explicit budget.")
end

_pi_spectrum_output(info,vectors::Bool,return_info::Bool)=
    vectors ? info : return_info ? merge(info,(vectors=nothing,)) : info.values

function _matrixfree_projector_for_spectrum(L,basis,symmetry,charge;atol,rtol,
                                             rng=Random.MersenneTwister(0))
    symmetry===nothing&&return (nothing,nothing,nothing)
    basis===nothing&&throw(ArgumentError("matrix-free symmetry projection requires basis=... or a PIModel"))
    if symmetry isa Union{MatrixFreeSymmetryProjector,JointSymmetryProjector}
        symmetry.basis===basis||throw(ArgumentError(
            "matrix-free symmetry projector uses a different PI basis"))
        check=_projected_symmetry_residual(L,symmetry;probes=4,rng=rng,
                                           atol=atol,rtol=rtol)
        check.symmetric||throw(ArgumentError(
            "supplied matrix-free symmetry projector is not invariant under the Liouvillian"))
        return (symmetry,:supplied_projector,check)
    end
    candidates = symmetry===:auto ? _usual_unitary_candidates(basis.d) : [nothing=>symmetry]
    for (name,U) in candidates
        P=try matrixfree_symmetry_projector(basis,U;charge=charge,atol=atol,rtol=rtol) catch;continue;end
        check=_projected_symmetry_residual(L,P;probes=4,rng=rng,atol=atol,rtol=rtol)
        check.symmetric&&return (P,name,check)
    end
    throw(ArgumentError("no invariant matrix-free unitary symmetry projector was found for the requested charge"))
end

"""
    pi_liouvillian_spectrum(L; sortby=:real, rev=true, vectors=false,
                            return_info=false,
                            memory_budget=512*1024^2)

With `method=:dense`, compute the complete spectrum in the polynomial-size PI
operator space. With `method=:krylov`, compute selected ordinary Ritz modes;
`method=:block_arnoldi` advances several directions through one batched
operator application and uses an explicitly named thick restart;
`method=:harmonic` computes thick-restarted harmonic Ritz modes near zero,
`method=:iram` applies implicit-QR restarting, and `method=:jd` uses
hard-locking Jacobi--Davidson near `target`.
`L`
may be a static `PIModel`, an assembled matrix, or a matrix-free PI
Liouvillian. Set `vectors=true` to also return consistently sorted right
eigenvectors. Matrix-free inputs are materialized in PI coordinates, never in
the full `d^(2N)` Liouville space, only when `method=:dense`; Krylov,
harmonic, IRAM, and Jacobi--Davidson methods apply them without materializing
the PI Liouvillian.

Complete dense diagonalization is guarded by a conservative work-array
estimate. Increase `memory_budget` only when the required RAM is available,
or pass `memory_budget=Inf` as an explicit opt-in. `return_info=true` keeps
solver diagnostics but, independently of that choice, right eigenvectors are
retained only when `vectors=true`.
"""
function pi_liouvillian_spectrum(x;sortby=:real,rev::Bool=true,vectors::Bool=false,
                                 method=:dense,nev::Integer=6,krylovdim::Integer=max(20,2nev+4),
                                 atol::Real=1e-10,rtol::Real=1e-8,
                                 require_convergence::Bool=true,basis=nothing,
                                 symmetry=nothing,charge=1,thickdim::Integer=max(nev+2,2nev),
                                 maxrestarts::Integer=20,target=nothing,
                                 retained_dimension::Integer=max(nev,min(2nev,krylovdim-1)),
                                 rng=Random.MersenneTwister(0),
                                 return_info::Bool=false,
                                 memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
                                 kwargs...)
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_canonical_spectrum_algorithm(method)
    method===:auto&&throw(ArgumentError(
        "pi_liouvillian_spectrum requires an explicit method; use liouvillian_spectrum for automatic selection"))
    _require_autonomous_spectral_input(x)
    x isa PIModel&&basis===nothing&&(basis=x.basis)
    x isa PIModel&&_require_model_preparation_budget(x,memory_budget;
        operation="Liouvillian spectral model preparation")
    if method===:harmonic
        L=x isa PIModel ? liouvillian(
            x;representation=:matrixfree,memory_budget) : x
        _guard_selected_spectrum_workspace(L,method,krylovdim,nev,vectors,
            memory_budget;projected=symmetry!==nothing,target,kwargs...)
        P,sname,scheck=_matrixfree_projector_for_spectrum(L,basis,symmetry,charge;
            atol=atol,rtol=rtol,rng=rng)
        info=harmonic_arnoldi_spectrum(L;nev=nev,krylovdim=krylovdim,
            thickdim=thickdim,maxrestarts=maxrestarts,projector=P,vectors=vectors,
            target=target===nothing ? 0 : target,
            atol=atol,rtol=rtol,require_convergence=require_convergence,rng=rng,kwargs...)
        info=merge(info,(symmetry_used=P!==nothing,symmetry_name=sname,
                         symmetry_charge=P===nothing ? nothing : P.charge,
                         symmetry_residual=scheck))
        return _pi_spectrum_output(info,vectors,return_info)
    end
    if method===:iram
        symmetry===nothing||throw(ArgumentError("implicit-QR Arnoldi does not implement matrix-free symmetry projection; use method=:harmonic"))
        haskey(kwargs,:which)&&throw(ArgumentError("pi_liouvillian_spectrum maps sortby to implicit-QR selection; pass sortby instead of which"))
        sortby in (:real,:magnitude)||throw(ArgumentError("implicit-QR spectra support sortby=:real or :magnitude"))
        which=sortby===:real ? (rev ? :LR : throw(ArgumentError("ascending-real implicit-QR selection is not supported"))) : (rev ? :LM : :SM)
        L=x isa PIModel ? liouvillian(
            x;representation=:matrixfree,memory_budget) : x
        _guard_selected_spectrum_workspace(L,method,krylovdim,nev,vectors,
            memory_budget;target,kwargs...)
        info=implicitly_restarted_arnoldi_spectrum(L;nev=nev,krylovdim=krylovdim,
            retained_dimension=retained_dimension,maxrestarts=maxrestarts,which=which,
            target=target,vectors=vectors,atol=atol,rtol=rtol,
            require_convergence=require_convergence,rng=rng,kwargs...)
        info=merge(info,(method=:iram,selection=target===nothing ? which : :near_target))
        return _pi_spectrum_output(info,vectors,return_info)
    end
    if method===:jd
        symmetry===nothing||throw(ArgumentError("Jacobi--Davidson does not implement matrix-free symmetry projection; use method=:harmonic"))
        haskey(kwargs,:subspace_dim)&&throw(ArgumentError("pi_liouvillian_spectrum maps krylovdim to the Jacobi--Davidson subspace; pass krylovdim instead of subspace_dim"))
        L=x isa PIModel ? liouvillian(
            x;representation=:matrixfree,memory_budget) : x
        _guard_selected_spectrum_workspace(L,method,krylovdim,nev,vectors,
            memory_budget;target,kwargs...)
        info=jacobi_davidson_spectrum(L;nev=nev,target=target===nothing ? 0 : target,
            subspace_dim=krylovdim,vectors=vectors,atol=atol,rtol=rtol,
            require_convergence=require_convergence,rng=rng,kwargs...)
        info=merge(info,(method=:jd,selection=:near_target))
        return _pi_spectrum_output(info,vectors,return_info)
    end
    if method===:block_arnoldi
        symmetry===nothing||throw(ArgumentError(
            "block Arnoldi does not implement matrix-free symmetry projection; use method=:harmonic"))
        sortby in (:real,:magnitude)||throw(ArgumentError(
            "block-Arnoldi spectra support sortby=:real or :magnitude"))
        which=sortby===:real ? (rev ? :LR : throw(ArgumentError(
            "ascending-real block-Arnoldi selection is not supported"))) :
            (rev ? :LM : :SM)
        L=x isa PIModel ? liouvillian(
            x;representation=:matrixfree,memory_budget) : x
        _guard_selected_spectrum_workspace(L,:block_arnoldi,krylovdim,nev,
            vectors,memory_budget;target,maxrestarts,kwargs...)
        info=block_arnoldi_spectrum(L;nev,krylovdim,which,target,
            vectors,atol,rtol,maxrestarts,retained_dimension,
            require_convergence,rng,memory_budget,kwargs...)
        return _pi_spectrum_output(info,vectors,return_info)
    end
    if method in (:krylov,:arnoldi)
        symmetry===nothing||throw(ArgumentError("use method=:harmonic for matrix-free symmetry projection"))
        sortby in (:real,:magnitude)||throw(ArgumentError("Krylov spectra support sortby=:real or :magnitude"))
        which=sortby===:real ? (rev ? :LR : throw(ArgumentError("ascending-real Krylov selection is not supported"))) : (rev ? :LM : :SM)
        L=x isa PIModel ? liouvillian(
            x;representation=:matrixfree,memory_budget) : x
        _guard_selected_spectrum_workspace(L,method,krylovdim,nev,vectors,
            memory_budget;kwargs...)
        info=krylov_liouvillian_spectrum(L;nev=nev,krylovdim=krylovdim,which=which,
            vectors=vectors,atol=atol,rtol=rtol,require_convergence=require_convergence,rng=rng,kwargs...)
        info=merge(info,(method=:arnoldi,))
        return _pi_spectrum_output(info,vectors,return_info)
    end
    # Prepare model geometry without assembling its PI-coordinate matrix, then
    # reject an oversized dense request before the first n-by-n allocation.
    L=x isa PIModel ? LiouvillianPlan(x) : x
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch(
        "Liouvillian must be square"))
    T=promote_type(_complex_float_type(eltype(L)),ComplexF64)
    estimate=_performance_array_bytes(n,T,vectors ? 7 : 5;
                                       linear_arrays=6)
    _require_performance_budget("complete Liouvillian eigendecomposition",
        estimate,memory_budget;guidance=
        "Use method=:krylov, :block_arnoldi, :harmonic, :iram, or :jd for selected modes.")
    M=_materialize(L);size(M,1)==size(M,2)||throw(DimensionMismatch("Liouvillian must be square"))
    if vectors
        E=eigen(Matrix(M));order=_spectrum_order(E.values,sortby,rev)
        return (values=E.values[order],vectors=E.vectors[:,order],dimension=size(M,1))
    end
    values=eigvals(Matrix(M))
    ordered=values[_spectrum_order(values,sortby,rev)]
    return_info ? (values=ordered,vectors=nothing,dimension=size(M,1),
                   method=:dense,partial_scope=false) : ordered
end

@doc """
    pi_liouvillian_gap(L; atol=1e-12, rtol=1e-10,
                       check_stability=true, return_info=false,
                       initial_vector=nothing)

Return the asymptotic Liouvillian decay gap
`-max(real(lambda))` after excluding the numerical stationary cluster around
zero. Unlike selecting the second sorted eigenvalue, this handles degenerate
stationary spaces and oscillatory decay modes. `return_info=true` also reports
the controlling eigenvalue, oscillation frequency, stationary multiplicity,
stability, tolerance, and PI dimension.

`method=:harmonic` is deliberately not accepted for a global gap: harmonic
Ritz values are selected by distance to zero, so they can miss a slow mode
with a large oscillation frequency. With a unitary `symmetry` and
`return_info=true`, it instead reports a near-zero decay estimate for the
selected `charge` sector. A nontrivial charge sector is not required to
contain a stationary eigenvalue. The estimate is certified only when the
complete selected sector was extracted; inspect `gap_certified`,
`sector_dimension`, and `scope` in the returned information.

For iterative methods, `initial_vector` supplies an explicit reproducible
Krylov seed. The solver projects it into the requested symmetry charge when
applicable and rejects a seed with no component in that sector.

`method=:iram` is a bounded-memory global-gap route because it selects
largest-real Ritz values. `method=:jd` is rejected here: near-target
Jacobi--Davidson has the same large-imaginary slow-mode caveat as harmonic
selection and currently has no charge-projector specialization.
""" pi_liouvillian_gap
function _liouvillian_gap_info(values;atol,rtol,check_stability)
    scale=max(maximum(abs,values;init=0.0),1.0);tol=atol+rtol*scale
    stationary=abs.(values).<=tol;nullity=count(stationary)
    nullity>0||throw(ArgumentError("no stationary eigenvalue was found within tolerance $tol"))
    spectral_abscissa=maximum(real,values)
    stable=spectral_abscissa<=tol
    check_stability&&!stable&&throw(ArgumentError("Liouvillian is unstable: spectral abscissa is $spectral_abscissa"))
    candidates=values[.!stationary]
    if isempty(candidates)
        gap=Inf;mode=nothing;frequency=NaN
    else
        j=argmax(real.(candidates));mode=candidates[j];gap=-real(mode)
        abs(gap)<=tol&&(gap=zero(gap));frequency=imag(mode)
    end
    info=(gap=gap,decay_eigenvalue=mode,oscillation_frequency=frequency,
          stationary_multiplicity=nullity,unique_stationary_mode=nullity==1,
          stable=stable,spectral_abscissa=spectral_abscissa,tolerance=tol,
          dimension=length(values))
    info
end

# A nontrivial unitary charge sector need not contain a stationary mode.  Its
# asymptotic decay rate is therefore determined from the sector spectral
# abscissa directly, excluding a zero cluster only when one is actually
# present (for example in a symmetry-broken stationary manifold).
function _sector_decay_info(values;atol,rtol,check_stability,
                            require_stationary::Bool=false)
    isempty(values)&&throw(ArgumentError("no eigenvalues were returned for the selected charge sector"))
    scale=max(maximum(abs,values;init=0.0),1.0);tol=atol+rtol*scale
    stationary=abs.(values).<=tol;nullity=count(stationary)
    require_stationary&&nullity==0&&throw(ArgumentError("the trivial charge sector did not contain a stationary eigenvalue within tolerance $tol"))
    spectral_abscissa=maximum(real,values);stable=spectral_abscissa<=tol
    check_stability&&!stable&&throw(ArgumentError("selected symmetry sector is unstable: spectral abscissa is $spectral_abscissa"))
    candidates=values[.!stationary]
    if isempty(candidates)
        gap=Inf;mode=nothing;frequency=NaN
    else
        j=argmax(real.(candidates));mode=candidates[j];gap=-real(mode)
        abs(gap)<=tol&&(gap=zero(gap));frequency=imag(mode)
    end
    (gap=gap,decay_eigenvalue=mode,oscillation_frequency=frequency,
     stationary_multiplicity=nullity,unique_stationary_mode=nullity==1,
     stable,spectral_abscissa,tolerance=tol,dimension=length(values))
end

function _group_unitary_eigenvalues(values,tol)
    groups=Vector{Vector{Int}}();labels=ComplexF64[]
    for i in eachindex(values)
        j=findfirst(z->abs(values[i]-z)<=tol,labels)
        if j===nothing;push!(labels,values[i]);push!(groups,[i]);else;push!(groups[j],i);end
    end
    labels,groups
end

function _symmetry_superoperator_for_gap(U,basis,n;atol,rtol,cache=nothing)
    if basis===nothing
        D=_check_unitary_matrix(U;atol=atol,rtol=rtol);D^2==n||throw(DimensionMismatch("Liouvillian dimension must equal size(U,1)^2"))
        sandwich_superoperator(U)
    else
        length(basis)==n||throw(DimensionMismatch("PI basis and Liouvillian dimensions differ"))
        _pi_conjugation_superoperator(basis,U;atol=atol,rtol=rtol,cache=cache)
    end
end

function _symmetry_block_spectrum(M,S;atol,rtol)
    residual=norm(M*S-S*M);scale=max(norm(M)*norm(S),1.0);symtol=atol+rtol*scale
    residual<=symtol||throw(ArgumentError("candidate is not a Liouvillian weak symmetry: residual=$residual, tolerance=$symtol"))
    svalues,W=_orthonormal_unitary_eigensystem(S;atol=atol,rtol=rtol)
    grouptol=atol+rtol*max(maximum(abs,svalues;init=0.0),1.0)
    labels,groups=_group_unitary_eigenvalues(svalues,grouptol)
    values=ComplexF64[];sectors=NamedTuple[]
    for (label,inds) in zip(labels,groups)
        k=length(inds);V=W[:,inds]
        B=adjoint(V)*M*V;vals=eigvals(Matrix(B));append!(values,vals)
        tol=atol+rtol*max(maximum(abs,vals;init=0.0),1.0);stationary=count(x->abs(x)<=tol,vals)
        abscissa=maximum(real,vals);nonzero=vals[abs.(vals).>tol]
        gap=isempty(nonzero) ? Inf : max(0.0,-maximum(real,nonzero))
        push!(sectors,(charge=label,dimension=k,stationary_multiplicity=stationary,
                       spectral_abscissa=abscissa,gap=gap))
    end
    values,sectors,(residual=residual,relative_residual=residual/scale,tolerance=symtol)
end

function _auto_gap_symmetry(x,basis,M;atol,rtol)
    basis===nothing&&return nothing
    best=nothing;bestblocks=1;cache=OneBodyGeometry(basis)
    for (name,U) in _usual_unitary_candidates(basis.d)
        S=_symmetry_superoperator_for_gap(U,basis,size(M,1);atol=atol,rtol=rtol,cache=cache)
        residual=norm(M*S-S*M);scale=max(norm(M)*norm(S),1.0)
        residual<=atol+rtol*scale||continue
        svalues,_=_orthonormal_unitary_eigensystem(S;atol=atol,rtol=rtol)
        labels,_=_group_unitary_eigenvalues(svalues,atol+rtol)
        length(labels)>bestblocks&&(best=(name,U,S);bestblocks=length(labels))
    end
    best
end

function pi_liouvillian_gap(x;atol::Real=1e-12,rtol::Real=1e-10,
                            check_stability::Bool=true,return_info::Bool=false,
                            symmetry=nothing,symmetry_kind=:unitary,basis=nothing,
                            method=:dense,nev::Integer=6,
                            krylovdim::Integer=max(20,2nev+4),
                            require_convergence::Bool=true,charge=1,
                            thickdim::Integer=max(nev+2,2nev),
                            maxrestarts::Integer=20,
                            retained_dimension::Integer=max(nev,min(2nev,krylovdim-1)),
                            initial_vector=nothing,
                            rng=Random.MersenneTwister(0),
                            memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
                            kwargs...)
    symmetry_kind===:unitary||throw(ArgumentError("only unitary weak symmetries define linear charge blocks for gap reduction"))
    _require_autonomous_spectral_input(x)
    x isa PIModel&&basis===nothing&&(basis=x.basis)
    x isa PIModel&&_require_model_preparation_budget(x,memory_budget;
        operation="Liouvillian-gap model preparation")
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_canonical_spectrum_algorithm(method)
    method===:auto&&throw(ArgumentError(
        "pi_liouvillian_gap requires an explicit method"))
    method===:block_arnoldi&&throw(ArgumentError(
        "block Arnoldi does not currently provide the global-gap certification used by pi_liouvillian_gap"))
    method===:jd&&throw(ArgumentError(
        "near-target Jacobi--Davidson cannot certify the global largest-real Liouvillian gap; use method=:iram or :krylov"))
    method===:iram&&(haskey(kwargs,:target)||haskey(kwargs,:which))&&throw(ArgumentError(
        "a Liouvillian gap requires largest-real implicit-QR selection; do not pass target or which"))
    if method===:harmonic
        L=x isa PIModel ? liouvillian(
            x;representation=:matrixfree,memory_budget) : x
        _guard_selected_spectrum_workspace(L,method,krylovdim,nev,false,
            memory_budget;projected=symmetry!==nothing,target=0,
            initial_vector,kwargs...)
        P,sname,scheck=_matrixfree_projector_for_spectrum(L,basis,symmetry,charge;
            atol=atol,rtol=rtol,rng=rng)
        P===nothing&&throw(ArgumentError("method=:harmonic selects modes nearest zero in modulus and cannot certify the global largest-real Liouvillian gap; use method=:krylov or supply a unitary symmetry and request return_info=true for a charge-sector estimate with explicit certification metadata"))
        sector_dimension=_projector_range_dimension(P,size(L,1))
        sector_nev=min(Int(nev),sector_dimension)
        har=harmonic_arnoldi_spectrum(L;nev=sector_nev,krylovdim=krylovdim,
            thickdim=thickdim,maxrestarts=maxrestarts,projector=P,
            initial_vector=initial_vector,atol=atol,rtol=rtol,
            require_convergence=require_convergence,rng=rng)
        qtol=atol+rtol
        require_stationary=_projector_has_only_trivial_charges(P,qtol)
        info=_sector_decay_info(har.values;atol=atol,rtol=rtol,
            check_stability=check_stability,require_stationary=require_stationary)
        complete_sector=length(har.values)==sector_dimension&&all(har.converged)
        stationary_complete=complete_sector||info.stationary_multiplicity<sector_nev
        info=merge(info,(dimension=size(L,1),method=:harmonic,
            symmetry_used=P!==nothing,symmetry_name=sname,
            symmetry_charge=P===nothing ? nothing : P.charge,
            symmetry_sectors=nothing,symmetry_residual=scheck,
            ritz_residuals=har.residuals,restarts=har.restarts,
            ritz_extraction=har.ritz_extraction,
            search_space_exhausted=har.search_space_exhausted,
            stationary_multiplicity_certified=stationary_complete,
            krylov_dimension=har.krylov_dimension,
            sector_dimension=sector_dimension,
            scope=:charge_sector,selection=:near_zero,
            gap_certified=complete_sector,stability_certified=complete_sector,
            certification_message=complete_sector ?
                "the complete selected charge sector was diagonalized" :
                "harmonic Ritz extraction orders by distance to zero, not by real part"))
        return_info&&return info
        complete_sector&&return info.gap
        throw(ArgumentError("a partial harmonic symmetry-sector calculation is a near-zero decay estimate, not a certified gap; set return_info=true and inspect gap_certified, or increase nev to the reported sector dimension"))
    end
    if method in (:krylov,:arnoldi,:iram,:implicit_qr)
        symmetry===nothing||throw(ArgumentError("use method=:harmonic for matrix-free symmetry projection"))
        L=x isa PIModel ? liouvillian(
            x;representation=:matrixfree,memory_budget) : x
        _guard_selected_spectrum_workspace(L,method,krylovdim,nev,false,
            memory_budget;initial_vector,kwargs...)
        arn = if method in (:iram,:implicit_qr)
            implicitly_restarted_arnoldi_spectrum(L;nev=nev,krylovdim=krylovdim,
                retained_dimension=retained_dimension,maxrestarts=maxrestarts,which=:LR,
                initial_vector=initial_vector,atol=atol,rtol=rtol,
                require_convergence=require_convergence,rng=rng,kwargs...)
        else
            krylov_liouvillian_spectrum(L;nev=nev,krylovdim=krylovdim,which=:LR,
                initial_vector=initial_vector,atol=atol,rtol=rtol,
                require_convergence=require_convergence,rng=rng,kwargs...)
        end
        info=_liouvillian_gap_info(arn.values;atol=atol,rtol=rtol,check_stability=check_stability)
        # A partial spectrum certifies the returned slow mode, but not that all
        # stationary modes have been counted if the requested window is full.
        stationary_complete=info.stationary_multiplicity<nev
        selected_method=method===:iram ? :iram : :arnoldi
        info=merge(info,(dimension=size(L,1),symmetry_used=false,symmetry_name=nothing,
            symmetry_sectors=nothing,symmetry_residual=nothing,method=selected_method,
            ritz_residuals=arn.residuals,stationary_multiplicity_certified=stationary_complete,
            krylov_dimension=arn.krylov_dimension,scope=:global,
            selection=:largest_real,gap_certified=stationary_complete,
            stability_certified=stationary_complete))
        return return_info ? info : info.gap
    end
    L=x isa PIModel ? LiouvillianPlan(x) : x
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch(
        "Liouvillian must be square"))
    T=promote_type(_complex_float_type(eltype(L)),ComplexF64)
    # Weak-symmetry discovery can retain the Liouvillian, conjugation map,
    # eigensystem, and dense charge blocks simultaneously. The larger factor
    # also safely covers the ordinary complete-gap path.
    estimate=_performance_array_bytes(n,T,12;linear_arrays=8)
    _require_performance_budget("complete Liouvillian gap analysis",estimate,
        memory_budget;guidance=
        "Use method=:krylov or :iram for a bounded largest-real calculation.")
    M=_materialize(L)
    size(M,1)==size(M,2)||throw(DimensionMismatch("Liouvillian must be square"))
    selected_name=nothing;selected=symmetry;selected_S=nothing
    if symmetry===:auto
        found=_auto_gap_symmetry(x,basis,M;atol=atol,rtol=rtol)
        if found===nothing;selected=nothing;else;selected_name,selected,selected_S=found;end
    end
    if selected===nothing
        values=eigvals(Matrix(M));info=_liouvillian_gap_info(values;atol=atol,rtol=rtol,check_stability=check_stability)
        info=merge(info,(symmetry_used=false,symmetry_name=selected_name,symmetry_sectors=nothing,symmetry_residual=nothing))
    else
        S=selected_S===nothing ? _symmetry_superoperator_for_gap(selected,basis,size(M,1);atol=atol,rtol=rtol) : selected_S
        values,sectors,residual=_symmetry_block_spectrum(M,S;atol=atol,rtol=rtol)
        info=_liouvillian_gap_info(values;atol=atol,rtol=rtol,check_stability=check_stability)
        info=merge(info,(symmetry_used=true,symmetry_name=selected_name,symmetry_sectors=sectors,symmetry_residual=residual))
    end
    return_info ? info : info.gap
end

"""Alias for [`pi_liouvillian_gap`](@ref)."""
liouvillian_gap(x;kwargs...)=pi_liouvillian_gap(x;kwargs...)

"""
    pi_density_spectrum(rho; expanded=false, max_expanded_dimension=10^7)

Diagonalize every physical Schur block of a PI density operator. The default
compressed result stores each irrep eigenvalue once together with its exact
symmetric-group degeneracy. `expanded=true` returns the full eigenvalue list
only when its size does not exceed `max_expanded_dimension`.
"""
function pi_density_spectrum(rho::PIState;expanded::Bool=false,
                             max_expanded_dimension::Integer=10^7,
                             sortby=:value,rev::Bool=true,atol::Real=1e-12)
    ishermitian(rho;atol=atol,rtol=0)||throw(ArgumentError("density operator must be Hermitian"))
    RT=typeof(real(zero(eltype(rho.data))))
    values=RT[];degeneracies=BigInt[];sectors=Partition[];indices=Int[]
    for p in rho.basis.sectors
        R=physical_block(rho,p);vals=eigvals(Hermitian(R));f=symmetric_group_dimension(p)
        for (i,v) in pairs(vals)
            push!(values,v);push!(degeneracies,f);push!(sectors,p);push!(indices,i)
        end
    end
    order = sortby===:value ? sortperm(values;rev=rev) :
            sortby===:magnitude ? sortperm(values;by=abs,rev=rev) :
            sortby===:none ? collect(eachindex(values)) :
            throw(ArgumentError("sortby must be :value, :magnitude, or :none"))
    values=values[order];degeneracies=degeneracies[order];sectors=sectors[order];indices=indices[order]
    total=sum(degeneracies;init=big(0))
    if expanded
        total<=max_expanded_dimension||throw(ArgumentError("expanded spectrum has dimension $total; increase max_expanded_dimension explicitly"))
        out=RT[];sizehint!(out,Int(total))
        for (v,g) in zip(values,degeneracies);append!(out,Iterators.repeated(v,Int(g)));end
        return out
    end
    (;values,degeneracies,sectors,sector_indices=indices,total_dimension=total,
      trace=sum(BigFloat(g)*v for (v,g) in zip(values,degeneracies)),
      minimum=minimum(values),maximum=maximum(values))
end

"""Alias for [`pi_density_spectrum`](@ref)."""
const density_operator_spectrum=pi_density_spectrum
"""Alias for [`pi_density_spectrum`](@ref)."""
const pi_density_operator_spectrum=pi_density_spectrum
