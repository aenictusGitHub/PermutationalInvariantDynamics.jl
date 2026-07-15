"""
    evans_uniqueness(H, jumps; atol=1e-12, rtol=1e-10, return_basis=false)

Apply the finite-dimensional Evans commutant criterion to explicit
Hilbert-space operators. The commutant of `{H,L_k,L_k'}` is obtained as the
nullspace of stacked vectorized commutators. A one-dimensional commutant
certifies a unique steady state (assuming the finite-dimensional
trace-preserving GKSL semigroup).
"""
function evans_uniqueness(H::AbstractMatrix,jumps;
                          atol::Real=1e-12,rtol::Real=1e-10,return_basis::Bool=false)
    n=size(H,1);size(H)==(n,n)||throw(DimensionMismatch("Hamiltonian must be square"))
    LinearAlgebra.ishermitian(H)||throw(ArgumentError("Hamiltonian must be Hermitian"))
    js=collect(jumps);all(L->L isa AbstractMatrix&&size(L)==(n,n),js)||throw(DimensionMismatch("every jump operator must match the Hamiltonian dimension"))
    ops=Any[H]
    for L in js;push!(ops,L);push!(ops,adjoint(L));end
    CT=promote_type(ComplexF64,eltype(H),(eltype(L) for L in js)...)
    C=Matrix{CT}(undef,length(ops)*n^2,n^2)
    for (q,A) in pairs(ops)
        rows=(q-1)*n^2+1:q*n^2
        C[rows,:].=left_superoperator(CT.(A))-right_superoperator(CT.(A))
    end
    S=svd(C);tol=atol+rtol*(isempty(S.S) ? 0 : maximum(S.S));nullity=count(<=(tol),S.S)
    # The identity always belongs to the exact commutant. A zero numerical
    # nullity signals tolerances below the factorization's roundoff floor.
    nullity>=1||throw(ArgumentError("numerical commutant lost the identity; increase atol or rtol"))
    result=(unique=nullity==1,certified=true,commutant_dimension=nullity,
            tolerance=tol,hilbert_dimension=n,criterion=:evans_commutant,
            reason=nullity==1 ? "the commutant of H, L_k, and L_k† is scalar" : "the Evans commutant is nontrivial")
    return_basis ? merge(result,(commutant_basis=S.V[:,end-nullity+1:end],)) : result
end

evans_uniqueness(jumps::AbstractVector{<:AbstractMatrix};kwargs...)=
    evans_uniqueness(zeros(ComplexF64,size(first(jumps),1),size(first(jumps),1)),jumps;kwargs...)

function _static_evans_rate(t)
    t.rate isa Number||throw(ArgumentError("Evans uniqueness requires a time-independent GKSL model"))
    real(t.rate)>=0&&iszero(imag(complex(t.rate)))||throw(ArgumentError("Evans uniqueness requires real nonnegative dissipative rates"))
    real(t.rate)
end

function _static_evans_hamiltonian_scale(t)
    t.rate isa Number||throw(ArgumentError("Evans uniqueness requires a time-independent Hamiltonian"))
    scale=t.rate/t.hbar
    iszero(imag(complex(scale)))||
        throw(ArgumentError("Evans uniqueness requires a Hermitian Hamiltonian with real coefficients"))
    real(scale)
end

function _check_evans_microscopic_operator(term::_BuiltinPITerm,b::PIBasis)
    operator=term.operator
    if term isa _HamiltonianPITerm
        hermitian=operator isa AbstractPIOperator ? ishermitian(operator) :
                                                   LinearAlgebra.ishermitian(operator)
        hermitian||throw(ArgumentError("Evans uniqueness requires a Hermitian Hamiltonian"))
    end
    p=body_order(term)
    if p>1
        R=_real_float_type(eltype(operator));scale=max(norm(operator,Inf),one(R))
        for adjacent in 1:p-1
            permutation=_tensor_swap_permutation(p,b.d,adjacent)
            maximum(abs,operator-operator[permutation,permutation])<=R(1e-10)*scale||
                throw(ArgumentError("p-body operator must be invariant under permutations of its p particles"))
        end
    end
    nothing
end

# `vec` initially groups all ket indices and then all bra indices.  Appendix-D
# instead expects one local label per particle.  This map sends
#
#     (a_1,...,a_p ; c_1,...,c_p)
#
# to the p-site alphabet `(a_s + d*c_s)`, whose local dimension is d^2.  The
# first particle remains the fastest tensor index in both conventions.
function _evans_regroup_indices(d::Integer,p::Integer)
    d>=1||throw(ArgumentError("d must be positive"))
    p>=1||throw(ArgumentError("p must be positive"))
    n=d^p;q=d^2;indices=Matrix{Int}(undef,n,n)
    for c in 0:n-1,a in 0:n-1
        index=0
        for site in 0:p-1
            ket=(a÷d^site)%d;bra=(c÷d^site)%d
            index+=(ket+d*bra)*q^site
        end
        indices[a+1,c+1]=index+1
    end
    indices
end

"""Local commutator `X -> A*X-X*A`, regrouped into p sites of dimension `d^2`."""
function _evans_regrouped_commutator(A::AbstractMatrix,p::Integer,d::Integer)
    n=d^p;size(A)==(n,n)||
        throw(DimensionMismatch("a $p-body operator must be $n×$n"))
    indices=_evans_regroup_indices(d,p)
    CT=promote_type(ComplexF64,eltype(A));D=zeros(CT,n^2,n^2)
    # Fill the two commutator actions directly.  This avoids materializing a
    # second, conventionally grouped n^2-by-n^2 Kronecker matrix.
    for c in 1:n,a in 1:n,u in 1:n
        D[indices[a,c],indices[u,c]]+=A[a,u]
    end
    for c in 1:n,a in 1:n,v in 1:n
        D[indices[a,c],indices[a,v]]-=A[v,c]
    end
    D
end

function _evans_missing_report(model::PIModel,reason;
                               criterion=:evans_auxiliary_pi,
                               memory_budget=nothing,estimated_bytes=nothing,
                               estimated_geometry_bytes=nothing)
    report=(unique=missing,certified=false,commutant_dimension=nothing,
            tolerance=nothing,hilbert_dimension=big(model.basis.d)^model.basis.N,
            criterion,scope=:full_hilbert_space,reason)
    memory_budget===nothing ? report :
        merge(report,(;memory_budget,estimated_bytes,
                       estimated_geometry_bytes))
end

function _evans_pbody_isometry_elements(N::Int,q::Int,p::Int)
    # PBodyGeometry retains one Float64 isometry for every corner-removal path.
    # Its shape is dim(lambda) x dim(mu) x q^p. Count those entries exactly
    # without constructing the auxiliary basis, GT patterns, paths, or arrays.
    total=big(0)
    for lambda in partitions(N,q)
        centers=Dict(lambda=>big(1))
        for _ in 1:p
            next=Dict{typeof(lambda),BigInt}()
            for (sector,count) in centers,corner in removable_corners(sector)
                child=remove_corner(sector,corner)
                next[child]=get(next,child,big(0))+count
            end
            centers=next
        end
        parent_dimension=unitary_group_dimension(lambda)
        for (center,count) in centers
            total+=count*parent_dimension*unitary_group_dimension(center)*big(q)^p
        end
    end
    total
end

function _evans_auxiliary_memory_estimate(model::PIModel,return_basis::Bool)
    b=model.basis;qbig=big(b.d)^2
    qbig<=typemax(Int)||return (;bytes=typemax(Int),coordinates=typemax(Int),
        auxiliary_local_dimension=typemax(Int),
        estimated_geometry_bytes=typemax(Int))
    q=Int(qbig);coordinates=binomial(big(b.N)+big(q)^2-1,big(b.N))
    retained_local=big(0);largest_local=big(0)
    body_orders=Set{Int}()
    for t in model.terms
        t isa _BuiltinPITerm||continue
        t isa Union{DirectPIHamiltonian,DirectPIJump}&&continue
        p=body_order(t);push!(body_orders,p);elements=big(q)^(2p)
        retained_local+=elements;largest_local=max(largest_local,elements)
    end
    # Sector matrices, Hermitian eigensolver scratch, representation geometry,
    # and (when requested) retained compressed eigenvectors.  This is a
    # deliberately conservative setup bound, not a promise about allocator
    # internals.
    factor=return_basis ? big(12) : big(8)
    matrix_bytes=big(sizeof(ComplexF64))*(factor*coordinates+retained_local+3*largest_local)
    isometry_elements=sum(p->_evans_pbody_isometry_elements(b.N,q,p),body_orders;
                           init=big(0))
    # Retained isometries dominate PBodyGeometry. A factor of three also covers
    # path dictionaries, multiplicity tables, and peak construction scratch.
    geometry_bytes=3big(sizeof(Float64))*isometry_elements
    bytes=matrix_bytes+geometry_bytes
    (;bytes,coordinates,auxiliary_local_dimension=q,
      estimated_geometry_bytes=geometry_bytes)
end

function _evans_auxiliary_descriptors(model::PIModel)
    b=model.basis;descriptors=NamedTuple[]
    for term in model.terms
        term isa _BuiltinPITerm||return nothing
        term isa Union{DirectPIHamiltonian,DirectPIJump}&&return nothing
        p=body_order(term);operator=Matrix{ComplexF64}(term.operator)
        D=_evans_regrouped_commutator(operator,p,b.d)
        if term isa _HamiltonianPITerm
            scale=_static_evans_hamiltonian_scale(term)
            iszero(scale)||push!(descriptors,(role=:hamiltonian,p,operator=scale.*D))
        else
            rate=_static_evans_rate(term)
            iszero(rate)&&continue
            if term isa Union{LocalJump,LocalPBodyJump}
                # The local channels impose both [L_S,X]=0 and [L_S†,X]=0.
                # Since ad_(L†)=ad_L† in Hilbert--Schmidt coordinates, their
                # positive local constraint is D†D + DD† before summing subsets.
                K=D'*D
                scratch=similar(K);mul!(scratch,D,D');K .+= scratch
                push!(descriptors,(role=:local_constraint,p,operator=rate.*K))
            else
                # Collective channels must first be summed over particles or
                # subsets.  Their positive square is therefore formed later,
                # inside each auxiliary Schur block.
                push!(descriptors,(role=:collective_constraint,p,
                                   operator=sqrt(rate).*D))
            end
        end
    end
    # Linearity lets Hamiltonian commutators and independent local constraint
    # squares of the same body order share one expensive Appendix-D
    # contraction.  Collective channels remain separate because each one must
    # be summed before it is squared.
    combined=Dict{Tuple{Symbol,Int},Matrix{ComplexF64}}();out=NamedTuple[]
    for descriptor in descriptors
        if descriptor.role===:collective_constraint
            push!(out,descriptor)
        else
            key=(descriptor.role,descriptor.p)
            if haskey(combined,key)
                combined[key].+=descriptor.operator
            else
                combined[key]=copy(descriptor.operator)
            end
        end
    end
    for key in sort!(collect(keys(combined));by=key->(string(key[1]),key[2]))
        role,p=key;push!(out,(;role,p,operator=combined[key]))
    end
    out
end

function _evans_auxiliary_block(auxiliary::PIBasis,sector::Partition,
                                descriptor,pcaches)
    # PBodyGeometry at p=1 is deliberately used here instead of the general
    # OneBodyGeometry.  Evans only needs sector-diagonal collective blocks;
    # building the latter's sector-to-sector local-kernel contractions would
    # waste substantial setup time and memory.
    geometry=get!(()->PBodyGeometry(auxiliary,descriptor.p),pcaches,
                  descriptor.p)
    pbody_collective_block(geometry,descriptor.operator,sector)
end

function _evans_auxiliary_report(model::PIModel;
                                 atol::Real,rtol::Real,return_basis::Bool,
                                 memory_budget)
    budget=_memory_budget_bytes(memory_budget)
    estimate=_evans_auxiliary_memory_estimate(model,return_basis)
    if estimate.bytes>budget
        return _evans_missing_report(model,
            "the auxiliary d^2-site Schur calculation exceeds the requested memory budget";
            memory_budget=budget,estimated_bytes=estimate.bytes,
            estimated_geometry_bytes=estimate.estimated_geometry_bytes)
    end
    descriptors=_evans_auxiliary_descriptors(model)
    descriptors===nothing&&return _evans_missing_report(model,
        "custom and direct PI terms lack the microscopic ket/bra recoupling required by the auxiliary Evans test";
        criterion=:unsupported_microscopic_recoupling)

    auxiliary=PIBasis(model.basis.N,model.basis.d^2)
    pcaches=Dict{Int,PBodyGeometry}()
    sector_data=Vector{Any}(undef,length(auxiliary.sectors))
    squared_scale=0.0;max_block_dimension=0
    for (index,sector) in pairs(auxiliary.sectors)
        n=length(auxiliary.patterns[index]);max_block_dimension=max(max_block_dimension,n)
        H=zeros(ComplexF64,n,n);K=zeros(ComplexF64,n,n);scratch=similar(K)
        for descriptor in descriptors
            block=_evans_auxiliary_block(auxiliary,sector,descriptor,pcaches)
            if descriptor.role===:hamiltonian
                H .+= block
            elseif descriptor.role===:local_constraint
                K .+= block
            else
                mul!(scratch,block',block);K .+= scratch
                mul!(scratch,block,block');K .+= scratch
            end
        end
        mul!(scratch,H',H);K .+= scratch
        # Averaging only removes multiplication/eigensolver roundoff; K is
        # mathematically Hermitian positive semidefinite by construction.
        decomposition=return_basis ? eigen(Hermitian((K+K')/2)) :
                                     eigvals(Hermitian((K+K')/2))
        values=return_basis ? decomposition.values : decomposition
        isempty(values)||(squared_scale=max(squared_scale,max(maximum(values),0.0)))
        sector_data[index]=(sector=sector,values=values,
                            vectors=return_basis ? decomposition.vectors : nothing)
    end

    requested_tolerance=Float64(atol)+Float64(rtol)*sqrt(squared_scale)
    # K is a normal-equation operator.  Exact zero modes consequently carry
    # O(eps*norm(K)) eigenvalue noise even when the requested singular-value
    # tolerance is much smaller.  Include that unavoidable floor explicitly.
    roundoff_floor=64*eps(Float64)*max(max_block_dimension,1)*squared_scale
    squared_tolerance=max(requested_tolerance^2,roundoff_floor)
    tolerance=sqrt(squared_tolerance)
    dimension=big(0);compressed_basis=NamedTuple[]
    for item in sector_data
        mask=item.values .<= squared_tolerance
        nullity=count(mask)
        multiplicity=symmetric_group_dimension(item.sector)
        dimension+=multiplicity*nullity
        if return_basis&&nullity>0
            push!(compressed_basis,(sector=item.sector,multiplicity,
                                    vectors=item.vectors[:,mask]))
        end
    end
    dimension>=1||throw(ArgumentError(
        "numerical auxiliary commutant lost the identity; increase atol or rtol"))
    result=(unique=dimension==1,certified=true,commutant_dimension=dimension,
            tolerance,requested_tolerance,squared_tolerance,
            hilbert_dimension=big(model.basis.d)^model.basis.N,
            criterion=:evans_auxiliary_pi,scope=:full_hilbert_space,
            auxiliary_local_dimension=estimate.auxiliary_local_dimension,
            auxiliary_pi_dimension=estimate.coordinates,
            memory_budget=budget,estimated_bytes=estimate.bytes,
            estimated_geometry_bytes=estimate.estimated_geometry_bytes,
            reason=dimension==1 ?
                "the auxiliary Schur decomposition proves that the joint Evans commutant is scalar" :
                "the auxiliary Schur decomposition finds a nontrivial joint Evans commutant")
    return_basis ? merge(result,(commutant_basis=compressed_basis,
        basis_representation=:auxiliary_schur_blocks,)) : result
end

function _sector_evans_operators(model::PIModel,p::Partition)
    b=model.basis;cache=OneBodyGeometry(b);pcaches=Dict{Int,PBodyGeometry}();n=length(b.patterns[b.index[p]])
    H=zeros(ComplexF64,n,n);jumps=Matrix{ComplexF64}[]
    for t in model.terms
        t.operator isa Function&&throw(ArgumentError("Evans uniqueness requires fixed operators"))
        if t isa Union{LocalHamiltonian,CollectiveHamiltonian}
            t.rate isa Number||throw(ArgumentError("Evans uniqueness requires time-independent coefficients"))
            H .+= (t.rate/t.hbar).*collective_block(b,t.operator,p;cache=cache)
        elseif t isa DirectPIHamiltonian
            t.rate isa Number||throw(ArgumentError("Evans uniqueness requires time-independent coefficients"))
            H .+= (t.rate/t.hbar).*physical_block(t.operator,p)
        elseif t isa PBodyHamiltonian
            t.rate isa Number||throw(ArgumentError("Evans uniqueness requires time-independent coefficients"))
            pc=get!(()->PBodyGeometry(b,t.p),pcaches,t.p);H .+= (t.rate/t.hbar).*pbody_collective_block(pc,t.operator,p)
        elseif t isa Union{LocalJump,CollectiveJump}
            r=_static_evans_rate(t);iszero(r)||push!(jumps,sqrt(r).*collective_block(b,t.operator,p;cache=cache))
        elseif t isa DirectPIJump
            r=_static_evans_rate(t);iszero(r)||push!(jumps,sqrt(r).*Matrix(physical_block(t.operator,p)))
        elseif t isa Union{LocalPBodyJump,CollectivePBodyJump}
            b.N==1||t isa CollectivePBodyJump||throw(ArgumentError("local p-body jumps do not preserve one Schur sector"))
            r=_static_evans_rate(t);pc=get!(()->PBodyGeometry(b,t.p),pcaches,t.p);iszero(r)||push!(jumps,sqrt(r).*pbody_collective_block(pc,t.operator,p))
        else
            throw(ArgumentError("local jump terms do not preserve one Schur sector"))
        end
    end
    H,jumps
end

"""
    evans_uniqueness(model::PIModel; atol=1e-12, rtol=1e-10,
                     memory_budget=512*1024^2)

Return a theorem-based uniqueness report without constructing the full
`d^N` Hilbert space. `unique` is `true`, `false`, or `missing` when the
efficient Evans tests cannot run within `memory_budget` or a direct/custom PI
term lacks a microscopic recoupling. After inexpensive special cases, the
general path constructs the positive joint-commutator operator on an
auxiliary PI problem with local dimension `d^2`, diagonalizes its Schur
blocks, and counts kernel vectors with their exact symmetric-group
multiplicities. It never constructs a `d^N` operator.
"""
function evans_uniqueness(model::PIModel;atol::Real=1e-12,rtol::Real=1e-10,
                          return_basis::Bool=false,memory_budget=512*1024^2)
    atol>=0||throw(ArgumentError("atol must be nonnegative"))
    rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
    budget=_memory_budget_bytes(memory_budget)
    b=model.basis
    for t in model.terms
        term_operator(t) isa Function&&throw(ArgumentError("Evans uniqueness requires fixed operators"))
        term_rate(t) isa Number||throw(ArgumentError("Evans uniqueness requires time-independent coefficients"))
    end
    all(t->t isa _BuiltinPITerm,model.terms)||return _evans_missing_report(model,
        "custom PI terms lack the microscopic ket/bra recoupling required by the auxiliary Evans test";
        criterion=:unsupported_microscopic_recoupling)
    foreach(t->_check_evans_microscopic_operator(t,b),model.terms)
    localjumps=Matrix{ComplexF64}[]
    for t in model.terms
        if t isa _HamiltonianPITerm
            _static_evans_hamiltonian_scale(t)
        elseif t isa LocalJump
            r=_static_evans_rate(t);iszero(r)||push!(localjumps,sqrt(r).*Matrix(t.operator))
        elseif t isa Union{LocalPBodyJump,CollectiveJump,DirectPIJump,CollectivePBodyJump}
            _static_evans_rate(t)
        end
    end
    if b.N==1 || length(b.sectors)==1 && isempty(localjumps) && !any(t->t isa LocalPBodyJump,model.terms)
        p=only(b.sectors);H,jumps=_sector_evans_operators(model,p)
        report=evans_uniqueness(H,jumps;atol=atol,rtol=rtol,return_basis=return_basis)
        return merge(report,(scope=b.N==1 ? :full_hilbert_space : :retained_schur_sector,sector=p))
    end
    if !isempty(localjumps)
        localreport=evans_uniqueness(zeros(ComplexF64,b.d,b.d),localjumps;atol=atol,rtol=rtol,return_basis=return_basis)
        if localreport.unique
            return merge(localreport,(scope=:full_hilbert_space,
                reason="the single-particle local-jump algebra has scalar commutant, hence its copies certify Evans uniqueness for every N"))
        end
    end
    haslocal=any(t->t isa Union{LocalJump,LocalPBodyJump}&&
                           !iszero(_static_evans_rate(t)),model.terms)
    if !haslocal
        return (unique=false,certified=true,commutant_dimension=nothing,
                tolerance=atol,hilbert_dimension=big(b.d)^b.N,criterion=:schur_sector_conservation,
                scope=:full_pi_space,conserved_sectors=length(b.sectors),
                reason="collective/direct terms conserve every retained Schur-sector population")
    end
    _evans_auxiliary_report(model;atol,rtol,return_basis,memory_budget=budget)
end

"""Boolean-or-`missing` convenience result from `evans_uniqueness(model)`."""
has_unique_steady_state_evans(model::PIModel;kwargs...)=evans_uniqueness(model;kwargs...).unique
