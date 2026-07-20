@doc raw"""
    negativity(rho, k; atol=_analysis_atol(rho), rtol=_state_rtol(rho),
               plan=nothing,
               workspace=nothing)

Compute the entanglement negativity across the bipartition `k | N-k`,

```math
\mathcal N(\rho) = (\|\rho^{T_A}\|_1-1)/2.
```

This method is polynomial in `N` for fixed local dimension `d`. Fully symmetric
states use occupation branching, general qubit states use SU(2) recoupling,
and general qudit states use Littlewood--Richardson intertwiner spaces computed
from U(d) generator equations.

On the fully symmetric branch, the calculation embeds `Sym^N(C^d)` into
`Sym^k(C^d) ⊗ Sym^(N-k)(C^d)` using exact occupation-number branching
coefficients. The partially transposed matrix has dimension
`binomial(k+d-1,k) * binomial(N-k+d-1,N-k)`.

Pass a `ReductionPlan(rho.basis,k)` to reuse the product-Schur recoupling or
subduction data across repeated states. A caller-owned
`ReductionWorkspace(plan,rho)` additionally reuses the numerical product-block
and partial-transpose scratch.
"""
function negativity(rho::PIState, k::Integer;
                    atol::Real=_analysis_atol(rho),
                    rtol::Real=_state_rtol(rho),
                    plan=nothing,workspace=nothing)
    b=rho.basis
    0 <= k <= b.N || throw(ArgumentError("bipartition size k must satisfy 0 ≤ k ≤ N"))
    validate_state(rho;atol=atol,rtol=rtol)
    if plan!==nothing||workspace!==nothing
        plan,workspace=_resolve_reduction_resources(b,k,plan,workspace;atol=atol)
        if workspace===nothing
            return _plan_negativity(rho,plan)
        end
        _check_reduction_workspace(workspace,plan,rho)
        return _plan_negativity(rho,plan,workspace;atol=atol,rtol=rtol)
    end

    # Every U(2) outer product is multiplicity-free.  This permits a complete
    # product-Schur construction without storing symmetric-group tableaux.
    b.d==2 && return _qubit_negativity(rho,k;atol=atol)

    sym=Partition(ntuple(i -> i==1 ? b.N : 0,b.d))
    haskey(b.index,sym) || throw(ArgumentError("the fully symmetric sector $sym is absent from this basis"))
    nonsymmetric=false
    for p in b.sectors
        p==sym && continue
        nonsymmetric |= norm(coefficient_block(rho,p)) > atol
    end
    nonsymmetric && return _qudit_negativity(rho,k;atol=atol)
    (k==0 || k==b.N) && return 0.0

    pa=Partition(ntuple(i -> i==1 ? k : 0,b.d))
    pb=Partition(ntuple(i -> i==1 ? b.N-k : 0,b.d))
    ga=gt_patterns(pa); gb=gt_patterns(pb); gn=b.patterns[b.index[sym]]
    ia=Dict(content(g)=>i for (i,g) in pairs(ga))
    ib=Dict(content(g)=>i for (i,g) in pairs(gb))
    da=length(ga); db=length(gb); dn=length(gn)
    V=zeros(ComplexF64,da*db,dn)
    denom=binomial(big(b.N),big(k))
    for (col,g) in pairs(gn)
        n=content(g)
        # Enumerate subsystem-A occupations recursively; B is fixed by n-a.
        function branch!(a::Vector{Int},level::Int,left::Int)
            if level==b.d
                left<=n[level] || return
                push!(a,left)
                aa=Tuple(a); bb=ntuple(i->n[i]-aa[i],b.d)
                weighta=prod(binomial(big(n[i]),big(aa[i])) for i in 1:b.d)
                row=ia[aa]+(ib[bb]-1)*da
                V[row,col]=sqrt(Float64(weighta//denom))
                pop!(a); return
            end
            for x in 0:min(n[level],left)
                push!(a,x);branch!(a,level+1,left-x);pop!(a)
            end
        end
        branch!(Int[],1,k)
    end

    rhoab=V*Matrix(physical_block(rho,sym))*V'
    tensor=reshape(rhoab,da,db,da,db)
    pt=reshape(permutedims(tensor,(3,2,1,4)),da*db,da*db)
    vals=eigvals(Hermitian((pt+pt')/2))
    max(0.0,(sum(abs,vals)-1)/2)
end

_fact(n::Integer)=n<0 ? zero(BigInt) : factorial(big(n))

# Qubit product-Schur construction evaluates many Clebsch--Gordan
# coefficients with the same bounded set of factorial arguments.  Keep this
# cache local to one ReductionPlan setup: plans remain immutable and ordinary
# one-off coefficient calls retain their allocation-light uncached route.
struct _SU2FactorialCache
    values::Vector{BigInt}
end

function _SU2FactorialCache(maximum_argument::Integer)
    maximum_argument>=0||throw(ArgumentError(
        "maximum factorial argument must be nonnegative"))
    nmax=Int(maximum_argument)
    values=Vector{BigInt}(undef,nmax+1)
    values[1]=one(BigInt)
    @inbounds for n in 1:nmax
        values[n+1]=values[n]*n
    end
    _SU2FactorialCache(values)
end

@inline _fact(::Nothing,n::Integer)=_fact(n)
@inline function _fact(cache::_SU2FactorialCache,n::Integer)
    n<0&&return zero(BigInt)
    ni=Int(n)
    ni<length(cache.values)||throw(ArgumentError(
        "factorial argument $ni exceeds the prepared SU(2) cache"))
    @inbounds cache.values[ni+1]
end

function _su2_selection_rule(j1::Int,m1::Int,j2::Int,m2::Int,j::Int,m::Int)
    m1+m2==m || return false
    abs(m1)<=j1 && abs(m2)<=j2 && abs(m)<=j || return false
    iseven(j1-m1) && iseven(j2-m2) && iseven(j-m) || return false
    abs(j1-j2)<=j<=j1+j2 || return false
    iseven(j1+j2-j)
end

# Allocation-light route retained for the ordinary small-spin reduction path.
# Its separately converted factorial products remain finite throughout this
# deliberately conservative range.
_su2_cgc_small(j1::Int,m1::Int,j2::Int,m2::Int,j::Int,m::Int)=
    _su2_cgc_small(nothing,j1,m1,j2,m2,j,m)

function _su2_cgc_small(factorials::Union{Nothing,_SU2FactorialCache},
                        j1::Int,m1::Int,j2::Int,m2::Int,j::Int,m::Int)
    a=(j1+j2-j)÷2; bb=(j1-j2+j)÷2; c=(-j1+j2+j)÷2
    pref=sqrt(Float64((big(j+1)*_fact(factorials,a)*_fact(factorials,bb)*
        _fact(factorials,c))//_fact(factorials,(j1+j2+j)÷2+1)))
    nums=_fact(factorials,(j1+m1)÷2)*_fact(factorials,(j1-m1)÷2)*
         _fact(factorials,(j2+m2)÷2)*_fact(factorials,(j2-m2)÷2)*
         _fact(factorials,(j+m)÷2)*_fact(factorials,(j-m)÷2)
    pref*=sqrt(Float64(nums))
    lo=max(0,-((j-j2+m1)÷2),-((j-j1-m2)÷2))
    hi=min(a,(j1-m1)÷2,(j2+m2)÷2)
    s=0.0
    for z in lo:hi
        den=_fact(factorials,z)*_fact(factorials,a-z)*
            _fact(factorials,(j1-m1)÷2-z)*
            _fact(factorials,(j2+m2)÷2-z)*
            _fact(factorials,(j-j2+m1)÷2+z)*
            _fact(factorials,(j-j1-m2)÷2+z)
        s += isodd(z) ? -inv(Float64(den)) : inv(Float64(den))
    end
    pref*s
end

# Stable Racah sum.  The alternating sum is accumulated exactly as a
# Rational{BigInt}; only the final nonnegative squared coefficient is
# converted, after cancellation, with the binary-scaled square-root helper.
# Consecutive summands are generated by a rational recurrence, so the large
# path needs only the six factorials in its first denominator rather than six
# new factorials at every summation index.
_su2_cgc_exact(j1::Int,m1::Int,j2::Int,m2::Int,j::Int,m::Int)=
    _su2_cgc_exact(nothing,j1,m1,j2,m2,j,m)

function _su2_cgc_exact(factorials::Union{Nothing,_SU2FactorialCache},
                        j1::Int,m1::Int,j2::Int,m2::Int,j::Int,m::Int)
    a=(j1+j2-j)÷2; bb=(j1-j2+j)÷2; c=(-j1+j2+j)÷2
    A=(j1-m1)÷2; B=(j2+m2)÷2
    C=(j-j2+m1)÷2; D=(j-j1-m2)÷2
    lo=max(0,-C,-D); hi=min(a,A,B)
    lo<=hi||return 0.0

    prefactor_squared=
        (big(j+1)*_fact(factorials,a)*_fact(factorials,bb)*
         _fact(factorials,c))//_fact(factorials,(j1+j2+j)÷2+1)
    prefactor_squared*=
        _fact(factorials,(j1+m1)÷2)*_fact(factorials,A)*
        _fact(factorials,(j2+m2)÷2)*_fact(factorials,(j2-m2)÷2)*
        _fact(factorials,(j+m)÷2)*_fact(factorials,(j-m)÷2)

    first_denominator=
        _fact(factorials,lo)*_fact(factorials,a-lo)*
        _fact(factorials,A-lo)*_fact(factorials,B-lo)*
        _fact(factorials,C+lo)*_fact(factorials,D+lo)
    magnitude=one(BigInt)//first_denominator
    racah_sum=isodd(lo) ? -magnitude : magnitude
    for z in lo:hi-1
        # |t_(z+1)/t_z|, with the alternating sign applied only when adding
        # the new term.  Every factor is positive on the selected range.
        magnitude*=
            (big(a-z)*big(A-z)*big(B-z))//
            (big(z+1)*big(C+z+1)*big(D+z+1))
        racah_sum+=isodd(z+1) ? -magnitude : magnitude
    end
    iszero(racah_sum)&&return 0.0
    squared=prefactor_squared*abs2(racah_sum)
    root=_checked_sqrt_exact_ratio(
        Float64,numerator(squared),denominator(squared);
        context="SU(2) Clebsch--Gordan coefficient")
    racah_sum<0 ? -root : root
end

# Standard Condon--Shortley SU(2) coefficient. All quantum numbers are stored
# doubled, making every factorial argument above an exact integer.  Small
# spins keep the previous fast path; larger spins use exact cancellation and
# a single scaled conversion, avoiding factorial overflow and Inf*0 products.
_su2_cgc(j1::Int,m1::Int,j2::Int,m2::Int,j::Int,m::Int)=
    _su2_cgc(nothing,j1,m1,j2,m2,j,m)

function _su2_cgc(factorials::Union{Nothing,_SU2FactorialCache},
                  j1::Int,m1::Int,j2::Int,m2::Int,j::Int,m::Int)
    _su2_selection_rule(j1,m1,j2,m2,j,m)||return 0.0
    # The alternating Float64 Racah sum begins losing several ulps in complete
    # coupled columns above doubled spin 32 (already ~2e-13 at 48 and ~6e-12
    # in individual entries at 64).  Keep it only where orthogonality remains
    # at the ordinary roundoff scale; larger labels use exact cancellation.
    max(j1,j2,j)<=32 && return _su2_cgc_small(
        factorials,j1,m1,j2,m2,j,m)
    _su2_cgc_exact(factorials,j1,m1,j2,m2,j,m)
end

_qubit_sectors(n::Int)=[(Partition((n-r,r)),n-2r) for r in 0:fld(n,2)]

function _pattern_m2(g::GTPattern{2},n::Int)
    # The package computational ordering follows content occupations directly:
    # local basis index 1 is the lower SU(2) weight.
    2content(g)[1]-n
end

"""Exact product-Schur partial-transpose trace norm for a general PI qubit state."""
function _qubit_negativity(rho::PIState,k::Integer;atol::Real=1e-12)
    b=rho.basis
    (k==0 || k==b.N) && return 0.0
    # Every recoupling coefficient below draws factorials from the same
    # bounded range.  Preparing that range once avoids rebuilding identical
    # BigInt factorials for every product-basis entry while retaining the
    # exact-cancellation route used for large doubled spins.
    factorials=_SU2FactorialCache(b.N+1)
    _qubit_negativity(rho,k,factorials;atol=atol)
end

# The explicit cache argument keeps an uncached oracle available to focused
# tests and benchmarks without changing the public negativity API.
function _qubit_negativity(rho::PIState,k::Integer,
                           factorials::Union{Nothing,_SU2FactorialCache};
                           atol::Real=1e-12)
    b=rho.basis; na=k; nb=b.N-k
    (na==0 || nb==0) && return 0.0
    total_norm=0.0
    for (pa,ja) in _qubit_sectors(na), (pb,jb) in _qubit_sectors(nb)
        parent_sectors=Partition{2}[]
        for p in b.sectors
            j=b.N-2p[2]
            abs(ja-jb)<=j<=ja+jb&&iseven(ja+jb-j)&&push!(parent_sectors,p)
        end
        isempty(parent_sectors)&&continue
        da=ja+1; db=jb+1
        block=zeros(ComplexF64,da*db,da*db)
        product_multiplicity=
            symmetric_group_dimension(pa)*symmetric_group_dimension(pb)
        scale_squared=product_multiplicity^2
        ma=collect(-ja:2:ja); mb=collect(-jb:2:jb)
        for p in parent_sectors
            j=b.N-2p[2]
            C=_scaled_reduction_parent_block(rho,p,scale_squared)
            patterns=b.patterns[b.index[p]]
            m_to_i=Dict(_pattern_m2(g,b.N)=>i for (i,g) in pairs(patterns))
            # Isometry from the total-J GT basis into the product basis.
            U=zeros(Float64,da*db,j+1)
            for (ia,m1) in pairs(ma),(ib,m2) in pairs(mb)
                m=m1+m2; abs(m)<=j || continue
                U[ia+(ib-1)*da,m_to_i[m]]=
                    _su2_cgc(factorials,ja,m1,jb,m2,j,m)
            end
            block .+= U*C*U'
        end
        tensor=reshape(block,da,db,da,db)
        pt=reshape(permutedims(tensor,(3,2,1,4)),da*db,da*db)
        vals=eigvals(Hermitian((pt+pt')/2))
        total_norm += sum(abs,vals)
    end
    max(0.0,(total_norm-1)/2)
end

# U(d) representation matrices are obtained from the already validated
# equation-(31) collective blocks, restricted to one partition.
function _irrep_generators(p::Partition{D}) where D
    b=PIBasis(weight(p),D;sectors=[p.parts]); cache=OneBodyGeometry(b)
    gens=Matrix{ComplexF64}[]
    for i in 1:D
        E=zeros(ComplexF64,D,D);E[i,i]=1
        push!(gens,collective_block(b,E,p;cache=cache))
    end
    for i in 1:D-1
        E=zeros(ComplexF64,D,D);E[i,i+1]=1
        push!(gens,collective_block(b,E,p;cache=cache))
        push!(gens,collective_block(b,E',p;cache=cache))
    end
    gens
end

# Exact Littlewood--Richardson coefficient from semistandard skew tableaux.
# Boxes are visited in the LR reading order (right-to-left, top-to-bottom), so
# the lattice-word condition can be checked on every prefix and no tableaux
# need to be retained.
function _lr_tableau_coefficient(alpha::Partition{D},beta::Partition{D},
                                 lambda::Partition{D}) where D
    weight(alpha)+weight(beta)==weight(lambda)||return big(0)
    all(i->lambda[i]>=alpha[i],1:D)||return big(0)
    boxes=Tuple{Int,Int}[]
    for row in 1:D,column in lambda[row]:-1:alpha[row]+1
        push!(boxes,(row,column))
    end
    length(boxes)==weight(beta)||return big(0)
    isempty(boxes)&&return big(1)
    remaining=collect(beta.parts);used=zeros(Int,D)
    entries=Dict{Tuple{Int,Int},Int}();count=Ref(big(0))
    function visit!(position::Int)
        if position>length(boxes)
            count[]+=1;return
        end
        row,column=boxes[position]
        right=get(entries,(row,column+1),D)
        above=get(entries,(row-1,column),0)
        for value in 1:D
            remaining[value]>0||continue
            value<=right||continue                 # weak rows
            (row==1||above==0||above<value)||continue # strict columns
            remaining[value]-=1;used[value]+=1
            lattice=all(used[level]>=used[level+1] for level in 1:D-1)
            if lattice
                entries[(row,column)]=value;visit!(position+1)
                delete!(entries,(row,column))
            end
            used[value]-=1;remaining[value]+=1
        end
    end
    visit!(1);count[]
end

function _weight_restricted_variables(ga,gb,gl)
    da=length(ga);db=length(gb);D=length(content(first(gl)))
    product_weights=Vector{NTuple{D,Int}}(undef,da*db)
    for ib in 1:db,ia in 1:da
        wa=content(ga[ia]);wb=content(gb[ib])
        product_weights[ia+(ib-1)*da]=ntuple(level->wa[level]+wb[level],D)
    end
    parent_by_weight=Dict{NTuple{D,Int},Vector{Int}}()
    for (column,g) in pairs(gl);push!(get!(()->Int[],parent_by_weight,content(g)),column);end
    variables=Tuple{Int,Int}[]
    for row in eachindex(product_weights),column in get(parent_by_weight,product_weights[row],Int[])
        push!(variables,(row,column))
    end
    variables
end

function _sparse_lr_nullspace(genera,genb,genl,patterna,patternb,patternl,
                              expected::Int;atol::Real)
    da=size(genera[1],1);db=size(genb[1],1);dl=size(genl[1],1);dp=da*db
    variables=_weight_restricted_variables(patterna,patternb,patternl)
    nvariables=length(variables)
    expected<=nvariables||throw(ErrorException(
        "LR coefficient $expected exceeds the weight-restricted variable count $nvariables"))
    expected==0&&return zeros(ComplexF64,dp*dl,0)

    # Cartan equations vanish identically on the equal-weight variables. Only
    # the simple-root raising/lowering equations remain, assembled directly in
    # sparse triplet form. Rows are introduced lazily, so identically zero
    # equations consume no storage.
    rowmap=Dict{Tuple{Int,Int,Int},Int}();rows=Int[];cols=Int[];values=ComplexF64[]
    equation(gen,row,column)=get!(rowmap,(gen,row,column)) do;length(rowmap)+1;end
    identitya=sparse(I,da,da);identityb=sparse(I,db,db)
    # Recover d from the `3d-2` cached Cartan/simple-root generators.
    D=(length(genl)+2)÷3
    for generator in D+1:length(genl)
        product_generator=kron(identityb,sparse(genera[generator]))+
                          kron(sparse(genb[generator]),identitya)
        for (variable,(source_row,parent_column)) in pairs(variables)
            for pointer in nzrange(product_generator,source_row)
                target_row=product_generator.rowval[pointer]
                coefficient=product_generator.nzval[pointer]
                iszero(coefficient)&&continue
                push!(rows,equation(generator,target_row,parent_column))
                push!(cols,variable);push!(values,coefficient)
            end
            for target_column in 1:dl
                coefficient=genl[generator][parent_column,target_column]
                iszero(coefficient)&&continue
                push!(rows,equation(generator,source_row,target_column))
                push!(cols,variable);push!(values,-coefficient)
            end
        end
    end
    constraints=sparse(rows,cols,values,length(rowmap),nvariables)
    factorization=qr(constraints;tol=atol)
    R=factorization.R;rank=nvariables-expected
    rank<=min(size(R)...)||throw(ErrorException(
        "sparse LR system has too few independent equations for coefficient $expected"))
    scale=maximum(abs,R;init=0.0);threshold=atol*max(scale,1.0)
    rank>0&&abs(R[rank,rank])<=threshold&&throw(ArgumentError(
        "sparse LR constraints are numerically rank deficient; decrease atol or use algorithm=:dense"))
    if rank<min(size(R)...)
        abs(R[rank+1,rank+1])<=100threshold||throw(ArgumentError(
            "sparse LR nullity disagrees with the exact tableau coefficient; increase atol"))
    end
    permuted=zeros(ComplexF64,nvariables,expected)
    if rank>0
        right=Matrix(R[1:rank,rank+1:nvariables])
        permuted[1:rank,:].=-(UpperTriangular(R[1:rank,1:rank])\right)
    end
    permuted[rank+1:end,:].=Matrix{ComplexF64}(I,expected,expected)
    restricted=zeros(ComplexF64,nvariables,expected)
    restricted[factorization.pcol,:].=permuted
    orthonormal=Matrix(qr(restricted).Q[:,1:expected])
    full=zeros(ComplexF64,dp*dl,expected)
    for (variable,(row,column)) in pairs(variables)
        full[row+(column-1)*dp,:].=orthonormal[variable,:]
    end
    residual=norm(constraints*orthonormal)
    residual<=max(atol,100eps(Float64))*max(norm(constraints),1)||
        throw(ArgumentError("sparse LR intertwiner residual $residual exceeds tolerance"))
    full
end

function _dense_lr_nullspace(genera,genb,genl;atol::Real)
    da=size(genera[1],1);db=size(genb[1],1);dl=size(genl[1],1);dp=da*db
    identitya=Matrix{ComplexF64}(I,da,da)
    identityb=Matrix{ComplexF64}(I,db,db)
    identityl=Matrix{ComplexF64}(I,dl,dl)
    identityp=Matrix{ComplexF64}(I,dp,dp)
    constraints=Matrix{ComplexF64}[]
    for generator in eachindex(genl)
        product_generator=kron(identityb,genera[generator])+
                          kron(genb[generator],identitya)
        push!(constraints,kron(identityl,product_generator)-
                          kron(transpose(genl[generator]),identityp))
    end
    nullspace(reduce(vcat,constraints);atol=atol)
end

"""
Return an orthonormal basis of Littlewood--Richardson intertwiners
`U(lambda) -> U(alpha) ⊗ U(beta)`. Its length is the LR coefficient.

The nullspace formulation avoids enumerating symmetric-group tableaux.  By
Schur--Weyl duality this is the same multiplicity space that occurs in the
restriction `S(lambda) ↓ S(a) × S(b)`.
"""
function _lr_intertwiners(alpha::Partition{D},beta::Partition{D},lambda::Partition{D};
                          atol::Real=2e-11,algorithm::Symbol=:auto,
                          gencache=Dict{Partition,Vector{Matrix{ComplexF64}}}()) where D
    algorithm in (:auto,:sparse,:dense)||
        throw(ArgumentError("LR algorithm must be :auto, :sparse, or :dense"))
    weight(alpha)+weight(beta)==weight(lambda) || return Matrix{ComplexF64}[]
    all(i->lambda[i]>=alpha[i]&&lambda[i]>=beta[i],1:D) || return Matrix{ComplexF64}[]
    coefficient=_lr_tableau_coefficient(alpha,beta,lambda)
    iszero(coefficient)&&return Matrix{ComplexF64}[]
    coefficient<=typemax(Int)||throw(OverflowError("LR coefficient is too large to materialize"))
    expected=Int(coefficient)
    genera=get!(gencache,alpha) do; _irrep_generators(alpha); end
    genb=get!(gencache,beta) do; _irrep_generators(beta); end
    genl=get!(gencache,lambda) do; _irrep_generators(lambda); end
    dl=size(genl[1],1);dp=size(genera[1],1)*size(genb[1],1)
    selected=algorithm===:auto ? :sparse : algorithm
    Z = if selected===:dense
        _dense_lr_nullspace(genera,genb,genl;atol=atol)
    else
        _sparse_lr_nullspace(genera,genb,genl,gt_patterns(alpha),
                             gt_patterns(beta),gt_patterns(lambda),expected;
                             atol=atol)
    end
    size(Z,2)==expected||throw(ArgumentError(
        "$selected LR nullity $(size(Z,2)) disagrees with exact coefficient $expected"))
    out=Matrix{ComplexF64}[]
    for r in axes(Z,2)
        T=reshape(Z[:,r],dp,dl)*sqrt(dl)
        # Numerical nullspaces may choose arbitrary LR multiplicity rotations;
        # Frobenius orthogonality is exactly the required intertwiner metric.
        push!(out,T)
    end
    out
end

"""
    subduction_intertwiners(alpha, beta, lambda; atol=2e-11,
                            algorithm=:auto)

Return an orthonormal set of intertwiners from `U(lambda)` into
`U(alpha) ⊗ U(beta)`. By Schur--Weyl duality these span the symmetric-group
subduction multiplicity space for
`S(lambda) ↓ S(weight(alpha)) × S(weight(beta))`. Each returned matrix has
orthonormal columns; its row basis is the product GT basis.

The number of matrices is the Littlewood--Richardson coefficient
`c^lambda_{alpha,beta}`.

The default backend removes forbidden weights first, assembles sparse
simple-root constraints, and uses a rank-revealing SPQR factorization.
`algorithm=:dense` retains the former stacked Kronecker construction as a
small-problem reference.
"""
subduction_intertwiners(alpha::Partition{D},beta::Partition{D},lambda::Partition{D};
                        atol=2e-11,algorithm::Symbol=:auto) where D =
    _lr_intertwiners(alpha,beta,lambda;atol=atol,algorithm=algorithm)

"""
Return the exact Littlewood--Richardson coefficient `c^lambda_{alpha,beta}`.
The lattice-tableau recursion performs no numerical nullspace construction.
"""
littlewood_richardson_coefficient(alpha::Partition{D},beta::Partition{D},
                                  lambda::Partition{D};atol=nothing) where D =
    _lr_tableau_coefficient(alpha,beta,lambda)

struct _ProductSchurCoupling{T,D}
    alpha::Partition{D}
    beta::Partition{D}
    da::Int
    db::Int
    alpha_multiplicity::BigInt
    beta_multiplicity::BigInt
    product_multiplicity::BigInt
    # Entries are `(parent-sector index, multiplicity-space intertwiners)`.
    intertwiners::Vector{Tuple{Int,Vector{Matrix{T}}}}
end

"""
    ReductionPlan(basis, k; atol=2e-11)

Prepare product-Schur intertwiners for the fixed bipartition `k | N-k`.
Reuse the plan with `reduced_state`, `reduced_purity`, `negativity`, and
`partial_transpose_spectrum` for many states on the exact same `PIBasis`.
Qubits store their multiplicity-free SU(2) recoupling matrices; qudits store
the Littlewood--Richardson intertwiner spaces.  No full `d^N` object is built.

The plan retains every required product-Schur intertwiner but no mutable
application scratch.  It is therefore most useful for state or parameter
scans at fixed `(basis,k)`. For qudits, exact lattice-tableau counts first
remove forbidden weight spaces, after which sparse simple-root constraints
are factorized with SPQR. The resulting intertwiner matrices are dense and can
still require substantial temporary and retained memory; a one-off call may
be cheaper without a retained plan.
"""
struct ReductionPlan{T,D,B<:PIBasis,O<:PIBasis}
    basis::B
    k::Int
    output_basis::O
    couplings::Vector{_ProductSchurCoupling{T,D}}
    atol::Float64
end
function show(io::IO,p::ReductionPlan)
    nintertwiners=sum(length(Ts) for c in p.couplings for (_,Ts) in c.intertwiners)
    print(io,"ReductionPlan(N=$(p.basis.N), d=$(p.basis.d), k=$(p.k), couplings=$(length(p.couplings)), intertwiners=$nintertwiners)")
end

"""
    ReductionWorkspace(plan; T=Float64, mode=:both)
    ReductionWorkspace(plan, rho; mode=:both)

Allocate mutable scratch for repeated applications of a [`ReductionPlan`](@ref).
The workspace reuses the largest product-Schur block, multiplication
intermediate, parent block, partial trace, partial transpose, and all
reduced-sector blocks. It also prepares the exact parent multiplicity factors
and scalar-type-matched recoupling matrices once in the workspace precision,
so repeated applications do not redo `BigInt`/rational setup or trigger
mixed-eltype matrix-multiplication fallbacks. The immutable plan keeps its
compact recoupling representation (real for qubits); an already type-matched
plan matrix is shared rather than copied into the workspace.
Use `mode=:reduction` when only reduced states or purities are required, or
`mode=:negativity` when only negativity is required. These modes omit the
same-size partial-transpose matrix or the partial-trace/reduced-block storage,
respectively. The default `mode=:both` preserves the general-purpose
workspace. A mode-specific workspace raises if passed to an incompatible
operation instead of allocating missing scratch implicitly.

It belongs to the exact `ReductionPlan` used at construction and must be used
by only one task at a time.

`ReductionWorkspace(plan, rho)` selects a scalar type that can represent both
the cached intertwiners and `rho`.  The keyword constructor interprets `T` as
the real floating-point component type, consistently with `PIState(...; T=...)`.
"""
mutable struct ReductionWorkspace{T,P<:ReductionPlan,S}
    plan::P
    Ttype::Type{T}
    product_block::Matrix{T}
    product_tmp::Matrix{T}
    parent_block::Matrix{T}
    partial_trace::Matrix{T}
    partial_transpose::Matrix{T}
    reduced_blocks::Vector{Matrix{T}}
    recoupling_intertwiners::Vector{Vector{Vector{Matrix{T}}}}
    reduction_parent_scales::S
    negativity_parent_scales::S
    mode::Symbol
end

_workspace_intertwiner(::Type{T},U::Matrix{T}) where T=U
_workspace_intertwiner(::Type{T},U::AbstractMatrix) where T=Matrix{T}(U)

function ReductionWorkspace(plan::ReductionPlan{G};T::Type{R}=Float64,
                            mode::Symbol=:both) where {G,R<:AbstractFloat}
    mode in (:both,:reduction,:negativity)||throw(ArgumentError(
        "ReductionWorkspace mode must be :both, :reduction, or :negativity"))
    CT=promote_type(Complex{R},G)
    max_product=maximum((c.da*c.db for c in plan.couplings);init=1)
    max_parent=maximum((size(U,2) for c in plan.couplings for (_,Us) in c.intertwiners
                                  for U in Us);init=1)
    max_output=maximum((length(patterns) for patterns in plan.output_basis.patterns);init=1)
    reduced=mode===:negativity ? Matrix{CT}[] :
        [zeros(CT,length(patterns),length(patterns))
         for patterns in plan.output_basis.patterns]
    recouplers=Vector{Vector{Vector{Matrix{CT}}}}(undef,length(plan.couplings))
    Scale=_PreparedExactScale{_real_float_type(CT),true}
    reduction_scales=Vector{Vector{Scale}}(undef,length(plan.couplings))
    negativity_scales=Vector{Vector{Scale}}(undef,length(plan.couplings))
    for (coupling_index,coupling) in pairs(plan.couplings)
        reduction_numerator=coupling.alpha_multiplicity*
            coupling.beta_multiplicity^2
        negativity_numerator=coupling.product_multiplicity^2
        recouplers[coupling_index]=Vector{Vector{Matrix{CT}}}(
            undef,length(coupling.intertwiners))
        reduction_scales[coupling_index]=Scale[]
        negativity_scales[coupling_index]=Scale[]
        for (connection_index,(sector_index,intertwiners)) in
            pairs(coupling.intertwiners)
            recouplers[coupling_index][connection_index]=
                [_workspace_intertwiner(CT,U) for U in intertwiners]
            denominator=symmetric_group_dimension(
                plan.basis.sectors[sector_index])
            if mode!==:negativity
                push!(reduction_scales[coupling_index],_prepare_exact_scale(
                    _real_float_type(CT),reduction_numerator,denominator,Val(true);
                    context="prepared reduced-state parent scale"))
            end
            if mode!==:reduction
                push!(negativity_scales[coupling_index],_prepare_exact_scale(
                    _real_float_type(CT),negativity_numerator,denominator,Val(true);
                    context="prepared partial-transpose parent scale"))
            end
        end
    end
    ReductionWorkspace{CT,typeof(plan),typeof(reduction_scales)}(
        plan,CT,zeros(CT,max_product,max_product),
        zeros(CT,max_product,max_parent),zeros(CT,max_parent,max_parent),
        mode===:negativity ? zeros(CT,0,0) : zeros(CT,max_output,max_output),
        mode===:reduction ? zeros(CT,0,0) : zeros(CT,max_product,max_product),
        reduced,recouplers,reduction_scales,negativity_scales,mode)
end

function ReductionWorkspace(plan::ReductionPlan,rho::PIState;
                            mode::Symbol=:both)
    _check_reduction_plan(plan,rho.basis,plan.k)
    ReductionWorkspace(plan;T=_real_float_type(eltype(rho.data)),mode)
end

function show(io::IO,w::ReductionWorkspace)
    print(io,"ReductionWorkspace(k=$(w.plan.k), mode=$(w.mode), scalar_type=$(w.Ttype), " *
             "max_product_dimension=$(size(w.product_block,1)))")
end

_qubit_reduction_couplings(b::PIBasis{2},k::Int)=
    _qubit_reduction_couplings(b,k,_SU2FactorialCache(b.N+1))

function _qubit_reduction_couplings(b::PIBasis{2},k::Int,
                                    factorials::Union{Nothing,_SU2FactorialCache})
    na=k;nb=b.N-k;out=_ProductSchurCoupling{Float64,2}[]
    m_maps=[Dict(_pattern_m2(g,b.N)=>i for (i,g) in pairs(patterns)) for patterns in b.patterns]
    for (pa,ja) in _qubit_sectors(na),(pb,jb) in _qubit_sectors(nb)
        da=ja+1;db=jb+1;ma=collect(-ja:2:ja);mb=collect(-jb:2:jb)
        intertwiners=Tuple{Int,Vector{Matrix{Float64}}}[]
        for (s,p) in pairs(b.sectors)
            j=b.N-2p[2]
            abs(ja-jb)<=j<=ja+jb&&iseven(ja+jb-j)||continue
            U=zeros(Float64,da*db,j+1);m_to_i=m_maps[s]
            for (ia,m1) in pairs(ma),(ib,m2) in pairs(mb)
                m=m1+m2;abs(m)<=j||continue
                U[ia+(ib-1)*da,m_to_i[m]]=
                    _su2_cgc(factorials,ja,m1,jb,m2,j,m)
            end
            push!(intertwiners,(s,Matrix{Float64}[U]))
        end
        isempty(intertwiners)&&continue
        fa=big(symmetric_group_dimension(pa));fb=big(symmetric_group_dimension(pb))
        push!(out,_ProductSchurCoupling{Float64,2}(pa,pb,da,db,fa,fb,fa*fb,intertwiners))
    end
    out
end

function _qudit_reduction_couplings(b::PIBasis{D},k::Int,atol::Real) where D
    na=k;nb=b.N-k;out=_ProductSchurCoupling{ComplexF64,D}[]
    cache=Dict{Tuple{Partition{D},Partition{D},Partition{D}},Vector{Matrix{ComplexF64}}}()
    gencache=Dict{Partition{D},Vector{Matrix{ComplexF64}}}()
    for alpha in partitions(na,D),beta in partitions(nb,D)
        da=Int(unitary_group_dimension(alpha));db=Int(unitary_group_dimension(beta))
        intertwiners=Tuple{Int,Vector{Matrix{ComplexF64}}}[]
        for (s,lambda) in pairs(b.sectors)
            key=(alpha,beta,lambda)
            Ts=get!(cache,key) do
                _lr_intertwiners(alpha,beta,lambda;atol=atol,gencache=gencache)
            end
            isempty(Ts)||push!(intertwiners,(s,Ts))
        end
        isempty(intertwiners)&&continue
        fa=big(symmetric_group_dimension(alpha));fb=big(symmetric_group_dimension(beta))
        push!(out,_ProductSchurCoupling{ComplexF64,D}(alpha,beta,da,db,fa,fb,fa*fb,intertwiners))
    end
    out
end

function ReductionPlan(b::PIBasis{D},k::Integer;atol::Real=2e-11) where D
    0<=k<=b.N||throw(ArgumentError("subsystem size k must satisfy 0 ≤ k ≤ N"))
    atol>=0||throw(ArgumentError("atol must be nonnegative"));ki=Int(k)
    output_basis=ki==b.N ? b : PIBasis(ki,D)
    if ki==0||ki==b.N
        T=D==2 ? Float64 : ComplexF64
        couplings=_ProductSchurCoupling{T,D}[]
        return ReductionPlan{T,D,typeof(b),typeof(output_basis)}(b,ki,output_basis,couplings,Float64(atol))
    end
    if D==2
        couplings=_qubit_reduction_couplings(b,ki)
        ReductionPlan{Float64,D,typeof(b),typeof(output_basis)}(b,ki,output_basis,couplings,Float64(atol))
    else
        tol=max(atol,2e-11);couplings=_qudit_reduction_couplings(b,ki,tol)
        ReductionPlan{ComplexF64,D,typeof(b),typeof(output_basis)}(b,ki,output_basis,couplings,Float64(tol))
    end
end
ReductionPlan(rho::PIState,k::Integer;kwargs...)=ReductionPlan(rho.basis,k;kwargs...)

function _check_reduction_plan(plan::ReductionPlan,b::PIBasis,k::Integer)
    plan.basis===b||throw(ArgumentError("ReductionPlan was prepared for a different PIBasis"))
    plan.k==k||throw(ArgumentError("ReductionPlan has k=$(plan.k), but the requested subsystem has k=$k"))
    plan
end

function _check_reduction_workspace(work::ReductionWorkspace,plan::ReductionPlan,
                                    rho::PIState)
    work.plan===plan||throw(ArgumentError(
        "ReductionWorkspace was prepared for a different ReductionPlan"))
    _check_reduction_plan(plan,rho.basis,plan.k)
    promote_type(work.Ttype,eltype(rho.data))===work.Ttype||throw(ArgumentError(
        "ReductionWorkspace scalar type $(work.Ttype) cannot represent state scalar type $(eltype(rho.data))"))
    work
end

function _require_reduction_workspace_mode(work::ReductionWorkspace,
                                           operation::Symbol)
    required=operation===:reduction ? (:both,:reduction) :
             operation===:negativity ? (:both,:negativity) : ()
    work.mode in required||throw(ArgumentError(
        "ReductionWorkspace(mode=$(work.mode)) cannot perform $operation; " *
        "construct it with mode=:$operation or mode=:both"))
    work
end

function _resolve_reduction_resources(b::PIBasis,k::Integer,plan,workspace;
                                      atol::Real)
    workspace===nothing||workspace isa ReductionWorkspace||throw(ArgumentError(
        "workspace must be a ReductionWorkspace"))
    if plan===nothing
        plan=workspace===nothing ? ReductionPlan(b,k;atol=max(atol,2e-11)) :
                                  workspace.plan
    else
        plan isa ReductionPlan||throw(ArgumentError("plan must be a ReductionPlan"))
    end
    _check_reduction_plan(plan,b,k)
    workspace!==nothing&&workspace.plan!==plan&&throw(ArgumentError(
        "ReductionWorkspace was prepared for a different ReductionPlan"))
    plan,workspace
end

function _scaled_reduction_parent_block(rho::PIState,p::Partition,
                                        scale_squared_numerator::Integer)
    f=symmetric_group_dimension(p)
    C=coefficient_block(rho,p);R=_real_float_type(eltype(rho.data))
    scale=try
        _checked_sqrt_exact_ratio(
            R,scale_squared_numerator,f;
            context="multiplicity-scaled parent block in sector $p")
    catch error
        error isa ArgumentError||rethrow()
        nothing
    end
    if scale!==nothing
        # Ordinary sectors retain one scalar matrix multiplication.  Reject a
        # direct result if either complex component of a nonzero input was
        # silently lost; the binary-scaled per-entry route below then either
        # recovers the bounded product or raises with wider-type guidance.
        # Materialize once and scale in place.  `Matrix(C)*scale` creates a
        # second dense temporary for every parent/coupling even when the
        # caller supplied a reusable ReductionWorkspace.
        result=Matrix(C)
        result.*=scale
        _ordinary_scaled_value_safe(result,C)&&return result
    end
    _checked_mul_sqrt_exact_ratio(
        C,scale_squared_numerator,f;
        context="multiplicity-scaled parent block in sector $p")
end

function _scaled_reduction_parent_block!(destination::AbstractMatrix,
                                         rho::PIState,p::Partition,
                                         scale::_PreparedExactScale)
    C=coefficient_block(rho,p)
    n=size(C,1)
    result=view(destination,1:n,1:n)
    if scale.direct
        @inbounds for index in eachindex(result,C)
            result[index]=C[index]*scale.factor
        end
        _ordinary_scaled_value_safe(result,C)&&return result
    end
    @inbounds for index in eachindex(result,C)
        result[index]=_apply_prepared_exact_scale(C[index],scale;
            context="multiplicity-scaled parent block in sector $p")
    end
    result
end

function _product_block(rho::PIState,c::_ProductSchurCoupling{T},
                        scale_squared_numerator::Integer=1) where T
    R=promote_type(eltype(rho.data),T);block=zeros(R,c.da*c.db,c.da*c.db)
    for (s,Ts) in c.intertwiners
        C=_scaled_reduction_parent_block(
            rho,rho.basis.sectors[s],scale_squared_numerator)
        for U in Ts;block .+= U*C*U';end
    end
    block
end

function _product_block!(rho::PIState,c::_ProductSchurCoupling,
                         work::ReductionWorkspace,
                         recouplers::AbstractVector,
                         parent_scales::AbstractVector)
    n=c.da*c.db
    block=view(work.product_block,1:n,1:n)
    fill!(block,zero(eltype(block)))
    length(parent_scales)==length(c.intertwiners)||error(
        "internal reduction parent-scale count mismatch")
    length(recouplers)==length(c.intertwiners)||error(
        "internal reduction recoupler count mismatch")
    for (connection_index,(s,_)) in pairs(c.intertwiners)
        Ts=recouplers[connection_index]
        C=_scaled_reduction_parent_block!(work.parent_block,
            rho,rho.basis.sectors[s],parent_scales[connection_index])
        dl=size(C,1)
        tmp=view(work.product_tmp,1:n,1:dl)
        for U in Ts
            mul!(tmp,U,C)
            mul!(block,tmp,adjoint(U),one(eltype(block)),one(eltype(block)))
        end
    end
    block
end

function _fill_partial_trace!(dest,block,da::Int,db::Int)
    R=view(dest,1:da,1:da);fill!(R,zero(eltype(R)))
    @inbounds for q in 1:db,j in 1:da,i in 1:da
        R[i,j]+=block[i+(q-1)*da,j+(q-1)*da]
    end
    R
end

function _fill_partial_transpose!(dest,block,da::Int,db::Int)
    n=da*db;pt=view(dest,1:n,1:n)
    @inbounds for jb in 1:db,ja in 1:da,ib in 1:db,ia in 1:da
        pt[ia+(ib-1)*da,ja+(jb-1)*da]=
            block[ja+(ib-1)*da,ia+(jb-1)*da]
    end
    pt
end

function _hermitianize_reduction_roundoff!(A;atol::Real,rtol::Real,
                                           label::AbstractString)
    R=_real_float_type(eltype(A));error=zero(R);scale=zero(R)
    @inbounds for j in axes(A,2),i in axes(A,1)
        scale=max(scale,R(abs(A[i,j])))
        error=max(error,R(abs(A[i,j]-conj(A[j,i]))))
    end
    requested=R(atol)+R(rtol)*scale
    roundoff=R(32)*eps(R)*max(scale,one(R))*max(size(A)...)
    error<=max(requested,roundoff)||throw(ErrorException(
        "$label lost Hermiticity: error=$error, tolerance=$(max(requested,roundoff))"))
    @inbounds for j in axes(A,2)
        A[j,j]=complex(real(A[j,j]),zero(R))
        for i in 1:j-1
            value=(A[i,j]+conj(A[j,i]))/2
            A[i,j]=value;A[j,i]=conj(value)
        end
    end
    A
end

function _accumulate_reduced_blocks!(rho::PIState,plan::ReductionPlan,
                                     work::ReductionWorkspace;
                                     atol::Real,rtol::Real)
    _require_reduction_workspace_mode(work,:reduction)
    for R in work.reduced_blocks;fill!(R,zero(eltype(R)));end
    for (coupling_index,c) in pairs(plan.couplings)
        # The output stores C_alpha=sqrt(f_alpha)R_alpha.  Fuse its complete
        # factor sqrt(f_alpha)*f_beta/sqrt(f_lambda) into the parent
        # coefficient block before floating conversion.
        block=_product_block!(rho,c,work,
            work.recoupling_intertwiners[coupling_index],
            work.reduction_parent_scales[coupling_index])
        partial=_fill_partial_trace!(work.partial_trace,block,c.da,c.db)
        ai=plan.output_basis.index[c.alpha]
        target=work.reduced_blocks[ai]
        @. target=target+partial
    end
    for R in work.reduced_blocks
        _hermitianize_reduction_roundoff!(R;atol=atol,rtol=rtol,
                                          label="reduced Schur block")
    end
    work.reduced_blocks
end

function _plan_negativity(rho::PIState,plan::ReductionPlan)
    (plan.k==0||plan.k==rho.basis.N)&&return 0.0
    total_norm=0.0
    for c in plan.couplings
        block=_product_block(rho,c,c.product_multiplicity^2)
        pt=reshape(permutedims(reshape(block,c.da,c.db,c.da,c.db),(3,2,1,4)),c.da*c.db,c.da*c.db)
        total_norm+=sum(abs,eigvals(Hermitian((pt+pt')/2)))
    end
    max(0.0,(total_norm-1)/2)
end

function _plan_negativity(rho::PIState,plan::ReductionPlan,
                          work::ReductionWorkspace;atol::Real,rtol::Real)
    _require_reduction_workspace_mode(work,:negativity)
    (plan.k==0||plan.k==rho.basis.N)&&return zero(_real_float_type(work.Ttype))
    R=_real_float_type(work.Ttype);total_norm=zero(R)
    for (coupling_index,c) in pairs(plan.couplings)
        block=_product_block!(rho,c,work,
            work.recoupling_intertwiners[coupling_index],
            work.negativity_parent_scales[coupling_index])
        pt=_fill_partial_transpose!(work.partial_transpose,block,c.da,c.db)
        _hermitianize_reduction_roundoff!(pt;atol=atol,rtol=rtol,
                                          label="partial transpose block")
        values=eigvals!(Hermitian(pt))
        total_norm+=sum(abs,values)
    end
    max(zero(R),(total_norm-one(R))/2)
end

function _qudit_negativity(rho::PIState,k::Integer;atol::Real=1e-12)
    b=rho.basis;na=k;nb=b.N-k
    (na==0||nb==0)&&return 0.0
    total_norm=0.0
    cache=Dict{Tuple{Partition,Partition,Partition},Vector{Matrix{ComplexF64}}}()
    gencache=Dict{Partition,Vector{Matrix{ComplexF64}}}()
    for alpha in partitions(na,b.d),beta in partitions(nb,b.d)
        da=Int(unitary_group_dimension(alpha));db=Int(unitary_group_dimension(beta))
        block=zeros(ComplexF64,da*db,da*db)
        for lambda in b.sectors
            key=(alpha,beta,lambda)
            Ts=get!(cache,key) do
                _lr_intertwiners(alpha,beta,lambda;atol=max(atol,2e-11),gencache=gencache)
            end
            isempty(Ts)&&continue
            multiplicity=symmetric_group_dimension(alpha)*
                         symmetric_group_dimension(beta)
            C=_scaled_reduction_parent_block(rho,lambda,multiplicity^2)
            for T in Ts
                block .+= T*C*T'
            end
        end
        tensor=reshape(block,da,db,da,db)
        pt=reshape(permutedims(tensor,(3,2,1,4)),da*db,da*db)
        vals=eigvals(Hermitian((pt+pt')/2))
        total_norm += sum(abs,vals)
    end
    max(0.0,(total_norm-1)/2)
end

"""
    logarithmic_negativity(rho, k; base=2, atol=_analysis_atol(rho),
                           rtol=_state_rtol(rho),
                           plan=nothing, workspace=nothing)

Return `log_base(‖rho^T_A‖₁) = log_base(2*negativity(rho,k)+1)` for the
general PI bipartitions supported by [`negativity`](@ref).
"""
function logarithmic_negativity(rho::PIState,k::Integer;base::Real=2,
                                atol::Real=_analysis_atol(rho),
                                rtol::Real=_state_rtol(rho),
                                plan=nothing,workspace=nothing)
    base>0 && base!=1 || throw(ArgumentError("logarithm base must be positive and different from one"))
    log(2negativity(rho,k;atol=atol,rtol=rtol,plan=plan,
                    workspace=workspace)+1)/log(base)
end

function _partial_trace_b(block::AbstractMatrix,da::Int,db::Int)
    R=zeros(eltype(block),da,da)
    A=reshape(block,da,db,da,db)
    for q in 1:db
        @views R .+= A[:,q,:,q]
    end
    R
end

function _plan_reduced_state(rho::PIState,plan::ReductionPlan{T};atol::Real,rtol::Real) where T
    b=rho.basis;k=plan.k
    k==b.N&&return PIState(b,copy(rho.data))
    CT=promote_type(eltype(rho.data),T);RT=_real_float_type(CT)
    out=PIState(plan.output_basis;T=RT)
    k==0&&(out.data[1]=one(eltype(out.data));return out)
    reduced_blocks=[zeros(CT,length(patterns),length(patterns)) for patterns in plan.output_basis.patterns]
    for c in plan.couplings
        scale_squared=c.alpha_multiplicity*c.beta_multiplicity^2
        block=_product_block(rho,c,scale_squared)
        ai=plan.output_basis.index[c.alpha]
        reduced_blocks[ai].+=_partial_trace_b(block,c.da,c.db)
    end
    for (s,alpha) in pairs(plan.output_basis.sectors)
        R=reduced_blocks[s];R=(R+R')/2
        coefficient_block(out,alpha).=R
    end
    abs(trace(out)-1)<=max(atol+rtol,5e-10)||throw(ErrorException("partial-trace normalization check failed"))
    out
end

function _plan_reduced_state!(out::PIState,rho::PIState,plan::ReductionPlan,
                              work::ReductionWorkspace;
                              atol::Real,rtol::Real)
    b=rho.basis;k=plan.k
    out.basis===plan.output_basis||throw(ArgumentError(
        "output state must use the ReductionPlan output_basis object"))
    promote_type(eltype(out.data),work.Ttype)===eltype(out.data)||throw(ArgumentError(
        "output state scalar type $(eltype(out.data)) cannot represent ReductionWorkspace scalar type $(work.Ttype)"))
    if k==b.N
        copyto!(out.data,rho.data)
        return out
    end
    fill!(out.data,zero(eltype(out.data)))
    if k==0
        out.data[1]=one(eltype(out.data))
        return out
    end
    blocks=_accumulate_reduced_blocks!(rho,plan,work;atol=atol,rtol=rtol)
    for (s,alpha) in pairs(plan.output_basis.sectors)
        coefficient_block(out,alpha).=blocks[s]
    end
    abs(trace(out)-1)<=max(atol+rtol,5e-10)||throw(ErrorException(
        "partial-trace normalization check failed"))
    out
end

"""
    reduced_state!(out, rho, k; plan, workspace,
                   atol=_analysis_atol(rho), rtol=_state_rtol(rho))

Write the `k`-particle reduced density matrix into `out` while reusing a
caller-owned [`ReductionWorkspace`](@ref). `out` must use
`plan.output_basis`, and the plan, workspace, and input state must belong to
the exact same parent `PIBasis` object.
"""
function reduced_state!(out::PIState,rho::PIState,k::Integer;
                        plan=nothing,workspace=nothing,
                        atol::Real=_analysis_atol(rho),
                        rtol::Real=_state_rtol(rho))
    workspace isa ReductionWorkspace||throw(ArgumentError(
        "reduced_state! requires a ReductionWorkspace"))
    0<=k<=rho.basis.N||throw(ArgumentError(
        "subsystem size k must satisfy 0 ≤ k ≤ N"))
    validate_state(rho;atol=atol,rtol=rtol)
    plan,workspace=_resolve_reduction_resources(
        rho.basis,k,plan,workspace;atol=atol)
    _check_reduction_workspace(workspace,plan,rho)
    _plan_reduced_state!(out,rho,plan,workspace;atol=atol,rtol=rtol)
end

reduced_state!(out::PIState,rho::PIState,plan::ReductionPlan,
               workspace::ReductionWorkspace;kwargs...)=
    reduced_state!(out,rho,plan.k;plan=plan,workspace=workspace,kwargs...)

function _qubit_product_block(rho::PIState,pa::Partition,ja::Int,pb::Partition,jb::Int)
    b=rho.basis;da=ja+1;db=jb+1
    block=zeros(ComplexF64,da*db,da*db)
    ma=collect(-ja:2:ja);mb=collect(-jb:2:jb)
    for p in b.sectors
        j=b.N-2p[2]
        abs(ja-jb)<=j<=ja+jb && iseven(ja+jb-j) || continue
        C=_scaled_reduction_parent_block(rho,p,1)
        patterns=b.patterns[b.index[p]]
        m_to_i=Dict(_pattern_m2(g,b.N)=>i for (i,g) in pairs(patterns))
        U=zeros(Float64,da*db,j+1)
        for (ia,m1) in pairs(ma),(ib,m2) in pairs(mb)
            m=m1+m2;abs(m)<=j||continue
            U[ia+(ib-1)*da,m_to_i[m]]=_su2_cgc(ja,m1,jb,m2,j,m)
        end
        block .+= U*C*U'
    end
    block
end

function _qudit_product_block(rho::PIState,alpha::Partition,beta::Partition,
                              cache,gencache;atol=2e-11)
    b=rho.basis
    da=Int(unitary_group_dimension(alpha));db=Int(unitary_group_dimension(beta))
    block=zeros(ComplexF64,da*db,da*db)
    for lambda in b.sectors
        key=(alpha,beta,lambda)
        Ts=get!(cache,key) do
            _lr_intertwiners(alpha,beta,lambda;atol=atol,gencache=gencache)
        end
        C=_scaled_reduction_parent_block(rho,lambda,1)
        for T in Ts
            block .+= T*C*T'
        end
    end
    block
end

"""
    reduced_state(rho, k; atol=_analysis_atol(rho),
                  rtol=_state_rtol(rho), plan=nothing,
                  workspace=nothing)

Return the reduced state of any `k` particles as a `PIState` on
`PIBasis(k,d)`. Permutation invariance makes the result independent of which
particles are retained.

The partial trace is evaluated directly in product Schur blocks. Qubits use
specialized SU(2) recoupling and qudits use cached Littlewood--Richardson
intertwiners. No computational-basis matrix of dimension `d^k` is formed.
The returned coefficient blocks obey the package's equation-(7)
normalization.
Pass `plan=ReductionPlan(rho.basis,k)` when the same reduction is evaluated
for several states. Add `workspace=ReductionWorkspace(plan,rho)` to reuse
application scratch, or call [`reduced_state!`](@ref) to reuse the output too.
"""
function reduced_state(rho::PIState,k::Integer;
                       atol::Real=_analysis_atol(rho),
                       rtol::Real=_state_rtol(rho),plan=nothing,
                       workspace=nothing)
    b=rho.basis
    0<=k<=b.N||throw(ArgumentError("subsystem size k must satisfy 0 ≤ k ≤ N"))
    validate_state(rho;atol=atol,rtol=rtol)
    plan,workspace=_resolve_reduction_resources(b,k,plan,workspace;atol=atol)
    if workspace===nothing
        return _plan_reduced_state(rho,plan;atol=atol,rtol=rtol)
    end
    _check_reduction_workspace(workspace,plan,rho)
    RT=_real_float_type(workspace.Ttype)
    out=PIState(plan.output_basis;T=RT)
    _plan_reduced_state!(out,rho,plan,workspace;atol=atol,rtol=rtol)
end

function _reduced_blocks_purity(blocks,::Type{R}) where R<:AbstractFloat
    result=zero(R)
    for block in blocks
        result+=real(dot(block,block))
    end
    isfinite(result)&&(!iszero(result)||all(block->all(iszero,block),blocks))&&
        return result
    scale=zero(R)
    for block in blocks
        scale=hypot(scale,norm(block))
    end
    isfinite(scale)&&!iszero(scale)||throw(ArgumentError(
        "reduced-state purity is outside the nonzero finite range of $R; use a wider scalar type"))
    result=scale*scale
    isfinite(result)&&!iszero(result)||throw(ArgumentError(
        "reduced-state purity is outside the nonzero finite range of $R; use a wider scalar type"))
    result
end

"""
    reduced_purity(rho, k; atol=_analysis_atol(rho),
                   rtol=_state_rtol(rho), plan=nothing,
                   workspace=nothing)

Return `tr(rho_k^2)` for the `k`-particle state returned by
[`reduced_state`](@ref). The result for `k=0` is one and that for `k=N` equals
[`purity`](@ref).
"""
function reduced_purity(rho::PIState,k::Integer;
                        atol::Real=_analysis_atol(rho),
                        rtol::Real=_state_rtol(rho),plan=nothing,
                        workspace=nothing)
    if workspace===nothing
        return purity(reduced_state(rho,k;atol=atol,rtol=rtol,plan=plan))
    end
    b=rho.basis;0<=k<=b.N||throw(ArgumentError(
        "subsystem size k must satisfy 0 ≤ k ≤ N"))
    validate_state(rho;atol=atol,rtol=rtol)
    plan,workspace=_resolve_reduction_resources(b,k,plan,workspace;atol=atol)
    _check_reduction_workspace(workspace,plan,rho)
    k==0&&return one(_real_float_type(workspace.Ttype))
    k==b.N&&return purity(rho)
    blocks=_accumulate_reduced_blocks!(rho,plan,workspace;
                                       atol=atol,rtol=rtol)
    _reduced_blocks_purity(blocks,_real_float_type(workspace.Ttype))
end

"""
    reduced_purities(rho; ks=0:N, atol=_analysis_atol(rho),
                     rtol=_state_rtol(rho), plans=nothing)

Compute reduced purities for the requested subsystem sizes and return them in
the same order as `ks`.
"""
function reduced_purities(rho::PIState;ks=0:rho.basis.N,
                          atol::Real=_analysis_atol(rho),
                          rtol::Real=_state_rtol(rho),plans=nothing)
    kvals=collect(ks)
    if plans===nothing
        return [reduced_purity(rho,k;atol=atol,rtol=rtol) for k in kvals]
    end
    ps=collect(plans);length(ps)==length(kvals)||throw(DimensionMismatch("one ReductionPlan is required per subsystem size"))
    [reduced_purity(rho,k;atol=atol,rtol=rtol,plan=p) for (k,p) in zip(kvals,ps)]
end

"""
    partial_transpose_spectrum(rho, k; plan=nothing)

Return product-Schur partial-transpose spectra as named tuples containing
`alpha`, `beta`, `multiplicity`, and `eigenvalues`. Multiplicities are not
expanded, avoiding exponentially repeated output.
"""
function partial_transpose_spectrum(rho::PIState,k::Integer;
                                    atol::Real=_analysis_atol(rho),
                                    rtol::Real=_state_rtol(rho),plan=nothing)
    b=rho.basis;0<=k<=b.N||throw(ArgumentError("invalid bipartition"));validate_state(rho;atol=atol,rtol=rtol)
    if plan!==nothing
        plan isa ReductionPlan||throw(ArgumentError("plan must be a ReductionPlan"))
        _check_reduction_plan(plan,b,k)
    end
    if k==0||k==b.N
        z=Partition(ntuple(_->0,b.d))
        return [(alpha=k==b.N ? p : z,beta=k==0 ? p : z,multiplicity=symmetric_group_dimension(p),eigenvalues=_sector_eigenvalues(rho,p;atol=atol,rtol=rtol)) for p in b.sectors]
    end
    plan===nothing&&(plan=ReductionPlan(b,k;atol=max(atol,2e-11)))
    out=NamedTuple[]
    for c in plan.couplings
        block=_product_block(rho,c)
        pt=reshape(permutedims(reshape(block,c.da,c.db,c.da,c.db),(3,2,1,4)),c.da*c.db,c.da*c.db)
        push!(out,(alpha=c.alpha,beta=c.beta,multiplicity=c.product_multiplicity,
                   eigenvalues=eigvals(Hermitian((pt+pt')/2))))
    end
    out
end

"""
    minimum_partial_transpose_eigenvalue(rho, k; kwargs...)

Return the smallest eigenvalue across the multiplicity-compressed
product-Schur blocks produced by `partial_transpose_spectrum(rho, k)`.
"""
minimum_partial_transpose_eigenvalue(rho::PIState,k::Integer;kwargs...)=minimum(minimum(x.eigenvalues) for x in partial_transpose_spectrum(rho,k;kwargs...))

"""Negativity for every inequivalent particle bipartition `1:floor(N/2)`."""
bipartition_negativities(rho::PIState;kwargs...)=[negativity(rho,k;kwargs...) for k in 1:fld(rho.basis.N,2)]

function _charge_pt_blocks(rho,k,Q;atol=_analysis_atol(rho),plan=nothing)
    b=rho.basis;out=NamedTuple[]; qcache=Dict{Partition,Matrix{ComplexF64}}()
    plan===nothing&&(plan=ReductionPlan(b,k;atol=max(atol,2e-11)))
    _check_reduction_plan(plan,b,k)
    qblock(p)=get!(qcache,p) do
        bp=PIBasis(weight(p),b.d;sectors=[p.parts]);collective_block(bp,Q,p)
    end
    for c in plan.couplings
        block=_product_block(rho,c,c.product_multiplicity^2)
        pt=reshape(permutedims(reshape(block,c.da,c.db,c.da,c.db),(3,2,1,4)),c.da*c.db,c.da*c.db)
        imbalance=kron(Matrix{ComplexF64}(I,c.db,c.db),transpose(qblock(c.alpha)))-
                  kron(qblock(c.beta),Matrix{ComplexF64}(I,c.da,c.da))
        push!(out,(pt=(pt+pt')/2,imbalance=(imbalance+imbalance')/2,
                   multiplicity=1.0))
    end
    out
end

"""Negativity contributions resolved by partial-transpose charge imbalance."""
function charge_resolved_negativity(rho::PIState,k::Integer,
                                    local_charge::AbstractMatrix;
                                    atol::Real=_state_rtol(rho),
                                    rtol::Real=_state_rtol(rho),plan=nothing)
    b=rho.basis;0<k<b.N||throw(ArgumentError("charge resolution requires 0 < k < N"));validate_state(rho;atol=atol,rtol=rtol)
    size(local_charge)==(b.d,b.d)||throw(DimensionMismatch("local charge has wrong size"));ishermitian(local_charge)||throw(ArgumentError("local charge must be Hermitian"))
    plan!==nothing&&(plan isa ReductionPlan||throw(ArgumentError("plan must be a ReductionPlan")))
    accum=Dict{Float64,Tuple{Float64,Float64}}()
    for x in _charge_pt_blocks(rho,k,local_charge;atol=atol,plan=plan)
        norm(x.pt*x.imbalance-x.imbalance*x.pt)<=atol*max(norm(x.pt)*norm(x.imbalance),1)||throw(ArgumentError("rho does not have the requested additive charge symmetry"))
        E=eigen(Hermitian(x.imbalance));groups=Vector{Vector{Int}}()
        for i in eachindex(E.values)
            j=findfirst(g->abs(E.values[first(g)]-E.values[i])<=atol,groups)
            j===nothing ? push!(groups,[i]) : push!(groups[j],i)
        end
        for inds in groups
            B=E.vectors[:,inds]'*x.pt*E.vectors[:,inds]; vals=eigvals(Hermitian((B+B')/2))
            q=E.values[first(inds)]; key=round(q;digits=max(0,ceil(Int,-log10(atol))))
            neg=x.multiplicity*sum(v->max(0,-v),vals);trb=x.multiplicity*real(sum(vals))
            old=get(accum,key,(0.0,0.0));accum[key]=(old[1]+neg,old[2]+trb)
        end
    end
    [(charge=q,negativity=accum[q][1],weight=accum[q][2]) for q in sort!(collect(keys(accum)))]
end

"""Charge-resolved negativity for the local number operator `diag(0:d-1)`."""
number_resolved_negativity(rho::PIState,k::Integer;kwargs...)=charge_resolved_negativity(rho,k,Diagonal(collect(0:rho.basis.d-1));kwargs...)
