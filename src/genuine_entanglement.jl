const _PPT_MIXTURE_CLARABEL_EXTENSION =
    :PermutationalInvariantDynamicsClarabelExt

struct _PPTMixtureBlock{D}
    cut::Int
    alpha::Partition{D}
    beta::Partition{D}
    da::Int
    db::Int
    dimension::Int
    variables::UnitRange{Int}
    positive_rows::UnitRange{Int}
    partial_transpose_rows::UnitRange{Int}
end

@doc raw"""
    PPTMixturePlan(basis; T=Float64, memory_budget=512*1024^2)

Prepare the polynomial-size semidefinite program of Novo, Moroder, and
Gühne, *Phys. Rev. A* **88**, 012305 (2013), for testing whether a PI qubit
state is a positive-partial-transpose (PPT) mixture.

For every inequivalent cut ``k\,|\,(N-k)`` the plan introduces one
``S_k\times S_{N-k}``-invariant operator.  Its product-Schur blocks are
represented by multiplicity-weighted Hermitian matrices ``Y_{k,\alpha,\beta}``
and constrained by

```math
Y_{k,\alpha,\beta}\succeq0,\qquad
Y_{k,\alpha,\beta}^{T_A}\succeq0.
```

The equality constraint is assembled directly in Schur blocks,

```math
\sqrt{f^\lambda}\,C_\lambda
=\sum_{k,\alpha,\beta,r}
U_{\alpha\beta\to\lambda,r}^{\dagger}
Y_{k,\alpha,\beta}
U_{\alpha\beta\to\lambda,r},
```

so no ``2^N`` state or operator is constructed.  An internally complete PI
basis is used even when `basis` is sector restricted; absent source sectors
therefore enter the equality constraint as exact zeros.

The immutable plan stores a sparse real conic map and may be reused for many
states on the exact same `PIBasis`.  `T` may be `Float32` or `Float64`.
`memory_budget=Inf` is the explicit opt-out from the conservative setup and
solver-storage guard.

The paper derives this construction for qubits.  Qudit inputs are rejected
rather than being passed through an unproved generalization.
"""
struct PPTMixturePlan{T<:AbstractFloat,B<:PIBasis,F<:PIBasis}
    basis::B
    full_basis::F
    cuts::Vector{Int}
    blocks::Vector{_PPTMixtureBlock{2}}
    constraint_matrix::SparseMatrixCSC{T,Int}
    objective::Vector{T}
    equality_rows::UnitRange{Int}
    equality_sector_rows::Vector{UnitRange{Int}}
    cone_dimensions::Vector{Int}
    estimated_setup_bytes::BigInt
    estimated_solve_bytes::BigInt
end

function show(io::IO,plan::PPTMixturePlan)
    print(io,"PPTMixturePlan(N=$(plan.basis.N), blocks=$(length(plan.blocks)), " *
             "variables=$(size(plan.constraint_matrix,2)), " *
             "constraints=$(size(plan.constraint_matrix,1)), " *
             "scalar_type=$(eltype(plan.objective)))")
end

@doc raw"""
    PPTMixtureResult

Result of [`ppt_mixture_test`](@ref).  `classification` is one of:

- `:gme_certified`: a converged, explicitly validated numerical dual
  certificate establishes, within the reported tolerances, that the state is
  not a PPT mixture and hence is genuinely multipartite entangled;
- `:ppt_mixture`: a converged primal decomposition passes the Schur equality,
  positivity, and partial-transpose checks;
- `:inconclusive`: neither numerical certificate passed.

For PI qubits with `N == 3` (and for the bipartite `N == 2` case), PPT-mixture
membership is also sufficient for biseparability.  For `N >= 4`, a
`:ppt_mixture` result means only that this criterion did not detect genuine
multipartite entanglement; consequently `genuinely_multipartite_entangled`
and `biseparable` are `missing`.

`scaled_margin` is the optimized slack for multiplicity-weighted Schur
variables.  Its sign has the paper's membership meaning, but its magnitude is
not the unscaled ``s`` of Eq. (31).
"""
struct PPTMixtureResult{T<:AbstractFloat}
    classification::Symbol
    ppt_mixture::Union{Bool,Missing}
    genuinely_multipartite_entangled::Union{Bool,Missing}
    biseparable::Union{Bool,Missing}
    scaled_margin::T
    primal_objective::T
    dual_objective::T
    equality_residual::T
    minimum_block_eigenvalue::T
    minimum_partial_transpose_eigenvalue::T
    dual_stationarity_residual::T
    minimum_dual_cone_eigenvalue::T
    solver_status::Symbol
    iterations::Int
    solve_time::Float64
    certificate_atol::T
    certificate_rtol::T
    message::String
end

function show(io::IO,result::PPTMixtureResult)
    print(io,"PPTMixtureResult(classification=$(result.classification), " *
             "solver_status=$(result.solver_status), " *
             "scaled_margin=$(result.scaled_margin), " *
             "equality_residual=$(result.equality_residual))")
end

@inline _ppt_triangle_length(n::Integer)=n*(n+1)÷2
@inline _ppt_svec_index(i::Int,j::Int)=((j-1)*j)÷2+i

@inline function _ppt_hreal_index(n::Int,i::Int,j::Int)
    i<j||throw(ArgumentError("Hermitian real coordinate requires i < j"))
    n+(j-1)*(j-2)+2*(i-1)+1
end

@inline _ppt_himag_index(n::Int,i::Int,j::Int)=
    _ppt_hreal_index(n,i,j)+1

function _ppt_product_partial_transpose(A::AbstractMatrix,da::Int,db::Int)
    n=da*db
    size(A)==(n,n)||throw(DimensionMismatch(
        "product block has the wrong dimension"))
    out=similar(A,n,n)
    @inbounds for jb in 1:db,ja in 1:da,ib in 1:db,ia in 1:da
        out[ia+(ib-1)*da,ja+(jb-1)*da]=
            A[ja+(ib-1)*da,ia+(jb-1)*da]
    end
    out
end

function _ppt_unpack_hermitian(x::AbstractVector{T},variables::UnitRange{Int},
                               n::Int) where T<:AbstractFloat
    length(variables)==n^2||throw(DimensionMismatch(
        "Hermitian coordinate range has the wrong length"))
    start=first(variables)-1
    R=typeof(sqrt(one(T)))
    invsqrt2=inv(sqrt(R(2)))
    X=zeros(Complex{T},n,n)
    @inbounds for i in 1:n
        X[i,i]=x[start+i]
    end
    @inbounds for j in 2:n,i in 1:j-1
        re=x[start+_ppt_hreal_index(n,i,j)]*invsqrt2
        im=x[start+_ppt_himag_index(n,i,j)]*invsqrt2
        value=complex(re,im)
        X[i,j]=value
        X[j,i]=conj(value)
    end
    X
end

function _ppt_pack_hermitian!(destination::AbstractVector{T},
                              rows::UnitRange{Int},A::AbstractMatrix) where
                              T<:AbstractFloat
    n=size(A,1)
    size(A,2)==n||throw(DimensionMismatch("Hermitian block must be square"))
    length(rows)==n^2||throw(DimensionMismatch(
        "Hermitian target range has the wrong length"))
    start=first(rows)-1
    root2=sqrt(T(2))
    @inbounds for i in 1:n
        destination[start+i]=T(real(A[i,i]))
    end
    @inbounds for j in 2:n,i in 1:j-1
        destination[start+_ppt_hreal_index(n,i,j)]=root2*T(real(A[i,j]))
        destination[start+_ppt_himag_index(n,i,j)]=root2*T(imag(A[i,j]))
    end
    destination
end

function _ppt_unpack_svec(z::AbstractVector{T},rows::AbstractUnitRange{Int},n::Int) where
        T<:AbstractFloat
    length(rows)==_ppt_triangle_length(n)||throw(DimensionMismatch(
        "semidefinite-cone coordinate range has the wrong length"))
    offset=first(rows)-1
    invsqrt2=inv(sqrt(T(2)))
    Z=zeros(T,n,n)
    @inbounds for j in 1:n,i in 1:j
        value=z[offset+_ppt_svec_index(i,j)]
        i==j||(value*=invsqrt2)
        Z[i,j]=value
        Z[j,i]=value
    end
    Z
end

function _ppt_transposed_coordinate(p::Int,q::Int,kind::Symbol,
                                    da::Int,db::Int)
    if p==q
        return p,p,one(Int)
    end
    ia=(p-1)%da+1
    ib=(p-1)÷da+1
    ja=(q-1)%da+1
    jb=(q-1)÷da+1
    r=ja+(ib-1)*da
    s=ia+(jb-1)*da
    r==s&&error("internal partial-transpose coordinate collision")
    if r<s
        return r,s,one(Int)
    end
    s,r,kind===:imag ? -one(Int) : one(Int)
end

function _ppt_append_real_embedding!(rows::Vector{Int},columns::Vector{Int},
        values::Vector{T},rowstart::Int,variable::Int,n::Int,
        i::Int,j::Int,kind::Symbol,sign::Int) where T<:AbstractFloat
    m=2n
    if kind===:diag
        push!(rows,rowstart+_ppt_svec_index(i,i)-1)
        push!(columns,variable);push!(values,-T(sign))
        push!(rows,rowstart+_ppt_svec_index(n+i,n+i)-1)
        push!(columns,variable);push!(values,-T(sign))
    elseif kind===:real
        push!(rows,rowstart+_ppt_svec_index(i,j)-1)
        push!(columns,variable);push!(values,-T(sign))
        push!(rows,rowstart+_ppt_svec_index(n+i,n+j)-1)
        push!(columns,variable);push!(values,-T(sign))
    elseif kind===:imag
        push!(rows,rowstart+_ppt_svec_index(i,n+j)-1)
        push!(columns,variable);push!(values,T(sign))
        push!(rows,rowstart+_ppt_svec_index(j,n+i)-1)
        push!(columns,variable);push!(values,-T(sign))
    else
        throw(ArgumentError("unknown Hermitian coordinate kind $kind"))
    end
    nothing
end

function _ppt_append_psd_constraint!(rows::Vector{Int},columns::Vector{Int},
        values::Vector{T},block::_PPTMixtureBlock,rowrange::UnitRange{Int};
        partial_transpose::Bool=false) where T<:AbstractFloat
    n=block.dimension
    m=2n
    rowstart=first(rowrange)
    variable_start=first(block.variables)-1
    @inbounds for p in 1:n
        i,j,sign=partial_transpose ?
            _ppt_transposed_coordinate(p,p,:diag,block.da,block.db) :
            (p,p,one(Int))
        _ppt_append_real_embedding!(rows,columns,values,rowstart,
            variable_start+p,n,i,j,:diag,sign)
    end
    @inbounds for q in 2:n,p in 1:q-1
        for kind in (:real,:imag)
            coordinate=kind===:real ? _ppt_hreal_index(n,p,q) :
                                      _ppt_himag_index(n,p,q)
            i,j,sign=partial_transpose ?
                _ppt_transposed_coordinate(p,q,kind,block.da,block.db) :
                (p,q,one(Int))
            _ppt_append_real_embedding!(rows,columns,values,rowstart,
                variable_start+coordinate,n,i,j,kind,sign)
        end
    end
    # `A*x + slack = 0`: the slack is Phi(X-tI), so the common slack
    # variable has coefficient +1 on every real-embedding diagonal.
    @inbounds for i in 1:m
        push!(rows,rowstart+_ppt_svec_index(i,i)-1)
        push!(columns,1);push!(values,one(T))
    end
    nothing
end

function _ppt_row_to_parent(U::AbstractMatrix{T}) where T
    n,d=size(U)
    columns=zeros(Int,n)
    amplitudes=zeros(T,n)
    @inbounds for row in 1:n
        found=0
        for column in 1:d
            value=U[row,column]
            iszero(value)&&continue
            found==0||throw(ErrorException(
                "a qubit CG row unexpectedly couples to several parent weights"))
            found=column
            amplitudes[row]=value
        end
        columns[row]=found
    end
    columns,amplitudes
end

function _ppt_append_equality_connection!(rows::Vector{Int},
        columns::Vector{Int},values::Vector{T},block::_PPTMixtureBlock,
        parent_rows::UnitRange{Int},U::AbstractMatrix) where T<:AbstractFloat
    n=block.dimension
    d=size(U,2)
    size(U,1)==n||throw(DimensionMismatch("CG intertwiner has the wrong size"))
    length(parent_rows)==d^2||throw(DimensionMismatch(
        "parent equality range has the wrong size"))
    parent_start=first(parent_rows)-1
    variable_start=first(block.variables)-1
    parent_columns,amplitudes=_ppt_row_to_parent(U)
    root2=sqrt(T(2))
    @inbounds for p in 1:n
        a=parent_columns[p]
        a==0&&continue
        push!(rows,parent_start+a)
        push!(columns,variable_start+p)
        push!(values,T(amplitudes[p])^2)
    end
    @inbounds for q in 2:n,p in 1:q-1
        a=parent_columns[p]
        b=parent_columns[q]
        (a==0||b==0)&&continue
        product=T(amplitudes[p])*T(amplitudes[q])
        real_variable=variable_start+_ppt_hreal_index(n,p,q)
        imaginary_variable=variable_start+_ppt_himag_index(n,p,q)
        if a==b
            push!(rows,parent_start+a)
            push!(columns,real_variable)
            push!(values,root2*product)
        else
            i=min(a,b);j=max(a,b)
            push!(rows,parent_start+_ppt_hreal_index(d,i,j))
            push!(columns,real_variable)
            push!(values,product)
            push!(rows,parent_start+_ppt_himag_index(d,i,j))
            push!(columns,imaginary_variable)
            push!(values,a<b ? product : -product)
        end
    end
    nothing
end

function _ppt_plan_shape(N::Int,::Type{T}) where T<:AbstractFloat
    variables=big(1)
    cone_rows=big(0)
    cone_nnz=big(0)
    equality_nnz_bound=big(0)
    block_count=0
    max_block=0
    for k in 1:fld(N,2)
        for (_,ja) in _qubit_sectors(k),(_,jb) in _qubit_sectors(N-k)
            n=(ja+1)*(jb+1)
            connections=0
            for (_,j) in _qubit_sectors(N)
                abs(ja-jb)<=j<=ja+jb&&iseven(ja+jb-j)&&
                    (connections+=1)
            end
            variables+=big(n)^2
            cone_rows+=2*big(_ppt_triangle_length(2*n))
            cone_nnz+=4*big(n)^2+4*big(n)
            equality_nnz_bound+=big(n)^2*connections
            block_count+=1
            max_block=max(max_block,n)
        end
    end
    equality_rows=sum(big(j+1)^2 for (_,j) in _qubit_sectors(N))
    rows=cone_rows+equality_rows
    nnz_bound=cone_nnz+equality_nnz_bound
    scalar_bytes=BigInt(sizeof(T));integer_bytes=BigInt(sizeof(Int))
    # Peak setup retains the triplets and the final CSC simultaneously.  The
    # solve allowance covers copied conic data, primal/dual work vectors, and
    # a conservative sparse-factorization multiple; solver fill remains
    # problem dependent and is reported as an estimate rather than a bound.
    setup_bytes=nnz_bound*(3integer_bytes+2scalar_bytes)+
                (variables+rows)*(4scalar_bytes+2integer_bytes)+
                (variables+1)*integer_bytes
    solve_bytes=setup_bytes+8*(nnz_bound*(integer_bytes+scalar_bytes)+
                (variables+rows)*scalar_bytes)
    (;variables,rows,nnz_bound,block_count,max_block,equality_rows,
      setup_bytes,solve_bytes)
end

function PPTMixturePlan(basis::PIBasis;T::Type{R}=Float64,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where R<:AbstractFloat
    basis.d==2||throw(ArgumentError(
        "PPTMixturePlan implements the qubit construction of Phys. Rev. A 88, 012305; d must equal 2"))
    basis.N>=2||throw(ArgumentError(
        "genuine multipartite entanglement requires at least two particles"))
    R in (Float32,Float64)||throw(ArgumentError(
        "PPTMixturePlan currently supports Float32 and Float64 conic data; use one of these types explicitly"))
    shape=_ppt_plan_shape(basis.N,R)
    _require_performance_budget("PI PPT-mixture SDP preparation",
        shape.setup_bytes,memory_budget;guidance=
        "Reduce N or prepare the plan on a machine with more memory.")
    _require_performance_budget("PI PPT-mixture SDP solve",
        shape.solve_bytes,memory_budget;guidance=
        "Reduce N or pass a larger budget after checking sparse-factorization memory.")
    all(value->value<=typemax(Int),
        (shape.variables,shape.rows,shape.nnz_bound))||throw(ArgumentError(
        "PI PPT-mixture SDP dimensions exceed addressable Int indices"))

    full_basis=PIBasis(basis.N,2)
    variable_offset=2
    row_offset=1
    cone_dimensions=Int[]
    cuts=collect(1:fld(basis.N,2))
    blocks=_PPTMixtureBlock{2}[]
    block_lookup=Dict{Tuple{Int,Partition{2},Partition{2}},Int}()
    for k in cuts
        for (alpha,ja) in _qubit_sectors(k),
            (beta,jb) in _qubit_sectors(basis.N-k)
            da=ja+1;db=jb+1;n=da*db
            variables=variable_offset:(variable_offset+n^2-1)
            variable_offset=last(variables)+1
            m=2n
            nr=_ppt_triangle_length(m)
            positive_rows=row_offset:(row_offset+nr-1)
            row_offset=last(positive_rows)+1
            partial_rows=row_offset:(row_offset+nr-1)
            row_offset=last(partial_rows)+1
            push!(cone_dimensions,m);push!(cone_dimensions,m)
            block=_PPTMixtureBlock{2}(k,alpha,beta,da,db,n,variables,
                positive_rows,partial_rows)
            push!(blocks,block)
            block_lookup[(k,alpha,beta)]=length(blocks)
        end
    end
    equality_start=row_offset
    equality_sector_rows=UnitRange{Int}[]
    for patterns in full_basis.patterns
        n=length(patterns)
        range=row_offset:(row_offset+n^2-1)
        push!(equality_sector_rows,range)
        row_offset=last(range)+1
    end
    equality_rows=equality_start:(row_offset-1)
    nvariables=variable_offset-1
    nrows=row_offset-1
    nvariables==Int(shape.variables)||error(
        "internal PPT-mixture variable-count mismatch")
    nrows==Int(shape.rows)||error(
        "internal PPT-mixture constraint-count mismatch")

    triplet_rows=Int[]
    triplet_columns=Int[]
    triplet_values=R[]
    sizehint!(triplet_rows,Int(shape.nnz_bound))
    sizehint!(triplet_columns,Int(shape.nnz_bound))
    sizehint!(triplet_values,Int(shape.nnz_bound))
    for block in blocks
        _ppt_append_psd_constraint!(triplet_rows,triplet_columns,
            triplet_values,block,block.positive_rows)
        _ppt_append_psd_constraint!(triplet_rows,triplet_columns,
            triplet_values,block,block.partial_transpose_rows;
            partial_transpose=true)
    end
    for k in cuts
        # Only one cut's CG data is live at once.  The final plan retains the
        # assembled sparse equality map, not every intermediate ReductionPlan.
        reduction=ReductionPlan(full_basis,k)
        for coupling in reduction.couplings
            block=blocks[block_lookup[(k,coupling.alpha,coupling.beta)]]
            for (sector_index,intertwiners) in coupling.intertwiners
                for U in intertwiners
                    _ppt_append_equality_connection!(triplet_rows,
                        triplet_columns,triplet_values,block,
                        equality_sector_rows[sector_index],U)
                end
            end
        end
    end
    A=sparse(triplet_rows,triplet_columns,triplet_values,nrows,nvariables)
    objective=zeros(R,nvariables)
    objective[1]=-one(R)
    PPTMixturePlan{R,typeof(basis),typeof(full_basis)}(
        basis,full_basis,cuts,blocks,A,objective,equality_rows,
        equality_sector_rows,cone_dimensions,shape.setup_bytes,
        shape.solve_bytes)
end

PPTMixturePlan(rho::PIState;kwargs...)=PPTMixturePlan(rho.basis;kwargs...)

function _ppt_check_plan(plan::PPTMixturePlan,rho::PIState)
    plan.basis===rho.basis||throw(ArgumentError(
        "PPTMixturePlan was prepared for a different PIBasis object"))
    state_type=_real_float_type(eltype(rho.data))
    plan_type=eltype(plan.objective)
    promote_type(plan_type,state_type)===plan_type||throw(ArgumentError(
        "PPTMixturePlan scalar type $plan_type cannot represent state scalar type $state_type; rebuild the plan with T=$state_type"))
    plan
end

function _ppt_mixture_rhs(plan::PPTMixturePlan{T},rho::PIState) where
        T<:AbstractFloat
    _ppt_check_plan(plan,rho)
    rhs=zeros(T,size(plan.constraint_matrix,1))
    for (sector_index,partition) in pairs(plan.full_basis.sectors)
        source_index=get(rho.basis.index,partition,0)
        source_index==0&&continue
        C=coefficient_block(rho,partition)
        D=_multiply_by_schur_multiplicity_scale(
            C,_real_float_type(eltype(rho.data)),partition)
        _ppt_pack_hermitian!(rhs,
            plan.equality_sector_rows[sector_index],D)
    end
    rhs
end

@inline function _ppt_finite_solver_vector(vector::AbstractVector,
                                            expected_length::Integer)
    length(vector)==expected_length&&all(isfinite,vector)
end

function _ppt_primal_diagnostics(plan::PPTMixturePlan{T},rhs,
                                 x::AbstractVector) where T<:AbstractFloat
    _ppt_finite_solver_vector(x,size(plan.constraint_matrix,2))||return (
        scaled_margin=T(NaN),equality_residual=T(Inf),
        equality_scale=one(T),minimum_block_eigenvalue=-T(Inf),
        minimum_partial_transpose_eigenvalue=-T(Inf),
        cone_scale=one(T))
    equality=view(plan.constraint_matrix,plan.equality_rows,:)*x
    target=view(rhs,plan.equality_rows)
    equality_residual=T(norm(equality-target,Inf))
    equality_scale=max(T(norm(target,Inf)),T(norm(equality,Inf)),one(T))
    minimum_block=T(Inf)
    minimum_partial=T(Inf)
    cone_scale=one(T)
    for block in plan.blocks
        X=_ppt_unpack_hermitian(x,block.variables,block.dimension)
        partial=_ppt_product_partial_transpose(X,block.da,block.db)
        cone_scale=max(cone_scale,T(opnorm(X,Inf)),T(opnorm(partial,Inf)))
        block_values=eigvals!(Hermitian(X))
        partial_values=eigvals!(Hermitian(partial))
        block_min=T(minimum(block_values))
        partial_min=T(minimum(partial_values))
        minimum_block=min(minimum_block,block_min)
        minimum_partial=min(minimum_partial,partial_min)
    end
    (;scaled_margin=T(x[1]),equality_residual,equality_scale,
      minimum_block_eigenvalue=minimum_block,
      minimum_partial_transpose_eigenvalue=minimum_partial,
      cone_scale)
end

function _ppt_dual_diagnostics(plan::PPTMixturePlan{T},rhs,
                               z::AbstractVector) where T<:AbstractFloat
    _ppt_finite_solver_vector(z,size(plan.constraint_matrix,1))||return (
        dual_objective=T(NaN),stationarity_residual=T(Inf),
        stationarity_scale=one(T),minimum_cone_eigenvalue=-T(Inf),
        cone_scale=one(T))
    adjoint_action=adjoint(plan.constraint_matrix)*z
    stationarity=adjoint_action+plan.objective
    stationarity_residual=T(norm(stationarity,Inf))
    stationarity_scale=max(T(norm(adjoint_action,Inf)),
                           T(norm(plan.objective,Inf)),one(T))
    minimum_cone=T(Inf)
    cone_scale=one(T)
    for block in plan.blocks
        for rows in (block.positive_rows,block.partial_transpose_rows)
            Z=_ppt_unpack_svec(z,rows,2*block.dimension)
            values=eigvals!(Symmetric(Z))
            minimum_cone=min(minimum_cone,T(minimum(values)))
            cone_scale=max(cone_scale,T(opnorm(Z,Inf)))
        end
    end
    (;dual_objective=-T(dot(rhs,z)),stationarity_residual,
      stationarity_scale,minimum_cone_eigenvalue=minimum_cone,cone_scale)
end

function _ppt_classified_result(plan::PPTMixturePlan{T},rhs,x,z;
        solver_status::Symbol,solved::Bool,primal_objective,
        solver_dual_objective,iterations::Integer,solve_time::Real,
        certificate_atol::Real,certificate_rtol::Real) where T<:AbstractFloat
    atol=T(certificate_atol)
    rtol=T(certificate_rtol)
    primal=_ppt_primal_diagnostics(plan,rhs,x)
    dual=_ppt_dual_diagnostics(plan,rhs,z)
    equality_tolerance=atol+rtol*primal.equality_scale
    primal_cone_scale=max(one(T),abs(primal.scaled_margin),primal.cone_scale)
    primal_cone_tolerance=atol+rtol*primal_cone_scale
    primal_valid=solved&&primal.equality_residual<=equality_tolerance&&
        primal.minimum_block_eigenvalue>=-primal_cone_tolerance&&
        primal.minimum_partial_transpose_eigenvalue>=-primal_cone_tolerance

    stationarity_tolerance=atol+rtol*dual.stationarity_scale
    dual_cone_tolerance=atol+rtol*dual.cone_scale
    objective_scale=max(one(T),abs(T(primal_objective)),
                        abs(dual.dual_objective),abs(T(solver_dual_objective)))
    objective_tolerance=atol+rtol*objective_scale
    dual_objective_consistent=isfinite(solver_dual_objective)&&
        abs(dual.dual_objective-T(solver_dual_objective))<=
            10*objective_tolerance
    dual_valid=solved&&isfinite(dual.dual_objective)&&
        dual.stationarity_residual<=stationarity_tolerance&&
        dual.minimum_cone_eigenvalue>=-dual_cone_tolerance&&
        dual_objective_consistent&&dual.dual_objective>10*objective_tolerance

    if primal_valid&&dual_valid
        classification=:inconclusive
        ppt=missing;gme=missing;biseparable=missing
        message="primal and dual numerical certificates conflict; tighten solver tolerances"
    elseif dual_valid
        classification=:gme_certified
        ppt=false;gme=true;biseparable=false
        message="validated numerical dual certificate establishes non-PPT-mixture membership within the reported tolerances"
    elseif primal_valid
        classification=:ppt_mixture
        ppt=true
        if plan.basis.N<=3
            gme=false;biseparable=true
            message=plan.basis.N==3 ?
                "validated numerical PPT-mixture decomposition; for PI three-qubit states the paper then implies biseparability" :
                "validated numerical PPT decomposition; for two qubits the PPT criterion then implies separability"
        else
            gme=missing;biseparable=missing
            message="validated numerical PPT-mixture decomposition; for N ≥ 4 the test is inconclusive about biseparability and genuine entanglement"
        end
    else
        classification=:inconclusive
        ppt=missing;gme=missing;biseparable=missing
        message=solved ?
            "the optimizer terminated, but neither returned numerical certificate passed the requested validation tolerances" :
            "the optimizer did not terminate with a full-accuracy solved status"
    end
    PPTMixtureResult{T}(classification,ppt,gme,biseparable,
        primal.scaled_margin,T(primal_objective),dual.dual_objective,
        primal.equality_residual,primal.minimum_block_eigenvalue,
        primal.minimum_partial_transpose_eigenvalue,
        dual.stationarity_residual,dual.minimum_cone_eigenvalue,
        solver_status,Int(iterations),Float64(solve_time),atol,rtol,message)
end

function _solve_ppt_mixture end

function _ppt_tolerance(::Type{T},value,name::AbstractString) where
        T<:AbstractFloat
    value isa Real&&isfinite(value)||throw(ArgumentError(
        "$name must be a finite real number"))
    value>=0||throw(ArgumentError("$name must be nonnegative"))
    if value isa Integer
        converted=T(value)
        exactly_represented=isfinite(converted)&&try
            BigInt(converted)==BigInt(value)
        catch
            false
        end
        exactly_represented||throw(ArgumentError(
            "$name is not exactly representable in solver precision $T"))
        return converted
    end
    promote_type(T,typeof(value))===T||throw(ArgumentError(
        "$name scalar type $(typeof(value)) would narrow in solver precision $T"))
    converted=T(value)
    isfinite(converted)||throw(ArgumentError(
        "$name is not finite in solver precision $T"))
    !iszero(value)&&iszero(converted)&&throw(ArgumentError(
        "$name underflows in solver precision $T; use wider precision"))
    converted
end

function _ppt_time_limit(value)
    value isa Real||throw(ArgumentError(
        "time_limit must be a positive real number or Inf"))
    value==Inf&&return Inf
    isfinite(value)&&value>0||throw(ArgumentError(
        "time_limit must be a positive finite real number or Inf"))
    converted=_ppt_tolerance(Float64,value,"time_limit")
    converted>0||throw(ArgumentError(
        "time_limit underflows in Float64; use a larger limit"))
    converted
end

@doc raw"""
    ppt_mixture_test(rho; plan=nothing, solver=:clarabel,
                     atol=_analysis_atol(rho), rtol=_state_rtol(rho),
                     certificate_atol=nothing, certificate_rtol=nothing,
                     solver_atol=nothing, solver_rtol=nothing,
                     max_iterations=300, time_limit=Inf,
                     verbose=false, solver_options=NamedTuple(),
                     memory_budget=512*1024^2)

Test a PI qubit state for genuine multipartite entanglement with the
permutationally invariant PPT-mixture SDP of Novo, Moroder, and Gühne,
*Phys. Rev. A* **88**, 012305 (2013).

Load the optional solver first:

```julia
import Clarabel
result = ppt_mixture_test(rho)
```

The routine maximizes a common slack in multiplicity-weighted product-Schur
blocks.  It returns a [`PPTMixtureResult`](@ref), never an ambiguous Boolean.
Only `result.classification == :gme_certified` reports genuine multipartite
entanglement for arbitrary `N`, based on a validated numerical dual
certificate and the PPT-mixture criterion.  For `N >= 4`, `:ppt_mixture`
means that the criterion did not detect GME, not that the state is
biseparable.  For PI three-qubit states the paper proves the criterion
necessary and sufficient.

`plan=PPTMixturePlan(rho.basis)` reuses the sparse SDP geometry across a state
scan.  Plans are immutable and solver state is call-local.  The state is
validated but never normalized, truncated, or repaired.  Solver output is
classified only after explicit primal Schur-equality/PSD/PPT checks or an
explicit dual stationarity/PSD check.  Early termination therefore returns
`:inconclusive`.

`solver_atol` and `solver_rtol` configure the optimizer; the separate
`certificate_atol` and `certificate_rtol` control post-solve validation.
Explicit tolerances must be representable without narrowing in the plan's
conic precision.
`solver_options` is a named tuple of additional backend settings.  The
default `memory_budget` checks both the prepared sparse data and a conservative
solver-storage estimate; `Inf` explicitly opts out.
"""
function ppt_mixture_test(rho::PIState;plan=nothing,
        solver::Symbol=:clarabel,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho),
        certificate_atol=nothing,certificate_rtol=nothing,
        solver_atol=nothing,solver_rtol=nothing,
        max_iterations::Integer=300,time_limit::Real=Inf,
        verbose::Bool=false,solver_options::NamedTuple=NamedTuple(),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    atol>=0&&isfinite(atol)||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    rtol>=0&&isfinite(rtol)||throw(ArgumentError(
        "rtol must be finite and nonnegative"))
    max_iterations>0||throw(ArgumentError(
        "max_iterations must be positive"))
    max_iterations<=typemax(UInt32)||throw(ArgumentError(
        "max_iterations exceeds the Clarabel UInt32 iteration counter"))
    checked_time_limit=_ppt_time_limit(time_limit)
    _performance_memory_limit(memory_budget)
    validate_state(rho;atol=atol,rtol=rtol)
    R=_real_float_type(eltype(rho.data))
    R in (Float32,Float64)||throw(ArgumentError(
        "the optional PPT-mixture SDP backend supports Float32 and Float64 states; convert explicitly to one of these precisions"))
    solver===:clarabel||throw(ArgumentError(
        "unsupported PPT-mixture solver $solver; the available backend is :clarabel"))
    extension=Base.get_extension(@__MODULE__,_PPT_MIXTURE_CLARABEL_EXTENSION)
    extension===nothing&&throw(ArgumentError(
        "the :clarabel PPT-mixture backend is optional; run `import Clarabel` before calling ppt_mixture_test"))
    build_plan=plan===nothing
    if build_plan
        T=R
    else
        plan isa PPTMixturePlan||throw(ArgumentError(
            "plan must be a PPTMixturePlan"))
        _ppt_check_plan(plan,rho)
        T=eltype(plan.objective)
    end
    default_certificate=T===Float32 ? T(1e-3) : T(1e-7)
    default_solver=T===Float32 ? T(1e-4) : T(1e-9)
    catol=certificate_atol===nothing ? max(default_certificate,T(100)*eps(T)) :
        _ppt_tolerance(T,certificate_atol,"certificate_atol")
    crtol=certificate_rtol===nothing ? max(default_certificate,T(100)*eps(T)) :
        _ppt_tolerance(T,certificate_rtol,"certificate_rtol")
    satol=solver_atol===nothing ? max(default_solver,T(100)*eps(T)) :
        _ppt_tolerance(T,solver_atol,"solver_atol")
    srtol=solver_rtol===nothing ? max(default_solver,T(100)*eps(T)) :
        _ppt_tolerance(T,solver_rtol,"solver_rtol")
    if build_plan
        plan=PPTMixturePlan(rho.basis;T,memory_budget)
    else
        _require_performance_budget("PI PPT-mixture SDP solve",
            plan.estimated_solve_bytes,memory_budget;guidance=
            "Reduce N or pass a larger budget after checking sparse-factorization memory.")
    end
    rhs=_ppt_mixture_rhs(plan,rho)
    _solve_ppt_mixture(Val(:clarabel),plan,rhs;
        certificate_atol=catol,certificate_rtol=crtol,
        solver_atol=satol,solver_rtol=srtol,
        max_iterations,time_limit=checked_time_limit,verbose,solver_options)
end
