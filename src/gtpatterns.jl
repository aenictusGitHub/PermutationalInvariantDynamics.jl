"""Immutable Gelfand–Tsetlin pattern, stored row-by-row for rows 1 through D."""
struct GTPattern{D,L}
    entries::NTuple{L,Int}
    function GTPattern{D}(entries::Tuple; check::Bool=true) where D
        L=D*(D+1)÷2
        length(entries) == L || throw(ArgumentError("wrong number of GT entries"))
        vals = ntuple(i->Int(entries[i]),L)
        g = new{D,L}(vals)
        check && !isvalid(g) && throw(ArgumentError("invalid GT pattern"))
        g
    end
end
GTPattern(rows::AbstractVector{<:AbstractVector}; check=true) = begin
    D=length(rows); all(k->length(rows[k])==k,1:D) || throw(ArgumentError("row k must have length k"))
    GTPattern{D}(Tuple(vcat(rows...)); check=check)
end
_gidx(i,k) = k*(k-1)÷2+i

"""
    gt_entry(g, i, k)

Return entry `i` of row `k` of the Gelfand--Tsetlin pattern `g`. Both indices
are one-based; row `k` contains `k` entries.
"""
gt_entry(g::GTPattern, i::Integer, k::Integer) = g.entries[_gidx(i,k)]

"""Return the top-row partition labeling the irrep of `g`."""
shape(g::GTPattern{D}) where D = Partition(ntuple(i->gt_entry(g,i,D),D))

"""
    isvalid(g)

Return whether all entries of `g` are nonnegative and satisfy the GT
interlacing inequalities.
"""
function isvalid(g::GTPattern{D}) where D
    all(x->x>=0,g.entries) || return false
    for k in 2:D, i in 1:k-1
        gt_entry(g,i,k) >= gt_entry(g,i,k-1) >= gt_entry(g,i+1,k) || return false
    end
    true
end

@inline function _content_entry(g::GTPattern,::Val{K}) where K
    value=0
    @inbounds for i in 1:K
        value+=g.entries[_gidx(i,K)]
    end
    @inbounds for i in 1:K-1
        value-=g.entries[_gidx(i,K-1)]
    end
    value
end
@inline _content_tuple(g::GTPattern,::Val{0})=()
@inline _content_tuple(g::GTPattern,::Val{K}) where K=
    (_content_tuple(g,Val(K-1))...,_content_entry(g,Val(K)))

"""
    content(g)

Return the length-`D` weight-content tuple of `g`, obtained from successive
row-sum differences. Its entries give the local-label occupations and sum to
`weight(shape(g))`.
"""
function content(g::GTPattern{D}) where D
    # `content` is used in the innermost staging loops for one-body geometry
    # and LR intertwiners. Compile-time tuple recursion is allocation-free on
    # Julia 1.10 as well as newer releases; a closure-based static `ntuple`
    # still boxed 32 bytes on the minimum supported release.
    _content_tuple(g,Val(D))
end
show(io::IO,g::GTPattern) = print(io,"GTPattern",g.entries)

function _interlacing_rows(upper::Vector{Int})
    k=length(upper)-1; out=Vector{Vector{Int}}()
    function rec(v,i)
        if i>k; push!(out,copy(v)); return; end
        for x in upper[i]:-1:upper[i+1]
            push!(v,x); rec(v,i+1); pop!(v)
        end
    end
    rec(Int[],1); out
end

"""
    gt_patterns(p)

Enumerate all Gelfand--Tsetlin patterns with top row `p`. Patterns are sorted
by the ascending lexicographic order of their stored entry tuples, which is
the ordering used inside every PI Schur block.
"""
function gt_patterns(p::Partition{D}) where D
    L=D*(D+1)÷2
    out=GTPattern{D,L}[]
    function descend(rows::Vector{Vector{Int}})
        if length(rows)==D
            ordered=reverse(rows)
            push!(out,GTPattern(ordered)); return
        end
        for row in _interlacing_rows(rows[end])
            push!(rows,row); descend(rows); pop!(rows)
        end
    end
    descend([collect(p.parts)])
    sort!(out; by=g->g.entries)
end

"""Return shift row positions `(tau_i,...,tau_D)`, or `nothing` if selection fails."""
function triangular_shift(mu::GTPattern{D}, lam::GTPattern{D}, j::Integer) where D
    0 <= j < D || return nothing
    tau=Int[]
    for k in 1:D
        dif=[gt_entry(lam,i,k)-gt_entry(mu,i,k) for i in 1:k]
        if k <= j
            all(iszero,dif) || return nothing
        else
            count(==(1),dif)==1 && all(x->x==0||x==1,dif) || return nothing
            push!(tau,findfirst(==(1),dif))
        end
    end
    tau
end
