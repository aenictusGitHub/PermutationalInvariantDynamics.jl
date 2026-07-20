@doc raw"""
    StabilizerRenyiPlan(basis; T=Float64, memory_budget=512*1024^2)

Prepare the permutation-symmetric Pauli transform used by
[`stabilizer_renyi_entropy`](@ref).

For a pure symmetric ``N``-qubit state the second stabilizer Rényi entropy is

```math
M_2=-\log\!\left[
2^{-N}\sum_{P\in\{I,X,Y,Z\}^{\otimes N}}
\left(\mathrm{tr}\,\rho P\right)^4
\right].
```

Permutation symmetry groups the ``4^N`` strings into only
``\binom{N+3}{3}`` representatives.  This plan stores bounded normalized
Krawtchouk transforms, hypergeometric modes, and logarithmic binomial data.
It is tied to the exact `PIBasis`, is read-only after construction, and can be
shared across tasks.  The corresponding mutable scratch is owned by
[`StabilizerRenyiWorkspace`](@ref).

The prepared contraction takes ``O(N^4)`` arithmetic, the plan retains
``O(N^3)`` scalars, and a workspace retains ``O(N^2)`` scalars.  Neither setup
nor evaluation constructs a ``2^N`` state vector, a ``4^N`` Pauli list, or the
paper's collection of representative Pauli matrices.

The paper defines this magic measure for pure symmetric qubit states.  The
plan therefore requires a qubit basis containing the fully symmetric sector;
evaluation performs the remaining state and support checks.  `memory_budget`
guards the conservative setup peak, including temporary exact binomial data;
pass `Inf` only after checking available memory.
"""
struct StabilizerRenyiPlan{T<:AbstractFloat,B<:PIBasis,RMode}
    basis::B
    symmetric_partition::Partition{2}
    symmetric_sector::Int
    excitation_indices::Vector{Int}
    log_binomials::Vector{Vector{T}}
    hypergeometric_modes::Matrix{Int}
    hypergeometric_mode_probabilities::Matrix{T}
    krawtchouk::Vector{Matrix{T}}
    precision_bits::Int
    rounding_mode::RMode
    estimated_plan_bytes::BigInt
    estimated_setup_bytes::BigInt
    estimated_workspace_bytes::BigInt
end

function show(io::IO,plan::StabilizerRenyiPlan)
    print(io,"StabilizerRenyiPlan(N=$(plan.basis.N), " *
             "scalar_type=$(eltype(plan.hypergeometric_mode_probabilities)), " *
             "plan_bytes=$(plan.estimated_plan_bytes), " *
             "workspace_bytes=$(plan.estimated_workspace_bytes))")
end

@doc raw"""
    StabilizerRenyiWorkspace(plan; memory_budget=512*1024^2)

Allocate the task-owned ``O(N^2)`` scratch for a
[`StabilizerRenyiPlan`](@ref).  A workspace contains the hypergeometric table,
two complex Krawtchouk copies, and three transform buffers.  Reuse it
sequentially for state scans, but never from concurrent tasks.  Concurrent use
is detected and raises rather than corrupting the result.

`memory_budget` guards this allocation.  The plan records both its retained
size estimate and this workspace estimate for reproducible memory planning.
"""
struct StabilizerRenyiWorkspace{T<:AbstractFloat,P<:StabilizerRenyiPlan}
    plan::P
    probabilities::Matrix{T}
    representative::Matrix{Complex{T}}
    first_transform::Matrix{Complex{T}}
    transformed::Matrix{Complex{T}}
    left_krawtchouk::Matrix{Complex{T}}
    right_krawtchouk::Matrix{Complex{T}}
    busy::Threads.Atomic{Int}
end

function show(io::IO,workspace::StabilizerRenyiWorkspace)
    print(io,"StabilizerRenyiWorkspace(N=$(workspace.plan.basis.N), " *
             "scalar_type=$(eltype(workspace.probabilities)), busy=$(workspace.busy[]))")
end

function _stabilizer_renyi_estimates(N::Int,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T<:AbstractFloat
    N>=0||throw(ArgumentError("N must be nonnegative"))
    n=BigInt(N)+1
    triangle_entries=n*(n+1)÷2
    krawtchouk_entries=n*(n+1)*(2n+1)÷6
    square_entries=n^2
    real_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    complex_bytes=_scalar_retained_bytes(Complex{T};bigfloat_precision)
    int_bytes=BigInt(sizeof(Int))
    pointer_bytes=BigInt(sizeof(Ptr{Cvoid}))

    # One logarithmic triangle, normalized Krawtchouk matrices, a mode table,
    # and its probability table are retained.  Header allowances deliberately
    # overestimate the small Vector/Matrix objects.
    headers=(BigInt(N)+8)*8pointer_bytes
    plan_bytes=(triangle_entries+krawtchouk_entries+square_entries)*real_bytes+
               square_entries*int_bytes+headers

    # Pascal's triangle is setup-only.  A binomial at row N has at most N+1
    # bits; include a generous boxed-BigInt/header allowance per entry.
    bigint_entry=8pointer_bytes+BigInt(cld(N+1,8))+32
    setup_bytes=plan_bytes+triangle_entries*bigint_entry

    # One real probability table and five complex transform matrices.
    workspace_bytes=square_entries*(real_bytes+5complex_bytes)+16pointer_bytes
    (;plan_bytes,setup_bytes,workspace_bytes)
end

function _stabilizer_log_exact_integer(::Type{T},value::BigInt) where
        T<:AbstractFloat
    value>0||throw(ArgumentError("logarithm input must be positive"))
    value==1&&return zero(T)
    direct=try
        T(value)
    catch
        T(Inf)
    end
    isfinite(direct)&&return log(direct)

    exponent=ndigits(value;base=2)-1
    W=T===Float16 ? Float64 : T
    mantissa=W(value//(big(1)<<exponent))
    wide=log(mantissa)+W(exponent)*log(W(2))
    isfinite(wide)&&wide<=W(floatmax(T))||throw(ArgumentError(
        "a logarithmic binomial coefficient is not representable in $T; use a wider scalar type"))
    result=T(wide)
    isfinite(result)||throw(ArgumentError(
        "a logarithmic binomial coefficient is not representable in $T; use a wider scalar type"))
    result
end

function _stabilizer_pascal_triangle(N::Int)
    rows=Vector{Vector{BigInt}}(undef,N+1)
    rows[1]=BigInt[1]
    for n in 1:N
        previous=rows[n]
        row=Vector{BigInt}(undef,n+1)
        row[1]=row[end]=1
        @inbounds for k in 1:n-1
            row[k+1]=previous[k]+previous[k+1]
        end
        rows[n+1]=row
    end
    rows
end

function _stabilizer_krawtchouk(size::Int,::Type{T}) where T<:AbstractFloat
    size>=0||throw(ArgumentError("Krawtchouk size must be nonnegative"))
    K=Matrix{T}(undef,size+1,size+1)
    if iszero(size)
        K[1,1]=one(T)
        return K
    end
    half=fld(size,2)
    s=T(size)
    @inbounds for d in 0:size
        K[1,d+1]=one(T)
        if half>=1
            K[2,d+1]=(T(size)-T(2d))/s
        end
        for k in 1:half-1
            denominator=T(size-k)
            K[k+2,d+1]=((T(size)-T(2d))/denominator)*K[k+1,d+1]-
                        (T(k)/denominator)*K[k,d+1]
        end
        if iseven(size)&&isodd(d)
            # K_{s/2}(d)=0 follows exactly from reflection symmetry.  Setting
            # this structural zero avoids retaining recurrence roundoff.
            K[half+1,d+1]=zero(T)
        end
        sign=isodd(d) ? -one(T) : one(T)
        for k in half+1:size
            K[k+1,d+1]=sign*K[size-k+1,d+1]
        end
    end
    all(isfinite,K)||throw(ArgumentError(
        "normalized Krawtchouk setup is not finite in $T; use a wider scalar type"))
    K
end

function _stabilizer_excitation_indices(basis::PIBasis,sector::Int)
    indices=zeros(Int,basis.N+1)
    for (pattern_index,pattern) in pairs(basis.patterns[sector])
        occupations=content(pattern)
        excitation=occupations[2]
        0<=excitation<=basis.N||error(
            "internal error: invalid symmetric-sector excitation")
        iszero(indices[excitation+1])||error(
            "internal error: duplicate symmetric-sector excitation")
        indices[excitation+1]=pattern_index
    end
    all(!iszero,indices)||error(
        "internal error: symmetric qubit sector lacks a Dicke excitation")
    indices
end

function _stabilizer_build_plan(basis::PIBasis,::Type{T},estimates,
        precision_bits::Int,rounding_mode) where T<:AbstractFloat
    N=basis.N
    partition=Partition((N,0))
    sector=get(basis.index,partition,0)
    sector>0||throw(ArgumentError(
        "stabilizer Rényi analysis requires the fully symmetric sector $(partition.parts)"))
    excitation_indices=_stabilizer_excitation_indices(basis,sector)

    binomials=_stabilizer_pascal_triangle(N)
    log_binomials=Vector{Vector{T}}(undef,N+1)
    for n in 0:N
        log_binomials[n+1]=T[_stabilizer_log_exact_integer(T,value)
                             for value in binomials[n+1]]
    end

    modes=Matrix{Int}(undef,N+1,N+1)
    mode_probabilities=Matrix{T}(undef,N+1,N+1)
    denominator_row=binomials[N+1]
    for L in 0:N,n in 0:N
        lower=max(0,L-(N-n))
        upper=min(n,L)
        # The exact mode formula avoids an Int multiplication overflow before
        # the memory safeguard has a chance to reject an impractical request.
        mode=clamp(Int(fld(big(L+1)*big(n+1),big(N+2))),lower,upper)
        numerator=binomials[n+1][mode+1]*
                  binomials[N-n+1][L-mode+1]
        probability=_checked_exact_ratio(T,numerator,denominator_row[L+1];
            context="hypergeometric mode probability for N=$N, n=$n, L=$L")
        modes[L+1,n+1]=mode
        mode_probabilities[L+1,n+1]=probability
    end

    krawtchouk=Matrix{T}[_stabilizer_krawtchouk(size,T) for size in 0:N]
    StabilizerRenyiPlan{T,typeof(basis),typeof(rounding_mode)}(
        basis,partition,sector,excitation_indices,log_binomials,modes,
        mode_probabilities,krawtchouk,precision_bits,rounding_mode,
        estimates.plan_bytes,estimates.setup_bytes,estimates.workspace_bytes)
end

function _with_stabilizer_precision(f,::Type{T},precision_bits,
        rounding_mode) where T<:AbstractFloat
    if T===BigFloat
        return setrounding(BigFloat,rounding_mode) do
            setprecision(BigFloat,precision_bits) do
                f()
            end
        end
    end
    f()
end

function StabilizerRenyiPlan(basis::PIBasis;
        T::Type{<:AbstractFloat}=Float64,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    isconcretetype(T)||throw(ArgumentError(
        "StabilizerRenyiPlan requires a concrete floating-point type"))
    basis.d==2||throw(ArgumentError(
        "stabilizer Rényi entropy is implemented for qubits only"))
    precision_bits=T===BigFloat ? precision(BigFloat) : 0
    rounding_mode=T===BigFloat ? rounding(BigFloat) : nothing
    estimates=_stabilizer_renyi_estimates(basis.N,T;
        bigfloat_precision=T===BigFloat ? precision_bits : precision(BigFloat))
    _require_performance_budget("stabilizer Rényi transform preparation",
        estimates.setup_bytes,memory_budget;guidance=
        "Reduce N, use a sector-restricted basis containing (N,0), or reuse one prepared plan.")
    _with_stabilizer_precision(T,precision_bits,rounding_mode) do
        _stabilizer_build_plan(
            basis,T,estimates,precision_bits,rounding_mode)
    end
end

function _stabilizer_workspace(plan::StabilizerRenyiPlan{T}) where
        T<:AbstractFloat
    dimension=plan.basis.N+1
    StabilizerRenyiWorkspace{T,typeof(plan)}(
        plan,
        zeros(T,dimension,dimension),
        zeros(Complex{T},dimension,dimension),
        zeros(Complex{T},dimension,dimension),
        zeros(Complex{T},dimension,dimension),
        zeros(Complex{T},dimension,dimension),
        zeros(Complex{T},dimension,dimension),
        Threads.Atomic{Int}(0))
end

function StabilizerRenyiWorkspace(plan::StabilizerRenyiPlan{T};
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where T<:AbstractFloat
    _require_performance_budget("stabilizer Rényi workspace construction",
        plan.estimated_workspace_bytes,memory_budget;guidance=
        "Reduce N or allocate one workspace and reuse it sequentially.")
    _with_stabilizer_precision(T,plan.precision_bits,plan.rounding_mode) do
        _stabilizer_workspace(plan)
    end
end

function _stabilizer_tolerance(::Type{T},value,name::AbstractString) where
        T<:AbstractFloat
    value isa Real&&isfinite(value)||throw(ArgumentError(
        "$name must be a finite real number"))
    value>=0||throw(ArgumentError("$name must be nonnegative"))
    converted=try
        T(value)
    catch error
        throw(ArgumentError(
            "$name is not representable in $T: $(sprint(showerror,error))"))
    end
    isfinite(converted)||throw(ArgumentError("$name is not finite in $T"))
    !iszero(value)&&iszero(converted)&&throw(ArgumentError(
        "$name underflows in $T; use wider precision"))
    converted
end

function _stabilizer_check_ownership(rho::PIState,
        plan::StabilizerRenyiPlan{T},workspace::StabilizerRenyiWorkspace) where
        T<:AbstractFloat
    rho.basis===plan.basis||throw(ArgumentError(
        "the stabilizer Rényi plan belongs to a different PIBasis"))
    workspace.plan===plan||throw(ArgumentError(
        "the stabilizer Rényi workspace belongs to a different plan"))
    S=_real_float_type(eltype(rho.data))
    promote_type(S,T)===T||throw(ArgumentError(
        "plan precision $T would narrow state precision $S; prepare a wider plan"))
    if T===BigFloat&&S===BigFloat
        input_precision=maximum(_bigfloat_input_precision,rho.data;
            init=precision(BigFloat))
        plan.precision_bits>=input_precision||throw(ArgumentError(
            "plan precision $(plan.precision_bits) bits would narrow the " *
            "$input_precision-bit BigFloat state; prepare the plan at wider precision"))
    end
    nothing
end

function _stabilizer_validate_state(rho::PIState,
        plan::StabilizerRenyiPlan{T},atol::T,rtol::T) where T<:AbstractFloat
    # The paper's pure-state vector lives in the symmetric irrep.  Requiring
    # exact zero support elsewhere prevents a tolerance from silently
    # truncating a genuinely multi-sector PI density operator.
    for sector in eachindex(rho.basis.sectors)
        sector==plan.symmetric_sector&&continue
        first=rho.basis.offsets[sector]
        last=rho.basis.offsets[sector+1]-1
        all(iszero,@view(rho.data[first:last]))||throw(ArgumentError(
            "the paper's stabilizer Rényi entropy requires exact support only in the fully symmetric sector"))
    end
    validate_state(rho;atol=atol,rtol=rtol)
    block=coefficient_block(rho,plan.symmetric_partition)
    block_purity=mapreduce(value->abs2(Complex{T}(value)),+,block;
                           init=zero(T))
    tolerance=atol+rtol*max(one(T),abs(block_purity))
    abs(block_purity-one(T))<=tolerance||throw(ArgumentError(
        "stabilizer Rényi entropy is a pure-state measure in the cited work; " *
        "the symmetric Schur block has purity $block_purity (tolerance $tolerance)"))
    block
end

function _stabilizer_fill_hypergeom!(workspace::StabilizerRenyiWorkspace{T},
        L::Int) where T<:AbstractFloat
    plan=workspace.plan
    N=plan.basis.N
    probabilities=workspace.probabilities
    @inbounds for n in 0:N
        for D in 0:L
            probabilities[n+1,D+1]=zero(T)
        end
        lower=max(0,L-(N-n))
        upper=min(n,L)
        mode=plan.hypergeometric_modes[L+1,n+1]
        probabilities[n+1,mode+1]=
            plan.hypergeometric_mode_probabilities[L+1,n+1]

        for D in mode:upper-1
            ratio=(T(n-D)/T(D+1))*(T(L-D)/T(N-n-L+D+1))
            probabilities[n+1,D+2]=probabilities[n+1,D+1]*ratio
        end
        for D in mode:-1:lower+1
            ratio=(T(D)/T(n-D+1))*(T(N-n-L+D)/T(L-D+1))
            probabilities[n+1,D]=probabilities[n+1,D+1]*ratio
        end

        total=zero(T)
        for D in lower:upper
            total+=probabilities[n+1,D+1]
        end
        isfinite(total)&&total>zero(T)||throw(ArgumentError(
            "hypergeometric recurrence failed in $T; use wider precision"))
        for D in lower:upper
            probabilities[n+1,D+1]/=total
        end
    end
    probabilities
end

@inline function _stabilizer_logsum_add(maximum_term::T,scaled_sum::T,
        term::T) where T<:AbstractFloat
    if term>maximum_term
        scaled=isfinite(maximum_term) ?
            scaled_sum*exp(maximum_term-term)+one(T) : one(T)
        return term,scaled
    end
    maximum_term,scaled_sum+exp(term-maximum_term)
end

@inline function _stabilizer_quarter_phase(::Type{T},power::Int) where
        T<:AbstractFloat
    residue=mod(power,4)
    residue==0&&return complex(one(T),zero(T))
    residue==1&&return complex(zero(T),one(T))
    residue==2&&return complex(-one(T),zero(T))
    complex(zero(T),-one(T))
end

function _stabilizer_entropy_impl(rho::PIState,
        plan::StabilizerRenyiPlan{T},workspace::StabilizerRenyiWorkspace{T};
        base,atol,rtol) where T<:AbstractFloat
    atolT=_stabilizer_tolerance(T,atol,"atol")
    rtolT=_stabilizer_tolerance(T,rtol,"rtol")
    baseT=try
        T(base)
    catch error
        throw(ArgumentError(
            "logarithm base is not representable in $T: $(sprint(showerror,error))"))
    end
    isfinite(baseT)&&baseT>zero(T)&&baseT!=one(T)||throw(ArgumentError(
        "logarithm base must be positive, finite, and different from one"))
    block=_stabilizer_validate_state(rho,plan,atolT,rtolT)

    N=plan.basis.N
    logtwo=log(T(2))
    maximum_fourth=T(-Inf);sum_fourth=zero(T)
    maximum_second=T(-Inf);sum_second=zero(T)
    expectation_scale=one(T)
    imaginary_tolerance=atolT+rtolT+T(64)*(T(N)+one(T))*eps(T)

    for L in 0:N
        M=N-L
        _stabilizer_fill_hypergeom!(workspace,L)
        H=@view workspace.representative[1:L+1,1:M+1]
        left=@view workspace.first_transform[1:L+1,1:M+1]
        transformed=@view workspace.transformed[1:L+1,1:M+1]
        leftK=@view workspace.left_krawtchouk[1:L+1,1:L+1]
        rightK=@view workspace.right_krawtchouk[1:M+1,1:M+1]
        source_left=plan.krawtchouk[L+1]
        source_right=plan.krawtchouk[M+1]

        @inbounds for column in 1:L+1,row in 1:L+1
            leftK[row,column]=complex(source_left[row,column],zero(T))
        end
        @inbounds for column in 1:M+1,row in 1:M+1
            rightK[row,column]=complex(source_right[row,column],zero(T))
        end
        @inbounds for C in 0:M,D in 0:L
            ket_excitation=C+D
            bra_excitation=M-C+D
            ket_index=plan.excitation_indices[ket_excitation+1]
            bra_index=plan.excitation_indices[bra_excitation+1]
            left_probability=workspace.probabilities[ket_excitation+1,D+1]
            right_probability=workspace.probabilities[bra_excitation+1,D+1]
            scale=sqrt(left_probability)*sqrt(right_probability)
            H[D+1,C+1]=scale*Complex{T}(block[ket_index,bra_index])
        end

        mul!(left,leftK,H)
        mul!(transformed,left,transpose(rightK))

        log_choose_split=plan.log_binomials[N+1][L+1]
        @inbounds for ny in 0:M
            phase=_stabilizer_quarter_phase(T,ny)
            log_choose_y=plan.log_binomials[M+1][ny+1]
            for nz in 0:L
                expectation_complex=phase*transformed[nz+1,ny+1]
                abs(imag(expectation_complex))<=imaginary_tolerance||throw(ArgumentError(
                    "a symmetric Pauli expectation has imaginary residual " *
                    "$(imag(expectation_complex)), exceeding $imaginary_tolerance; " *
                    "use wider precision"))
                expectation=real(expectation_complex)
                isfinite(expectation)||throw(ArgumentError(
                    "a symmetric Pauli expectation is not finite in $T"))
                expectation_scale=max(expectation_scale,abs(expectation))
                iszero(expectation)&&continue
                log_degeneracy=log_choose_split+
                    plan.log_binomials[L+1][nz+1]+log_choose_y
                log_absolute=log(abs(expectation))
                maximum_fourth,sum_fourth=_stabilizer_logsum_add(
                    maximum_fourth,sum_fourth,log_degeneracy+T(4)*log_absolute)
                maximum_second,sum_second=_stabilizer_logsum_add(
                    maximum_second,sum_second,log_degeneracy+T(2)*log_absolute)
            end
        end
    end

    isfinite(maximum_fourth)&&sum_fourth>zero(T)||error(
        "internal error: identity Pauli representative was not accumulated")
    log_fourth=maximum_fourth+log(sum_fourth)-T(N)*logtwo
    log_second=maximum_second+log(sum_second)-T(N)*logtwo

    # Pauli Parseval is an independent end-to-end check of the transform.
    # Its tolerance includes the length-N matrix-product accumulation but is
    # not used to relax the direct pure-state test above.
    parseval_tolerance=max(atolT+rtolT,
        T(64)*(T(N)+one(T))*eps(T)*expectation_scale)
    abs(expm1(log_second))<=parseval_tolerance||throw(ArgumentError(
        "symmetric Pauli transform failed its purity sum rule: " *
        "2^-N sum_P <P>^2=$(exp(log_second)) (tolerance $parseval_tolerance); " *
        "use wider precision"))

    # Exact M2 is nonnegative.  Repair only a positive log-fourth residual
    # that is already certified as transform roundoff; larger violations are
    # reported instead of clipped.
    if log_fourth>zero(T)
        log_fourth<=parseval_tolerance||throw(ArgumentError(
            "computed Pauli fourth moment exceeds one beyond numerical tolerance; use wider precision"))
        log_fourth=zero(T)
    end
    -log_fourth/log(baseT)
end

@doc raw"""
    stabilizer_renyi_entropy(rho; base=\mathit{e}, plan=nothing,
                             workspace=nothing,
                             atol=_analysis_atol(rho),
                             rtol=_state_rtol(rho),
                             memory_budget=512*1024^2)
    stabilizer_renyi_entropy(rho, plan, workspace; base=\mathit{e}, ...)

Compute the second stabilizer Rényi entropy ``M_2`` (nonstabilizerness or
"magic") of a pure permutation-symmetric qubit state using the method of
Passarelli, Fazio, and Lucignano, *Phys. Rev. A* **110**, 022436 (2024).
The default natural logarithm matches the paper; `base=2` reports bits.

The algorithm sums one representative for each Pauli multiplicity quadruple
with its exact multinomial degeneracy.  A bounded hypergeometric--Krawtchouk
transform improves the paper's stored-matrix contraction to ``O(N^4)`` time,
with an ``O(N^3)`` reusable plan and ``O(N^2)`` task-owned workspace.  The
final fourth moment is reduced by log-sum-exp, so neither ``2^N`` nor a large
multinomial coefficient is converted separately.

This function follows the paper's pure-state definition strictly.  It
validates the density operator, verifies unit purity, and requires exact zero
support outside the fully symmetric Schur sector.  Mixed states and general
multi-sector PI states are rejected: applying the same fourth-moment formula
to a mixed state is not the cited magic monotone (the maximally mixed state is
a simple counterexample).  The input is never normalized, projected,
symmetrized, or otherwise repaired.

For one call, omit `plan` and `workspace`.  For a scan, construct one
`StabilizerRenyiPlan(rho.basis; T=...)` and one workspace per concurrent task.
Both are tied to their exact owner.  A wider plan may analyze a narrower
state, but a narrower plan is rejected.  `memory_budget` guards only storage
allocated by the convenience route; a fully supplied plan and workspace have
already paid that cost.
"""
function stabilizer_renyi_entropy(rho::PIState;
        base=ℯ,plan=nothing,workspace=nothing,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    if workspace!==nothing
        workspace isa StabilizerRenyiWorkspace||throw(ArgumentError(
            "workspace must be a StabilizerRenyiWorkspace"))
        if plan===nothing
            plan=workspace.plan
        elseif workspace.plan!==plan
            throw(ArgumentError(
                "the supplied workspace belongs to a different stabilizer Rényi plan"))
        end
    end

    if plan===nothing
        T=_real_float_type(eltype(rho.data))
        precision_bits=T===BigFloat ? precision(BigFloat) : 0
        estimates=_stabilizer_renyi_estimates(rho.basis.N,T;
            bigfloat_precision=T===BigFloat ? precision_bits : precision(BigFloat))
        combined_peak=max(estimates.setup_bytes,
                          estimates.plan_bytes+estimates.workspace_bytes)
        _require_performance_budget("stabilizer Rényi convenience evaluation",
            combined_peak,memory_budget;guidance=
            "Prepare and reuse a plan/workspace, reduce N, or pass a larger explicit budget.")
        plan=StabilizerRenyiPlan(rho.basis;T=T,memory_budget=Inf)
    else
        plan isa StabilizerRenyiPlan||throw(ArgumentError(
            "plan must be a StabilizerRenyiPlan"))
    end
    if workspace===nothing
        workspace=StabilizerRenyiWorkspace(plan;memory_budget)
    end
    stabilizer_renyi_entropy(rho,plan,workspace;base,atol,rtol)
end

function stabilizer_renyi_entropy(rho::PIState,
        plan::StabilizerRenyiPlan,workspace::StabilizerRenyiWorkspace;
        base=ℯ,atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho))
    _stabilizer_check_ownership(rho,plan,workspace)
    Threads.atomic_cas!(workspace.busy,0,1)==0||throw(ArgumentError(
        "StabilizerRenyiWorkspace is already in use; allocate one workspace per concurrent task"))
    try
        T=eltype(plan.hypergeometric_mode_probabilities)
        _with_stabilizer_precision(T,plan.precision_bits,plan.rounding_mode) do
            _stabilizer_entropy_impl(rho,plan,workspace;base,atol,rtol)
        end
    finally
        workspace.busy[]=0
    end
end

function stabilizer_renyi_entropy(rho::PIState,plan::StabilizerRenyiPlan;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    workspace=StabilizerRenyiWorkspace(plan;memory_budget)
    stabilizer_renyi_entropy(rho,plan,workspace;kwargs...)
end
