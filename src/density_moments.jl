"""
    DensityPowerWorkspace(basis; T=Float64, memory_budget=512*1024^2)
    DensityPowerWorkspace(rho; memory_budget=512*1024^2)
    DensityPowerWorkspace(reduction_workspace; memory_budget=512*1024^2)

Allocate the three task-owned square buffers used by [`trace_power`](@ref) and
[`reduced_trace_power`](@ref).  Their side length is the largest Schur-block
dimension retained by `basis`, rather than the exponentially large Hilbert
dimension `d^N`.  The state constructor selects its real component type and,
for `BigFloat`, its stored precision.

For a nontrivial reduction `0 < k < N`, the reduction-workspace constructor
selects the exact output basis and arithmetic context of a reduction-capable
[`ReductionWorkspace`](@ref), which is the least error-prone setup for
repeated reduced moments. Endpoint moments need no reduction scratch; use the
state constructor for `k=N`.

A workspace is tied to one exact [`PIBasis`](@ref), scalar type, and
`BigFloat` precision/rounding context.  The immutable basis may be shared, but
one mutable workspace must not be used concurrently.  Construction checks the
requested `memory_budget`; `Inf` is the explicit opt-out.
"""
mutable struct DensityPowerWorkspace{R<:AbstractFloat,B,M,Q,S}
    basis::B
    first::M
    second::M
    third::M
    multiplicity_scales::S
    precision_bits::Int
    rounding_mode::Q
    retained_bytes::BigInt
end

function _density_power_multiplicity_bits_bound(basis::PIBasis)
    basis.N<=1&&return big(1)
    # Every symmetric-group multiplicity is at most N! <= N^N.  This bound
    # is intentionally coefficient-free so it can be checked before creating
    # even one potentially large exact multiplicity.
    BigInt(basis.N)*BigInt(ndigits(basis.N;base=2))+1
end

function _density_power_scale_storage_bound(
        basis::PIBasis,::Type{R},precision_bits::Int) where R<:AbstractFloat
    count=BigInt(length(basis.sectors))
    payload=cld(_density_power_multiplicity_bits_bound(basis),BigInt(8))+1
    pointer_bytes=BigInt(sizeof(Ptr{Cvoid}))
    integer_bytes=BigInt(sizeof(Int))
    scalar_bytes=_scalar_retained_bytes(
        R;bigfloat_precision=precision_bits)
    # Each scale owns two BigInts, two floating factors, a Boolean/exponent,
    # and allocator-dependent headers.  The generous metadata allowance
    # keeps this a construction-time upper bound rather than a post-hoc size
    # measurement.
    count*(2payload+2pointer_bytes+2scalar_bytes+2integer_bytes+512)+512
end

function _density_power_workspace_storage_bound(
        basis::PIBasis,::Type{R},precision_bits::Int) where R<:AbstractFloat
    largest=maximum(length,basis.patterns;init=1)
    matrix_bytes=_performance_array_bytes(
        largest,Complex{R},3;bigfloat_precision=precision_bits)
    matrix_bytes+
        _density_power_scale_storage_bound(basis,R,precision_bits)+1024
end

function _density_power_scale_peak_bound(basis::PIBasis,order::Int)
    order<=2&&return big(0)
    bit_count=BigInt(order-1)*
        _density_power_multiplicity_bits_bound(basis)+1
    4cld(bit_count,BigInt(8))+4096
end

function _density_power_workspace(
        basis::PIBasis,::Type{R},precision_bits::Int,rounding_mode,
        memory_budget) where R<:AbstractFloat
    if R===BigFloat&&
            (precision(BigFloat)!=precision_bits||
             rounding(BigFloat)!=rounding_mode)
        return setrounding(BigFloat,rounding_mode) do
            setprecision(BigFloat,precision_bits) do
                _density_power_workspace(
                    basis,R,precision_bits,rounding_mode,memory_budget)
            end
        end
    end
    largest=maximum(length,basis.patterns;init=1)
    retained=_density_power_workspace_storage_bound(
        basis,R,precision_bits)
    _require_performance_budget(
        "density-power workspace",retained,memory_budget;guidance=
        "Raise the explicit budget or evaluate only the dedicated q=1 or q=2 path.")
    scales=[_prepare_exact_scale(
        R,symmetric_group_dimension(sector),one(BigInt),Val(true);
        context="prepared multiplicity-weighted density block")
        for sector in basis.sectors]
    matrix()=zeros(Complex{R},largest,largest)
    DensityPowerWorkspace{
        R,typeof(basis),Matrix{Complex{R}},typeof(rounding_mode),typeof(scales)}(
        basis,matrix(),matrix(),matrix(),scales,precision_bits,rounding_mode,
        retained)
end

function DensityPowerWorkspace(
        basis::PIBasis;T::Type{R}=Float64,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where R<:AbstractFloat
    isconcretetype(R)||throw(ArgumentError(
        "DensityPowerWorkspace requires a concrete AbstractFloat type"))
    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    _density_power_workspace(
        basis,R,precision_bits,rounding_mode,memory_budget)
end

function DensityPowerWorkspace(
        rho::PIState;memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    R=_real_float_type(eltype(rho.data))
    precision_bits=R===BigFloat ? begin
        bounds=_reduction_state_precision_bounds(rho)
        bounds[2]
    end : precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    _density_power_workspace(
        rho.basis,R,precision_bits,rounding_mode,memory_budget)
end

function DensityPowerWorkspace(
        work::ReductionWorkspace;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _require_reduction_workspace_mode(work,:reduction)
    0<work.plan.k<work.plan.basis.N||throw(ArgumentError(
        "DensityPowerWorkspace(reduction_workspace) requires 0 < k < N; " *
        "endpoint moments use no reduction scratch"))
    R=_real_float_type(work.Ttype)
    _density_power_workspace(
        work.plan.output_basis,R,work.precision_bits,work.rounding_mode,
        memory_budget)
end

function Base.show(io::IO,workspace::DensityPowerWorkspace{R}) where R
    print(io,"DensityPowerWorkspace(maximum_block=",
          size(workspace.first,1),", scalar_type=",R)
    R===BigFloat&&print(io,", precision_bits=",workspace.precision_bits)
    print(io,")")
end

@inline function _density_power_with_precision(
        f,workspace::DensityPowerWorkspace{R}) where R
    R===BigFloat||return f()
    setrounding(BigFloat,workspace.rounding_mode) do
        setprecision(BigFloat,workspace.precision_bits) do
            f()
        end
    end
end

function _check_density_power_workspace(
        workspace::DensityPowerWorkspace{R},basis::PIBasis,
        ::Type{S}) where {R<:AbstractFloat,S<:AbstractFloat}
    workspace.basis===basis||throw(ArgumentError(
        "DensityPowerWorkspace was prepared for a different PIBasis"))
    R===S||throw(ArgumentError(
        "DensityPowerWorkspace scalar type $R does not match the required scalar type $S"))
    largest=maximum(length,basis.patterns;init=1)
    expected=(largest,largest)
    for (label,buffer) in (("first",workspace.first),
                           ("second",workspace.second),
                           ("third",workspace.third))
        size(buffer)==expected||throw(DimensionMismatch(
            "DensityPowerWorkspace $label buffer has dimensions $(size(buffer)); expected $expected"))
        if R===BigFloat
            bounds=_local_factor_precision_bounds(buffer)
            bounds==(workspace.precision_bits,workspace.precision_bits)||
                throw(ArgumentError(
                    "DensityPowerWorkspace $label buffer has precision range $bounds, expected $(workspace.precision_bits) bits"))
        end
    end
    buffers=(workspace.first,workspace.second,workspace.third)
    for right in 2:length(buffers),left in 1:right-1
        Base.mightalias(buffers[left],buffers[right])&&throw(ArgumentError(
            "DensityPowerWorkspace buffers must not alias each other"))
    end
    length(workspace.multiplicity_scales)==length(basis.sectors)||
        throw(DimensionMismatch(
            "DensityPowerWorkspace has the wrong number of multiplicity scales"))
    if R===BigFloat
        for scale in workspace.multiplicity_scales
            precision(scale.factor)==workspace.precision_bits&&
                precision(scale.mantissa)==workspace.precision_bits||
                throw(ArgumentError(
                    "DensityPowerWorkspace has a multiplicity scale at the wrong BigFloat precision"))
        end
    end
    workspace
end

function _check_density_power_workspace(
        workspace,basis::PIBasis,::Type{R}) where R<:AbstractFloat
    workspace isa DensityPowerWorkspace||throw(ArgumentError(
        "workspace must be a DensityPowerWorkspace"))
    _check_density_power_workspace(workspace,basis,R)
end

function _density_power_order(order)
    order isa Integer&&!(order isa Bool)||throw(ArgumentError(
        "density-matrix power order must be an integer, not $(typeof(order))"))
    order>=1||throw(ArgumentError(
        "density-matrix power order must be positive"))
    order<=typemax(Int)||throw(ArgumentError(
        "density-matrix power order exceeds the addressable Int range"))
    Int(order)
end

@inline function _density_power_buffer(
        workspace::DensityPowerWorkspace,index::Int,n::Int)
    index==1&&return view(workspace.first,1:n,1:n)
    index==2&&return view(workspace.second,1:n,1:n)
    index==3&&return view(workspace.third,1:n,1:n)
    error("internal error: invalid density-power buffer index $index")
end

# Binary powering with one setup copy and no multiplication by an identity.
# `result_index` and `factor_index` always name distinct buffers once the first
# set bit has been copied.  The remaining buffer is therefore a legal,
# non-aliasing `mul!` destination.
function _density_matrix_power!(
        workspace::DensityPowerWorkspace,C::AbstractMatrix,
        exponent::Int)
    exponent>=1||throw(ArgumentError(
        "internal density-matrix exponent must be positive"))
    n=size(C,1)
    size(C,2)==n||throw(DimensionMismatch(
        "a density Schur block must be square"))
    factor_index=1
    factor=_density_power_buffer(workspace,factor_index,n)
    Base.mightalias(factor,C)||copyto!(factor,C)
    result_index=0
    remaining=exponent
    while remaining>0
        if isodd(remaining)
            if result_index==0
                if remaining==1
                    # The final set bit needs no independent result copy.
                    result_index=factor_index
                else
                    result_index=factor_index==1 ? 2 : 1
                    copyto!(_density_power_buffer(workspace,result_index,n),
                            _density_power_buffer(workspace,factor_index,n))
                end
            else
                destination=6-result_index-factor_index
                mul!(_density_power_buffer(workspace,destination,n),
                     _density_power_buffer(workspace,result_index,n),
                     _density_power_buffer(workspace,factor_index,n))
                result_index=destination
            end
        end
        remaining>>=1
        remaining==0&&break
        destination=result_index==0 ? (factor_index==1 ? 2 : 1) :
                                      6-result_index-factor_index
        mul!(_density_power_buffer(workspace,destination,n),
             _density_power_buffer(workspace,factor_index,n),
             _density_power_buffer(workspace,factor_index,n))
        factor_index=destination
    end
    _density_power_buffer(workspace,result_index,n),result_index
end

function _density_power_scale_peak_bytes(
        multiplicity::Integer,order::Int)
    (order<=2||multiplicity==1)&&return big(0)
    # `f^(q-1)`, its exact-scale copy, and exponentiation scratch can coexist.
    # Four payloads plus a small header allowance give a conservative guard
    # before the potentially large `BigInt` is constructed.
    bit_count=BigInt(order-1)*BigInt(ndigits(big(multiplicity);base=2))+1
    4cld(bit_count,BigInt(8))+4096
end

function _density_power_peak_bytes(
        workspace::DensityPowerWorkspace,order::Int)
    scale_peak=maximum((
        _density_power_scale_peak_bytes(
            scale.numerator,order)
        for scale in workspace.multiplicity_scales);init=big(0))
    workspace.retained_bytes+scale_peak
end

function _density_power_requested_peak_bytes(
        basis::PIBasis,::Type{R},order::Int,power_workspace) where
        R<:AbstractFloat
    if power_workspace isa DensityPowerWorkspace
        return order<=2 ? power_workspace.retained_bytes :
                          _density_power_peak_bytes(power_workspace,order)
    end
    order<=2&&return big(0)
    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    _density_power_workspace_storage_bound(basis,R,precision_bits)+
        _density_power_scale_peak_bound(basis,order)
end

function _density_intertwiner_conversion_bytes(
        U::AbstractMatrix,::Type{CT},precision_bits::Int) where CT
    U isa Matrix{CT}&&return big(0)
    _performance_entries_bytes(
        length(U),CT;bigfloat_precision=precision_bits)+512
end

function _density_intertwiner_conversion_bytes(
        U::_PackedLRIntertwiner,::Type{CT},precision_bits::Int) where CT
    U isa _PackedLRIntertwiner{CT}&&return big(0)
    scalar_bytes=_scalar_retained_bytes(
        CT;bigfloat_precision=precision_bits)
    index_bytes=BigInt(sizeof(Int))
    sum(U.blocks;init=big(0)) do block
        BigInt(nnz(block))*(scalar_bytes+index_bytes)+
            BigInt(size(block,2)+1)*index_bytes+512
    end
end

function _density_unprepared_reduction_extra_bytes(
        plan::ReductionPlan,::Type{R}) where R<:AbstractFloat
    (plan.k==0||plan.k==plan.basis.N)&&return big(0)
    CT=Complex{R}
    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    output_entries=BigInt(length(plan.output_basis))
    maximum_output=maximum(length,plan.output_basis.patterns;init=1)
    maximum_parent=1
    maximum_tmp=big(0)
    maximum_conversion=big(0)
    for coupling in plan.couplings
        for (_,intertwiners) in coupling.intertwiners,U in intertwiners
            parent=size(U,2)
            maximum_parent=max(maximum_parent,parent)
            maximum_tmp=max(
                maximum_tmp,BigInt(coupling.da)*BigInt(parent))
            maximum_conversion=max(
                maximum_conversion,
                _density_intertwiner_conversion_bytes(
                    U,CT,precision_bits))
        end
    end
    # `_plan_reduced_state` owns the output and accumulated reduced blocks.
    # Its largest live coupling also owns a scaled/converted parent pair, one
    # rectangular multiplication buffer, one possible recoupler conversion,
    # and Hermitianization scratch. Count all of them before entering that
    # allocation route.
    entries=2output_entries+2BigInt(maximum_parent)^2+
        maximum_tmp+2BigInt(maximum_output)^2
    _performance_entries_bytes(
        entries,CT;bigfloat_precision=precision_bits)+
        maximum_conversion+
        BigInt(length(plan.output_basis.sectors))*512+4096
end

function _reduced_density_power_preflight(
        plan::ReductionPlan,workspace,power_workspace,
        ::Type{R},order::Int,memory_budget) where R<:AbstractFloat
    _performance_memory_limit(memory_budget)===nothing&&return big(0)
    retained=BigInt(Base.summarysize(
        workspace isa ReductionWorkspace ? workspace : plan))
    reduction_extra=workspace===nothing&&order>=2 ?
        _density_unprepared_reduction_extra_bytes(plan,R) : big(0)
    power_peak=_density_power_requested_peak_bytes(
        plan.output_basis,R,order,power_workspace)
    peak=retained+reduction_extra+power_peak
    _require_performance_budget(
        "reduced density-matrix power trace",peak,memory_budget;guidance=
        "Reuse a reduction-capable workspace and DensityPowerWorkspace, reduce the retained sectors, or raise memory_budget explicitly.")
    peak
end

function _density_power_scale(
        ::Type{R},multiplicity::Integer,order::Int) where R<:AbstractFloat
    order>=3||error("internal density-power scale requires order at least three")
    multiplicity==1&&return _prepare_exact_scale(
        R,one(BigInt),one(BigInt),Val(false);
        context="density-moment multiplicity scale")
    denominator=big(multiplicity)^(order-1)
    _prepare_exact_scale(
        R,one(BigInt),denominator,Val(false);
        context="density-moment multiplicity scale")
end

function _fill_density_weighted_block!(
        workspace::DensityPowerWorkspace,source::AbstractMatrix,
        sector_index::Int,destination_index::Int=1)
    n=size(source,1)
    destination=_density_power_buffer(workspace,destination_index,n)
    scale=workspace.multiplicity_scales[sector_index]
    if scale.direct
        @inbounds for index in eachindex(destination,source)
            destination[index]=source[index]*scale.factor
        end
        _ordinary_scaled_value_safe(destination,source)&&return destination
    end
    @inbounds for index in eachindex(destination,source)
        destination[index]=_apply_prepared_exact_scale(
            source[index],scale;
            context="multiplicity-weighted density block")
    end
    destination
end

function _density_power_real(value,::Type{R};atol::Real,rtol::Real,
        label::AbstractString) where R<:AbstractFloat
    real_value=real(value)
    imaginary_value=imag(value)
    isfinite(real_value)&&isfinite(imaginary_value)||throw(ArgumentError(
        "$label is nonfinite in $R; use a wider scalar type"))
    tolerance=atol+rtol*max(abs(real_value),one(R))
    abs(imaginary_value)<=tolerance||throw(ArgumentError(
        "$label has an imaginary residual $imaginary_value exceeding tolerance $tolerance"))
    real_value
end

function _density_power_raw(
        source::AbstractMatrix,sector_index::Int,order::Int,
        workspace::DensityPowerWorkspace,
        ::Type{R};atol::Real,rtol::Real) where R<:AbstractFloat
    half=order>>1
    weighted=_fill_density_weighted_block!(
        workspace,source,sector_index,1)
    powered,result_index=_density_matrix_power!(workspace,weighted,half)
    if all(iszero,powered)&&!all(iszero,source)
        throw(ArgumentError(
            "density-block power underflows in $R; use a wider scalar type"))
    end
    raw = if iseven(order)
        real(dot(powered,powered))
    else
        weighted_index = if half==1
            result_index
        else
            index=result_index==1 ? 2 : 1
            _fill_density_weighted_block!(
                workspace,source,sector_index,index)
            index
        end
        weighted=_density_power_buffer(
            workspace,weighted_index,size(source,1))
        product_index = half==1 ? (result_index==1 ? 2 : 1) :
                                  6-result_index-weighted_index
        product=_density_power_buffer(
            workspace,product_index,size(source,1))
        mul!(product,weighted,powered)
        _density_power_real(dot(powered,product),R;atol,rtol,
                            label="density-block power trace")
    end
    isfinite(raw)||throw(ArgumentError(
        "density-block power trace is nonfinite in $R; use a wider scalar type"))
    if iszero(raw)&&!all(iszero,source)
        throw(ArgumentError(
            "density-block power trace underflows in $R; use a wider scalar type"))
    end
    raw
end

function _trace_power_from_blocks(
        blocks,basis::PIBasis,order::Int,
        workspace::DensityPowerWorkspace{R};atol::Real,rtol::Real,
        memory_budget) where R<:AbstractFloat
    _check_density_power_workspace(workspace,basis,R)
    peak=_density_power_peak_bytes(workspace,order)
    _require_performance_budget(
        "density-matrix power trace",peak,memory_budget;guidance=
        "Use a wider budget only after checking the largest Schur block and requested power.")
    total_parts=(zero(R),0)
    any_nonzero=false
    for (sector_index,block) in enumerate(blocks)
        all(iszero,block)&&continue
        any_nonzero=true
        raw=_density_power_raw(
            block,sector_index,order,workspace,R;atol,rtol)
        scale=_density_power_scale(
            R,workspace.multiplicity_scales[sector_index].numerator,order)
        sector_parts=_scaled_real_product_parts(
            raw,one(R),scale;
            context="density-moment sector contribution")
        total_parts=_combine_binary_parts(total_parts,sector_parts)
    end
    result=any_nonzero ? _checked_binary_parts_value(
        total_parts,"density-matrix power trace") : zero(R)
    isfinite(result)&&(!iszero(result)||!any_nonzero)&&return result
    throw(ArgumentError(
        "density-matrix power trace is outside the nonzero finite range of $R; use a wider scalar type"))
end

function _density_trace_real(rho::PIState,::Type{R};atol::Real,
        rtol::Real) where R<:AbstractFloat
    value=trace(rho)
    result=_density_power_real(value,R;atol,rtol,
                               label="density-matrix trace")
    converted=try
        R(result)
    catch
        throw(ArgumentError(
            "density-matrix trace is not representable in $R; use a wider scalar type"))
    end
    isfinite(converted)||throw(ArgumentError(
        "density-matrix trace is nonfinite in $R"))
    !iszero(result)&&iszero(converted)&&throw(ArgumentError(
        "density-matrix trace underflows in $R; use a wider scalar type"))
    converted
end

"""
    trace_power(rho, order; workspace=nothing, check=true,
                atol=_analysis_atol(rho), rtol=_state_rtol(rho),
                memory_budget=512*1024^2)

Return `tr(rho^order)` for a positive integer `order`, directly in Schur
coordinates.  If `C_nu` is the stored coefficient block and `f^nu` its exact
symmetric-group multiplicity, the evaluated formula is

`tr(rho^q) = sum_nu (f^nu)^(1-q/2) tr(C_nu^q)`.

`order=1` uses the physical trace and `order=2` uses the allocation-light
[`purity`](@ref) contraction.  Higher orders use binary block powering and
exact multiplicity scaling. For stability they power
`M_nu=sqrt(f^nu)*C_nu` and apply the equivalent factor `f^(1-order)` only to
the resulting sector trace. Reuse a [`DensityPowerWorkspace`](@ref) for
repeated calls. No `d^N` state is constructed.

By default `rho` is validated as a density operator.  Set `check=false` only
when an enclosing workflow has already validated it.  This does not disable
finite-range or workspace-ownership checks.
"""
function trace_power(
        rho::PIState,order;
        workspace=nothing,check::Bool=true,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    q=_density_power_order(order)
    _check_reduction_tolerances(atol,rtol)
    _performance_memory_limit(memory_budget)
    R=_real_float_type(eltype(rho.data))
    workspace===nothing||_check_density_power_workspace(workspace,rho.basis,R)
    if workspace isa DensityPowerWorkspace{BigFloat}
        bounds=_reduction_state_precision_bounds(rho)
        bounds==(workspace.precision_bits,workspace.precision_bits)||
            throw(ArgumentError(
                "source state BigFloat storage has precision range $bounds, but DensityPowerWorkspace requires $(workspace.precision_bits) bits"))
    end
    if workspace isa DensityPowerWorkspace{BigFloat}&&
            (precision(BigFloat)!=workspace.precision_bits||
             rounding(BigFloat)!=workspace.rounding_mode)
        return _density_power_with_precision(workspace) do
            trace_power(rho,q;workspace,check,atol,rtol,memory_budget)
        end
    end
    if workspace===nothing&&R===BigFloat
        bounds=_reduction_state_precision_bounds(rho)
        if precision(BigFloat)!=bounds[2]
            return setprecision(BigFloat,bounds[2]) do
                trace_power(
                    rho,q;workspace,check,atol,rtol,memory_budget)
            end
        end
    end
    check&&validate_state(rho;atol,rtol)
    if q<=2
        retained=workspace===nothing ? big(0) : workspace.retained_bytes
        _require_performance_budget(
            "density-matrix power trace",retained,memory_budget;guidance=
            "Omit the unused workspace for the dedicated q=1 or q=2 path.")
        q==1&&return _density_trace_real(rho,R;atol,rtol)
        return purity(rho)
    end
    work=workspace===nothing ?
        DensityPowerWorkspace(rho;memory_budget) : workspace
    blocks=(coefficient_block(rho,sector) for sector in rho.basis.sectors)
    _trace_power_from_blocks(
        blocks,rho.basis,q,work;atol,rtol,memory_budget)
end

function _reduced_trace_power_target_type(
        rho::PIState,k::Int,plan::ReductionPlan,workspace)
    k==rho.basis.N&&return _real_float_type(eltype(rho.data))
    workspace isa ReductionWorkspace&&return _real_float_type(workspace.Ttype)
    k==0&&return _real_float_type(eltype(rho.data))
    _reduction_source_plan_context(plan,rho).WorkR
end

function _check_reduced_power_precision_context(
        rho::PIState,plan,workspace,power_workspace)
    power_workspace isa DensityPowerWorkspace{BigFloat}||return nothing
    power_bits=power_workspace.precision_bits
    if workspace isa ReductionWorkspace&&
            _real_float_type(workspace.Ttype)===BigFloat
        workspace.precision_bits==power_bits&&
            workspace.rounding_mode==power_workspace.rounding_mode||
            throw(ArgumentError(
                "ReductionWorkspace and DensityPowerWorkspace use different BigFloat contexts"))
    elseif workspace===nothing
        required=_reduction_unprepared_precision(rho,plan)
        required===nothing||required==power_bits||throw(ArgumentError(
            "DensityPowerWorkspace precision $power_bits does not match the reduced-state precision $required"))
    end
    nothing
end

"""
    reduced_trace_power(rho, k, order; plan=nothing, workspace=nothing,
                        power_workspace=nothing, check=true,
                        atol=_analysis_atol(rho), rtol=_state_rtol(rho),
                        memory_budget=512*1024^2)

Return `tr(rho_k^order)` for the `k`-particle reduction and a positive integer
`order`.  `order=2` delegates to the optimized [`reduced_purity`](@ref) path.
With a [`ReductionWorkspace`](@ref), higher moments are evaluated directly
from its accumulated reduced Schur blocks without constructing a
[`PIState`](@ref).  `power_workspace` supplies the independent task-owned
square buffers used for block powering.

The endpoints obey `tr(rho_0^q)=1` and
`tr(rho_N^q)=tr(rho^q)`.  Supplied plans and workspaces are still checked at
the endpoints, so a shortcut cannot hide an ownership or precision error.
Set `check=false` only after validating the parent state.
"""
function reduced_trace_power(
        rho::PIState,k::Integer,order;
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho),
        plan=nothing,workspace=nothing,power_workspace=nothing,
        check::Bool=true,memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    b=rho.basis
    k isa Bool&&throw(ArgumentError("subsystem size k must be an integer"))
    0<=k<=b.N||throw(ArgumentError(
        "subsystem size k must satisfy 0 ≤ k ≤ N"))
    ki=Int(k)
    q=_density_power_order(order)
    _check_reduction_tolerances(atol,rtol)
    _performance_memory_limit(memory_budget)
    workspace===nothing||workspace isa ReductionWorkspace||throw(ArgumentError(
        "workspace must be a ReductionWorkspace"))
    power_workspace===nothing||power_workspace isa DensityPowerWorkspace||
        throw(ArgumentError(
            "power_workspace must be a DensityPowerWorkspace"))
    _check_reduced_power_precision_context(
        rho,plan,workspace,power_workspace)
    if workspace isa ReductionWorkspace&&
            _real_float_type(workspace.Ttype)===BigFloat&&
            (precision(BigFloat)!=workspace.precision_bits||
             rounding(BigFloat)!=workspace.rounding_mode)
        return _reduction_with_precision(workspace) do
            reduced_trace_power(
                rho,ki,q;atol,rtol,plan,workspace,power_workspace,check,
                memory_budget)
        end
    end
    if power_workspace isa DensityPowerWorkspace{BigFloat}&&
            (precision(BigFloat)!=power_workspace.precision_bits||
             rounding(BigFloat)!=power_workspace.rounding_mode)
        return _density_power_with_precision(power_workspace) do
            reduced_trace_power(
                rho,ki,q;atol,rtol,plan,workspace,power_workspace,check,
                memory_budget)
        end
    end
    if workspace===nothing
        precision_bits=_reduction_unprepared_precision(rho,plan)
        if precision_bits!==nothing&&precision(BigFloat)!=precision_bits
            return setprecision(BigFloat,precision_bits) do
                reduced_trace_power(
                    rho,ki,q;atol,rtol,plan,workspace,power_workspace,check,
                    memory_budget)
            end
        end
    end
    check&&validate_state(rho;atol,rtol)

    if plan===nothing&&workspace===nothing&&power_workspace===nothing
        q==1&&return _density_trace_real(
            rho,_real_float_type(eltype(rho.data));atol,rtol)
        ki==0&&return one(_real_float_type(eltype(rho.data)))
        ki==b.N&&return trace_power(
            rho,q;check=false,atol,rtol,memory_budget)
    end
    plan,workspace=_resolve_reduction_resources(
        b,ki,plan,workspace;atol)
    workspace===nothing||_check_reduction_workspace(workspace,plan,rho)
    targetR=_reduced_trace_power_target_type(rho,ki,plan,workspace)
    power_workspace===nothing||
        _check_density_power_workspace(
            power_workspace,plan.output_basis,targetR)
    _reduced_density_power_preflight(
        plan,workspace,power_workspace,targetR,q,memory_budget)

    if q==1
        return _density_trace_real(rho,targetR;atol,rtol)
    elseif q==2
        return reduced_purity(
            rho,ki;atol,rtol,plan,workspace,check=false)
    elseif ki==0
        return one(targetR)
    elseif ki==b.N
        return trace_power(
            rho,q;workspace=power_workspace,check=false,atol,rtol,
            memory_budget)
    end

    if workspace===nothing
        reduced=_plan_reduced_state(rho,plan;atol,rtol)
        return trace_power(
            reduced,q;workspace=power_workspace,check=false,atol,rtol,
            memory_budget)
    end
    blocks=_accumulate_reduced_blocks!(rho,plan,workspace;atol,rtol)
    work=power_workspace===nothing ? DensityPowerWorkspace(
        plan.output_basis;T=targetR,memory_budget) : power_workspace
    _trace_power_from_blocks(
        blocks,plan.output_basis,q,work;atol,rtol,memory_budget)
end

function _reduced_density_power_batch_preflight(
        rho::PIState,order::Int,kvals,plans,workspaces,power_workspaces,
        memory_budget)
    _performance_memory_limit(memory_budget)===nothing&&return nothing
    retained=big(0)
    maximum_extra=big(0)
    for (k,prepared,reduction_work,power_work) in
            zip(kvals,plans,workspaces,power_workspaces)
        reduction_work===nothing||reduction_work isa ReductionWorkspace||
            throw(ArgumentError(
                "workspace entries must be ReductionWorkspace objects or nothing"))
        power_work===nothing||power_work isa DensityPowerWorkspace||
            throw(ArgumentError(
                "power_workspace entries must be DensityPowerWorkspace objects or nothing"))
        active_plan=reduction_work isa ReductionWorkspace ?
            reduction_work.plan : prepared
        if reduction_work isa ReductionWorkspace
            retained+=BigInt(Base.summarysize(reduction_work))
        elseif active_plan isa ReductionPlan
            retained+=BigInt(Base.summarysize(active_plan))
        end
        power_work isa DensityPowerWorkspace&&
            (retained+=power_work.retained_bytes)
        supplied_power_extra=power_work isa DensityPowerWorkspace&&order>2 ?
            _density_power_peak_bytes(power_work,order)-
                power_work.retained_bytes : big(0)
        if !(active_plan isa ReductionPlan)
            maximum_extra=max(maximum_extra,supplied_power_extra)
            continue
        end
        targetR=_reduced_trace_power_target_type(
            rho,Int(k),active_plan,reduction_work)
        reduction_extra=reduction_work===nothing&&order>=2 ?
            _density_unprepared_reduction_extra_bytes(
                active_plan,targetR) : big(0)
        power_extra = if power_work isa DensityPowerWorkspace
            supplied_power_extra
        else
            _density_power_requested_peak_bytes(
                active_plan.output_basis,targetR,order,nothing)
        end
        maximum_extra=max(
            maximum_extra,reduction_extra+power_extra)
    end
    _require_performance_budget(
        "reduced density-matrix power trace batch",
        retained+maximum_extra,memory_budget;guidance=
        "Retain fewer prepared subsystem workspaces at once, split the batch, or raise memory_budget explicitly.")
    nothing
end

"""
    reduced_trace_powers(rho, order; ks=0:N, plans=nothing,
                         workspaces=nothing, power_workspaces=nothing,
                         check=true, kwargs...)
    reduced_trace_powers(rho, plan_set, order; workspace=nothing,
                         power_workspaces=nothing, check=true, kwargs...)

Evaluate the fixed positive integer `order` for several reduced subsystem
sizes, returning results in the requested order.  The parent state is
validated once.  The first form accepts optional plan, reduction-workspace,
and density-power-workspace collections aligned with `ks`.  The prepared
[`ReductionPlanSet`](@ref) form uses its stored ordering and accepts a matching
[`ReductionWorkspaceSet`](@ref).

For repeated state families, prepare both workspace collections once.  Each
workspace remains task owned; no hidden buffer is grown during evaluation.
"""
function reduced_trace_powers(
        rho::PIState,order;ks=0:rho.basis.N,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho),
        plans=nothing,workspaces=nothing,power_workspaces=nothing,
        check::Bool=true,memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    q=_density_power_order(order)
    kvals=collect(ks)
    for k in kvals
        k isa Integer&&!(k isa Bool)||throw(ArgumentError(
            "subsystem sizes must be integers"))
        0<=k<=rho.basis.N||throw(ArgumentError(
            "subsystem size k must satisfy 0 ≤ k ≤ N"))
    end
    _check_reduction_tolerances(atol,rtol)
    _performance_memory_limit(memory_budget)
    state_bounds=_reduction_state_precision_bounds(rho)
    if state_bounds!==nothing&&precision(BigFloat)!=state_bounds[2]
        return setprecision(BigFloat,state_bounds[2]) do
            reduced_trace_powers(
                rho,q;ks=kvals,atol,rtol,plans,workspaces,power_workspaces,
                check,memory_budget)
        end
    end
    check&&validate_state(rho;atol,rtol)
    if plans===nothing&&workspaces===nothing&&power_workspaces===nothing
        if q==1
            value=_density_trace_real(
                rho,_real_float_type(eltype(rho.data));atol,rtol)
            return fill(value,length(kvals))
        end
    end
    count=length(kvals)
    ps=plans===nothing ? fill(nothing,count) : collect(plans)
    ws=workspaces===nothing ? fill(nothing,count) : collect(workspaces)
    pws=power_workspaces===nothing ? fill(nothing,count) :
                                    collect(power_workspaces)
    length(ps)==count||throw(DimensionMismatch(
        "one ReductionPlan is required per subsystem size"))
    length(ws)==count||throw(DimensionMismatch(
        "one ReductionWorkspace entry is required per subsystem size"))
    length(pws)==count||throw(DimensionMismatch(
        "one DensityPowerWorkspace entry is required per subsystem size"))
    _reduced_density_power_batch_preflight(
        rho,q,kvals,ps,ws,pws,memory_budget)
    map(kvals,ps,ws,pws) do k,prepared,reduction_work,power_work
        reduced_trace_power(
            rho,k,q;atol,rtol,plan=prepared,workspace=reduction_work,
            power_workspace=power_work,check=false,memory_budget)
    end
end

function reduced_trace_powers(
        rho::PIState,set::ReductionPlanSet,order;
        workspace=nothing,power_workspaces=nothing,check::Bool=true,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    q=_density_power_order(order)
    _check_reduction_tolerances(atol,rtol)
    _performance_memory_limit(memory_budget)
    _check_reduction_plan_set(set)
    rho.basis===set.basis||throw(ArgumentError(
        "state and ReductionPlanSet use different PIBasis objects"))
    state_bounds=_reduction_state_precision_bounds(rho)
    if state_bounds!==nothing&&precision(BigFloat)!=state_bounds[2]
        return setprecision(BigFloat,state_bounds[2]) do
            reduced_trace_powers(
                rho,set,q;workspace,power_workspaces,check,atol,rtol,
                memory_budget)
        end
    end
    check&&validate_state(rho;atol,rtol)
    works=workspace===nothing ? ntuple(_->nothing,length(set.plans)) :
        _check_reduction_workspace_set(workspace,set,rho).workspaces
    power_works=power_workspaces===nothing ?
        ntuple(_->nothing,length(set.plans)) : Tuple(power_workspaces)
    length(power_works)==length(set.plans)||throw(DimensionMismatch(
        "one DensityPowerWorkspace entry is required per prepared subsystem size"))
    _reduced_density_power_batch_preflight(
        rho,q,set.ks,set.plans,works,power_works,memory_budget)
    map(set.plans,works,power_works) do plan,reduction_work,power_work
        reduced_trace_power(
            rho,plan.k,q;plan,workspace=reduction_work,
            power_workspace=power_work,check=false,atol,rtol,memory_budget)
    end
end
