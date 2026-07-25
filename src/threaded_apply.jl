# Deterministic task-parallel application of a prepared PI Liouvillian.
#
# Every spawned task owns complete output Schur sectors.  Source blocks and
# prepared kernel data are read-only during the parallel phase, so no atomic
# numerical updates or task-order-dependent reductions are required. Dynamic operator
# schedules and scalar rates are evaluated once, on the calling task, before
# any worker is launched.

struct _ThreadedSectorScratch{T}
    input::Matrix{T}
    left::Matrix{T}
    right::Matrix{T}
    rectangular::Matrix{T}
end

function _ThreadedSectorScratch(::Type{T},largest::Int) where T
    _ThreadedSectorScratch(zeros(T,largest,largest),
                           zeros(T,largest,largest),
                           zeros(T,largest,largest),
                           zeros(T,largest,largest))
end

"""
    ThreadedLiouvillianWorkspace(plan_or_compiled; tasks=Threads.nthreads())

Caller-owned scratch and deterministic Schur-sector partition for
[`threaded_apply!`](@ref) and [`threaded_apply_adjoint!`](@ref).

The prepared plan, supplied directly or through a `CompiledPIModel`, is
read-only. Dynamic schedules are evaluated exactly
once in `preparation`; each worker then reads those values and owns disjoint
output sectors together with one private matrix scratch. `tasks` is capped by
the number of retained Schur sectors and must be positive. The workspace may
be reused sequentially but must not be used concurrently.

This API does not change BLAS's global thread count. For reproducible
benchmarking, choose the BLAS configuration before constructing the
workspace and compare it separately with Julia task parallelism.
"""
struct ThreadedLiouvillianWorkspace{P,W,S,A,B}
    plan::P
    preparation::W
    workers::S
    assignments::A
    busy::B
end

_threaded_kernel_supported(kernel)=false
_threaded_kernel_supported(::Union{
    HamiltonianPIKernel,DissipatorPIKernel,LocalJumpPIKernel,
    FactorizedLocalJumpPIKernel,FactorizedLocalPBodyJumpPIKernel,
    FusedStaticPIKernel,InPlaceHamiltonianPIKernel,
    InPlaceDissipatorPIKernel,InPlaceLocalJumpPIKernel,
    InPlaceLocalPBodyJumpPIKernel,
    InPlaceCorrelatedCollectiveJumpPIKernel,
    InPlaceCorrelatedLocalJumpPIKernel})=true

function _threaded_sector_costs(basis)
    # A dense Schur-block sandwich is cubic in the irrep dimension.  The
    # strictly positive linear term gives empty/one-dimensional sectors a
    # meaningful balancing weight. BigInt keeps setup accounting exact even
    # when a user inspects a basis whose cubic estimate exceeds machine Int.
    [begin
         dimension=length(basis.patterns[sector])
         BigInt(dimension)^3+dimension
     end for sector in eachindex(basis.sectors)]
end

function _threaded_sector_assignments(basis,tasks::Int)
    number_sectors=length(basis.sectors)
    number_workers=min(tasks,max(number_sectors,1))
    assignments=[Int[] for _ in 1:number_workers]
    loads=zeros(BigInt,number_workers)
    costs=_threaded_sector_costs(basis)
    # Largest-processing-time scheduling has deterministic tie breaking.
    order=sortperm(eachindex(costs);by=sector->(-costs[sector],sector))
    for sector in order
        worker=findmin(loads)[2]
        push!(assignments[worker],sector)
        loads[worker]+=costs[sector]
    end
    foreach(sort!,assignments)
    assignments
end

function ThreadedLiouvillianWorkspace(plan::LiouvillianPlan;
                                      tasks::Integer=Threads.nthreads())
    tasks>0||throw(ArgumentError("tasks must be positive"))
    plan.kernels===nothing&&throw(ArgumentError(
        "threaded Schur-sector application requires a prepared PI kernel plan; " *
        "wrap callable operators in InPlaceTimeOperator before compiling"))
    if !all(_threaded_kernel_supported,plan.kernels)
        unsupported=[typeof(kernel) for kernel in plan.kernels
                     if !_threaded_kernel_supported(kernel)]
        throw(ArgumentError(
            "threaded Schur-sector application does not support prepared " *
            "kernel types $unsupported; extend the target-sector kernel " *
            "contract before enabling threading"))
    end
    assignments=_threaded_sector_assignments(plan.basis,Int(tasks))
    largest=maximum(length,plan.basis.patterns;init=1)
    workers=[_ThreadedSectorScratch(plan.Ttype,largest)
             for _ in eachindex(assignments)]
    ThreadedLiouvillianWorkspace(plan,LiouvillianWorkspace(plan),workers,
                                 assignments,Threads.Atomic{Int}(0))
end

ThreadedLiouvillianWorkspace(compiled::CompiledPIModel;kwargs...)=
    ThreadedLiouvillianWorkspace(compiled.plan;kwargs...)

function _check_threaded_workspace(work::ThreadedLiouvillianWorkspace,
                                   plan::LiouvillianPlan)
    work.plan===plan||throw(ArgumentError(
        "threaded Liouvillian workspace belongs to a different plan"))
    _check_liouvillian_workspace(work.preparation,plan)
    work
end

@inline function _threaded_source_block(source,basis,sector::Int)
    dimension=length(basis.patterns[sector])
    offset=basis.offsets[sector]
    reshape(view(source,offset:(offset+dimension*dimension-1)),
            dimension,dimension)
end

@inline function _threaded_destination_block(destination,basis,sector::Int)
    dimension=length(basis.patterns[sector])
    offset=basis.offsets[sector]
    reshape(view(destination,offset:(offset+dimension*dimension-1)),
            dimension,dimension)
end

@inline function _threaded_copy_source!(scratch,source,basis,sector::Int)
    source_block=_threaded_source_block(source,basis,sector)
    dimension=size(source_block,1)
    input=@view scratch.input[1:dimension,1:dimension]
    copyto!(input,source_block)
    input
end

# Multi-worker application packs the full source once on the calling task.
# Workers then share those immutable matrices while retaining private
# left/right/rectangular scratch and disjoint destination sectors.
struct _ThreadedPreparedSource{V,B}
    vector::V
    blocks::B
end
Base.getindex(source::_ThreadedPreparedSource,index::Int)=
    source.vector[index]
@inline _threaded_copy_source!(scratch,source::_ThreadedPreparedSource,
        basis,sector::Int)=source.blocks[sector][3]

@inline _threaded_forward_scale(kernel::HamiltonianPIKernel,::Type{T},t,p) where T =
    convert(T,value_at(kernel.scale,t,p))
@inline _threaded_forward_scale(kernel::DissipatorPIKernel,::Type{T},t,p) where T =
    convert(T,_evaluated_dissipative_rate(kernel.scale,t,p))
@inline _threaded_forward_scale(kernel::LocalJumpPIKernel,::Type{T},t,p) where T =
    convert(T,_evaluated_dissipative_rate(kernel.scale,t,p))
@inline _threaded_forward_scale(kernel::FactorizedLocalJumpPIKernel,::Type{T},t,p) where T =
    convert(T,_evaluated_dissipative_rate(kernel.scale,t,p))
@inline _threaded_forward_scale(kernel::FactorizedLocalPBodyJumpPIKernel,::Type{T},t,p) where T =
    convert(T,_evaluated_dissipative_rate(kernel.scale,t,p))
@inline _threaded_forward_scale(::FusedStaticPIKernel,::Type{T},t,p) where T = one(T)
@inline _threaded_forward_scale(kernel::AbstractDynamicPIKernel,::Type{T},t,p) where T =
    kernel isa InPlaceHamiltonianPIKernel ? convert(T,value_at(kernel.scale,t,p)) :
    convert(T,_evaluated_dissipative_rate(kernel.scale,t,p))

@inline _threaded_adjoint_scale(kernel,T,t,p)=
    conj(_threaded_forward_scale(kernel,T,t,p))

_threaded_scales(::Tuple{},T,t,p,adjoint)=()
function _threaded_scales(kernels::Tuple{K,Vararg{Any}},T,t,p,adjoint) where K
    kernel=first(kernels)
    scale=adjoint ? _threaded_adjoint_scale(kernel,T,t,p) :
                    _threaded_forward_scale(kernel,T,t,p)
    (scale,_threaded_scales(Base.tail(kernels),T,t,p,adjoint)...)
end

@inline function _threaded_add_block!(destination,source,scale)
    @inbounds for index in eachindex(destination,source)
        destination[index]+=scale*source[index]
    end
    destination
end

function _threaded_hamiltonian_sector!(destination,source,block,scale,scratch;
                                       adjoint::Bool=false)
    dimension=size(source,1)
    left=@view scratch.left[1:dimension,1:dimension]
    right=@view scratch.right[1:dimension,1:dimension]
    if adjoint
        effective=LinearAlgebra.adjoint(block)
        mul!(left,effective,source)
        mul!(right,source,effective)
        @inbounds for index in eachindex(destination,left,right)
            destination[index]+=(1im*scale)*(left[index]-right[index])
        end
    else
        mul!(left,block,source)
        mul!(right,source,block)
        @inbounds for index in eachindex(destination,left,right)
            destination[index]+=(-1im*scale)*(left[index]-right[index])
        end
    end
    destination
end

function _threaded_dissipator_sector!(destination,source,block,qblock,scale,
                                      scratch;adjoint::Bool=false)
    dimension=size(source,1)
    left=@view scratch.left[1:dimension,1:dimension]
    right=@view scratch.right[1:dimension,1:dimension]
    if adjoint
        mul!(left,LinearAlgebra.adjoint(block),source)
        mul!(right,left,block)
        _threaded_add_block!(destination,right,scale)
        effective_q=LinearAlgebra.adjoint(qblock)
        mul!(left,effective_q,source)
        mul!(right,source,effective_q)
    else
        mul!(left,block,source)
        mul!(right,left,LinearAlgebra.adjoint(block))
        _threaded_add_block!(destination,right,scale)
        mul!(left,qblock,source)
        mul!(right,source,qblock)
    end
    half=-scale/2
    @inbounds for index in eachindex(destination,left,right)
        destination[index]+=half*(left[index]+right[index])
    end
    destination
end

function _threaded_anticommutator_sector!(destination,source,qblock,scale,
                                          scratch;adjoint::Bool=false)
    dimension=size(source,1)
    left=@view scratch.left[1:dimension,1:dimension]
    right=@view scratch.right[1:dimension,1:dimension]
    effective_q=adjoint ? LinearAlgebra.adjoint(qblock) : qblock
    mul!(left,effective_q,source)
    mul!(right,source,effective_q)
    half=-scale/2
    @inbounds for index in eachindex(destination,left,right)
        destination[index]+=half*(left[index]+right[index])
    end
    destination
end

function _threaded_rectangular_sandwich!(destination,source,operator,scale,
                                         scratch;adjoint::Bool=false)
    output_dimension=size(destination,1)
    input_dimension=size(source,1)
    rectangular=@view scratch.rectangular[1:output_dimension,1:input_dimension]
    output=@view scratch.left[1:output_dimension,1:output_dimension]
    effective=adjoint ? LinearAlgebra.adjoint(operator) : operator
    _rectangular_sandwich!(output,rectangular,effective,source)
    _threaded_add_block!(destination,output,scale)
end

function _threaded_rectangular_sandwich!(destination,source,
        contraction::_StaticOneBodyContraction,scale,scratch;
        adjoint::Bool=false)
    if !contraction.use_support
        return _threaded_rectangular_sandwich!(destination,source,
            contraction.matrix,scale,scratch;adjoint)
    end
    rows=contraction.output_rows;columns=contraction.input_columns
    values=contraction.values
    if adjoint
        @inbounds for right in eachindex(values)
            output_column=columns[right]
            input_column=rows[right]
            right_value=values[right]
            for left in eachindex(values)
                destination[columns[left],output_column]+=
                    scale*conj(values[left])*right_value*
                    source[rows[left],input_column]
            end
        end
    else
        @inbounds for right in eachindex(values)
            output_column=rows[right]
            input_column=columns[right]
            right_value=values[right]
            for left in eachindex(values)
                destination[rows[left],output_column]+=
                    scale*values[left]*conj(right_value)*
                    source[columns[left],input_column]
            end
        end
    end
    destination
end

function _threaded_apply_sector!(destination,source,
        kernel::HamiltonianPIKernel,::Nothing,basis,scale,scratch,sector;
        adjoint::Bool=false)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_hamiltonian_sector!(output,input,kernel.blocks[sector],scale,
                                  scratch;adjoint)
end

function _threaded_apply_sector!(destination,source,
        kernel::DissipatorPIKernel,::Nothing,basis,scale,scratch,sector;
        adjoint::Bool=false)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_dissipator_sector!(output,input,kernel.blocks[sector],
        kernel.qblocks[sector],scale,scratch;adjoint)
end

# Compatibility for the pre-factorized static gain representation.  The
# destination ownership test is coordinate-local and therefore race free.
# Prepared fixed-gain kernels specialize this below without scanning unrelated
# triplets.
function _threaded_apply_sector!(destination,source,
        kernel::LocalJumpPIKernel,::Nothing,basis,scale,scratch,sector;
        adjoint::Bool=false)
    offset=basis.offsets[sector]
    stop=basis.offsets[sector+1]-1
    if adjoint
        @inbounds for index in eachindex(kernel.gain.V)
            target=kernel.gain.J[index]
            offset<=target<=stop||continue
            destination[target]+=scale*conj(kernel.gain.V[index])*
                                 source[kernel.gain.I[index]]
        end
    else
        @inbounds for index in eachindex(kernel.gain.V)
            target=kernel.gain.I[index]
            offset<=target<=stop||continue
            destination[target]+=scale*kernel.gain.V[index]*
                                 source[kernel.gain.J[index]]
        end
    end
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_anticommutator_sector!(output,input,kernel.qblocks[sector],
                                     scale,scratch;adjoint)
end

function _threaded_apply_sector!(destination,source,
        kernel::InPlaceHamiltonianPIKernel,
        prepared::InPlaceHamiltonianKernelWorkspace,basis,scale,scratch,sector;
        adjoint::Bool=false)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_hamiltonian_sector!(output,input,prepared.blocks[sector],scale,
                                  scratch;adjoint)
end

function _threaded_apply_sector!(destination,source,
        kernel::InPlaceDissipatorPIKernel,
        prepared::InPlaceDissipatorKernelWorkspace,basis,scale,scratch,sector;
        adjoint::Bool=false)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_dissipator_sector!(output,input,prepared.blocks[sector],
        prepared.qblocks[sector],scale,scratch;adjoint)
end

function _threaded_onebody_gain_sector!(destination,source,branches,
        contractions,basis,scale,scratch,sector,channels::Int;
        adjoint::Bool=false)
    branch_count=length(branches.entries)
    @inbounds for branch_index in 1:branch_count
        branch=branches.entries[branch_index]
        target=adjoint ? branch.input_sector : branch.output_sector
        target==sector||continue
        source_sector=adjoint ? branch.output_sector : branch.input_sector
        input=_threaded_copy_source!(scratch,source,basis,source_sector)
        output=_threaded_destination_block(destination,basis,target)
        branch_scale=scale*branch.scale
        for channel in 1:channels
            contraction=contractions[(channel-1)*branch_count+branch_index]
            _threaded_rectangular_sandwich!(output,input,contraction,
                branch_scale,scratch;adjoint)
        end
    end
    nothing
end

function _threaded_apply_sector!(destination,source,
        kernel::InPlaceLocalJumpPIKernel,
        prepared::InPlaceLocalJumpKernelWorkspace,basis,scale,scratch,sector;
        adjoint::Bool=false)
    _threaded_onebody_gain_sector!(destination,source,kernel.branches,
        prepared.contractions,basis,scale,scratch,sector,1;adjoint)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_anticommutator_sector!(output,input,prepared.qblocks[sector],
                                     scale,scratch;adjoint)
end

function _threaded_apply_sector!(destination,source,
        kernel::FactorizedLocalJumpPIKernel,
        ::StaticFactorizedGainKernelWorkspace,
        basis,scale,scratch,sector;adjoint::Bool=false)
    _threaded_onebody_gain_sector!(destination,source,kernel.branches,
        kernel.contractions,basis,scale,scratch,sector,1;adjoint)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_anticommutator_sector!(output,input,kernel.qblocks[sector],
                                     scale,scratch;adjoint)
end

function _threaded_apply_sector!(destination,source,
        kernel::InPlaceCorrelatedCollectiveJumpPIKernel,
        prepared::InPlaceCorrelatedCollectiveJumpKernelWorkspace,
        basis,scale,scratch,sector;adjoint::Bool=false)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    dimension=size(input,1)
    left=@view scratch.left[1:dimension,1:dimension]
    right=@view scratch.right[1:dimension,1:dimension]
    for channel in 1:prepared.rank[]
        block=prepared.effective_blocks[channel][sector]
        if adjoint
            mul!(left,LinearAlgebra.adjoint(block),input)
            mul!(right,left,block)
        else
            mul!(left,block,input)
            mul!(right,left,LinearAlgebra.adjoint(block))
        end
        _threaded_add_block!(output,right,scale)
    end
    _threaded_anticommutator_sector!(output,input,prepared.qblocks[sector],
                                     scale,scratch;adjoint)
end


function _threaded_apply_sector!(destination,source,
        kernel::InPlaceCorrelatedLocalJumpPIKernel,
        prepared::InPlaceCorrelatedLocalJumpKernelWorkspace,
        basis,scale,scratch,sector;adjoint::Bool=false)
    _threaded_onebody_gain_sector!(destination,source,kernel.branches,
        prepared.contractions,basis,scale,scratch,sector,prepared.rank[];adjoint)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_anticommutator_sector!(output,input,prepared.qblocks[sector],
                                     scale,scratch;adjoint)
end

function _threaded_pbody_gain_sector!(destination,source,kernel,prepared,
        basis,scale,scratch,sector;adjoint::Bool=false)
    @inbounds for (output_sector,input_sector,first_pair,last_pair) in kernel.groups
        target=adjoint ? input_sector : output_sector
        target==sector||continue
        source_sector=adjoint ? output_sector : input_sector
        input=_threaded_copy_source!(scratch,source,basis,source_sector)
        output=_threaded_destination_block(destination,basis,target)
        for pair in first_pair:last_pair
            contraction=prepared.contractions[pair]
            exact_scale=kernel.pair_scales[pair]
            if exact_scale.direct
                _threaded_rectangular_sandwich!(output,input,contraction,
                    scale*exact_scale.factor,scratch;adjoint)
            else
                output_dimension=size(output,1)
                input_dimension=size(input,1)
                rectangular=@view scratch.rectangular[1:output_dimension,
                                                       1:input_dimension]
                term=@view scratch.left[1:output_dimension,1:output_dimension]
                effective=adjoint ? LinearAlgebra.adjoint(contraction) : contraction
                mul!(rectangular,effective,input)
                mul!(term,rectangular,LinearAlgebra.adjoint(effective))
                for index in eachindex(output,term)
                    contribution=scale*term[index]
                    output[index]+=_apply_prepared_exact_scale(contribution,
                        exact_scale;context=adjoint ?
                        "threaded adjoint local p-body gain contribution" :
                        "threaded local p-body gain contribution")
                end
            end
        end
    end
    nothing
end

function _threaded_apply_sector!(destination,source,
        kernel::InPlaceLocalPBodyJumpPIKernel,
        prepared::InPlaceLocalPBodyJumpKernelWorkspace,
        basis,scale,scratch,sector;adjoint::Bool=false)
    _threaded_pbody_gain_sector!(destination,source,kernel,prepared,basis,
                                 scale,scratch,sector;adjoint)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_anticommutator_sector!(output,input,prepared.qblocks[sector],
                                     scale,scratch;adjoint)
end

function _threaded_apply_sector!(destination,source,
        kernel::FactorizedLocalPBodyJumpPIKernel,
        ::StaticFactorizedGainKernelWorkspace,
        basis,scale,scratch,sector;adjoint::Bool=false)
    _threaded_pbody_gain_sector!(destination,source,kernel,kernel,basis,
                                 scale,scratch,sector;adjoint)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)
    _threaded_anticommutator_sector!(output,input,kernel.qblocks[sector],
                                     scale,scratch;adjoint)
end


function _threaded_apply_sector!(destination,source,
        kernel::FusedStaticPIKernel,::StaticFactorizedGainKernelWorkspace,
        basis,scale,scratch,sector;adjoint::Bool=false)
    input=_threaded_copy_source!(scratch,source,basis,sector)
    output=_threaded_destination_block(destination,basis,sector)

    if kernel.hamiltonian_blocks!==nothing
        _threaded_hamiltonian_sector!(output,input,
            kernel.hamiltonian_blocks[sector],one(scale),scratch;adjoint)
    end

    for gain in kernel.collective_gains
        block=gain.blocks[sector]
        gain_scale=adjoint ? conj(gain.scale) : gain.scale
        dimension=size(input,1)
        left=@view scratch.left[1:dimension,1:dimension]
        right=@view scratch.right[1:dimension,1:dimension]
        if adjoint
            mul!(left,LinearAlgebra.adjoint(block),input)
            mul!(right,left,block)
        else
            mul!(left,block,input)
            mul!(right,left,LinearAlgebra.adjoint(block))
        end
        _threaded_add_block!(output,right,gain_scale)
    end

    if kernel.loss_blocks!==nothing
        _threaded_anticommutator_sector!(output,input,
            kernel.loss_blocks[sector],one(scale),scratch;adjoint)
    end

    for gain in kernel.onebody_gains
        gain_scale=adjoint ? conj(gain.scale) : gain.scale
        _threaded_onebody_gain_sector!(destination,source,gain.branches,
            gain.contractions,basis,gain_scale,scratch,sector,1;adjoint)
    end
    for gain in kernel.pbody_gains
        gain_scale=adjoint ? conj(gain.scale) : gain.scale
        _threaded_pbody_gain_sector!(destination,source,gain,gain,basis,
                                     gain_scale,scratch,sector;adjoint)
    end
    nothing
end

@inline _threaded_apply_sector_kernels!(destination,source,::Tuple{},::Tuple{},
        ::Tuple{},basis,scratch,sector;adjoint::Bool=false)=nothing

@inline function _threaded_apply_sector_kernels!(destination,source,
        kernels::Tuple{K,Vararg{Any}},prepared::Tuple{W,Vararg{Any}},
        scales::Tuple{S,Vararg{Any}},basis,scratch,sector;
        adjoint::Bool=false) where {K,W,S}
    _threaded_apply_sector!(destination,source,first(kernels),first(prepared),
        basis,first(scales),scratch,sector;adjoint)
    _threaded_apply_sector_kernels!(destination,source,Base.tail(kernels),
        Base.tail(prepared),Base.tail(scales),basis,scratch,sector;adjoint)
end

function _threaded_apply_impl!(destination::AbstractVector,
        plan::LiouvillianPlan,source::AbstractVector,time,parameters,
        work::ThreadedLiouvillianWorkspace;adjoint::Bool=false)
    number_coordinates=length(plan.basis)
    length(source)==number_coordinates&&length(destination)==number_coordinates||
        throw(DimensionMismatch("Liouvillian vector has the wrong length"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "threaded Liouvillian source and destination must not share storage"))
    _check_threaded_workspace(work,plan)
    _check_liouvillian_apply_types(destination,source,plan)

    Threads.atomic_cas!(work.busy,0,1)==0||throw(ArgumentError(
        "a ThreadedLiouvillianWorkspace cannot be used concurrently"))
    try
        preparation=work.preparation
        if length(work.assignments)==1
            return adjoint ?
                apply_adjoint!(destination,plan,source,time,parameters,preparation) :
                apply!(destination,plan,source,time,parameters,preparation)
        end
        _prepare_kernels!(plan.kernels,preparation.kernel_workspaces,
                          plan.basis,time,parameters)
        scales=_threaded_scales(plan.kernels,plan.Ttype,time,parameters,adjoint)
        fill!(destination,zero(eltype(destination)))
        _copy_input_blocks!(preparation.blocks,source,plan.basis)
        prepared_source=_ThreadedPreparedSource(source,preparation.blocks)

        # One Julia task per private scratch.  Static scheduling and disjoint
        # output sectors make the arithmetic order for every coordinate
        # independent of task scheduling. Every worker reads the same
        # caller-packed source blocks.
        @sync for worker_index in eachindex(work.assignments)
            Threads.@spawn begin
                scratch=work.workers[worker_index]
                for sector in work.assignments[worker_index]
                    _threaded_apply_sector_kernels!(destination,prepared_source,
                        plan.kernels,preparation.kernel_workspaces,scales,
                        plan.basis,scratch,sector;adjoint)
                end
            end
        end
        destination
    finally
        work.busy[]=0
    end
end

"""
    threaded_apply!(destination, plan_or_compiled, source, time, parameters, workspace)
    threaded_apply!(destination, plan_or_compiled, source, workspace)

Apply a prepared PI Liouvillian with deterministic, output-sector-owned Julia
tasks. The short form requires an autonomous plan. Unlike compatibility
`mul!`, this opt-in route never changes global BLAS threading and never uses
atomic output updates or task-order-dependent reductions.
"""
function threaded_apply!(destination,plan::LiouvillianPlan,source,time,
                         parameters,work::ThreadedLiouvillianWorkspace)
    _threaded_apply_impl!(destination,plan,source,time,parameters,work)
end

function threaded_apply!(destination,plan::LiouvillianPlan,source,
                         work::ThreadedLiouvillianWorkspace)
    _require_autonomous(plan,"threaded_apply!")
    threaded_apply!(destination,plan,source,
                    zero(_real_float_type(plan.Ttype)),nothing,work)
end

function threaded_apply!(destination,compiled::CompiledPIModel,source,time,
                         parameters,work::ThreadedLiouvillianWorkspace)
    threaded_apply!(destination,compiled.plan,source,time,parameters,work)
end

function threaded_apply!(destination,compiled::CompiledPIModel,source,
                         work::ThreadedLiouvillianWorkspace)
    _require_autonomous(compiled,"threaded_apply!")
    threaded_apply!(destination,compiled.plan,source,work)
end

"""
    threaded_apply_adjoint!(destination, plan_or_compiled, source, time, parameters, workspace)
    threaded_apply_adjoint!(destination, plan_or_compiled, source, workspace)

Apply the Frobenius adjoint of a prepared PI Liouvillian using the same
deterministic output-sector ownership as [`threaded_apply!`](@ref).
"""
function threaded_apply_adjoint!(destination,plan::LiouvillianPlan,source,time,
                                 parameters,
                                 work::ThreadedLiouvillianWorkspace)
    _threaded_apply_impl!(destination,plan,source,time,parameters,work;
                          adjoint=true)
end

function threaded_apply_adjoint!(destination,plan::LiouvillianPlan,source,
                                 work::ThreadedLiouvillianWorkspace)
    _require_autonomous(plan,"threaded_apply_adjoint!")
    threaded_apply_adjoint!(destination,plan,source,
        zero(_real_float_type(plan.Ttype)),nothing,work)
end

function threaded_apply_adjoint!(destination,compiled::CompiledPIModel,
                                 source,time,parameters,
                                 work::ThreadedLiouvillianWorkspace)
    threaded_apply_adjoint!(destination,compiled.plan,source,time,parameters,
                            work)
end

function threaded_apply_adjoint!(destination,compiled::CompiledPIModel,
                                 source,work::ThreadedLiouvillianWorkspace)
    _require_autonomous(compiled,"threaded_apply_adjoint!")
    threaded_apply_adjoint!(destination,compiled.plan,source,work)
end
