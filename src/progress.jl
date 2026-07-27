"""
    CancellationToken()

Thread-safe, dependency-free cancellation token for long-running package
workflows. Request cancellation with [`cancel!`](@ref) and inspect it with
[`iscancelled`](@ref). Cancellation is cooperative: a workflow observes the
request only at a documented safe boundary, so numerical kernels are never
interrupted while mutating shared scratch.
"""
struct CancellationToken
    requested::Threads.Atomic{Bool}
end

CancellationToken()=CancellationToken(Threads.Atomic{Bool}(false))

"""Request cooperative cancellation. This operation is idempotent."""
function cancel!(token::CancellationToken)
    token.requested[]=true
    token
end

"""Return whether cooperative cancellation has been requested."""
iscancelled(token::CancellationToken)=token.requested[]

"""
    reset_cancellation!(token)

Clear a cancellation request before reusing a token for a new operation.
Resetting a token while an operation is using it is a caller error.
"""
function reset_cancellation!(token::CancellationToken)
    token.requested[]=false
    token
end

"""
    ProgressEvent

Dependency-free progress record emitted by long-running workflows.

- `operation` identifies the workflow.
- `stage` is normally `:started`, `:advanced`, `:completed`, `:stopped`, or
  `:cancelled`.
- `completed` and `total` describe safe completed units; `total=nothing`
  denotes work whose size is not known in advance.
- `metadata` contains workflow-specific, read-only context.

An `on_event` callback may return `nothing` to continue or `:cancel` to
request cooperative cancellation.
"""
struct ProgressEvent{M}
    operation::Symbol
    stage::Symbol
    completed::Int
    total::Union{Nothing,Int}
    elapsed_seconds::Float64
    message::String
    metadata::M
end

"""
    OperationCancelled

Exception raised when a workflow that cannot return a meaningful partial
result observes cooperative cancellation. The mutable destination, when one
exists, contains the last fully completed numerical unit.
"""
struct OperationCancelled <: Exception
    operation::Symbol
    completed::Int
    total::Union{Nothing,Int}
end

function Base.showerror(io::IO,error::OperationCancelled)
    print(io,"$(error.operation) cancelled after $(error.completed)")
    error.total===nothing||print(io," of $(error.total)")
    print(io," completed units")
end

mutable struct _ProgressContext{F,I}
    operation::Symbol
    callback::F
    io::I
    token::CancellationToken
    started_ns::UInt64
    terminal_emitted::Bool
end

function _prepare_progress(operation::Symbol;
        progress=false,on_event=nothing,cancellation_token=nothing)
    progress isa Union{Bool,IO}||throw(ArgumentError(
        "progress must be false, true, or an IO"))
    on_event===nothing||applicable(on_event,
        ProgressEvent(operation,:started,0,nothing,0.0,"",NamedTuple()))||
        throw(ArgumentError("on_event must accept one ProgressEvent"))
    cancellation_token===nothing||
        cancellation_token isa CancellationToken||throw(ArgumentError(
            "cancellation_token must be a CancellationToken or nothing"))
    if progress===false&&on_event===nothing&&cancellation_token===nothing
        return nothing
    end
    output=progress===true ? stderr : progress isa IO ? progress : nothing
    token=cancellation_token===nothing ? CancellationToken() :
                                        cancellation_token
    _ProgressContext(operation,on_event,output,token,time_ns(),false)
end

@inline _progress_cancelled(::Nothing)=false
@inline _progress_cancelled(context::_ProgressContext)=
    iscancelled(context.token)

function _print_progress_event(io::IO,event::ProgressEvent)
    print(io,'[',event.operation,"] ",event.stage,' ',event.completed)
    event.total===nothing||print(io,'/',event.total)
    isempty(event.message)||print(io," — ",event.message)
    println(io)
    flush(io)
    nothing
end

@inline function _progress_emit!(::Nothing,stage::Symbol,
                                 completed::Integer,total;
                                 message="",metadata=NamedTuple())
    false
end

function _progress_emit!(context::_ProgressContext,stage::Symbol,
                         completed::Integer,total;
                         message="",metadata=NamedTuple())
    completed>=0||throw(ArgumentError("completed progress must be nonnegative"))
    completed_int=Int(completed)
    total_int=if total===nothing
        nothing
    else
        total isa Integer&&!(total isa Bool)&&total>=0||throw(ArgumentError(
            "total progress must be a nonnegative integer or nothing"))
        Int(total)
    end
    total_int===nothing||completed_int<=total_int||throw(ArgumentError(
        "completed progress cannot exceed total progress"))
    event=ProgressEvent(context.operation,stage,completed_int,total_int,
        (time_ns()-context.started_ns)/1e9,string(message),metadata)
    context.io===nothing||_print_progress_event(context.io,event)
    if context.callback!==nothing
        response=context.callback(event)
        if response===:cancel
            cancel!(context.token)
        elseif response!==nothing
            throw(ArgumentError(
                "on_event must return nothing or :cancel"))
        end
    end
    stage in (:completed,:stopped,:cancelled)&&
        (context.terminal_emitted=true)
    _progress_cancelled(context)
end

@inline function _progress_cancel!(::Nothing,completed,total)
    false
end

function _progress_cancel!(context::_ProgressContext,completed,total)
    context.terminal_emitted||_progress_emit!(
        context,:cancelled,completed,total;message="cancellation requested")
    true
end

@inline function _progress_throw_if_cancelled!(::Nothing,completed,total)
    nothing
end

function _progress_throw_if_cancelled!(
        context::_ProgressContext,completed,total)
    _progress_cancelled(context)||return nothing
    _progress_cancel!(context,completed,total)
    throw(OperationCancelled(context.operation,Int(completed),
                             total===nothing ? nothing : Int(total)))
end
