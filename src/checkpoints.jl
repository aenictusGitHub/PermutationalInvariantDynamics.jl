"""Current on-disk schema version written by [`save_checkpoint`](@ref)."""
const PI_CHECKPOINT_VERSION=UInt16(1)
const _PI_CHECKPOINT_MAGIC=UInt8[0x50,0x49,0x44,0x43,0x48,0x4b,0x50,0x54]

"""
    PIStateCheckpoint(state; time=nothing, metadata=Dict())

Versioned, detached checkpoint of one PI state.  Metadata keys and values are
stored as strings for portability across the built-in binary, JLD2, and HDF5
backends.  The exact retained partition list and coefficient precision are
part of the schema; loading never substitutes a complete basis for a
restricted one.
"""
struct PIStateCheckpoint{S,T}
    schema_version::UInt16
    state::S
    time::T
    metadata::Dict{String,String}
end

function PIStateCheckpoint(state::PIState;time=nothing,metadata=Dict())
    R=_real_float_type(eltype(state.data))
    stored_time=if time===nothing
        nothing
    else
        time isa Real||throw(ArgumentError("checkpoint time must be real"))
        converted=R(time)
        isfinite(converted)||throw(ArgumentError(
            "checkpoint time is not finite in state precision $R"))
        converted==time||throw(ArgumentError(
            "checkpoint time cannot be represented exactly in state precision $R"))
        converted
    end
    stored_metadata=Dict{String,String}()
    for (key,value) in pairs(metadata)
        stored_metadata[string(key)]=string(value)
    end
    PIStateCheckpoint(PI_CHECKPOINT_VERSION,copy(state),stored_time,
                      stored_metadata)
end

"""Return a detached `PIState` from a [`PIStateCheckpoint`](@ref)."""
checkpoint_state(checkpoint::PIStateCheckpoint)=copy(checkpoint.state)

function _checkpoint_format(path,format)
    format===:auto||return Symbol(format)
    extension=lowercase(splitext(String(path))[2])
    extension in (".jld2", ".jld")&&return :jld2
    extension in (".h5", ".hdf5")&&return :hdf5
    :pid
end

function _checkpoint_scalar_code(::Type{Float16});UInt8(16);end
function _checkpoint_scalar_code(::Type{Float32});UInt8(32);end
function _checkpoint_scalar_code(::Type{Float64});UInt8(64);end
function _checkpoint_scalar_code(::Type{BigFloat});UInt8(128);end
function _checkpoint_scalar_code(::Type{R}) where R
    throw(ArgumentError("portable PI checkpoints do not support scalar type $R; use Float16, Float32, Float64, or BigFloat"))
end

_checkpoint_scalar_type(code::UInt8)=code==16 ? Float16 : code==32 ? Float32 :
    code==64 ? Float64 : code==128 ? BigFloat : throw(ArgumentError(
        "unknown PI checkpoint scalar code $code"))

function _write_checkpoint_string(io,value::AbstractString)
    bytes=codeunits(value);length(bytes)<=typemax(UInt32)||throw(ArgumentError(
        "checkpoint string is too long"))
    write(io,UInt32(length(bytes)));write(io,bytes)
end

function _read_checkpoint_string(io)
    length_value=Int(read(io,UInt32))
    String(read(io,length_value))
end

function _write_checkpoint_real(io,value::R) where R<:Union{Float16,Float32,Float64}
    write(io,value)
end
_write_checkpoint_real(io,value::BigFloat)=_write_checkpoint_string(io,string(value))

_read_checkpoint_real(io,::Type{R}) where R<:Union{Float16,Float32,Float64}=read(io,R)
_read_checkpoint_real(io,::Type{BigFloat})=parse(BigFloat,_read_checkpoint_string(io))

function _checkpoint_bigfloat_precision(checkpoint::PIStateCheckpoint)
    stored_precision=0
    function include_precision(value::BigFloat)
        value_precision=precision(value)
        if iszero(stored_precision)
            stored_precision=value_precision
        elseif value_precision!=stored_precision
            throw(ArgumentError(
                "portable BigFloat checkpoints require one common precision for every real and imaginary coefficient and the optional time"))
        end
    end
    for value in checkpoint.state.data
        include_precision(real(value));include_precision(imag(value))
    end
    checkpoint.time===nothing||include_precision(checkpoint.time)
    stored_precision>0||throw(ArgumentError(
        "cannot determine the precision of an empty BigFloat checkpoint"))
    stored_precision<=typemax(Int32)||throw(ArgumentError(
        "BigFloat checkpoint precision exceeds the portable schema limit"))
    stored_precision
end

function _save_checkpoint(::Val{:pid},path,checkpoint::PIStateCheckpoint)
    state=checkpoint.state;R=_real_float_type(eltype(state.data));code=_checkpoint_scalar_code(R)
    open(path,"w") do io
        write(io,_PI_CHECKPOINT_MAGIC);write(io,checkpoint.schema_version)
        write(io,code)
        code==128&&write(io,Int32(_checkpoint_bigfloat_precision(checkpoint)))
        write(io,Int64(state.basis.N));write(io,Int32(state.basis.d))
        write(io,Int32(length(state.basis.sectors)))
        for sector in state.basis.sectors,part in sector.parts
            write(io,Int64(part))
        end
        write(io,UInt8(checkpoint.time===nothing ? 0 : 1))
        checkpoint.time===nothing||_write_checkpoint_real(io,R(checkpoint.time))
        write(io,Int64(length(state.data)))
        for value in state.data
            _write_checkpoint_real(io,R(real(value)))
            _write_checkpoint_real(io,R(imag(value)))
        end
        write(io,Int32(length(checkpoint.metadata)))
        for (key,value) in sort!(collect(checkpoint.metadata);by=first)
            _write_checkpoint_string(io,key);_write_checkpoint_string(io,value)
        end
    end
    String(path)
end

function _load_checkpoint(::Val{:pid},path)
    open(path,"r") do io
        read(io,length(_PI_CHECKPOINT_MAGIC))==_PI_CHECKPOINT_MAGIC||
            throw(ArgumentError("file is not a PI checkpoint"))
        version=read(io,UInt16);version==PI_CHECKPOINT_VERSION||throw(ArgumentError(
            "unsupported PI checkpoint schema version $version"))
        code=read(io,UInt8);R=_checkpoint_scalar_type(code)
        precision_value=code==128 ? Int(read(io,Int32)) : 0
        code==128&&precision_value<=0&&throw(ArgumentError(
            "invalid BigFloat checkpoint precision $precision_value"))
        loader=function ()
            N=Int(read(io,Int64));d=Int(read(io,Int32));nsectors=Int(read(io,Int32))
            N>=0&&d>=1&&nsectors>=1||throw(ArgumentError(
                "invalid PI checkpoint basis metadata"))
            sectors=[ntuple(_->Int(read(io,Int64)),d) for _ in 1:nsectors]
            basis=PIBasis(N,d;sectors)
            has_time=read(io,UInt8);has_time in (0,1)||throw(ArgumentError(
                "invalid PI checkpoint time flag"))
            time=has_time==1 ? _read_checkpoint_real(io,R) : nothing
            coordinate_count=Int(read(io,Int64));coordinate_count==length(basis)||
                throw(DimensionMismatch("PI checkpoint coefficient length does not match its basis"))
            data=Vector{Complex{R}}(undef,coordinate_count)
            for index in eachindex(data)
                data[index]=complex(_read_checkpoint_real(io,R),
                                    _read_checkpoint_real(io,R))
            end
            metadata_count=Int(read(io,Int32));metadata_count>=0||throw(ArgumentError(
                "invalid PI checkpoint metadata count"))
            metadata=Dict{String,String}()
            for _ in 1:metadata_count
                key=_read_checkpoint_string(io);haskey(metadata,key)&&throw(ArgumentError(
                    "duplicate PI checkpoint metadata key $key"))
                metadata[key]=_read_checkpoint_string(io)
            end
            eof(io)||throw(ArgumentError("trailing data in PI checkpoint"))
            PIStateCheckpoint(version,PIState(basis,data),time,metadata)
        end
        if code==128
            setprecision(BigFloat,precision_value) do
                loader()
            end
        else
            loader()
        end
    end
end

function _save_checkpoint(::Val{F},path,checkpoint) where F
    F in (:jld2,:hdf5)&&throw(ArgumentError(
        "checkpoint format :$F requires loading the corresponding optional JLD2 or HDF5 package"))
    throw(ArgumentError("unknown PI checkpoint format :$F"))
end
function _load_checkpoint(::Val{F},path) where F
    F in (:jld2,:hdf5)&&throw(ArgumentError(
        "checkpoint format :$F requires loading the corresponding optional JLD2 or HDF5 package"))
    throw(ArgumentError("unknown PI checkpoint format :$F"))
end

"""
    save_checkpoint(path, checkpoint; format=:auto)
    save_checkpoint(path, state; time=nothing, metadata=Dict(), format=:auto)

Write a versioned PI state checkpoint.  `format=:auto` selects JLD2 for
`.jld2`, HDF5 for `.h5`/`.hdf5`, and the dependency-free `:pid` format
otherwise.  Optional formats become available when their package extension is
loaded.  Existing files are replaced only because the caller explicitly
requested this write.
"""
function save_checkpoint(path,checkpoint::PIStateCheckpoint;format=:auto)
    checkpoint.schema_version==PI_CHECKPOINT_VERSION||throw(ArgumentError(
        "cannot write unsupported checkpoint schema $(checkpoint.schema_version)"))
    selected=_checkpoint_format(path,format)
    _save_checkpoint(Val(selected),path,checkpoint)
end
function save_checkpoint(path,state::PIState;time=nothing,metadata=Dict(),
                         format=:auto)
    save_checkpoint(path,PIStateCheckpoint(state;time,metadata);format)
end

"""
    load_checkpoint(path; format=:auto)

Load and validate a versioned [`PIStateCheckpoint`](@ref).  The coefficient
vector, restricted sector list, time, and string metadata are detached from
the file backend.  State physicality is not silently repaired; call
[`validate_state`](@ref) when loading untrusted numerical data.
"""
function load_checkpoint(path;format=:auto)
    isfile(path)||throw(ArgumentError("checkpoint file does not exist: $path"))
    selected=_checkpoint_format(path,format)
    checkpoint=_load_checkpoint(Val(selected),path)
    checkpoint isa PIStateCheckpoint||throw(ArgumentError(
        "checkpoint backend returned an invalid object"))
    checkpoint
end

# Primitive backend-neutral schema used by optional extensions.
function _checkpoint_payload(checkpoint::PIStateCheckpoint)
    state=checkpoint.state
    _real_float_type(eltype(state.data))===BigFloat&&
        _checkpoint_bigfloat_precision(checkpoint)
    (;schema_version=Int(checkpoint.schema_version),N=state.basis.N,d=state.basis.d,
      sectors=reduce(vcat,(collect(sector.parts)' for sector in state.basis.sectors)),
      scalar_type=string(_real_float_type(eltype(state.data))),
      real=real.(state.data),imag=imag.(state.data),
      has_time=checkpoint.time!==nothing,
      time=checkpoint.time===nothing ? zero(_real_float_type(eltype(state.data))) : checkpoint.time,
      metadata_keys=collect(keys(checkpoint.metadata)),
      metadata_values=[checkpoint.metadata[key] for key in keys(checkpoint.metadata)])
end

function _checkpoint_from_payload(payload)
    UInt16(payload.schema_version)==PI_CHECKPOINT_VERSION||throw(ArgumentError(
        "unsupported PI checkpoint schema version $(payload.schema_version)"))
    sectors=[Tuple(Int.(payload.sectors[row,:])) for row in axes(payload.sectors,1)]
    basis=PIBasis(Int(payload.N),Int(payload.d);sectors)
    length(payload.real)==length(payload.imag)==length(basis)||throw(DimensionMismatch(
        "checkpoint payload coefficient length does not match its basis"))
    R=eltype(payload.real);R<:AbstractFloat||throw(ArgumentError(
        "checkpoint payload coefficients must use a floating scalar type"))
    string(payload.scalar_type)==string(R)||throw(ArgumentError(
        "checkpoint payload scalar type $(payload.scalar_type) does not match coefficient type $R"))
    eltype(payload.imag)===R||throw(ArgumentError(
        "checkpoint payload real and imaginary coefficients use different scalar types"))
    data=complex.(Vector{R}(payload.real),Vector{R}(payload.imag))
    keys_vector=String.(payload.metadata_keys);values_vector=String.(payload.metadata_values)
    length(keys_vector)==length(values_vector)||throw(DimensionMismatch(
        "checkpoint metadata key/value lengths differ"))
    metadata=Dict(keys_vector .=> values_vector)
    time=Bool(payload.has_time) ? R(payload.time) : nothing
    PIStateCheckpoint(PI_CHECKPOINT_VERSION,PIState(basis,data),time,metadata)
end
