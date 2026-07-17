"""Supertype for composable linear channels on one retained PI operator algebra."""
abstract type AbstractPIChannel end

"""
    PIChannel(basis, matrix; check=false, atol=1e-12, rtol=1e-10)

Wrap an explicit dense or sparse PI-coefficient channel.  `matrix` acts on the
orthonormal equation-(7) coordinates.  With `check=true`, construction requires
complete positivity and trace preservation on the retained direct-sum Schur
algebra; this scoped certificate is not a claim about sectors omitted from a
restricted basis.
"""
struct PIChannel{B,M} <: AbstractPIChannel
    basis::B
    matrix::M
    function PIChannel(basis::B,matrix::M;check::Bool=false,
            atol::Real=1e-12,rtol::Real=1e-10) where {B<:PIBasis,M<:AbstractMatrix}
        size(matrix)==(length(basis),length(basis))||throw(DimensionMismatch(
            "PI channel matrix has the wrong dimensions"))
        eltype(matrix)<:Number||throw(ArgumentError(
            "PI channel matrix must have numeric entries"))
        channel=new{B,M}(basis,matrix)
        if check
            report=check_pi_channel(channel;atol,rtol)
            report.completely_positive===true||throw(ArgumentError(
                "PI channel is not completely positive on the retained Schur algebra"))
            report.trace_preserving===true||throw(ArgumentError(
                "PI channel is not trace preserving"))
        end
        channel
    end
end

"""
    MatrixFreePIChannel(basis, T, action!, adjoint_action!)

Construct a matrix-free PI channel from preallocated forward and adjoint
callbacks `callback(destination, source)`.  Compatibility calls are
synchronized; callbacks themselves may be used with task-owned external
scratch when their implementation provides it.  Complete positivity cannot
be certified without materialization, while trace preservation is checked
through the supplied adjoint.
"""
struct MatrixFreePIChannel{B,T,F,A,K} <: AbstractPIChannel
    basis::B
    Ttype::Type{T}
    action!::F
    adjoint_action!::A
    lock::K
end

function MatrixFreePIChannel(basis::PIBasis,::Type{T},action!,adjoint_action!) where T<:Number
    MatrixFreePIChannel(basis,T,action!,adjoint_action!,ReentrantLock())
end

struct _ComposedPIChannel{A,B} <: AbstractPIChannel
    outer::A
    inner::B
end
struct _AdjointPIChannel{C} <: AbstractPIChannel
    parent::C
end

_channel_basis(channel::PIChannel)=channel.basis
_channel_basis(channel::MatrixFreePIChannel)=channel.basis
_channel_basis(channel::_ComposedPIChannel)=_channel_basis(channel.outer)
_channel_basis(channel::_AdjointPIChannel)=_channel_basis(channel.parent)

Base.size(channel::AbstractPIChannel)=(length(_channel_basis(channel)),
                                      length(_channel_basis(channel)))
Base.size(channel::AbstractPIChannel,index::Integer)=index in (1,2) ?
    length(_channel_basis(channel)) : 1
Base.eltype(channel::PIChannel)=eltype(channel.matrix)
Base.eltype(channel::MatrixFreePIChannel)=channel.Ttype
Base.eltype(channel::_ComposedPIChannel)=promote_type(eltype(channel.outer),
                                                      eltype(channel.inner))
Base.eltype(channel::_AdjointPIChannel)=eltype(channel.parent)

function mul!(destination::AbstractVector,channel::PIChannel,source::AbstractVector)
    mul!(destination,channel.matrix,source)
end

function mul!(destination::AbstractVector,channel::MatrixFreePIChannel,
              source::AbstractVector)
    length(destination)==length(source)==length(channel.basis)||
        throw(DimensionMismatch("matrix-free PI channel vector has wrong length"))
    lock(channel.lock)
    try
        result=channel.action!(destination,source)
        result===nothing||result===destination||throw(ArgumentError(
            "matrix-free PI channel callback must return its destination or nothing"))
    finally
        unlock(channel.lock)
    end
    destination
end

function mul!(destination::AbstractVector,channel::_AdjointPIChannel,
              source::AbstractVector)
    parent=channel.parent
    if parent isa PIChannel
        mul!(destination,adjoint(parent.matrix),source)
    elseif parent isa MatrixFreePIChannel
        lock(parent.lock)
        try
            result=parent.adjoint_action!(destination,source)
            result===nothing||result===destination||throw(ArgumentError(
                "matrix-free PI channel adjoint callback must return its destination or nothing"))
        finally
            unlock(parent.lock)
        end
        destination
    elseif parent isa _ComposedPIChannel
        mul!(destination,compose_channels(adjoint(parent.inner),
                                          adjoint(parent.outer)),source)
    elseif parent isa _AdjointPIChannel
        mul!(destination,parent.parent,source)
    else
        throw(ArgumentError("unsupported PI channel adjoint"))
    end
end

function mul!(destination::AbstractVector,channel::_ComposedPIChannel,
              source::AbstractVector)
    destination===source&&throw(ArgumentError(
        "composed PI channel destination must not alias its source"))
    temporary=similar(destination,promote_type(eltype(channel.inner),
                                               eltype(source)),length(source))
    mul!(temporary,channel.inner,source)
    mul!(destination,channel.outer,temporary)
end

function Base.:*(channel::AbstractPIChannel,source::AbstractVector)
    destination=zeros(promote_type(eltype(channel),eltype(source)),size(channel,1))
    mul!(destination,channel,source)
end

"""
    apply_channel!(destination, channel, source)
    apply_channel(channel, source)

Apply a PI channel to compatible `PIState` or `PIOperator` data.  The result is
not normalized, clipped, symmetrized, or positivity-repaired.
"""
function apply_channel!(destination::A,channel::AbstractPIChannel,
                        source::A) where A<:AbstractPIOperator
    basis=_channel_basis(channel)
    destination.basis===basis&&source.basis===basis||throw(ArgumentError(
        "PI channel and operators use incompatible bases"))
    destination===source&&throw(ArgumentError(
        "PI channel destination must not alias its source"))
    mul!(destination.data,channel,source.data);destination
end

"""Allocate and return the result of applying `channel` to a PI state or operator."""
function apply_channel(channel::AbstractPIChannel,source::PIState)
    output=PIState(source.basis;T=_real_float_type(promote_type(
        eltype(channel),eltype(source.data))))
    apply_channel!(output,channel,source)
end
function apply_channel(channel::AbstractPIChannel,source::PIOperator)
    output=PIOperator(source.basis;T=_real_float_type(promote_type(
        eltype(channel),eltype(source.data))))
    apply_channel!(output,channel,source)
end

"""
    compose_channels(outer, inner)

Return the lazy composition `outer o inner`.  Only one PI-sized temporary is
created per compatibility application; explicit matrices are multiplied once
at construction.
"""
function compose_channels(outer::AbstractPIChannel,inner::AbstractPIChannel)
    _channel_basis(outer)===_channel_basis(inner)||throw(ArgumentError(
        "composed PI channels must use the exact same basis object"))
    outer isa PIChannel&&inner isa PIChannel ?
        PIChannel(outer.basis,outer.matrix*inner.matrix) :
        _ComposedPIChannel(outer,inner)
end

"""Return the Hilbert--Schmidt adjoint of a PI channel."""
channel_adjoint(channel::AbstractPIChannel)=adjoint(channel)
Base.adjoint(channel::PIChannel)=PIChannel(channel.basis,adjoint(channel.matrix))
Base.adjoint(channel::_AdjointPIChannel)=channel.parent
Base.adjoint(channel::AbstractPIChannel)=_AdjointPIChannel(channel)

"""Construct the identity channel on a retained PI operator algebra."""
identity_channel(basis::PIBasis;T::Type{<:AbstractFloat}=Float64)=
    PIChannel(basis,sparse(I,length(basis),length(basis)).*one(Complex{T}))

"""
    kraus_channel(operators; check=false)

Construct the sector-preserving channel `rho -> sum_a K_a rho K_a'` from PI
Kraus operators on one exact basis.  Physical Schur blocks are used directly,
and no full-system Kraus matrix is formed.
"""
function kraus_channel(operators;check::Bool=false,atol::Real=1e-12,
                       rtol::Real=1e-10)
    ks=collect(operators);isempty(ks)&&throw(ArgumentError(
        "at least one PI Kraus operator is required"))
    all(K->K isa PIOperator,ks)||throw(ArgumentError(
        "Kraus operators must be PIOperators"))
    basis=first(ks).basis
    all(K->K.basis===basis,ks)||throw(ArgumentError(
        "all PI Kraus operators must share the exact basis"))
    T=foldl(promote_type,(eltype(K.data) for K in ks))
    blocks=SparseMatrixCSC{T,Int}[]
    for sector in basis.sectors
        n=size(coefficient_block(first(ks),sector),1)
        block=spzeros(T,n*n,n*n)
        for K in ks
            physical=Matrix(physical_block(K,sector))
            block+=sparse(kron(conj(physical),physical))
        end
        push!(blocks,block)
    end
    PIChannel(basis,blockdiag(blocks...);check,atol,rtol)
end

"""Scoped CP/TP diagnostics for a retained PI channel algebra."""
struct PIChannelCheck{R}
    completely_positive::Union{Bool,Missing}
    trace_preserving::Bool
    unital::Bool
    minimum_choi_eigenvalue::Union{R,Missing}
    trace_residual::R
    unital_residual::R
    tolerance::R
    scope::Symbol
    materialized::Bool
end

function _materialize_channel(channel::AbstractPIChannel)
    n=size(channel,1);T=_complex_float_type(eltype(channel))
    matrix=zeros(T,n,n);source=zeros(T,n)
    for column in 1:n
        fill!(source,zero(T));source[column]=one(T)
        mul!(view(matrix,:,column),channel,source)
    end
    matrix
end

function _channel_choi(component,nout,nin)
    T=eltype(component);choi=zeros(T,nout*nin,nout*nin)
    for j in 1:nin,i in 1:nin,b in 1:nout,a in 1:nout
        output=a+(b-1)*nout;input=i+(j-1)*nin
        choi[(a-1)*nin+i,(b-1)*nin+j]=component[output,input]
    end
    choi
end

"""
    check_pi_channel(channel; atol=1e-12, rtol=1e-10,
                     materialize=false)

Check trace preservation and unitality from PI coefficient trace/identity
vectors.  For an explicit channel, also test complete positivity of every map
between source and destination Schur matrix algebras by Choi eigenvalues.
For a matrix-free/lazy channel CP is reported as `missing` unless
`materialize=true` is explicitly requested.  The scope is always
`:retained_pi_algebra`; omitted Schur sectors are outside the certificate.
"""
function check_pi_channel(channel::AbstractPIChannel;atol::Real=1e-12,
        rtol::Real=1e-10,materialize::Bool=false)
    atol>=0&&rtol>=0||throw(ArgumentError("channel tolerances must be nonnegative"))
    basis=_channel_basis(channel);R=_real_float_type(eltype(channel))
    matrix=channel isa PIChannel ? channel.matrix :
           materialize ? _materialize_channel(channel) : nothing
    tau=_trace_vector(basis,Complex{R});identity=identity_operator(basis;T=R).data
    adjoint_identity=similar(identity);mul!(adjoint_identity,adjoint(channel),identity)
    trace_residual=R(norm(adjoint_identity-identity))
    image_identity=similar(identity);mul!(image_identity,channel,identity)
    unital_residual=R(norm(image_identity-identity))
    scale=max(norm(identity),one(R));tolerance=R(atol)+R(rtol)*R(scale)
    # `identity` is the Riesz vector of the trace functional in orthonormal
    # coefficient coordinates; retain an explicit consistency assertion.
    isapprox(identity,tau;atol=zero(R),rtol=R(8)*eps(R))||throw(ErrorException(
        "internal PI identity/trace-vector convention mismatch"))
    minimum_eigenvalue=missing;cp=missing
    if matrix!==nothing
        minimum_value=R(Inf);cp_value=true
        for (out_index,out_sector) in pairs(basis.sectors),
            (in_index,in_sector) in pairs(basis.sectors)
            rout=basis.offsets[out_index]:basis.offsets[out_index+1]-1
            rin=basis.offsets[in_index]:basis.offsets[in_index+1]-1
            nout=length(basis.patterns[out_index]);nin=length(basis.patterns[in_index])
            fin=symmetric_group_dimension(in_sector)
            fout=symmetric_group_dimension(out_sector)
            factor=_checked_sqrt_exact_ratio(R,fin,fout;
                context="PI channel physical Schur-map normalization")
            component=Matrix(matrix[rout,rin]);component.*=factor
            choi=_channel_choi(component,nout,nin)
            hermitian_error=norm(choi-choi',Inf)
            choi_scale=max(norm(choi,Inf),one(R))
            local_tolerance=R(atol)+R(rtol)*R(choi_scale)
            if hermitian_error>local_tolerance
                cp_value=false;minimum_value=R(-Inf);continue
            end
            values=_hermitian_eigvals(Hermitian((choi+choi')/2);
                                      operation="PI channel Choi check")
            # Every Schur-matrix algebra has positive dimension, so the Choi
            # spectrum is nonempty.  Supplying a zero initializer would bias
            # the reported minimum for strictly positive Choi matrices.
            local_min=minimum(values);minimum_value=min(minimum_value,local_min)
            cp_value&=local_min>=-local_tolerance
        end
        minimum_eigenvalue=minimum_value;cp=cp_value
    end
    PIChannelCheck(cp,trace_residual<=tolerance,unital_residual<=tolerance,
        minimum_eigenvalue,trace_residual,unital_residual,tolerance,
        :retained_pi_algebra,matrix!==nothing)
end
