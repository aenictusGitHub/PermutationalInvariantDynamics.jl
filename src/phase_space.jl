"""
    SpinPhaseSpaceData

Numerical, sector-aware spin phase-space data for a qubit `PIState`.  The
sample matrix `values` is indexed as `values[phi_index, theta_index]` and is
the aggregate obtained by summing the selected Schur-sector distributions.
`sector_values` is either `nothing` or a vector of matrices in the same order
as `sectors`, depending on the `resolved` keyword used during construction.

Every sector distribution is normalized as a density with respect to the
sphere measure `sin(theta) dtheta dphi`: its integral is the corresponding
entry of `populations`.  Thus the aggregate integrates to
`sum(populations)`, which is one only when all sectors of a normalized state
are selected.  Combining different total-spin sectors is explicitly a
marginal over the discrete sector label; the sector-resolved spheres remain
the primary representation.
"""
struct SpinPhaseSpaceData{R<:AbstractFloat,P,S,M}
    kind::Symbol
    theta::Vector{R}
    phi::Vector{R}
    values::Matrix{R}
    sectors::Vector{P}
    twice_spins::Vector{Int}
    multiplicities::Vector{BigInt}
    populations::Vector{R}
    sector_values::S
    normalization::Symbol
    metadata::M
end

function show(io::IO,data::SpinPhaseSpaceData)
    print(io,"SpinPhaseSpaceData(kind=",data.kind,
          ", grid=",length(data.phi),"×",length(data.theta),
          ", sectors=",length(data.sectors),
          ", resolved=",data.sector_values!==nothing,")")
end

function show(io::IO,::MIME"text/plain",data::SpinPhaseSpaceData)
    show(io,data)
    print(io,"\n  normalization: ",data.normalization,
          "\n  selected population: ",sum(data.populations),
          "\n  matrix indexing: values[phi_index, theta_index]")
end

function _phase_space_real_type(rho,theta,phi)
    promote_type(_real_float_type(eltype(rho.data)),
                 _real_float_type(eltype(theta)),
                 _real_float_type(eltype(phi)))
end

function _phase_space_angles(rho::PIState,theta::AbstractVector{<:Real},
                             phi::AbstractVector{<:Real})
    Base.require_one_based_indexing(theta,phi)
    isempty(theta)&&throw(ArgumentError("theta cannot be empty"))
    isempty(phi)&&throw(ArgumentError("phi cannot be empty"))
    R=_phase_space_real_type(rho,theta,phi)
    R<:AbstractFloat||throw(ArgumentError(
        "phase-space coordinates must promote to an AbstractFloat type"))
    theta_values=R.(theta)
    phi_values=R.(phi)
    pi_value=R(pi)
    all(value->isfinite(value)&&zero(R)<=value<=pi_value,theta_values)||
        throw(ArgumentError("theta values must be finite and lie in [0, pi]"))
    all(isfinite,phi_values)||throw(ArgumentError("phi values must be finite"))
    theta_values,phi_values
end

function _default_phase_space_angles(rho::PIState,ntheta::Integer,nphi::Integer)
    ntheta>=2||throw(ArgumentError("ntheta must be at least 2"))
    nphi>=2||throw(ArgumentError("nphi must be at least 2"))
    ntheta<=typemax(Int)||throw(ArgumentError("ntheta is too large"))
    nphi<=typemax(Int)||throw(ArgumentError("nphi is too large"))
    R=_real_float_type(eltype(rho.data))
    nt=Int(ntheta);np=Int(nphi);pi_value=R(pi)
    theta=R[pi_value*R(index-1)/R(nt-1) for index in 1:nt]
    # The periodic endpoint is deliberately omitted: zero and 2pi represent
    # the same meridian and retaining both would duplicate one grid column.
    phi=R[R(2)*pi_value*R(index-1)/R(np) for index in 1:np]
    theta,phi
end

function _phase_space_sector_indices(basis::PIBasis,sectors)
    basis.d==2||throw(ArgumentError(
        "spin phase-space functions currently require a qubit basis (d == 2)"))
    indices=Int[]
    if sectors===:all
        append!(indices,eachindex(basis.sectors))
    elseif sectors isa Partition ||
           (sectors isa Tuple&&length(sectors)==2&&all(x->x isa Integer,sectors)) ||
           (sectors isa AbstractVector&&length(sectors)==2&&
            all(x->x isa Integer,sectors))
        push!(indices,basis.index[_schur_block_partition(basis,sectors)])
    else
        sectors isa Symbol&&throw(ArgumentError(
            "sectors must be :all, one retained partition, or an iterable of retained partitions"))
        for label in sectors
            partition=_schur_block_partition(basis,label)
            push!(indices,basis.index[partition])
        end
    end
    isempty(indices)&&throw(ArgumentError("at least one Schur sector must be selected"))
    length(unique(indices))==length(indices)||throw(ArgumentError(
        "duplicate Schur sector in phase-space selection"))
    sort!(indices)
end

function _phase_space_validation(rho,normalization,atol,rtol)
    normalization===:sphere_density||throw(ArgumentError(
        "normalization must be :sphere_density"))
    validate_state(rho;atol=atol,rtol=rtol)
    nothing
end

function _multiplicity_weighted_block(rho::PIState,sector_index::Int,::Type{R}) where R
    partition=rho.basis.sectors[sector_index]
    # Reuse the fused exact-scaling analysis path.  The weighted block has the
    # sector population as its trace and can remain representable even when
    # `sqrt(f^nu)` itself is outside `R`.
    source=_multiplicity_weighted_block(rho,partition)
    block=(source isa Matrix{Complex{R}} ? source :
           Matrix{Complex{R}}(source))::Matrix{Complex{R}}
    # State validation has already certified the skew-Hermitian residual.
    # Averaging only removes accepted roundoff and is never applied to an
    # unchecked or invalid state.
    half=one(R)/R(2)
    @inbounds for column in axes(block,2)
        block[column,column]=complex(real(block[column,column]),zero(R))
        for row in 1:column-1
            value=(block[row,column]+conj(block[column,row]))*half
            block[row,column]=value
            block[column,row]=conj(value)
        end
    end
    population=real(LinearAlgebra.tr(block))
    block,population
end

# Stable square roots of the binomial probabilities in descending-m order.
# Index one is m=+j.  Starting at the probability mode avoids separately
# converting a potentially enormous binomial coefficient to the work type.
function _spin_coherent_amplitudes!(amplitudes::AbstractVector{R},
                                    theta::R) where R<:AbstractFloat
    dimension=length(amplitudes);two_j=dimension-1
    half=one(R)/R(2)
    if iszero(theta)
        fill!(amplitudes,zero(R));amplitudes[1]=one(R)
        return amplitudes
    elseif theta==R(pi)
        fill!(amplitudes,zero(R));amplitudes[end]=one(R)
        return amplitudes
    end
    cosine=cos(theta*half);sine=sin(theta*half)
    fill!(amplitudes,zero(R))
    if iszero(sine)
        amplitudes[1]=one(R)
        return amplitudes
    elseif iszero(cosine)
        amplitudes[end]=one(R)
        return amplitudes
    end
    probability=cosine*cosine
    mode=clamp(floor(Int,R(two_j+1)*probability),0,two_j)
    amplitudes[two_j-mode+1]=one(R)
    # k=j+m is the number of local spin-up factors.  Array indices reverse k
    # because the package's qubit GT blocks are ordered m=+j,...,-j.
    for k in mode+1:two_j
        ratio=sqrt(R(two_j-k+1)/R(k))*(cosine/sine)
        amplitudes[two_j-k+1]=amplitudes[two_j-(k-1)+1]*ratio
    end
    for k in mode-1:-1:0
        ratio=sqrt(R(k+1)/R(two_j-k))*(sine/cosine)
        amplitudes[two_j-k+1]=amplitudes[two_j-(k+1)+1]*ratio
    end
    scale=norm(amplitudes)
    iszero(scale)&&error("internal error: coherent-spin amplitude recurrence vanished")
    amplitudes./=scale
    amplitudes
end

function _azimuthal_phases(phi::Vector{R},dimension::Int) where R
    phases=Matrix{Complex{R}}(undef,dimension,length(phi))
    @inbounds for column in eachindex(phi)
        phases[1,column]=one(Complex{R})
        step=cis(phi[column])
        for q in 1:dimension-1
            phases[q+1,column]=phases[q,column]*step
        end
    end
    phases
end

function _spin_husimi_sector!(destination::Matrix{R},
                              block::Matrix{Complex{R}},
                              theta::Vector{R},phi::Vector{R}) where R
    dimension=size(block,1)
    amplitudes=zeros(R,dimension)
    coefficients=zeros(Complex{R},dimension)
    phases=_azimuthal_phases(phi,dimension)
    scale=R(dimension)/(R(4)*R(pi))
    @inbounds for theta_index in eachindex(theta)
        _spin_coherent_amplitudes!(amplitudes,theta[theta_index])
        coefficients[1]=sum(real(block[index,index])*amplitudes[index]^2
                            for index in 1:dimension)
        for q in 1:dimension-1
            coefficient=zero(Complex{R})
            for row in 1:dimension-q
                column=row+q
                coefficient+=block[row,column]*amplitudes[row]*amplitudes[column]
            end
            coefficients[q+1]=coefficient
        end
        for phi_index in eachindex(phi)
            value=real(coefficients[1])
            for q in 1:dimension-1
                value+=R(2)*real(coefficients[q+1]*phases[q+1,phi_index])
            end
            destination[phi_index,theta_index]=scale*value
        end
    end
    destination
end

function _normalized_highest_tensors(::Type{R},dimension::Int) where R
    ladder=R[sqrt(R(index)*R(dimension-index)) for index in 1:dimension-1]
    highest=fill(inv(sqrt(R(dimension))),dimension)
    ladder,highest
end

function _multipole_coefficients(block::Matrix{Complex{R}}) where R
    dimension=size(block,1)
    # Polarization-tensor recurrences below use ascending m.  PI qubit GT
    # blocks use the opposite order, so both axes are explicitly reversed.
    ascending=@view block[dimension:-1:1,dimension:-1:1]
    coefficients=zeros(Complex{R},dimension,dimension)
    ladder,positive_highest=_normalized_highest_tensors(R,dimension)
    next_highest=zeros(R,dimension)
    tensor_a=zeros(R,dimension)
    tensor_b=zeros(R,dimension)
    for k in 0:dimension-1
        highest_length=dimension-k
        sign=isodd(k) ? -one(R) : one(R)
        @inbounds for index in 1:highest_length
            tensor_a[index]=sign*positive_highest[index]
        end
        tensor=tensor_a;lowered=tensor_b
        q=k
        while q>=0
            coefficient=zero(Complex{R})
            @inbounds for column in 1:dimension-q
                row=column+q
                coefficient+=conj(tensor[column])*ascending[row,column]
            end
            coefficients[k+1,q+1]=coefficient
            q==0&&break
            lowered_length=dimension-q+1
            @inbounds for column in 1:lowered_length
                value=zero(R)
                column<=dimension-q &&
                    (value+=ladder[column+q-1]*tensor[column])
                column>=2 && (value-=tensor[column-1]*ladder[column-1])
                lowered[column]=value
            end
            denominator=sqrt(R((k+q)*(k-q+1)))
            @inbounds for column in 1:lowered_length
                lowered[column]/=denominator
            end
            tensor,lowered=lowered,tensor
            q-=1
        end
        if k<dimension-1
            next_length=dimension-k-1
            @inbounds for column in 1:next_length
                next_highest[column]=ladder[column+k]*positive_highest[column]
            end
            next_norm=norm(view(next_highest,1:next_length))
            iszero(next_norm)&&error(
                "internal error: highest polarization tensor recurrence vanished")
            @inbounds for column in 1:next_length
                next_highest[column]/=next_norm
            end
            positive_highest,next_highest=next_highest,positive_highest
        end
    end
    coefficients
end

function _spherical_fourier_coefficients!(destination::Vector{Complex{R}},
        multipoles::Matrix{Complex{R}},theta::R) where R
    dimension=size(multipoles,1)
    fill!(destination,zero(Complex{R}))
    cosine,sine = if iszero(theta)
        (one(R),zero(R))
    elseif theta==R(pi)
        (-one(R),zero(R))
    else
        (cos(theta),sin(theta))
    end
    diagonal=inv(sqrt(R(4)*R(pi))) # Y_0^0(theta,0)
    for q in 0:dimension-1
        if q>0
            diagonal*=-sqrt(R(2q+1)/R(2q))*sine
        end
        previous_previous=zero(R)
        previous=diagonal
        destination[q+1]+=multipoles[q+1,q+1]*previous
        q==dimension-1&&continue
        current=sqrt(R(2q+3))*cosine*previous
        destination[q+1]+=multipoles[q+2,q+1]*current
        previous_previous,previous=previous,current
        for k in q+2:dimension-1
            denominator=R(k*k-q*q)
            a=sqrt(R(4k*k-1)/denominator)
            b=sqrt(R((2k+1)*((k-1)*(k-1)-q*q))/
                   (R(2k-3)*denominator))
            current=a*cosine*previous-b*previous_previous
            destination[q+1]+=multipoles[k+1,q+1]*current
            previous_previous,previous=previous,current
        end
    end
    destination
end

function _spin_wigner_sector!(destination::Matrix{R},
                              block::Matrix{Complex{R}},
                              theta::Vector{R},phi::Vector{R}) where R
    dimension=size(block,1)
    multipoles=_multipole_coefficients(block)
    fourier=zeros(Complex{R},dimension)
    phases=_azimuthal_phases(phi,dimension)
    scale=sqrt(R(dimension)/(R(4)*R(pi)))
    @inbounds for theta_index in eachindex(theta)
        _spherical_fourier_coefficients!(fourier,multipoles,theta[theta_index])
        for phi_index in eachindex(phi)
            value=real(fourier[1])
            for q in 1:dimension-1
                value+=R(2)*real(fourier[q+1]*phases[q+1,phi_index])
            end
            destination[phi_index,theta_index]=scale*value
        end
    end
    destination
end

function _spin_phase_space(rho::PIState,theta::AbstractVector{<:Real},
        phi::AbstractVector{<:Real},kind::Symbol;
        sectors=:all,resolved::Bool=false,
        normalization::Symbol=:sphere_density,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho))
    theta_values,phi_values=_phase_space_angles(rho,theta,phi)
    _phase_space_validation(rho,normalization,atol,rtol)
    indices=_phase_space_sector_indices(rho.basis,sectors)
    R=eltype(theta_values)
    aggregate=zeros(R,length(phi_values),length(theta_values))
    sector_data=resolved ? Matrix{R}[] : nothing
    scratch=resolved ? nothing : similar(aggregate)
    selected_sectors=eltype(rho.basis.sectors)[]
    twice_spins=Int[];multiplicities=BigInt[];populations=R[]
    for sector_index in indices
        partition=rho.basis.sectors[sector_index]
        block,population=_multiplicity_weighted_block(rho,sector_index,R)
        values=resolved ? zeros(R,size(aggregate)) : scratch
        if kind===:husimi_q
            _spin_husimi_sector!(values,block,theta_values,phi_values)
        else
            _spin_wigner_sector!(values,block,theta_values,phi_values)
        end
        aggregate .+= values
        resolved&&push!(sector_data,values)
        push!(selected_sectors,partition)
        push!(twice_spins,partition[1]-partition[2])
        push!(multiplicities,symmetric_group_dimension(partition))
        push!(populations,population)
    end
    metadata=(N=rho.basis.N,d=rho.basis.d,
              selected_population=sum(populations),
              aggregate=:marginal_over_schur_sectors,
              ordering=:phi_theta,source_type=typeof(rho),
              resolved=resolved)
    SpinPhaseSpaceData(kind,theta_values,phi_values,aggregate,
        selected_sectors,twice_spins,multiplicities,populations,sector_data,
        normalization,metadata)
end

"""
    spin_husimi_q(rho, theta, phi; sectors=:all, resolved=false,
                  normalization=:sphere_density, atol=..., rtol=...)
    spin_husimi_q(rho; ntheta=91, nphi=180, kwargs...)

Compute the spin Husimi-Q distribution of a qubit PI state without forming a
`2^N` state.  For Schur sector `nu` with spin `j`, dimension `n_j=2j+1`,
multiplicity `f^nu`, physical block `rho_j`, and coherent state
`|theta,phi;j>`, the returned sector density is

``P_Q(j,theta,phi) = n_j/(4pi) *
    <theta,phi;j| f^nu rho_j |theta,phi;j>``.

It integrates over the sphere to that sector's physical population.  The
aggregate `values` is the sum over the selected sector spheres and is
therefore explicitly a marginal over `j`.  Use `resolved=true` to retain the
individual matrices in the returned [`SpinPhaseSpaceData`](@ref); the default
streams them into the aggregate to bound retained memory.  `theta` must lie
in `[0,pi]`; arbitrary finite azimuths are accepted and are not wrapped.

The state is validated but never normalized, clipped, or otherwise repaired.
Only `normalization=:sphere_density` is currently defined.
"""
function spin_husimi_q(rho::PIState,theta::AbstractVector{<:Real},
                       phi::AbstractVector{<:Real};kwargs...)
    _spin_phase_space(rho,theta,phi,:husimi_q;kwargs...)
end

function spin_husimi_q(rho::PIState;ntheta::Integer=91,nphi::Integer=180,
                       kwargs...)
    theta,phi=_default_phase_space_angles(rho,ntheta,nphi)
    spin_husimi_q(rho,theta,phi;kwargs...)
end

"""
    spin_wigner(rho, theta, phi; sectors=:all, resolved=false,
                 normalization=:sphere_density, atol=..., rtol=...)
    spin_wigner(rho; ntheta=91, nphi=180, kwargs...)

Compute the sector-resolved Agarwal spin-Wigner distribution of a qubit PI
state.  In spin-`j` sector the implementation uses orthonormal
Condon--Shortley polarization tensors `T_kq`, coefficients
`t_kq=tr(T_kq' * f^nu rho_j)`, and spherical harmonics `Y_kq`.  The returned
sector density is

``P_W(j,theta,phi) = sqrt((2j+1)/(4pi)) *
    sum(k,q, t_kq * Y_kq(theta,phi))``.

Its sphere integral is the sector population, and a maximally mixed
conditional spin block contributes the uniform density `p_j/(4pi)`. Wigner
negativity is retained exactly; values are never clipped. The aggregate is
the explicit marginal over the selected Schur sectors rather than a claim
that a multi-sector state belongs to one spin irrep.

`resolved=true` retains every selected sector matrix.  Tensor coefficients
are generated from diagonal recurrences with call-local workspace; no dense
`2^N` object or global mutable cache is constructed.  The state and angle
validation rules are the same as for [`spin_husimi_q`](@ref).
"""
function spin_wigner(rho::PIState,theta::AbstractVector{<:Real},
                     phi::AbstractVector{<:Real};kwargs...)
    _spin_phase_space(rho,theta,phi,:wigner;kwargs...)
end

function spin_wigner(rho::PIState;ntheta::Integer=91,nphi::Integer=180,
                     kwargs...)
    theta,phi=_default_phase_space_angles(rho,ntheta,nphi)
    spin_wigner(rho,theta,phi;kwargs...)
end
