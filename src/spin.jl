"""
    spin_matrices(d=2; T=Float64)

Return the standard spin-`j` matrices with `j=(d-1)/2` and `hbar=1` in the
ascending magnetic ordering ``|-j>,...,|j>``.  The result is the named tuple
`(j, jx, jy, jz, jp, jm)`, where `jp` and `jm` are the raising and lowering
operators.  In particular, for `d == 2`, `jx`, `jy`, and `jz` are Pauli
matrices divided by two, while `jm = |g><e|` in the package's local ordering
`(|g>, |e>)`.

`T` selects the real floating-point type retained by all returned matrices.
"""
function spin_matrices(d::Integer=2;T::Type{<:AbstractFloat}=Float64)
    d>=1||throw(ArgumentError("d must be positive"))
    dimension=try
        Int(d)
    catch error
        error isa InexactError||rethrow()
        throw(ArgumentError("d must be representable as an Int"))
    end
    half=one(T)/T(2)
    j=T(dimension-1)*half
    jp=zeros(Complex{T},dimension,dimension)
    for column in 1:dimension-1
        m=-j+T(column-1)
        jp[column+1,column]=sqrt((j-m)*(j+m+one(T)))
    end
    jm=copy(adjoint(jp))
    jx=(jp+jm)*half
    jy=(jp-jm)/(T(2)*im)
    jz=zeros(Complex{T},dimension,dimension)
    for index in 1:dimension
        jz[index,index]=-j+T(index-1)
    end
    (;j,jx,jy,jz,jp,jm)
end

"""
    collective_spin(basis, component; cache=nothing)

Construct one collective spin operator ``J_alpha=sum_i j_alpha^(i)`` without
forming the full computational-space matrix. `component` may be `:x`, `:y`,
`:z`, `:plus`, or `:minus`. The one-particle matrices use the ascending
``|-j>,...,|j>`` convention of [`spin_matrices`](@ref), with
`j=(basis.d-1)/2`.

The sole fully symmetric sector uses a lightweight occupation-number lift.
Pass a shared `OneBodyGeometry` when constructing several components on a
general Schur basis.
"""
function collective_spin(b::PIBasis,component::Symbol;
                         cache=nothing)
    cache=_collective_geometry(b,Float64,cache)
    matrices=spin_matrices(b.d;T=geometry_scalar_type(cache))
    local_operator=if component===:x
        matrices.jx
    elseif component===:y
        matrices.jy
    elseif component===:z
        matrices.jz
    elseif component===:plus||component===Symbol("+")
        matrices.jp
    elseif component===:minus||component===Symbol("-")
        matrices.jm
    else
        throw(ArgumentError(
            "component must be :x, :y, :z, :plus, or :minus"))
    end
    collective_operator(b,local_operator;cache=cache)
end

"""
    computational_product_state(basis, level; T=Float64)

Construct the tensor-power state ``|level><level|^otimes N`` directly in the
fully symmetric Schur sector. `level` is a one-based Julia local-state index
in `1:basis.d`; no full ``d^N`` vector is formed.
"""
function computational_product_state(b::PIBasis,level::Integer;
                                     T::Type{<:AbstractFloat}=Float64)
    1<=level<=b.d||throw(ArgumentError(
        "level must be a one-based local-state index in 1:$(b.d)"))
    psi=zeros(Complex{T},b.d)
    psi[Int(level)]=one(T)
    iid_pure_state(b,psi)
end

function _twice_qubit_label(value::Real,name::AbstractString)
    isfinite(value)||throw(ArgumentError("$name must be finite"))
    doubled=if value isa Integer||value isa Rational
        big(2)*big(value)
    else
        value+value
    end
    isfinite(doubled)&&isinteger(doubled)||throw(ArgumentError(
        "$name must be an integer or half-integer"))
    try
        Int(doubled)
    catch
        throw(ArgumentError(
            "$name is not representable as an integer or half-integer label"))
    end
end

function _qubit_dicke_labels(b::PIBasis,j::Real,m::Real)
    b.d==2||throw(ArgumentError("Dicke (j,m) labels require a qubit basis"))
    twice_j=_twice_qubit_label(j,"j")
    twice_m=_twice_qubit_label(m,"m")
    0<=twice_j<=b.N||throw(ArgumentError("j must satisfy 0 <= j <= N/2"))
    iseven(b.N-twice_j)||throw(ArgumentError(
        "j has the wrong integer/half-integer parity for N=$(b.N)"))
    -twice_j<=twice_m<=twice_j||throw(ArgumentError(
        "m must satisfy -j <= m <= j"))
    iseven(twice_j-twice_m)||throw(ArgumentError(
        "m must differ from j by an integer"))
    partition=Partition((div(b.N+twice_j,2),div(b.N-twice_j,2)))
    occupations=(div(b.N-twice_m,2),div(b.N+twice_m,2))
    partition,occupations
end

"""
    dicke_state(basis, j, m; T=Float64)

Construct the normalized qubit PI state with total-spin labels `(j,m)`.  With
the package ordering `(|g>,|e>)`, ``m=(n_e-n_g)/2`` and the corresponding
partition is ``(N/2+j,N/2-j)``.

For a sector of symmetric-group multiplicity `f`, this state is uniform over
the `f` indistinguishable multiplicity copies. Consequently its full
Hilbert-space purity is `1/f`; it is pure only when `f == 1`. The requested
sector must be present in `basis`.
"""
function dicke_state(b::PIBasis,j::Real,m::Real;
                     T::Type{<:AbstractFloat}=Float64)
    partition,occupations=_qubit_dicke_labels(b,j,m)
    sector_index=_sidx(b,partition)
    pattern_index=findfirst(pattern->content(pattern)==occupations,
                            b.patterns[sector_index])
    pattern_index===nothing&&error(
        "internal error: no GT pattern realizes Dicke labels (j=$j,m=$m)")
    basis_state(b,partition,b.patterns[sector_index][pattern_index];T=T)
end

"""
    dicke_operator(basis, j, m, mp; T=Float64)

Construct the qubit PI operator corresponding to ``|j,m><j,mp|``, uniformly
over the symmetric-group multiplicity copies of its Schur sector. If that
multiplicity is `f`, the selected physical Schur-block entry is `1/f` and the
stored equation-(7) coefficient is `1/sqrt(f)`. Consequently, the diagonal
case has the same coefficient data as [`dicke_state`](@ref).

Both magnetic labels must belong to the requested `j` ladder, and its sector
must be present in `basis`.
"""
function dicke_operator(b::PIBasis,j::Real,m::Real,mp::Real;
                        T::Type{<:AbstractFloat}=Float64)
    partition,occupations=_qubit_dicke_labels(b,j,m)
    second_partition,second_occupations=_qubit_dicke_labels(b,j,mp)
    partition==second_partition||error(
        "internal error: equal Dicke j labels produced different sectors")
    sector_index=_sidx(b,partition)
    patterns=b.patterns[sector_index]
    row=findfirst(pattern->content(pattern)==occupations,patterns)
    column=findfirst(pattern->content(pattern)==second_occupations,patterns)
    (row===nothing||column===nothing)&&error(
        "internal error: no GT pattern realizes the requested Dicke labels")
    operator=PIOperator(b;T=T)
    coefficient_block(operator,partition)[row,column]=
        _schur_inverse_multiplicity_scale(T,partition)
    operator
end

function _spin_state_real_type(requested,values...)
    if requested===nothing
        types=DataType[_real_float_type(typeof(value)) for value in values
                       if value isa AbstractFloat]
        return isempty(types) ? Float64 : foldl(promote_type,types)
    end
    requested isa Type&&requested<:AbstractFloat||throw(ArgumentError(
        "T must be an AbstractFloat type or nothing"))
    for value in values
        converted=try
            convert(requested,value)
        catch
            throw(ArgumentError("angle or phase is not representable in $requested"))
        end
        isfinite(converted)&&converted==value||throw(ArgumentError(
            "angle or phase would be narrowed by conversion to $requested"))
    end
    requested
end

"""
    ghz_state(basis; phase=0, T=nothing)

Construct the qubit GHZ state
``(|g>^otimes N + exp(im*phase)|e>^otimes N)/sqrt(2)`` directly in the fully
symmetric Schur sector. `basis` must contain that sector and have `N >= 1`;
no full ``2^N`` vector is formed. The scalar type is inferred from a floating
`phase` (or defaults to `Float64` for the exact default phase); pass `T`
explicitly to select a non-narrowing output type.
"""
function ghz_state(b::PIBasis;phase::Real=0,T=nothing)
    b.d==2||throw(ArgumentError("ghz_state requires a qubit basis"))
    b.N>=1||throw(ArgumentError("ghz_state requires at least one qubit"))
    R=_spin_state_real_type(T,phase)
    converted_phase=convert(R,phase)
    isfinite(converted_phase)||throw(ArgumentError("phase must be finite"))
    partition=Partition((b.N,0))
    sector_index=_sidx(b,partition)
    patterns=b.patterns[sector_index]
    ground=findfirst(pattern->content(pattern)==(b.N,0),patterns)
    excited=findfirst(pattern->content(pattern)==(0,b.N),patterns)
    (ground===nothing||excited===nothing)&&error(
        "internal error: symmetric qubit sector lacks a polarized GT pattern")
    amplitudes=zeros(Complex{R},length(patterns))
    scale=inv(sqrt(R(2)))
    amplitudes[ground]=scale
    amplitudes[excited]=scale*cis(converted_phase)
    sector_density_matrix(b,partition,amplitudes*amplitudes')
end

"""
    spin_coherent_state(basis, theta, phi=0; T=nothing)

Construct a qubit coherent-spin product state directly in the fully symmetric
Schur sector. In the local ordering `(|g>,|e>)`, the tensor-power factor is

``exp(im*phi) sin(theta/2)|g> + cos(theta/2)|e>``.

Thus the collective-spin mean points along
``(sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))``. Angles are used as
supplied and are not wrapped into a principal interval. The output type is
inferred from floating angle inputs; an explicit `T` must represent both
angles without narrowing.
"""
function spin_coherent_state(b::PIBasis,theta::Real,phi::Real=0;T=nothing)
    b.d==2||throw(ArgumentError(
        "spin_coherent_state currently requires a qubit basis"))
    R=_spin_state_real_type(T,theta,phi)
    converted_theta=convert(R,theta)
    converted_phi=convert(R,phi)
    isfinite(converted_theta)&&isfinite(converted_phi)||throw(ArgumentError(
        "theta and phi must be finite"))
    half=one(R)/R(2)
    psi=Complex{R}[
        cis(converted_phi)*sin(converted_theta*half),
        cos(converted_theta*half),
    ]
    iid_pure_state(b,psi)
end

_fixed_zero_rate(rate)=rate isa Number&&iszero(rate)

function _qubit_ensemble_real_type(requested,hamiltonian,rates)
    if requested!==nothing
        requested isa Type&&requested<:AbstractFloat||throw(ArgumentError(
            "T must be an AbstractFloat type or nothing"))
        return requested
    end
    types=DataType[]
    prototype=hamiltonian===nothing ? nothing : _operator_prototype(hamiltonian)
    if prototype isa AbstractPIOperator
        push!(types,_real_float_type(eltype(prototype.data)))
    elseif prototype isa AbstractMatrix
        push!(types,_real_float_type(eltype(prototype)))
    end
    for rate in rates
        rate isa Number&&!iszero(rate)&&
            push!(types,_real_float_type(typeof(rate)))
    end
    isempty(types) ? Float64 : foldl(promote_type,types)
end

function _qubit_ensemble_hamiltonian(b::PIBasis,hamiltonian)
    hamiltonian===nothing&&return ()
    if hamiltonian isa PIOperator
        return (DirectPIHamiltonian(hamiltonian),)
    elseif hamiltonian isa AbstractMatrix
        return (CollectiveHamiltonian(hamiltonian),)
    elseif hamiltonian isa InPlaceTimeOperator
        prototype=hamiltonian.prototype
        prototype isa PIOperator&&return (DirectPIHamiltonian(hamiltonian),)
        prototype isa AbstractMatrix&&return (CollectiveHamiltonian(hamiltonian),)
    end
    throw(ArgumentError(
        "hamiltonian must be nothing, a 2x2 one-particle matrix, a PIOperator, " *
        "or an InPlaceTimeOperator with one of those prototypes"))
end

function _append_qubit_jump(terms,rate,operator,collective::Bool)
    _fixed_zero_rate(rate)&&return terms
    term=collective ? CollectiveJump(operator;rate=rate) :
                      LocalJump(operator;rate=rate)
    (terms...,term)
end

"""
    qubit_ensemble_model(basis; hamiltonian=nothing,
                         emission=0, dephasing=0, pumping=0,
                         collective_emission=0, collective_dephasing=0,
                         collective_pumping=0, T=nothing)
    qubit_ensemble_model(N; sectors=nothing, kwargs...)

Construct the standard six-rate PI model for identical qubits. In the package
ordering `(|g>,|e>)`, emission uses `jm=|g><e|`, pumping uses `jp=jm'`, and
dephasing uses the spin operator `jz=Diagonal(-1/2,1/2)`. The contribution of
each keyword rate is

```
emission              * sum_i D[jm_i]
dephasing             * sum_i D[jz_i]
pumping               * sum_i D[jp_i]
collective_emission   * D[sum_i jm_i]
collective_dephasing  * D[sum_i jz_i]
collective_pumping    * D[sum_i jp_i]
```

where ``D[L](rho)=L*rho*L' - {L'*L,rho}/2``. Therefore the local dephasing
keyword damps a one-qubit off-diagonal element at `dephasing/2`; it is not the
same normalization as using a Pauli `sigma_z` jump.

`hamiltonian` may be a Hermitian `2x2` one-particle matrix, interpreted as
`sum_i h_i`, or a `PIOperator` on the exact model basis. An
`InPlaceTimeOperator` with either prototype is also accepted. Fixed numerical
zero rates are omitted, which permits collective-only models on a restricted
Schur basis. Callable rates are retained, and negative rates are passed
through unchanged. By default the spin-matrix precision is inferred from the
fixed nonzero numerical rates and Hamiltonian prototype; an all-zero or
callable-only model defaults to `Float64`. Pass `T` explicitly when a callable
rate needs another working precision. Wider rates or a wider Hamiltonian may
promote the compiled generator.
"""
function qubit_ensemble_model(b::PIBasis;hamiltonian=nothing,
        emission=0,dephasing=0,pumping=0,
        collective_emission=0,collective_dephasing=0,
        collective_pumping=0,T=nothing)
    b.d==2||throw(ArgumentError("qubit_ensemble_model requires d == 2"))
    rates=(emission,dephasing,pumping,collective_emission,
           collective_dephasing,collective_pumping)
    R=_qubit_ensemble_real_type(T,hamiltonian,rates)
    spin=spin_matrices(2;T=R)
    terms=_qubit_ensemble_hamiltonian(b,hamiltonian)
    terms=_append_qubit_jump(terms,emission,spin.jm,false)
    terms=_append_qubit_jump(terms,dephasing,spin.jz,false)
    terms=_append_qubit_jump(terms,pumping,spin.jp,false)
    terms=_append_qubit_jump(terms,collective_emission,spin.jm,true)
    terms=_append_qubit_jump(terms,collective_dephasing,spin.jz,true)
    terms=_append_qubit_jump(terms,collective_pumping,spin.jp,true)
    PIModel(b,terms)
end

function qubit_ensemble_model(N::Integer;sectors=nothing,kwargs...)
    basis=PIBasis(N,2;sectors=sectors)
    qubit_ensemble_model(basis;kwargs...)
end
