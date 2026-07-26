"""
    AffineCompiledPIModelFamily

Prepared family for a Liouvillian which is affine in named real parameters,

```math
\\mathcal L(\\theta)=\\mathcal L_0+\\sum_a\\theta_a\\mathcal L_a.
```

Each component is expressed with ordinary physical PI terms.  The complete
term geometry is lowered once through [`compile_family`](@ref); parameter
specialization only forms the scalar rates required by the already prepared
kernels.  The object is immutable and may be shared between tasks.

The affine coefficients multiply complete generator contributions.  For a
jump operator written as `a * L`, supply `abs2(a)` as the coefficient of the
`D[L]` component.  A parameter-dependent Kossakowski matrix can be represented
as a sum of fixed positive-semidefinite correlated-jump components when that
decomposition is physically appropriate.
"""
struct AffineCompiledPIModelFamily{B,C,N,D,F,E}
    base_model::B
    components::C
    names::N
    defaults::D
    rate_family::F
    estimates::E
end

function Base.show(io::IO,family::AffineCompiledPIModelFamily)
    print(io,"AffineCompiledPIModelFamily(N=$(family.base_model.basis.N), ",
          "d=$(family.base_model.basis.d), ",
          "parameters=$(family.names), ",
          "dimension=$(size(family.rate_family.plan,1)))")
end

function _affine_component_terms(component,basis,name)
    terms=component isa PIModel ? component.terms :
        component isa AbstractPITerm ? (component,) : Tuple(component)
    component isa PIModel&&component.basis!==basis&&throw(ArgumentError(
        "affine component $name uses a different PIBasis object"))
    isempty(terms)&&throw(ArgumentError(
        "affine component $name must contain at least one physical term"))
    all(term->term isa AbstractPITerm,terms)||throw(ArgumentError(
        "affine component $name must be a PIModel, an AbstractPITerm, or a " *
        "collection of AbstractPITerm values"))
    for term in terms
        term_isautonomous(term)||throw(ArgumentError(
            "affine component $name must use fixed operators and numeric rates"))
        rate=term_rate(term)
        rate isa Real&&isfinite(rate)||throw(ArgumentError(
            "affine component $name has a nonfinite or non-real prototype rate"))
    end
    terms
end

function _affine_components(components,basis)
    components isa NamedTuple||throw(ArgumentError(
        "components must be a nonempty NamedTuple of physical term groups"))
    isempty(components)&&throw(ArgumentError(
        "components must contain at least one named generator contribution"))
    names=keys(components)
    terms=map(name->_affine_component_terms(
        getproperty(components,name),basis,name),names)
    # Canonicalize every group before retaining it. In particular, a caller
    # may supply a Vector of terms, but an immutable prepared family must not
    # revisit that mutable source collection during specialization.
    names,NamedTuple{names}(terms)
end

function _affine_defaults(names,defaults)
    values = defaults===nothing ? ntuple(_->1,length(names)) :
        defaults isa NamedTuple ?
            ntuple(index->begin
                length(keys(defaults))==length(names)||throw(ArgumentError(
                    "defaults must contain exactly the affine parameters $names"))
                name=names[index]
                hasproperty(defaults,name)||throw(ArgumentError(
                    "defaults is missing affine parameter $name"))
                getproperty(defaults,name)
            end,length(names)) :
            Tuple(defaults)
    length(values)==length(names)||throw(DimensionMismatch(
        "expected $(length(names)) affine defaults, got $(length(values))"))
    for (index,value) in pairs(values)
        value isa Real&&!(value isa Bool)&&isfinite(value)||throw(ArgumentError(
            "default affine parameter $(names[index]) must be finite and real"))
    end
    NamedTuple{names}(values)
end

"""
    compile_affine_family(base_model, components;
                          defaults=nothing,
                          bigfloat_precision=precision(BigFloat),
                          coefficient_cache=nothing)

Prepare a fixed affine generator family. `base_model` supplies
``\\mathcal L_0``. `components` is a nonempty `NamedTuple`; each value is a
`PIModel` on the exact same basis, one `AbstractPITerm`, or a tuple/vector of
terms.  A component's scalar parameter multiplies every prototype term rate in
that component.

All operators and prototype rates must be autonomous.  Negative specialized
coefficients are accepted because deterministic time-local generators may
have negative rates; stochastic use must perform its own nonnegative-rate
validation.  Geometry, selection rules, and output precision are identical to
compiling the explicitly specialized model.
"""
function compile_affine_family(base_model::PIModel,components;
        defaults=nothing,bigfloat_precision::Integer=precision(BigFloat),
        coefficient_cache=nothing)
    isautonomous(base_model)||throw(ArgumentError(
        "compile_affine_family requires an autonomous base model"))
    names,canonical_components=_affine_components(
        components,base_model.basis)
    component_terms=Tuple(values(canonical_components))
    default_values=_affine_defaults(names,defaults)
    flattened=()
    ranges=Vector{UnitRange{Int}}(undef,length(names))
    next_index=length(base_model.terms)+1
    for index in eachindex(component_terms)
        terms=component_terms[index]
        ranges[index]=next_index:(next_index+length(terms)-1)
        flattened=(flattened...,terms...)
        next_index+=length(terms)
    end
    combined=PIModel(base_model.basis,(base_model.terms...,flattened...))
    selected=Tuple(Iterators.flatten(ranges))
    rate_family=compile_family(combined;rate_indices=selected,
        bigfloat_precision,coefficient_cache)
    estimates=merge(rate_family.estimates,(;
        affine_parameters=names,
        component_term_ranges=Tuple(ranges),
        base_term_count=length(base_model.terms),
        component_term_count=length(flattened),
        specialization=:affine_generator_coefficients))
    AffineCompiledPIModelFamily(
        base_model,canonical_components,names,default_values,
        rate_family,estimates)
end

function _affine_parameter_values(family::AffineCompiledPIModelFamily,
                                  parameters)
    values = parameters===nothing ? Tuple(values(family.defaults)) :
        parameters isa NamedTuple ?
            ntuple(index->begin
                length(keys(parameters))==length(family.names)||
                    throw(ArgumentError(
                        "affine parameters must contain exactly $(family.names)"))
                name=family.names[index]
                hasproperty(parameters,name)||throw(ArgumentError(
                    "affine parameters are missing $name"))
                getproperty(parameters,name)
            end,length(family.names)) :
        parameters isa Number&&length(family.names)==1 ? (parameters,) :
        Tuple(parameters)
    length(values)==length(family.names)||throw(DimensionMismatch(
        "expected $(length(family.names)) affine parameters, got $(length(values))"))
    for (index,value) in pairs(values)
        value isa Real&&!(value isa Bool)&&isfinite(value)||throw(ArgumentError(
            "affine parameter $(family.names[index]) must be finite and real"))
    end
    values
end

@inline function _affine_checked_product(coefficient,prototype_rate,
        ::Type{R},name::Symbol) where R<:AbstractFloat
    left=_checked_prepared_real(
        coefficient,R,"affine parameter $name")
    right=_checked_prepared_real(
        prototype_rate,R,"prototype rate for affine parameter $name")
    (iszero(left)||iszero(right))&&return zero(R)

    product=left*right
    if isfinite(product)&&!iszero(product)
        # An endpoint can be a rounded image of a mathematical value just
        # outside the representable interval. The check is cold and uses a
        # wider type only at those two rare boundaries.
        fixed_ieee=R===Float16||R===Float32||R===Float64
        endpoint=fixed_ieee&&(
            abs(product)==floatmax(R)||
            abs(product)==nextfloat(zero(R)))
        if endpoint
            W=widen(R)
            wide_product=W(left)*W(right)
            minimum=W(nextfloat(zero(R)))
            maximum=W(floatmax(R))
            minimum<=abs(wide_product)<=maximum||throw(ArgumentError(
                "affine rate for parameter $name is outside the nonzero " *
                "finite range of prepared precision $R; compile the model " *
                "at a wider precision"))
        end
        return product
    end

    # Recompute only the exceptional path in a wider type. This distinguishes
    # arithmetic range loss from invalid input without adding wider arithmetic
    # to ordinary parameter-scan specializations.
    W=widen(R)
    wide_product=W(left)*W(right)
    reason=iszero(product)&&!iszero(wide_product) ? "underflows" :
        !isfinite(product)&&isfinite(wide_product) ? "overflows" :
        "is outside the nonzero finite range"
    throw(ArgumentError(
        "affine rate for parameter $name $reason in prepared precision $R; " *
        "compile the model at a wider precision"))
end

function _affine_bound_rates(family::AffineCompiledPIModelFamily,parameters)
    coefficients=_affine_parameter_values(family,parameters)
    combined=family.rate_family.model
    R=_real_float_type(family.rate_family.plan.Ttype)
    rates=Vector{Any}()
    sizehint!(rates,length(family.rate_family.rate_indices))
    offset=length(family.base_model.terms)
    for (component_index,terms) in
            pairs(Tuple(values(family.components)))
        coefficient=coefficients[component_index]
        for term_index in eachindex(terms)
            prototype_rate=term_rate(combined.terms[offset+term_index])
            push!(rates,_affine_checked_product(
                coefficient,prototype_rate,R,
                family.names[component_index]))
        end
        offset+=length(terms)
    end
    Tuple(rates)
end

"""
    specialize(family::AffineCompiledPIModelFamily,
               parameters=family.defaults; kwargs...)

Bind named or positional affine coefficients and return an ordinary
[`SpecializedPIModel`](@ref).  All sparse, matrix-free, solver, scan, and
workspace APIs therefore work without an affine-specific execution path.
"""
function specialize(family::AffineCompiledPIModelFamily,
        parameters=family.defaults;kwargs...)
    rates=_affine_bound_rates(family,parameters)
    specialized=specialize(family.rate_family,rates;kwargs...)
    estimates=merge(specialized.estimates,(;
        affine_parameters=NamedTuple{family.names}(
            _affine_parameter_values(family,parameters)),
        affine_family=true))
    SpecializedPIModel(
        specialized.family,specialized.model,specialized.plan,
        specialized.operator,specialized.rates,specialized.backend,estimates)
end
