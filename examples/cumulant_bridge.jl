using PermutationalInvariantDynamics
using LinearAlgebra

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Exact PI reference data for an order-three cumulant closure.  The local
# alphabet is deliberately small: permutation symmetry stores only one moment
# per multiset of labels.
N=6
basis=PIBasis(N,2)
sx=ComplexF64[0 1;1 0]
sz=ComplexF64[1 0;0 -1]
sm=ComplexF64[0 1;0 0]
local_operators=(x=sx,z=sz,minus=sm)

# A product input provides a strict factorization sanity check.
sigma=ComplexF64[0.6 0.18;0.18 0.4]
rho_product=iid_state(basis,sigma)
product_moments=ordered_local_moments(
    rho_product,local_operators;order=3)

function product_closure(table,one_site_state)
    Dict(key=>prod(tr(getproperty(local_operators,label)*one_site_state)
                   for label in key)
         for key in keys(table))
end

factorized=product_closure(product_moments,sigma)
product_check=compare_cumulant_closure(product_moments,factorized)
@assert product_check.within_tolerance

# A GHZ state is not captured by the first-order product closure formed from
# its one-body marginal.  The exact table quantifies the missing correlations
# without reconstructing a 2^N density matrix.
rho_correlated=ghz_state(basis)
exact=ordered_local_moments(rho_correlated,local_operators;order=3)
factorized_correlated=product_closure(exact,one_body_rdm(rho_correlated))
correlation_error=compare_cumulant_closure(
    exact,factorized_correlated;atol=1e-12,rtol=1e-10)
@assert !correlation_error.within_tolerance

# The neutral model payload contains copied microscopic operators and all
# conventions needed by a package-specific symbolic adapter.  Direct PI terms
# would be marked microscopic=false instead of being assigned a guessed local
# realization.
pair_zz=kron(sz,sz)
model=PIModel(basis,(
    LocalHamiltonian(sx;rate=0.2),
    PBodyHamiltonian(pair_zz,2;rate=0.05),
    CollectiveJump(sm;rate=0.1),
))
bridge=cumulant_bridge_payload(
    model,rho_correlated,local_operators;order=3)

println(product_check)
println(correlation_error)
println("bridge schema: ",bridge.schema_version)
println("neutral terms: ",length(bridge.model.terms))
println("canonical exact moments: ",length(bridge.moments))

if makie_available()
    M=makie_module()
    orders=collect(1:3)
    product_errors=[
        maximum(
            (row.absolute_error for row in product_check.rows
             if length(row.labels)==order);
            init=0.0)
        for order in orders
    ]
    correlated_errors=[
        maximum(
            (row.absolute_error for row in correlation_error.rows
             if length(row.labels)==order);
            init=0.0)
        for order in orders
    ]
    canonical_counts=[
        count(row->length(row.labels)==order,correlation_error.rows)
        for order in orders
    ]
    ordered_counts=length(local_operators) .^ orders

    figure=M.Figure(size=(1080,430),fontsize=17)
    error_axis=M.Axis(
        figure[1,1];xlabel="moment order",
        ylabel="maximum absolute closure error",yscale=log10,
        xticks=orders,title="Product closure (display floor ε)")
    count_axis=M.Axis(
        figure[1,2];xlabel="moment order",ylabel="stored values",
        xticks=orders,title="Permutation-symmetric moment table")

    M.scatterlines!(
        error_axis,orders,max.(product_errors,eps(Float64));
        color=:seagreen4,linewidth=2.2,markersize=10,
        label="product input")
    M.scatterlines!(
        error_axis,orders,max.(correlated_errors,eps(Float64));
        color=:firebrick3,linewidth=2.2,markersize=10,
        label="GHZ input")
    M.scatterlines!(
        count_axis,orders,canonical_counts;
        color=:dodgerblue3,linewidth=2.2,markersize=10,
        label="canonical multisets")
    M.scatterlines!(
        count_axis,orders,ordered_counts;
        color=:gray35,linewidth=2,markersize=9,linestyle=:dash,
        label="all ordered words")
    M.axislegend(error_axis;position=:lt)
    M.axislegend(count_axis;position=:lt)
    save_example_figure(figure,"cumulant_bridge")
end

# Optional QuantumCumulants.jl 0.5 integration is loaded only when requested:
#
#   import QuantumCumulants
#   symbolic_map = Dict((:x,) => symbolic_average_x,
#                       (:x,:z) => symbolic_average_xz)
#   u0 = quantumcumulants_initial_values(exact, symbolic_map)
#
# The symbolic averages come from the user's chosen QuantumCumulants/SQA
# Hilbert and index spaces.  The extension validates their orders with the
# public QuantumCumulants.get_order API; it intentionally does not guess this
# model-specific symbolic mapping.
