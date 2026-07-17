### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# ╔═╡ 24f8ce2e-20ea-4d3b-8e5c-7b25f11ca0bc
md"""
# Prepared PI research workflow

Launch Pluto from the dedicated `notebooks` environment. This notebook demonstrates a
continuation scan, a convergence report, and plotting through the optional
Makie conversion layer without constructing a full ``2^N`` state.
"""

# ╔═╡ 7ee76596-e5e3-42a4-946e-bc31ecde68cb
begin
    using LinearAlgebra
    using PermutationalInvariantDynamics
end

# ╔═╡ 6ed43866-cab5-467c-bb12-84e4c115a855
begin
    basis = PIBasis(8, 2)
    sm = ComplexF64[0 1; 0 0]
    sp = sm'
    sz = ComplexF64[1 0; 0 -1] / 2
    pump_rates = collect(range(0.05, 0.8; length=18))
    builder = pump -> PIModel(basis, (
        LocalJump(sm; rate=1.0),
        LocalJump(sp; rate=pump),
    ))
end

# ╔═╡ 73032742-ecaf-41b0-a734-4938b64f92ad
begin
    scan_plan = ParameterScanPlan(pump_rates, builder;
        task=:steady_state,
        algorithm=GMRESAlgorithm(krylovdim=24, maxiter=300),
        continuation=true,
        diagnostic=(rho, pump) -> (
            magnetization=real(collective_expectation(rho, sz)) / basis.N,
            pump=pump,
        ))
    scan = parameter_scan(scan_plan)
end

# ╔═╡ e4c8c39a-59ba-4588-8cab-f455b70a3b26
parameter_scan_rows(scan)

# ╔═╡ 59d71902-b0b8-4f8c-bde6-61af749f3a7d
begin
    rho0 = computational_product_state(basis, 1)
    timestep_report = timestep_convergence([0.04, 0.02, 0.01, 0.005];
        atol=2e-7, rtol=2e-5) do dt
        time_evolve(builder(0.3), rho0, (0.0, 0.5);
                    steps=ceil(Int, 0.5 / dt))
    end
end

# ╔═╡ 113745e4-27bb-410b-b5d6-aa2ee08bceae
timestep_report

# ╔═╡ 2946f5cf-6803-4a66-9334-8d92d2f21c1e
md"""
To plot directly, add CairoMakie to the notebook environment and evaluate
`lines(timestep_report)`. The package extension converts the already-computed
pairwise errors; it does not rerun the evolution.
"""

# ╔═╡ Cell order:
# ╠═24f8ce2e-20ea-4d3b-8e5c-7b25f11ca0bc
# ╠═7ee76596-e5e3-42a4-946e-bc31ecde68cb
# ╠═6ed43866-cab5-467c-bb12-84e4c115a855
# ╠═73032742-ecaf-41b0-a734-4938b64f92ad
# ╠═e4c8c39a-59ba-4588-8cab-f455b70a3b26
# ╠═59d71902-b0b8-4f8c-bde6-61af749f3a7d
# ╠═113745e4-27bb-410b-b5d6-aa2ee08bceae
# ╠═2946f5cf-6803-4a66-9334-8d92d2f21c1e
