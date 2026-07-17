using LinearAlgebra
using Random
using PermutationalInvariantDynamics

# Spectral functionals and population-coordinate metadata stay compressed.
basis=PIBasis(3,2)
rho=maximally_mixed_state(basis)
@assert spectral_trace(rho,identity)≈1
coordinates=each_population_coordinate(basis)
@assert length(coordinates)==population_dimension(basis)

sm=ComplexF64[0 1;0 0]
population_plan=PopulationPlan(PIModel(basis,(LocalJump(sm;rate=0.2),)))
transitions=population_transitions(population_plan)
@assert !isempty(transitions)

# A one-qubit amplitude-damping channel, PI POVM, and constrained tomography.
single=PIBasis(1,2)
probability=0.25
K0=collective_operator(single,ComplexF64[1 0;0 sqrt(1-probability)])
K1=collective_operator(single,ComplexF64[0 sqrt(probability);0 0])
channel=kraus_channel((K0,K1);check=true)
input=iid_state(single,ComplexF64[0.35 0.1;0.1 0.65])
output=apply_channel(channel,input)
@assert check_pi_channel(channel).trace_preserving
@assert isapprox(trace(output),1;atol=2e-12)

E0=collective_operator(single,ComplexF64[1 0;0 0])
E1=collective_operator(single,ComplexF64[0 0;0 1])
sample=sample_pi_povm(output,(E0,E1),2_000;rng=MersenneTwister(8))
estimate=maximum_likelihood_tomography(
    single,(E0,E1),sample.counts;maxiter=1_000)
@assert estimate.converged

# The dependency-free checkpoint retains the exact restricted basis and type.
mktempdir() do directory
    path=joinpath(directory,"state.pid")
    save_checkpoint(path,estimate.state;time=0.5,metadata=(source="MLE",))
    restored=load_checkpoint(path)
    @assert restored.state.data==estimate.state.data
    @assert restored.metadata["source"]=="MLE"
end

# Simultaneous weak-symmetry charges are intersected matrix-free.
sx=ComplexF64[0 1;1 0];sz=ComplexF64[1 0;0 -1]
joint=joint_symmetry_projector(PIBasis(2,2),(sx=>1,sz=>1))
@assert joint.range_dimension>0

println("population coordinates = ",length(coordinates))
println("directed population transitions = ",length(transitions))
println("channel CP/TP = ",check_pi_channel(channel))
println("tomography probabilities = ",estimate.probabilities)
println("joint symmetry rank = ",joint.range_dimension)
