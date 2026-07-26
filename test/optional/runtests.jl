using Test
using LinearAlgebra
using SparseArrays
using PermutationalInvariantDynamics
using Clarabel
using Makie
using QuantumCumulants
import QuantumOptics
import QuantumToolbox
using JLD2
using HDF5

@testset "optional package extensions" begin
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsClarabelExt)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsMakieExt)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsQuantumCumulantsExt)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsJLD2Ext)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsHDF5Ext)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsQuantumOpticsExt)!==nothing
    @test Base.get_extension(PermutationalInvariantDynamics,
        :PermutationalInvariantDynamicsQuantumToolboxExt)!==nothing

    spectrum=liouvillian_spectrum_data(ComplexF64[0,-1+0.25im])
    x,y=Makie.convert_arguments(Makie.Scatter,spectrum)
    @test collect(x)==[0.0,-1.0]
    @test collect(y)==[0.0,0.25]

    report=convergence_study(h->1+h^2,[0.4,0.2,0.1];
        consecutive=1,atol=0.1,rtol=0)
    refinements,errors=Makie.convert_arguments(Makie.Lines,report)
    @test collect(refinements)==[0.2,0.1]
    @test all(isfinite,errors)

    basis=PIBasis(1,3)
    rho=computational_product_state(basis,3)
    data=qudit_husimi_q(rho,zeros(ComplexF64,3,3);
        representation=:generator)
    points,values=Makie.convert_arguments(Makie.Lines,data)
    @test collect(points)==[1]
    @test collect(values)==data.values

    checkpoint_basis=PIBasis(2,2)
    checkpoint_state=iid_pure_state(
        checkpoint_basis,ComplexF32[1,im]/sqrt(2.0f0))
    checkpoint=PIStateCheckpoint(checkpoint_state;time=0.25f0,
        metadata=Dict("source"=>"optional-extension smoke"))
    mktempdir() do directory
        for (format,extension) in ((:jld2,"jld2"),(:hdf5,"h5"))
            path=joinpath(directory,"state.$extension")
            @test save_checkpoint(path,checkpoint;format)==path
            loaded=load_checkpoint(path;format)
            @test loaded.schema_version==PI_CHECKPOINT_VERSION
            @test loaded.state.basis.N==checkpoint_basis.N
            @test loaded.state.basis.d==checkpoint_basis.d
            @test loaded.state.basis.sectors==checkpoint_basis.sectors
            @test loaded.state.data==checkpoint_state.data
            @test loaded.time===0.25f0
            @test loaded.metadata==checkpoint.metadata
        end
    end
end

@testset "optional local-operator interchange" begin
    basis=PIBasis(3,2)
    expected=ComplexF64[0 1;0 0]

    qo_basis=QuantumOptics.NLevelBasis(2)
    qo_lowering=QuantumOptics.DenseOperator(qo_basis,expected)
    @test local_operator_matrix(qo_lowering;dimension=2)==expected
    @test LocalJump(qo_lowering;rate=0.2).operator==expected
    @test collective_operator(basis,qo_lowering).data≈
        collective_operator(basis,expected).data
    qo_sparse=QuantumOptics.SparseOperator(qo_basis,sparse(expected))
    @test local_operator_matrix(qo_sparse) isa SparseMatrixCSC
    @test_throws ArgumentError local_operator_matrix(
        QuantumOptics.Ket(qo_basis,ComplexF64[1,0]))
    @test_throws ArgumentError local_operator_matrix(
        QuantumOptics.spre(qo_lowering))
    qo_other_basis=QuantumOptics.FockBasis(1)
    qo_basis_map=QuantumOptics.DenseOperator(
        qo_basis,qo_other_basis,expected)
    @test_throws ArgumentError local_operator_matrix(qo_basis_map)
    @test_throws ArgumentError LocalJump(qo_basis_map;rate=0.2)
    @test_throws ArgumentError local_operator_matrix(
        QuantumOptics.DenseOperator(
            qo_basis,ComplexF64[NaN 0;0 0]))

    qt_lowering=QuantumToolbox.QuantumObject(expected)
    @test local_operator_matrix(qt_lowering;dimension=2)==expected
    @test CollectiveJump(qt_lowering;rate=0.2).operator==expected
    @test collective_operator(basis,qt_lowering).data≈
        collective_operator(basis,expected).data
    qt_sparse=QuantumToolbox.QuantumObject(sparse(expected))
    @test local_operator_matrix(qt_sparse) isa SparseMatrixCSC
    @test_throws ArgumentError local_operator_matrix(
        QuantumToolbox.QuantumObject(ComplexF64[1,0]))
    @test_throws ArgumentError local_operator_matrix(
        QuantumToolbox.spre(qt_lowering))
    @test_throws ArgumentError local_operator_matrix(
        QuantumToolbox.QuantumObject(ComplexF64[NaN 0;0 0]))
    @test_throws DimensionMismatch local_operator_matrix(
        QuantumToolbox.QuantumObject(zeros(ComplexF64,3,3));dimension=2)
end

@testset "optional Clarabel PI PPT-mixture solver" begin
    mix(state,mixed,weight)=PIState(state.basis,
        weight.*state.data.+(1-weight).*mixed.data)

    # For two qubits PPT-mixture membership is exactly ordinary PPT.  The
    # Werner threshold is q=1/3.
    basis2=PIBasis(2,2)
    singlet_partition=Partition((1,1))
    singlet=basis_state(basis2,singlet_partition,
        only(basis2.patterns[basis2.index[singlet_partition]]))
    mixed2=maximally_mixed_state(basis2)
    plan2=PPTMixturePlan(basis2)
    separable=ppt_mixture_test(mix(singlet,mixed2,0.2);plan=plan2,
        solver_options=(equilibrate_enable=false,))
    boundary=ppt_mixture_test(mix(singlet,mixed2,1/3);plan=plan2)
    entangled=ppt_mixture_test(mix(singlet,mixed2,0.6);plan=plan2)
    @test separable.classification==:ppt_mixture
    @test separable.ppt_mixture===true
    @test separable.genuinely_multipartite_entangled===false
    @test separable.biseparable===true
    @test boundary.classification!=:gme_certified
    @test entangled.classification==:gme_certified
    @test entangled.ppt_mixture===false
    @test entangled.genuinely_multipartite_entangled===true
    @test_throws ArgumentError ppt_mixture_test(mixed2;plan=plan2,
        solver_options=(not_a_clarabel_setting=true,))
    @test_throws ArgumentError ppt_mixture_test(mixed2;plan=plan2,
        solver_options=(max_iter=5,))
    @test_throws ArgumentError ppt_mixture_test(mixed2;plan=plan2,
        solver_options=(max_step_fraction="invalid",))

    # For PI three-qubit GHZ plus white noise, the exact PPT-mixture and
    # biseparability threshold is p=3/7.
    basis3=PIBasis(3,2)
    ghz=ghz_state(basis3;phase=pi/2)
    mixed3=maximally_mixed_state(basis3)
    plan3=PPTMixturePlan(basis3)
    below=ppt_mixture_test(mix(ghz,mixed3,0.35);plan=plan3)
    above=ppt_mixture_test(mix(ghz,mixed3,0.55);plan=plan3)
    @test below.classification==:ppt_mixture
    @test below.biseparable===true
    @test above.classification==:gme_certified
    @test above.dual_objective>0
    @test isfinite(above.dual_stationarity_residual)

    # A biseparable PI state may be NPT across every fixed cut.  This oracle
    # prevents replacing the PPT-mixture SDP by a cut-negativity shortcut.
    symmetric=Partition((3,0))
    lower=Partition((2,1))
    symmetric_block=zeros(ComplexF64,4,4)
    lower_block=zeros(ComplexF64,2,2)
    symmetric_patterns=basis3.patterns[basis3.index[symmetric]]
    lower_patterns=basis3.patterns[basis3.index[lower]]
    ground=findfirst(pattern->
        PermutationalInvariantDynamics.content(pattern)==(3,0),
        symmetric_patterns)
    double=findfirst(pattern->
        PermutationalInvariantDynamics.content(pattern)==(1,2),
        symmetric_patterns)
    lower_double=findfirst(pattern->
        PermutationalInvariantDynamics.content(pattern)==(1,2),
        lower_patterns)
    symmetric_block[ground,ground]=1/2
    symmetric_block[double,double]=1/6
    symmetric_block[ground,double]=sqrt(3)/6
    symmetric_block[double,ground]=sqrt(3)/6
    lower_block[lower_double,lower_double]=1/6
    twirled_biseparable=state_from_schur_blocks(basis3,
        [symmetric=>symmetric_block,lower=>lower_block];
        representation=:physical,validate=true)
    strict_biseparable=mix(twirled_biseparable,mixed3,0.9)
    @test negativity(strict_biseparable,1)>0.19
    false_positive_guard=ppt_mixture_test(strict_biseparable;plan=plan3)
    @test false_positive_guard.classification==:ppt_mixture
    @test false_positive_guard.biseparable===true

    # A feasible result for N>=4 is only a failure to detect GME.
    basis4=PIBasis(4,2)
    # This larger, highly degenerate SDP can reach Clarabel's `AlmostSolved`
    # status at the stricter default optimizer tolerance on some LAPACK
    # builds.  Match the solver tolerance to the independently checked
    # Float64 certificate tolerance for this result-semantics regression.
    result4=ppt_mixture_test(maximally_mixed_state(basis4);
        solver_atol=1e-7,solver_rtol=1e-7)
    @test result4.classification==:ppt_mixture
    @test result4.ppt_mixture===true
    @test result4.genuinely_multipartite_entangled===missing
    @test result4.biseparable===missing

    # Restricted inputs are embedded into all global Schur equality blocks.
    pure_full=ppt_mixture_test(ghz;plan=plan3)
    restricted_basis=PIBasis(3,2;sectors=[(3,0)])
    restricted_result=ppt_mixture_test(
        ghz_state(restricted_basis;phase=pi/2))
    @test restricted_result.classification==:gme_certified
    @test restricted_result.dual_objective≈pure_full.dual_objective atol=2e-8

    early=ppt_mixture_test(ghz;plan=plan3,max_iterations=1)
    @test early.iterations<=1
    @test early.solver_status isa Symbol
    if early.solver_status!=:solved
        @test early.classification==:inconclusive
        @test early.ppt_mixture===missing
    end

    basis32=PIBasis(3,2)
    result32=ppt_mixture_test(ghz_state(basis32;T=Float32);
        plan=PPTMixturePlan(basis32;T=Float32))
    @test result32 isa PPTMixtureResult{Float32}
    @test result32.classification==:gme_certified
end

# Reuse the package's exact normalization regressions after loading the weak
# dependency, which activates the conditional extension testset at its end.
include(joinpath(@__DIR__,"..","test_cumulants.jl"))
