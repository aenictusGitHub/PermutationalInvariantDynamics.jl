using Test
using PermutationalInvariantDynamics

@testset "prepared geometry bundles" begin
    basis=PIBasis(3,2)
    bundle=prepare_geometry(
        basis;T=Float32,pbody_orders=(2,),reduction_ks=(1,2),
        memory_budget=Inf)

    @test validate_prepared_geometry(bundle,basis)===bundle
    @test PermutationalInvariantDynamics.geometry_scalar_type(
        onebody_geometry(bundle))===Float32
    @test PermutationalInvariantDynamics.geometry_scalar_type(
        pbody_geometry(bundle,2))===Float32
    @test prepared_reductions(bundle).ks==(1,2)
    @test reduction_plan(prepared_reductions(bundle),1).basis===basis
    @test bundle.estimates.retained_bytes>0
    @test bundle.estimates.known_setup_peak_bytes>0
    @test_throws ArgumentError pbody_geometry(bundle,1)

    truncated=PreparedGeometryBundle(
        bundle.basis,bundle.basis_layout,bundle.scalar_type,
        bundle.precision_bits,bundle.rounding_mode,bundle.coefficient_cache,
        bundle.one_body,bundle.pbody_orders,(),
        bundle.reduction_plans,bundle.estimates)
    @test_throws DimensionMismatch validate_prepared_geometry(truncated)

    reductions=bundle.reduction_plans
    missing_plan=ReductionPlanSet(
        basis,reductions.ks,(first(reductions.plans),),reductions.estimates)
    malformed_reductions=PreparedGeometryBundle(
        bundle.basis,bundle.basis_layout,bundle.scalar_type,
        bundle.precision_bits,bundle.rounding_mode,bundle.coefficient_cache,
        bundle.one_body,bundle.pbody_orders,bundle.pbody_geometries,
        missing_plan,bundle.estimates)
    @test_throws DimensionMismatch validate_prepared_geometry(
        malformed_reductions)

    mismatched_k=ReductionPlanSet(
        basis,reverse(reductions.ks),reductions.plans,reductions.estimates)
    mismatched_reductions=PreparedGeometryBundle(
        bundle.basis,bundle.basis_layout,bundle.scalar_type,
        bundle.precision_bits,bundle.rounding_mode,bundle.coefficient_cache,
        bundle.one_body,bundle.pbody_orders,bundle.pbody_geometries,
        mismatched_k,bundle.estimates)
    @test_throws ArgumentError validate_prepared_geometry(
        mismatched_reductions)

    same_shape=PIBasis(3,2)
    @test_throws ArgumentError validate_prepared_geometry(bundle,same_shape)
    @test_throws ArgumentError onebody_geometry(bundle,same_shape)

    without=prepare_geometry(
        basis;one_body=false,coefficient_cache=nothing,memory_budget=Inf)
    @test without.coefficient_cache===nothing
    @test_throws ArgumentError onebody_geometry(without)
    @test_throws ArgumentError prepared_reductions(without)

    @test_throws ArgumentError prepare_geometry(
        basis;pbody_orders=(2,2),memory_budget=Inf)
    @test_throws ArgumentError prepare_geometry(
        basis;pbody_orders=(4,),memory_budget=Inf)
    @test_throws ArgumentError prepare_geometry(
        basis;reduction_ks=(1,1),memory_budget=Inf)
    @test_throws ArgumentError prepare_geometry(
        basis;reduction_ks=(1,),reduction_atol=big"1e-1000",
        memory_budget=Inf)
    @test_throws ArgumentError prepare_geometry(basis;memory_budget=0)

    qudit_basis=PIBasis(2,3)
    @test_throws ArgumentError prepare_geometry(
        qudit_basis;reduction_ks=(1,),memory_budget=512*1024^2)
end

@testset "prepared geometry BigFloat context" begin
    basis=PIBasis(2,2)
    bundle=setprecision(BigFloat,128) do
        prepare_geometry(
            basis;T=BigFloat,pbody_orders=(2,),memory_budget=Inf)
    end
    setprecision(BigFloat,128) do
        @test validate_prepared_geometry(bundle)===bundle
    end
    setprecision(BigFloat,192) do
        @test_throws ArgumentError validate_prepared_geometry(bundle)
    end
end

@testset "user-owned preparation cache" begin
    basis=PIBasis(3,2)
    cache=PreparationCache(memory_budget=Inf)
    first=prepare_geometry!(
        cache,basis;pbody_orders=(2,),reduction_ks=(1,))
    second=prepare_geometry!(
        cache,basis;pbody_orders=(2,),reduction_ks=(1,))
    @test first===second
    @test preparation_cache_summary(cache).entry_count==1

    distinct=prepare_geometry!(
        cache,basis;T=Float32,pbody_orders=(2,),reduction_ks=(1,))
    @test distinct!==first
    @test preparation_cache_summary(cache).entry_count==2
    equal_basis=PIBasis(3,2)
    equal_bundle=prepare_geometry!(cache,equal_basis;one_body=false)
    @test equal_bundle.basis===equal_basis
    @test preparation_cache_summary(cache).basis_count==2

    coefficients=OneBoxCGCache(
        basis;max_depth=2,memory_budget=Inf)
    external=prepare_geometry!(
        cache,basis;pbody_orders=(2,),coefficient_cache=coefficients)
    @test external.coefficient_cache===coefficients
    @test evict_prepared_geometry!(cache,first)
    @test !evict_prepared_geometry!(cache,first)
    @test preparation_cache_summary(cache).entry_count==3
    @test clear_preparation_cache!(cache)===cache
    @test preparation_cache_summary(cache).entry_count==0
    @test preparation_cache_summary(cache).retained_bytes==0

    tiny=PreparationCache(memory_budget=0)
    @test_throws ArgumentError prepare_geometry!(tiny,basis)
    @test preparation_cache_summary(tiny).entry_count==0
end

@testset "bundle detects basis layout mutation" begin
    basis=PIBasis(2,2)
    bundle=prepare_geometry(
        basis;one_body=false,coefficient_cache=nothing,memory_budget=Inf)
    old=basis.offsets[1]
    basis.offsets[1]=old+1
    @test_throws ArgumentError validate_prepared_geometry(bundle)
    basis.offsets[1]=old
    @test validate_prepared_geometry(bundle)===bundle

    sector=first(basis.sectors)
    old_index=basis.index[sector]
    basis.index[sector]=old_index+1
    @test_throws ArgumentError validate_prepared_geometry(bundle)
    basis.index[sector]=old_index
    @test validate_prepared_geometry(bundle)===bundle

    bogus=Partition((3,0))
    basis.index[bogus]=length(basis.index)+1
    @test_throws ArgumentError validate_prepared_geometry(bundle)
    delete!(basis.index,bogus)
    @test validate_prepared_geometry(bundle)===bundle
end
