function _content_hot_allocations(pattern)
    content(pattern)
    @allocated content(pattern)
end

@testset "GT Clebsch-Gordan coefficients" begin
    allocation_pattern=first(gt_patterns(Partition((4,2,1))))
    @test _content_hot_allocations(allocation_pattern)==0

    for d in 2:4, N in 1:5, mu in partitions(N-1,d)
        mus=gt_patterns(mu)
        lambdas=unique(add_corner(mu,r) for r in addable_corners(mu))
        states=[(l,w) for l in lambdas for w in gt_patterns(l)]
        C=zeros(Float64,sum(length(gt_patterns(l)) for l in lambdas),d*length(mus))
        for (a,(l,w)) in pairs(states), (q,wm) in pairs(mus),j in 0:d-1
            C[a,q+(j*length(mus))]=cgc(wm,j,w)
        end
        @test C*C' ≈ I atol=2e-13
    end
end

@testset "basis-owned sparse one-box coefficient cache" begin
    PID=PermutationalInvariantDynamics
    basis=PIBasis(4,3)
    cache=OneBoxCGCache(basis;max_depth=2,T=Float64)
    @test cache.basis===basis
    @test cache.max_depth==2
    @test cache.coefficient_count<=cache.candidate_count
    @test cache.estimated_bytes>0
    @test Base.summarysize(cache)<=cache.estimated_bytes

    dense_candidates=sum((basis.d*length(cache.patterns[lower])*
                           length(cache.patterns[upper])
                           for (lower,upper) in keys(cache.transitions));init=0)
    @test cache.candidate_count<dense_candidates

    # Every cached sparse edge agrees with the direct exact-rational route,
    # including structural zeros omitted from retained storage.
    for ((lower,upper),_) in cache.transitions
        for mu in cache.patterns[lower],lam in cache.patterns[upper],
            local_label in 0:basis.d-1
            @test cgc(mu,local_label,lam;cache=cache)==
                  cgc(mu,local_label,lam;T=Float64)
        end
    end

    edge=first(keys(cache.transitions));lower,upper=edge
    table=cache.transitions[edge]
    lower_index=findfirst(i->table.offsets[i]<table.offsets[i+1],
                          1:length(cache.patterns[lower]))
    @test lower_index!==nothing
    upper_index,_,_=table.terms[table.offsets[lower_index]]
    Wm=cache.patterns[lower][lower_index]
    WL=cache.patterns[upper][upper_index]
    symbol=three_nu_symbol(WL,Wm,WL;cache=cache)
    @test symbol≈three_nu_symbol(WL,Wm,WL;T=Float64)
    symbol[1,1]+=1
    @test three_nu_symbol(WL,Wm,WL;cache=cache)≈
          three_nu_symbol(WL,Wm,WL;T=Float64)

    cache32=OneBoxCGCache(basis;max_depth=2,T=Float32)
    @test cgc(Wm,0,WL;cache=cache32) isa Float32
    @test eltype(three_nu_symbol(WL,Wm,WL;cache=cache32))===Float32
    @test (@inferred cgc(Wm,0,WL;cache=cache32)) isa Float32
    @test (@inferred cgc(Wm,0,WL;T=Float32,cache=cache32)) isa Float32
    @test eltype(@inferred three_nu_symbol(
        WL,Wm,WL;cache=cache32))===Float32
    @test eltype(@inferred three_nu_symbol(
        WL,Wm,WL;T=Float32,cache=cache32))===Float32
    @test_throws ArgumentError cgc(Wm,0,WL;T=Float64,cache=cache32)

    one_cached=OneBodyGeometry(basis,Float64;coefficient_cache=cache)
    one_direct=OneBodyGeometry(basis,Float64)
    local_matrix=ComplexF64[0 1 0;1 0 1;0 1 0]
    @test collective_operator(basis,local_matrix;cache=one_cached).data≈
          collective_operator(basis,local_matrix;cache=one_direct).data
    p_cached=PBodyGeometry(basis,2,Float64;coefficient_cache=cache)
    p_direct=PBodyGeometry(basis,2,Float64)
    @test keys(p_cached.isometries)==keys(p_direct.isometries)
    @test all(p_cached.isometries[path]≈p_direct.isometries[path]
              for path in keys(p_cached.isometries))

    shallow=OneBoxCGCache(basis;max_depth=0)
    @test isempty(shallow.transitions)
    @test_throws ArgumentError OneBodyGeometry(
        basis;coefficient_cache=shallow)
    depth_one=OneBoxCGCache(basis;max_depth=1)
    @test_throws ArgumentError PBodyGeometry(
        basis,2;coefficient_cache=depth_one)
    zero_basis=PIBasis(0,3)
    zero_cache=OneBoxCGCache(zero_basis)
    @test zero_cache.max_depth==0
    @test OneBoxCGCache(zero_basis,Float32).max_depth==0
    @test OneBodyGeometry(
        zero_basis;coefficient_cache=zero_cache).basis===zero_basis
    depth_two_edge=first(edge for edge in keys(cache.transitions)
                         if cache.depths[first(edge)]==2)
    deep_lower,deep_upper=depth_two_edge
    @test_throws ArgumentError cgc(first(cache.patterns[deep_lower]),0,
        first(cache.patterns[deep_upper]);cache=depth_one)

    other_basis=PIBasis(4,3)
    @test_throws ArgumentError OneBodyGeometry(
        other_basis;coefficient_cache=cache)
    @test_throws ArgumentError PBodyGeometry(
        other_basis,2;coefficient_cache=cache)
    @test_throws ArgumentError OneBodyGeometry(basis;coefficient_cache=:bad)
    @test_throws ArgumentError PBodyGeometry(basis,2;coefficient_cache=:bad)
    restricted=PIBasis(4,3;sectors=[(4,0,0)])
    restricted_cache=OneBoxCGCache(restricted;max_depth=1)
    foreign_upper=first(gt_patterns(Partition((3,1,0))))
    reachable_lower=first(gt_patterns(Partition((3,0,0))))
    @test_throws ArgumentError cgc(
        reachable_lower,0,foreign_upper;cache=restricted_cache)

    @test_throws ArgumentError OneBoxCGCache(basis;max_depth=-1)
    @test_throws ArgumentError OneBoxCGCache(basis;max_depth=basis.N+1)
    @test_throws ArgumentError OneBoxCGCache(
        basis;max_depth=1,memory_budget=0)
    @test OneBoxCGCache(
        basis;max_depth=1,memory_budget=Inf).max_depth==1

    # The structural memory envelope must cover dictionary and allocator
    # overhead through the largest automatic-cache qudit thresholds.
    for (large_N,large_d) in ((8,3),(5,4))
        guarded_basis=PIBasis(large_N,large_d)
        guarded_cache=OneBoxCGCache(
            guarded_basis;max_depth=large_N,memory_budget=Inf)
        @test Base.summarysize(guarded_cache)<=guarded_cache.estimated_bytes
    end

    big_cache=setprecision(BigFloat,128) do
        OneBoxCGCache(basis;max_depth=1,T=BigFloat)
    end
    big_edge=first(keys(big_cache.transitions));big_lower,big_upper=big_edge
    big_table=big_cache.transitions[big_edge]
    big_lower_index=findfirst(i->big_table.offsets[i]<big_table.offsets[i+1],
                              1:length(big_cache.patterns[big_lower]))
    big_upper_index,_,_=big_table.terms[big_table.offsets[big_lower_index]]
    big_mu=big_cache.patterns[big_lower][big_lower_index]
    big_lam=big_cache.patterns[big_upper][big_upper_index]
    setprecision(BigFloat,128) do
        @test cgc(big_mu,0,big_lam;cache=big_cache) isa BigFloat
    end
    setprecision(BigFloat,256) do
        @test_throws ArgumentError cgc(big_mu,0,big_lam;cache=big_cache)
        @test_throws ArgumentError OneBodyGeometry(
            basis,BigFloat;coefficient_cache=big_cache)
    end
    setprecision(BigFloat,128) do
        setrounding(BigFloat,RoundDown) do
            @test_throws ArgumentError cgc(
                big_mu,0,big_lam;cache=big_cache)
            @test_throws ArgumentError PBodyGeometry(
                basis,1,BigFloat;coefficient_cache=big_cache)
        end
    end
end
