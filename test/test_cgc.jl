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
