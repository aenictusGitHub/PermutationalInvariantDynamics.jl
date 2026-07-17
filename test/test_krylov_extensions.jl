@testset "Block, shifted, recycled, and exponential Krylov methods" begin
    rng=MersenneTwister(0x4b72796c)

    # Upper triangular coupling makes this deterministic reference genuinely
    # nonnormal while retaining a well-conditioned inverse.
    n=12
    A=Matrix(Diagonal(ComplexF64.(2 .+ rand(rng,n))))
    for i in 1:n-1
        A[i,i+1]=0.18+0.07im
    end

    @testset "block GMRES" begin
        B=randn(rng,ComplexF64,n,3)
        workspace=BlockGMRESWorkspace(ComplexF64,n,3,4)
        X=zeros(ComplexF64,n,3)
        result=block_gmres!(X,A,B,workspace;
            atol=1e-12,rtol=1e-10,maxiter=12)
        @test all(result.converged)
        @test result.iterations==4
        @test result.operator_applications>=result.iterations
        @test X≈A\B atol=2e-11 rtol=2e-11
        @test maximum(result.residuals)<2e-11

        # Duplicate and zero residuals exercise deterministic block deflation.
        b=randn(rng,ComplexF64,8)
        Adef=randn(rng,ComplexF64,8,8)+5I
        Bdef=hcat(b,b,zeros(ComplexF64,8))
        deflated=block_gmres(Adef,Bdef;block_krylovdim=8,
            atol=1e-12,rtol=1e-10,maxiter=16)
        @test all(deflated.converged)
        @test norm(Adef*deflated.solution-Bdef)<2e-11
        @test iszero(norm(deflated.solution[:,3]))

        # The low-level path also accepts a mutating callable operator.
        callable=(y,x)->mul!(y,A,x)
        Xcall=zeros(ComplexF64,n,3)
        called=block_gmres!(Xcall,callable,B,workspace;
            atol=1e-12,rtol=1e-10,maxiter=12)
        @test all(called.converged)
        @test Xcall≈X atol=2e-11

        # Explicit workspaces must not narrow a typed operator or
        # preconditioner before convergence is checked.
        narrow_block=BlockGMRESWorkspace(ComplexF32,n,3,4)
        @test_throws ArgumentError block_gmres!(zeros(ComplexF32,n,3),A,
            ComplexF32.(B),narrow_block)
        @test_throws ArgumentError block_gmres!(zeros(ComplexF32,n,3),
            ComplexF32.(A),ComplexF32.(B),narrow_block;preconditioner=lu(A))

        inconsistent=block_gmres(zeros(ComplexF64,2,2),
            reshape(ComplexF64[1,0],2,1);block_krylovdim=2,
            require_convergence=false)
        @test !only(inconsistent.converged)
        @test only(inconsistent.residuals)≈1

        # Callable fixed preconditioners follow the same mutating contract as
        # callable operators; they need not fabricate a matrix `size` method.
        identity_preconditioner=(y,x)->copyto!(y,x)
        fill!(Xcall,0)
        callable_preconditioned=block_gmres!(Xcall,callable,B,workspace;
            preconditioner=identity_preconditioner,
            atol=1e-12,rtol=1e-10,maxiter=12)
        @test all(callable_preconditioned.converged)
        @test Xcall≈X atol=2e-11
    end

    @testset "shared-Arnoldi shifted solves" begin
        b=randn(rng,ComplexF64,n)
        shifts=ComplexF64[-0.2,0.11im,0.3-0.04im]
        workspace=MultiShiftGMRESWorkspace(A,length(shifts),n)
        result=multishift_gmres(A,b,shifts;workspace,
            krylovdim=n,atol=1e-12,rtol=1e-10)
        @test all(result.converged)
        @test result.shared_arnoldi
        @test result.operator_applications<=n+length(shifts)
        for j in eachindex(shifts)
            @test result.solutions[:,j]≈(A-shifts[j]*I)\b atol=3e-11 rtol=3e-11
        end

        partial=multishift_gmres(A,b,shifts;krylovdim=2,
            atol=0,rtol=1e-14,require_convergence=false)
        @test !all(partial.converged)
        @test_throws ArgumentError multishift_gmres(A,b,shifts;
            krylovdim=2,atol=0,rtol=1e-14)
        nonzero=ones(ComplexF64,n,length(shifts))
        @test_throws ArgumentError multishift_gmres!(nonzero,A,b,shifts,workspace)
        @test_throws ArgumentError multishift_gmres(A,b,shifts;
            preconditioner=lu(A))
        narrow_shift=MultiShiftGMRESWorkspace(ComplexF32,n,length(shifts),n)
        @test_throws ArgumentError multishift_gmres!(
            zeros(ComplexF32,n,length(shifts)),A,ComplexF32.(b),
            ComplexF32.(shifts),narrow_shift)
        inconsistent=multishift_gmres(zeros(ComplexF64,2,2),
            ComplexF64[1,0],ComplexF64[0];krylovdim=2,
            require_convergence=false)
        @test !only(inconsistent.converged)
        @test only(inconsistent.residuals)≈1
        alias_parent=ComplexF64[0,0,1]
        @test_throws ArgumentError multishift_gmres!(
            reshape(view(alias_parent,1:2),2,1),Matrix{ComplexF64}(I,2,2),
            view(alias_parent,2:3),ComplexF64[0],
            MultiShiftGMRESWorkspace(ComplexF64,2,1,2))

        # A small but nonzero Arnoldi remainder is not a happy breakdown when
        # it remains larger than the requested solve tolerance.
        near_breakdown=ComplexF64[1 0;1e-9 2]
        tight=multishift_gmres(near_breakdown,ComplexF64[1,0],
            ComplexF64[0];krylovdim=2,atol=0,rtol=1e-12)
        @test only(tight.converged)
        @test only(tight.residuals)<=1e-12
    end

    @testset "recycled GCRO solves" begin
        workspace=RecycledGMRESWorkspace(ComplexF64,n,8,3)
        b1=randn(rng,ComplexF64,n);x1=zeros(ComplexF64,n)
        first=recycled_gmres!(x1,A,b1,workspace;
            atol=1e-12,rtol=1e-10,maxiter=80)
        @test first.converged
        @test !first.recycled_initially
        @test first.recycle_dimension==3
        @test norm(A*x1-b1)<=first.residual+20eps(Float64)

        # A nearby matrix is the continuation use case: C=A*U is rebuilt for
        # the new operator before the retained space is projected out.
        A2=copy(A);A2[1,1]+=0.015;A2[4,5]-=0.01im
        b2=randn(rng,ComplexF64,n);x2=zeros(ComplexF64,n)
        second=recycled_gmres!(x2,A2,b2,workspace;
            atol=1e-11,rtol=2e-9,maxiter=100)
        @test second.converged
        @test second.recycled_initially
        @test !second.recycle_reset
        @test second.recycle_dimension==3
        @test x2≈A2\b2 atol=2e-8 rtol=2e-8
        @test second.operator_applications>=second.iterations

        # The projected path also accepts a callable fixed preconditioner.
        callable_workspace=RecycledGMRESWorkspace(ComplexF64,n,n,2)
        callable_solution=zeros(ComplexF64,n)
        callable_preconditioner=(y,x)->copyto!(y,x)
        callable_result=recycled_gmres!(callable_solution,A,b1,
            callable_workspace;preconditioner=callable_preconditioner,
            atol=1e-12,rtol=1e-10,maxiter=n)
        @test callable_result.converged
        @test callable_solution≈A\b1 atol=2e-10 rtol=2e-10

        narrow_recycle=RecycledGMRESWorkspace(ComplexF32,n,8,3)
        @test_throws ArgumentError recycled_gmres!(zeros(ComplexF32,n),A,
            ComplexF32.(b1),narrow_recycle)
        @test_throws ArgumentError recycled_gmres!(zeros(ComplexF32,n),
            ComplexF32.(A),ComplexF32.(b1),narrow_recycle;
            preconditioner=lu(A))
        inconsistent=recycled_gmres(zeros(ComplexF64,2,2),
            ComplexF64[1,0];krylovdim=2,recycle_dim=0,
            require_convergence=false)
        @test !inconsistent.converged
        @test inconsistent.residual≈1
    end

    @testset "adaptive matrix-free exponential action" begin
        b=randn(rng,ComplexF64,n);t=0.4
        workspace=KrylovExpvWorkspace(ComplexF64,n,6)
        result=krylov_expv(A,b,t;workspace,krylovdim=6,
            atol=1e-11,rtol=1e-9)
        @test result.converged
        @test result.reached_time==t
        @test result.accepted_steps>0
        @test result.operator_applications>=result.accepted_steps
        @test result.value≈exp(t*A)*b atol=3e-9 rtol=3e-9

        partial=krylov_expv(A,b,10.0;krylovdim=2,max_steps=1,
            atol=0,rtol=1e-14,require_convergence=false)
        @test !partial.converged
        @test partial.reached_time<10
        @test_throws ArgumentError krylov_expv(A,b,10.0;krylovdim=2,
            max_steps=1,atol=0,rtol=1e-14)

        # A total time below eps(1) must not be mistaken for an absolute
        # unit-scale rounding remainder after the first accepted substep.
        tiny_operator=ComplexF64[1e20;;]
        tiny_time=krylov_expv(tiny_operator,ComplexF64[1],1e-20;
            krylovdim=1,maximum_step=1e-21,atol=1e-13,rtol=1e-13)
        @test tiny_time.converged
        @test tiny_time.accepted_steps>1
        @test tiny_time.value[1]≈exp(1) atol=2e-12 rtol=2e-12

        # A small nonzero Arnoldi remainder still carries the leading defect;
        # it is not an exact happy breakdown with zero error.
        near_breakdown_operator=ComplexF64[1 0;1e-9 0]
        near_breakdown=krylov_expv(near_breakdown_operator,
            ComplexF64[1,0],1.0;krylovdim=1,atol=1e-14,rtol=1e-14)
        @test near_breakdown.converged
        # The nominal one-vector space is enlarged and verified with a
        # second matvec; zero estimated error is then legitimate because the
        # enlarged space is the complete two-dimensional space.
        @test near_breakdown.operator_applications>=2
        @test near_breakdown.value≈
            exp(near_breakdown_operator)*ComplexF64[1,0] atol=2e-13 rtol=2e-13

        narrow_expv=KrylovExpvWorkspace(ComplexF32,n,6)
        @test_throws ArgumentError krylov_expv!(zeros(ComplexF32,n),A,
            ComplexF32.(b),0.4f0,narrow_expv)
    end

    @testset "Float32 storage and accuracy" begin
        n32=9
        A32=randn(rng,ComplexF32,n32,n32)+4f0I
        for i in 1:n32-1
            A32[i,i+1]+=0.15f0im
        end
        B32=randn(rng,ComplexF32,n32,2)
        block32=block_gmres(A32,B32;block_krylovdim=5,
            atol=2f-5,rtol=2f-4,maxiter=20)
        @test eltype(block32.solution)===ComplexF32
        @test eltype(block32.residuals)===Float32
        @test norm(A32*block32.solution-B32)<3f-4

        b32=randn(rng,ComplexF32,n32)
        @test_throws ArgumentError krylov_expv(A32,b32,0.2f0;
            minimum_step=-1e-300)
        @test_throws ArgumentError krylov_expv(A32,b32,0.2f0;
            safety=0.9999999999)
        @test_throws ArgumentError krylov_expv(A32,b32,0.2f0;
            initial_step=0.1)
        @test_throws ArgumentError krylov_expv(A32,b32,0.2f0;
            atol=-1e-300)
        shifts32=ComplexF32[-0.1f0,0.2f0im]
        shifted32=multishift_gmres(A32,b32,shifts32;krylovdim=n32,
            atol=2f-5,rtol=2f-4)
        @test eltype(shifted32.solutions)===ComplexF32
        @test eltype(shifted32.residuals)===Float32

        recycle32=RecycledGMRESWorkspace(ComplexF32,n32,n32,2)
        x32=zeros(ComplexF32,n32)
        recycled32=recycled_gmres!(x32,A32,b32,recycle32;
            atol=2f-5,rtol=2f-4,maxiter=n32)
        @test recycled32.converged
        @test eltype(recycled32.solution)===ComplexF32

        exp32=krylov_expv(A32,b32,0.2f0;krylovdim=5,
            atol=2f-5,rtol=2f-4)
        @test eltype(exp32.value)===ComplexF32
        @test exp32.estimated_error isa Float32
        @test exp32.value≈exp(0.2f0*A32)*b32 atol=5f-4 rtol=5f-4

        # Exactly representable integer times do not widen Float32 storage.
        integer_time=krylov_expv(A32,b32,1;krylovdim=5,
            atol=2f-5,rtol=2f-4)
        @test eltype(integer_time.value)===ComplexF32

        wide_workspace=KrylovExpvWorkspace(ComplexF64,n32,5)
        wide_result=krylov_expv(A32,b32,0.2f0;workspace=wide_workspace,
            atol=2e-7,rtol=2e-6)
        @test eltype(wide_result.value)===ComplexF64
        narrow_destination=zeros(ComplexF32,n32)
        @test_throws ArgumentError krylov_expv!(narrow_destination,A32,b32,
            0.2f0,wide_workspace)
    end
end
