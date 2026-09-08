using PermutationalInvariantDynamics
using LinearAlgebra
using Random

# Run from the repository root: julia --project=. benchmark/no_jump_kernels.jl
# Reference kernels are benchmark-only, not alternate production solvers.
const PID = PermutationalInvariantDynamics

function measured(f; samples=12)
    f(); f()
    bytes=@allocated f()
    seconds=minimum([@elapsed f() for _ in 1:samples])
    (;seconds,bytes)
end

function scalar_norm_reference(basis,data)
    R=typeof(real(zero(eltype(data))));result=zero(R)
    for (sector,partition) in pairs(basis.sectors)
        for index in basis.offsets[sector]:basis.offsets[sector+1]-1
            value=PID._divide_by_schur_multiplicity_scale(data[index],R,partition)
            result=max(result,abs(value))
        end
    end
    result
end

function benchmark_case(N)
    basis=PIBasis(N,2);spin=spin_matrices(2)
    model=PIModel(basis,(
        LocalHamiltonian(0.23spin.jz),
        LocalJump(spin.jm;rate=0.31),LocalJump(spin.jp;rate=0.09)))
    plan=NoJumpIterativePlan(model)
    work=NoJumpResolventWorkspace(plan.no_jump)
    x=randn(MersenneTwister(0xb33f),ComplexF64,length(basis));y=similar(x)
    expected_norm=scalar_norm_reference(basis,x)
    @assert PID._no_jump_iterative_physical_maximum(plan,x)==expected_norm
    norm_reference=measured(()->scalar_norm_reference(basis,x))
    norm_prepared=measured(()->PID._no_jump_iterative_physical_maximum(plan,x))

    # Force the previous general Schur route on the very same blocks to
    # isolate the exact-diagonal specialization, excluding factorization.
    p=plan.no_jump
    general=PID.NoJumpResolventPlan(p.basis,p.generator_blocks,
        map(PID._no_jump_iterative_schur_factor,p.generator_blocks),
        p.Ttype,merge(p.metadata,(diagonal_sectors=0,)))
    general_work=NoJumpResolventWorkspace(general);reference=similar(x)
    no_jump_resolvent!(reference,general,x,0.4,general_work)
    no_jump_resolvent!(y,p,x,0.4,work)
    relative_error=norm(y-reference)/norm(reference)
    @assert relative_error<=1e-12
    action_reference=measured(()->no_jump_resolvent!(reference,general,x,0.4,general_work))
    action_diagonal=measured(()->no_jump_resolvent!(y,p,x,0.4,work))

    driven=PIModel(basis,(LocalHamiltonian(0.7spin.jx+0.23spin.jz),
        LocalJump(spin.jm;rate=0.31),LocalJump(spin.jp;rate=0.09)))
    driven_plan=NoJumpIterativePlan(driven)
    driven_work=NoJumpIterativeWorkspace(driven_plan;krylovdim=40,recycle_dim=8)
    solve_case()=no_jump_iterative_steady_state(driven_plan;
        workspace=driven_work,maxiter=1000,atol=1e-9,rtol=1e-7,return_info=true)
    result=solve_case()
    expected_z=((0.09-0.31)/0.4)*0.4*(0.2^2+0.23^2)/
        (2*(0.4*(0.2^2+0.23^2)+0.7^2*0.2))
    @assert result.converged&&result.state_diagnostics.valid
    @assert isapprox(real(collective_expectation(result.state,spin.jz))/N,
        expected_z;atol=5e-7,rtol=5e-7)
    solve_time=measured(solve_case;samples=3)
    (;N,dimension=length(basis),norm_reference,norm_prepared,
        action_reference,action_diagonal,relative_error,
        diagonal_scratch_bytes=sum(pair->sum(sizeof,pair),work.blocks),
        previous_diagonal_scratch_bytes=2length(basis)*sizeof(eltype(x)),
        stationary=solve_time,iterations=result.linear_solver.iterations,
        physical_residual=result.physical_residual_inf,trace_error=result.trace_error)
end

function blocked_sylvester_case(n,::Type{R},adjoint_action) where R
    rng=MersenneTwister(0x5a1+n)
    triangular=triu(randn(rng,Complex{R},n,n)*(R(0.05)/sqrt(R(n))))
    for i in 1:n
        triangular[i,i]=complex(-one(R)-R(i)/R(n),R(0.2))
    end
    rhs=randn(rng,Complex{R},n,n);output=similar(rhs)
    shift=Complex{R}(0.4,0.2)
    scalar! = adjoint_action ? PID._solve_nojump_adjoint_sector_scalar! :
                              PID._solve_nojump_sector_scalar!
    selected! = adjoint_action ? PID._solve_nojump_adjoint_sector! :
                                PID._solve_nojump_sector!
    reference=scalar!(copy(rhs),triangular,shift)
    scalar=measured(()->scalar!(copyto!(output,rhs),triangular,shift))
    selected=measured(()->selected!(copyto!(output,rhs),triangular,shift))
    A=adjoint_action ? adjoint(triangular) : triangular
    relative_error=norm(output-reference)/norm(reference)
    residual=norm(shift*output-A*output-output*adjoint(A)-rhs)/norm(rhs)
    @assert max(relative_error,residual)<=100eps(R)
    @assert selected.bytes<=4096
    (;operation=:triangular_sylvester,n,R,adjoint_action,
        blocked=PID._no_jump_iterative_uses_blocked_sylvester(output,triangular),
        scalar,selected,speedup=scalar.seconds/selected.seconds,
        relative_error,residual)
end

function symmetric_schur_reference!(y,plan,x,shift,work)
    factor=only(plan.factors);n=length(factor.values)
    left,right=only(work.blocks)
    X=reshape(x,n,n);Y=reshape(y,n,n)
    mul!(left,adjoint(factor.vectors),X)
    mul!(right,left,factor.vectors)
    PID._solve_nojump_sector_scalar!(right,factor.triangular,shift)
    mul!(left,factor.vectors,right)
    mul!(Y,left,adjoint(factor.vectors))
    y
end

function symmetric_resolvent_case(N)
    basis=PIBasis(N,2;sectors=[(N,0)]);spin=spin_matrices(2)
    model=PIModel(basis,(CollectiveHamiltonian(0.7spin.jx+0.23spin.jz),
        CollectiveJump(spin.jm;rate=0.31),CollectiveJump(spin.jp;rate=0.09)))
    plan=NoJumpResolventPlan(model)
    work=NoJumpResolventWorkspace(plan)
    x=randn(MersenneTwister(N),ComplexF64,length(basis));y=similar(x)
    reference=similar(x)
    scalar=measured(()->symmetric_schur_reference!(reference,plan,x,0.4,work))
    selected=measured(()->no_jump_resolvent!(y,plan,x,0.4,work))
    G=only(plan.generator_blocks);n=size(G,1)
    X=reshape(x,n,n);Y=reshape(y,n,n)
    relative_error=norm(y-reference)/norm(reference)
    residual=norm(0.4Y-G*Y-Y*adjoint(G)-X)/norm(X)
    @assert max(relative_error,residual)<=2e-11
    @assert selected.bytes<=4096
    (;operation=:symmetric_resolvent,N,n,dimension=length(basis),
        scalar,selected,speedup=scalar.seconds/selected.seconds,
        relative_error,residual)
end

function main()
    previous_threads=BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        println((julia=VERSION,cpu=Sys.CPU_NAME,threads=Threads.nthreads(),
            blas_threads=BLAS.get_num_threads(),samples=12,
            timing=:minimum_warmed_seconds,allocations=:bytes))
        for N in (4,16,32)
            println(benchmark_case(N))
        end
        for R in (Float64,Float32),n in (32,64,128,256),
                adjoint_action in (false,true)
            println(blocked_sylvester_case(n,R,adjoint_action))
        end
        for N in (32,64,128,256)
            println(symmetric_resolvent_case(N))
        end
    finally
        BLAS.set_num_threads(previous_threads)
    end
end

main()
