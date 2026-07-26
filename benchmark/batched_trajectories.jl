using BenchmarkTools
using LinearAlgebra
using PermutationalInvariantDynamics

const PID=PermutationalInvariantDynamics

function scalar_conditional_action!(Y,lambdas,X,trajectory_plan,workspaces,t)
    basis=trajectory_plan.model.basis
    tau=trajectory_plan.liouvillian.tracevec
    for column in axes(X,2)
        workspace=workspaces[column]
        PID._reset_effective_jump_cache!(workspace)
        lambdas[column]=PID._conditional_action_and_intensity!(
            view(Y,:,column),view(X,:,column),workspace,basis,t,nothing,tau)
    end
    Y
end

function main(;N=48,columns=16,symmetric=true)
    basis=symmetric ?
        PIBasis(N,2;sectors=[(N,0)]) :
        PIBasis(N,2)
    spin=spin_matrices()
    model=symmetric ?
        PIModel(basis,(
            CollectiveHamiltonian(spin.jx;rate=0.23),
            CollectiveJump(spin.jm;rate=0.08),
        )) :
        PIModel(basis,(
            LocalHamiltonian(spin.jx;rate=0.23),
            LocalJump(spin.jm;rate=0.71),
            CollectiveJump(spin.jm;rate=0.08),
        ))
    rho=iid_pure_state(basis,ComplexF64[0,1])
    trajectory_plan=TrajectoryPlan(model)
    plan=PID.BatchedConditionalPlan(trajectory_plan)
    batch_work=PID.BatchedConditionalWorkspace(plan,rho,columns)
    scalar_work=[
        TrajectoryWorkspace(trajectory_plan,rho;mode=:fixed)
        for _ in 1:columns
    ]
    X=repeat(reshape(rho.data,:,1),1,columns)
    Y=similar(X)
    lambdas=zeros(columns)

    PID.batched_conditional_action!(
        Y,lambdas,plan,X,0.2,nothing,batch_work)
    scalar_conditional_action!(
        Y,lambdas,X,trajectory_plan,scalar_work,0.2)

    batched=@benchmark PID.batched_conditional_action!(
        $Y,$lambdas,$plan,$X,0.2,nothing,$batch_work)
    repeated=@benchmark scalar_conditional_action!(
        $Y,$lambdas,$X,$trajectory_plan,$scalar_work,0.2)
    batched_time=median(batched).time
    repeated_time=median(repeated).time

    println("Matrix-RHS conditional trajectory benchmark")
    println("N=$N, PI coordinates=$(length(basis)), columns=$columns, " *
            "symmetric_sector=$symmetric")
    println("batched median:  $(batched_time/1e6) ms")
    println("repeated median: $(repeated_time/1e6) ms")
    println("speedup:         $(repeated_time/batched_time)x")
    println("batched memory:  $(median(batched).memory) bytes")
    println("repeated memory: $(median(repeated).memory) bytes")
    nothing
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
