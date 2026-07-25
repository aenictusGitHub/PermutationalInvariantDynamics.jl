# Deliberately exponential, test-only helpers.  The largest oracle below is
# 2^3-by-2^3 (the qutrit case is 3^2-by-3^2), so these never exercise or
# authorize a production full-Hilbert route.
function _analysis_oracle_schur_reference(N::Int,d::Int)
    PID=PermutationalInvariantDynamics
    rows=Matrix{Float64}[]
    labels=NamedTuple[]
    offset=1
    for sector in partitions(N,d)
        paths=PID._removal_paths(sector,N)
        @assert length(paths)==symmetric_group_dimension(sector)
        for path in paths
            tensor=PID._path_isometry(path)
            block=reshape(tensor,size(tensor,1),d^N)
            range=offset:offset+size(block,1)-1
            push!(rows,block)
            push!(labels,(;sector,range))
            offset=last(range)+1
        end
    end
    (;transform=reduce(vcat,rows),labels)
end

function _analysis_oracle_dense(A)
    reference=_analysis_oracle_schur_reference(A.basis.N,A.basis.d)
    T=promote_type(eltype(A.data),Float64)
    schur=zeros(T,size(reference.transform))
    for label in reference.labels
        schur[label.range,label.range].=
            Matrix(physical_block(A,label.sector))
    end
    reference.transform'*schur*reference.transform
end

function _analysis_oracle_partial_trace(rho::AbstractMatrix,d::Int,N::Int,k::Int)
    da=d^k
    db=d^(N-k)
    reduced=zeros(eltype(rho),da,da)
    for a in 0:da-1,b in 0:da-1,q in 0:db-1
        reduced[a+1,b+1]+=rho[a*db+q+1,b*db+q+1]
    end
    reduced
end

function _analysis_oracle_partial_transpose(
        rho::AbstractMatrix,d::Int,N::Int,k::Int)
    da=d^k
    db=d^(N-k)
    transposed=similar(rho)
    for a in 0:da-1,b in 0:da-1,q in 0:db-1,r in 0:db-1
        transposed[a*db+q+1,b*db+r+1]=
            rho[b*db+q+1,a*db+r+1]
    end
    transposed
end

function _analysis_oracle_collective(X::AbstractMatrix,N::Int)
    d=size(X,1)
    identity=Matrix{eltype(X)}(I,d,d)
    sum(reduce(kron,
        [site==active ? X : identity for site in 1:N])
        for active in 1:N)
end

function _analysis_oracle_random_state(basis,rng)
    PID=PermutationalInvariantDynamics
    rho=PIState(basis;T=Float64)
    for partition in basis.sectors
        n=length(basis.patterns[basis.index[partition]])
        A=randn(rng,ComplexF64,n,n)
        coefficient_block(rho,partition).=
            A*A'+0.25*Matrix{ComplexF64}(I,n,n)
    end
    PID.normalize!(rho)
end

function _analysis_oracle_entropy(rho::AbstractMatrix)
    values=eigvals(Hermitian((rho+rho')/2))
    -sum(value*log2(value) for value in values if value>0)
end

function _analysis_oracle_fidelity(
        rho::AbstractMatrix,sigma::AbstractMatrix)
    er=eigen(Hermitian((rho+rho')/2))
    sqrt_rho=er.vectors*Diagonal(sqrt.(max.(er.values,0)))*er.vectors'
    kernel=sqrt_rho*sigma*sqrt_rho
    values=eigvals(Hermitian((kernel+kernel')/2))
    sum(sqrt.(max.(values,0)))^2
end

function _analysis_oracle_relative_entropy(
        rho::AbstractMatrix,sigma::AbstractMatrix)
    er=eigen(Hermitian((rho+rho')/2))
    es=eigen(Hermitian((sigma+sigma')/2))
    logrho=er.vectors*Diagonal(log.(er.values))*er.vectors'
    logsigma=es.vectors*Diagonal(log.(es.values))*es.vectors'
    real(tr(rho*(logrho-logsigma)))/log(2)
end

function _analysis_oracle_qfim(rho::AbstractMatrix,generators)
    E=eigen(Hermitian((rho+rho')/2))
    transformed=[E.vectors'*G*E.vectors for G in generators]
    n=length(generators)
    F=zeros(Float64,n,n)
    cutoff=64eps(Float64)*maximum(E.values)
    for a in eachindex(E.values),b in eachindex(E.values)
        denominator=E.values[a]+E.values[b]
        denominator>cutoff||continue
        weight=2*(E.values[a]-E.values[b])^2/denominator
        for mu in 1:n,nu in mu:n
            F[mu,nu]+=weight*real(
                transformed[mu][a,b]*transformed[nu][b,a])
        end
    end
    for mu in 1:n,nu in 1:mu-1
        F[mu,nu]=F[nu,mu]
    end
    F
end

@testset "bounded dense oracles for general PI analysis" begin
    rng=MersenneTwister(0x5a17e)
    for (N,d) in ((3,2),(2,3))
        basis=PIBasis(N,d)
        rho=_analysis_oracle_random_state(basis,rng)
        sigma=_analysis_oracle_random_state(basis,rng)
        dense_rho=_analysis_oracle_dense(rho)
        dense_sigma=_analysis_oracle_dense(sigma)

        @test tr(dense_rho)≈1 atol=2e-13
        @test minimum(eigvals(Hermitian(dense_rho)))>-2e-13
        @test purity(rho)≈real(tr(dense_rho*dense_rho)) atol=3e-12
        @test von_neumann_entropy(rho)≈
            _analysis_oracle_entropy(dense_rho) atol=4e-11
        @test renyi_entropy(rho,2)≈
            -log2(real(tr(dense_rho*dense_rho))) atol=4e-11
        @test trace_distance(rho,sigma)≈
            sum(abs,eigvals(Hermitian(dense_rho-dense_sigma)))/2 atol=4e-11
        @test fidelity(rho,sigma)≈
            _analysis_oracle_fidelity(dense_rho,dense_sigma) atol=8e-11
        @test quantum_relative_entropy(rho,sigma)≈
            _analysis_oracle_relative_entropy(dense_rho,dense_sigma) atol=8e-11

        dense_one=_analysis_oracle_partial_trace(dense_rho,d,N,1)
        @test one_body_rdm(rho)≈dense_one atol=5e-11

        for k in 1:N-1
            plan=ReductionPlan(basis,k)
            workspace=ReductionWorkspace(plan,rho;mode=:both)
            reduced=reduced_state(rho,k;plan,workspace)
            dense_reduced=_analysis_oracle_partial_trace(dense_rho,d,N,k)
            @test _analysis_oracle_dense(reduced)≈dense_reduced atol=8e-11
            @test reduced_purity(rho,k;plan,workspace)≈
                real(tr(dense_reduced*dense_reduced)) atol=8e-11

            dense_pt=_analysis_oracle_partial_transpose(dense_rho,d,N,k)
            dense_pt_values=eigvals(Hermitian((dense_pt+dense_pt')/2))
            dense_negativity=-sum(value for value in dense_pt_values
                                  if value<0)
            @test negativity(rho,k;plan,workspace)≈
                dense_negativity atol=8e-11
            @test logarithmic_negativity(rho,k;plan,workspace)≈
                log2(2dense_negativity+1) atol=8e-11

            compressed=partial_transpose_spectrum(rho,k;plan)
            expanded=Float64[]
            for block in compressed
                for _ in 1:Int(block.multiplicity)
                    append!(expanded,real.(block.eigenvalues))
                end
            end
            @test sort!(expanded)≈sort!(dense_pt_values) atol=8e-11
        end

        X=Matrix{ComplexF64}(Diagonal(range(-0.4,0.7;length=d)))
        Y=zeros(ComplexF64,d,d)
        for level in 1:d-1
            X[level,level+1]=X[level+1,level]=0.13level
            Y[level,level+1]=-0.17im*level
            Y[level+1,level]=conj(Y[level,level+1])
        end
        GX=_analysis_oracle_collective(X,N)
        GY=_analysis_oracle_collective(Y,N)
        mean_x=tr(dense_rho*GX)
        mean_y=tr(dense_rho*GY)
        variance_x=real(tr(dense_rho*GX*GX)-mean_x^2)
        covariance_xy=real(
            tr(dense_rho*(GX*GY+GY*GX)/2)-mean_x*mean_y)
        @test collective_expectation(rho,X)≈mean_x atol=4e-11
        @test collective_variance(rho,X)≈variance_x atol=4e-11
        @test collective_covariance(rho,X,Y)≈covariance_xy atol=4e-11

        dense_F=_analysis_oracle_qfim(dense_rho,(GX,GY))
        plan_x=CollectiveObservablePlan(basis,X)
        plan_y=CollectiveObservablePlan(basis,Y)
        @test qfim(rho,[plan_x,plan_y])≈dense_F atol=8e-11
        @test qfi(rho,collective_operator(plan_x))≈dense_F[1,1] atol=8e-11
    end
end

@testset "analysis workspace ownership and precondition failures" begin
    rng=MersenneTwister(0x0bad5eed)
    basis=PIBasis(3,2)
    rho=_analysis_oracle_random_state(basis,rng)
    source_before=copy(rho.data)

    plan=ReductionPlan(basis,1)
    first_work=ReductionWorkspace(plan,rho;mode=:both)
    second_work=ReductionWorkspace(plan,rho;mode=:both)
    @test first_work.product_block!==second_work.product_block
    @test first_work.product_tmp!==second_work.product_tmp
    @test all(pair->pair[1]!==pair[2],
        zip(first_work.reduced_blocks,second_work.reduced_blocks))

    first_output=PIState(plan.output_basis;T=Float64)
    second_output=PIState(plan.output_basis;T=Float64)
    reduced_state!(
        first_output,rho,plan,first_work;check=false)
    reduced_state!(
        second_output,rho,plan,second_work;check=false)
    @test first_output.data≈second_output.data atol=3e-12
    @test rho.data==source_before

    geometry=OneBodyGeometry(basis)
    one_work=OneBodyRDMWorkspace(geometry,rho)
    aliased_destination=reshape(view(rho.data,1:4),2,2)
    @test Base.mightalias(aliased_destination,rho.data)
    @test_throws ArgumentError one_body_rdm!(
        aliased_destination,rho,one_work;check=false)
    @test rho.data==source_before

    equivalent_basis=PIBasis(3,2)
    equivalent_state=PIState(equivalent_basis,copy(rho.data))
    @test_throws ArgumentError trace_distance(rho,equivalent_state)
    @test_throws ArgumentError reduced_state(
        equivalent_state,1;plan,workspace=first_work)

    other_plan=ReductionPlan(basis,1)
    other_work=ReductionWorkspace(other_plan,rho)
    @test_throws ArgumentError reduced_state(
        rho,1;plan,workspace=other_work)

    X=ComplexF64[0 1;1 0]
    Y=ComplexF64[1 0;0 -1]
    plan_x=CollectiveObservablePlan(basis,X)
    plan_y=CollectiveObservablePlan(basis,Y)
    @test_throws DimensionMismatch qfim(
        rho,[X,Y];plans=(plan_x,))
    @test_throws ArgumentError qfim(
        rho,[X,Y];cache=geometry,plans=(plan_x,plan_y))
    @test_throws ArgumentError qfim(
        rho,[X,collective_operator(plan_y)];plans=(nothing,plan_y))
end
