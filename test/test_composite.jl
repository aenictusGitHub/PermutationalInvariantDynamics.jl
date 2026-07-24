struct _CountingCompositeDenseMatrix{T} <: DenseArray{T,2}
    data::Matrix{T}
    pointer_calls::Base.RefValue{Int}
end
Base.size(A::_CountingCompositeDenseMatrix)=size(A.data)
Base.getindex(A::_CountingCompositeDenseMatrix,i::Int,j::Int)=A.data[i,j]
Base.setindex!(A::_CountingCompositeDenseMatrix,value,i::Int,j::Int)=
    (A.data[i,j]=value)
Base.strides(A::_CountingCompositeDenseMatrix)=strides(A.data)
Base.copy(A::_CountingCompositeDenseMatrix)=
    _CountingCompositeDenseMatrix(copy(A.data),A.pointer_calls)
function Base.pointer(A::_CountingCompositeDenseMatrix)
    A.pointer_calls[]+=1
    pointer(A.data)
end
Base.unsafe_convert(::Type{Ptr{T}},
                    A::_CountingCompositeDenseMatrix{T}) where T=
    pointer(A)

@testset "Composite PI operator spaces" begin
    bpi=PIBasis(1,2)
    baux=FiniteOperatorBasis(2;label=:cavity)
    basis=CompositePIBasis(bpi,baux)
    @test basis.dimensions==(length(bpi),4)
    @test length(basis)==length(bpi)*4
    @test pi_dimension(basis)==length(basis)

    rho_pi=iid_state(bpi,ComplexF64[0.7 0.1im;-0.1im 0.3])
    rho_aux=ComplexF64[0.6 0.2;0.2 0.4]
    rho=composite_tensor_state(basis,rho_pi,rho_aux)
    # First factor is fastest in composite operator coordinates.
    @test rho.data==kron(vec(rho_aux),rho_pi.data)
    @test trace(rho)≈trace(rho_pi)*LinearAlgebra.tr(rho_aux)
    @test purity(rho)≈purity(rho_pi)*real(LinearAlgebra.tr(rho_aux*rho_aux))

    Xpi=collective_operator(bpi,ComplexF64[0 1;1 0])
    Zaux=ComplexF64[1 0;0 -1]
    observable=composite_tensor_operator(basis,Xpi,Zaux)
    @test observable.data==kron(vec(Zaux),Xpi.data)
    @test expectation(rho,observable)≈
        expectation(rho_pi,Xpi)*LinearAlgebra.tr(adjoint(Zaux)*rho_aux)

    identity=composite_identity_operator(basis)
    @test expectation(rho,identity)≈trace(rho)
    @test trace(identity)≈4

    @test_throws ArgumentError CompositePIBasis()
    @test_throws DimensionMismatch composite_tensor_state(basis,rho_pi)
    @test_throws DimensionMismatch composite_tensor_state(basis,rho_pi,zeros(3,3))

    @testset "factor reductions without full trace vectors" begin
        reduced_pi=PermutationalInvariantDynamics.composite_reduced_state(
            rho,1)
        reduced_aux=PermutationalInvariantDynamics.composite_reduced_state(
            rho,2)
        @test reduced_pi isa PIState
        @test reduced_pi.basis===bpi
        @test reduced_pi.data≈LinearAlgebra.tr(rho_aux).*rho_pi.data
        @test reduced_aux≈trace(rho_pi).*rho_aux
        @test trace(reduced_pi)≈trace(rho)
        @test LinearAlgebra.tr(reduced_aux)≈trace(rho)

        pi_destination=PIState(bpi)
        finite_destination=zeros(ComplexF64,2,2)
        @test PermutationalInvariantDynamics.composite_reduced_state!(
            pi_destination,rho,1)===pi_destination
        @test PermutationalInvariantDynamics.composite_reduced_state!(
            finite_destination,rho,2)===finite_destination
        @test pi_destination.data≈reduced_pi.data
        @test finite_destination≈reduced_aux

        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state(rho,true)
        @test_throws BoundsError PermutationalInvariantDynamics.
            composite_reduced_state(rho,3)
        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state!(PIState(PIBasis(1,2)),rho,1)
        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state!(pi_destination,rho,2)
        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state!(finite_destination,rho,1)
        @test_throws DimensionMismatch PermutationalInvariantDynamics.
            composite_reduced_state!(zeros(ComplexF64,3,3),rho,2)
        @test_throws ArgumentError PermutationalInvariantDynamics.
            _composite_reduced_state_data!(
                view(rho.data,1:length(bpi)),rho,1)

        # Multiplicity weights of a general PI factor are part of the
        # physical partial trace, even when that factor is the one removed.
        multi=PIBasis(2,2)
        multi_state=maximally_mixed_state(multi)
        multi_basis=CompositePIBasis(multi,baux)
        multi_product=composite_tensor_state(
            multi_basis,multi_state,rho_aux)
        @test PermutationalInvariantDynamics.composite_reduced_state(
            multi_product,1).data≈multi_state.data
        @test PermutationalInvariantDynamics.composite_reduced_state(
            multi_product,2)≈rho_aux

        third=FiniteOperatorBasis(2;label=:second_auxiliary)
        rho_third=ComplexF64[0.4 0.07im;-0.07im 0.6]
        three_basis=CompositePIBasis(multi,baux,third)
        three_product=composite_tensor_state(
            three_basis,multi_state,rho_aux,rho_third)
        @test PermutationalInvariantDynamics.composite_reduced_state(
            three_product,1).data≈multi_state.data
        @test PermutationalInvariantDynamics.composite_reduced_state(
            three_product,2)≈rho_aux
        @test PermutationalInvariantDynamics.composite_reduced_state(
            three_product,3)≈rho_third

        # Repeated contractions prepare the exact diagonal geometry once.
        system_plan=PermutationalInvariantDynamics.CompositeReductionPlan(
            three_product,1)
        auxiliary_plan=
            PermutationalInvariantDynamics.CompositeReductionPlan(
                three_product,2)
        third_plan=PermutationalInvariantDynamics.CompositeReductionPlan(
            three_product,3)
        @test system_plan.basis===three_basis
        @test system_plan.kept_basis===multi
        @test system_plan.kept_factor==1
        @test system_plan.direct_scales
        # Two diagonal choices in each finite factor.
        @test system_plan.estimates.traced_diagonal_count==4
        @test system_plan.exact_multiplicities==BigInt[1]
        @test auxiliary_plan.exact_multiplicities==
            symmetric_group_dimension.(multi.sectors)
        @test all(
            index->auxiliary_plan.exact_multiplicities[index] ===
                auxiliary_plan.prepared_scales[index].numerator,
            eachindex(auxiliary_plan.exact_multiplicities))
        @test PermutationalInvariantDynamics.composite_reduced_state(
            three_product,system_plan).data≈multi_state.data
        @test PermutationalInvariantDynamics.composite_reduced_state(
            three_product,auxiliary_plan)≈rho_aux
        @test PermutationalInvariantDynamics.composite_reduced_state(
            three_product,third_plan)≈rho_third

        prepared_pi=PIState(multi)
        prepared_aux=zeros(ComplexF64,2,2)
        PermutationalInvariantDynamics.composite_reduced_state!(
            prepared_pi,three_product,system_plan)
        PermutationalInvariantDynamics.composite_reduced_state!(
            prepared_aux,three_product,auxiliary_plan)
        @test @allocated(
            PermutationalInvariantDynamics.composite_reduced_state!(
                prepared_pi,three_product,system_plan))==0
        @test @allocated(
            PermutationalInvariantDynamics.composite_reduced_state!(
                prepared_aux,three_product,auxiliary_plan))==0
        @test prepared_pi.data≈multi_state.data
        @test prepared_aux≈rho_aux

        arbitrary_data=ComplexF64[
            sin(0.19index)+im*cos(0.13index)
            for index in 1:length(multi_basis)]
        arbitrary=CompositePIState(multi_basis,arbitrary_data)
        arbitrary_system=
            PermutationalInvariantDynamics.CompositeReductionPlan(
                arbitrary,1)
        arbitrary_auxiliary=
            PermutationalInvariantDynamics.CompositeReductionPlan(
                arbitrary,2)
        system_reference=reshape(
            arbitrary_data,length(multi),length(baux))*
            PermutationalInvariantDynamics._factor_trace_vector(
                baux,Float64)
        auxiliary_reference=Vector{ComplexF64}(undef,length(baux))
        system_trace=PermutationalInvariantDynamics._factor_trace_vector(
            multi,Float64)
        for coordinate in eachindex(auxiliary_reference)
            source=view(
                arbitrary_data,
                (coordinate-1)*length(multi)+1:
                    coordinate*length(multi))
            auxiliary_reference[coordinate]=dot(system_trace,source)
        end
        @test PermutationalInvariantDynamics.composite_reduced_state(
            arbitrary,arbitrary_system).data≈system_reference
        @test vec(PermutationalInvariantDynamics.composite_reduced_state(
            arbitrary,arbitrary_auxiliary))≈auxiliary_reference

        @test_throws ArgumentError PermutationalInvariantDynamics.
            CompositeReductionPlan(
                three_basis,1;memory_budget=1)
        @test_throws ArgumentError PermutationalInvariantDynamics.
            CompositeReductionPlan(
                three_product,1;T=Float32)
        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state!(
                PIState(multi;T=Float32),three_product,system_plan)
        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state!(
                prepared_pi,
                composite_tensor_state(
                    CompositePIBasis(multi,baux),
                    multi_state,rho_aux),
                system_plan)
        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state!(
                prepared_pi,three_product,auxiliary_plan)
        nonfinite=copy(three_product)
        nonfinite.data[1]=Inf+0im
        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state!(
                prepared_pi,nonfinite,system_plan)

        big_source,big_plan,big_finite_plan,big_destination,
        big_expected_pi,big_expected_finite=
        setprecision(BigFloat,192) do
            local_state=iid_state(
                multi,Complex{BigFloat}[
                    big"0.7" big"0.0";big"0.0" big"0.3"])
            finite_state=Complex{BigFloat}[
                big"0.6" big"0.0";big"0.0" big"0.4"]
            source=composite_tensor_state(
                multi_basis,local_state,finite_state)
            plan=PermutationalInvariantDynamics.CompositeReductionPlan(
                source,1)
            finite_plan=
                PermutationalInvariantDynamics.CompositeReductionPlan(
                    source,2)
            destination=PIState(multi;T=BigFloat)
            source,plan,finite_plan,destination,local_state,finite_state
        end
        setprecision(BigFloat,128) do
            PermutationalInvariantDynamics.composite_reduced_state!(
                big_destination,big_source,big_plan)
        end
        @test big_plan.precision_bits==192
        lower_ambient_plan=setprecision(BigFloat,64) do
            PermutationalInvariantDynamics.CompositeReductionPlan(
                big_source,1)
        end
        higher_ambient_plan=setprecision(BigFloat,256) do
            PermutationalInvariantDynamics.CompositeReductionPlan(
                big_source,1)
        end
        @test lower_ambient_plan.precision_bits==192
        @test higher_ambient_plan.precision_bits==192
        @test lower_ambient_plan.scales==big_plan.scales
        @test higher_ambient_plan.scales==big_plan.scales
        @test isapprox(
            trace(big_destination),trace(big_source);
            atol=big"1e-50",rtol=big"1e-50")
        wrong_precision=setprecision(BigFloat,128) do
            PIState(multi;T=BigFloat)
        end
        @test_throws ArgumentError PermutationalInvariantDynamics.
            composite_reduced_state!(
                wrong_precision,big_source,big_plan)
        mixed_precision=copy(big_source)
        mixed_precision.data[1]=setprecision(BigFloat,128) do
            Complex{BigFloat}(1,0)
        end
        @test_throws ArgumentError PermutationalInvariantDynamics.
            CompositeReductionPlan(mixed_precision,1)

        # Allocating plan and factor routes must create their destinations in
        # the source/plan context, not at either lower or higher ambient
        # precision. Cover both retained PI and retained finite factors.
        for ambient_precision in (64,256)
            pi_from_plan,finite_from_plan,pi_from_factor,
            finite_from_factor=setprecision(
                    BigFloat,ambient_precision) do
                (
                    PermutationalInvariantDynamics.
                        composite_reduced_state(big_source,big_plan),
                    PermutationalInvariantDynamics.
                        composite_reduced_state(
                            big_source,big_finite_plan),
                    PermutationalInvariantDynamics.
                        composite_reduced_state(big_source,1),
                    PermutationalInvariantDynamics.
                        composite_reduced_state(big_source,2),
                )
            end
            for values in (
                    pi_from_plan.data,finite_from_plan,
                    pi_from_factor.data,finite_from_factor)
                @test extrema(
                    value->max(
                        precision(real(value)),
                        precision(imag(value))),
                    values)==(192,192)
            end
            @test pi_from_plan.data≈big_expected_pi.data
            @test pi_from_factor.data≈big_expected_pi.data
            @test finite_from_plan≈big_expected_finite
            @test finite_from_factor≈big_expected_finite
        end
    end

    @testset "owned fresh coordinates and dense first-factor batch" begin
        # Public vector constructors retain their copy-on-input contract even
        # though package-owned freshly allocated coordinates can now transfer
        # ownership internally.
        input=fill(1.0+2.0im,length(basis))
        copied_state=CompositePIState(basis,input)
        copied_operator=CompositePIOperator(basis,input)
        input[1]=9.0-3.0im
        @test copied_state.data[1]==1.0+2.0im
        @test copied_operator.data[1]==1.0+2.0im

        allocation_pi=PIBasis(7,2)
        allocation_finite=FiniteOperatorBasis(10)
        allocation_basis=CompositePIBasis(allocation_pi,allocation_finite)
        allocation_rho=maximally_mixed_state(allocation_pi)
        allocation_aux=Matrix{ComplexF64}(I,10,10)/10
        CompositePIState(allocation_basis)
        CompositePIOperator(allocation_basis)
        composite_tensor_state(allocation_basis,allocation_rho,allocation_aux)
        payload=sizeof(ComplexF64)*length(allocation_basis)
        @test @allocated(CompositePIState(allocation_basis))<3payload÷2
        @test @allocated(CompositePIOperator(allocation_basis))<3payload÷2
        @test @allocated(composite_tensor_state(
            allocation_basis,allocation_rho,allocation_aux))<3payload÷2

        # A dense action on the contiguous first factor is one matrix-matrix
        # call over all tensor fibers, not one matrix-vector call per fiber.
        pointer_calls=Ref(0)
        dense_data=ComplexF64[0.2 0.3im 0 0;
                             -0.1im -0.4 0 0;
                             0 0 0.5 0.2;
                             0 0 -0.3 0.1]
        counted=_CountingCompositeDenseMatrix(dense_data,pointer_calls)
        batched=CompositeSuperoperator(
            basis,local_superoperator_term(basis,1,counted))
        batched_workspace=CompositeSuperoperatorWorkspace(batched)
        x=ComplexF64.(1:length(basis))./(length(basis)+1)
        y=similar(x)
        apply!(y,batched,x,0.0,nothing,batched_workspace)
        @test pointer_calls[]==1
        @test y≈kron(Matrix{ComplexF64}(I,4,4),dense_data)*x atol=2e-14
        pointer_calls[]=0
        @test @allocated(apply!(y,batched,x,0.0,nothing,
                                batched_workspace))==0
        @test pointer_calls[]==1
    end

    @testset "preallocated sum of Kronecker maps" begin
        A=ComplexF64[0.2 0.3im 0 0;
                     -0.1im -0.4 0 0;
                     0 0 0.5 0.2;
                     0 0 -0.3 0.1]
        B=ComplexF64[0.1 0.2 0 0;
                     0.3 -0.2 0 0;
                     0 0 0.4 -0.1im;
                     0 0 0.2im -0.5]
        C=ComplexF64[0.3 0 0 0;
                     0 -0.2 0.1 0;
                     0 0.4 0.5 0;
                     0 0 0 -0.1]
        local_term=local_superoperator_term(basis,1,A;coefficient=0.7)
        cross=factorized_superoperator_term(basis,1=>B,2=>C;
                                            coefficient=-0.25im)
        S=CompositeSuperoperator(basis,local_term,cross)
        @test_throws ArgumentError CompositeSuperoperator(
            basis,local_term;T=Float32)
        @test_throws ArgumentError CompositeSuperoperatorWorkspace(S;T=Float32)
        x=ComplexF64.(1:length(basis))./(length(basis)+1)
        y=similar(x)
        workspace=CompositeSuperoperatorWorkspace(S,x)
        apply!(y,S,x,0.0,nothing,workspace)
        reference=(0.7*kron(Matrix{ComplexF64}(I,4,4),A)-
                   0.25im*kron(C,B))*x
        @test y≈reference atol=2e-14 rtol=2e-14
        apply_adjoint!(y,S,x,0.0,nothing,workspace)
        @test y≈adjoint(0.7*kron(Matrix{ComplexF64}(I,4,4),A)-
                        0.25im*kron(C,B))*x atol=2e-14 rtol=2e-14
        @test @allocated(apply_adjoint!(
            y,S,x,0.0,nothing,workspace))==0
        @test S*x≈reference atol=2e-14 rtol=2e-14
        @test_throws ArgumentError apply!(x,S,x,0.0,nothing,workspace)
        @test_throws ArgumentError apply_adjoint!(
            x,S,x,0.0,nothing,workspace)

        wrapped=composite_matrixfree(S)
        @test wrapped.workspace isa CompositeSuperoperatorWorkspace
        @test wrapped*x≈reference atol=2e-14 rtol=2e-14
        @test adjoint(wrapped)*x≈
            adjoint(0.7*kron(Matrix{ComplexF64}(I,4,4),A)-
                    0.25im*kron(C,B))*x atol=2e-14 rtol=2e-14

        # A callable coefficient uses the explicit-time path and the same
        # preallocated tensor-mode storage.
        coefficient_calls=Ref(0)
        driven=factorized_superoperator_term(basis,2=>C;
            coefficient=(t,p)->(coefficient_calls[]+=1;p*t))
        Sd=CompositeSuperoperator(basis,driven)
        wd=CompositeSuperoperatorWorkspace(Sd,x)
        apply!(y,Sd,x,0.4,2.0,wd)
        @test y≈0.8*kron(C,Matrix{ComplexF64}(I,4,4))*x
        @test coefficient_calls[]==1
        apply_adjoint!(y,Sd,x,0.4,2.0,wd)
        @test y≈0.8adjoint(kron(C,Matrix{ComplexF64}(I,4,4)))*x
        @test coefficient_calls[]==2
        @test_throws ArgumentError apply_adjoint!(y,Sd,x,wd)
        @test !isautonomous(Sd)
    end

    @testset "fixed-capacity matrix right-hand sides" begin
        A=ComplexF64[0.2 0.3im 0 0;
                     -0.1im -0.4 0 0;
                     0 0 0.5 0.2;
                     0 0 -0.3 0.1]
        B=ComplexF64[0.1 0.2 0 0;
                     0.3 -0.2 0 0;
                     0 0 0.4 -0.1im;
                     0 0 0.2im -0.5]
        calls=Ref(0)
        counted=_CountingCompositeDenseMatrix(A,calls)
        S=CompositeSuperoperator(
            basis,
            local_superoperator_term(basis,1,counted;coefficient=0.7),
            local_superoperator_term(basis,2,B;coefficient=-0.25im))
        rng=MersenneTwister(9917)
        X=randn(rng,ComplexF64,length(basis),3)
        Y=similar(X)
        batch_work=CompositeSuperoperatorBatchWorkspace(
            S;capacity=4)
        apply!(Y,S,X,0.0,nothing,batch_work)
        reference=hcat((begin
            output=similar(view(X,:,column))
            apply!(output,S,view(X,:,column),0.0,nothing,
                   CompositeSuperoperatorWorkspace(S))
            output
        end for column in axes(X,2))...)
        @test Y≈reference atol=3e-14 rtol=3e-14
        # The contiguous first factor consumes every tensor fiber and every
        # right-hand side in one GEMM.
        @test calls[]==1+size(X,2)
        @test (@allocated apply!(
            Y,S,X,0.0,nothing,batch_work))<=1024

        apply_adjoint!(Y,S,X,0.0,nothing,batch_work)
        adjoint_reference=hcat((begin
            output=similar(view(X,:,column))
            apply_adjoint!(
                output,S,view(X,:,column),0.0,nothing,
                CompositeSuperoperatorWorkspace(S))
            output
        end for column in axes(X,2))...)
        @test Y≈adjoint_reference atol=4e-14 rtol=4e-14
        @test (@allocated apply_adjoint!(
            Y,S,X,0.0,nothing,batch_work))<=1024
        @test_throws ArgumentError apply!(
            X,S,X,0.0,nothing,batch_work)
        @test_throws ArgumentError apply_adjoint!(
            X,S,X,0.0,nothing,batch_work)
        oversized=randn(rng,ComplexF64,length(basis),5)
        @test_throws ArgumentError apply!(
            similar(oversized),S,oversized,0.0,nothing,batch_work)
        @test_throws ArgumentError apply_adjoint!(
            similar(oversized),S,oversized,0.0,nothing,batch_work)
        @test S*X≈reference atol=3e-14 rtol=3e-14

        PID=PermutationalInvariantDynamics
        vector_bytes=PID._performance_linear_operator_workspace_bytes(S)
        batch_bytes=PID._performance_linear_operator_workspace_bytes(
            S;batch_columns=4)
        @test batch_bytes==4vector_bytes
        @test batch_bytes>=
            2sizeof(ComplexF64)*length(basis)*batch_work.capacity
        @test_throws ArgumentError CompositeSuperoperatorBatchWorkspace(
            S;capacity=0)
        @test_throws ArgumentError CompositeSuperoperatorBatchWorkspace(
            S;capacity=4,T=Float32)

        # The plan-less compatibility wrapper still exposes the immutable
        # composite plan through its retained vector workspace. Prepared
        # consumers must recover a fresh batch workspace instead of invoking
        # the synchronized vector callback once per right-hand side.
        wrapper_calls=Ref(0)
        driven_term=local_superoperator_term(
            basis,2,B;coefficient=(t,p)->begin
                wrapper_calls[]+=1
                t*p
            end)
        driven_superoperator=CompositeSuperoperator(basis,driven_term)
        driven_wrapper=composite_matrixfree(driven_superoperator)
        wrapper_work=PID._linear_operator_batch_workspace(
            driven_wrapper,3,ComplexF64)
        @test wrapper_work isa CompositeSuperoperatorBatchWorkspace
        apply!(Y,driven_wrapper,X,0.4,1.5,wrapper_work)
        driven_reference=0.6kron(
            B,Matrix{ComplexF64}(I,4,4))*X
        @test Y≈driven_reference atol=3e-14 rtol=3e-14
        @test wrapper_calls[]==1
        apply_adjoint!(
            Y,driven_wrapper,X,0.4,1.5,wrapper_work)
        @test Y≈0.6adjoint(kron(
            B,Matrix{ComplexF64}(I,4,4)))*X atol=3e-14 rtol=3e-14
        @test wrapper_calls[]==2
        @test PID._performance_linear_operator_workspace_bytes(
            driven_wrapper;batch_columns=3)==
            PID._performance_linear_operator_workspace_bytes(
                driven_superoperator;batch_columns=3)

        # `sensitivity_problem` requests that same batch protocol. The dummy
        # one-particle, four-level PI state has the matching 16 coordinates;
        # only its storage shape is relevant to this augmented-RHS test.
        dummy_basis=PIBasis(1,4)
        dummy_state=PIState(dummy_basis,copy(view(X,:,1)))
        wrapper_calls[]=0
        problem=sensitivity_problem(
            driven_wrapper,dummy_state,(0.0,1.0),
            (zeros(ComplexF64,length(basis),length(basis)),);
            parameters=1.5)
        derivative=similar(problem.u0)
        problem.f(derivative,problem.u0,problem.p,0.4)
        @test wrapper_calls[]==1
        @test derivative[:,1]≈driven_reference[:,1] atol=3e-14 rtol=3e-14
        @test iszero(norm(view(derivative,:,2)))

        # A composite map may itself be used as one finite factor action. Its
        # nested vector and matrix workspaces must be prepared recursively,
        # rather than falling through to an allocation-heavy compatibility
        # call.
        inner_basis=CompositePIBasis(
            FiniteOperatorBasis(2;label=:inner_left),
            FiniteOperatorBasis(2;label=:inner_right))
        inner=CompositeSuperoperator(
            inner_basis,
            local_superoperator_term(inner_basis,1,A;coefficient=0.7),
            local_superoperator_term(inner_basis,2,B;coefficient=-0.25im))
        outer_basis=CompositePIBasis(
            FiniteOperatorBasis(4;label=:nested_composite))
        outer=CompositeSuperoperator(
            outer_basis,
            local_superoperator_term(outer_basis,1,inner))
        nested_x=randn(rng,ComplexF64,length(outer_basis))
        nested_y=similar(nested_x)
        nested_vector_work=CompositeSuperoperatorWorkspace(outer)
        inner_vector_work=CompositeSuperoperatorWorkspace(inner)
        nested_vector_reference=similar(nested_x)
        apply!(nested_y,outer,nested_x,nested_vector_work)
        apply!(nested_vector_reference,inner,nested_x,inner_vector_work)
        @test nested_y≈nested_vector_reference atol=4e-14 rtol=4e-14
        apply_adjoint!(nested_y,outer,nested_x,nested_vector_work)
        apply_adjoint!(
            nested_vector_reference,inner,nested_x,inner_vector_work)
        @test nested_y≈nested_vector_reference atol=4e-14 rtol=4e-14
        nested_X=randn(rng,ComplexF64,length(outer_basis),3)
        nested_Y=similar(nested_X)
        nested_work=CompositeSuperoperatorBatchWorkspace(
            outer;capacity=3)
        apply!(nested_Y,outer,nested_X,nested_work)
        inner_work=CompositeSuperoperatorBatchWorkspace(inner;capacity=3)
        nested_reference=similar(nested_X)
        apply!(nested_reference,inner,nested_X,inner_work)
        @test nested_Y≈nested_reference atol=4e-14 rtol=4e-14
        apply_adjoint!(nested_Y,outer,nested_X,nested_work)
        apply_adjoint!(nested_reference,inner,nested_X,inner_work)
        @test nested_Y≈nested_reference atol=4e-14 rtol=4e-14
    end

    @testset "PI Liouvillian lift" begin
        sm=ComplexF64[0 1;0 0]
        model=PIModel(bpi,(LocalJump(sm;rate=0.35),))
        compiled=compile(model;backend=:matrixfree)
        lifted=local_superoperator_term(basis,1,compiled)
        S=CompositeSuperoperator(basis,lifted)
        @test_throws ArgumentError CompositeSuperoperatorWorkspace(S;T=BigFloat)
        x=copy(rho.data);y=similar(x)
        workspace=CompositeSuperoperatorWorkspace(S,x)
        apply!(y,S,x,0.0,nothing,workspace)
        Lpi=Matrix(liouvillian(model;representation=:sparse))
        @test y≈kron(Matrix{ComplexF64}(I,4,4),Lpi)*x atol=3e-13
        apply_adjoint!(y,S,x,0.0,nothing,workspace)
        @test y≈adjoint(kron(
            Matrix{ComplexF64}(I,4,4),Lpi))*x atol=3e-13
        @test @allocated(apply_adjoint!(
            y,S,x,0.0,nothing,workspace))==0

        # Heterogeneous tuples previously fell back to runtime indexing and
        # allocated despite every numerical kernel being preallocated.
        cross=factorized_superoperator_term(
            basis,
            1=>factor_left_superoperator(
                bpi,collective_operator(bpi,ComplexF64[0 1;0 0])),
            2=>left_superoperator(ComplexF64[0 1;0 0]);
            coefficient=0.13,
        )
        combined=CompositeSuperoperator(basis,lifted,cross)
        combined_workspace=CompositeSuperoperatorWorkspace(combined,x)
        apply!(y,combined,x,0.0,nothing,combined_workspace)
        @test @allocated(apply!(y,combined,x,0.0,nothing,
                                combined_workspace))==0
    end

    @testset "cross-factor Hamiltonian and jump" begin
        # N=1 makes the PI factor physically identical to a two-level matrix,
        # while still exercising compressed PI-coordinate lifts.
        sx=ComplexF64[0 1;1 0]
        sm=ComplexF64[0 1;0 0]
        A=collective_operator(bpi,sx)
        H=composite_hamiltonian_superoperator(basis,1=>A,2=>sx;rate=0.3)
        D=composite_dissipator_superoperator(basis,1=>collective_operator(bpi,sm),
                                             2=>sm;rate=0.2)
        x=copy(rho.data);yh=similar(x);yd=similar(x)
        apply!(yh,H,x,0.0,nothing,CompositeSuperoperatorWorkspace(H,x))
        apply!(yd,D,x,0.0,nothing,CompositeSuperoperatorWorkspace(D,x))

        LA=factor_left_superoperator(bpi,A)
        RA=factor_right_superoperator(bpi,A)
        LB=left_superoperator(sx);RB=right_superoperator(sx)
        Href=-0.3im*(kron(LB,LA)-kron(RB,RA))
        @test yh≈Href*x atol=3e-13

        Jpi=collective_operator(bpi,sm)
        gain=kron(sandwich_superoperator(sm),
                  factor_sandwich_superoperator(bpi,Jpi))
        Qpi=adjoint(Jpi)*Jpi;Qaux=adjoint(sm)*sm
        Dref=0.2*(gain-
            (kron(left_superoperator(Qaux),factor_left_superoperator(bpi,Qpi))+
             kron(right_superoperator(Qaux),factor_right_superoperator(bpi,Qpi)))/2)
        @test yd≈Dref*x atol=3e-13
        apply_adjoint!(
            yh,H,x,0.0,nothing,CompositeSuperoperatorWorkspace(H,x))
        apply_adjoint!(
            yd,D,x,0.0,nothing,CompositeSuperoperatorWorkspace(D,x))
        @test yh≈adjoint(Href)*x atol=3e-13
        @test yd≈adjoint(Dref)*x atol=3e-13
        @test_throws ArgumentError composite_hamiltonian_superoperator(
            basis,1=>collective_operator(bpi,sm),2=>sx)
        @test_throws ArgumentError composite_hamiltonian_superoperator(
            basis,1=>A,2=>sx;rate=1+im)
        @test_throws ArgumentError composite_dissipator_superoperator(
            basis,1=>Jpi;rate=1+im)

        # The generic RK4 layer discovers the composite workspace and retains
        # trace under a trace-preserving Hamiltonian-plus-jump generator.
        generator=H+D
        evolved=copy(x)
        evolution_workspace=EvolutionWorkspace(generator,x)
        evolve!(evolved,generator,x,(0.0,0.2);steps=16,
                workspace=evolution_workspace)
        @test trace(CompositePIState(basis,evolved))≈trace(rho) atol=2e-12
        evolved_state=time_evolve(generator,rho,(0.0,0.2);steps=16)
        @test evolved_state.data≈evolved atol=2e-14
        @test time_evolve(rho,generator,(0.0,0.2);steps=16).data≈
            evolved atol=2e-14
        sampled=time_evolution(generator,rho,[0.0,0.1,0.2];
                               steps_per_interval=8)
        @test sampled[end].data≈evolved atol=2e-14
        @test time_evolution(rho,generator,[0.0,0.1,0.2];
            steps_per_interval=8)[end].data≈evolved atol=2e-14
        @test pi_dimension(evolved_state)==length(basis)
        @test_throws ArgumentError CompositeSuperoperator(
            basis,first(generator.terms);T=Complex{Int})
    end


    @testset "multiple compressed PI factors" begin
        two_pi=CompositePIBasis(bpi,bpi)
        product=composite_tensor_state(two_pi,rho_pi,rho_pi)
        @test product.data==kron(rho_pi.data,rho_pi.data)
        @test trace(product)≈trace(rho_pi)^2
        localmap=Matrix(liouvillian(PIModel(bpi,(LocalJump(
            ComplexF64[0 1;0 0];rate=0.1),));representation=:sparse))
        term=local_superoperator_term(two_pi,2,localmap)
        S=CompositeSuperoperator(two_pi,term)
        @test S*product.data≈kron(localmap,
                                      Matrix{ComplexF64}(I,length(bpi),length(bpi)))*product.data

        other_basis=PIBasis(1,2)
        other_model=compile(PIModel(other_basis,(LocalJump(
            ComplexF64[0 1;0 0];rate=0.1),));backend=:matrixfree)
        @test_throws ArgumentError local_superoperator_term(two_pi,1,other_model)

        S2=CompositeSuperoperator(two_pi,
            local_superoperator_term(two_pi,1,localmap))
        @test_throws ArgumentError apply!(similar(product.data),S2,product.data,
            0.0,nothing,CompositeSuperoperatorWorkspace(S,product.data))
    end

    @testset "multi-sector PI factor" begin
        multi=PIBasis(2,2)
        @test length(multi.sectors)==2
        finite=FiniteOperatorBasis(2;label=:ancilla)
        composite=CompositePIBasis(multi,finite)
        local_density=ComplexF64[0.65 0.08im;-0.08im 0.35]
        multi_state=iid_state(multi,local_density)
        auxiliary_state=ComplexF64[0.7 0.1;0.1 0.3]
        state=composite_tensor_state(composite,multi_state,auxiliary_state)
        @test trace(state)≈trace(multi_state)*LinearAlgebra.tr(auxiliary_state)

        identity=composite_identity_operator(composite)
        @test expectation(state,identity)≈trace(state)
        @test trace(identity)≈8

        sm=ComplexF64[0 1;0 0]
        model=PIModel(multi,(LocalJump(sm;rate=0.17),))
        compiled=compile(model;backend=:matrixfree)
        lifted=CompositeSuperoperator(composite,
            local_superoperator_term(composite,1,compiled))
        destination=similar(state.data)
        workspace=CompositeSuperoperatorWorkspace(lifted,state.data)
        apply!(destination,lifted,state.data,0.0,nothing,workspace)
        dense_local=Matrix(liouvillian(model;representation=:sparse))
        reference=kron(Matrix{ComplexF64}(I,length(finite),length(finite)),
                       dense_local)*state.data
        @test destination≈reference atol=5e-13 rtol=5e-13
        @test abs(dot(composite_trace_vector(composite),destination))<5e-13
    end

    ambiguities=Test.detect_ambiguities(PermutationalInvariantDynamics;
                                        recursive=true)
    @test !any(pair->occursin("CompositePIState",sprint(show,pair)),ambiguities)
end
