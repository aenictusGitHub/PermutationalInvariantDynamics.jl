# Deliberately exponential, test-only Schur oracle. Each standard Young path
# from the empty partition to lambda supplies one multiplicity copy, while the
# one-box CG recurrence supplies its U(d) rows. Production code must never call
# this helper.
function _dense_schur_reference(N::Int,d::Int)
    PID=PermutationalInvariantDynamics
    pieces=Matrix{Float64}[]
    labels=NamedTuple[]
    offset=1
    for sector in partitions(N,d)
        paths=PID._removal_paths(sector,N)
        @assert length(paths)==symmetric_group_dimension(sector)
        for (multiplicity,path) in pairs(paths)
            tensor=PID._path_isometry(path)
            block=reshape(tensor,size(tensor,1),d^N)
            range=offset:offset+size(block,1)-1
            push!(pieces,block)
            push!(labels,(;sector,multiplicity,range,path))
            offset=last(range)+1
        end
    end
    transform=reduce(vcat,pieces)
    (;transform,labels)
end

function _dense_tensor_power(A,N)
    N==0&&return reshape(one(eltype(A)),1,1)
    reduce(kron,fill(A,N))
end

function _dense_collective_sum(X,N,d)
    identity=Matrix{eltype(X)}(I,d,d)
    sum(reduce(kron,[site==active ? X : identity for site in 1:N])
        for active in 1:N)
end

@testset "test-only recursive dense Schur reference" begin
    for (N,d,local_state) in (
        (3,2,ComplexF64[0.65 0.1im;-0.1im 0.35]),
        (3,3,ComplexF64[0.6 0.1 0;0.1 0.4 0;0 0 0]),
    )
        reference=_dense_schur_reference(N,d);S=reference.transform
        @test size(S)==(d^N,d^N)
        @test S*S'≈I atol=2e-13 rtol=2e-13

        full_state=_dense_tensor_power(local_state,N)
        transformed=S*full_state*S'
        pi_state=iid_state(PIBasis(N,d),local_state)
        off_block=copy(transformed)
        for label in reference.labels
            expected=Matrix(physical_block(pi_state,label.sector))
            @test transformed[label.range,label.range]≈expected atol=3e-13 rtol=3e-13
            off_block[label.range,label.range].=0
        end
        @test norm(off_block)<5e-13

        offdiagonal=zeros(ComplexF64,d,d)
        for level in 1:d-1
            offdiagonal[level,level+1]=0.2+0.1im*level
            offdiagonal[level+1,level]=conj(offdiagonal[level,level+1])
        end
        for X in (Matrix{ComplexF64}(Diagonal(collect(0:d-1))),offdiagonal)
            full_observable=_dense_collective_sum(X,N,d)
            transformed_observable=S*full_observable*S'
            pi_observable=collective_operator(pi_state.basis,X)
            for label in reference.labels
                @test transformed_observable[label.range,label.range]≈
                      physical_block(pi_observable,label.sector) atol=3e-13 rtol=3e-13
            end
        end
    end
end
