"""Matrix representing `X -> A*X` under Julia column-major vectorization."""
function left_superoperator(A::AbstractMatrix)
    m,n=size(A);m==n||throw(DimensionMismatch("A must be square"))
    E=issparse(A) ? sparse(I,n,n) : Matrix{eltype(A)}(I,n,n)
    kron(E,A)
end

"""Matrix representing `X -> X*A` under Julia column-major vectorization."""
function right_superoperator(A::AbstractMatrix)
    m,n=size(A);m==n||throw(DimensionMismatch("A must be square"))
    E=issparse(A) ? sparse(I,n,n) : Matrix{eltype(A)}(I,n,n)
    kron(transpose(A),E)
end

"""Matrix representing `X -> A*X*B'` under column-major vectorization."""
function sandwich_superoperator(A::AbstractMatrix,B::AbstractMatrix=A)
    size(A,1)==size(A,2)||throw(DimensionMismatch("A must be square"));size(B)==size(A)||throw(DimensionMismatch("A and B must have equal square sizes"))
    kron(conj(B),A)
end

"""Vectorized generator `X -> -im*(H*X-X*H)`."""
commutator_superoperator(H::AbstractMatrix)=-im*(left_superoperator(H)-right_superoperator(H))

"""Vectorized Lindblad dissipator `D[L](X)`."""
function dissipator_superoperator(L::AbstractMatrix)
    Q=adjoint(L)*L
    sandwich_superoperator(L)-(left_superoperator(Q)+right_superoperator(Q))/2
end

"""
    is_pi_operator(A, N, d; atol=1e-12, rtol=sqrt(eps()))

Test whether a full Hilbert-space operator commutes with every particle
permutation. Adjacent transpositions generate the symmetric group, so only
`N-1` tensor-index permutations are tested and no permutation matrices are
constructed.
"""
function is_pi_operator(A::AbstractMatrix,N::Integer,d::Integer;atol::Real=1e-12,rtol::Real=sqrt(eps(Float64)))
    size(A,1)==size(A,2)||return false;_check_tensor_dimension(size(A,1),N,d)
    for k in 1:N-1
        p=_tensor_swap_permutation(N,d,k)
        isapprox(A,A[p,p];atol=atol,rtol=rtol)||return false
    end
    true
end
is_pi_operator(::AbstractPIOperator,args...;kwargs...)=true

"""
    is_pi_superoperator(S, N, d; atol=1e-12, rtol=sqrt(eps()))

Test permutation covariance of a full Liouville-space matrix `S`, namely
`S C_pi = C_pi S` with `C_pi vec(rho)=vec(P_pi*rho*P_pi')`. The test uses the
induced index permutations directly and follows the package's column-major
vectorization convention.
"""
function is_pi_superoperator(S::AbstractMatrix,N::Integer,d::Integer;atol::Real=1e-12,rtol::Real=sqrt(eps(Float64)))
    D=d^N;size(S)==(D^2,D^2)||throw(DimensionMismatch("superoperator must have size d^(2N) × d^(2N)"))
    for k in 1:N-1
        p=_tensor_swap_permutation(N,d,k);q=Vector{Int}(undef,D^2)
        for j in 1:D,i in 1:D;q[i+(j-1)*D]=p[i]+(p[j]-1)*D;end
        isapprox(S,S[q,q];atol=atol,rtol=rtol)||return false
    end
    true
end
is_pi_superoperator(::PIModel,args...;kwargs...)=true

"""Alias for `is_pi_operator`."""
is_permutationally_invariant(args...;kwargs...)=is_pi_operator(args...;kwargs...)
