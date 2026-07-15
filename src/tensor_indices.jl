# Small full-space index helpers used only by validation and by checking the
# permutation symmetry of a p-particle input operator. Production PI dynamics
# never constructs a d^N state or superoperator through these functions.
function _tensor_swap_permutation(N::Integer,d::Integer,k::Integer)
    1<=k<N||throw(ArgumentError("adjacent swap index must satisfy 1 ≤ k < N"));D=d^N;perm=Vector{Int}(undef,D)
    for q in 0:D-1
        a=(q÷d^(k-1))%d;b=(q÷d^k)%d
        perm[q+1]=q+1+(b-a)*d^(k-1)+(a-b)*d^k
    end
    perm
end

function _check_tensor_dimension(D,N,d)
    N>=1&&d>=1||throw(ArgumentError("N and d must be positive"));d^N==D||throw(DimensionMismatch("matrix dimension must equal d^N"))
end

