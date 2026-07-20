function _magic_dicke_state(basis::PIBasis,
                            amplitudes::AbstractVector{<:Complex})
    N=basis.N
    length(amplitudes)==N+1||throw(DimensionMismatch(
        "a symmetric N-qubit vector needs N+1 Dicke amplitudes"))
    partition=Partition((N,0))
    patterns=basis.patterns[basis.index[partition]]
    ordered=zeros(eltype(amplitudes),N+1)
    for excitations in 0:N
        index=findfirst(pattern->content(pattern)==
            (N-excitations,excitations),patterns)
        index===nothing&&error(
            "symmetric sector lacks excitation number $excitations")
        ordered[index]=amplitudes[excitations+1]
    end
    sector_density_matrix(basis,partition,ordered*ordered')
end

# Explicitly bounded full-Hilbert oracle.  It exists only in the tests and is
# never called above N=4.  The vector uses the package's local order
# (|g>,|e>), with a Dicke amplitude divided uniformly among bit strings of the
# same excitation number.
function _magic_full_vector(N::Int,
                            amplitudes::AbstractVector{<:Complex})
    N<=4||throw(ArgumentError("the dense Pauli oracle is limited to N <= 4"))
    R=typeof(real(zero(eltype(amplitudes))))
    psi=zeros(eltype(amplitudes),1<<N)
    for word in 0:((1<<N)-1)
        excitations=count_ones(word)
        psi[word+1]=amplitudes[excitations+1]/
            sqrt(R(binomial(N,excitations)))
    end
    psi
end

function _magic_pauli_expectation(psi::AbstractVector{<:Complex},
                                  N::Int,pauli_word::Int)
    result=zero(eltype(psi))
    for column in 0:((1<<N)-1)
        row=column
        phase=one(eltype(psi))
        encoded=pauli_word
        for site in 0:N-1
            pauli=encoded&3
            bit=(column>>site)&1
            if pauli==1 # sigma_x
                row=xor(row,1<<site)
            elseif pauli==2 # sigma_y
                row=xor(row,1<<site)
                phase*=bit==0 ? im : -im
            elseif pauli==3 # sigma_z
                phase*=bit==0 ? 1 : -1
            end
            encoded>>=2
        end
        result+=conj(psi[row+1])*phase*psi[column+1]
    end
    result
end

function _magic_dense_stabilizer_renyi(amplitudes::AbstractVector{<:Complex})
    N=length(amplitudes)-1
    psi=_magic_full_vector(N,amplitudes)
    R=typeof(real(zero(eltype(psi))))
    fourth_moment=zero(R)
    for word in 0:(4^N-1)
        value=_magic_pauli_expectation(psi,N,word)
        abs(imag(value))<=R(200)*eps(R)||error(
            "Hermitian Pauli expectation acquired an imaginary component")
        fourth_moment+=real(value)^4
    end
    -log(fourth_moment/R(2)^N)
end

@testset "PI second stabilizer Renyi entropy" begin
    @testset "bounded full-Hilbert Pauli oracle" begin
        rng=MersenneTwister(220436)
        for N in 1:4
            basis=PIBasis(N,2)
            amplitudes=normalize(randn(rng,ComplexF64,N+1))
            rho=_magic_dicke_state(basis,amplitudes)
            expected=_magic_dense_stabilizer_renyi(amplitudes)
            @test stabilizer_renyi_entropy(rho)≈expected atol=2e-11 rtol=2e-11
        end
    end

    @testset "analytic stabilizer, product, GHZ, and W states" begin
        vacuum_basis=PIBasis(0,2;sectors=[(0,0)])
        vacuum=PIState(vacuum_basis;T=Float64)
        coefficient_block(vacuum,Partition((0,0)))[1,1]=1
        @test stabilizer_renyi_entropy(vacuum)==0.0

        for N in 1:5
            basis=PIBasis(N,2)
            ground=computational_product_state(basis,1)
            plus=iid_pure_state(basis,ComplexF64[1,1]/sqrt(2))
            @test stabilizer_renyi_entropy(ground)≈0 atol=3e-12
            @test stabilizer_renyi_entropy(plus)≈0 atol=3e-12

            t_state=iid_pure_state(
                basis,ComplexF64[1,cis(pi/4)]/sqrt(2))
            expected_t=N*log(4/3)
            @test isapprox(stabilizer_renyi_entropy(t_state),expected_t;
                atol=3e-11,rtol=3e-11)
            @test isapprox(stabilizer_renyi_entropy(t_state;base=2),
                N*log2(4/3);atol=3e-11,rtol=3e-11)

            @test stabilizer_renyi_entropy(
                ghz_state(basis;phase=0))≈0 atol=3e-11
            @test stabilizer_renyi_entropy(
                ghz_state(basis;phase=pi/2))≈0 atol=3e-11
            ghz_phase=pi/4
            expected_ghz=-log(1-sin(2ghz_phase)^2/4)
            @test isapprox(stabilizer_renyi_entropy(
                ghz_state(basis;phase=ghz_phase)),expected_ghz;
                atol=4e-11,rtol=4e-11)
        end

        for N in 2:5
            basis=PIBasis(N,2)
            w_state=dicke_state(basis,N/2,1-N/2)
            expected=log(N^3/(7N-6))
            @test isapprox(stabilizer_renyi_entropy(w_state),expected;
                atol=5e-11,rtol=5e-11)
        end
    end

    @testset "prepared reuse, restricted bases, and scalar precision" begin
        N=4
        basis=PIBasis(N,2)
        rho=ghz_state(basis;phase=pi/4)
        original=copy(rho.data)
        plan=StabilizerRenyiPlan(basis)
        workspace=StabilizerRenyiWorkspace(plan)
        direct=stabilizer_renyi_entropy(rho)
        @test stabilizer_renyi_entropy(
            rho;plan=plan,workspace=workspace)≈direct atol=3e-12 rtol=3e-12
        @test stabilizer_renyi_entropy(
            rho;plan=plan,workspace=workspace)≈direct atol=3e-12 rtol=3e-12
        @test rho.data==original

        restricted_basis=PIBasis(N,2;sectors=[(N,0)])
        restricted=ghz_state(restricted_basis;phase=pi/4)
        restricted_plan=StabilizerRenyiPlan(restricted_basis)
        restricted_workspace=StabilizerRenyiWorkspace(restricted_plan)
        @test isapprox(stabilizer_renyi_entropy(restricted;
            plan=restricted_plan,workspace=restricted_workspace),direct;
            atol=3e-12,rtol=3e-12)

        basis32=PIBasis(3,2)
        state32=iid_pure_state(
            basis32,ComplexF32[1,cis(Float32(pi/4))]/sqrt(2f0))
        plan32=StabilizerRenyiPlan(basis32;T=Float32)
        workspace32=StabilizerRenyiWorkspace(plan32)
        value32=stabilizer_renyi_entropy(
            state32;plan=plan32,workspace=workspace32)
        @test value32 isa Float32
        @test value32≈Float32(3*log(4/3)) atol=8f-5 rtol=8f-5

        setprecision(BigFloat,512) do
            basis_big=PIBasis(2,2;sectors=[(2,0)])
            state_big=iid_pure_state(basis_big,
                Complex{BigFloat}[inv(sqrt(big(2))),inv(sqrt(big(2)))])
            wide_plan=StabilizerRenyiPlan(basis_big;T=BigFloat)
            wide_workspace=StabilizerRenyiWorkspace(wide_plan)
            wide_value=stabilizer_renyi_entropy(state_big;
                plan=wide_plan,workspace=wide_workspace)
            @test wide_value isa BigFloat
            @test abs(wide_value)<=big"1e-130"
            narrow_plan=setprecision(BigFloat,256) do
                StabilizerRenyiPlan(basis_big;T=BigFloat)
            end
            narrow_workspace=StabilizerRenyiWorkspace(narrow_plan)
            @test_throws ArgumentError stabilizer_renyi_entropy(state_big;
                plan=narrow_plan,workspace=narrow_workspace)
        end

        @test (@doc StabilizerRenyiPlan)!==nothing
        @test (@doc StabilizerRenyiWorkspace)!==nothing
        @test (@doc stabilizer_renyi_entropy)!==nothing
    end

    @testset "strict domain and ownership validation" begin
        basis=PIBasis(3,2)
        rho=ghz_state(basis)
        plan=StabilizerRenyiPlan(basis)
        workspace=StabilizerRenyiWorkspace(plan)

        @test_throws ArgumentError stabilizer_renyi_entropy(
            maximally_mixed_state(basis))
        @test_throws ArgumentError stabilizer_renyi_entropy(
            PIState(basis,2rho.data))
        # The two-qubit singlet is pure and PI as a density operator, but its
        # vector is not in the fully symmetric sector assumed by the paper.
        @test_throws ArgumentError stabilizer_renyi_entropy(
            dicke_state(PIBasis(2,2),0,0))
        @test_throws ArgumentError stabilizer_renyi_entropy(
            iid_pure_state(PIBasis(2,3),ComplexF64[1,0,0]))

        equal_but_distinct=PIBasis(3,2)
        wrong_plan=StabilizerRenyiPlan(equal_but_distinct)
        wrong_workspace=StabilizerRenyiWorkspace(wrong_plan)
        @test_throws ArgumentError stabilizer_renyi_entropy(rho;plan=wrong_plan)
        @test_throws ArgumentError stabilizer_renyi_entropy(
            rho;plan=plan,workspace=wrong_workspace)

        workspace.busy[]=1
        @test_throws ArgumentError stabilizer_renyi_entropy(
            rho;plan=plan,workspace=workspace)
        workspace.busy[]=0

        @test_throws ArgumentError StabilizerRenyiPlan(basis;memory_budget=1)
        @test_throws ArgumentError StabilizerRenyiWorkspace(
            plan;memory_budget=1)
        @test_throws ArgumentError stabilizer_renyi_entropy(
            rho;memory_budget=1)
        @test_throws ArgumentError stabilizer_renyi_entropy(rho;base=1)
        @test_throws ArgumentError stabilizer_renyi_entropy(rho;base=0)
        @test_throws ArgumentError stabilizer_renyi_entropy(rho;base=Inf)
    end
end
