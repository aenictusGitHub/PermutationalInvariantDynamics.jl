@testset "Evans uniqueness certificates" begin
    sm=ComplexF64[0 1;0 0];sz=ComplexF64[1 0;0 -1];sx=ComplexF64[0 1;1 0]
    lowering=evans_uniqueness(zeros(2,2),[sm];return_basis=true)
    @test lowering.unique
    @test lowering.commutant_dimension==1
    @test size(lowering.commutant_basis)==(4,1)
    dephasing=evans_uniqueness(zeros(2,2),[sz])
    @test !dephasing.unique
    @test dephasing.commutant_dimension==2

    b=PIBasis(4,2)
    local_decay=PIModel(b,[LocalJump(sm)])
    @test has_unique_steady_state_evans(local_decay)===true
    @test evans_uniqueness(local_decay).scope==:full_hilbert_space

    collective_decay=PIModel(b,[CollectiveJump(sm)])
    cr=evans_uniqueness(collective_decay)
    @test cr.unique===false
    @test cr.criterion==:schur_sector_conservation

    sym=Partition((4,0));bs=PIBasis(4,2;sectors=[sym.parts])
    restricted=evans_uniqueness(PIModel(bs,[CollectiveJump(sm)]))
    @test restricted.unique
    @test restricted.scope==:retained_schur_sector

    combined=evans_uniqueness(PIModel(b,[LocalJump(sz),CollectiveHamiltonian(sx)]))
    @test combined.unique===true
    @test combined.certified
    @test combined.criterion==:evans_auxiliary_pi
    @test combined.auxiliary_local_dimension==4
    @test combined.commutant_dimension==1
    @test combined.estimated_geometry_bytes>0
    @test combined.estimated_geometry_bytes<=combined.estimated_bytes

    budgeted=evans_uniqueness(PIModel(b,[LocalJump(sz),CollectiveHamiltonian(sx)]);
                              memory_budget=1)
    @test budgeted.unique===missing
    @test !budgeted.certified
    @test budgeted.estimated_bytes>budgeted.memory_budget

    # The budget includes retained Appendix-D path isometries, not just the
    # auxiliary PI coordinates and sector eigensolver matrices.
    triple=kron(sz,kron(sz,sz))
    geometry_budgeted=evans_uniqueness(PIModel(PIBasis(5,2),[
        LocalJump(sz),LocalPBodyJump(triple,3)]);memory_budget=3_000_000)
    @test geometry_budgeted.unique===missing
    @test geometry_budgeted.estimated_geometry_bytes>3_000_000

    direct=DirectPIHamiltonian(collective_operator(b,sx))
    unsupported=evans_uniqueness(PIModel(b,[LocalJump(sz),direct]))
    @test unsupported.unique===missing
    @test unsupported.criterion==:unsupported_microscopic_recoupling
    @test_throws ArgumentError evans_uniqueness(PIModel(b,[LocalJump(sm;rate=(t,p)->1)]))
    @test_throws ArgumentError evans_uniqueness(PIModel(b,[LocalJump(sm;rate=-1)]))
    @test_throws ArgumentError evans_uniqueness(local_decay;atol=-1)
end

@testset "Auxiliary-Schur Evans/Davies algebra" begin
    PID=PermutationalInvariantDynamics

    function subsets_of_size(N,p)
        out=Vector{Vector{Int}}()
        function visit(next,left,current)
            if left==0
                push!(out,copy(current));return
            end
            for site in next:N-left+1
                push!(current,site);visit(site+1,left-1,current);pop!(current)
            end
        end
        visit(1,p,Int[]);out
    end

    # Particle 1 is the fastest tensor index, as in Appendix D.
    function embed_subset(A,subset,N,d)
        dimension=d^N;embedded=zeros(ComplexF64,dimension,dimension)
        for row in 0:dimension-1,column in 0:dimension-1
            rowdigits=[(row÷d^(site-1))%d for site in 1:N]
            coldigits=[(column÷d^(site-1))%d for site in 1:N]
            all(site->site in subset||rowdigits[site]==coldigits[site],1:N)||continue
            localrow=1+sum(rowdigits[subset[index]]*d^(index-1) for index in eachindex(subset))
            localcolumn=1+sum(coldigits[subset[index]]*d^(index-1) for index in eachindex(subset))
            embedded[row+1,column+1]=A[localrow,localcolumn]
        end
        embedded
    end

    function explicit_model_evans(model)
        b=model.basis;dimension=b.d^b.N
        H=zeros(ComplexF64,dimension,dimension);jumps=Matrix{ComplexF64}[]
        for term in model.terms
            p=PID.body_order(term);subsets=subsets_of_size(b.N,p)
            embedded=[embed_subset(PID.term_operator(term),subset,b.N,b.d) for subset in subsets]
            if PID.term_process(term) isa Val{:hamiltonian}
                H .+=(PID.term_rate(term)/PID.term_hbar(term)).*sum(embedded)
            elseif PID.term_scope(term) isa Val{:local}
                append!(jumps,[sqrt(PID.term_rate(term)).*operator for operator in embedded])
            else
                push!(jumps,sqrt(PID.term_rate(term)).*sum(embedded))
            end
        end
        evans_uniqueness(H,jumps)
    end

    # Verify the ket/bra-to-local-pair permutation independently for p=1,2.
    for (d,p) in ((2,1),(2,2),(3,1))
        n=d^p;A=reshape(ComplexF64.(1:n^2),n,n)
        conventional=left_superoperator(A)-right_superoperator(A)
        new_of_old=vec(PID._evans_regroup_indices(d,p))
        old_of_new=invperm(new_of_old)
        @test PID._evans_regrouped_commutator(A,p,d)≈
              conventional[old_of_new,old_of_new]
    end

    sm=ComplexF64[0 1;0 0];sx=ComplexF64[0 1;1 0];sz=ComplexF64[1 0;0 -1]
    models=[
        # A nontrivial local commutant becomes scalar only after including H.
        PIModel(PIBasis(2,2),[LocalJump(sz),CollectiveHamiltonian(sx)]),
        # The conserved computational-basis algebra has dimension 2^N.
        PIModel(PIBasis(3,2),[LocalJump(sz)]),
        # Pair-local and collective constraints exercise Appendix-D blocks.
        PIModel(PIBasis(3,2),[
            LocalPBodyJump(kron(sm,sm),2),CollectiveHamiltonian(sx),
            CollectiveJump(sm;rate=0.3)]),
        # Pair Hamiltonians contribute through their summed microscopic
        # commutator before the positive Evans normal equation is formed.
        PIModel(PIBasis(3,2),[
            PBodyHamiltonian(kron(sz,sz),2;rate=0.4),LocalJump(sz)]),
        # A coherent collective pair channel must be treated as one summed
        # jump, not as independent subset jumps. The local dephasing term also
        # prevents the collective-only early certificate from bypassing the
        # auxiliary-Schur construction under test.
        PIModel(PIBasis(3,2),[
            CollectivePBodyJump(kron(sm,sm),2;rate=0.2),
            LocalJump(sz),CollectiveHamiltonian(sx)]),
    ]
    for model in models
        compressed=evans_uniqueness(model)
        explicit=explicit_model_evans(model)
        @test compressed.unique==explicit.unique
        @test compressed.commutant_dimension==explicit.commutant_dimension
        @test compressed.estimated_geometry_bytes>0
    end
    @test evans_uniqueness(models[2]).commutant_dimension==8

    # Qutrit sanity check: diagonal local noise plus a connected transverse
    # Hamiltonian has a scalar global commutant for N=2.
    diagonal=ComplexF64[1 0 0;0 0 0;0 0 -1]
    connected=ComplexF64[0 1 0;1 0 1;0 1 0]
    qutrit=PIModel(PIBasis(2,3),[
        LocalJump(diagonal),CollectiveHamiltonian(connected)])
    qreport=evans_uniqueness(qutrit;return_basis=true)
    qexplicit=explicit_model_evans(qutrit)
    @test qreport.unique==qexplicit.unique==true
    @test qreport.commutant_dimension==qexplicit.commutant_dimension==1
    @test qreport.basis_representation==:auxiliary_schur_blocks
    @test sum(item.multiplicity*size(item.vectors,2)
              for item in qreport.commutant_basis)==qreport.commutant_dimension
end
