@testset "Appendix-D p-body paths and kernels" begin
    PID=PermutationalInvariantDynamics
    sx=ComplexF64[0 1;1 0];sz=ComplexF64[1 0;0 -1];sm=ComplexF64[0 1;0 0]
    for d in (2,3),N in 2:4
        b=PIBasis(N,d);X=diagm(ComplexF64.(collect(1:d)))
        @test pbody_collective_operator(b,X,1).data≈collective_operator(b,X).data atol=3e-11
        @test Matrix(pbody_kernel_operator(b,X,X,1))≈Matrix(local_kernel_operator(b,X,X)) atol=3e-11
    end
    b1=PIBasis(3,2)
    @test Matrix(liouvillian(PIModel(b1,[PBodyHamiltonian(sz,1)]);representation=:sparse))≈Matrix(liouvillian(PIModel(b1,[LocalHamiltonian(sz)]);representation=:sparse))
    @test Matrix(liouvillian(PIModel(b1,[LocalPBodyJump(sm,1)]);representation=:sparse))≈Matrix(liouvillian(PIModel(b1,[LocalJump(sm)]);representation=:sparse))
    @test Matrix(liouvillian(PIModel(b1,[CollectivePBodyJump(sm,1)]);representation=:sparse))≈Matrix(liouvillian(PIModel(b1,[CollectiveJump(sm)]);representation=:sparse))

    for N in 2:5
        b=PIBasis(N,2);pair=kron(sz,sz);A=pbody_collective_operator(b,pair,2)
        J=collective_operator(b,sz);reference=(J*J-N*identity_operator(b))*(1/2)
        @test A.data≈reference.data atol=5e-10
    end

    b=PIBasis(4,2);pairjump=kron(sm,sm)
    terms=(PBodyHamiltonian(kron(sz,sz),2;rate=.3),LocalPBodyJump(pairjump,2;rate=.2),CollectivePBodyJump(pairjump,2;rate=.1))
    m=PIModel(b,collect(terms));Ls=liouvillian(m;representation=:sparse);Lm=liouvillian(m;representation=:matrixfree)
    rho=iid_pure_state(b,ComplexF64[0,1]);@test Ls*rho.data≈Lm*rho.data atol=2e-10
    @test check_generator(m).trace_preservation_error<2e-9

    # With p=N there is only one subset, so local and collective dissipators coincide.
    b2=PIBasis(2,2);X=kron(sm,sm)
    Ll=liouvillian(PIModel(b2,[LocalPBodyJump(X,2)]);representation=:sparse)
    Lc=liouvillian(PIModel(b2,[CollectivePBodyJump(X,2)]);representation=:sparse)
    @test Matrix(Ll)≈Matrix(Lc) atol=3e-11

    # Local Young-graph neighbourhood generation must reproduce the explicit
    # all-partition Appendix-D closure oracle on small qubit and qudit cases.
    # The production route then remains independent of the total partition
    # count for a huge N with only a few retained sectors.
    for dcheck in 2:3,Ncheck in 2:7,pcheck in 1:min(Ncheck,3)
        closure_basis=PIBasis(Ncheck,dcheck)
        closure_term=LocalPBodyJump(
            Matrix{Float64}(I,dcheck^pcheck,dcheck^pcheck),pcheck)
        centers=Dict(sector=>Set(first.(PID._removal_paths(sector,pcheck)))
                     for sector in closure_basis.sectors)
        oracle=unique(output for input in closure_basis.sectors,
            output in closure_basis.sectors
            if !isempty(intersect(centers[input],centers[output])))
        @test Set(PID.required_sectors(closure_term,closure_basis))==Set(oracle)
    end

    nonsymmetric=kron(sm,sx)
    @test_throws ArgumentError pbody_collective_operator(b2,nonsymmetric,2)
    @test_throws ArgumentError PIModel(PIBasis(3,2;sectors=[(3,0)]),[LocalPBodyJump(pairjump,2)])

    # Independent computational-basis reference (particle 1 is the fastest
    # tensor index, matching Appendix D's tuple flattening).
    function embed_subset(X,subset,N,d)
        D=d^N;M=zeros(ComplexF64,D,D)
        for a in 0:D-1,c in 0:D-1
            da=[(a÷d^(r-1))%d for r in 1:N];dc=[(c÷d^(r-1))%d for r in 1:N]
            all(r->r in subset||da[r]==dc[r],1:N)||continue
            ia=1+sum(da[subset[r]]*d^(r-1) for r in eachindex(subset));ic=1+sum(dc[subset[r]]*d^(r-1) for r in eachindex(subset))
            M[a+1,c+1]=X[ia,ic]
        end
        M
    end
    N=3;d=2;b3=PIBasis(N,d);psi=normalize(ComplexF64[1,1im]);rho=iid_pure_state(b3,psi)
    A=ComplexF64[1 2im 3 1; -2im 0 2 4;3 2 2 -im;1 4 im -1];A=(A+A')/2
    swap=[1,3,2,4];X=(A+A[swap,swap])/2
    Hp=sum(embed_subset(X,[i,j],N,d) for i in 1:N-1 for j in i+1:N)
    fullpsi=reduce(kron,reverse(fill(psi,N)))
    @test expectation(rho,pbody_collective_operator(b3,X,2))≈dot(fullpsi,Hp*fullpsi) atol=2e-10

    Jzfull=sum(embed_subset(sz,[i],N,d) for i in 1:N)
    Ldense=zeros(ComplexF64,d^N,d^N)
    rhofull=fullpsi*fullpsi'
    for i in 1:N-1,j in i+1:N
        ell=embed_subset(pairjump,[i,j],N,d);Ldense .+= ell*rhofull*ell'-(ell'*ell*rhofull+rhofull*ell'*ell)/2
    end
    dpi=PIState(b3,liouvillian(PIModel(b3,[LocalPBodyJump(pairjump,2)]);representation=:sparse)*rho.data)
    @test expectation(dpi,collective_operator(b3,sz))≈tr(Jzfull*Ldense) atol=3e-10

    # Large exact Appendix-D factors must be combined before conversion.  For
    # the restricted N=18 singlet, intermediate multiplicity products overflow
    # Float16 although both the collective identity block and identity gain map
    # are simply binomial(18,2)=153.
    singlet18=Partition((9,9))
    b18=PIBasis(18,2;sectors=[singlet18.parts])
    identity_pair16=Matrix{Complex{Float16}}(I,4,4)
    cache18=PBodyGeometry(b18,2,Float16)
    block18=pbody_collective_block(cache18,identity_pair16,singlet18)
    kernel18=Matrix(pbody_kernel_operator(
        b18,identity_pair16,identity_pair16,2;cache=cache18))
    expected18=Float16(exact_binomial(18,2))
    @test all(isfinite,block18)
    @test all(isfinite,kernel18)
    @test block18[1,1]≈expected18 atol=Float16(1)
    @test kernel18[1,1]≈expected18 atol=Float16(1)

    # A Kac-normalized operator remains finite even though its subset count is
    # not representable in Float16. The gain combines two small contractions
    # with the exact path-pair scale before either product underflows.
    Nlarge=1000;singlet=Partition((500,500))
    blarge=PIBasis(Nlarge,2;sectors=[singlet.parts])
    large_count=exact_binomial(Nlarge,2)
    inverse_count=Float16(inv(BigFloat(large_count)))
    normalized_pair=inverse_count.*Matrix{ComplexF16}(I,4,4)
    large_cache=PBodyGeometry(blarge,2,Float16)
    normalized_block=pbody_collective_block(
        large_cache,normalized_pair,singlet)
    normalized_gain=Matrix(pbody_kernel_operator(
        blarge,normalized_pair,normalized_pair,2;cache=large_cache))
    @test normalized_block[1,1]≈Float16(BigFloat(large_count)*inverse_count) rtol=Float16(3e-2)
    @test normalized_gain[1,1]≈Float16(
        BigFloat(large_count)*BigFloat(inverse_count)^2) rtol=Float16(3e-2)

    # Fixed-spin pair blocks lose an O(1) splitting through O(N^2) path
    # cancellation in native Float64; the risk-triggered wide route restores it.
    Ncancellation=10^8;spinone=Partition((Ncancellation÷2+1,Ncancellation÷2-1))
    bcancellation=PIBasis(Ncancellation,2;sectors=[spinone.parts])
    jz=ComplexF64[-0.5 0;0 0.5]
    cancellation_block=pbody_collective_block(
        PBodyGeometry(bcancellation,2,Float64),kron(jz,jz),spinone)
    @test real(cancellation_block[1,1]-cancellation_block[2,2])≈0.5 atol=2eps(Float64)

    # Local gain coordinates can be much more ill-conditioned than either
    # endpoint path scale suggests: all individual Float32 path-pair factors
    # below are finite/direct, while their O(10^-3) contributions cancel to a
    # representable O(10^-14) result.  Risky static groups therefore certify
    # the native sum and reuse guarded-wide contractions when it is severe.
    Nkernel=10^12
    kernel_sectors=[(Nkernel÷2+r,Nkernel÷2-r) for r in 1:2]
    bkernel=PIBasis(Nkernel,2;sectors=kernel_sectors)
    # Model validation detects restriction from the exact retained PI
    # dimension; it must not enumerate O(N) qubit partitions here.
    @test PIModel(bkernel,()).basis===bkernel
    kernel_rng=MersenneTwister(7)
    pair_operator=randn(kernel_rng,ComplexF64,4,4)
    pair_swap=PID._tensor_swap_permutation(2,2,1)
    pair_operator=(pair_operator+pair_operator[pair_swap,pair_swap])/2
    pair_operator/=norm(pair_operator,Inf)
    pair32=ComplexF32.(pair_operator/Nkernel)
    cache32=PBodyGeometry(bkernel,2,Float32)
    l=bkernel.sectors[2];n=bkernel.sectors[1]
    a=3;bb=2;c=2;dindex=3
    element32=pbody_kernel_element(cache32,pair32,pair32,
                                   l,a,bb,n,c,dindex)
    reference=setprecision(BigFloat,256) do
        pairwide=Complex{BigFloat}.(pair32)
        widecache=PBodyGeometry(bkernel,2,BigFloat)
        pbody_kernel_element(widecache,pairwide,pairwide,
                             l,a,bb,n,c,dindex)
    end
    @test element32≈ComplexF32(reference) rtol=8f-5 atol=1f-20

    # Materializing the complete map also checks every other coordinate. Some
    # genuine entries of this fixed-spin map fall below Float32's storage
    # range, so use Float64 for the complete triplet oracle while retaining the
    # same N and cancellation condition number. The element test above remains
    # the narrow Float32 regression.
    triplet_pair64=ComplexF64.(pair_operator)
    cache64=PBodyGeometry(bkernel,2,Float64)
    triplet_reference=setprecision(BigFloat,256) do
        pairwide=Complex{BigFloat}.(triplet_pair64)
        widecache=PBodyGeometry(bkernel,2,BigFloat)
        pbody_kernel_element(widecache,pairwide,pairwide,
                             l,a,bb,n,c,dindex)
    end
    triplets64=PID.pbody_kernel_triplets(
        cache64,triplet_pair64,triplet_pair64)
    li=PID._sidx(bkernel,l);ni=PID._sidx(bkernel,n)
    nl=length(bkernel.patterns[li]);nn=length(bkernel.patterns[ni])
    row=bkernel.offsets[li]+a-1+(bb-1)*nl
    column=bkernel.offsets[ni]+c-1+(dindex-1)*nn
    triplet_value=sparse(triplets64.I,triplets64.J,triplets64.V,
                         length(bkernel),length(bkernel))[row,column]
    @test triplet_value≈ComplexF64(triplet_reference) rtol=5e-12 atol=1e-3

    # A preallocated dynamic gain cannot widen its fixed contraction scratch.
    # Reject even the all-direct large-factor case at compilation, rather than
    # silently applying an uncertified Float32 map.
    # Exercise the compile helper directly: a generic local term requires a
    # complete retained basis before PIModel validation, which is unrelated to
    # the path-scale diagnostic tested here.
    Ndynamic=200
    dynamic_sectors=[(Ndynamic÷2+1,Ndynamic÷2-1)]
    bdynamic=PIBasis(Ndynamic,2;sectors=dynamic_sectors)
    dynamic_identity=Matrix{ComplexF32}(I,4,4)/Float32(Ndynamic)
    dynamic_schedule=InPlaceTimeOperator(
        dynamic_identity,(destination,t,p)->nothing)
    dynamic_term=LocalPBodyJump(dynamic_schedule,2)
    dynamic_context=PID.TermCompileContext(
        bdynamic,nothing,
        Dict{Int,PBodyGeometry{Float32,2,3,typeof(bdynamic)}}(),Float32)
    dynamic_builder=PID._pbody_block_builder(dynamic_context,dynamic_term)
    @test dynamic_builder.cancellation_risk
    @test all(entry[2].direct for entries in dynamic_builder.block_entries
                              for entry in entries)
    @test_throws ArgumentError PID._pbody_gain_factorization(dynamic_builder)
    @test PID._dynamic_pbody_block_uncertified(1.0f0+0.0f0im,1.0f0,2)
    dynamic_blocks=[zeros(ComplexF32,length(patterns),length(patterns))
                    for patterns in bdynamic.patterns]
    @test_throws ArgumentError PID._fill_dynamic_blocks!(
        dynamic_blocks,dynamic_builder,dynamic_identity)

    # A feasible complete basis exercises the public compilation route too.
    # At N=24 the Float16 path factors are direct but exceed the cancellation
    # certification threshold, while the trace vector remains representable.
    public_dynamic_basis=PIBasis(24,2)
    public_dynamic_identity=Matrix{Complex{Float16}}(I,4,4)/Float16(24)
    public_dynamic_schedule=InPlaceTimeOperator(
        public_dynamic_identity,(destination,t,p)->nothing)
    @test_throws ArgumentError LiouvillianPlan(PIModel(public_dynamic_basis,[
        LocalPBodyJump(public_dynamic_schedule,2)]))

    # End-to-end identity dissipators vanish exactly.  This catches a former
    # O(1) residual from independently rounded O(N^2) gain/anticommutator data.
    Ndissipator=10^8
    dissipator_sectors=[(Ndissipator÷2+1,Ndissipator÷2-1)]
    bdissipator=PIBasis(Ndissipator,2;sectors=dissipator_sectors)
    identity_pair=Matrix{ComplexF64}(I,4,4)
    identity_gain=pbody_kernel_operator(
        bdissipator,identity_pair,identity_pair,2)
    subset_count=Float64(exact_binomial(Ndissipator,2))
    identity_residual=identity_gain-
        subset_count*sparse(I,length(bdissipator),length(bdissipator))
    @test norm(identity_residual,Inf)<=32eps(Float64)

    # The ordinary small-N branch remains the original direct accumulation
    # path and retains its allocation-light setup-free element evaluation.
    bsmall=PIBasis(4,2);cachesmall=PBodyGeometry(bsmall,2,Float64)
    small_sector=first(bsmall.sectors);small_pair=kron(sz,sz)
    @test !PID._pbody_kernel_cancellation_possible(
        cachesmall,small_sector,small_sector,Float64)
    pbody_kernel_element(cachesmall,small_pair,small_pair,
                         small_sector,1,1,small_sector,1,1)
    @test (@allocated pbody_kernel_element(
        cachesmall,small_pair,small_pair,small_sector,1,1,
        small_sector,1,1))<=64*1024

    # A forward-error bound is not a proof that a nonzero sum is exactly zero.
    # Guarded-wide conversion must raise on an unresolved representable value
    # instead of silently deleting that kernel coordinate.
    setprecision(BigFloat,128) do
        absolute_sum=one(BigFloat)
        bound=PID._wide_pbody_zero_bound(
            cachesmall,small_sector,small_sector,BigFloat,absolute_sum,1)
        unresolved=Complex{BigFloat}(bound/2)
        @test !iszero(ComplexF64(unresolved))
        @test_throws ArgumentError PID._convert_checked_pbody_kernel_sum(
            Float64,unresolved,absolute_sum,cachesmall,
            small_sector,small_sector,1;context="test p-body sum")
    end
end
