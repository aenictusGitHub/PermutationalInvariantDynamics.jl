@testset "Automated convergence studies" begin
    @testset "generic refinement evidence" begin
        timesteps=[0.4,0.2,0.1,0.05]
        report=convergence_study(timesteps;parameter=:mesh,
            refinement_scale=identity,rtol=0.06) do h
            (value=1+2h^2,converged=true,iterations=round(Int,1/h))
        end
        @test report isa ConvergenceStudyResult
        @test report.parameter===:mesh
        @test report.converged
        @test report.reason===:converged
        @test report.first_passing_index==4
        @test report.consecutive_required==2
        @test length(report)==length(timesteps)
        @test report[2]===report.results[2]
        @test all(report.solver_converged)
        @test ismissing(report.pairwise_errors[1])
        @test ismissing(report.observed_rates[2])
        @test report.observed_rates[3]≈2 atol=2e-14
        @test report.observed_rates[4]≈2 atol=2e-14
        @test convergence_estimate(report)≈1.005
        @test occursin("converged=true",sprint(show,report))

        # An earlier passing window cannot hide loss of stability at the
        # finest requested refinement.
        unstable_values=Dict(1=>1.0,2=>1.001,3=>1.0015,4=>2.0)
        unstable=convergence_study(i->unstable_values[i],1:4;
            atol=0.002,rtol=0,consecutive=2)
        @test !unstable.converged
        @test unstable.first_passing_index==3
        @test unstable.reason===:refinement_not_stable
        @test_throws ArgumentError convergence_estimate(unstable)
        @test convergence_estimate(unstable;require_convergence=false)==2

        # Two agreeing values are insufficient under the conservative
        # two-comparison default.
        insufficient=convergence_study(_->1.0,[1,2];atol=1e-12)
        @test !insufficient.converged
        @test insufficient.reason===:insufficient_refinements

        # Inner iterative-solver failure blocks the outer convergence claim.
        statuses=Bool[true,false,true]
        inner=convergence_study(i->(value=1.0,converged=statuses[i]),1:3;
            atol=1e-12)
        @test !inner.converged
        @test inner.reason===:inner_solver_not_converged
        @test inner.pairwise_converged[2:3]==[true,true]
        @test_throws ArgumentError convergence_study(
            i->(value=1.0,converged=statuses[i]),1:3;
            atol=1e-12,require_convergence=true)

        # Vector estimates use their normed full difference by default.
        vector_report=convergence_study(i->Float64[1,inv(i)^2],[2,4,8,16];
            atol=0.05,rtol=0)
        @test vector_report.converged
        @test vector_report.estimates[end]==[1.0,1/256]

        first_basis=PIBasis(1,2;sectors=[(1,0)])
        second_basis=PIBasis(3,2;sectors=[(2,1)])
        first_state=PIState(first_basis,zeros(ComplexF64,length(first_basis)))
        second_state=PIState(second_basis,zeros(ComplexF64,length(second_basis)))
        @test length(first_state.data)==length(second_state.data)
        @test_throws ArgumentError convergence_study(
            index->index==1 ? first_state : second_state,[1,2];
            consecutive=1)
    end

    @testset "domain-specific refinement wrappers" begin
        time_report=timestep_convergence([0.4,0.2,0.1,0.05];rtol=0.06) do h
            1+2h^2
        end
        @test time_report.parameter===:time_step
        @test time_report.converged
        @test time_report.metadata.refinement_scales==[0.4,0.2,0.1,0.05]

        krylov_report=krylov_dimension_convergence([4,8,16,32];rtol=0.02) do m
            (solution=1+inv(float(m))^2,converged=true,iterations=m)
        end
        @test krylov_report.parameter===:krylov_dimension
        @test krylov_report.converged
        @test krylov_report.observed_rates[end]≈2 atol=2e-14
        @test krylov_report.diagnostics[end].iterations==32

        hierarchy_report=hierarchy_depth_convergence([0,1,3,7];rtol=0.3) do depth
            1+2/(depth+1)^2
        end
        @test hierarchy_report.parameter===:hierarchy_depth
        @test hierarchy_report.converged
        @test hierarchy_report.observed_rates[end]≈2 atol=2e-14

        nongeometric=timestep_convergence([0.4,0.2,0.05];
            atol=1,rtol=0,consecutive=1) do h
            1+h^2
        end
        @test ismissing(nongeometric.observed_rates[end])

        sector_report=sector_cutoff_convergence([2,4,8,16];rtol=0.05) do cutoff
            1+inv(float(cutoff))^2
        end
        @test sector_report.parameter===:sector_cutoff
        @test sector_report.converged

        @test_throws ArgumentError timestep_convergence(identity,[0.1,0.2])
        @test_throws ArgumentError krylov_dimension_convergence(identity,[4,4,8])
        @test_throws ArgumentError hierarchy_depth_convergence(identity,[-1,0,1])
        @test_throws ArgumentError sector_cutoff_convergence(identity,[0,1,2])
    end

    @testset "precision and invalid data" begin
        h32=Float32[0.4,0.2,0.1,0.05]
        report32=timestep_convergence(h->1f0+2f0*h^2,h32;rtol=0.06f0)
        @test report32.metadata.atol isa Float32
        @test report32.metadata.rtol isa Float32
        @test all(value->value isa Float32,skipmissing(report32.pairwise_errors))
        @test all(value->value isa Float32,skipmissing(report32.observed_rates))

        @test_throws ArgumentError convergence_study(_->NaN,1:3)
        @test_throws ArgumentError convergence_study(_->[NaN],1:3;
            estimate_norm=_ -> 0.0)
        @test_throws ArgumentError convergence_study(_->1.0,1:3;
            refinement_scale=i->i)
        @test_throws ArgumentError convergence_study(_->1.0,1:3;
            distance=(a,b)->-1)
        @test_throws ArgumentError convergence_study(_->1.0,[1])
        @test_throws ArgumentError convergence_study(_->1.0,1:3;
            consecutive=0)
        extreme=convergence_study(_->1.0,1:3;
            consecutive=typemax(Int),atol=0,rtol=0)
        @test !extreme.converged
        @test extreme.reason===:insufficient_refinements
        @test_throws ArgumentError convergence_study(_->1.0,1:3;
            atol=1e308,rtol=1e308,consecutive=1)
        calls=Ref(0)
        evaluator=_ -> begin
            calls[]+=1
            1.0
        end
        @test_throws ArgumentError convergence_study(evaluator,1:3;atol=-1)
        @test calls[]==0
        @test_throws ArgumentError convergence_study(evaluator,1:3;
            refinement_scale=identity)
        @test calls[]==0
    end
end
