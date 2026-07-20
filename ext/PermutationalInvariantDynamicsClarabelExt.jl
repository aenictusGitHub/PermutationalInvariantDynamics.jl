module PermutationalInvariantDynamicsClarabelExt

using PermutationalInvariantDynamics
using SparseArrays
import Clarabel

const PID=PermutationalInvariantDynamics

function _clarabel_settings(::Type{T};solver_atol,solver_rtol,
        max_iterations,time_limit,verbose,solver_options) where
        T<:AbstractFloat
    settings=Clarabel.Settings{T}()
    settings.max_iter=UInt32(max_iterations)
    settings.time_limit=Float64(time_limit)
    settings.verbose=verbose
    settings.tol_gap_abs=T(solver_atol)
    settings.tol_gap_rel=T(solver_rtol)
    settings.tol_feas=T(solver_atol)
    settings.tol_infeas_abs=T(solver_atol)
    settings.tol_infeas_rel=T(solver_rtol)
    if T===Float32
        # Clarabel accepts Float32, but its Float64-oriented regularization
        # defaults lie below useful single-precision scales.  These settings
        # follow the solver documentation's recommendation to relax the
        # numerical linear-algebra tolerances for 32-bit data.
        settings.static_regularization_constant=T(1e-5)
        settings.dynamic_regularization_eps=T(1e-6)
        settings.dynamic_regularization_delta=T(1e-3)
        settings.iterative_refinement_reltol=T(1e-5)
        settings.iterative_refinement_abstol=T(1e-5)
    end
    # The returned dual vector must use the original full PSD cones so core
    # can independently validate stationarity and cone membership.
    settings.chordal_decomposition_enable=false
    settings.chordal_decomposition_complete_dual=true

    reserved=Set((:max_iter,:time_limit,:verbose,:tol_gap_abs,:tol_gap_rel,
        :tol_feas,:tol_infeas_abs,:tol_infeas_rel,
        :chordal_decomposition_enable,
        :chordal_decomposition_complete_dual))
    for (name,value) in pairs(solver_options)
        name in reserved&&throw(ArgumentError(
            "Clarabel setting $name is controlled by an explicit ppt_mixture_test keyword"))
        hasproperty(settings,name)||throw(ArgumentError(
            "unknown Clarabel setting $name"))
        field_type=fieldtype(typeof(settings),name)
        converted=try
            convert(field_type,value)
        catch error
            throw(ArgumentError(
                "Clarabel setting $name cannot represent $(repr(value)): $(sprint(showerror,error))"))
        end
        setproperty!(settings,name,converted)
    end
    settings
end

function PID._solve_ppt_mixture(::Val{:clarabel},
        plan::PID.PPTMixturePlan{T},rhs::AbstractVector{T};
        certificate_atol,certificate_rtol,solver_atol,solver_rtol,
        max_iterations,time_limit,verbose,solver_options) where
        T<:AbstractFloat
    settings=_clarabel_settings(T;solver_atol,solver_rtol,max_iterations,
        time_limit,verbose,solver_options)
    cones=Clarabel.SupportedCone[]
    for dimension in plan.cone_dimensions
        push!(cones,Clarabel.PSDTriangleConeT(dimension))
    end
    push!(cones,Clarabel.ZeroConeT(length(plan.equality_rows)))
    nvariables=size(plan.constraint_matrix,2)
    quadratic=spzeros(T,nvariables,nvariables)
    solver=Clarabel.Solver(quadratic,plan.objective,
        plan.constraint_matrix,rhs,cones,settings)
    solution=Clarabel.solve!(solver)
    status_symbol=Symbol(lowercase(string(solution.status)))
    PID._ppt_classified_result(plan,rhs,solution.x,solution.z;
        solver_status=status_symbol,
        solved=solution.status==Clarabel.SOLVED,
        primal_objective=solution.obj_val,
        solver_dual_objective=solution.obj_val_dual,
        iterations=Int(solution.iterations),
        solve_time=solution.solve_time,
        certificate_atol,certificate_rtol)
end

end
