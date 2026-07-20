"""
    state_at(solution, time)

Return the state represented by `solution` at physical `time`. Saved-grid
results require `time` to match a stored sample (up to `isapprox`); SciML
solutions use their continuous interpolation. Unlike `state(solution, i)`, an
integer passed to `state_at` is always interpreted as a physical time rather
than a saved-state index. Package solvers return nondecreasing saved grids;
manually constructed result objects must preserve that invariant so the
allocation-free binary lookup remains valid.
"""
function state_at end

function _saved_time_index(times::AbstractVector,time::Real)
    isempty(times)&&throw(ArgumentError("the solution has no saved times"))
    position=searchsortedfirst(times,time)
    index=if position<=firstindex(times)
        firstindex(times)
    elseif position>lastindex(times)
        lastindex(times)
    else
        lower=position-1
        abs(times[lower]-time)<=abs(times[position]-time) ? lower : position
    end
    isapprox(times[index],time)||throw(ArgumentError(
        "time $time was not saved; nearest saved time is $(times[index])"))
    index
end
