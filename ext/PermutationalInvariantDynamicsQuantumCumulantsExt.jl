module PermutationalInvariantDynamicsQuantumCumulantsExt

using PermutationalInvariantDynamics
import QuantumCumulants

function _neutral_key(moments,key)
    key isa Symbol ?
        PermutationalInvariantDynamics._canonical_moment_key(moments,(key,)) :
        PermutationalInvariantDynamics._canonical_moment_key(moments,key)
end

function quantumcumulants_initial_values(
        moments::PermutationalInvariantDynamics.OrderedLocalMoments,
        symbolic_map)
    entries=try
        symbolic_map isa Union{NamedTuple,AbstractDict} ?
            collect(pairs(symbolic_map)) : collect(symbolic_map)
    catch error
        throw(ArgumentError(
            "symbolic_map must be a dictionary or named collection of neutral-key => symbolic-average entries: $(sprint(showerror,error))"))
    end
    output=Dict{Any,eltype(moments)}()
    for entry in entries
        neutral=first(entry);symbolic=last(entry)
        key=_neutral_key(moments,neutral)
        haskey(moments,key)||throw(KeyError(key))
        symbolic_order=try
            QuantumCumulants.get_order(symbolic)
        catch error
            throw(ArgumentError(
                "QuantumCumulants.get_order failed for the symbolic target of $key: $(sprint(showerror,error))"))
        end
        symbolic_order==length(key)||throw(ArgumentError(
            "neutral moment $key has order $(length(key)), but its QuantumCumulants target has order $symbolic_order"))
        haskey(output,symbolic)&&throw(ArgumentError(
            "multiple neutral moments map to the same QuantumCumulants symbolic average"))
        output[symbolic]=moments[key]
    end
    output
end

end
