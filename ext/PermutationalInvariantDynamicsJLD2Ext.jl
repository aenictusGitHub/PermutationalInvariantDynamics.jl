module PermutationalInvariantDynamicsJLD2Ext

using PermutationalInvariantDynamics
using JLD2

import PermutationalInvariantDynamics: _save_checkpoint, _load_checkpoint,
    _checkpoint_payload, _checkpoint_from_payload, PIStateCheckpoint,
    _save_result, _result_detached_columns, PI_RESULT_ARCHIVE_VERSION

function _save_checkpoint(::Val{:jld2},path,checkpoint::PIStateCheckpoint)
    payload=_checkpoint_payload(checkpoint)
    JLD2.jldopen(path,"w") do file
        for name in propertynames(payload)
            file[string(name)]=getproperty(payload,name)
        end
    end
    String(path)
end

function _load_checkpoint(::Val{:jld2},path)
    payload=JLD2.jldopen(path,"r") do file
        (;schema_version=file["schema_version"],N=file["N"],d=file["d"],
          sectors=file["sectors"],scalar_type=file["scalar_type"],
          real=file["real"],imag=file["imag"],
          has_time=file["has_time"],time=file["time"],
          metadata_keys=file["metadata_keys"],
          metadata_values=file["metadata_values"])
    end
    _checkpoint_from_payload(payload)
end

function _save_result(::Val{:jld2},path,result,table,summary,metadata)
    columns=_result_detached_columns(table)
    JLD2.jldopen(path,"w") do file
        file["schema_version"]=Int(PI_RESULT_ARCHIVE_VERSION)
        file["package_version"]=string(
            Base.pkgversion(PermutationalInvariantDynamics))
        file["julia_version"]=string(VERSION)
        file["summary"]=summary
        file["metadata"]=metadata
        file["columns"]=columns
        # JLD2 is the explicitly Julia-native backend. Retaining the object
        # here is intentional; the portable reconstructing state and
        # experiment formats remain save_checkpoint/save_experiment.
        file["result"]=result
    end
    String(path)
end

end
