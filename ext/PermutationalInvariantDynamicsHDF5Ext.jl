module PermutationalInvariantDynamicsHDF5Ext

using PermutationalInvariantDynamics
using HDF5

import PermutationalInvariantDynamics: _save_checkpoint, _load_checkpoint,
    _checkpoint_payload, _checkpoint_from_payload, PIStateCheckpoint,
    _save_result, _result_state_records, _result_text,
    PI_RESULT_ARCHIVE_VERSION

_hdf5_payload_value(value::AbstractArray)=Array(value)
_hdf5_payload_value(value)=value

function _save_checkpoint(::Val{:hdf5},path,checkpoint::PIStateCheckpoint)
    payload=_checkpoint_payload(checkpoint)
    HDF5.h5open(path,"w") do file
        for name in propertynames(payload)
            HDF5.write(file,string(name),
                _hdf5_payload_value(getproperty(payload,name)))
        end
    end
    String(path)
end

function _load_checkpoint(::Val{:hdf5},path)
    payload=HDF5.h5open(path,"r") do file
        (;schema_version=HDF5.read(file,"schema_version"),
          N=HDF5.read(file,"N"),d=HDF5.read(file,"d"),
          sectors=HDF5.read(file,"sectors"),
          scalar_type=HDF5.read(file,"scalar_type"),
          real=HDF5.read(file,"real"),imag=HDF5.read(file,"imag"),
          has_time=HDF5.read(file,"has_time"),time=HDF5.read(file,"time"),
          metadata_keys=HDF5.read(file,"metadata_keys"),
          metadata_values=HDF5.read(file,"metadata_values"))
    end
    _checkpoint_from_payload(payload)
end

function _hdf5_write_checkpoint_group(group,payload)
    for name in propertynames(payload)
        HDF5.write(group,string(name),
            _hdf5_payload_value(getproperty(payload,name)))
    end
end

function _save_result(::Val{:hdf5},path,result,table,summary,metadata)
    HDF5.h5open(path,"w") do file
        HDF5.attributes(file)["schema_version"]=Int(PI_RESULT_ARCHIVE_VERSION)
        HDF5.attributes(file)["package_version"]=string(
            Base.pkgversion(PermutationalInvariantDynamics))
        HDF5.attributes(file)["julia_version"]=string(VERSION)
        summary_group=HDF5.create_group(file,"summary")
        for name in propertynames(summary)
            HDF5.attributes(summary_group)[string(name)]=
                _result_text(getproperty(summary,name))
        end
        metadata_group=HDF5.create_group(file,"metadata")
        for (key,value) in sort!(collect(metadata);by=first)
            HDF5.attributes(metadata_group)[key]=value
        end
        columns_group=HDF5.create_group(file,"columns")
        for name in propertynames(table.columns)
            column=getproperty(table.columns,name)
            # A text representation accepts heterogeneous scan diagnostics
            # without narrowing or HDF5 compound-type inference.
            HDF5.write(columns_group,string(name),_result_text.(column))
        end
        states_group=HDF5.create_group(file,"states")
        for (index,record) in enumerate(_result_state_records(result))
            group=HDF5.create_group(
                states_group,"state_$(lpad(index,6,'0'))")
            HDF5.attributes(group)["label"]=record.label
            HDF5.attributes(group)["has_time"]=record.time!==nothing
            record.time===nothing||
                (HDF5.attributes(group)["time"]=_result_text(record.time))
            _hdf5_write_checkpoint_group(
                group,_checkpoint_payload(PIStateCheckpoint(
                    record.state;time=record.time)))
        end
    end
    String(path)
end

end
