module PermutationalInvariantDynamicsHDF5Ext

using PermutationalInvariantDynamics
using HDF5

import PermutationalInvariantDynamics: _save_checkpoint, _load_checkpoint,
    _checkpoint_payload, _checkpoint_from_payload, PIStateCheckpoint

function _save_checkpoint(::Val{:hdf5},path,checkpoint::PIStateCheckpoint)
    payload=_checkpoint_payload(checkpoint)
    HDF5.h5open(path,"w") do file
        for name in propertynames(payload)
            HDF5.write(file,string(name),getproperty(payload,name))
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

end
