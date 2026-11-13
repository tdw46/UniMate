"""Hierarchy/aspect bucket modes and verified named torso overrides; research subset.

No geometric rig-map fallback or face-specific reassignment is claimed here.
"""
from .rig_mapping import validate,detect_spine

def assign(names,parents,*,spine_shapes=None,spine_from_names=True,bucket_modes=(2,14,6,14,6),prepared=None):
    _,p,_=validate(names,parents,None)
    children=[0]*len(names)
    for i in p:
        if i>=0:children[i]+=1
    modes=[2 if count>=3 else 14 if count else 6 for count in children]
    mapping=detect_spine(names,p,[True]*len(names)) if spine_from_names else None
    if mapping and mapping['verified']:
        # Named torso mapping limits hub seeds to pelvis/spine. Other branches
        # remain cylinders, including hands and branching hair joints. This is
        # independent of hand/finger names; disabling mapping keeps tree hubs.
        hubs={mapping['pelvis'],*mapping['spine']}
        modes=[14 if mode==2 and i not in hubs else mode for i,mode in enumerate(modes)]
    if bucket_modes is None:
        modes=[0]*len(names)
    elif prepared is not None:
        import numpy as np
        h,t,r=prepared
        delta=np.asarray(t,'f4')-np.asarray(h,'f4')
        length=np.sqrt((delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2])
        length=np.where(length>np.float32(1e-6),length,np.float32(1e-6))
        ratio=np.asarray(r,'f4')/length
        kinds=[0 if modes[i]==2 else ((3 if ratio[i]<=np.float32(.55) else 1)+(not count)) for i,count in enumerate(children)]
        modes=[bucket_modes[k] for k in kinds]
    elif len(set(bucket_modes))==1:
        modes=[bucket_modes[0]]*len(names)
    # Native unknown bytes select the regular profile; mode 10 without an
    # oriented box falls back to the statistical ellipsoid (5).
    modes=[0 if x>18 else 5 if x==10 else x for x in modes]
    override_modes=[255]*len(names)
    if spine_shapes is not None and bucket_modes is not None:
        if any(not isinstance(x,int) or not 0<=x<=255 for x in spine_shapes):
            raise ValueError('invalid torso mode overrides')
        if mapping and mapping['verified']:
            settings=list(spine_shapes)+[6,6]
            if len(settings)<5:
                from . import _bvt_solver as runtime
                raise runtime.PanicException(f'index out of bounds: the len is {len(settings)} but the index is {len(settings)}')
            groups=[[mapping['pelvis']] if mapping['pelvis']>=0 else [],mapping['spine'][:-1],mapping['spine'][-1:],mapping['neck'],[mapping['head']] if mapping['head']>=0 else []]
            for ids,value in zip(groups,settings):
                if value==255:continue # native sentinel preserves hierarchy selection
                for i in ids:modes[i]=value;override_modes[i]=value
    modes=[0 if x>18 else 5 if x==10 else x for x in modes]
    trunk=None
    if mapping and mapping['verified']:
        trunk=[int(i in mapping['spine']) for i in range(len(names))]
    return dict(modes=modes,torso=mapping,trunk=trunk,override_modes=override_modes,complete_rig_map=False)
