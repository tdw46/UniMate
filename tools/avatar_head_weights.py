"""Head ownership for an upright bare-skin bust in rig-local coordinates.

Heat supplies the neck transition. A measured neck-shaft envelope distinguishes
projecting jaw/skull surfaces from the shaft; the cranial region is rigid Head.
This semantic correction does not change geometry or expand the voxel patch.
"""
import numpy as np
from avatar_apparel_weights import weights, assign


def smoothstep(low, high, value):
    t=np.clip((value-low)/(high-low),0.,1.)
    return t*t*(3-2*t)


def head_ownership(points, neck_base, atlas, surface_samples=None):
    points=np.asarray(points,dtype=float)
    base=np.asarray(neck_base,dtype=float);top=np.asarray(atlas,dtype=float)
    height=top[2]-base[2]
    audit={'applied':False}
    if height<=1e-8:return np.zeros(len(points)),audit
    unique=np.unique(points if surface_samples is None else surface_samples,axis=0)
    sections=[]
    # Lower shaft avoids jaw contamination. Use section medians so duplicate
    # UV vertices and denser tessellation in one slice cannot dominate the fit.
    for f in np.linspace(.3,.55,7):
        p=unique[abs(unique[:,2]-(base[2]+f*height))<height*.04]
        if len(p)<12:continue
        lo,hi=np.quantile(p[:,:2],[.025,.975],axis=0)
        sections.append((lo,hi))
    if len(sections)<4:
        audit['reason']='Insufficient neck shaft sections'
        return np.zeros(len(points)),audit
    lo,hi=np.median(np.array(sections),axis=0)
    radii=(hi-lo)*.5;center=(hi+lo)*.5
    if np.any(radii<height*.03):
        audit['reason']='Degenerate neck shaft envelope'
        return np.zeros(len(points)),audit
    radial=np.linalg.norm((points[:,:2]-center)/radii,axis=1)
    z=(points[:,2]-base[2])/height
    # End the shaft transition at the atlas. The projecting chin/jaw can lie
    # below that joint, so it acquires Head ownership outside the shaft earlier.
    cranial=smoothstep(.55,1.,z)
    jaw=smoothstep(.35,.60,z)*smoothstep(1.05,1.40,radial)
    amount=np.maximum(cranial,jaw)
    audit.update(applied=True,neck_center_xy=center.tolist(),neck_radii_xy=radii.tolist(),
                 correction_min_z=float(base[2]+height*.35),rigid_head_min_z=float(top[2]),
                 transition='shaft upper 45%; projecting jaw outside measured shaft',
                 fully_head_vertices=int(np.count_nonzero(amount>=1.)),
                 affected_region_vertices=int(np.count_nonzero(amount>0.)))
    return amount,audit


def correct_head_weights(meshes,rig,surface_samples=None):
    # Deliberately scoped to skin, never infer semantic ownership for garments
    # or hair using this envelope. Explicit separated heads have their own rule.
    meshes=[o for o in meshes if o.get('binding_surface')=='skin' and o.get('binding_role')=='body']
    points=np.array([v.co[:] for o in meshes for v in o.data.vertices]).reshape((-1,3))
    neck=rig.data.bones['Neck']
    amount,audit=head_ownership(points,neck.head_local,neck.tail_local,surface_samples=surface_samples)
    if not audit['applied']:return audit
    cursor=changed=0
    for obj in meshes:
        for v in obj.data.vertices:
            t=float(amount[cursor]);cursor+=1
            if t<=0:continue
            old=weights(obj,v.index)
            if not old:continue
            new={n:w*(1-t) for n,w in old.items()}
            new['Head']=new.get('Head',0)+t
            new=dict(sorted(new.items(),key=lambda item:item[1],reverse=True)[:4])
            total=sum(new.values());new={n:w/total for n,w in new.items() if w>1e-8}
            if new!=old:assign(obj,[v.index],new);changed+=1
    audit['changed_vertices']=changed
    return audit
