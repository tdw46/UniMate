"""Local surface ownership for small hair islands missed by guide cuts."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
from avatar_apparel_weights import components


def attach_satellites(obj,points,allowed,regions,height):
    from avatar_springs import _vertical_weights
    owners={i:r for r in regions if r['obj']==obj and r['role']=='hair' for i in r['ids']}
    if not owners:return None
    obj.data.calc_loop_triangles();faces=[]
    for tri in obj.data.loop_triangles:
        ids=tuple(tri.vertices);owned=[owners[i] for i in ids if i in owners]
        if len(owned)>=2 and all(r is owned[0] for r in owned) and all(i in allowed for i in ids):faces.append(ids)
    if not faces:return None
    surface=BVHTree.FromPolygons(points.tolist(),faces,all_triangles=True)
    candidates=[];isolated=[]
    for component in components(obj.data):
        if len(component)<=200 and set(component)<=allowed:
            candidates.extend(i for i in component if i not in owners)
            if len(component)<=32 and max(np.ptp(points[list(component)],axis=0))<height*.06:isolated.append(list(component))
    values={}
    for i in candidates:
        location,_,face,distance=surface.find_nearest(Vector(points[i]))
        if location is None or distance>height*.012:continue
        tri=faces[face]
        a,b,c=(Vector(points[j]) for j in tri)
        if (b-a).cross(c-a).length_squared<1e-20:
            donor=min(tri,key=lambda j:float(np.linalg.norm(points[j]-points[i])))
            factors=[1. if j==donor else 0. for j in tri]
        else:
            factors=np.maximum(np.asarray(barycentric_transform(location,a,b,c,Vector((1,0,0)),Vector((0,1,0)),Vector((0,0,1)))),0)
            factors/=max(float(sum(factors)),1e-12)
        result={}
        for j,factor in zip(tri,factors):
            region=owners.get(j)
            donor=(_vertical_weights(region['names'][0],'Head',(region['hi']-points[j,2])/(region['hi']-region['lo']),True)
                   if region else {'Head':1.})
            for name,weight in donor.items():result[name]=result.get(name,0.)+float(factor)*weight
        if sum(w for n,w in result.items() if n!='Head')>1e-8:values[i]=result
    # A tiny floating patch is one attachment, not several unrelated nearest
    # points. Keep its vertices together even if one projects onto the cap.
    for component in isolated:
        donors=[]
        for i in component:
            if i in values:donors.append(values[i])
            elif i in owners:
                r=owners[i]
                donors.append(_vertical_weights(r['names'][0],'Head',(r['hi']-points[i,2])/(r['hi']-r['lo']),True))
        if len(donors)<len(component)*.8:continue
        blended={}
        for donor in donors:
            for name,weight in donor.items():blended[name]=blended.get(name,0.)+weight/len(donors)
        for i in component:values[i]=dict(blended)
    return dict(obj=obj,role='hair',ids=sorted(values),values=values) if values else None
