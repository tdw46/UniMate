"""Conservative thin hanging-panel fallback for anonymous, fused GLB surfaces.

Semantic garments retain their existing route. This fallback admits only thin
opposed surfaces below the pelvis with a substantial connected vertical run.
Hands, feet and sleeves are protected geometrically. No vertices are moved.
"""
import math
import re
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from avatar_virtual_topology import connected_graph, protected_extremities
from avatar_apparel_weights import weights
from avatar_surface_attachments import normalized


def plan_panels(rig, obj, points, excluded, height, prefix, start_index=0):
    if any(m and not re.fullmatch(r'(?:material|default)(?:[ ._\d]*)',m.name,re.I)
           for m in obj.data.materials):
        return [], []
    hips_bone=rig.data.bones.get('Hips')
    feet=[rig.data.bones.get('Foot.'+s) for s in ('L','R')]
    feet=[bone for bone in feet if bone is not None]
    # Without pelvis and foot landmarks the conservative panel region is unknown.
    if hips_bone is None or not feet:
        return [], []
    hips=hips_bone.head_local
    floor=max(b.head_local.z for b in feet)+height*.055
    pelvis_limit = hips.z + height*.055
    anchors = [bone for name in ('Spine', 'UpperSpine')
               if (bone := rig.data.bones.get(name)) and bone.head_local.z > pelvis_limit]
    probe_limit = max((bone.head_local.z + height*.012 for bone in anchors), default=pelvis_limit)
    eligible=(points[:,2]<probe_limit)&(points[:,2]>floor)
    eligible &= ~protected_extremities(rig,points,height)
    for side in ('L','R'):
        for token in ('UpperArm.','Forearm.'):
            bone=rig.data.bones.get(token+side)
            if bone is None:continue
            a=np.array(bone.head_local);d=np.array(bone.tail_local)-a
            t=np.clip((points-a)@d/max(d@d,1e-16),0,1)
            eligible &= np.linalg.norm(points-(a+t[:,None]*d),axis=1)>height*.045
    bank=obj.data.attributes.get('hallway_arm_bank')
    if bank:
        owned=np.empty(len(points),dtype=np.int32);bank.data.foreach_get('value',owned)
        eligible &= owned==0
    eligible[list(excluded)]=False
    obj.data.calc_loop_triangles()
    triangles=np.array([t.vertices[:] for t in obj.data.loop_triangles])
    tree=BVHTree.FromPolygons(points.tolist(),triangles.tolist(),all_triangles=True)
    normal_matrix=(rig.matrix_world.inverted_safe()@obj.matrix_world).to_3x3().inverted_safe().transposed()
    seeds=set()
    for i in np.flatnonzero(eligible):
        n=(normal_matrix@obj.data.vertices[i].normal).normalized()
        hit,other,_,distance=tree.ray_cast(Vector(points[i])-n*height*1e-5,-n,height*.024)
        if hit is not None and height*.0005<distance and other.dot(n)<-.25:
            seeds.add(int(i))
    lower = eligible & (points[:,2] < pelvis_limit)
    lower_seeds = {i for i in seeds if lower[i]}
    if not lower_seeds:
        return [], []

    def thin_components(mask, roots):
        # Close only small tessellation gaps in the same thin surface.
        import heapq
        adjacency = {int(i): [] for i in np.flatnonzero(mask)}
        for edge in obj.data.edges:
            a, b = edge.vertices
            if a in adjacency and b in adjacency:
                adjacency[a].append(b)
                adjacency[b].append(a)
        distance = {i: 0. for i in roots}
        queue = [(0., i) for i in roots]
        heapq.heapify(queue)
        while queue:
            d, i = heapq.heappop(queue)
            if d != distance[i]:
                continue
            for j in adjacency[i]:
                trial = d + float(np.linalg.norm(points[j]-points[i]))
                if trial <= height*.012 and trial < distance.get(j, math.inf):
                    distance[j] = trial
                    heapq.heappush(queue, (trial, j))
        return connected_graph(obj.data, points, distance, height*.0001)[1]

    components = thin_components(lower, lower_seeds)
    extended_components = thin_components(eligible, seeds) if anchors else []
    extended_lookup = {i: index for index, component in enumerate(extended_components) for i in component}
    chains=[];regions=[]
    from avatar_garment_straps import _centerline
    from avatar_springs import _vertical_weights
    from avatar_skirt_surface import angle_weights
    for component in components:
        ids=set(component);p=points[component]
        if len(ids)<200 or np.ptp(p[:,2])<height*.065 or np.ptp(p[:,0])<height*.035:continue
        faces=triangles[np.all(np.isin(triangles,list(ids)),axis=1)]
        if len(faces)<100:continue
        top=float(np.quantile(p[:,2],.99));bottom=float(p[:,2].min())
        center=np.array(hips)[:2];angles=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])%math.tau
        guides=[];directions=[]
        for sector in range(16):
            angle=sector*math.tau/16
            delta=(angles[faces].mean(1)-angle+math.pi)%math.tau-math.pi
            # Circular means avoid a false gap at the wrap seam.
            face_angle=np.arctan2(np.sin(angles[faces]).mean(1),np.cos(angles[faces]).mean(1))
            delta=(face_angle-angle+math.pi)%math.tau-math.pi
            subset=points[faces[abs(delta)<math.tau/16*.75]]
            if not len(subset):continue
            lo=float(subset[:,:,2].min());hi=min(top,float(subset[:,:,2].max()))
            if hi-lo<height*.055:continue
            levels=np.linspace(hi-height*.001,lo+height*.001,int(np.clip(math.ceil((hi-lo)/(height*.035)),3,10))+1)
            line=_centerline(subset,levels)
            if not line:continue
            names=[f'{prefix}Accessory_SurfacePanel_{start_index+len(regions):02d}_{sector:02d}_{j:02d}' for j in range(len(line))]
            chain=dict(role='accessory',names=names,line=line,parent='Hips',garment=True,physics_start=1,anonymous_panel=True)
            guides.append(chain);directions.append(angle)
        if not guides:continue
        # Reuse the original lower guides and their sample heights. The upper
        # section grows only on the connected thin surface above each guide.
        extensions = {}
        if anchors and bottom < hips.z-height*.15:
            overlap = {}
            for i in component:
                if i in extended_lookup:
                    key = extended_lookup[i]
                    overlap[key] = overlap.get(key, 0) + 1
            if overlap:
                continuous = extended_components[max(overlap, key=overlap.get)]
                upper_faces = triangles[np.all(np.isin(triangles, continuous), axis=1)]
                directions_on_faces = np.arctan2(
                    np.sin(angles[upper_faces]).mean(1), np.cos(angles[upper_faces]).mean(1))
                for index, (guide, angle) in enumerate(zip(guides, directions)):
                    root = guide['line'][0]
                    if guide['line'][-1].z >= hips.z-height*.15:
                        continue
                    delta = (directions_on_faces-angle+math.pi) % math.tau - math.pi
                    strip = points[upper_faces[np.abs(delta) < math.tau/16*.75]]
                    if not len(strip):
                        continue
                    top = float(strip[:,:,2].max())
                    anchor = max((b for b in anchors if b.head_local.z <= top+height*.015),
                                 key=lambda b: b.head_local.z, default=None)
                    if anchor is None:
                        continue
                    top = min(top-height*.001, anchor.head_local.z)
                    span = top-root.z
                    if span < height*.035:
                        continue
                    segments = max(2, int(math.ceil(span/(height*.035))))
                    line = _centerline(strip, np.linspace(top, root.z, segments+1))
                    if not line or (line[-1]-root).length > height*.022:
                        continue
                    line[-1] = root.copy()
                    original = guide['names']
                    stem = original[0].rsplit('_',1)[0]
                    upper_names = [f'{stem}_Torso_U{n:02d}' for n in range(segments)]
                    extensions[index] = (upper_names, line, anchor.name)
        values={}
        for i in component:
            base=normalized(weights(obj,i)) or {'Hips':1.}
            row={}
            adjacent=angle_weights(directions,angles[i]) if len(guides)>1 else [(0,1.)]
            for s,f in adjacent:
                guide=guides[s];hi=guide['line'][0].z;lo=guide['line'][-1].z
                t=float(np.clip((hi-points[i,2])/(hi-lo),0,1))
                share=float(np.clip((hi-points[i,2])/(height*.045),0,1));share=share*share*(3-2*share)
                for n,w in base.items():row[n]=row.get(n,0)+f*(1-share)*w
                for n,w in _vertical_weights(guide['names'],'Hips',t,True).items():row[n]=row.get(n,0)+f*share*w
            values[i]=normalized(row)
        # Use the same continuous interpolation on both sides of a missing
        # rail. The connected surface, not rail availability, owns the seam.
        from avatar_secondary_smoothing import smooth_domain
        values,_=smooth_domain(points,[tuple(e.vertices) for e in obj.data.edges],values,iterations=16,factor=.5,tolerance=height*.0001)
        if extensions:
            for i in continuous:
                if i in ids or points[i,2] <= hips.z:
                    continue
                adjacent = angle_weights(directions, angles[i]) if len(guides)>1 else [(0, 1.)]
                base = normalized(weights(obj, i)) or {'Hips': 1.}
                row = {}
                support = 0.
                for index, factor in adjacent:
                    if index not in extensions:
                        continue
                    names, line, _ = extensions[index]
                    lo, hi = line[-1].z, line[0].z
                    z = points[i,2]
                    if not lo < z < hi:
                        continue
                    # Fade against the original waist and torso attachment.
                    f = min(1., (z-lo)/(height*.028), (hi-z)/(height*.034))
                    share = .65*factor*f*f*(3.-2.*f)
                    support += share
                    for name, weight in _vertical_weights(names+[guides[index]['names'][0]],
                                                          'Hips', (hi-z)/(hi-lo), True).items():
                        row[name] = row.get(name, 0.)+weight*share
                if support:
                    for name, weight in base.items():
                        row[name] = row.get(name, 0.)+(1.-support)*weight
                    values[i] = normalized(row)
            for index, (names, line, anchor) in extensions.items():
                guide = guides[index]
                # A separate, coincident terminal lets the lower guide keep
                # its pelvis attachment while the upper cloth can flex.
                terminal = names[0].rsplit('_', 1)[0] + '_UTip'
                chains.append(dict(role='accessory', names=names+[terminal], line=line,
                                   parent='Hips', garment=True, physics_start=1,
                                   anonymous_panel=True, coat_spine=anchor, coat_torso=True))
        chains.extend(guides)
        regions.append(dict(obj=obj,role='garment',ids=sorted(values),values=values,anonymous_panel=True))
    return chains,regions
