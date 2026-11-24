"""Head ownership for upright characters in rig-local coordinates.

Heat supplies the neck transition. A measured neck-shaft envelope distinguishes
projecting jaw/skull surfaces from the shaft; the cranial region is rigid Head.
This semantic correction does not change geometry or expand the voxel patch.
"""
import re
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


def projecting_chin_ownership(points, head, neck, front_depth):
    """Find a projecting jaw above an anonymous neck or collar surface."""
    root = np.asarray(head.head_local)
    length = head.length
    columns = root[0] + np.linspace(-.62, .62, 39)*length
    levels = np.linspace(neck.head_local.z-.12*length, root[2]+.96*length, 121)
    profile = np.array([[front_depth(x, z) for z in levels] for x in columns])
    # A jaw projects forward suddenly as the scan passes its lower rim.
    # Search below the mouth and require a coherent ledge across the face.
    search = np.flatnonzero(levels[2:] < root[2]-.11*length)
    if not len(search):
        return np.zeros(len(points))
    delta = profile[:, :-2]-profile[:, 2:]
    delta = np.where(np.isfinite(delta), delta, 0.)
    strongest = search[np.argmax(delta[:, search], axis=1)]
    accepted = (delta[np.arange(len(columns)), strongest] > .055*length
                ) & np.isfinite(profile[:, -1])
    middle = len(columns)//2
    center = strongest[middle-2:middle+3][accepted[middle-2:middle+3]]
    if len(center) < 3 or not accepted[middle]:
        return np.zeros(len(points))
    anchor = np.median(center)
    accepted &= abs(levels[strongest]-levels[int(anchor)]) < .26*length
    first = last = middle
    while first > 0 and accepted[first-1]:
        first -= 1
    while last+1 < len(columns) and accepted[last+1]:
        last += 1
    if last-first < 8:
        return np.zeros(len(points))
    chin_levels = levels[np.minimum(strongest+2, len(levels)-1)]
    # If the lip is hidden at an outer cheek, raise the lower boundary
    # toward the atlas as it approaches the lateral edge of the face.
    for outside, edge in ((slice(0,first), first), (slice(last+1,None), last)):
        distance = np.abs(columns[outside]-columns[edge])/length
        blend = smoothstep(0., .22, distance)
        chin_levels[outside] = ((1.-blend)*chin_levels[edge]+blend*(root[2]+.07*length))
    # Avoid a step at quantized silhouette samples, especially at the lower jaw.
    chin_levels[1:-1] = .25*chin_levels[:-2] + .5*chin_levels[1:-1] + .25*chin_levels[2:]
    x, y, z = points.T
    inside = ((x >= columns[0]) & (x <= columns[-1]) &
              (z >= chin_levels.min()-.14*length) & (z <= levels[-1]))
    indices = np.flatnonzero(inside)
    xi = np.clip(np.searchsorted(columns, x[indices])-1, 0, len(columns)-2)
    xt = (x[indices]-columns[xi])/(columns[xi+1]-columns[xi])
    chin = (1-xt)*chin_levels[xi]+xt*chin_levels[xi+1]
    probe = np.maximum(z[indices], chin)
    zi = np.clip(np.searchsorted(levels, probe)-1, 0, len(levels)-2)
    zt = (probe-levels[zi])/(levels[zi+1]-levels[zi])
    depth = ((1-xt)*((1-zt)*profile[xi,zi]+zt*profile[xi,zi+1])
             + xt*((1-zt)*profile[xi+1,zi]+zt*profile[xi+1,zi+1]))
    margin = .125+.14*smoothstep(-.40,.05,(z[indices]-root[2])/length)
    side = np.minimum(x[indices]-columns[0], columns[-1]-x[indices])/length
    amount = (smoothstep(-.14,-.028,(z[indices]-chin)/length)
              *(1.-smoothstep(margin,margin+.065,(y[indices]-depth)/length))
              *smoothstep(0.,.10,side)
              *smoothstep(.22,.37,(neck.head_local.y-y[indices])/length)
              *(1.-smoothstep(.83,.96,(z[indices]-root[2])/length)))
    result = np.zeros(len(points))
    result[indices] = np.where(np.isfinite(amount), amount, 0.)
    return result


def cranial_ownership(points, head, radial_depth):
    """Trace the skull from inside so an outer collar cannot seed ownership."""
    root = np.asarray(head.head_local)
    axis = np.asarray(head.tail_local) - root
    length = head.length
    height = (points[:, 2] - root[2]) / length
    indices = np.flatnonzero(height > -.16)
    z = points[indices, 2]
    center_xy = root[:2] + np.maximum(0., (z - root[2]) / axis[2])[:, None] * axis[:2]
    radial = points[indices, :2] - center_xy
    radius = np.linalg.norm(radial, axis=1)
    angle = np.arctan2(radial[:, 0], -radial[:, 1])
    # The jaw sits lower in front while the back of the neck needs a longer
    # blend. The ear sits beside the atlas and is already on the skull.
    front = smoothstep(-.46, .62, np.cos(angle))
    rigid_height = .26 - .34 * front
    height_amount = smoothstep(rigid_height - .16, rigid_height, height[indices])
    depth = radial_depth(z, angle)
    surface_amount = 1. - smoothstep(.038, .105, (radius - depth) / length)
    near_axis = 1. - smoothstep(.78, .98, radius / length)
    # Anchor the whole cap above the ears. Folded rims can hide themselves
    # from a radial ray, producing isolated lower weights on a rigid surface.
    cap = smoothstep(.58, .72, height[indices])
    result = np.zeros(len(points))
    amount = height_amount * np.where(np.isfinite(depth), surface_amount, 0.)
    result[indices] = np.maximum(amount, cap) * near_axis
    return result


def correct_generated_faces(rig, meshes):
    """Anchor fused facial hair on generated copies with a measured chin ledge."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    head, neck = (rig.data.bones.get(n) for n in ('Head', 'Neck'))
    if head is None or neck is None or head.length < 1e-6:
        return {}
    report = {}
    for obj in meshes:
        if any(m and not re.fullmatch(r'(?:material|default)(?:[ ._\d]*)',m.name,re.I)
               for m in obj.data.materials):
            continue
        transform = rig.matrix_world.inverted_safe() @ obj.matrix_world
        points = np.asarray([transform @ v.co for v in obj.data.vertices])
        if not len(points):
            continue
        obj.data.calc_loop_triangles()
        triangles = [t.vertices[:] for t in obj.data.loop_triangles]
        if not triangles:
            continue
        tree = BVHTree.FromPolygons(points.tolist(), triangles, all_triangles=True)
        front = float(points[:,1].min()-head.length)
        reach = float(points[:,1].max()-front+head.length)

        def depth(x, z):
            p, *_ = tree.ray_cast(Vector((x,front,z)), Vector((0,1,0)), reach)
            return p.y if p is not None else np.nan

        amount = projecting_chin_ownership(points, head, neck, depth)
        # Only use the anonymous fused head pass when a forward chin ledge
        # corroborates the landmarks. It can coexist with explicit skin binding.
        if not np.any(amount):
            continue
        axis = head.tail_local - head.head_local
        if axis.z > head.length * .5:
            def radial_depth(z, angle):
                result = np.empty(len(z))
                for i, (height, phi) in enumerate(zip(z, angle)):
                    offset = max(0., (height - head.head_local.z) / axis.z)
                    center = head.head_local + axis * offset
                    center.z = height
                    direction = Vector((np.sin(phi), -np.cos(phi), 0.))
                    _, _, _, distance = tree.ray_cast(center, direction, head.length)
                    result[i] = np.nan if distance is None else distance
                return result

            amount = np.maximum(amount, cranial_ownership(points, head, radial_depth))
        fixed = np.flatnonzero(amount >= 1.-1e-7).tolist()
        if fixed:
            assign(obj, fixed, {'Head':1.})
        blend = np.flatnonzero((amount > 1e-7) & (amount < 1.-1e-7)).tolist()
        for i in blend:
            t = amount[i]
            row = {n: w*(1.-t) for n,w in weights(obj,i).items()}
            row['Head'] = row.get('Head',0.)+t
            row = dict(sorted(row.items(),key=lambda item: (-item[1],item[0]))[:4])
            total = sum(row.values())
            assign(obj,[i],{n:w/total for n,w in row.items()})
        if fixed:
            report[obj.name] = dict(rigid=len(fixed), blended=len(blend))
    return report
