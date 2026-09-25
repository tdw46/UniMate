"""Experimental surface-to-segment binding for generated skirt chains.

Ordinary normalized skin weights export without a runtime extension. The fixed
waist transition is short; a segment owns its interior, with smooth blends only
around joints. This avoids pulling an entire upper segment toward fixed hips.
"""
import math
import bpy
from mathutils import Vector
from avatar_apparel_weights import weights, assign


def fit_rest_clearance(rig, parts, clearance=0.002):
    raise RuntimeError('Disabled: final mesh geometry and existing shape keys must remain unchanged')
    """Experimental radial garment fitting against the actual rest leg mesh.

    `parts` contains (object, garment triangles, leg triangles). This changes
    garment rest geometry, including every shape key by the same displacement;
    it must only be used on an explicitly prepared comparison copy.
    """
    if not bpy.app.background:
        raise RuntimeError('Experimental rest fitting is restricted to isolated background tests')
    from mathutils.bvhtree import BVHTree
    vertices=[];triangles=[]
    for obj,_,body in parts:
        transform=rig.matrix_world.inverted()@obj.matrix_world
        offset=len(vertices)
        vertices.extend(transform@v.co for v in obj.data.vertices)
        triangles.extend(tuple(i+offset for i in tri) for tri in body)
    if not triangles:return dict(vertices=0,maximum_displacement=0.)
    body=BVHTree.FromPolygons(vertices,triangles,all_triangles=True)
    hips=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.hips.node.bone_name
    center=rig.data.bones[hips].head_local
    changed=0;maximum=0.
    for obj,garment,_ in parts:
        transform=rig.matrix_world.inverted()@obj.matrix_world
        inverse=transform.inverted().to_3x3()
        ids={i for tri in garment for i in tri}
        shifts={}
        for i in ids:
            point=transform@obj.data.vertices[i].co
            origin=Vector((center.x,center.y,point.z))
            radial=point-origin;radius=radial.length
            if radius<1e-8:continue
            direction=radial/radius
            start=origin.copy();outer=0.
            for _ in range(32):
                hit,_,_,_=body.ray_cast(start,direction,10.)
                if hit is None:break
                outer=max(outer,(hit-origin).dot(direction))
                start=hit+direction*1e-5
            shifts[i]=max(0.,outer+clearance-radius) if outer else 0.
        adjacency={i:set() for i in ids}
        for tri in garment:
            for i in tri:adjacency[i].update(j for j in tri if j!=i)
        # Spread the fitted displacement without reducing any contact bound.
        for _ in range(2):
            shifts={i:max(shifts.get(i,0.),sum(shifts.get(j,0.) for j in adjacency[i])/max(1,len(adjacency[i]))) for i in ids}
        # Vertices alone can clear a convex body while an edge or face still
        # cuts through it. Test edge midpoints and face centers as well.
        originals={i:transform@obj.data.vertices[i].co for i in ids}
        directions={i:Vector((p.x-center.x,p.y-center.y,0.)).normalized() for i,p in originals.items()}
        for _ in range(4):
            updates=dict(shifts)
            for tri in garment:
                for members in (tri,tri[:2],tri[1:],(tri[0],tri[2])):
                    point=sum((originals[i]+directions[i]*shifts.get(i,0.) for i in members),Vector())/len(members)
                    origin=Vector((center.x,center.y,point.z));radial=point-origin
                    if radial.length<1e-8:continue
                    direction=radial.normalized();start=origin.copy();outer=0.
                    for _ in range(32):
                        hit,_,_,_=body.ray_cast(start,direction,10.)
                        if hit is None:break
                        outer=max(outer,(hit-origin).dot(direction));start=hit+direction*1e-5
                    deficit=outer+clearance-radial.length if outer else 0.
                    if deficit<=0:continue
                    projection=min(directions[i].dot(direction) for i in members)
                    if projection<=.25:continue
                    for i in members:updates[i]=max(updates[i],shifts[i]+deficit/projection)
            shifts=updates
        for i,amount in shifts.items():
            if amount<1e-8:continue
            original=obj.data.vertices[i].co.copy();point=transform@original
            direction=Vector((point.x-center.x,point.y-center.y,0.)).normalized()
            delta=inverse@(direction*amount)
            if obj.data.shape_keys:
                values=[(key,key.data[i].co.copy()) for key in obj.data.shape_keys.key_blocks]
                for key,value in values:key.data[i].co=value+delta
            obj.data.vertices[i].co=original+delta
            changed+=1;maximum=max(maximum,amount)
        obj.data.update()
    return dict(vertices=changed,maximum_displacement=maximum,clearance=clearance)


def rebind_skirt_strips(rig, meshes, joint_blend=0.2, waist_fraction=0.05):
    sb = rig.data.vrm_addon_extension.spring_bone1
    families = {}
    for spring in sb.springs:
        if not spring.vrm_name.startswith('Secondary_Skirt_'):
            continue
        names = [j.node.bone_name for j in spring.joints]
        points = [rig.data.bones[n].head_local.copy() for n in names]
        families.setdefault(spring.vrm_name.rsplit('_', 1)[0], []).append((names, points))
    hips = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.hips.node.bone_name
    changed = 0
    def smooth(x):
        x=max(0., min(1., x));return x*x*(3-2*x)
    for obj in meshes:
        matrix=rig.matrix_world.inverted()@obj.matrix_world
        for vertex in obj.data.vertices:
            old=weights(obj, vertex.index)
            family=next((chains for label,chains in families.items()
                         if any(n.startswith(label+'_') and w>1e-6 for n,w in old.items())),None)
            if family is None:
                continue
            p=matrix@vertex.co
            names_set={n for names,_ in family for n in names}
            amount=sum(w for n,w in old.items() if n in names_set or n==hips)
            if amount<.99:
                continue  # Ambiguous mixed bindings are not ours to rewrite.
            center=sum((points[0] for _,points in family),Vector())/len(family)
            angular=sorted((math.atan2(points[0].y-center.y,points[0].x-center.x)%math.tau,names,points)
                           for names,points in family)
            angle=math.atan2(p.y-center.y,p.x-center.x)%math.tau
            k=max((i for i,c in enumerate(angular) if c[0]<=angle),default=len(angular)-1)
            a,names_a,points_a=angular[k];b,names_b,points_b=angular[(k+1)%len(angular)]
            fraction=((angle-a)%math.tau)/((b-a)%math.tau)
            values={}
            for names,points,angular_weight in ((names_a,points_a,1-fraction),(names_b,points_b,fraction)):
                if not all(points[i].z>points[i+1].z for i in range(len(points)-1)):
                    raise ValueError('Strip binding requires descending generated skirt guides')
                index=next((i for i in range(len(points)-1) if p.z>=points[i+1].z),len(points)-2)
                local={names[index]:1.}
                for j in range(1,len(points)-1):
                    band=min(points[j-1].z-points[j].z,points[j].z-points[j+1].z)*joint_blend
                    if abs(p.z-points[j].z)<band:
                        t=smooth((points[j].z+band-p.z)/(2*band))
                        local={names[j-1]:1-t,names[j]:t}
                        break
                free=smooth((points[0].z-p.z)/((points[0].z-points[-1].z)*waist_fraction))
                local={n:w*free for n,w in local.items()}
                local[hips]=1-free
                for n,w in local.items():values[n]=values.get(n,0.)+w*angular_weight
            assign(obj,[vertex.index],values)
            changed+=1
    return {'vertices':changed,'joint_blend':joint_blend,'waist_fraction':waist_fraction}
