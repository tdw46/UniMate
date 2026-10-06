"""Experimental surface-to-segment binding for generated skirt chains.

Ordinary normalized skin weights export without a runtime extension. A fixed
top band anchors a smooth body/spring overlap above and below the chain roots.
Each segment owns its interior with local joint blends and at most two chains.
"""
import math
import bpy
from mathutils import Vector
from avatar_apparel_weights import weights, assign
from avatar_waist_attachment import WaistAttachment, WAIST_FRACTION, TRANSITION_FRACTION


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


def rebind_skirt_strips(rig, meshes, joint_blend=0.2, waist_fraction=WAIST_FRACTION, transition_fraction=TRANSITION_FRACTION, regions=None, knee_joint_blend=0.4, smooth_iterations=20, smooth_factor=.2):
    from avatar_dress import full_chain_names
    sb = rig.data.vrm_addon_extension.spring_bone1
    if not 0 < joint_blend <= .5 or not 0 < knee_joint_blend <= .5:
        raise ValueError('Joint blend fractions must be in (0, 0.5] to avoid overlapping bands')
    families = {}
    knee_joints = {}
    for spring in sb.springs:
        if not spring.vrm_name.startswith('Secondary_Skirt_'):
            continue
        names = full_chain_names(rig,spring)
        if names is None:continue
        knee_joints[names[0]] = rig.data.bones[names[0]].get('hallway_dress_knee_index')
        points = [rig.data.bones[n].head_local.copy() for n in names]
        families.setdefault(spring.vrm_name.rsplit('_', 1)[0], []).append((names, points))
    from avatar_skirt_support import ensure_support, PREFIX as SUPPORT_PREFIX
    support = ensure_support(rig, meshes, families) if families else None
    attachment = WaistAttachment(rig, meshes)
    bounds={}
    for label,family in families.items():
        root=rig.data.bones[family[0][0][0]]
        top=root.get('hallway_garment_top',max(points[0].z for _,points in family))
        bottom=root.get('hallway_garment_bottom',min(points[-1].z for _,points in family))
        old_start=top-waist_fraction*(top-bottom)
        first_segment=min(points[0].z-points[1].z for _,points in family)
        lift=min((top-old_start)*.65,first_segment*.25)
        bounds[label]=(top,bottom,old_start,first_segment,lift)
    explicit = {(r['obj'].name, i): r['names'][0][0].rsplit('_', 2)[0]
                for r in (regions or []) for i in r['ids']}
    changed = 0
    attached = 0
    domains = {}
    handoffs = []
    def smooth(x):
        x=max(0., min(1., x));return x*x*(3-2*x)
    for obj in meshes:
        matrix=rig.matrix_world.inverted()@obj.matrix_world
        candidates=[]
        for vertex in obj.data.vertices:
            old=weights(obj, vertex.index)
            label = explicit.get((obj.name, vertex.index))
            if label is None:
                label = next((label for label in families
                              if any((n.startswith(label+'_') or (n in rig.data.bones and str(rig.data.bones[n].get('hallway_support_chain','')).startswith(label+'_'))) and w>1e-6 for n,w in old.items())), None)
            if label not in families:
                continue
            family = families[label]
            p=matrix@vertex.co
            names_set={n for names,_ in family for n in names}
            amount=sum(w for n,w in old.items() if n in names_set or n in attachment.allowed or n.startswith(SUPPORT_PREFIX))
            if (obj.name, vertex.index) not in explicit and amount<.99:
                continue  # Ambiguous mixed bindings are not ours to rewrite.
            candidates.append((vertex.index,label,p))
        # Previously only already-spring-weighted vertices were candidates,
        # making it impossible to blend upward into the transferred bodice.
        # Grow within the same connected garment/material and bounded height.
        from collections import deque
        adjacency={}
        for edge in obj.data.edges:
            a,b=edge.vertices
            adjacency.setdefault(a,[]).append(b);adjacency.setdefault(b,[]).append(a)
        owned={i for i,_,_ in candidates}
        for label in sorted({row[1] for row in candidates}):
            top,bottom,old_start,first_segment,lift=bounds[label]
            seed={i for i,name,_ in candidates if name==label}
            slots={f.material_index for f in obj.data.polygons if any(i in seed for i in f.vertices)}
            allowed={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}
            queue=deque(seed);seen=set(seed)
            while queue:
                for i in adjacency.get(queue.popleft(),()):
                    if i in seen:continue
                    seen.add(i)
                    if i not in allowed or i in owned:continue
                    p=matrix@obj.data.vertices[i].co
                    if not old_start-1e-6<=p.z<=top:continue
                    old=weights(obj,i)
                    if sum(w for n,w in old.items() if n in attachment.allowed)<.99:continue
                    candidates.append((i,label,p));owned.add(i);queue.append(i)
        blend_mass={}
        for label in sorted({row[1] for row in candidates}):
            family=families[label]
            rows=[(i,p) for i,name,p in candidates if name==label]
            ids={i for i,p in rows};coords=dict(rows)
            top,bottom,old_start,first_segment,lift=bounds[label]
            start=old_start+lift
            edges=[tuple(e.vertices) for e in obj.data.edges if all(i in ids for i in e.vertices)]
            steps=[abs(coords[a].z-coords[b].z) for a,b in edges
                   if min(coords[a].z,coords[b].z)>old_start-(top-bottom)*.15 and max(coords[a].z,coords[b].z)<=old_start
                   and abs(coords[a].z-coords[b].z)>(top-bottom)*1e-4]
            import statistics
            # Cover several actual mesh rings, bounded before the next spring
            # joint. Widen the scalar handoff, never the chain neighborhood.
            span=max(transition_fraction*(top-bottom),4*statistics.median(steps) if steps else 0.)
            span=min(span,first_segment*.55)+lift
            span=max(span,(top-bottom)*1e-6)
            seed={i:smooth((start-p.z)/span) for i,p in rows}
            from avatar_weight_smoothing import smooth_attachment_mass
            mass=smooth_attachment_mass(seed,edges,iterations=smooth_iterations,factor=smooth_factor)
            blend_mass.update(mass)
            handoffs.append(dict(object=obj.name,family=label,start=start,span=span,
                                 nominal_span=transition_fraction*(top-bottom),upward_overlap=lift,previous_start=old_start))
        for vertex_index,label,p in candidates:
            family=families[label]
            domains.setdefault(obj.name,{}).setdefault(label,[]).append(vertex_index)
            free=blend_mass[vertex_index]
            body = attachment.sample(p) if free < 1. else {}
            if free == 0.:
                assign(obj, [vertex_index], body)
                changed += 1
                attached += 1
                continue
            center=sum((points[0] for _,points in family),Vector())/len(family)
            angular=sorted((math.atan2(points[0].y-center.y,points[0].x-center.x)%math.tau,names,points)
                           for names,points in family)
            angle=math.atan2(p.y-center.y,p.x-center.x)%math.tau
            k=max((i for i,c in enumerate(angular) if c[0]<=angle),default=len(angular)-1)
            a,names_a,points_a=angular[k];b,names_b,points_b=angular[(k+1)%len(angular)]
            fraction=((angle-a)%math.tau)/((b-a)%math.tau)
            if attachment.lateral:
                side_a=attachment.lateral_coordinate(points_a[0])
                side_b=attachment.lateral_coordinate(points_b[0])
                if side_a*side_b<0:
                    # Front/back bridge is physically narrow even on a flared
                    # hem; angular interpolation otherwise widens with radius.
                    lateral=attachment.lateral_coordinate(p)
                    fraction=smooth(.5+.5*lateral/attachment.midline_width*(1 if side_b>0 else -1))
            values={}
            for names,points,angular_weight in ((names_a,points_a,1-fraction),(names_b,points_b,fraction)):
                if not all(points[i].z>points[i+1].z for i in range(len(points)-1)):
                    raise ValueError('Strip binding requires descending generated skirt guides')
                index=next((i for i in range(len(points)-1) if p.z>=points[i+1].z),len(points)-2)
                local={names[index]:1.}
                for j in range(1,len(points)-1):
                    # Broaden only the thigh/calf spring junction. Keep the
                    # cloth spring-weighted and the fixed waist band intact.
                    blend=knee_joint_blend if j==knee_joints[names[0]] else joint_blend
                    band=min(points[j-1].z-points[j].z,points[j].z-points[j+1].z)*blend
                    if abs(p.z-points[j].z)<band:
                        t=smooth((points[j].z+band-p.z)/(2*band))
                        local={names[j-1]:1-t,names[j]:t}
                        break
                local={n:w*free for n,w in local.items()}
                for n,w in body.items():local[n]=local.get(n,0.)+w*(1-free)
                for n,w in local.items():values[n]=values.get(n,0.)+w*angular_weight
            assign(obj,[vertex_index],values)
            changed+=1
    from avatar_weight_smoothing import smooth_skirt_weights
    smoothing=smooth_skirt_weights(rig,meshes,domains,iterations=smooth_iterations,factor=smooth_factor) if smooth_iterations else None
    return {'support':support,'handoffs':handoffs,'smoothing':smoothing,'vertices':changed,'joint_blend':joint_blend,'knee_joint_blend':knee_joint_blend,'waist_fraction':waist_fraction,
            'transition_fraction':transition_fraction,'fully_attached_vertices':attached,
            'body_surface_samples':attachment.sampled,'humanoid_fallback_samples':attachment.fallback}
