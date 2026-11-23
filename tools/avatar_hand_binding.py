"""Local heat cleanup for generated hands; body/secondary weights stay owned.

Full-body volumetric sampling can blur adjacent fingers and detached nails.
Use the existing split-surface heat binder on small, isolated hand crops only.
No source skeleton, source weights or Preserve Volume is used.
"""
import bpy
import numpy as np
from avatar_apparel_weights import assign, weights


def refine_hand_weights(rig, meshes):
    from avatar_arp_binding import _solve
    context = bpy.context
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active
    pose = rig.data.pose_position
    reports = []
    confinement = confine_hand_weights(rig, meshes)
    try:
        rig.data.pose_position = 'REST'
        for side in ('L', 'R'):
            hand = 'Hand.'+side; forearm = 'Forearm.'+side
            digits = {f'{digit}{n}.{side}' for digit in
                      ('Thumb', 'Index', 'Middle', 'Ring', 'Little') for n in (1,2,3)}
            allowed = digits | {hand, forearm}
            if any(n not in rig.data.bones for n in allowed):
                continue
            bone = rig.data.bones[hand]
            head = np.array(bone.head_local); axis = np.array(bone.tail_local)-head
            length = float(np.linalg.norm(axis))
            if length < 1e-8:
                continue
            axis /= length
            for obj in meshes:
                transform = rig.matrix_world.inverted_safe()@obj.matrix_world
                vertices = np.array([transform@v.co for v in obj.data.vertices])
                delta = vertices-head
                region = (np.linalg.norm(delta, axis=1)<length*3)
                region &= np.einsum('ni,i->n', delta, axis)>-length*.8
                # Existing generated ownership excludes nearby hair, clothing
                # and the other hand. This is not imported-model weight reuse.
                original = {i:weights(obj,i) for i in np.flatnonzero(region)}
                eligible = {i for i,row in original.items()
                            if sum(w for n,w in row.items() if n in allowed)>.5}
                polygons = [list(p.vertices) for p in obj.data.polygons
                            if all(i in eligible for i in p.vertices)]
                ids = sorted({i for polygon in polygons for i in polygon})
                if len(ids)<50:
                    continue
                remap = {i:n for n,i in enumerate(ids)}
                mesh = bpy.data.meshes.new('Temporary hand binding surface')
                temp = target = None
                armature = None
                try:
                    mesh.from_pydata(vertices[ids].tolist(), [],
                                     [[remap[i] for i in p] for p in polygons])
                    mesh.update()
                    temp = bpy.data.objects.new('Temporary hand binding surface', mesh)
                    context.scene.collection.objects.link(temp)
                    armature = rig.data.copy()
                    target = bpy.data.objects.new('Temporary hand binding skeleton', armature)
                    context.scene.collection.objects.link(target)
                    for b in armature.bones:
                        b.use_deform = b.name in allowed
                    rows, _ = _solve(target, temp)
                    # Retry only missing rows, using the binder's existing scale
                    # recovery. Never discard a valid first-pass surface solve.
                    missing = [i for i,row in enumerate(rows) if not row]
                    if missing:
                        retry, _ = _solve(target, temp, 20.)
                        for i in missing:
                            rows[i] = retry[i]
                    isolated=isolate_distal_digits(rig,side,vertices[ids],rows)
                    planned = [];retained=0
                    for index, row in zip(ids, rows):
                        row = {n:w for n,w in row.items() if n in allowed and w>0}
                        total = sum(row.values())
                        if not total:
                            # Optional cropped heat may fail on a newly capped
                            # contact. Keep the already validated full-body row;
                            # never replace it with an empty refinement.
                            baseline=original[index]
                            if not baseline or abs(sum(baseline.values())-1)>1e-5:
                                raise ValueError('Hand binding left vertices without weights')
                            retained+=1
                            continue
                        # Keep the wristward crop boundary and its body blending
                        # exactly as the full-body solver produced them.
                        if sum(w for n,w in row.items() if n!=forearm)<.01:
                            continue
                        planned.append((index,{n:w/total for n,w in row.items()}))
                    for index,row in planned:
                        assign(obj,[index],row)
                    reports.append(dict(side=side,object=obj.name,vertices=len(planned),
                                        crop_vertices=len(ids),scale_retry_vertices=len(missing),retained_solver_vertices=retained,isolated_digit_vertices=isolated))
                finally:
                    for item in (temp,target):
                        if item is not None:
                            bpy.data.objects.remove(item,do_unlink=True)
                    if mesh.users==0:
                        bpy.data.meshes.remove(mesh)
                    if armature is not None and armature.users==0:
                        bpy.data.armatures.remove(armature)
    finally:
        rig.data.pose_position = pose
        for obj in list(context.selected_objects):
            obj.select_set(False)
        for obj in selected:
            if obj.name in context.view_layer.objects:
                obj.select_set(True)
        if active and active.name in context.view_layer.objects:
            context.view_layer.objects.active = active
    return dict(engine='Local split-surface hand heat cleanup',hands=reports,confinement=confinement,
                source_weights_used=False,preserve_volume=False)


def isolate_distal_digits(rig, side, points, rows):
    """Remove cross-digit leakage only where rest geometry establishes an owner.

    The wrist/palm and ambiguous webbing retain their heat solution. Distances
    are scale-relative; nearest segment sampling is bounded to five small chains.
    """
    names=('Thumb','Index','Middle','Ring','Little')
    hand=rig.data.bones['Hand.'+side];scale=hand.length
    distances=[];parameters=[]
    for digit in names:
        ds=[];ts=[]
        for i in (1,2,3):
            b=rig.data.bones[f'{digit}{i}.{side}'];a=np.array(b.head_local);d=np.array(b.tail_local)-a
            t=np.clip((points-a)@d/max(float(d@d),1e-12),0,1)
            ds.append(np.linalg.norm(points-a-t[:,None]*d,axis=1));ts.append(t+i-1)
        ds=np.array(ds);best=ds.argmin(0)
        distances.append(ds.min(0));parameters.append(np.array(ts)[best,np.arange(len(points))])
    distances=np.array(distances);owner=distances.argmin(0);ordered=np.sort(distances,axis=0)
    changed=0
    for i,row in enumerate(rows):
        digit=names[owner[i]];t=float(parameters[owner[i]][i])
        if t<.6 or ordered[0,i]>scale*.13 or ordered[0,i]>ordered[1,i]*.7:continue
        foreign=[n for n in row if n.endswith('.'+side) and any(n.startswith(d) for d in names) and not n.startswith(digit)]
        leakage=sum(row[n] for n in foreign)
        if leakage<.5:continue
        # Preserve the heat solver's same-finger falloff; remove only the
        # independently disproven owner. If it gave this finger no influence,
        # use continuous segment-center interpolation as a bounded fallback.
        kept={n:w for n,w in row.items() if n not in foreign}
        own=sum(w for n,w in kept.items() if n.startswith(digit))
        if own>.02:
            for n in kept:
                if n.startswith(digit):kept[n]+=leakage*kept[n]/own
        else:
            u=float(np.clip(t-.5,0,2));a=min(int(u),1);f=u-a
            for segment,weight in [(a+1,1-f),(a+2,f)]:
                n=f'{digit}{segment}.{side}';kept[n]=kept.get(n,0)+leakage*weight
        kept=dict(sorted(((n,w) for n,w in kept.items() if w>1e-8),key=lambda x:-x[1])[:4]);total=sum(kept.values())
        rows[i]={n:w/total for n,w in kept.items()};changed+=1
    return changed


def confine_hand_weights(rig, meshes):
    """Stop a volumetric hand field from claiming a nearby welded garment.

    Keep ordinary palm/finger skin untouched. Outside a generous hand-sized
    capsule envelope, fade invalid hand mass into the remaining body solution.
    Rows containing only invalid hand influences use the nearest body segments;
    they must not be sent to the isolated finger heat solver as hand surfaces.
    """
    bones=rig.data.bones
    hands={side:[n for n in bones.keys() if n=='Hand.'+side or
                (n.endswith('.'+side) and n.startswith(('Thumb','Index','Middle','Ring','Little')))]
           for side in ('L','R') if 'Hand.'+side in bones}
    body=[b for b in bones if b.use_deform and b.name not in set(sum(hands.values(),[])) and not b.name.startswith('Secondary_')]
    def distances(points, selected):
        result=[]
        for b in selected:
            a=np.array(b.head_local);d=np.array(b.tail_local)-a
            t=np.clip((points-a)@d/max(float(d@d),1e-16),0,1)
            result.append(np.linalg.norm(points-a-t[:,None]*d,axis=1))
        return np.array(result)
    reports=[]
    for obj in meshes:
        matrix=rig.matrix_world.inverted_safe()@obj.matrix_world
        points=np.array([matrix@v.co for v in obj.data.vertices]);changed=0
        keep={}
        for side,names in hands.items():
            d=distances(points,[bones[n] for n in names]).min(0);length=bones['Hand.'+side].length
            q=np.clip((d/length-.60)/.40,0,1);keep[side]=1-q*q*(3-2*q)
        for v in obj.data.vertices:
            row=weights(obj,v.index);removed=0.
            for side,names in hands.items():
                factor=float(keep[side][v.index])
                if factor>=1:continue
                for n in row.keys()&set(names):
                    value=row[n]*(1-factor);row[n]-=value;removed+=value
            if removed<1e-6:continue
            nonhand={n:w for n,w in row.items() if n not in set(sum(hands.values(),[]))}
            total=sum(nonhand.values())
            if total>.001:
                for n,w in nonhand.items():row[n]+=removed*w/total
            else:
                d=distances(points[v.index:v.index+1],body)[:,0];ids=np.argsort(d)[:2]
                mass=1/np.maximum(d[ids],bones['Hips'].length*.05)**4;mass/=mass.sum()
                for k,w in zip(ids,mass):row[body[k].name]=row.get(body[k].name,0)+removed*float(w)
            row={n:w for n,w in row.items() if w>1e-8};total=sum(row.values())
            assign(obj,[v.index],{n:w/total for n,w in row.items()});changed+=1
        reports.append(dict(object=obj.name,confined_vertices=changed))
    return reports
