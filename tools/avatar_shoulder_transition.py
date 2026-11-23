"""Local bone-heat shoulder solve, preserving hands and the rest of the bind."""
from contextlib import contextmanager
import numpy as np
from avatar_apparel_weights import weights,assign


def refine_shoulder_heat(rig,meshes,sources=None,*,world_to_rig=None,include_root=True):
    import bpy
    from avatar_arp_binding import _solve
    from avatar_skirt_fit_io import rest_edit
    meshes=list(meshes);sources=list(sources) if sources is not None else meshes
    if len(meshes)!=len(sources):
        raise ValueError('Shoulder reference meshes must correspond one-to-one with prepared meshes')
    if not all('UpperArm.'+s in rig.data.bones for s in ('L','R')):return []
    if not any(obj.data.attributes.get('hallway_arm_bank') for obj in meshes):return []
    world_to_rig=world_to_rig if world_to_rig is not None else rig.matrix_world.inverted_safe()
    reports=[]
    with rest_edit(rig):
        target=bpy.data.objects.new('Temporary shoulder heat rig',rig.data.copy())
        bpy.context.scene.collection.objects.link(target)
        data=target.data
        try:
            for b in data.bones:
                b.use_deform=b.name in {'Spine','UpperSpine','Chest','Neck','Head','Clavicle.L','Clavicle.R','UpperArm.L','UpperArm.R','Forearm.L','Forearm.R'}
            for obj,source in zip(meshes,sources):
                if not obj.data.attributes.get('hallway_arm_bank'):continue
                to_rig=rig.matrix_world.inverted_safe()@obj.matrix_world
                points=np.array([to_rig@v.co for v in obj.data.vertices])
                blend=np.zeros(len(points));under=np.zeros(len(points))
                for side in ('L','R'):
                    b=rig.data.bones['UpperArm.'+side];a=np.array(b.head_local);d=np.array(b.tail_local)-a
                    if -d[2]<b.length*.55:continue
                    t=(points-a)@d/max(float(d@d),1e-16)
                    radius=np.linalg.norm(points-a-np.clip(t,0,1)[:,None]*d,axis=1)/b.length
                    # Continue the uncut surface heat only through the attachment.
                    # By 45% of upper-arm length, separate cut-bank ownership is
                    # exact again; broad heat transfer pulls the coat into a wing.
                    strength=np.clip((.8-radius)/.3,0,1)*np.clip((t+.5)/.2,0,1)*np.clip((.45-t)/.35,0,1)
                    blend=np.maximum(blend,strength*strength*(3-2*strength))
                    sign=1 if side=='L' else -1
                    lateral=(points[:,0]-a[0])*sign/b.length
                    depth=np.abs(points[:,1]-a[1])/b.length
                    # Use the same Blender bone-length normalization as the
                    # accepted underside solve (the cap keeps its existing frame).
                    t=(points-a)@d/(b.length*b.length)
                    # Underside only: retain the cap solve and limit body pull.
                    u=_smooth((t-.08)/.15)*(1-_smooth((t-.55)/.3))
                    u*=_smooth((lateral+.65)/.25)*(1-_smooth((lateral-.05)/.3))
                    u*=_smooth((.55-depth)/.25)*(1-_smooth((t-.4)/.2))*.5
                    under=np.maximum(under,u)
                if not include_root:blend[:]=0
                rows={int(i):weights(obj,int(i)) for i in np.flatnonzero((blend>1e-6)|(under>1e-6))}
                # A trace hip weight is not pelvis ownership: treating it as
                # an exclusion punches discontinuous holes through the blend.
                ids=[i for i,row in rows.items() if row.get('Hips',0)<.5 and not any(n.startswith(('Secondary_','Hand.','Thumb','Index','Middle','Ring','Little','Thigh.','Shin.','Foot.')) and w>1e-5 for n,w in row.items())]
                if not ids:continue
                source_transform=world_to_rig@source.matrix_world
                source_points=[source_transform@v.co for v in source.data.vertices]
                with _heat_surface(source_points,source.data.polygons,float(np.ptp(points[:,2]))) as temp:
                    from mathutils.kdtree import KDTree
                    tree=KDTree(len(temp.data.vertices))
                    for v in temp.data.vertices:tree.insert(v.co,v.index)
                    tree.balance();mapping={i:tree.find(points[i])[1] for i in ids}
                    heat,log=_solve(target,temp)
                    if any(not heat[mapping[i]] for i in ids):
                        retry,_=_solve(target,temp,20.)
                        for i in ids:
                            if not heat[mapping[i]]:heat[mapping[i]]=retry[mapping[i]]
                    changed=0
                    for i in ids:
                        if not heat[mapping[i]]:continue
                        row=rows[i];raw=heat[mapping[i]];new=raw;total=sum(new.values())
                        new={n:w/total for n,w in new.items()}
                        # Match the accepted two-stage blend, including limiting
                        # between stages, while sharing one uncut heat solve.
                        for stage,f in enumerate((float(blend[i]),float(under[i]))):
                            if f<=1e-6:continue
                            reference=new if stage==0 else raw
                            # Preserve weak heat support in the underside pass;
                            # normalizing the donor first amplifies it beyond
                            # the approved half-strength blend.
                            row={n:row.get(n,0)*(1-f)+reference.get(n,0)*f for n in row.keys()|reference.keys()}
                            row=dict(sorted(((n,w) for n,w in row.items() if w>(1e-6 if stage==0 else 1e-8)),key=lambda p:-p[1])[:4]);total=sum(row.values())
                            row={n:w/total for n,w in row.items()}
                        assign(obj,[i],row);changed+=1
                    reports.append(dict(object=obj.name,vertices=changed,missing=sum(not heat[mapping[i]] for i in ids),geometry_changed=False,bones_changed=False,distal_cut_banks_preserved=True,underside_vertices=int(np.count_nonzero(under>1e-6)),underside_strength=.5))
        finally:
            bpy.data.objects.remove(target,do_unlink=True);bpy.data.armatures.remove(data)
    return reports


@contextmanager
def _heat_surface(points,polygons,height):
    """Disposable, welded solve surface; never modify the reference datablock."""
    import bpy,bmesh
    data=bpy.data.meshes.new('Temporary shoulder heat surface');obj=None
    try:
        data.from_pydata(points,[],[tuple(p.vertices) for p in polygons]);data.update()
        bm=bmesh.new()
        try:
            bm.from_mesh(data)
            bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=max(height*1e-7,1e-9))
            bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(data)
        finally:bm.free()
        if not data.vertices:raise ValueError('Shoulder reference surface is empty')
        obj=bpy.data.objects.new('Temporary shoulder heat surface',data)
        bpy.context.scene.collection.objects.link(obj)
        yield obj
    finally:
        if obj is not None:bpy.data.objects.remove(obj,do_unlink=True)
        bpy.data.meshes.remove(data)


def _smooth(value):
    value=np.clip(value,0.,1.)
    return value*value*(3-2*value)
