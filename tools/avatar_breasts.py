"""Paired frontal torso lobes from mesh geometry, with a protected center seam."""
import numpy as np
import bpy
from mathutils import Vector
from avatar_apparel_weights import weights,assign


def fit_lobes(points,chest_a,chest_b,shoulder_width,height):
    """Require two frontal peaks, a central valley and meaningful excess volume.

    Canonical inferred humanoid frame: Z up, -Y forward. Binned front envelopes
    avoid vertex-density bias and do not use semantic names or source weights.
    """
    a=np.asarray(chest_a,dtype=float);b=np.asarray(chest_b,dtype=float);cx=(a[0]+b[0])*.5
    lo=a[2]-(b[2]-a[2])*.3;hi=b[2]
    half=shoulder_width*.9
    p=np.asarray(points);p=p[(p[:,2]>=lo)&(p[:,2]<=hi)&(np.abs(p[:,0]-cx)<half)&(p[:,1]<max(a[1],b[1])+height*.06)]
    if len(p)<60:return []
    slices=[]
    for z in np.linspace(lo,hi,25):
        band=p[np.abs(p[:,2]-z)<(hi-lo)/8]
        center=band[np.abs(band[:,0]-cx)<half*.18]
        if len(center)<3:continue
        baseline=float(np.quantile(center[:,1],.2))
        row=[]
        for sign in (1,-1):
            lateral=band[((band[:,0]-cx)*sign>half*.2)&((band[:,0]-cx)*sign<half*.9)]
            if len(lateral)<8:break
            bins=[]
            for x in np.linspace(half*.25,half*.8,7):
                local=lateral[np.abs((lateral[:,0]-cx)*sign-x)<half*.13]
                if len(local)>=3:bins.append((float(np.quantile(local[:,1],.12)),x))
            if not bins:break
            front,x=min(bins)
            excess=baseline-front
            if excess<max(height*.006,(hi-lo)*.06):break
            row.append(dict(x=cx+sign*x,y=front,z=z,baseline=baseline,excess=excess))
        if len(row)==2:slices.append(row)
    # A single protruding vertex or buckle is not a pair of breast volumes.
    if len(slices)<2:return []
    specs=[]
    for side,si in (('L',0),('R',1)):
        candidates=[row[si] for row in slices];peak=max(candidates,key=lambda r:r['excess'])
        if sum(r['excess'] for r in candidates)*(hi-lo)/25 < height*height*.00015:return []
        # Separation qualifies a lobe; it must not choose the tip height.
        # The actual forwardmost supported geometry supplies that level.
        sign=1 if side=='L' else -1
        lateral=p[((p[:,0]-cx)*sign>half*.2)&((p[:,0]-cx)*sign<half*.9)]
        apex=lateral[np.argmin(lateral[:,1])]
        patch=lateral[lateral[:,1]<=apex[1]+height*.002]
        center=np.array((float(np.median(patch[:,0])),float(apex[1]),float(apex[2])))
        # Estimate the attachment loop around the lobe in the frontal plane.
        # Sampling angular sectors avoids a dense seam dominating its center.
        rx=half*.48;rz=(hi-lo)*.55
        surface=np.asarray(points)
        q=surface[((surface[:,0]-cx)*sign>half*.08)&(np.abs(surface[:,0]-cx)<half*1.25)&(surface[:,1]<max(a[1],b[1]))]
        radial=(q[:,[0,2]]-center[[0,2]])/np.array([rx,rz])
        radius=np.linalg.norm(radial,axis=1);angle=np.arctan2(radial[:,1],radial[:,0])
        loop=[]
        for theta in np.linspace(-np.pi,np.pi,16,endpoint=False):
            candidates=q[(np.abs((angle-theta+np.pi)%(2*np.pi)-np.pi)<np.pi/12)&(radius>.75)&(radius<1.3)]
            if len(candidates):loop.append(np.median(candidates,axis=0))
        if len(loop)<6:continue
        loop=np.array(loop);support=np.median(loop,axis=0)
        support[0]=center[0]
        support[2]=float(np.clip(support[2],center[2]-(hi-lo)*.2,center[2]+(hi-lo)*.2))
        support[1]=min(max(a[1],b[1]),float(np.quantile(loop[:,1],.65))+height*.008)
        if support[1]-center[1]<height*.015:continue
        specs.append(dict(side=side,head=support.tolist(),tail=center.tolist(),
                          radii=[rx*1.25,max((support[1]-center[1])*1.2,height*.025),rz*1.2],
                          center_x=float(cx),excess=peak['excess'],supporting_slices=len(slices),
                          base_loop=loop.tolist(),apex_level=float(apex[2]),
                          attachment_y=float(max(a[1],b[1]))))
    return specs if len(specs)==2 else []


def lobe_influence(points, spec):
    """Smooth attachment envelope with a strong interior, in inferred rig space.

    The pivot is inside the lobe, not its back boundary. Fade toward the torso
    axis instead, independently of the frontal-plane perimeter. A tip-centered
    3D distance incorrectly attenuates most of the breast surface twice.
    All dimensions come from the fitted mesh; no authored weights are inputs.
    """
    points = np.asarray(points, dtype=float)
    tip = np.asarray(spec['tail'], dtype=float)
    radii = np.asarray(spec['radii'], dtype=float)
    rear = float(spec['attachment_y'])
    depth = rear - tip[1]
    if depth <= 0 or np.any(radii <= 0):
        return np.zeros(len(points))
    def smooth(value):
        value = np.clip(value, 0., 1.)
        return value * value * (3. - 2. * value)
    radial = np.linalg.norm((points[:, [0, 2]] - tip[[0, 2]]) / radii[[0, 2]], axis=1)
    perimeter = smooth((1.4 - radial) / .55)
    attachment = smooth((rear - points[:, 1]) / (depth * .45))
    sign = 1 if spec['side'] == 'L' else -1
    medial = (points[:, 0] - spec['center_x']) * sign
    seam = smooth(medial / (radii[0] * .25))
    return .95 * perimeter * attachment * seam


def paired_influences(points, specs):
    """Blend the central attachment 25% body, 37.5% to each lobe.

    The seam blend is local to the frontal breast envelope; the back, neck and
    abdomen retain their body weights. Away from the narrow seam band, each
    side keeps its independent lobe envelope.
    """
    points=np.asarray(points,dtype=float)
    result=np.stack([lobe_influence(points,s) for s in specs],axis=1)
    if len(specs)!=2:return result
    def smooth(x):
        x=np.clip(x,0,1);return x*x*(3-2*x)
    cx=float(np.mean([s['center_x'] for s in specs]))
    width=min(s['radii'][0] for s in specs)*.25
    seam=1-smooth((np.abs(points[:,0]-cx)-width*.15)/(width*.85))
    supports=[]
    for s in specs:
        tip=np.asarray(s['tail']);rear=s['attachment_y']
        z=np.abs(points[:,2]-tip[2])/s['radii'][2]
        supports.append(smooth((1.4-z)/.55)*smooth((rear-points[:,1])/((rear-tip[1])*.45)))
    support=np.minimum(*supports)
    return result*(1-seam[:,None])+.375*(seam*support)[:,None]


def generate_breasts(context,rig,meshes,prefix='UniMate_', symmetric=False):
    chest=rig.data.bones.get('Chest')
    if chest is None:return []
    shoulders=[rig.data.bones.get('UpperArm.'+s) for s in ('L','R')]
    if any(b is None for b in shoulders):return []
    # The builder just placed the rig; publish inherited child transforms
    # before measuring mesh coordinates in its local frame.
    context.view_layer.update()
    local={o.name:np.array([tuple(rig.matrix_world.inverted()@o.matrix_world@v.co) for v in o.data.vertices]) for o in meshes}
    points=np.concatenate(list(local.values()));height=float(np.ptp(points[:,2]))
    width=max(abs(b.head_local.x-chest.head_local.x) for b in shoulders)
    specs=fit_lobes(points,chest.head_local,chest.tail_local,width,height)
    if symmetric and len(specs) == 2:
        from mesh_landmarks.rig_frames import symmetrize_guides
        guides = [dict(role='breast', line=[s['head'],s['tail']]) for s in specs]
        symmetrize_guides(guides, rig, height)
        for spec, guide in zip(specs, guides):
            spec['head'], spec['tail'] = map(list, guide['line'])

    if not specs:return []
    selected=list(context.selected_objects);active=context.view_layer.objects.active
    try:
        for o in context.view_layer.objects:o.select_set(False)
        rig.select_set(True);context.view_layer.objects.active=rig
        bpy.ops.object.mode_set(mode='EDIT')
        for spec in specs:
            a=Vector(spec['head']);b=Vector(spec['tail']);side=spec['side']
            bone=rig.data.edit_bones.new(prefix+'Breast.'+side);bone.head=a;bone.tail=b;bone.parent=rig.data.edit_bones['Chest']
            end=rig.data.edit_bones.new(prefix+'BreastEnd.'+side);end.head=b;end.tail=b+(b-a)*.2;end.parent=bone;end.use_connect=True;end.use_deform=False
        bpy.ops.object.mode_set(mode='OBJECT')
        for obj in meshes:
            influences = paired_influences(local[obj.name], specs)
            for index,p in enumerate(local[obj.name]):
                old=weights(obj,index)
                # Only newly generated torso weights may be redistributed.
                amount=sum(old.get(n,0.) for n in ('Chest','UpperSpine'))
                if amount<=0 or any('Secondary_' in n for n in old):continue
                fractions=influences[index]
                fraction=float(fractions.sum())
                if fraction<1e-6:continue
                values=dict(old)
                for n in ('Chest','UpperSpine'):
                    if n in values:values[n]*=1-fraction
                for spec,value in zip(specs,fractions):
                    values[prefix+'Breast.'+spec['side']]=amount*float(value)
                assign(obj,[index],values)
    finally:
        if context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        for o in context.view_layer.objects:o.select_set(o in selected)
        context.view_layer.objects.active=active
    import json
    rig['hallway_breast_geometry']=json.dumps(specs)
    return [s['side'] for s in specs]

def install_physics(rig,sides,meshes,*,bone_prefix='UniMate_'):
    """Public VRM schema; independent of BVT's simulation implementation."""
    height=max(v.co.z for o in meshes for v in o.data.vertices)-min(v.co.z for o in meshes for v in o.data.vertices)
    sb=rig.data.vrm_addon_extension.spring_bone1
    for side in sides:
        spring=sb.springs.add();spring.vrm_name='Secondary_Breast_'+side
        for i,name in enumerate((bone_prefix+'Breast.'+side,bone_prefix+'BreastEnd.'+side)):
            joint=spring.joints.add();joint.node.bone_name=name
            joint.stiffness=1.-i*.1;joint.drag_force=.25
            joint.gravity_power=.01;joint.gravity_dir=(0,0,-1)
            joint.hit_radius=height*.01
    if sides:
        from avatar_colliders import rebuild_colliders
        rebuild_colliders(rig,meshes,collider_roles=('breast',))
