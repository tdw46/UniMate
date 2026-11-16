"""Replace a neutral MMD deformation rig using semantic joints and fresh heat.

Keeps mesh topology, UVs, materials and every shape-key coordinate unchanged.
The source rig supplies joint landmarks only, never bones, weights or physics.
Run on an isolated scene; live application is a separate, guarded operation.
"""
from avatar_material_roles import semantic_label

import hashlib
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar_apparel_weights import assign, weights, enforce_head_cap
from avatar_springs import generate_secondary
from autorig_bust import repaired_heat
from avatar_voxel_seams import correct_skin_seams


def rename_mixamo(rig, meshes):
    """Use unprefixed Mixamo names while retaining all VRM node references."""
    mapping={'Chest':'Spine1'}
    for side,prefix in (('L','Left'),('R','Right')):
        for source,target in (('Clavicle','Shoulder'),('UpperArm','Arm'),('Forearm','ForeArm'),
                              ('Hand','Hand'),('Thigh','UpLeg'),('Shin','Leg'),('Foot','Foot'),('Toe','ToeBase')):
            mapping[source+'.'+side]=prefix+target
        for digit in ('Thumb','Index','Middle','Ring','Little'):
            for index in range(1,4):
                mapping[f'{digit}{index}.{side}']=f'{prefix}Hand{"Pinky" if digit=="Little" else digit}{index}'
    ext=rig.data.vrm_addon_extension
    from avatar_springs import humanoid_roles
    import re
    nodes=[getattr(ext.vrm1.humanoid.human_bones,re.sub(r'(?<!^)(?=[A-Z])','_',role).lower()).node
           for role in humanoid_roles()]
    nodes += [s.center for s in ext.spring_bone1.springs]
    nodes += [j.node for s in ext.spring_bone1.springs for j in s.joints]
    nodes += [c.node for c in ext.spring_bone1.colliders]
    references=[(node,node.bone_name) for node in nodes]
    optional_centers={s.center.as_pointer() for s in ext.spring_bone1.springs}
    for old,new in mapping.items():
        if old in rig.data.bones:rig.data.bones[old].name=new
    for node,old in references:node.bone_name=mapping.get(old,old)
    for obj in meshes:
        for old,new in mapping.items():
            if old in obj.vertex_groups:obj.vertex_groups[old].name=new
    for pb in rig.pose.bones:
        for constraint in pb.constraints:
            if hasattr(constraint,'subtarget'):constraint.subtarget=mapping.get(constraint.subtarget,constraint.subtarget)
    assert all(node.bone_name in rig.data.bones or
               (not node.bone_name and node.as_pointer() in optional_centers)
               for node,_ in references)
    assert all(g.name in rig.data.bones for o in meshes for g in o.vertex_groups)
    return mapping


def fingerprint(obj):
    h = hashlib.sha256()
    for v in obj.data.vertices:
        h.update(repr(tuple(v.co)).encode())
    for p in obj.data.polygons:
        h.update(repr((tuple(p.vertices), p.material_index)).encode())
    for uv in obj.data.uv_layers:
        h.update(uv.name.encode())
        for d in uv.data:
            h.update(repr(tuple(d.uv)).encode())
    keys = obj.data.shape_keys
    if keys:
        for key in keys.key_blocks:
            h.update(repr((key.name, key.value, key.vertex_group, key.relative_key.name)).encode())
            for v in key.data:
                h.update(repr(tuple(v.co)).encode())
    return h.hexdigest()


def specs_from_mmd(source):
    """Symmetrize corresponding semantic landmarks, discarding MMD controls."""
    bones = source.data.bones
    def point(name, tail=False):
        return (bones[name].tail_local if tail else bones[name].head_local).copy()
    specs = []
    def add(name, a, b, parent):
        specs.append((name, a, b, parent))
    hip = point('LowerBody'); hip.z = (point('Leg.L').z+point('Leg.R').z)/2
    height=point('Head',True).z-min(point('Ankle.L').z,point('Ankle.R').z)
    add('Root', Vector((0,0,0)), Vector((0,0,height*.07)), None)
    add('Hips', hip, point('UpperBody'), 'Root')
    add('Spine', point('UpperBody'), point('UpperBody2'), 'Hips')
    add('Chest', point('UpperBody2'), point('Neck'), 'Spine')
    add('Neck', point('Neck'), point('Head'), 'Chest')
    add('Head', point('Head'), point('Head', True), 'Neck')
    chain = [('Clavicle','Shoulder','Chest'), ('UpperArm','Arm','Clavicle'),
             ('Forearm','Elbow','UpperArm'), ('Hand','Wrist','Forearm'),
             ('Thigh','Leg','Hips'), ('Shin','Knee','Thigh'), ('Foot','Ankle','Shin'), ('Toe','Toe','Foot')]
    for side in ('L','R'):
        for target, origin, parent in chain:
            a,b = point(origin+'.'+side), point(origin+'.'+side, True)
            if target == 'Toe':
                direction = a-point('Ankle.'+side); direction.z = 0
                b = a+direction.normalized()*height*.028
            add(target+'.'+side, a,b,parent if parent in ('Chest','Hips') else parent+'.'+side)
        for target, origin in [('Thumb','Thumb'),('Index','IndexFinger'),('Middle','MiddleFinger'),('Ring','RingFinger'),('Little','LittleFinger')]:
            indices = (0,1,2) if target == 'Thumb' else (1,2,3)
            names = [origin+str(i)+'.'+side for i in indices]
            tip = ('ThumbTip' if target=='Thumb' else origin.replace('Finger','FingerTip'))+'.'+side
            for i,n in enumerate(names):
                add(f'{target}{i+1}.{side}',point(n),point(names[i+1] if i<2 else tip),
                    'Hand.'+side if i==0 else f'{target}{i}.{side}')
    lookup = {n:(a,b) for n,a,b,_ in specs}
    for name,a,b,_ in specs:
        if name.endswith('.L'):
            ra,rb = lookup[name[:-1]+'R']
            for left,right in ((a,ra),(b,rb)):
                mean=(left+Vector((-right.x,right.y,right.z)))*.5
                left[:]=mean;right[:]=(-mean.x,mean.y,mean.z)
        elif not name.endswith('.R'):
            a.x=b.x=0
    return specs


def rebuild(source, meshes, specs=None, skirt_segments=None, *, secondary=True, binding=None):
    if source is not None:
        assert all(pb.matrix_basis == Matrix.Identity(4) for pb in source.pose.bones), 'Source must be neutral'
        assert source.matrix_world == Matrix.Identity(4), 'Apply rig transforms first'
    elif specs is None:
        raise ValueError('Mesh-only generation requires predicted joints')
    assert all(o.matrix_world == Matrix.Identity(4) for o in meshes), 'Mesh transforms must match rig space'
    before = {o.name:fingerprint(o) for o in meshes}
    specs = specs_from_mmd(source) if specs is None else specs
    data=bpy.data.armatures.new('Hallway fresh humanoid')
    rig=bpy.data.objects.new((source.name if source else 'Character')+'_Hallway',data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action='DESELECT'); rig.select_set(True)
    bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name,a,b,parent in specs:
        bone=data.edit_bones.new(name);bone.head=a;bone.tail=b
        bone.parent=data.edit_bones.get(parent) if parent else None
        bone.use_deform=name!='Root'
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj in meshes:
        obj.vertex_groups.clear();obj.parent=None
        for mod in list(obj.modifiers):
            if mod.type=='ARMATURE':obj.modifiers.remove(mod)
    if source is not None: bpy.data.objects.remove(source,do_unlink=True)
    if binding is None:
        from avatar_arp_binding import bind_generated
        binding=bind_generated(rig,meshes)
    else:
        from mesh_landmarks import skin_binding
        from avatar_influence_domains import head_domains
        binding=skin_binding.apply(rig,meshes,binding,head_domains(rig,meshes))
        from avatar_hand_binding import refine_hand_weights
        binding['hand_cleanup']=refine_hand_weights(rig,meshes)
    from avatar_springs import plan_secondary
    springs=generate_secondary(rig,meshes,skirt_segments=skirt_segments,refine_cloth=False) if secondary and plan_secondary(rig,meshes)['chains'] else dict(chains=0)
    after={o.name:fingerprint(o) for o in meshes}
    assert before==after,'Geometry or shape keys changed'
    # Binding remains geometry preserving. Local attached-cloth tessellation
    # is a separate stage with exact old-vertex and interpolated surface checks.
    from avatar_cloth_overlays import refine_bound_overlays
    refine_bound_overlays(rig,meshes,springs.get('cloth_overlays',{}))
    from avatar_hinge_weights import refine_hinge_weights
    binding['hinge_cleanup'] = refine_hinge_weights(rig, meshes)
    sums=[sum(weights(o,v.index).values()) for o in meshes for v in o.data.vertices]
    assert min(sums)>.9999 and max(sums)<1.0001, (min(sums),max(sums))
    assert all(g.name in rig.data.bones for o in meshes for g in o.vertex_groups)
    return rig,dict(mesh_hashes=before,refined_mesh_hashes={o.name:fingerprint(o) for o in meshes},binding=binding,springs=springs,
                    bones=len(rig.data.bones),vertices=len(sums),weight_sum_range=[min(sums),max(sums)],
                    shape_keys={o.name:len(o.data.shape_keys.key_blocks) if o.data.shape_keys else 0 for o in meshes})


def main():
    if not bpy.app.background:
        raise RuntimeError('Build replacements in an isolated background Blender')
    directory=Path(sys.argv[sys.argv.index('--')+1]).resolve()
    inventory=json.loads((directory/'source_inventory.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str(directory/'source_scene.blend'))
    source=bpy.data.objects[inventory['active']]
    scene=next(s for s in bpy.data.scenes if source.name in s.objects)
    bpy.context.window.scene=scene
    meshes=[bpy.data.objects[n] for n in inventory['affected_meshes']]
    # A live snapshot can contain a posed or simulated source. Landmarks come
    # from its rest bones; discard only the isolated source's pose, never bake
    # that pose into the original vertices or existing shape keys.
    posed=[p.name for p in source.pose.bones if p.matrix_basis != Matrix.Identity(4)]
    source.animation_data_clear()
    for pb in source.pose.bones:pb.matrix_basis=Matrix.Identity(4)
    if scene.rigidbody_world:scene.rigidbody_world.enabled=False
    for obj in list(bpy.data.objects):
        if obj not in meshes and obj!=source:bpy.data.objects.remove(obj,do_unlink=True)
    for repo in bpy.context.preferences.extensions.repos:
        if repo.module == 'user_default':
            repo.use_custom_directory = True
            repo.custom_directory = str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    rig,report=rebuild(source,meshes)
    report['source_pose_reset_in_isolation']=posed
    rig['unimate_source']='MMD semantic rest joints; new heat weights and geometry-derived springs'
    (directory/'generated_weights.json').write_text(json.dumps({o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}))
    (directory/'generation_report.json').write_text(json.dumps(report,indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(directory/'generated.blend'))
    print('REBUILD_OK',json.dumps({k:v for k,v in report.items() if k in ('bones','vertices','weight_sum_range')}),flush=True)

if __name__=='__main__':main()
