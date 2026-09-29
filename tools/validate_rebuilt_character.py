"""Validate and package a fresh MMD rig in isolated Blender, without rendering."""
import importlib
import json
import math
from pathlib import Path
import sys
import types
import bpy
from mathutils import Matrix, Quaternion, Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from rebuild_mmd_character import fingerprint,rename_mixamo
from avatar_apparel_weights import weights
from avatar_colliders import rest_contacts
from avatar_physics_preview import set_simulation,background_step


def run(directory):
    if not bpy.app.background:raise RuntimeError('Validation requires isolated background Blender')
    directory=Path(directory).resolve()
    for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
        if repo.module=='user_default':
            repo.use_custom_directory=True
            repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    bpy.ops.wm.open_mainfile(filepath=str(directory/'generated.blend'))
    inventory=json.loads((directory/'source_inventory.json').read_text())
    source=json.loads((directory/'generation_report.json').read_text())
    rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    meshes=[bpy.data.objects[n] for n in inventory['affected_meshes']]
    assert {o.name:fingerprint(o) for o in meshes}==source['mesh_hashes']==inventory['mesh_hashes']
    rename_mixamo(rig,meshes)
    sb=rig.data.vrm_addon_extension.spring_bone1
    assert not rest_contacts(rig)['contacts']
    assert all(not s.center.bone_name for s in sb.springs)
    joints=[j.node.bone_name for s in sb.springs for j in s.joints]
    assert len(joints)==len(set(joints))
    skirt=[s for s in sb.springs if s.vrm_name.startswith('Secondary_Skirt_')]
    assert all('_Contact_' not in s.vrm_name for s in skirt)
    symmetry=0.
    for bone in rig.data.bones:
        if not bone.name.startswith('Left'):continue
        other=rig.data.bones['Right'+bone.name[4:]]
        for a,b in ((bone.head_local,other.head_local),(bone.tail_local,other.tail_local)):
            symmetry=max(symmetry,(a-Vector((-b.x,b.y,b.z))).length)
    assert symmetry<1e-6
    values={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
    assert all(w and abs(sum(w.values())-1)<1e-5 and all(n in rig.data.bones for n in w) for rows in values.values() for w in rows)
    # Load only the installed BVT solver for numerical stepping; unrelated UI
    # dependencies are unnecessary in a factory-startup validation process.
    name='bl_ext.user_default.beyond_vrm_extension_suite'
    package=types.ModuleType(name);package.__path__=[str(Path.home()/'Documents/Blender/extensions/user_default/beyond_vrm_extension_suite')]
    sys.modules[name]=package
    solver=importlib.import_module(name+'.VRM_SpringSimulation')
    for cls in (solver.BVT_OT_SpringSimulationEngine,solver.BVT_OT_SetSpringSimulation,solver.BVT_OT_SetSpringLoopPhysics):bpy.utils.register_class(cls)
    solver.register_runtime()
    bpy.context.view_layer.objects.active=rig
    # The scene snapshot may have physics enabled in its saved ID properties.
    set_simulation(False);set_simulation(True)
    changed=0;last=None
    try:
        for frame in range(180):
            pulse=math.sin(math.tau*frame/179)*math.sin(math.pi*frame/179)
            for bone,axis,angle in (('LeftUpLeg',(1,0,0),30),('RightUpLeg',(0,0,1),20),('Head',(0,1,0),20),('LeftArm',(0,0,1),25)):
                pb=rig.pose.bones[bone];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion(axis,math.radians(angle)*pulse)
            bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
            current=[tuple(rig.pose.bones[n].matrix.to_quaternion()) for n in joints]
            assert all(math.isfinite(v) for q in current for v in q)
            if last and any(abs(a-b)>1e-7 for qa,qb in zip(current,last) for a,b in zip(qa,qb)):changed+=1
            last=current
    finally:
        set_simulation(False)
        for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
        bpy.context.view_layer.update()
    assert changed>0
    assert {o.name:fingerprint(o) for o in meshes}==source['mesh_hashes']
    report=dict(geometry_and_all_shape_keys_unchanged=True,changed_frames=changed,physics_steps=180,
                mesh_count=len(meshes),vertices=sum(len(o.data.vertices) for o in meshes),
                bones=len(rig.data.bones),springs=len(sb.springs),skirt_chains=len(skirt),
                hair_chains=sum(s.vrm_name.startswith('Secondary_Hair_') for s in sb.springs),
                colliders=len(sb.colliders),rest_audit=rest_contacts(rig),humanoid_symmetry_error=symmetry,
                normalized_fresh_weights=True,unique_spring_joints=True,centers_empty=True,
                method='MMD rest landmarks, fresh Blender heat weights, generated native VRM physics; BVT simulation',
                scope='Execution and data-integrity checks only; visual and clipping review is left to the user')
    (directory/'validated_weights.json').write_text(json.dumps(values))
    (directory/'validation.json').write_text(json.dumps(report,indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(directory/'validated.blend'))
    print('REBUILT_VALIDATION_OK',json.dumps(report),flush=True)


if __name__=='__main__':run(sys.argv[sys.argv.index('--')+1])
