"""Isolated native collider lifecycle/export regression on a generated rig.

blender --background --factory-startup --python tools/test_avatar_skirt_ceiling.py -- source.blend report.json
"""
import json
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Quaternion, Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run(source, output):
    if not bpy.app.background:
        raise RuntimeError('Use isolated background Blender')
    for repo in getattr(getattr(bpy.context.preferences, 'extensions', None), 'repos', []):
        if repo.module == 'user_default':
            repo.use_custom_directory = True
            repo.custom_directory = str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    from properties_hallway_rig import register, skirt_colliders
    register()
    bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
    from avatar_skirt_ceiling import install_skirt_ceiling, plan_ceiling, GROUP_PREFIX
    from avatar_skirt_fit_io import rest_edit
    from avatar_colliders import rest_contacts
    from avatar_mesh_invariant import snapshot, verify
    from avatar_apparel_weights import weights
    from bl_ext.user_default.vrm.exporter.vrm1_exporter import Vrm1Exporter
    rig = next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))
    meshes = [o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig]
    geometry = snapshot(meshes)
    binding = {o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
    sb = rig.data.vrm_addon_extension.spring_bone1
    with rest_edit(rig):
        plans = plan_ceiling(rig)
        for plan in plans:
            springs = [s for s in sb.springs if s.vrm_name in plan['springs']]
            clearance = max(j.hit_radius for s in springs for j in s.joints[:-1]) + plan['margin']
            assert plan['height'] >= plan['garment_top'] + clearance - 1e-6
        # Old rigs lack garment bounds; their root attachment must still
        # define a safe minimum. Restore authored metadata after the check.
        roots = [rig.data.bones[s.joints[0].node.bone_name] for s in sb.springs
                 if s.vrm_name.startswith('Secondary_Skirt_') and len(s.joints)>1]
        bounds = {b.name:b.get('hallway_garment_top') for b in roots}
        try:
            for b in roots:
                if 'hallway_garment_top' in b:del b['hallway_garment_top']
            for plan in plan_ceiling(rig):
                assert plan['height'] >= plan['garment_top'] + plan['margin'] - 1e-6
        finally:
            for b in roots:
                if bounds[b.name] is not None:b['hallway_garment_top']=bounds[b.name]
        first = install_skirt_ceiling(rig)
        counts = (len(sb.colliders), len(sb.collider_groups), len(bpy.data.objects))
        second = install_skirt_ceiling(rig)
        assert first == second and counts == (len(sb.colliders),len(sb.collider_groups),len(bpy.data.objects))
        audit = rest_contacts(rig)
        assert not audit['contacts']
        groups = [g for g in sb.collider_groups if g.vrm_name.startswith(GROUP_PREFIX)]
        ids = {c.collider_uuid for g in groups for c in g.colliders}
        roof = [c for c in sb.colliders if c.uuid in ids]
        assert not (ids & {c.uuid for c in skirt_colliders(rig)})  # Thickness cannot open roof gaps.
        for group in groups:
            owners = [s.vrm_name for s in sb.springs if any(r.collider_group_uuid==group.uuid for r in s.collider_groups)]
            assert len(owners)==1 and owners[0].startswith('Secondary_Skirt_')
        used=[]
        data, lookup = Vrm1Exporter.create_spring_bone_collider_dicts(used,sb,{b.name:i for i,b in enumerate(rig.data.bones)})
        assert all('capsule' in data[lookup[c.uuid]]['shape'] and 'extensions' not in data[lookup[c.uuid]] for c in roof)
        native = {c.uuid:(Vector(c.shape.capsule.offset),Vector(c.shape.capsule.tail),c.shape.capsule.radius) for c in roof}
    hips = rig.pose.bones[rig.data.vrm_addon_extension.vrm1.humanoid.human_bones.hips.node.bone_name]
    basis, world, position = hips.matrix_basis.copy(), rig.matrix_world.copy(), rig.data.pose_position
    try:
        rig.data.pose_position='POSE'
        hips.matrix_basis=Matrix.Translation((.02,.03,.06))@Quaternion((1,0,0),.3).to_matrix().to_4x4()
        rig.location.z+=.2
        bpy.context.view_layer.update()
        for c in roof:
            a,b,r=native[c.uuid]
            matrix=rig.matrix_world@hips.matrix
            assert (c.bpy_object.matrix_world.translation-matrix@a).length<1e-5
            assert (c.bpy_object.children[0].matrix_world.translation-matrix@b).length<1e-5
            assert abs(c.shape.capsule.radius-r)<1e-6
            assert c.bpy_object.show_in_front and not c.bpy_object.hide_viewport and not c.bpy_object.hide_get()
    finally:
        hips.matrix_basis=basis;rig.matrix_world=world;rig.data.pose_position=position;bpy.context.view_layer.update()
    verify(meshes,geometry)
    assert binding=={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
    report=dict(setup=second,idempotent=True,geometry_and_weights_unchanged=True,
                portable_capsule_export=True,pose_attachment_verified=True,rest_audit=audit)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print('SKIRT_CEILING_LIFECYCLE_OK',json.dumps(report),flush=True)


if __name__=='__main__':run(*sys.argv[sys.argv.index('--')+1:])
