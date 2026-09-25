"""Isolated regression: blender --background --factory-startup --python this.py -- source.blend output.json."""
import argparse
import json
from pathlib import Path
import sys
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_mesh_invariant import snapshot,verify
from avatar_apparel_weights import weights
from avatar_directional_contacts import install_skirt_contact_rig,skirt_owner_leg
from properties_hallway_rig import initialize


def run(source,output):
    if not bpy.app.background:raise RuntimeError('Run against an isolated scene copy')
    if hasattr(bpy.context.preferences,'extensions'):
        for repo in bpy.context.preferences.extensions.repos:
            if repo.module=='user_default':
                repo.use_custom_directory=True
                repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    try:bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    except (RuntimeError,ModuleNotFoundError):bpy.ops.preferences.addon_enable(module='VRM_Addon_for_Blender')
    bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
    candidates=[o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator')]
    if len(candidates)!=1:raise ValueError('Supply a comparison scene with one generated rig')
    rig=candidates[0];meshes=[o for o in rig.children if o.type=='MESH'];initialize(rig)
    geometry=snapshot(meshes);binding={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
    report=install_skirt_contact_rig(rig,meshes)
    if not report['segments']:raise ValueError('Fixture has no skirt springs')
    sb=rig.data.vrm_addon_extension.spring_bone1
    counts=lambda:(len(rig.data.bones),len(sb.springs),len(sb.colliders),len(sb.collider_groups))
    first=counts();install_skirt_contact_rig(rig,meshes);assert counts()==first
    hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    thighs=[getattr(hum,s+'_upper_leg').node.bone_name for s in ('left','right')]
    colliders={c.uuid:c for c in sb.colliders};groups={g.uuid:g for g in sb.collider_groups}
    from avatar_colliders import plan_colliders,rest_contacts
    from mathutils import Vector
    plan=plan_colliders(rig,meshes)
    body_radius={s['bone']:s['body_radius'] for s in plan['collider_details'] if s['role']=='skirt'}
    max_thigh_radius=max(body_radius[n] for n in thighs if n in body_radius)
    assert not rest_contacts(rig)['contacts']
    guard_uses={c.uuid:0 for c in sb.colliders if c.bpy_object.get('hallway_directional_guard')}
    for spring in sb.springs:
        if not spring.vrm_name.startswith('Secondary_Skirt_'):continue
        owner=skirt_owner_leg(rig,spring,thighs)
        guards=[colliders[x.collider_uuid] for ref in spring.collider_groups for x in groups[ref.collider_group_uuid].colliders if colliders[x.collider_uuid].bpy_object.get('hallway_directional_guard')]
        assert len(guards)==2
        assert {c.node.bone_name for c in guards}=={owner,hum.hips.node.bone_name}
        for c in guards:
            guard_uses[c.uuid]+=1
            assert c.shape.capsule.radius<=body_radius.get(c.node.bone_name,max_thigh_radius)+1e-6
            # Radius is body-sized while contact remains continuous along the
            # thigh, including guards whose attachment is transformed to Hips.
            assert (Vector(c.shape.capsule.tail)-Vector(c.shape.capsule.offset)).length<=max(rig.data.bones[n].length for n in thighs)+1e-5
    assert all(count==1 for count in guard_uses.values())
    import properties_hallway_rig as props
    assert abs(props.HALLWAY_PG_Follow.bl_rna.properties['influence'].default-.55)<1e-6
    assert abs(props.HALLWAY_PG_Spring.bl_rna.properties['drag'].default-.4)<1e-6

    verify(meshes,geometry)
    assert binding=={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
    assert len({j.node.bone_name for s in sb.springs for j in s.joints})==sum(len(s.joints) for s in sb.springs)
    own=[g for g in sb.collider_groups if g.vrm_name.startswith('Secondary_SkirtContact_')]
    assert all(g.colliders and any(ref.collider_group_uuid==g.uuid for s in sb.springs for ref in s.collider_groups) for g in own)
    group=rig.hallway_rig.spring_groups['Skirt'];old=group.stiffness
    values={s.vrm_name:s.joints[0].stiffness for s in sb.springs if s.vrm_name.startswith('Secondary_Skirt_')}
    try:
        group.stiffness=old*1.1
        assert all(abs(s.joints[0].stiffness-values[s.vrm_name]*1.1)<1e-5 for s in sb.springs if s.vrm_name in values)
    finally:group.stiffness=old
    Path(output).write_text(json.dumps(dict(counts=first,report=report,geometry_unchanged=True,weights_unchanged=True,repeat_no_duplicates=True,stiffness_callback=True),indent=2))
    print('SKIRT_CONTACT_REGRESSION_OK',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.source,a.output)
