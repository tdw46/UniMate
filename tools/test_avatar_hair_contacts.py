"""Isolated hair refit/scope regression. Pass source.blend and report.json."""
import json
from pathlib import Path
import sys
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))


def run(source,output):
    if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
    for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
        if repo.module=='user_default':
            repo.use_custom_directory=True
            repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
    from avatar_colliders import rebuild_colliders,rest_contacts
    from avatar_mesh_invariant import snapshot,verify
    from avatar_apparel_weights import weights
    from avatar_skirt_fit_io import rest_edit
    rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig]
    geometry=snapshot(meshes);binding={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
    sb=rig.data.vrm_addon_extension.spring_bone1
    def role_ids(role):
        groups={g.uuid:g for g in sb.collider_groups}
        return {ref.collider_uuid for s in sb.springs if s.vrm_name.startswith('Secondary_'+role+'_')
                for group in s.collider_groups for ref in groups[group.collider_group_uuid].colliders}
    def specs(ids):
        return {c.uuid:(c.node.bone_name,list(c.shape.capsule.offset),list(c.shape.capsule.tail),c.shape.capsule.radius)
                for c in sb.colliders if c.uuid in ids}
    with rest_edit(rig):
        skirt=role_ids('Skirt');old=specs(skirt)
        report=rebuild_colliders(rig,meshes,collider_roles=('hair',))
        counts=(len(sb.colliders),len(sb.collider_groups),len(bpy.data.objects))
        rebuild_colliders(rig,meshes,collider_roles=('hair',))
        assert counts==(len(sb.colliders),len(sb.collider_groups),len(bpy.data.objects))
        assert not (role_ids('Hair')&role_ids('Skirt'))
        assert specs(skirt)==old
        audit=rest_contacts(rig);assert not audit['contacts']
    verify(meshes,geometry)
    assert binding=={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
    report.update(idempotent=True,hair_skirt_colliders_disjoint=True,skirt_capsules_unchanged=True,
                  mesh_and_weights_unchanged=True,rest_audit=audit)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print('HAIR_CONTACTS_REGRESSION_OK',json.dumps(report),flush=True)


if __name__=='__main__':run(*sys.argv[sys.argv.index('--')+1:])
