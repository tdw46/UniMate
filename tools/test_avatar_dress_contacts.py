"""Check long-dress contact scopes, native export and idempotent replacement.

Run in isolated Blender: -- source.blend report.json
"""
import json,sys
from pathlib import Path
from mathutils import Vector
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated background Blender')
source,output=sys.argv[sys.argv.index('--')+1:]
for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
    if repo.module=='user_default':
        repo.use_custom_directory=True
        repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register
register()
bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_directional_contacts import install_skirt_contact_rig,skirt_owner_leg
from avatar_mesh_invariant import snapshot,verify
from avatar_apparel_weights import weights
from avatar_skirt_fit_io import rest_edit
from avatar_colliders import rest_contacts
from bl_ext.user_default.vrm.exporter.vrm1_exporter import Vrm1Exporter
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig]
geometry=snapshot(meshes)
binding={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
pose={p.name:p.matrix_basis.copy() for p in rig.pose.bones}
first=install_skirt_contact_rig(rig,meshes)
sb=rig.data.vrm_addon_extension.spring_bone1
counts=(len(sb.colliders),len(sb.collider_groups),len(bpy.data.objects))
second=install_skirt_contact_rig(rig,meshes)
assert first==second and counts==(len(sb.colliders),len(sb.collider_groups),len(bpy.data.objects))
assert all(rig.pose.bones[n].matrix_basis==m for n,m in pose.items())
hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
thighs=[getattr(hum,side+'_upper_leg').node.bone_name for side in ('left','right')]
calves={thighs[i]:getattr(hum,side+'_lower_leg').node.bone_name for i,side in enumerate(('left','right'))}
groups={g.uuid:g for g in sb.collider_groups};colliders={c.uuid:c for c in sb.colliders}
shared=set(second['shared_midline_chains']);side_count=0
hair=set();dress=set()
for s in sb.springs:
    ids={r.collider_uuid for cg in s.collider_groups for r in groups[cg.collider_group_uuid].colliders}
    if s.vrm_name.startswith('Secondary_Hair_'):hair.update(ids)
    if not s.vrm_name.startswith('Secondary_Skirt_'):continue
    dress.update(ids)
    guards={colliders[i].node.bone_name for i in ids if colliders[i].bpy_object.get('hallway_directional_guard')}
    owner=skirt_owner_leg(rig,s,thighs)
    lower=rig.data.bones[s.joints[0].node.bone_name].get('hallway_dress_lower')
    expected=set(calves.values()) if lower else set(thighs)
    if s.vrm_name in shared:
        assert expected<=guards,(s.vrm_name,guards)
        if not lower:
            crossbars=[colliders[i] for i in ids if colliders[i].bpy_object.get('hallway_directional_guard') and colliders[i].node.bone_name in thighs]
            assert len(crossbars)==2
            for c in crossbars:
                axis=Vector(c.shape.capsule.tail)-Vector(c.shape.capsule.offset)
                assert axis.length>0 and abs(axis.normalized().y)<1e-5

    else:
        other=next(n for n in thighs if n!=owner)
        assert other not in guards and calves[other] not in guards,(s.vrm_name,guards)
        side_count+=1
assert side_count and not hair&dress
if rig.get('hallway_bilateral_skirt_layout'):assert not shared
with rest_edit(rig):
    rest=rest_contacts(rig);assert not rest['contacts']
    data,lookup=Vrm1Exporter.create_spring_bone_collider_dicts([],sb,{b.name:i for i,b in enumerate(rig.data.bones)})
    for uuid in dress:
        c=colliders[uuid];entry=data[lookup[uuid]]
        assert 'capsule' in entry['shape'] and 'extensions' not in entry
        assert c.bpy_object.parent==rig and c.bpy_object.parent_bone==c.node.bone_name
        assert c.bpy_object.show_in_front and not c.bpy_object.hide_viewport and not c.bpy_object.hide_get()
verify(meshes,geometry)
assert binding=={o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
report=dict(shared_sections=sorted(shared),side_sections=side_count,native_counts=counts,
    hair_disjoint=True,portable_capsules=True,idempotent=True,pose_preserved=True,
    geometry_and_weights_unchanged=True,rest=rest)
Path(output).write_text(json.dumps(report,indent=2)+'\n')
print('DRESS_CONTACTS_OK',json.dumps(report),flush=True)
