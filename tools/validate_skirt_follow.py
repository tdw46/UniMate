"""Exercise local XYZ follow, rolled/mirrored rest frames and VRM export."""
import argparse,importlib,json,math,struct,sys
from pathlib import Path
import bpy
from mathutils import Matrix,Quaternion,Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_skirt_follow import upgrade_skirt_follow,INFLUENCE,PREFIX,set_rest_frame,set_constraint


def check_rotations(rig):
    followers=[p for p in rig.pose.bones if p.name.startswith(PREFIX)]
    saved={p.name:(p.rotation_mode,p.matrix_basis.copy()) for p in rig.pose.bones}
    results=[]
    try:
        for parent_angle in (0.,.23):
            for axis in ((1,0,0),(0,1,0),(0,0,1)):
                for angle in (-.6,.6):
                    for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
                    hips=rig.pose.bones.get('Hips')
                    if hips:hips.rotation_mode='QUATERNION';hips.rotation_quaternion=Quaternion((0,0,1),parent_angle)
                    sources={pb.constraints[0].subtarget for pb in followers}
                    for name in sources:
                        pb=rig.pose.bones[name];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion(axis,angle)
                    bpy.context.view_layer.update()
                    maximum=0.;point_error=0.
                    for follow in followers:
                        c=follow.constraints[0];leg=rig.pose.bones[c.subtarget]
                        parent=leg.parent
                        pre=parent.matrix@parent.bone.matrix_local.inverted()@leg.bone.matrix_local if parent else leg.bone.matrix_local
                        expected=pre@Quaternion(axis,angle*INFLUENCE).to_matrix().to_4x4()
                        maximum=max(maximum,max(abs(a-b) for row,old in zip(follow.matrix,expected) for a,b in zip(row,old)))
                        for child in follow.children:
                            predicted=expected@follow.bone.matrix_local.inverted()@child.bone.head_local
                            point_error=max(point_error,(predicted-child.head).length)
                    assert maximum<3e-6 and point_error<3e-6,(axis,angle,maximum,point_error)
                    results.append(dict(axis=axis,angle=angle,parent_angle=parent_angle,matrix_error=maximum,skirt_root_error=point_error))
    finally:
        for name,(mode,basis) in saved.items():rig.pose.bones[name].matrix_basis=basis;rig.pose.bones[name].rotation_mode=mode
        bpy.context.view_layer.update()
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('output');args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    out=Path(args.output).resolve();out.mkdir(exist_ok=True,parents=True)
    for repo in bpy.context.preferences.extensions.repos:
        if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.source).resolve()))
    rig=next(o for o in bpy.data.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))
    bpy.context.window.scene=next(s for s in bpy.data.scenes if rig.name in s.objects)
    keep={rig,*rig.children_recursive}
    for obj in list(bpy.data.objects):
        if obj not in keep:bpy.data.objects.remove(obj,do_unlink=True)
    bpy.context.view_layer.objects.active=rig
    if bpy.context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()
    mesh=[o for o in rig.children if o.type=='MESH']
    from rebuild_mmd_character import fingerprint
    from avatar_apparel_weights import weights
    before={o.name:(fingerprint(o),[weights(o,v.index) for v in o.data.vertices]) for o in mesh}
    followers=upgrade_skirt_follow(rig)
    assert all((fingerprint(o),[weights(o,v.index) for v in o.data.vertices])==before[o.name] for o in mesh)
    errors=check_rotations(rig)
    search=importlib.import_module('bl_ext.user_default.vrm.editor.search')
    _,accepted,problems=search.export_constraints([rig]+mesh,rig)
    assert not problems and set(accepted.rotation_constraints)=={x['bone'] for x in followers},problems
    # The same setup helper must also work on zero-length newly allocated bones,
    # arbitrary roll and opposing thigh directions, not only migrated avatars.
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for i,roll in enumerate((-.8,0.,1.1)):
        leg=rig.data.edit_bones.new('TestLeg'+str(i));leg.head=(i*.2,0,1);leg.tail=(i*.2+.04,.03,.5);leg.roll=roll
        helper=rig.data.edit_bones.new('TestFollow'+str(i));set_rest_frame(helper,leg)
        assert max(abs(a-b) for row,old in zip(helper.matrix,leg.matrix) for a,b in zip(row,old))<1e-6
    for b in list(rig.data.edit_bones):
        if b.name.startswith(('TestLeg','TestFollow')):rig.data.edit_bones.remove(b)
    bpy.ops.object.mode_set(mode='OBJECT');bpy.context.view_layer.update()
    meta=rig.data.vrm_addon_extension.vrm1.meta;meta.vrm_name='Skirt Follow Validation'
    if not meta.authors:meta.authors.add().value='Local validation'
    bpy.ops.object.select_all(action='DESELECT')
    for obj in [rig]+mesh:obj.select_set(True)
    bpy.context.view_layer.objects.active=rig
    path=out/'skirt_follow_90.vrm'
    result=bpy.ops.export_scene.vrm(filepath=str(path),armature_object_name=rig.name,export_only_selections=True,export_invisibles=True,export_gltf_animations=False,ignore_warning=True)
    assert result=={'FINISHED'},result
    raw=path.read_bytes();size,kind=struct.unpack_from('<II',raw,12);gltf=json.loads(raw[20:20+size])
    exported={n['name']:n['extensions']['VRMC_node_constraint']['constraint']['rotation'] for n in gltf['nodes'] if 'VRMC_node_constraint' in n.get('extensions',{})}
    assert set(exported)=={x['bone'] for x in followers}
    assert all(abs(c.get('weight',1)-INFLUENCE)<1e-6 for c in exported.values())
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'validated.blend'))
    for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
    assert bpy.ops.import_scene.vrm(filepath=str(path))=={'FINISHED'}
    imported=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    imported_errors=check_rotations(imported)
    _,accepted,problems=search.export_constraints(list(bpy.context.scene.objects),imported)
    assert not problems and len(accepted.rotation_constraints)==len(followers)
    report=dict(followers=followers,axis_checks=errors,reimport_axis_checks=imported_errors,exported_constraints=len(exported),geometry_and_weights_unchanged=True,fresh_rolled_helpers_passed=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print('SKIRT_FOLLOW_OK',len(exported),max(x['matrix_error'] for x in errors),max(x['matrix_error'] for x in imported_errors),flush=True)

if __name__=='__main__':main()
