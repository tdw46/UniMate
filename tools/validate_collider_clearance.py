"""Numerical rest-clearance and moving-leg checks, using the BVT runtime."""
import argparse,json,math,sys
from pathlib import Path
import bpy
from mathutils import Matrix,Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_colliders import rebuild_colliders,rest_contacts
from avatar_physics_preview import set_simulation, background_step


def exercise(rig):
    original={p.name:p.matrix_basis.copy() for p in rig.pose.bones}
    bones=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    legs=[getattr(bones,side+'_upper_leg').node.bone_name for side in ('left','right')]
    samples={};reports={}
    sb=rig.data.vrm_addon_extension.spring_bone1
    refs=[[r.collider_group_uuid for r in spring.collider_groups] for spring in sb.springs]
    pairs=[(head.node.bone_name,tail.node.bone_name) for spring in sb.springs for head,tail in zip(spring.joints,spring.joints[1:])]
    def restore_groups(collisions=True):
        for spring,ids in zip(sb.springs,refs):
            spring.collider_groups.clear()
            if collisions:
                for uuid in ids:spring.collider_groups.add().collider_group_uuid=uuid
    def neutral():
        for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
        bpy.context.view_layer.update()
    try:
        for moving in (False,True):
            for collisions in (False,True):
                set_simulation(False);neutral();restore_groups(collisions);set_simulation(True);trajectory=[]
                for frame in range(150):
                    if moving:
                        for side,name in enumerate(legs):
                            if name:
                                pb=rig.pose.bones[name];rest=pb.bone.matrix_local.to_quaternion()
                                angle=.85*math.sin(frame/150*math.tau)*(1 if side==0 else -1)
                                pb.rotation_mode='QUATERNION';pb.rotation_quaternion=rest.inverted()@Quaternion((1,0,0),angle)@rest
                    bpy.context.view_layer.update()
                    background_step(1/30);bpy.context.view_layer.update()
                    trajectory.append({name:(rig.matrix_world@rig.pose.bones[next_name].head).copy()
                                       for name,next_name in pairs})
                samples[moving,collisions]=trajectory
                reports[f'{"moving" if moving else "rest"}_{"collisions" if collisions else "free"}']={
                    'maximum_angle':max((rig.pose.bones[n].matrix_basis.to_quaternion().angle for n,_ in pairs),default=0)}
                set_simulation(False)
        def difference(moving):
            return max(((p-samples[moving,False][i][name]).length for i,frame in enumerate(samples[moving,True]) for name,p in frame.items()),default=0)
        reports['rest_collision_displacement']=difference(False)
        reports['moving_collision_displacement']=difference(True)
        height=max(b.head_local.z for b in rig.data.bones)-min(b.head_local.z for b in rig.data.bones)
        assert reports['rest_collision_displacement']<height*.003,reports
        if any('Skirt' in s.vrm_name for s in rig.data.vrm_addon_extension.spring_bone1.springs):
            assert reports['moving_collision_displacement']>height*.001,reports
        return reports
    finally:
        set_simulation(False);restore_groups()
        for p in rig.pose.bones:p.matrix_basis=original[p.name]
        bpy.context.view_layer.update()


def main():
    args=argparse.ArgumentParser();args.add_argument('source');args.add_argument('output');args=args.parse_args(sys.argv[sys.argv.index('--')+1:])
    for repo in bpy.context.preferences.extensions.repos:
        if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.source).resolve()))
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite')
    set_simulation(False)
    rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))
    meshes=[o for o in rig.children if o.type=='MESH']
    for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()
    report={'fit':rebuild_colliders(rig,meshes),'clearance':rest_contacts(rig)}
    assert not report['clearance']['contacts']
    report['physics']=exercise(rig)
    Path(args.output).write_text(json.dumps(report,indent=2))
    print('CLEARANCE_AND_MOTION_OK',json.dumps(report['physics']),flush=True)

if __name__=='__main__':main()
