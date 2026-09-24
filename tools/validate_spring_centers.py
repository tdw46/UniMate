"""Check free translation response and explicitly requested compensation."""
import json,math,sys
from pathlib import Path
import bpy
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_springs import generate_secondary,set_generated_spring_center
from avatar_physics_preview import set_simulation, background_step
from rebuild_mmd_character import rename_mixamo
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/bvt_solver_only_20260924/centers'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    for repo in bpy.context.preferences.extensions.repos:
        if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'outputs/fullbody_avatar_grid/avatars/avatarsample_b/02_fresh_rig.blend'))
    rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    rig.animation_data_clear()
    for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' and m.object==rig for m in o.modifiers)]
    report=generate_secondary(rig,meshes)
    sb=rig.data.vrm_addon_extension.spring_bone1
    assert sb.springs and all(not s.center.bone_name for s in sb.springs)
    rename_mixamo(rig,meshes)  # Empty optional centers must survive renaming.
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite')
    set_simulation(False)
    for spring in sb.springs:
        spring.collider_groups.clear()
        for joint in spring.joints:joint.gravity_power=0.
    simulated={j.node.bone_name for s in sb.springs for j in list(s.joints)[:-1]}
    initial_location=rig.location.copy();results={};trajectories={};absolute_peaks={}
    for center in (None,'Hips'):
        set_generated_spring_center(rig,center)
        for motion in ('stationary','object','hips'):
            rig.location=initial_location
            for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
            bpy.context.view_layer.update()
            set_simulation(True);peak=0.;trajectory=[]
            for frame in range(90):
                delta=.08*math.sin(frame/90*math.tau)
                if motion=='object':rig.location.x=initial_location.x+delta
                elif motion=='hips':rig.pose.bones['Hips'].location.x=delta
                bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
                pose={name:rig.pose.bones[name].matrix_basis.to_quaternion().copy() for name in simulated}
                trajectory.append(pose)
                peak=max(peak,max((q.angle for q in pose.values()),default=0))
            key=f'{center or "none"}_{motion}'
            absolute_peaks[key]=peak
            trajectories[key]=trajectory
            set_simulation(False)
    # Compare matching stationary runs: measure translation response separately
    # from the external solver's initialization/settling behavior.
    for center in ('none','Hips'):
        for motion in ('object','hips'):
            reference=trajectories[center+'_stationary']
            results[center+'_'+motion]=max(abs(a[name].rotation_difference(b[name]).angle) for a,b in zip(reference,trajectories[center+'_'+motion]) for name in simulated)
    assert results['none_object']>.01 and results['none_hips']>.01,results
    # BVT owns whole-object motion semantics, including its object-delta
    # handling. The explicit ancestor center must compensate that joint's motion.
    assert results['Hips_hips']<.002,results
    # Reject misspelled explicit centers without mutating the existing setup.
    before=[s.center.bone_name for s in sb.springs]
    try:set_generated_spring_center(rig,'missing_center')
    except ValueError:pass
    else:raise AssertionError('Unknown center was accepted')
    assert before==[s.center.bone_name for s in sb.springs]
    set_generated_spring_center(rig)
    assert all(not s.center.bone_name for s in sb.springs)
    result=dict(generated_springs=len(sb.springs),default_centers_empty=True,explicit_centers_preserved=True,translation_response_radians=results,absolute_peak_radians=absolute_peaks,rename_accepts_empty_centers=True)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2))
    print('CENTER_BEHAVIOR_OK',json.dumps(result),flush=True)

if __name__=='__main__':main()
