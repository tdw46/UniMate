"""Build, audit and record secondary rigs using BVT physics.

Run with Blender --background --factory-startup --python-exit-code 1 --python
tools/demo_avatar_springs.py -- [--prepare-only] [--frames 300] [--size 1080].
Generation and bake recording run here; BVT supplies all spring simulation.
Optional VRMA clips use the official VRM importer.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT))
from avatar_springs import generate_secondary, plan_secondary, PREFIX
from avatar_apparel_weights import weights
from avatar_fullbody import solve_leg
from avatar_physics_preview import set_simulation, enabled, reset, cached_animation


def bundled_walk(rig, item_id, frames, user_library=False):
    """Import a local VRMA with the official importer and repeat its FK take."""
    from data_process.utils.blender_actions import iter_fcurves
    from avatar_springs import humanoid_roles
    path=Path(item_id).expanduser().resolve()
    if not path.is_file() or path.suffix.lower()!='.vrma':
        raise ValueError('Supply a local .vrma path; external add-on library IDs are no longer used')
    set_simulation(False)
    rig.animation_data_clear()
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.context.view_layer.update()
    result = bpy.ops.import_scene.vrma(filepath=str(path), armature_object_name=rig.name)
    assert result == {'FINISHED'}, result
    action = rig.animation_data.action
    action.use_fake_user = True
    first, last = map(round, action.frame_range)
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    names = ['Root'] + list(humanoid_roles().values())
    poses = []
    for frame in range(first, last+1):
        bpy.context.scene.frame_set(frame)
        poses.append({n:rig.pose.bones[n].matrix_basis.copy() for n in names})
    # Last frame closes this bundled loop. Retain any real forward displacement.
    period = last-first
    if frames <= 0:
        frames = len(poses)
    endpoint_delta = {n:poses[-1][n].translation-poses[0][n].translation for n in ('Root', 'Hips')}
    rig.animation_data_clear()
    for frame in range(frames):
        for name in names:
            pb = rig.pose.bones[name]
            pb.rotation_mode = 'QUATERNION'
            sample_index = min(frame, len(poses)-1) if user_library else frame % period
            pb.matrix_basis = poses[sample_index][name]
            if name in endpoint_delta and not user_library:
                pb.location += endpoint_delta[name]*(frame//period)
            for prop in ('location', 'rotation_quaternion', 'scale'):
                pb.keyframe_insert(prop, frame=frame)
    rig.animation_data.action.name = action.name+(' - body take' if user_library else ' - repeated body take')
    assert not any(PREFIX in f.data_path for f in iter_fcurves(rig.animation_data.action))
    return dict(operator='bpy.ops.import_scene.vrma', source=str(path),
                action=action.name, source_frame_range=[first, last], loop_period=None if user_library else period,
                recorded_frames=frames, fps=fps,
                endpoint_translation={n:list(v) for n, v in endpoint_delta.items()})


def drag_armature(rig, frames):
    """A static relaxed pose; only the armature object's location is animated."""
    from data_process.utils.blender_actions import iter_fcurves
    rig.animation_data_clear()
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    for side, sign in (('L', 1), ('R', -1)):
        name = 'UpperArm.'+side
        basis = rig.data.bones[name].matrix_local.to_quaternion()
        rig.pose.bones[name].rotation_mode = 'QUATERNION'
        rig.pose.bones[name].rotation_quaternion = basis.inverted() @ Quaternion((0,1,0), math.radians(sign*70)) @ basis
    start = rig.location.copy()
    # Smooth drag gestures, reversals, and a final stationary settle interval.
    stops = [(0,(0,0,0)), (.12,(-.45,0,.12)), (.28,(.45,.12,.18)),
             (.42,(-.30,-.12,.35)), (.58,(.25,0,.08)), (.72,(0,0,0)), (1,(0,0,0))]
    for frame in range(frames):
        t = frame/max(1, frames-1)
        for (a, va), (b, vb) in zip(stops, stops[1:]):
            if a <= t <= b:
                blend = (t-a)/(b-a)
                blend = blend*blend*(3-2*blend)
                rig.location = start+Vector(va).lerp(Vector(vb), blend)
                rig.keyframe_insert('location', frame=frame)
                break
    rig.animation_data.action.name = 'UniMate armature object drag only'
    assert all(f.data_path == 'location' for f in iter_fcurves(rig.animation_data.action))
    return dict(animated_channels=['armature object location'], frames=frames,
                final_settle_frames=round(frames*.28))


def fingerprint(obj):
    mesh = obj.data
    return hashlib.sha256(repr((
        [tuple(v.co) for v in mesh.vertices],
        [tuple(p.vertices) for p in mesh.polygons],
        [p.material_index for p in mesh.polygons],
        [m.name if m else None for m in mesh.materials],
        [[tuple(v.uv) for v in layer.data] for layer in mesh.uv_layers],
    )).encode()).hexdigest()


def enable(module):
    result = bpy.ops.preferences.addon_enable(module='bl_ext.user_default.'+module)
    if result != {'FINISHED'}:
        raise RuntimeError(f'Could not enable {module}: {result}')


def animate(rig, frames):
    rig.animation_data_clear()
    ankles = {s: rig.data.bones['Foot.'+s].head_local.copy() for s in ('L', 'R')}
    directions = {s: rig.data.bones['Foot.'+s].tail_local-ankles[s] for s in ankles}
    def turn(name, axis, angle):
        basis = rig.data.bones[name].matrix_local.to_quaternion()
        rig.pose.bones[name].rotation_quaternion = basis.inverted() @ Quaternion(axis, math.radians(angle)) @ basis
    max_error = 0
    for frame in range(frames):
        seconds = frame/30
        moving_end = max(1, frames/30-2)
        env = min(1, seconds/.8, max(0, moving_end-seconds)/.8)
        env = env*env*(3-2*env)
        wave = math.sin(seconds*math.tau/2.8)
        for pb in rig.pose.bones:
            pb.rotation_mode = 'QUATERNION'
            pb.matrix_basis = Matrix.Identity(4)
        root = rig.pose.bones['Root']
        root.location = rig.data.bones['Root'].matrix_local.to_quaternion().inverted() @ Vector((.085*wave*env, 0, -.07))
        turn('Hips', (0, 0, 1), 9*wave*env)
        turn('Spine', (0, 0, 1), 8*wave*env)
        turn('Chest', (0, 0, 1), 10*math.sin(seconds*math.tau/3.5)*env)
        turn('Neck', (0, 1, 0), 5*wave*env)
        turn('Head', (0, 0, 1), 26*math.sin(seconds*math.tau/2.2)*env)
        for side, sign in (('L', 1), ('R', -1)):
            turn('UpperArm.'+side, (0, 1, 0), sign*65)
            turn('Forearm.'+side, (0, 0, 1), -sign*(12+8*wave*env))
            for digit in ('Index', 'Middle', 'Ring', 'Little'):
                for joint in (1, 2, 3):
                    turn(f'{digit}{joint}.{side}', (0, sign, 0), 12)
            bpy.context.view_layer.update()
            target = ankles[side].copy()
            target.z += .07*max(0, sign*wave)*env
            target.y -= .05*sign*wave*env
            max_error = max(max_error, solve_leg(rig, side, target, directions[side]))
        for pb in rig.pose.bones:
            if pb.name.startswith(PREFIX):
                continue
            pb.keyframe_insert('rotation_quaternion', frame=frame)
            if pb.name == 'Root':
                pb.keyframe_insert('location', frame=frame)
    rig.animation_data.action.name = 'UniMate spring study - authored body'
    return max_error


def studio(scene, frames, size, rig=None, orbit=True, fps=30):
    engines = {x.identifier for x in scene.render.bl_rna.properties['engine'].enum_items}
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
    if hasattr(scene, 'eevee') and hasattr(scene.eevee, 'taa_render_samples'):
        scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps)/fps
    scene.frame_start, scene.frame_end = 0, frames-1
    scene.world = bpy.data.worlds.new('Spring study world')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.035, .045, .065, 1)
    scene.view_settings.view_transform = 'Standard'
    camera = bpy.data.objects.new('Study camera', bpy.data.cameras.new('Study camera'))
    scene.collection.objects.link(camera)
    camera.data.type = 'ORTHO'
    scene.camera = camera
    for f in range(frames):
        # Smooth orbit exposes long back hair as well as the front skirt.
        t = f/max(1, frames-1)
        angle = .25+(math.tau*t if orbit else 0)
        focus = Vector((0, 0, 1.32))
        if rig is not None:
            scene.frame_set(f)
            hips = rig.matrix_world @ rig.pose.bones['Hips'].head
            focus.x, focus.y = hips.x, hips.y
        camera.location = focus+Vector((5*math.sin(angle), -5*math.cos(angle), .30))
        camera.rotation_euler = (focus-camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera.data.ortho_scale = 3.10 if orbit else 3.65
        camera.keyframe_insert('location', frame=f)
        camera.keyframe_insert('rotation_euler', frame=f)
    for loc, power in (((-3, -4, 5), 450), ((3, -1, 4), 300), ((0, 3, 4), 450)):
        light = bpy.data.objects.new('Softbox', bpy.data.lights.new('Softbox', 'AREA'))
        scene.collection.objects.link(light)
        light.location = loc
        light.rotation_euler = (Vector((0, 0, 1.4))-light.location).to_track_quat('-Z', 'Y').to_euler()
        light.data.energy, light.data.size = power, 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(ROOT/'outputs/fullbody_avatar_grid/avatars/avatarsample_b/02_fresh_rig.blend'))
    parser.add_argument('--output', default=str(ROOT/'outputs/spring_avatar_demo'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--size', type=int, default=1080)
    parser.add_argument('--preroll', type=int, default=60)
    parser.add_argument('--bundled-animation')
    parser.add_argument('--user-library-animation')
    parser.add_argument('--drag-armature', action='store_true')
    parser.add_argument('--name', default='sample_b_springs')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=args.source)
    scene = bpy.context.scene
    scene.frame_set(0)
    rig = next(o for o in scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in scene.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)]
    for repo in bpy.context.preferences.extensions.repos:
        if repo.module == 'user_default':
            repo.use_custom_directory = True
            repo.custom_directory = str(Path.home()/'Documents/Blender/extensions/user_default')
    enable('vrm')
    enable('beyond_vrm_extension_suite')
    set_simulation(False)
    before = {o.name: fingerprint(o) for o in meshes}
    before_weights = {o.name: [weights(o, i) for i in range(len(o.data.vertices))] for o in meshes}
    plan = plan_secondary(rig, meshes)
    touched = {o.name: set() for o in meshes}
    for region in plan['regions']:
        touched[region['obj'].name].update(region['ids'])
    report = generate_secondary(rig, meshes)
    report['generator_without_bvt'] = True
    report['geometry_uv_materials_unchanged'] = all(before[o.name] == fingerprint(o) for o in meshes)
    outside = all(before_weights[o.name][i] == weights(o, i) for o in meshes for i in range(len(o.data.vertices)) if i not in touched[o.name])
    report['unrelated_weights_unchanged'] = outside
    report['maximum_weight_sum_error'] = max(abs(sum(weights(o, i).values())-1) for o in meshes for i in touched[o.name])
    assert report['geometry_uv_materials_unchanged'] and outside
    assert report['maximum_weight_sum_error'] < 1e-6
    try:
        generate_secondary(rig, meshes)
    except ValueError as exc:
        assert 'already has' in str(exc)
        report['duplicate_generation_rejected'] = True
    else:
        raise AssertionError('Duplicate generation was not rejected')
    sb = rig.data.vrm_addon_extension.spring_bone1
    assert all(j.node.bone_name in rig.data.bones for s in sb.springs for j in s.joints)
    assert all(len(s.joints) == 5 and not rig.data.bones[s.joints[-1].node.bone_name].use_deform for s in sb.springs)
    for spring in sb.springs:
        if 'Hair' in spring.vrm_name:
            first = rig.data.bones[spring.joints[0].node.bone_name]
            assert first.name.endswith('_01') and first.parent.get('unimate_fixed_hair_root')
            assert first.parent.parent.name == 'Head'
            assert first.parent.name not in {j.node.bone_name for s in sb.springs for j in s.joints}
            assert not rig.pose.bones[first.parent.name].constraints
        else:
            assert all(abs(j.drag_force-.60) < 1e-6 for j in spring.joints)
    assert all(r.collider_uuid in {c.uuid for c in sb.colliders} for g in sb.collider_groups for r in g.colliders)
    assert all(r.collider_group_uuid in {g.uuid for g in sb.collider_groups} for s in sb.springs for r in s.collider_groups)
    report['vrm_references_and_terminal_nodes_valid'] = True
    (out/'generation.json').write_text(json.dumps(report, indent=2))
    print('GENERATION_AUDIT', json.dumps(report), flush=True)
    if sum(bool(v) for v in (args.bundled_animation, args.user_library_animation, args.drag_armature)) > 1:
        raise ValueError('Choose one motion source')
    if args.bundled_animation or args.user_library_animation:
        report['bundled_animation'] = bundled_walk(rig, args.bundled_animation or args.user_library_animation,
                                                   args.frames, user_library=bool(args.user_library_animation))
        args.frames = report['bundled_animation']['recorded_frames']
    elif args.drag_armature:
        report['object_drag'] = drag_armature(rig, args.frames)
    else:
        report['maximum_ankle_target_error'] = animate(rig, args.frames)
    studio(scene, args.frames, args.size, rig=rig if args.bundled_animation or args.user_library_animation else None,
           orbit=not args.drag_armature, fps=report.get('bundled_animation', {}).get('fps', 30))
    scene.frame_set(0)
    set_simulation(True)
    assert enabled()
    report['simulation_runtime'] = 'BVT VRM_SpringSimulation'
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(out/(args.name+'.blend')))
    if args.prepare_only:
        return
    frames_dir = out/'frames'
    frames_dir.mkdir(exist_ok=True)
    scene.render.filepath = str(frames_dir)+'/'
    samples = []
    spring_names = [b.name for b in rig.data.bones if b.name.startswith(PREFIX) and b.use_deform
                    and 'SkirtFollow_' not in b.name and not b.get('unimate_fixed_hair_root')]
    spring_roles = [role for role in ('Hair', 'Skirt') if any(role in n for n in spring_names)]
    frame_samples={}
    def capture(scene):
        sample = {'frame': scene.frame_current-args.preroll, 'bones': {n: list(rig.pose.bones[n].matrix_basis.to_quaternion()) for n in spring_names}}
        sample['object_location'] = list(rig.location)
        if args.drag_armature:
            sample['body_basis'] = {p.name:[v for row in p.matrix_basis for v in row]
                                    for p in rig.pose.bones if not p.name.startswith(PREFIX)}
        sample['maximum_angle_degrees'] = {role: max(math.degrees(rig.pose.bones[n].matrix_basis.to_quaternion().angle) for n in spring_names if role in n) for role in spring_roles}
        sample['heads'] = {n:list(rig.pose.bones[n].head) for n in spring_names}
        return sample
    def record(scene, *_):
        if scene.frame_current < args.preroll:
            return
        sample=frame_samples[scene.frame_current]
        samples.append(sample)
        (out/'motion_samples.json').write_text(json.dumps(samples))
    bpy.app.handlers.render_post.append(record)
    # Let the actual render-driven solver settle at the first body pose before
    # the recorded take. Keep one continuous render so its state is not reset.
    from data_process.utils.blender_actions import iter_fcurves
    actions = {owner.animation_data.action for obj in scene.objects
               for owner in (obj, obj.data) if owner is not None
               and getattr(owner, 'animation_data', None) and owner.animation_data.action}
    def shift_actions(offset):
        for action in actions:
            for curve in iter_fcurves(action):
                for key in curve.keyframe_points:
                    key.co.x += offset
                    key.handle_left.x += offset
                    key.handle_right.x += offset
                curve.update()
    shift_actions(args.preroll)
    warmup_keys = []
    if args.preroll and args.bundled_animation:
        # Settle on preceding walk cycles rather than holding a mid-stride pose.
        # Preserve the take's phase and translation at the recording boundary.
        motion = report['bundled_animation']
        period = motion['loop_period']
        for curve in iter_fcurves(rig.animation_data.action):
            if not curve.data_path.startswith('pose.bones['):
                continue
            delta = 0.
            for name, values in motion['endpoint_translation'].items():
                if curve.data_path == f'pose.bones["{name}"].location':
                    delta = values[curve.array_index]
            values = [(frame, curve.evaluate(args.preroll+(frame-args.preroll) % period)
                       +delta*((frame-args.preroll)//period)) for frame in range(args.preroll)]
            for frame, value in values:
                curve.keyframe_points.insert(frame, value)
                warmup_keys.append((curve, frame))
            curve.update()
    scene.frame_end += args.preroll
    reset()
    scene.frame_set(0)
    recorded=[]
    with cached_animation(scene) as apply_frame:
        for frame in range(scene.frame_start,scene.frame_end+1):
            apply_frame(frame)
            recorded.append((frame,{n:list(rig.pose.bones[n].matrix_basis.to_quaternion()) for n in spring_names}))
            frame_samples[frame]=capture(scene)
    set_simulation(False)
    for frame,rotations in recorded:
        for name,rotation in rotations.items():
            pb=rig.pose.bones[name];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=rotation
            pb.keyframe_insert('rotation_quaternion',frame=frame)
    try:
        bpy.ops.render.render(animation=True)
    finally:
        bpy.app.handlers.render_post.remove(record)
        for curve, frame in warmup_keys:
            key = next(k for k in curve.keyframe_points if abs(k.co.x-frame) < 1e-5)
            curve.keyframe_points.remove(key)
        shift_actions(-args.preroll)
        scene.frame_end = args.frames-1
    # Present the recorded take with its original frame numbers.
    for frame in range(args.frames):
        source = frames_dir/f'{frame+args.preroll:04d}.png'
        target = frames_dir/f'{frame:04d}.png'
        if source != target:
            source.replace(target)
    report['simulation_preroll_frames'] = args.preroll
    report['recorded_frames'] = len(samples)
    report['maximum_spring_angle_degrees'] = {role: max(s['maximum_angle_degrees'][role] for s in samples) for role in spring_roles}
    report['spring_rotation_variation_degrees'] = {
        role:max(math.degrees(Quaternion(samples[0]['bones'][n]).rotation_difference(Quaternion(s['bones'][n])).angle)
                 for s in samples for n in spring_names if role in n) for role in spring_roles}
    if args.drag_armature:
        report['object_drag']['maximum_translation'] = max((Vector(s['object_location'])-Vector(samples[0]['object_location'])).length for s in samples)
        report['object_drag']['maximum_body_pose_change'] = max(abs(v-samples[0]['body_basis'][n][i])
            for s in samples for n, values in s['body_basis'].items() for i, v in enumerate(values))
        assert report['object_drag']['maximum_body_pose_change'] < 1e-6
        assert report['object_drag']['maximum_translation'] > .4
        assert min(report['spring_rotation_variation_degrees'].values()) > 2
    assert len(samples) == args.frames
    assert all(v > .5 for v in report['maximum_spring_angle_degrees'].values()), report
    (out/'validation.json').write_text(json.dumps(report, indent=2))
    # Save a separate portable take with the actual simulated rotations baked.
    # The editable setup remains unbaked with the requested operator enabled.
    set_simulation(False)
    rig.animation_data.action = rig.animation_data.action.copy()
    rig.animation_data.action.name = 'UniMate body + recorded spring simulation'
    from data_process.utils.blender_actions import bind_action_slot
    bind_action_slot(rig, rig.animation_data.action)
    for sample in samples:
        for name, rotation in sample['bones'].items():
            pb = rig.pose.bones[name]
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = rotation
            pb.keyframe_insert('rotation_quaternion', frame=sample['frame'])
    scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/(args.name+'_baked.blend')))
    print('SPRING_DEMO_OK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
