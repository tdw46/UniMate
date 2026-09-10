"""Create original generic skinned busts and deterministic diagnostic clips.

Run only in a disposable Blender process with --factory-startup.
These animations are authored fixtures, not UniMate model predictions.
"""
import math
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/bust_evaluation'
OUT.mkdir(parents=True, exist_ok=True)
FRAMES = 120


def material(name, color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    shader = m.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = 0.42
    return m


def ellipsoid(name, location, scale, mat=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=20, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat:
        obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True
    return obj


def make_bust(name, broad=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = FRAMES - 1
    width = 1.15 if broad else 1.0
    skin = material('Porcelain ' + name, (0.15, 0.53, 0.58) if not broad else (0.73, 0.34, 0.16))
    dark = material('Facial landmarks', (0.025, 0.035, 0.05))
    parts = []
    for label, loc, scale in [
        ('Torso', (0, 0, 1.55), (.63*width, .32, .83)),
        ('Chest', (0, 0, 2.04), (.71*width, .35, .42)),
        ('Neck', (0, 0, 2.48), (.19, .19, .37)),
        ('Cranium', (0, 0, 2.98), (.32*width, .29, .43)),
        ('Jaw', (0, -.07, 2.78), (.26*width, .24, .22)),
        ('Nose', (0, -.285, 2.99), (.072, .12, .115)),
    ]:
        parts.append(ellipsoid(label, loc, scale))
    for side in (-1, 1):
        for label, x, z, scale in [
            ('Shoulder', .69*width, 2.20, (.29, .28, .28)),
            ('Upper arm', 1.00*width, 2.20, (.43*width, .18, .19)),
            ('Elbow', 1.32*width, 2.20, (.17, .17, .17)),
            ('Forearm', 1.59*width, 2.20, (.35*width, .145, .155)),
            ('Hand', 1.95*width, 2.20, (.23, .12, .105)),
            ('Ear', .315*width, 2.97, (.07, .08, .14)),
        ]:
            parts.append(ellipsoid(label, (side*x, 0, z), scale))
    bpy.ops.object.select_all(action='DESELECT')
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    body = bpy.context.object
    body.name = name + '_Skin'
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    remesh = body.modifiers.new('Continuous mannequin surface', 'REMESH')
    remesh.mode = 'VOXEL'
    remesh.voxel_size = .045
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    smooth = body.modifiers.new('Smooth union seams', 'SMOOTH')
    smooth.factor = 1.1
    smooth.iterations = 5
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    body.data.materials.append(skin)
    for p in body.data.polygons:
        p.use_smooth = True

    rig = bpy.data.objects.new(name + '_Rig', bpy.data.armatures.new(name))
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    body.select_set(False)
    bpy.ops.object.mode_set(mode='EDIT')
    specs = [
        ('Root', (0,0,.72), (0,0,1.15), None),
        ('Spine', (0,0,1.15), (0,0,1.75), 'Root'),
        ('Chest', (0,0,1.75), (0,0,2.30), 'Spine'),
        ('Neck', (0,0,2.30), (0,0,2.49 if broad else 2.68), 'Chest'),
    ]
    if broad:
        specs.append(('Neck2', (0,0,2.49), (0,0,2.68), 'Neck'))
    specs.append(('Head', (0,0,2.68), (0,0,3.35), 'Neck2' if broad else 'Neck'))
    for s, suffix in ((1, 'L'), (-1, 'R')):
        specs += [
            ('Clavicle.'+suffix, (0,0,2.20), (s*.70*width,0,2.20), 'Chest'),
            ('UpperArm.'+suffix, (s*.70*width,0,2.20), (s*1.32*width,0,2.20), 'Clavicle.'+suffix),
            ('Forearm.'+suffix, (s*1.32*width,0,2.20), (s*1.85*width,0,2.20), 'UpperArm.'+suffix),
            ('Hand.'+suffix, (s*1.85*width,0,2.20), (s*2.13*width,0,2.20), 'Forearm.'+suffix),
        ]
    for bone_name, head, tail, parent in specs:
        b = rig.data.edit_bones.new(bone_name)
        b.head, b.tail = head, tail
        if parent:
            b.parent = rig.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    body.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    assert all(v.groups for v in body.data.vertices), 'Unweighted fixture vertices'

    for s in (-1, 1):
        eye = ellipsoid('Eye', (s*.125*width, -.265, 3.04), (.064, .035, .047), dark)
        vg = eye.vertex_groups.new(name='Head')
        vg.add(list(range(len(eye.data.vertices))), 1, 'REPLACE')
        mod = eye.modifiers.new('Head attachment', 'ARMATURE')
        mod.object = rig
        eye.parent = rig
    mouth = ellipsoid('Mouth landmark', (0, -.294, 2.825), (.095, .018, .018), dark)
    mouth.vertex_groups.new(name='Head').add(list(range(len(mouth.data.vertices))), 1, 'REPLACE')
    mouth.modifiers.new('Head attachment', 'ARMATURE').object = rig
    mouth.parent = rig

    def world_rotation(bone, axis, angle):
        basis = rig.data.bones[bone].matrix_local.to_quaternion()
        return basis.inverted() @ Quaternion(axis, math.radians(angle)) @ basis

    for clip in ('head', 'neck', 'arms'):
        rig.animation_data_create()
        rig.animation_data.action = None
        for frame in range(FRAMES):
            t = frame / (FRAMES - 1)
            pulse = math.sin(2*math.pi*t) * math.sin(math.pi*t)
            for pb in rig.pose.bones:
                pb.rotation_mode = 'QUATERNION'
                pb.rotation_quaternion = (1,0,0,0)
                pb.location = (0,0,0)
                pb.scale = (1,1,1)
            for side, suffix in ((1,'L'), (-1,'R')):
                raise_angle = 65 - 100 * math.sin(math.pi*t)**2 if clip == 'arms' else 65
                rig.pose.bones['UpperArm.'+suffix].rotation_quaternion = world_rotation('UpperArm.'+suffix, (0,1,0), side*raise_angle)
                if clip == 'arms':
                    rig.pose.bones['Forearm.'+suffix].rotation_quaternion = world_rotation('Forearm.'+suffix, (0,0,1), -side*80*math.sin(math.pi*t)**2)
            if clip == 'head':
                rig.pose.bones['Head'].rotation_quaternion = world_rotation('Head', (0,0,1), 45*pulse) @ world_rotation('Head', (1,0,0), 22*math.sin(4*math.pi*t)*math.sin(math.pi*t))
            if clip == 'neck':
                for bone in (('Neck','Neck2') if broad else ('Neck',)):
                    div = 2 if broad else 1
                    rig.pose.bones[bone].rotation_quaternion = world_rotation(bone, (0,1,0), 24*pulse/div) @ world_rotation(bone, (1,0,0), 18*math.sin(4*math.pi*t)*math.sin(math.pi*t)/div)
            for pb in rig.pose.bones:
                pb.keyframe_insert('rotation_quaternion', frame=frame)
        rig.animation_data.action.name = clip
        rig.animation_data.action.use_fake_user = True
    bpy.context.scene.frame_set(0)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.gltf(filepath=str(OUT / f'{name}.glb'), export_format='GLB', export_animations=True, export_animation_mode='ACTIONS', export_force_sampling=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / f'{name}.blend'))
    print('FIXTURE_READY', name, len(rig.data.bones), len(body.data.vertices))


for name, broad in (('slender', False), ('broad', True)):
    make_bust(name, broad)
