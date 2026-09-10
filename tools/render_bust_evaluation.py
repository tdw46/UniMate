"""Render the actual reconstructed GLBs in a separate Blender 5.2 process."""
import math
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/bust_evaluation'
PREVIEW = '--preview' in sys.argv
OBLIQUE_ONLY = '--oblique-only' in sys.argv


def text(camera, body, x, y, size, color, align='CENTER'):
    curve = bpy.data.curves.new(body, 'FONT')
    curve.body = body
    curve.size = size
    curve.align_x = align
    obj = bpy.data.objects.new(body, curve)
    bpy.context.collection.objects.link(obj)
    obj.parent = camera
    obj.location = (x, y, -9)
    mat = bpy.data.materials.new(body)
    mat.diffuse_color = (*color, 1)
    obj.data.materials.append(mat)
    obj.color = (*color, 1)
    return obj


def stage(clip, view):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start = 0
    scene.frame_end = 119
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 900
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.studio_light = 'paint.sl'
    shading.color_type = 'MATERIAL'
    shading.show_shadows = True
    shading.show_cavity = True
    shading.cavity_type = 'BOTH'
    shading.curvature_ridge_factor = 1.1
    shading.curvature_valley_factor = .65
    shading.show_specular_highlight = True
    shading.background_type = 'WORLD'
    scene.world = bpy.data.worlds.new('Evaluation background')
    scene.world.color = (.025, .035, .055)
    scene.view_settings.view_transform = 'Standard'
    scene.display.render_aa = '16'
    for name, x in (('slender', -2.7), ('broad', 2.7)):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(OUT / 'reconstructed' / name / f'{name}-{clip}.glb'))
        rig = next(o for o in set(bpy.data.objects)-before if o.type == 'ARMATURE')
        rig.location.x = x
        if view == 'oblique':
            rig.rotation_mode = 'QUATERNION'
            rig.rotation_quaternion = Quaternion((0,0,1), math.radians(-35)) @ rig.rotation_quaternion
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=.80, depth=.20, location=(x, 0, .68))
        pedestal = bpy.context.object
        m = bpy.data.materials.new('Plinth')
        m.diffuse_color = (.07,.10,.14,1)
        pedestal.data.materials.append(m)
        bevel = pedestal.modifiers.new('Edge softness', 'BEVEL')
        bevel.width = .04
        bevel.segments = 3
    cam = bpy.data.objects.new('Evaluation camera', bpy.data.cameras.new('Evaluation camera'))
    scene.collection.objects.link(cam)
    cam.location = (0, -15, 4.5)
    cam.rotation_euler = (Vector((0,0,1.95))-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 10.8
    scene.camera = cam
    title = {'head':'01  /  HEAD TURNS + NODS', 'neck':'02  /  NECK BENDS + TILTS', 'arms':'03  /  ARM RAISES + ELBOW FLEXION'}[clip]
    text(cam, 'UNIMATE  /  BLENDER 5.2', -4.9, 2.65, .16, (.46,.66,.77), 'LEFT')
    text(cam, title, -4.9, 2.24, .26, (.87,.92,.96), 'LEFT')
    text(cam, view.upper() + ' VIEW', 4.9, 2.65, .15, (.46,.66,.77), 'RIGHT')
    text(cam, 'SLENDER  /  13 BONES', -2.7, -1.87, .18, (.3,.72,.76))
    text(cam, 'BROAD  /  14 BONES', 2.7, -1.87, .18, (.95,.58,.32))
    text(cam, 'Single neck joint', -2.7, -2.17, .14, (.65,.72,.8))
    text(cam, 'Two-joint neck chain', 2.7, -2.17, .14, (.65,.72,.8))
    text(cam, 'Authored diagnostics  |  UniMate GLB > NPZ > GLB  |  No model inference', 0, -2.73, .145, (.62,.70,.8))
    scene['evaluation_source'] = 'Authored diagnostic motion reconstructed through UniMate; not AI-generated.'
    scene.frame_set(60)
    folder = OUT / 'frames' / f'{clip}_{view}'
    folder.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(folder / '') + '/'
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / f'evaluation_{clip}_{view}.blend'))
    if PREVIEW:
        scene.render.filepath = str(OUT / f'preview_{clip}_{view}.png')
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


if PREVIEW:
    stage('arms', 'oblique' if OBLIQUE_ONLY else 'front')
else:
    for clip in ('head','neck','arms'):
        for view in (('oblique',) if OBLIQUE_ONLY else ('front', 'oblique')):
            stage(clip, view)
print('RENDERS_COMPLETE', flush=True)
