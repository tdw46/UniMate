"""Render a 3x3 grid of nine independently reconstructed avatars."""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/avatar_grid'
PREVIEW = '--preview' in sys.argv
PREPARED = '--prepared' in sys.argv
VIEW = 'oblique' if '--oblique' in sys.argv else 'front'
FRAME = int(sys.argv[sys.argv.index('--frame')+1]) if '--frame' in sys.argv else 299


def material(name,color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color,1)
    return m


def text(body,x,z,size,color,align='CENTER'):
    data = bpy.data.curves.new(body,'FONT')
    data.body, data.align_x, data.size = body,align,size
    obj = bpy.data.objects.new(body,data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (x,-3,z)
    obj.rotation_euler.x = math.pi/2
    obj.data.materials.append(material(body,color))
    return obj


def stage():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x = scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.frame_start,scene.frame_end = 0,359
    scene.render.image_settings.file_format = 'PNG'
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.studio_light = 'paint.sl'
    shading.color_type = 'TEXTURE'
    shading.show_shadows = False
    shading.show_cavity = True
    shading.cavity_type = 'BOTH'
    shading.curvature_ridge_factor = .6
    shading.curvature_valley_factor = .4
    shading.show_specular_highlight = False
    shading.background_type = 'WORLD'
    scene.world = bpy.data.worlds.new('Gallery background')
    scene.world.color = (.013,.021,.035)
    scene.view_settings.view_transform = 'Standard'
    scene.display.render_aa = '16'
    entries = json.loads((OUT/'sources/manifest.json').read_text())
    layout = []
    for index,entry in enumerate(entries):
        name = entry['id']
        before = set(scene.objects)
        source = OUT/'avatars'/name/('02_fresh_rig.glb' if PREPARED else f'reconstructed/{name}-evaluation.glb')
        bpy.ops.import_scene.gltf(filepath=str(source))
        imported = set(scene.objects)-before
        rigs = [o for o in imported if o.type=='ARMATURE']
        assert len(rigs)==1
        meshes = [o for o in imported if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers)]
        for obj in meshes:
            for m in obj.data.materials:
                if m and m.use_nodes:
                    images = [n for n in m.node_tree.nodes if n.type=='TEX_IMAGE' and n.image]
                    if images:
                        m.node_tree.nodes.active = images[0]
        group = bpy.data.objects.new(name+'_Display',None)
        scene.collection.objects.link(group)
        for obj in imported:
            if obj.type=='MESH' and obj not in meshes:
                obj.hide_render = True
            if obj.parent is None or obj.parent not in imported:
                world = obj.matrix_world.copy()
                obj.parent = group
                obj.matrix_world = world
        if VIEW=='oblique':
            group.rotation_mode = 'QUATERNION'
            group.rotation_quaternion = Quaternion((0,0,1),math.radians(-30))
        bounds = []
        for frame in range(360):
            scene.frame_set(frame)
            deps = bpy.context.evaluated_depsgraph_get()
            for obj in meshes:
                ev = obj.evaluated_get(deps)
                bounds.extend(tuple(ev.matrix_world @ Vector(c)) for c in ev.bound_box)
        low = Vector(tuple(min(v[i] for v in bounds) for i in range(3)))
        high = Vector(tuple(max(v[i] for v in bounds) for i in range(3)))
        width,height = high.x-low.x, high.z-low.z
        scale = min(3.53/width,2.60/height)
        x = (index%3-1)*4.05
        base = (2-index//3)*3.6+.2
        group.scale = (scale,)*3
        group.location = (x-(low.x+high.x)*scale/2,0,base+.15-low.z*scale)
        layout.append({'id':name,'bounds_min':list(low),'bounds_max':list(high),'display_scale':scale})
        bpy.ops.mesh.primitive_cube_add(size=1,location=(x,1.0,base+1.16))
        card = bpy.context.object
        card.name = name+'_Card'
        card.dimensions = (3.93,.06,3.34)
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        card.data.materials.append(material('Card',(0.026,.038,.058)))
        bevel = card.modifiers.new('Rounded panel','BEVEL')
        bevel.width,bevel.segments = .06,4
        text(f'{index+1:02d}  {entry["name"].upper()}',x,base-.13,.19,(.85,.9,.96))
        text('BOOLEAN BUST  /  FRESH 13-BONE RIG',x,base-.36,.085,(.45,.62,.72))
    camera = bpy.data.objects.new('Grid camera',bpy.data.cameras.new('Grid camera'))
    scene.collection.objects.link(camera)
    camera.location = (0,-25,5.12)
    camera.rotation_euler = (Vector((0,0,5.12))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 12.9
    scene.camera = camera
    text('AVATAR RIG STUDY  /  09',-5.9,11.0,.32,(.91,.95,.99),'LEFT')
    text('BLENDER 5.2  /  '+VIEW.upper(),5.9,11.09,.13,(.48,.70,.81),'RIGHT')
    for phase,label in enumerate(('01  HEAD TURNS + NODS','02  NECK BENDS + TILTS','03  ARM RAISES + ELBOW FLEXION')):
        obj = text(label,-5.9,10.54,.18,(.38,.77,.78),'LEFT')
        for f,hide in [(0,phase!=0),(120,phase!=1),(240,phase!=2),(359,phase!=2)]:
            obj.hide_render = hide
            obj.keyframe_insert('hide_render',frame=f)
    text('9 CC0 avatars by Quaternius  |  Posed, baked, cut, stripped, freshly weighted',0,-.76,.135,(.62,.72,.82))
    text('Authored evaluation motion through UniMate GLB > NPZ > GLB. No model inference.',0,-1.02,.12,(.44,.56,.68))
    scene['pipeline_provenance'] = 'New rigs and new weights after destructive Boolean cuts; UniMate reconstruction of authored diagnostics.'
    scene['layout_audit'] = json.dumps(layout)
    scene.frame_set(FRAME if PREVIEW else 0)
    bpy.ops.file.pack_all()
    for area in bpy.context.screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            area.spaces.active.shading.type = 'SOLID'
            area.spaces.active.shading.color_type = 'TEXTURE'
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/f'avatar_grid_{VIEW}.blend'))
    (OUT/f'layout_{VIEW}.json').write_text(json.dumps(layout,indent=2))
    if PREVIEW:
        scene.render.filepath = str(OUT/f'preview_{VIEW}_{FRAME:04d}.png')
        bpy.ops.render.render(write_still=True)
    else:
        folder = OUT/'frames'/VIEW
        folder.mkdir(parents=True,exist_ok=True)
        scene.render.filepath = str(folder)+'/'
        bpy.ops.render.render(animation=True)
    print('GRID_RENDER_COMPLETE',VIEW,flush=True)


stage()
