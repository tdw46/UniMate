"""Render the requested card's front and back arm-motion comparison."""
import json, math, os, sys
from pathlib import Path
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('AVATAR_EVAL_ROOT',ROOT/'outputs/complex_avatar_grid'))
# Presentation selection only; binding has no character-specific rules.
entry=json.loads((OUT/'sources/manifest.json').read_text())[6]
preview='--preview' in sys.argv
for state,folder in [('before','before_shoulder_strain_fix'),('after','avatars')]:
 for view,sign in [('front',-1),('back',1)]:
  bpy.ops.wm.open_mainfile(filepath=str(OUT/folder/entry['id']/'02_fresh_rig.blend'))
  scene=bpy.context.scene
  engines={e.identifier for e in scene.render.bl_rna.properties['engine'].enum_items}
  scene.render.engine='BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
  if hasattr(scene,'eevee') and hasattr(scene.eevee,'taa_render_samples'):scene.eevee.taa_render_samples=16
  scene.render.resolution_x=scene.render.resolution_y=768
  scene.render.resolution_percentage=100
  scene.render.image_settings.file_format='PNG'
  scene.view_settings.view_transform='Standard'
  world=bpy.data.worlds.new('Shoulder comparison background');scene.world=world;world.use_nodes=True
  world.node_tree.nodes.get('Background').inputs['Color'].default_value=(.045,.055,.075,1)
  camera=bpy.data.objects.new('Shoulder camera',bpy.data.cameras.new('Shoulder camera'))
  scene.collection.objects.link(camera);camera.location=(0,sign*8,1.48)
  camera.rotation_euler=(Vector((0,0,1.48))-camera.location).to_track_quat('-Z','Y').to_euler()
  camera.data.type='ORTHO';camera.data.ortho_scale=2.75;scene.camera=camera
  for x in (-3,3):
   light=bpy.data.objects.new('Softbox',bpy.data.lights.new('Softbox','AREA'))
   scene.collection.objects.link(light);light.location=(x,sign*4,4)
   light.rotation_euler=(Vector((0,0,1.3))-light.location).to_track_quat('-Z','Y').to_euler()
   light.data.energy=300;light.data.shape='DISK';light.data.size=5
  target=OUT/'shoulder_comparison_frames'/f'{state}_{view}';target.mkdir(parents=True,exist_ok=True)
  for frame in ([275] if preview else range(240,360)):
   scene.frame_set(frame);scene.render.filepath=str(target/f'{frame-240:04d}.png')
   bpy.ops.render.render(write_still=True)
print('SHOULDER_COMPARISON_RENDERED',flush=True)
