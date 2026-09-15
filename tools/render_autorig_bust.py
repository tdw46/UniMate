"""Render saved bust motion without modifying the working rig artifact."""
import argparse, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

parser=argparse.ArgumentParser()
parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
parser.add_argument('--preview',action='store_true')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(Path(args.input).resolve()))
s=bpy.context.scene;s.render.engine='BLENDER_EEVEE';s.render.resolution_x=1100;s.render.resolution_y=1000;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG'
if hasattr(s,'eevee') and hasattr(s.eevee,'taa_render_samples'):s.eevee.taa_render_samples=16
w=bpy.data.worlds.new('Evaluation world');w.use_nodes=True;w.node_tree.nodes.get('Background').inputs['Color'].default_value=(.06,.075,.10,1);s.world=w
c=bpy.data.objects.new('Evaluation camera',bpy.data.cameras.new('Evaluation camera'));s.collection.objects.link(c);c.data.type='ORTHO';c.data.ortho_scale=1.55;s.camera=c
target=Vector((0,0,0));c.location=(.7,-4,.12);c.rotation_euler=(target-c.location).to_track_quat('-Z','Y').to_euler()
for x,y,z,power in [(-2,-2,3,160),(2,-2,1,100),(0,2,2,180)]:
 l=bpy.data.objects.new('Softbox',bpy.data.lights.new('Softbox','AREA'));s.collection.objects.link(l);l.location=(x,y,z);l.rotation_euler=(target-l.location).to_track_quat('-Z','Y').to_euler();l.data.energy=power;l.data.size=3
s.frame_set(0)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'evaluation_stage.blend'))
if args.preview:
 for name,frame in [('rest',0),('head',26),('neck',106),('arms',200)]:
  s.frame_set(frame);s.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
else:
 (out/'frames').mkdir(exist_ok=True)
 for frame in range(s.frame_start,s.frame_end+1):
  s.frame_set(frame);s.render.filepath=str(out/'frames'/f'{frame:04d}.png');bpy.ops.render.render(write_still=True)
