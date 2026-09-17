"""Full-character overview and camera tour of nine freshly reconstructed rigs."""
import argparse,json,math,sys
from pathlib import Path
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from avatar_fullbody import PHASES,FRAMES
parser=argparse.ArgumentParser();parser.add_argument('--prepared',action='store_true');parser.add_argument('--first-only',action='store_true');parser.add_argument('--preview',type=int);parser.add_argument('--tour',action='store_true')
a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);out=ROOT/'outputs/fullbody_avatar_grid';view='tour' if a.tour else 'overview'
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
engines={x.identifier for x in scene.render.bl_rna.properties['engine'].enum_items};scene.render.engine='BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
if hasattr(scene,'eevee') and hasattr(scene.eevee,'taa_render_samples'):scene.eevee.taa_render_samples=16
scene.render.resolution_x=scene.render.resolution_y=1920;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.fps=30;scene.frame_start=0;scene.frame_end=FRAMES-1
scene.world=bpy.data.worlds.new('Gallery world');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.018,.025,.045,1);scene.view_settings.view_transform='Standard'

def mat(name,color):
 m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;n.clear();e=n.new('ShaderNodeEmission');e.inputs[0].default_value=(*color,1);o=n.new('ShaderNodeOutputMaterial');m.node_tree.links.new(e.outputs[0],o.inputs['Surface']);return m

def text(body,x,z,size,color):
 d=bpy.data.curves.new(body,'FONT');d.body=body;d.align_x='CENTER';d.size=size;o=bpy.data.objects.new(body,d);scene.collection.objects.link(o);o.location=(x,-2,z);o.rotation_euler.x=math.pi/2;d.materials.append(mat(body,color));return o

entries=json.loads((out/'sources/manifest.json').read_text());entries=entries[:1] if a.first_only else entries
rigs=[];audit=[]
for index,entry in enumerate(entries):
 folder=out/'avatars'/entry['id'];path=folder/('02_fresh_rig.glb' if a.prepared else 'reconstructed/'+entry['id']+'-evaluation.glb')
 before=set(scene.objects);bpy.ops.import_scene.gltf(filepath=str(path));imported=set(scene.objects)-before;rig=next(o for o in imported if o.type=='ARMATURE');meshes=[o for o in imported if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers)];group=bpy.data.objects.new(entry['id']+'_Display',None);scene.collection.objects.link(group)
 for o in imported:
  if o.parent not in imported:world=o.matrix_world.copy();o.parent=group;o.matrix_world=world
 low=Vector((float('inf'),)*3);high=-low
 for frame in range(FRAMES):
  scene.frame_set(frame);deps=bpy.context.evaluated_depsgraph_get()
  for o in meshes:
   ev=o.evaluated_get(deps)
   for corner in ev.bound_box:
    p=ev.matrix_world@Vector(corner)
    for k in range(3):low[k]=min(low[k],p[k]);high[k]=max(high[k],p[k])
 scale=min(3.48/(high.x-low.x),2.85/(high.z-low.z));x=(index%3-1)*4.05;z=(2-index//3)*3.72+.23
 group.scale=(scale,)*3;group.location=(x-scale*(low.x+high.x)*.5,0,z-scale*low.z)
 rigs.append(rig);audit.append({'id':entry['id'],'scale':scale,'motion_bounds_min':list(low),'motion_bounds_max':list(high),'full_body_fit_all_frames':True})
 text(f'{index+1:02d}  {entry.get("display_name",entry["name"]).upper()}',x,z-.23,.17,(.87,.93,.98))
 text('FRESH FULL RIG  /  52 BONES',x,z-.43,.095,(.43,.63,.75))
 # Stage discs are visual floor references for foot placement.
 bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=.85,depth=.025,location=(x,0,z-.04));bpy.context.object.data.materials.append(mat('Stage',(.025,.038,.06)))
text('FULL CHARACTER STUDY  /  09',0,11.37,.30,(.9,.95,.99))
text('HEAD  /  NECK  /  TORSO  /  ARMS  /  LEGS  /  HANDS',0,11.03,.14,(.43,.70,.81))
for phase,label in enumerate(PHASES):
 obj=text(f'{phase+1:02d}  {label}',0,10.68,.17,(.45,.79,.79))
 for f in (0,120,240,360,480,600,719):obj.hide_render=(f//120!=phase);obj.keyframe_insert('hide_render',frame=f)
text('VRoid / pixiv sources  |  Fresh weights + aligned rest mesh  |  Authored movement through UniMate',0,-.65,.12,(.50,.63,.73))
c=bpy.data.objects.new('Gallery camera',bpy.data.cameras.new('Gallery camera'));scene.collection.objects.link(c);c.data.type='ORTHO';scene.camera=c
camera_audit=[]
for frame in range(FRAMES):
 scene.frame_set(frame);phase=frame//120;t=(frame%120)/119;target=Vector((0,0,5.3));size=12.9;position=target+Vector((0,-25,0))
 if a.tour and phase in (1,2,3,4):
  selected={1:0,2:4,3:6,4:1}[phase]%len(rigs);r=rigs[selected]
  names={1:['Head','Neck','Chest'],2:['Thigh.L','Shin.L','Foot.L','Toe.L','Foot.R'],3:['Hips','Chest'],4:['Hand.R','Middle2.R','Thumb2.R']}[phase]
  focus=sum((r.matrix_world@r.pose.bones[n].head for n in names),Vector())/len(names)
  amount=min(1.,t/.25,(1-t)/.25);amount=amount*amount*(3-2*amount)
  close={1:1.35,2:1.6,3:3.0,4:.72}[phase]
  target=target.lerp(focus,amount);size=size*(1-amount)+close*amount
  angle=Vector((14,-22,5)) if phase==3 else Vector((4,-24,8 if phase==4 else 3))
  extra=angle*amount+Vector((0,-25,0))*(1-amount)
  position=target+extra
 c.location=position;c.rotation_euler=(target-position).to_track_quat('-Z','Y').to_euler();c.data.ortho_scale=size
 c.keyframe_insert('location',frame=frame);c.keyframe_insert('rotation_euler',frame=frame);c.data.keyframe_insert('ortho_scale',frame=frame)
 if frame%120==60:camera_audit.append({'frame':frame,'target':list(target),'ortho_scale':size})
for loc,power in [((-7,-8,14),1800),((6,-6,10),1200),((0,5,12),1600)]:
 o=bpy.data.objects.new('Softbox',bpy.data.lights.new('Softbox','AREA'));scene.collection.objects.link(o);o.location=loc;o.rotation_euler=(Vector((0,0,5))-o.location).to_track_quat('-Z','Y').to_euler();o.data.energy=power;o.data.size=8
scene.frame_set(a.preview if a.preview is not None else 0);bpy.ops.file.pack_all();scene['fullbody_layout']=json.dumps(audit)
bpy.ops.wm.save_as_mainfile(filepath=str(out/f'fullbody_{view}.blend'))
if a.preview is not None:scene.render.filepath=str(out/f'preview_{view}_{a.preview:04d}.png');bpy.ops.render.render(write_still=True)
else:
 folder=out/'frames'/view;folder.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(folder)+'/';bpy.ops.render.render(animation=True)
(out/f'layout_{view}.json').write_text(json.dumps({'avatars':audit,'camera':camera_audit,'frames':FRAMES},indent=2))
