"""Render paired hand closeups for nine fresh VRoid rigs; retain full rig assets."""
import argparse,json,math,sys
from pathlib import Path
import bpy,bmesh
from mathutils import Matrix,Vector,Quaternion
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from avatar_source_import import material_role
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'outputs/finger_avatar_grid');parser.add_argument('--preview',type=int);parser.add_argument('--first-only',action='store_true');parser.add_argument('--prepared',action='store_true');parser.add_argument('--palm',action='store_true')
a=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);out=a.root.resolve();view='palm' if a.palm else 'dorsal'
bpy.ops.wm.read_factory_settings(use_empty=True);scene=bpy.context.scene
engines={x.identifier for x in scene.render.bl_rna.properties['engine'].enum_items}
scene.render.engine='BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
if hasattr(scene,'eevee') and hasattr(scene.eevee,'taa_render_samples'):scene.eevee.taa_render_samples=16
scene.render.resolution_x=scene.render.resolution_y=1920;scene.render.resolution_percentage=100
scene.render.fps=30;scene.frame_start=0;scene.frame_end=359;scene.render.image_settings.file_format='PNG'
scene.world=bpy.data.worlds.new('Hand study world');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.08,.10,.14,1)
scene.view_settings.view_transform='Standard'

def emission(name,color):
 m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
 bs=m.node_tree.nodes.get('Principled BSDF');m.node_tree.nodes.remove(bs);e=m.node_tree.nodes.new('ShaderNodeEmission');e.inputs[0].default_value=(*color,1);m.node_tree.links.new(e.outputs[0],m.node_tree.nodes.get('Material Output').inputs['Surface']);return m

def label(body,x,z,size,color):
 data=bpy.data.curves.new(body,'FONT');data.body=body;data.align_x='CENTER';data.size=size
 obj=bpy.data.objects.new(body,data);scene.collection.objects.link(obj);obj.location=(x,-2,z);obj.rotation_euler.x=math.pi/2;data.materials.append(emission(body,color));return obj

entries=json.loads((out/'sources/manifest.json').read_text())
if a.first_only:entries=entries[:1]
audit=[]
for index,entry in enumerate(entries):
 folder=out/'avatars'/entry['id'];before=set(scene.objects)
 path=folder/('02_fresh_rig.glb' if a.prepared else 'reconstructed/'+entry['id']+'-evaluation.glb')
 bpy.ops.import_scene.gltf(filepath=str(path));imported=set(scene.objects)-before
 rig=next(o for o in imported if o.type=='ARMATURE');scene.frame_set(0)
 meshes=[o for o in imported if o.type=='MESH']
 cell_x=(index%3-1)*4.05;cell_z=(2-index//3)*3.8+.4
 for side,sign,offset in [('L',1,-1.00),('R',-1,1.00)]:
  r=rig.copy();r.data=rig.data.copy();scene.collection.objects.link(r)
  wrist=rig.data.bones['Hand.'+side].head_local.copy()
  forward=(rig.data.bones['Hand.'+side].tail_local-wrist).normalized()
  across=rig.data.bones['Index1.'+side].head_local-rig.data.bones['Little1.'+side].head_local
  across=(across-forward*across.dot(forward)).normalized();right=across*sign
  depth=forward.cross(right).normalized();rotation=Matrix((right,depth,forward)).to_4x4()
  if a.palm:rotation=Quaternion((0,0,1),math.pi).to_matrix().to_4x4()@rotation
  # A slight oblique view exposes flexion depth while retaining readable fingers.
  rotation=Quaternion((1,0,0),math.radians(12)).to_matrix().to_4x4()@rotation
  palm=(rig.data.bones['Middle1.'+side].head_local-wrist).length
  length=max((rig.data.bones[d+'3.'+side].tail_local-wrist).length for d in ('Thumb','Index','Middle','Ring','Little'))
  display=rotation@Matrix.Translation(-wrist)
  r.matrix_world=display@rig.matrix_world
  kept=0;cropped=[]
  for obj in meshes:
   data=obj.data.copy();bm=bmesh.new();bm.from_mesh(data)
   # Reveal the hand for evaluation; keep sleeves on the delivered full avatar.
   skin={i for i,m in enumerate(data.materials) if material_role(m)=='body'}
   bmesh.ops.delete(bm,geom=[f for f in bm.faces if f.material_index not in skin],context='FACES')
   # Isolate hands visually on display copies only. Include a short wrist cuff.
   remove=[v for v in bm.verts if (v.co-wrist).dot(forward)<-palm*.28 or (v.co-wrist).length>length*1.35]
   bmesh.ops.delete(bm,geom=remove,context='VERTS');bm.to_mesh(data);bm.free()
   if not len(data.polygons):bpy.data.meshes.remove(data);continue
   crop=obj.copy();crop.data=data;scene.collection.objects.link(crop);crop.parent=r
   crop.matrix_world=display@obj.matrix_world
   for mod in crop.modifiers:
    if mod.type=='ARMATURE':mod.object=r
   kept+=len(data.vertices);cropped.append(crop)
  assert kept>50,(entry['id'],side,'No visible hand geometry')
  low=Vector((float('inf'),)*3);high=-low
  # Fit every evaluated pose to its own hand slot: thumbs and wrist deviations
  # must not touch the other hand or neighboring avatar cards.
  for frame in range(360):
   scene.frame_set(frame);deps=bpy.context.evaluated_depsgraph_get()
   for crop in cropped:
    ev=crop.evaluated_get(deps)
    for corner in ev.bound_box:
     point=ev.matrix_world@Vector(corner)
     for k in range(3):low[k]=min(low[k],point[k]);high[k]=max(high[k],point[k])
  scale=min(1.75/(high.x-low.x),2.75/(high.z-low.z))
  shift=Vector((cell_x+offset-scale*(low.x+high.x)*.5,0,cell_z+.22-scale*low.z))
  r.matrix_world=Matrix.Translation(shift)@Matrix.Scale(scale,4)@display@rig.matrix_world
  scene.frame_set(0)
  audit.append({'id':entry['id'],'side':side,'display_vertices':kept,'source':str(path),
                'motion_bounds_min':list(low),'motion_bounds_max':list(high),'scale':scale,
                'slot_bounds_min':list(low*scale+shift),'slot_bounds_max':list(high*scale+shift),
                'all_360_frames_fitted':True})
  label(side,cell_x+offset,cell_z+.10,.13,(.45,.66,.79))
 for obj in imported:bpy.data.objects.remove(obj,do_unlink=True)
 label(f'{index+1:02d}  {entry.get("display_name",entry["name"]).upper()}',cell_x,cell_z-.16,.20,(.87,.93,.98))
 label('SPREAD BIND  /  ALIGNED REST',cell_x,cell_z-.40,.11,(.44,.65,.74))
label('VRoid HAND STUDY  /  09',0,11.65,.32,(.90,.95,.99))
label('30 FINGER BONES PER AVATAR  /  '+view.upper(),0,11.29,.16,(.40,.74,.81))
for phase,text in enumerate(('01  WRIST FLEX + DEVIATION','02  INDIVIDUAL FINGER CURLS','03  FIST + THUMB OPPOSITION + SPREAD')):
 obj=label(text,0,10.94,.17,(.50,.80,.80))
 for f in (0,120,240,359):obj.hide_render=(f//120!=phase);obj.keyframe_insert('hide_render',frame=f)
label('VRoid / pixiv samples  |  Fresh bones + heat weights  |  Authored diagnostic motion',0,-.65,.13,(.55,.65,.73))
camera=bpy.data.objects.new('Hand grid camera',bpy.data.cameras.new('Hand grid camera'));scene.collection.objects.link(camera)
camera.location=(0,-25,5.5);camera.rotation_euler=(Vector((0,0,5.5))-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.type='ORTHO';camera.data.ortho_scale=13;scene.camera=camera
for loc,power in [((-6,-9,14),1800),((6,-6,8),1200),((0,4,13),1600)]:
 light=bpy.data.objects.new('Softbox',bpy.data.lights.new('Softbox','AREA'));scene.collection.objects.link(light);light.location=loc;light.rotation_euler=(Vector((0,0,5))-light.location).to_track_quat('-Z','Y').to_euler();light.data.energy=power;light.data.size=8
scene.frame_set(a.preview if a.preview is not None else 0);bpy.ops.file.pack_all()
scene['hand_grid_audit']=json.dumps(audit)
scene['display_scope']='Hand skin closeups; sleeves hidden on display copies only. Full avatars remain in avatars/. '
bpy.ops.wm.save_as_mainfile(filepath=str(out/f'hand_grid_{view}.blend'))
if a.preview is not None:
 scene.render.filepath=str(out/f'preview_{view}_{a.preview:04d}.png');bpy.ops.render.render(write_still=True)
else:
 folder=out/'frames'/view;folder.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(folder)+'/';bpy.ops.render.render(animation=True)
(out/f'grid_{view}_validation.json').write_text(json.dumps({'hands':audit,'rigs':sum(o.type=='ARMATURE' for o in scene.objects),'frames':360},indent=2))
