"""Side ownership and smooth body/spring handoff regression in isolated Blender."""
import bpy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if not bpy.app.background:raise RuntimeError('Use isolated Blender')
source,dest=sys.argv[sys.argv.index('--')+1:];out=Path(dest).resolve();out.mkdir(exist_ok=True,parents=True)
for repo in bpy.context.preferences.extensions.repos:
 if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
from properties_hallway_rig import register
register();bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
from avatar_apparel_weights import weights
from avatar_skirt_binding import rebind_skirt_strips
from avatar_pelvis_binding import transfer_garment_waist
from avatar_waist_attachment import WaistAttachment
from avatar_dress import full_chain_names
from avatar_mesh_invariant import snapshot,verify
from avatar_skirt_fit_io import rest_edit
r=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('hallway_bilateral_skirt_layout'))
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==r]
geometry=snapshot(meshes);pose={p.name:p.matrix_basis.copy() for p in r.pose.bones}
def binding():return {o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}
before=binding();attach=WaistAttachment(r,meshes)
sides={}
for s in r.data.vrm_addon_extension.spring_bone1.springs:
 if not s.vrm_name.startswith('Secondary_Skirt_'):continue
 ns=full_chain_names(r,s)
 if ns:
  side=1 if attach.lateral_coordinate(r.data.bones[ns[0]].head_local)>0 else -1
  sides.update({n:side for n in ns})
def audit(data):
 cross_leg=[];cross_chain=[];max_jump=0.;energy=[];full_energy=[];direct_leg=0;max_influences=0;max_chains=0
 for o in meshes:
  m=r.matrix_world.inverted()@o.matrix_world;ws=data[o.name]
  mass=[sum(v for n,v in w.items() if n in sides) for w in ws]
  for i,w in enumerate(ws):
   if not mass[i]>1e-6:continue
   if sum(w.get(n,0.) for n in attach.legs)>1e-6:direct_leg+=1
   p=m@o.data.vertices[i].co;x=attach.lateral_coordinate(p)
   if abs(x)>1e-6:
    opposite=attach.legs[1 if x>0 else 0]
    if w.get(opposite,0)>1e-6:cross_leg.append((o.name,i,w[opposite]))
   if abs(x)>attach.midline_width+1e-6:
    total=sum(v for n,v in w.items() if n in sides and x*sides[n]<0)
    if total>1e-6:cross_chain.append((o.name,i,total))
   max_influences=max(max_influences,len(w));max_chains=max(max_chains,len({n.rsplit('_',1)[0] for n in w if n in sides}))
  for e in o.data.edges:
   a,b=e.vertices
   if 0<mass[a]+mass[b]<2-1e-6:
    delta=abs(mass[a]-mass[b]);max_jump=max(max_jump,delta);energy.append(delta**2)
    full_energy.append(sum((ws[a].get(n,0)-ws[b].get(n,0))**2 for n in ws[a].keys()|ws[b].keys()))
 return dict(cross_leg_count=len(cross_leg),cross_chain_count=len(cross_chain),direct_leg_vertices=direct_leg,max_handoff_jump=max_jump,handoff_edge_energy=sum(energy)/max(1,len(energy)),full_handoff_edge_energy=sum(full_energy)/max(1,len(full_energy)),max_influences=max_influences,max_chains=max_chains)
base=audit(before)
with rest_edit(r):
 report=rebind_skirt_strips(r,meshes);strip_binding=binding()
 transfer=transfer_garment_waist(r,meshes)
after=binding();result=audit(after)
upper_gains=0
for obj in meshes:
 m=r.matrix_world.inverted()@obj.matrix_world
 for i,(old,new) in enumerate(zip(before[obj.name],after[obj.name])):
  if any(n in sides for n in strip_binding[obj.name][i]):
   assert strip_binding[obj.name][i]==new,'Body transfer erased spring overlap'
  if any(n in sides for n in new) and not any(n in sides for n in old):
   z=(m@obj.data.vertices[i].co).z
   if any(h['previous_start']<z<h['start'] for h in report['handoffs']):upper_gains+=1
assert result['cross_leg_count']==0 and result['cross_chain_count']==0,result
assert result['max_influences']<=4 and result['max_chains']<=2,result
assert result['direct_leg_vertices']==0,result
assert result['max_handoff_jump']<=base['max_handoff_jump']+1e-6,(base,result)
assert result['handoff_edge_energy']<=base['handoff_edge_energy']+1e-6,(base,result)
assert result['full_handoff_edge_energy']<base['full_handoff_edge_energy'],(base,result)
for o in meshes:
 owned={i for i,w in enumerate(before[o.name]) if any(n in sides for n in w)}
 slots={f.material_index for f in o.data.polygons if any(i in owned for i in f.vertices)}
 allowed={i for f in o.data.polygons if f.material_index in slots for i in f.vertices}
 for i,(a,b) in enumerate(zip(before[o.name],after[o.name])):
  assert abs(sum(b.values())-1)<1e-6
  if a!=b:
   assert i in allowed,(o.name,i,'non-garment changed')
   assert not any(n.startswith('Secondary_Hair_') for n in a)
with rest_edit(r):
 rebind_skirt_strips(r,meshes);transfer_garment_waist(r,meshes)
again=binding()
assert all(max([abs(a.get(n,0)-b.get(n,0)) for n in a.keys()|b.keys()]+[0])<1e-6 for name,ws in after.items() for a,b in zip(ws,again[name]))
verify(meshes,geometry);assert all(r.pose.bones[n].matrix_basis==v for n,v in pose.items())
output=dict(before=base,after=result,binding=report,transfer=transfer,new_upper_blend_vertices=upper_gains,body_transfer_preserves_springs=True,midline_half_width=attach.midline_width,geometry_unchanged=True,body_hair_unchanged=True,pose_preserved=True,idempotent=True)
(out/'regression.json').write_text(json.dumps(output,indent=2));(out/'weights.json').write_text(json.dumps(dict(geometry=geometry,weights=after)))
bpy.ops.wm.save_as_mainfile(filepath=str(out/'fixed.blend'));print('HANDOFF_OK',json.dumps(output),flush=True)
