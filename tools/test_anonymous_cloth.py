"""Anonymous thin-panel admission and solid-body rejection, without artist data."""
import bpy,bmesh,numpy as np,os,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_anonymous_cloth import plan_panels


def surface(name,location,scale):
    bpy.ops.mesh.primitive_cube_add(size=2,location=location)
    obj=bpy.context.object;obj.name=name;obj.scale=scale
    bpy.ops.object.transform_apply(location=True,rotation=False,scale=True)
    bm=bmesh.new();bm.from_mesh(obj.data);bmesh.ops.subdivide_edges(bm,edges=list(bm.edges),cuts=20,use_grid_fill=True);bm.to_mesh(obj.data);bm.free();obj.data.update()
    obj.data.materials.append(bpy.data.materials.new('Material'))
    group=obj.vertex_groups.new(name='Hips');group.add(list(range(len(obj.data.vertices))),1.,'REPLACE')
    return obj

bpy.ops.wm.read_factory_settings(use_empty=True)
rig=bpy.data.objects.new('Test rig',bpy.data.armatures.new('Test rig'));bpy.context.scene.collection.objects.link(rig);bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
for n,a,b in [('Hips',(0,0,.6),(0,0,.7)),('Foot.L',(.1,0,0),(.1,-.1,0)),('Foot.R',(-.1,0,0),(-.1,-.1,0))]:
 bone=rig.data.edit_bones.new(n);bone.head=a;bone.tail=b
bpy.ops.object.mode_set(mode='OBJECT')
reports=[]
for thin in (False,True):
 obj=surface('Unlabelled surface',(0,-.13,.45),(.14,.004 if thin else .09,.15))
 points=np.array([v.co[:] for v in obj.data.vertices]);before=points.copy()
 chains,regions=plan_panels(rig,obj,points,set(),1.,'Secondary_')
 assert bool(chains)==thin,(thin,len(chains))
 if thin:
  assert all(len(row)<=4 and abs(sum(row.values())-1)<1e-6 for r in regions for row in r['values'].values())
  assert all(c['parent']=='Hips' and c['physics_start']==1 for c in chains)
  obj.data.materials[0].name='Skin'
  assert not plan_panels(rig,obj,points,set(),1.,'Secondary_')[0],'Semantic body should not enter fallback'
 np.testing.assert_array_equal(before,np.array([v.co[:] for v in obj.data.vertices]))
 reports.append(dict(thin=thin,chains=len(chains)))
 bpy.data.objects.remove(obj,do_unlink=True)
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,cases=reports)))
print('ANONYMOUS_CLOTH_PASSED',reports)
