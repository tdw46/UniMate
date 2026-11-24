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
for n,a,b in [('Hips',(0,0,.6),(0,0,.7)),('Spine',(0,0,.7),(0,0,.77)),('UpperSpine',(0,0,.77),(0,0,.84)),('Foot.L',(.1,0,0),(.1,-.1,0)),('Foot.R',(-.1,0,0),(-.1,-.1,0))]:
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
coat = surface('Unlabelled long coat', (0, -.13, .53), (.14, .004, .31))
points = np.array([v.co[:] for v in coat.data.vertices])
chains, regions = plan_panels(rig, coat, points, set(), 1., 'Secondary_')
extended = [c for c in chains if c.get('coat_spine') == 'UpperSpine']
assert extended and all(c['line'][0].z > .75 for c in extended)
lower_guides = {c['names'][0].rsplit('_', 1)[0]: c for c in chains if not c.get('coat_torso')}
for upper in extended:
 assert upper.get('coat_torso')
 stem = upper['names'][0].rsplit('_Torso_', 1)[0]
 lower = lower_guides[stem]
 assert upper['line'][-1] == lower['line'][0]
 assert min(p.z for p in lower['line']) < .45
assert any(row.get('Hips',0.) > .25 for r in regions for i,row in r['values'].items() if points[i,2]>.66)
# Removing the scaffold landmarks must leave the original lower guides identical.
bpy.ops.object.select_all(action='DESELECT')
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='EDIT')
for name in ('UpperSpine', 'Spine'):
 rig.data.edit_bones.remove(rig.data.edit_bones[name])
bpy.ops.object.mode_set(mode='OBJECT')
shorter, _ = plan_panels(rig, coat, points, set(), 1., 'Secondary_')
base = {c['names'][0].rsplit('_', 1)[0]: c for c in shorter}
for chain in (c for c in chains if not c.get('coat_torso')):
 old_guide = base[chain['names'][0].rsplit('_', 1)[0]]
 count = len(old_guide['names'])
 assert chain['names'][-count:] == old_guide['names']
 assert chain['line'][-count:] == old_guide['line']
reports.append(dict(long_coat=True, extended=len(extended), lower_guides_preserved=True))
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,cases=reports)))
print('ANONYMOUS_CLOTH_PASSED',reports)
