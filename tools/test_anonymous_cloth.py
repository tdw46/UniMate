"""Anonymous thin-panel admission and solid-body rejection, without artist data."""
import bpy,bmesh,numpy as np,os,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_anonymous_cloth import _bind_cut_fill, _close_coat_gaps, _opening_mask, plan_panels
from avatar_coat_binding import _limit_four, close_enclosed_patches, interpolate_coat


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
  assert any(row.get('Hips',0.) > .25 for r in regions for i,row in r['values'].items()
             if points[i,2] > .575)
  obj.data.materials[0].name='Skin'
  assert not plan_panels(rig,obj,points,set(),1.,'Secondary_')[0],'Semantic body should not enter fallback'
 np.testing.assert_array_equal(before,np.array([v.co[:] for v in obj.data.vertices]))
 reports.append(dict(thin=thin,chains=len(chains)))
 bpy.data.objects.remove(obj,do_unlink=True)
coat = surface('Unlabelled long coat', (0, -.13, .53), (.14, .004, .31))
underlayer = surface('Unlabelled lining test', (0, -.113, .693), (.08, .003, .063))
bm = bmesh.new()
bm.from_mesh(coat.data)
outer = min(bm.verts, key=lambda v: (v.co.x**2 + (v.co.z-.64)**2))
original = set(bm.verts)
bm.from_mesh(underlayer.data)
inner = min(set(bm.verts)-original, key=lambda v: (v.co-outer.co).length)
# A fused contact must not make a covered surface part of the coat.
bm.edges.new((outer, inner))
bm.to_mesh(coat.data)
bm.free()
bpy.data.objects.remove(underlayer, do_unlink=True)
points = np.array([v.co[:] for v in coat.data.vertices])
chains, regions = plan_panels(rig, coat, points, set(), 1., 'Secondary_')
extended = [c for c in chains if c.get('coat_spine') == 'UpperSpine']
assert extended and all(c['line'][0].z > .75 for c in extended)
assert all(not c.get('coat_torso') for c in chains)
for guide in extended:
 assert guide['coat_lower_root'] in guide['names'][1:]
 assert guide['line'][-1].z < .45
rows = {i: row for region in regions for i,row in region['values'].items()}
owned = set().union(*(region['coat_owned_ids'] for region in regions))
# Check free cloth away from the synthetic fused contact at x=0.
waist = [i for i in owned if .58 < points[i,2] < .65 and .045 < abs(points[i,0]) < .10]
upper = [i for i in owned if .67 < points[i,2] < .71]
attachment = [i for i in owned if .74 < points[i,2] < .77]
assert waist and upper and attachment
# The new boundary taper can touch the underlayer contact, but the free
# mid-panel remains predominantly coat at both heights.
assert np.quantile([rows[i].get('Hips',0.) for i in waist], .85) < 1e-6
assert np.median([rows[i].get('Hips',0.) for i in upper]) < .05
assert any(rows[i].get('Hips',0.) > .5 for i in attachment)
assert all(len(row) <= 4 and abs(sum(row.values())-1.) < 1e-6 for row in rows.values())
hidden = {i for i, p in enumerate(points) if p[1] > -.12}
assert hidden and all(not any(n.startswith('Secondary_') for n in rows.get(i, {})) for i in hidden)
# A recessed coat rim borrows through the source cloth, and its cap stays
# separate from the nearby body and from the opposite arm bank.
cut_points = np.array([(0, 0, 0), (.004, 0, 0), (.014, 0, 0),
                       (.027, 0, 0), (.025, 0, 0), (.014, .001, 0)])
cut_rows = _bind_cut_fill(cut_points, [(0,1), (1,2), (2,3), (3,4), (1,5)],
                          np.array([False, False, True, True, False, True]),
                          np.array([True, True, True, True, True, False]),
                          {0: {'Secondary_Coat':1.}}, {0}, 1.)
assert cut_rows == {1: {'Secondary_Coat':1.}, 2: {'Secondary_Coat':1.}}
# A belt across the opening cannot become a coat panel. A true closed upper
# front can still carry cloth where the hem opening ends.
opening = [(z, (-.4, .4)) for z in (.38, .42, .47, .51)]
probes = np.array([(0, -.13, .57), (.025, -.13, .57), (.068, -.13, .57), (0, -.13, .72)])
fused = SimpleNamespace(opening=opening, top=.65,
                        section=lambda z: SimpleNamespace(gaps=[(-.39,.4)] if z > .61 else []))
buttoned = SimpleNamespace(opening=opening, top=.65,
                           section=lambda z: SimpleNamespace(gaps=[]))
assert _opening_mask(fused, probes, (0., 0.), 1.).tolist() == [True, True, False, True]
assert not _opening_mask(buttoned, probes, (0., 0.), 1.).any()
# Close a missing cloth sample between two valid neighbors, without growing
# into the continuing body surface after the end of the panel.
strip = np.array([(i*.006, 0, 0) for i in range(10)])
strip_edges = [(i, i+1) for i in range(9)]
cloth = {'Secondary_Coat':1.}
closed = _close_coat_gaps(strip, strip_edges, np.ones(len(strip), dtype=bool),
                          {0,2}, {0:cloth,2:cloth}, lambda i: True, .020)
assert closed == {1: cloth}
# A small enclosed pocket moves with its three cloth neighbors. The second
# patch reaches protected underclothes and must remain on the body.
pocket_points = np.array([(0,0,0),(.001,0,0),(-.001,0,0),(0,.001,0),
                          (.004,0,0),(.005,0,0)])
pocket_edges = [(0,1),(0,2),(0,3),(4,1),(4,2),(4,3),(4,5)]
pocket_mass = {i:1. for i in (1,2,3)}
repairable = np.array([True,True,True,True,True,False])
assert close_enclosed_patches(pocket_points,pocket_edges,pocket_mass,repairable,1.) == {0:1.}
fringe_points = np.array([(i*.006,-.13,0) for i in [0,1,2,3,4,2]])
fringe_rows = {i:{} for i in range(1,len(fringe_points))}
fringe_weights = interpolate_coat(
    fringe_points, [(0,1),(1,2),(2,3),(3,4),(5,3)], fringe_rows, lambda i:{'Hips':1.},
    [(np.array([-.1,.1]), ['Secondary_A','Secondary_B'])], [0.], (0.,0.), 1.,
    dict.fromkeys(fringe_rows,1.), -np.inf)
body = [fringe_weights[i].get('Hips',0.) for i in (1,2,3,4)]
assert 1. > body[0] > body[1] > body[2] > body[3] == 0.
assert fringe_weights[2] == fringe_weights[5]
assert set(fringe_weights) == set(fringe_rows)
assert len(_limit_four(dict(a=.4,b=.3,c=.15,d=.09,e=.06))) == 4
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
assert len(chains) == len(shorter)
for chain in chains:
 old_guide = base[chain['names'][0].rsplit('_', 1)[0]]
 count = len(old_guide['names'])
 assert chain['names'][-count:] == old_guide['names']
 assert chain['line'][-count:] == old_guide['line']
# Exercise the same Blender RNA writer used during generation.
from avatar_apparel_weights import assign
for region in regions:
 for index, row in region['values'].items():
  assign(coat, [index], row)
reports.append(dict(long_coat=True, extended=len(extended), lower_guides_preserved=True))
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,cases=reports)))
print('ANONYMOUS_CLOTH_PASSED',reports)
