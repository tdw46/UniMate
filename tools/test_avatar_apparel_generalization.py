"""Name/tessellation/scale invariance and unseen procedural garment cases."""
import json
import math
from pathlib import Path
import sys

import bpy
import bmesh
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from avatar_apparel_weights import assign, classify_regions, correct_apparel, weights

OUT = ROOT/'outputs/avatar_grid'


def signature(regions, scale=1, offset=(0,0,0)):
    return [(tuple((v-offset[i])/scale for i,v in enumerate(r['low'])),
             tuple((v-offset[i])/scale for i,v in enumerate(r['high'])), r['role']) for r in regions]


def verify_role_weights(regions):
    for r in regions:
        if r['role'] == 'neckwear':
            assert all(abs(weights(r['object'], i).get('Neck', 0)-1) < 1e-6 for i in r['indices'])
        if r['role'] == 'torso':
            assert all(weights(r['object'], i).get('Head', 0) == 0 for i in r['indices'])


report = {'invariance': [], 'procedural': []}
for entry in json.loads((OUT/'sources/manifest.json').read_text()):
    bpy.ops.wm.open_mainfile(filepath=str(OUT/'avatars'/entry['id']/'02_fresh_rig.blend'))
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    rigid = json.loads(bpy.context.scene['preparation_audit'])['fresh_rig']['rigid_parts']
    expected = signature(classify_regions(meshes, rig, rigid)[1])
    renamed = {}
    offset = (3., -2., 4.)
    transform = Matrix.Translation(offset) @ Matrix.Scale(2.3, 4)
    for i, obj in enumerate(meshes):
        attachment = rigid.get(obj.name)
        obj.name = f'Anonymous_{i:03d}'
        if attachment:
            renamed[obj.name] = attachment
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=1, use_grid_fill=True)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.transform(transform)
    rig.data.transform(transform)
    regions = classify_regions(list(reversed(meshes)), rig, renamed)[1]
    actual = signature(regions, 2.3, offset)
    assert len(actual) == len(expected)
    for low, high, role in expected:
        matches = [a for a in actual if a[2] == role and
                   max(abs(x-y) for x,y in zip(low+high, a[0]+a[1])) < 1e-5]
        assert matches, (entry['id'], low, high, role)
        actual.remove(matches[0])
    correct_apparel(meshes, rig, renamed)
    verify_role_weights(regions)
    report['invariance'].append({'id': entry['id'], 'renamed': True, 'object_order_reversed': True,
                                  'triangulated_and_subdivided': True, 'scale': 2.3,
                                  'origin_offset': offset, 'passed': True})


def synthetic_rig():
    rig = bpy.data.objects.new('Skeleton', bpy.data.armatures.new('Skeleton'))
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    specs = [('Root',(0,0,0),(0,0,.15),None), ('Spine',(0,0,.15),(0,0,.4),'Root'),
             ('Chest',(0,0,.4),(0,0,1),'Spine'), ('Neck',(0,0,1),(0,0,1.25),'Chest'),
             ('Head',(0,0,1.25),(0,0,1.6),'Neck')]
    for side, sign in [('L',1),('R',-1)]:
        specs += [('Clavicle.'+side,(sign*.15,0,.95),(sign*.5,0,.9),'Chest'),
                  ('UpperArm.'+side,(sign*.5,0,.9),(sign*1.3,0,.65),'Clavicle.'+side),
                  ('Forearm.'+side,(sign*1.3,0,.65),(sign*2,0,.4),'UpperArm.'+side),
                  ('Hand.'+side,(sign*2,0,.4),(sign*2.4,0,.4),'Forearm.'+side)]
    for name, head, tail, parent in specs:
        b = rig.data.edit_bones.new(name)
        b.head, b.tail = head, tail
        if parent:
            b.parent = rig.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig


def bake_object(obj):
    obj.data.transform(obj.matrix_world)
    obj.matrix_world = Matrix.Identity(4)
    assign(obj, list(range(len(obj.data.vertices))), {'Head': 1.})
    return obj


for count, segments, scale in [(5,8,.5), (8,12,1.), (13,20,2.5)]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig = synthetic_rig()
    outline = [(-.5,0),(.5,0),(.5,.6),(2.5,.3),(2.55,.55),(.55,1),
               (.2,1.1),(.2,1.4),(-.2,1.4),(-.2,1.1),(-.55,1),(-2.55,.55),(-2.5,.3),(-.5,.6)]
    n = len(outline)
    points = [(x,y,z) for y in (-.2,.2) for x,z in outline]
    faces = [tuple(reversed(range(n))), tuple(range(n,2*n))]
    faces += [(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
    data = bpy.data.meshes.new('Carrier geometry')
    data.from_pydata(points, [], faces)
    body = bpy.data.objects.new('Anonymous carrier', data)
    bpy.context.scene.collection.objects.link(body)
    assign(body, list(range(len(data.vertices))), {'Chest': 1.})
    beads, hair = [], []
    for i in range(count):
        t = math.tau*i/count
        bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=segments//2,
                                            radius=.10, location=(.4*math.cos(t),.35*math.sin(t),1-.08*math.sin(t)))
        beads.append(bake_object(bpy.context.object))
    for sign in (-1,1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=segments//2,
                                            radius=1, location=(sign*.18,-.23,1.4))
        obj = bpy.context.object
        obj.scale = (.07,.06,.3)
        obj.rotation_euler.y = sign*.6
        hair.append(bake_object(obj))
    bpy.ops.mesh.primitive_torus_add(major_segments=segments, minor_segments=6,
                                    major_radius=.38, minor_radius=.06, location=(0,0,1.04))
    collar = bake_object(bpy.context.object)
    # A lone pendant has ambiguous geometry; explicit semantics must override
    # the geometric fallback without requiring a recognizable object name.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0,-.4,.75))
    pendant = bpy.context.object
    pendant.scale = (.08,.04,.35)
    bake_object(pendant)
    pendant['binding_role'] = 'neckwear'
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,.65))
    jacket = bpy.context.object
    jacket.scale = (1.4,.55,.95)
    bake_object(jacket)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    rigid = {o.name: 'Head' for o in beads+hair+[collar,jacket]}
    for obj in meshes:
        obj.data.transform(Matrix.Scale(scale,4))
    rig.data.transform(Matrix.Scale(scale,4))
    _, regions = classify_regions(meshes, rig, rigid)
    roles = {r['object']: r['role'] for r in regions}
    assert all(roles[o] == 'neckwear' for o in beads+[collar,pendant]), roles
    assert all(roles[o] == 'head' for o in hair), roles
    assert roles[jacket] == 'torso', roles
    correct_apparel(meshes, rig, rigid)
    verify_role_weights(regions)
    report['procedural'].append({'beads': count, 'sphere_segments': segments,
                                  'scale': scale, 'collar_and_jacket': True,
                                  'explicit_ambiguous_pendant_role': True,
                                  'elongated_hair_decoys': len(hair), 'passed': True})
report['passed'] = True
(OUT/'apparel_generalization.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
