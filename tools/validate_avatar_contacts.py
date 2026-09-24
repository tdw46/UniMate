"""Audit evaluated skirt triangles against surfaces bound to the legs.

Includes stockings and shoes. This detects intersecting triangle surfaces, not
cloth self-collision or a body surface entirely enclosed by another solid.
Run on a recorded walk with simulation disabled so every frame is reproducible.
"""
import argparse
import bpy, json, sys
from pathlib import Path
from mathutils.bvhtree import BVHTree
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('output', nargs='?', default=str(root/'outputs/spring_walk_demo/final'))
parser.add_argument('--name', default='sample_o_walk')
parser.add_argument('--report-only', action='store_true')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
out=Path(args.output)
bpy.ops.wm.open_mainfile(filepath=str(out/(args.name+'_baked.blend')))
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
parts=[]
for o in meshes:
    o.data.calc_loop_triangles()
    groups={g.index:g.name for g in o.vertex_groups}
    skirt={v.index for v in o.data.vertices if any(groups[g.group].startswith('Secondary_Skirt_') and g.weight>1e-5 for g in v.groups)}
    legs={v.index for v in o.data.vertices if any(groups[g.group].startswith(('Thigh.','Shin.')) and g.weight>.4 for g in v.groups)}
    st=[tuple(t.vertices) for t in o.data.loop_triangles if all(i in skirt for i in t.vertices)]
    lt=[tuple(t.vertices) for t in o.data.loop_triangles if all(i in legs for i in t.vertices)]
    if st or lt: parts.append((o,st,lt))
    print('PART',o.name,len(st),len(lt))
results=[]
for f in range(bpy.context.scene.frame_end+1):
    bpy.context.scene.frame_set(f)
    verts=[]; skirt=[]; legs=[]
    for o,st,lt in parts:
        e=o.evaluated_get(bpy.context.evaluated_depsgraph_get()); m=e.to_mesh(); start=len(verts)
        verts.extend([o.matrix_world@v.co for v in m.vertices]); e.to_mesh_clear()
        skirt.extend([tuple(i+start for i in t) for t in st]); legs.extend([tuple(i+start for i in t) for t in lt])
    hits=BVHTree.FromPolygons(verts,skirt,all_triangles=True).overlap(BVHTree.FromPolygons(verts,legs,all_triangles=True))
    results.append({'frame':f,'intersections':len(hits)})
print('SKIRT_CONTACT_AUDIT',len(results),'frames; maximum triangle intersections:',max(r['intersections'] for r in results))
(out/'surface_contact.json').write_text(json.dumps(results,indent=2))

if not args.report_only:
    assert all(r['intersections']==0 for r in results), 'Skirt/leg surface intersections remain'
