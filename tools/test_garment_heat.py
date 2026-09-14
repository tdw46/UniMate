"""General garment heat selection: complete shells, failures and detached details."""
import sys,json
import bpy
from pathlib import Path
from types import SimpleNamespace
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from avatar_apparel_weights import garment_heat_solution, preserved_garment_vertices, garment_heat_blend, garment_strain_score
neck=SimpleNamespace(head_local=Vector((0,0,1.2)),length=.3)
rig=SimpleNamespace(data=SimpleNamespace(bones={'Neck':neck,'UpperArm.L':SimpleNamespace(head_local=Vector((.4,0,1))),
                                             'UpperArm.R':SimpleNamespace(head_local=Vector((-.4,0,1)))}))
report=[]
for count in (7,53,401):
    obj=SimpleNamespace(name='Anonymous shell '+str(count))
    region={'object':obj,'indices':list(range(count)),'role':'torso','low':Vector((-.8,-.2,0)), 'high':Vector((.8,.2,1.1))}
    valid={obj.name:{i:{'Chest':.7,'UpperArm.L':.3} for i in range(count)}}
    assert garment_heat_solution(region,rig,valid) is valid[obj.name]
    unnormalized={obj.name:{i:{'Chest':.6,'UpperArm.L':.3} for i in range(count)}}
    assert garment_heat_solution(region,rig,unnormalized) is unnormalized[obj.name]
    incomplete={obj.name:dict(valid[obj.name])};incomplete[obj.name][count//2]={}
    assert garment_heat_solution(region,rig,incomplete) is None
    detail=dict(region,low=Vector((-.1,-.4,.9)),high=Vector((.1,-.3,1.1)))
    assert garment_heat_solution(detail,rig,valid) is None
    assert garment_heat_solution(dict(region,role='head'),rig,valid) is None
    assert garment_heat_solution(region,rig,None) is None
    report.append({'vertices':count,'complete_shell_preserved':True,'incomplete_shell_falls_back_as_whole':True,'small_accessory_unchanged':True,'passed':True})
# Independent face fixtures: top, underside, front/back, and flat cut cap.
for side,sign in [('L',1),('R',-1)]:
    arm=rig.data.bones['UpperArm.'+side];arm.tail_local=arm.head_local+Vector((sign*.5,0,0));arm.length=.5
points=[];faces=[]
for center,normal in [((.5,0,1.1),(0,0,1)),((.5,0,.9),(0,0,-1)),((.5,-.2,1),(0,-1,0)),((.5,.2,1),(0,1,0)),((0,0,0),(0,0,-1))]:
    start=len(points);points.extend([SimpleNamespace(co=Vector(center)) for _ in range(3)])
    faces.append(SimpleNamespace(vertices=(start,start+1,start+2),center=Vector(center),normal=Vector(normal)))
obj=SimpleNamespace(data=SimpleNamespace(vertices=points,polygons=faces))
region={'object':obj,'indices':list(range(15)),'low':Vector((0,0,0))}
assert preserved_garment_vertices(region,rig)=={0,1,2,3,4,5,12,13,14}
attachments={i:{'Chest':1.} for i in range(15)}
attachments[6]={'Neck':.2,'Chest':.8}
assert preserved_garment_vertices(region,rig,attachments)=={0,1,2,3,4,5,6,7,8,12,13,14}
chain=SimpleNamespace(data=SimpleNamespace(vertices=[SimpleNamespace(co=Vector((i*.05,0,0))) for i in range(5)],
                                          edges=[SimpleNamespace(vertices=(i,i+1)) for i in range(4)]))
blend=garment_heat_blend({'object':chain,'indices':list(range(5))},rig,{0})
assert blend[0]==0 and blend[4]==1 and 0<blend[1]<blend[2]<blend[3]
mesh=bpy.data.meshes.new('Anonymous test surface')
mesh.from_pydata([(.2+x*.3,-.2,.9+z*.15) for z in range(3) for x in range(3)],[],
                 [(z*3+x,z*3+x+1,(z+1)*3+x+1,(z+1)*3+x) for z in range(2) for x in range(2)])
obj=bpy.data.objects.new('Anonymous test surface',mesh)
region={'object':obj,'indices':list(range(9))}
coherent={i:{'Chest':.5,'UpperArm.L':.5} for i in range(9)}
alternating={i:{'Chest' if i%2 else 'UpperArm.L':1.} for i in range(9)}
assert garment_strain_score(region,rig,coherent)<garment_strain_score(region,rig,alternating)
report.append({'protected_complete_faces':True,'surface_distance_fade':True,'strain_prefers_coherent_weights':True,'passed':True})
(ROOT/'outputs/complex_avatar_grid/garment_heat_generalization.json').write_text(json.dumps(report,indent=2))
print('GARMENT_HEAT_GENERALIZATION_PASSED',flush=True)
