"""Independent shoulder cylinders: density, mirrored arms, scale and exclusions."""
import json, math, sys
from pathlib import Path
import bpy
from mathutils import Matrix, Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from avatar_shoulder_weights import smooth_shoulders
from avatar_apparel_weights import assign, weights

def make_case(segments,scale,offset):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig=bpy.data.objects.new('Skeleton',bpy.data.armatures.new('Bones'));bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
    specs=[('Root',(0,0,0),(0,0,.2)),('Spine',(0,0,.2),(0,0,.5)),('Chest',(0,0,.5),(0,0,1.1)),('Neck',(0,0,1.1),(0,0,1.4)),('Head',(0,0,1.4),(0,0,1.8))]
    for side,sign in [('L',1),('R',-1)]:
        for prefix,a,b in [('Clavicle',.1,.5),('UpperArm',.5,1.),('Forearm',1.,1.5),('Hand',1.5,1.7)]:
            specs.append((prefix+'.'+side,(sign*a,0,1),(sign*b,0,1)))
    for name,a,b in specs:
        bone=rig.data.edit_bones.new(name);bone.head=a;bone.tail=b
    bpy.ops.object.mode_set(mode='OBJECT')
    meshes=[];protected={};old={};geometry={}
    for side,sign in [('L',1),('R',-1)]:
        points=[(sign*(.25+k*.05),.12*math.cos(math.tau*j/segments),1+.12*math.sin(math.tau*j/segments)) for k in range(11) for j in range(segments)]
        faces=[(k*segments+j,k*segments+(j+1)%segments,(k+1)*segments+(j+1)%segments,(k+1)*segments+j) for k in range(10) for j in range(segments)]
        data=bpy.data.meshes.new('Surface');data.from_pydata(points,[],faces);data.update()
        obj=bpy.data.objects.new('Anonymous '+side,data);bpy.context.scene.collection.objects.link(obj);obj['binding_role']='body';meshes.append(obj)
        uv=data.uv_layers.new()
        for loop in data.loops:uv.data[loop.index].uv=(loop.vertex_index/len(points),.5)
        for v in data.vertices:assign(obj,[v.index],{'Chest' if abs(v.co.x)<.5 else 'UpperArm.'+side:1.})
        # Explicit Head boundary and Neck influences remain immutable even if
        # unusually placed inside a candidate shoulder sector.
        assign(obj,[5*segments],{'Head':1.});assign(obj,[6*segments],{'Neck':.3,'Chest':.7})
        protected[obj]={i:weights(obj,i) for i,p in enumerate(points) if abs(p[2]-1)>.10 or i in (5*segments,6*segments)}
        old[obj]=[weights(obj,i) for i in range(len(points))]
    transform=Matrix.Translation(offset) @ Matrix.Scale(scale,4);rig.data.transform(transform)
    for obj in meshes:
        obj.data.transform(transform);obj.data.update()
        geometry[obj]=([tuple(v.co) for v in obj.data.vertices],[tuple(x.uv) for x in obj.data.uv_layers.active.data])
    audit=smooth_shoulders(list(reversed(meshes)),rig,{})
    changes=0;energy=[0.,0.]
    for obj in meshes:
        assert geometry[obj]==([tuple(v.co) for v in obj.data.vertices],[tuple(x.uv) for x in obj.data.uv_layers.active.data])
        new=[weights(obj,i) for i in range(len(obj.data.vertices))]
        for i,w in protected[obj].items():assert new[i]==w
        for i,w in enumerate(new):
            assert len(w)<=4 and abs(sum(w.values())-1)<1e-6
            changes+=w!=old[obj][i]
        for edge in obj.data.edges:
            i,j=edge.vertices
            for n,values in enumerate((old[obj],new)):
                a,b=values[i],values[j];energy[n]+=sum((a.get(k,0)-b.get(k,0))**2 for k in a.keys()|b.keys())
    assert changes>0 and energy[1]<energy[0]
    return {'segments':segments,'scale':scale,'offset':offset,'changed_vertices':changes,'edge_energy':energy,'protected_top_underside_head_neck':True,'geometry_uvs_unchanged':True,'passed':True}
report=[make_case(*case) for case in [(24,.6,(0,0,0)),(48,1.,(3,-2,4)),(64,2.3,(-1,4,2))]]
(ROOT/'outputs/complex_avatar_grid/shoulder_generalization.json').write_text(json.dumps(report,indent=2))
print('SHOULDER_GENERALIZATION_PASSED',json.dumps(report))
