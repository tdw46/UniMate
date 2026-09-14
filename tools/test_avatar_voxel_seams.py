"""Procedural neck seams with split UVs, varied tessellation, scale, and origin."""
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Matrix, Quaternion

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from avatar_voxel_seams import correct_skin_seams
from avatar_apparel_weights import assign, weights


def skeleton():
    r=bpy.data.objects.new('Anonymous skeleton',bpy.data.armatures.new('Bones'))
    bpy.context.scene.collection.objects.link(r)
    bpy.context.view_layer.objects.active=r
    r.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    specs=[('Root',(0,0,0),(0,0,.2),None),('Spine',(0,0,.2),(0,0,.5),'Root'),
           ('Chest',(0,0,.5),(0,0,1),'Spine'),('Neck',(0,0,1),(0,0,1.3),'Chest'),
           ('Head',(0,0,1.3),(0,0,1.8),'Neck')]
    for side,sign in [('L',1),('R',-1)]:
        for prefix,a,b,parent in [('Clavicle',.1,.4,'Chest'),('UpperArm',.4,.8,'Clavicle.'+side),
                                  ('Forearm',.8,1.2,'UpperArm.'+side),('Hand',1.2,1.4,'Forearm.'+side)]:
            specs.append((prefix+'.'+side,(sign*a,0,.95),(sign*b,0,.9),parent))
    for name,a,b,parent in specs:
        bone=r.data.edit_bones.new(name);bone.head=a;bone.tail=b
        if parent:bone.parent=r.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    return r


def tube(name,segments,heights,cap,role,rig):
    # Each ring duplicates its first vertex at the UV cut, intentionally.
    points=[(.22*math.cos(math.tau*j/segments),.22*math.sin(math.tau*j/segments),z)
            for z in heights for j in range(segments+1)]
    stride=segments+1
    faces=[(k*stride+j,k*stride+j+1,(k+1)*stride+j+1,(k+1)*stride+j)
           for k in range(len(heights)-1) for j in range(segments)]
    faces.append(tuple(reversed(range(segments))) if cap=='bottom' else
                 tuple((len(heights)-1)*stride+j for j in range(segments)))
    data=bpy.data.meshes.new(name);data.from_pydata(points,[],faces)
    obj=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(obj)
    uv=data.uv_layers.new()
    for loop in data.loops:
        uv.data[loop.index].uv=(loop.vertex_index%stride/segments,loop.vertex_index//stride/(len(heights)-1))
    obj['binding_surface']='skin';obj['binding_role']=role
    assign(obj,list(range(len(points))),{'Head' if role=='head' else 'Neck':1.})
    mod=obj.modifiers.new('Fresh deformation','ARMATURE');mod.object=rig
    return obj


def evaluated(obj):
    ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
    points=[v.co.copy() for v in mesh.vertices];ev.to_mesh_clear();return points


report=[]
for segments,scale,offset in [(12,.6,(0,0,0)),(24,1.,(3,-2,4)),(32,2.3,(-1,4,2))]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig=skeleton()
    body=tube('Anonymous A',segments,[.6,.8,1.,1.1,1.2],'bottom','body',rig)
    head=tube('Anonymous B',segments,[1.2,1.35,1.6,1.85],'top','head',rig)
    clothing=body.copy();clothing.data=body.data.copy();clothing.name='Nearby garment'
    bpy.context.scene.collection.objects.link(clothing)
    clothing['binding_surface']='cloth';clothing['binding_role']='torso'
    assign(clothing,list(range(len(clothing.data.vertices))),{'Chest':1.})
    hair=head.copy();hair.data=head.data.copy();hair.name='Nearby hair'
    bpy.context.scene.collection.objects.link(hair);hair['binding_surface']='hair'
    meshes=[body,head,clothing,hair]
    transform=Matrix.Translation(offset) @ Matrix.Scale(scale,4)
    rig.data.transform(transform)
    for o in meshes:o.data.transform(transform)
    geometry={o:([tuple(v.co) for v in o.data.vertices],[tuple(v.uv) for v in o.data.uv_layers.active.data]) for o in meshes}
    untouched={o:[weights(o,i) for i in range(len(o.data.vertices))] for o in (clothing,hair)}
    audit=correct_skin_seams(list(reversed(meshes)),rig)
    assert audit['applied'] and audit['head_boundary_clusters']>=segments
    for o in meshes:
        assert geometry[o]==([tuple(v.co) for v in o.data.vertices],[tuple(v.uv) for v in o.data.uv_layers.active.data])
    for o in (clothing,hair):assert untouched[o]==[weights(o,i) for i in range(len(o.data.vertices))]
    stride=segments+1
    assert all(weights(body,4*stride+i)=={'Head':1.} for i in range(stride))
    assert all(weights(head,i)=={'Head':1.} for i in range(stride))
    max_gap=0.
    for bone in ('Head','Neck'):
        for axis in ((1,0,0),(0,1,0),(0,0,1)):
            for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
            basis=rig.data.bones[bone].matrix_local.to_quaternion()
            rig.pose.bones[bone].rotation_mode='QUATERNION'
            rig.pose.bones[bone].rotation_quaternion=basis.inverted() @ Quaternion(axis,.8) @ basis
            bpy.context.view_layer.update();a=evaluated(body);b=evaluated(head)
            max_gap=max(max_gap,max((a[4*stride+i]-b[i]).length for i in range(stride)))
            for o,points in [(body,a),(head,b)]:
                max_gap=max(max_gap,max((points[i]-points[i+segments]).length for i in range(0,len(points),stride)))
    assert max_gap<scale*1e-5
    report.append({'segments':segments,'scale':scale,'offset':offset,'uv_duplicates':True,
                   'head_boundary_weight':1.,'garments_and_hair_unchanged':True,
                   'max_seam_gap':max_gap,'proxy':audit,'passed':True})
(ROOT/'outputs/complex_avatar_grid/seam_generalization.json').write_text(json.dumps(report,indent=2))
print('PROCEDURAL_SEAMS_PASSED',json.dumps(report),flush=True)
