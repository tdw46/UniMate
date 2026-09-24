"""Replace a neutral MMD deformation rig using semantic joints and fresh heat.

Keeps mesh topology, UVs, materials and every shape-key coordinate unchanged.
The source rig supplies joint landmarks only, never bones, weights or physics.
Run on an isolated scene; live application is a separate, guarded operation.
"""
import hashlib
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar_apparel_weights import assign, weights, enforce_head_cap
from avatar_springs import generate_secondary
from autorig_bust import repaired_heat
from avatar_voxel_seams import correct_skin_seams


def rename_mixamo(rig, meshes):
    """Use unprefixed Mixamo names while retaining all VRM node references."""
    mapping={'Chest':'Spine1'}
    for side,prefix in (('L','Left'),('R','Right')):
        for source,target in (('Clavicle','Shoulder'),('UpperArm','Arm'),('Forearm','ForeArm'),
                              ('Hand','Hand'),('Thigh','UpLeg'),('Shin','Leg'),('Foot','Foot'),('Toe','ToeBase')):
            mapping[source+'.'+side]=prefix+target
        for digit in ('Thumb','Index','Middle','Ring','Little'):
            for index in range(1,4):
                mapping[f'{digit}{index}.{side}']=f'{prefix}Hand{"Pinky" if digit=="Little" else digit}{index}'
    ext=rig.data.vrm_addon_extension
    from avatar_springs import humanoid_roles
    import re
    nodes=[getattr(ext.vrm1.humanoid.human_bones,re.sub(r'(?<!^)(?=[A-Z])','_',role).lower()).node
           for role in humanoid_roles()]
    nodes += [s.center for s in ext.spring_bone1.springs]
    nodes += [j.node for s in ext.spring_bone1.springs for j in s.joints]
    nodes += [c.node for c in ext.spring_bone1.colliders]
    references=[(node,node.bone_name) for node in nodes]
    optional_centers={s.center.as_pointer() for s in ext.spring_bone1.springs}
    for old,new in mapping.items():
        if old in rig.data.bones:rig.data.bones[old].name=new
    for node,old in references:node.bone_name=mapping.get(old,old)
    for obj in meshes:
        for old,new in mapping.items():
            if old in obj.vertex_groups:obj.vertex_groups[old].name=new
    for pb in rig.pose.bones:
        for constraint in pb.constraints:
            if hasattr(constraint,'subtarget'):constraint.subtarget=mapping.get(constraint.subtarget,constraint.subtarget)
    assert all(node.bone_name in rig.data.bones or
               (not node.bone_name and node.as_pointer() in optional_centers)
               for node,_ in references)
    assert all(g.name in rig.data.bones for o in meshes for g in o.vertex_groups)
    return mapping


def fingerprint(obj):
    h = hashlib.sha256()
    for v in obj.data.vertices:
        h.update(repr(tuple(v.co)).encode())
    for p in obj.data.polygons:
        h.update(repr((tuple(p.vertices), p.material_index)).encode())
    for uv in obj.data.uv_layers:
        h.update(uv.name.encode())
        for d in uv.data:
            h.update(repr(tuple(d.uv)).encode())
    keys = obj.data.shape_keys
    if keys:
        for key in keys.key_blocks:
            h.update(repr((key.name, key.value, key.vertex_group, key.relative_key.name)).encode())
            for v in key.data:
                h.update(repr(tuple(v.co)).encode())
    return h.hexdigest()


def specs_from_mmd(source):
    """Symmetrize corresponding semantic landmarks, discarding MMD controls."""
    bones = source.data.bones
    def point(name, tail=False):
        return (bones[name].tail_local if tail else bones[name].head_local).copy()
    specs = []
    def add(name, a, b, parent):
        specs.append((name, a, b, parent))
    hip = point('LowerBody'); hip.z = (point('Leg.L').z+point('Leg.R').z)/2
    height=point('Head',True).z-min(point('Ankle.L').z,point('Ankle.R').z)
    add('Root', Vector((0,0,0)), Vector((0,0,height*.07)), None)
    add('Hips', hip, point('UpperBody'), 'Root')
    add('Spine', point('UpperBody'), point('UpperBody2'), 'Hips')
    add('Chest', point('UpperBody2'), point('Neck'), 'Spine')
    add('Neck', point('Neck'), point('Head'), 'Chest')
    add('Head', point('Head'), point('Head', True), 'Neck')
    chain = [('Clavicle','Shoulder','Chest'), ('UpperArm','Arm','Clavicle'),
             ('Forearm','Elbow','UpperArm'), ('Hand','Wrist','Forearm'),
             ('Thigh','Leg','Hips'), ('Shin','Knee','Thigh'), ('Foot','Ankle','Shin'), ('Toe','Toe','Foot')]
    for side in ('L','R'):
        for target, origin, parent in chain:
            a,b = point(origin+'.'+side), point(origin+'.'+side, True)
            if target == 'Toe':
                direction = a-point('Ankle.'+side); direction.z = 0
                b = a+direction.normalized()*height*.028
            add(target+'.'+side, a,b,parent if parent in ('Chest','Hips') else parent+'.'+side)
        for target, origin in [('Thumb','Thumb'),('Index','IndexFinger'),('Middle','MiddleFinger'),('Ring','RingFinger'),('Little','LittleFinger')]:
            indices = (0,1,2) if target == 'Thumb' else (1,2,3)
            names = [origin+str(i)+'.'+side for i in indices]
            tip = ('ThumbTip' if target=='Thumb' else origin.replace('Finger','FingerTip'))+'.'+side
            for i,n in enumerate(names):
                add(f'{target}{i+1}.{side}',point(n),point(names[i+1] if i<2 else tip),
                    'Hand.'+side if i==0 else f'{target}{i}.{side}')
    lookup = {n:(a,b) for n,a,b,_ in specs}
    for name,a,b,_ in specs:
        if name.endswith('.L'):
            ra,rb = lookup[name[:-1]+'R']
            for left,right in ((a,ra),(b,rb)):
                mean=(left+Vector((-right.x,right.y,right.z)))*.5
                left[:]=mean;right[:]=(-mean.x,mean.y,mean.z)
        elif not name.endswith('.R'):
            a.x=b.x=0
    return specs


def rebuild(source, meshes):
    assert all(pb.matrix_basis == Matrix.Identity(4) for pb in source.pose.bones), 'Source must be neutral'
    assert source.matrix_world == Matrix.Identity(4), 'Apply rig transforms first'
    assert all(o.matrix_world == Matrix.Identity(4) for o in meshes), 'Mesh transforms must match rig space'
    before = {o.name:fingerprint(o) for o in meshes}
    specs = specs_from_mmd(source)
    data=bpy.data.armatures.new('UniMate fresh humanoid')
    rig=bpy.data.objects.new(source.name+'_UniMate',data)
    bpy.context.scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action='DESELECT'); rig.select_set(True)
    bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='EDIT')
    for name,a,b,parent in specs:
        bone=data.edit_bones.new(name);bone.head=a;bone.tail=b
        bone.parent=data.edit_bones.get(parent) if parent else None
        bone.use_deform=name!='Root'
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj in meshes:
        obj.vertex_groups.clear();obj.parent=None
        for mod in list(obj.modifiers):
            if mod.type=='ARMATURE' and mod.object==source:obj.modifiers.remove(mod)
    bpy.data.objects.remove(source,do_unlink=True)
    # One joined temporary surface gives disconnected facial parts and the neck
    # the same heat domain; original meshes and their shape keys are untouched.
    vertices,faces,ranges=[],[],{}
    rigid = set()
    for obj in meshes:
        labels={obj.data.materials[p.material_index].name.lower() for p in obj.data.polygons}
        facial=('face','eye','mouth','teeth','double','hair','headaccessory')
        if all(any(t in label for t in facial) for label in labels):
            rigid.add(obj.name)
            continue
        ranges[obj.name]=(len(vertices),len(obj.data.vertices))
        vertices.extend(tuple(v.co) for v in obj.data.vertices)
        offset=ranges[obj.name][0]
        faces.extend(tuple(offset+i for i in p.vertices) for p in obj.data.polygons)
    proxy_data=bpy.data.meshes.new('Fresh heat surface')
    proxy_data.from_pydata(vertices,[],faces);proxy_data.update()
    proxy=bpy.data.objects.new('Fresh heat surface',proxy_data)
    bpy.context.scene.collection.objects.link(proxy)
    heat=repaired_heat([proxy],rig,allow_unweighted=True)
    solved={v.index:weights(proxy,v.index) for v in proxy.data.vertices}
    proxy.data.calc_loop_triangles()
    triangles=[tuple(t.vertices) for t in proxy.data.loop_triangles if all(solved[i] for i in t.vertices)]
    assert triangles,'Heat produced no usable surface'
    tree=BVHTree.FromPolygons([v.co for v in proxy.data.vertices],triangles,all_triangles=True)
    transferred=0;max_distance=0.
    for v in proxy.data.vertices:
        if solved[v.index]:continue
        hit,normal,index,distance=tree.find_nearest(v.co)
        assert index is not None and distance<rig.data.bones['UpperArm.L'].length*.75, 'Unseeded surface too far from heat solution'
        tri=triangles[index]
        bary=barycentric_transform(hit,*(proxy.data.vertices[i].co for i in tri),Vector((1,0,0)),Vector((0,1,0)),Vector((0,0,1)))
        values={}
        for i,factor in zip(tri,bary):
            for name,w in solved[i].items():values[name]=values.get(name,0)+max(0.,factor)*w
        total=sum(values.values());assert total>0
        assign(proxy,[v.index],{n:w/total for n,w in values.items()})
        transferred+=1;max_distance=max(max_distance,distance)
    heat.append(dict(unseeded_vertices_transferred=transferred,maximum_surface_distance=max_distance,source='new heat surface only'))
    for obj in meshes:
        count=len(obj.data.vertices)
        if obj.name not in rigid:
            start,_=ranges[obj.name]
            for v in obj.data.vertices:
                values=weights(proxy,start+v.index);total=sum(values.values())
                assert total>0
                assign(obj,[v.index],{n:w/total for n,w in values.items()})
        labels={obj.data.materials[p.material_index].name.lower() for p in obj.data.polygons}
        # Semantic facial surfaces and hair are anchored to Head before adding
        # free hair chains. Mixed body objects retain their ordinary heat solve.
        facial=('face','eye','mouth','teeth','double','hair','headaccessory')
        if all(any(t in label for t in facial) for label in labels):
            assign(obj,list(range(count)),{'Head':1.});obj['binding_role']='head'
        else:obj['binding_role']='body'
        for label in obj.data.materials:
            if label and any(t in label.name.lower() for t in ('skin','face')):label['binding_surface']='skin'
        mod=obj.modifiers.new('UniMate deform','ARMATURE');mod.object=rig
        obj.parent=rig
    bpy.data.objects.remove(proxy,do_unlink=True);bpy.data.meshes.remove(proxy_data)
    enforce_head_cap(meshes,rig)
    seams=correct_skin_seams(meshes,rig,max_rings=2)
    springs=generate_secondary(rig,meshes)
    after={o.name:fingerprint(o) for o in meshes}
    assert before==after,'Geometry or shape keys changed'
    sums=[sum(weights(o,v.index).values()) for o in meshes for v in o.data.vertices]
    assert min(sums)>.9999 and max(sums)<1.0001, (min(sums),max(sums))
    assert all(g.name in rig.data.bones for o in meshes for g in o.vertex_groups)
    return rig,dict(mesh_hashes=before,heat=heat,seams=seams,springs=springs,
                    bones=len(rig.data.bones),vertices=len(sums),weight_sum_range=[min(sums),max(sums)],
                    shape_keys={o.name:len(o.data.shape_keys.key_blocks) if o.data.shape_keys else 0 for o in meshes})


def main():
    directory=Path(sys.argv[sys.argv.index('--')+1]).resolve()
    inventory=json.loads((directory/'source_inventory.json').read_text())
    bpy.ops.wm.open_mainfile(filepath=str(directory/'source_scene.blend'))
    source=bpy.data.objects[inventory['active']]
    scene=next(s for s in bpy.data.scenes if source.name in s.objects)
    bpy.context.window.scene=scene
    meshes=[bpy.data.objects[n] for n in inventory['affected_meshes']]
    for obj in list(bpy.data.objects):
        if obj not in meshes and obj!=source:bpy.data.objects.remove(obj,do_unlink=True)
    for repo in bpy.context.preferences.extensions.repos:
        if repo.module == 'user_default':
            repo.use_custom_directory = True
            repo.custom_directory = str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
    rig,report=rebuild(source,meshes)
    rig['unimate_source']='MMD semantic rest joints; new heat weights and geometry-derived springs'
    (directory/'generated_weights.json').write_text(json.dumps({o.name:[weights(o,v.index) for v in o.data.vertices] for o in meshes}))
    (directory/'generation_report.json').write_text(json.dumps(report,indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(directory/'generated.blend'))
    print('REBUILD_OK',json.dumps({k:v for k,v in report.items() if k in ('bones','vertices','weight_sum_range')}),flush=True)

if __name__=='__main__':main()
