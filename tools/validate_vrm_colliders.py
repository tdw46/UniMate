"""Compare leg/skirt contact and prove VRM collider parenting/export round trips."""
import json,math,sys,struct,argparse
from pathlib import Path
import bpy
from mathutils import Matrix,Quaternion,Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_physics_preview import set_simulation,background_step
from avatar_colliders import rebuild_colliders,rest_contacts
from avatar_apparel_weights import weights
parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('output');args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
for repo in bpy.context.preferences.extensions.repos:
    if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
bpy.ops.wm.open_mainfile(filepath=str(Path(args.source).resolve()))
rig=next(o for o in bpy.data.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'))
bpy.context.window.scene=next(s for s in bpy.data.scenes if rig.name in s.objects)
scene=bpy.context.scene
bpy.context.view_layer.objects.active=rig
if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
body={p.name:p.matrix_basis.copy() for p in rig.pose.bones if not p.name.startswith('Secondary_')}
keep={rig,*rig.children_recursive}
for obj in list(bpy.data.objects):
    if obj not in keep:bpy.data.objects.remove(obj,do_unlink=True)
meshes=[o for o in rig.children if o.type=='MESH']
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite')
set_simulation(False);rig.animation_data_clear()
for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
parts=[]
humanoid=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
leg_names={getattr(humanoid,s+'_'+n).node.bone_name for s in ('left','right') for n in ('upper_leg','lower_leg')}
for obj in meshes:
    obj.data.calc_loop_triangles()
    ws=[weights(obj,v.index) for v in obj.data.vertices]
    skirt={i for i,w in enumerate(ws) if sum(v for n,v in w.items() if n.startswith('Secondary_Skirt_'))>.05}
    legs={i for i,w in enumerate(ws) if sum(v for n,v in w.items() if n in leg_names)>.4}
    st=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in skirt for i in t.vertices)]
    lt=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in legs for i in t.vertices)]
    if st or lt:parts.append((obj,st,lt))

def contacts():
    vertices=[];skirt=[];legs=[]
    for obj,st,lt in parts:
        ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();start=len(vertices)
        vertices.extend(obj.matrix_world@v.co for v in mesh.vertices);ev.to_mesh_clear()
        skirt.extend(tuple(i+start for i in tri) for tri in st);legs.extend(tuple(i+start for i in tri) for tri in lt)
    hits=BVHTree.FromPolygons(vertices,skirt,all_triangles=True).overlap(BVHTree.FromPolygons(vertices,legs,all_triangles=True))
    return len(hits)

def evaluate():
    result={}
    left=humanoid.left_upper_leg.node.bone_name
    for name,target in [('rest',Quaternion()),('user',body[left].to_quaternion()),('forward',Quaternion((1,0,0),.7)),('outward',Quaternion((0,0,1),-.7))]:
        set_simulation(False)
        for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
        bpy.context.view_layer.update();set_simulation(True)
        values=[]
        for frame in range(75):
            pb=rig.pose.bones[left];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion().slerp(target,min(1.,frame/35))
            bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
            if frame in (35,50,74):values.append(contacts())
        result[name]=values
    set_simulation(False)
    for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()
    return result
report={'before':evaluate()}
hair_ids={c.uuid for c in rig.data.vrm_addon_extension.spring_bone1.colliders if c.node.bone_name not in leg_names}
report['fit']=rebuild_colliders(rig,meshes,collider_roles=('skirt',))
first_count=len(rig.data.vrm_addon_extension.spring_bone1.colliders)
rebuild_colliders(rig,meshes,collider_roles=('skirt',))
assert len(rig.data.vrm_addon_extension.spring_bone1.colliders)==first_count
assert hair_ids<={c.uuid for c in rig.data.vrm_addon_extension.spring_bone1.colliders}
report['repeat_refit_stable']=True
report['hair_collider_ids_preserved']=True
report['rest_clearance']=rest_contacts(rig)
assert not report['rest_clearance']['contacts']
report['after']=evaluate()
assert sum(map(sum,report['after'].values())) <= sum(map(sum,report['before'].values())),report
(out/'contacts.json').write_text(json.dumps(report,indent=2))
print('COLLIDER_CONTACT_COMPARISON',json.dumps({k:report[k] for k in ('before','after')}),flush=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'fitted.blend'))
# Cache local RNA values once, then test actual helper movement at other poses.
sb=rig.data.vrm_addon_extension.spring_bone1
specs=[(c,Vector(c.shape.capsule.offset),Vector(c.shape.capsule.tail),c.shape.capsule.radius) for c in sb.colliders]
maximum=0.
for axis in ((1,0,0),(0,0,1)):
    for angle in (-.7,.7):
        for side in ('left','right'):
            for role in ('upper_leg','lower_leg'):
                pb=rig.pose.bones[getattr(humanoid,side+'_'+role).node.bone_name];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion(axis,angle)
        bpy.context.view_layer.update()
        for c,a,b,radius in specs:
            assert c.bpy_object.parent==rig and c.bpy_object.parent_bone==c.node.bone_name
            transform=rig.matrix_world@rig.pose.bones[c.node.bone_name].matrix
            maximum=max(maximum,(c.bpy_object.matrix_world.translation-transform@a).length,(c.bpy_object.children[0].matrix_world.translation-transform@b).length)
original_location=rig.location.copy()
rig.location+=Vector((.23,-.17,.09));bpy.context.view_layer.update()
for c,a,b,radius in specs:
    transform=rig.matrix_world@rig.pose.bones[c.node.bone_name].matrix
    maximum=max(maximum,(c.bpy_object.matrix_world.translation-transform@a).length,(c.bpy_object.children[0].matrix_world.translation-transform@b).length)
rig.location=original_location;bpy.context.view_layer.update()
assert maximum<1e-5,maximum
for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
meta=rig.data.vrm_addon_extension.vrm1.meta
meta.vrm_name='Collider Validation'
if not meta.authors:meta.authors.add().value='Local validation'
bpy.ops.object.select_all(action='DESELECT')
for o in [rig]+meshes:o.select_set(True)
path=out/'colliders.vrm'
assert bpy.ops.export_scene.vrm(filepath=str(path),armature_object_name=rig.name,export_only_selections=True,export_invisibles=True,export_gltf_animations=False,ignore_warning=True)=={'FINISHED'}
raw=path.read_bytes();size,kind=struct.unpack_from('<II',raw,12);gltf=json.loads(raw[20:20+size]);ext=gltf['extensions']['VRMC_springBone']
assert len(ext['colliders'])==len(sb.colliders)
assert all('capsule' in c['shape'] for c in ext['colliders'])
assert all(g['colliders'] and max(g['colliders'])<len(ext['colliders']) for g in ext['colliderGroups'])
# Endpoints in armature rest coordinates should survive import regardless of bone roll.
expected=[(rig.data.bones[c.node.bone_name].matrix_local@a,rig.data.bones[c.node.bone_name].matrix_local@b,r) for c,a,b,r in specs]
for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
assert bpy.ops.import_scene.vrm(filepath=str(path))=={'FINISHED'}
imported=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
actual=imported.data.vrm_addon_extension.spring_bone1.colliders
assert len(actual)==len(expected)
errors=[]
for c,(a,b,r) in zip(actual,expected):
    matrix=imported.data.bones[c.node.bone_name].matrix_local
    errors.append(max((matrix@Vector(c.shape.capsule.offset)-a).length,(matrix@Vector(c.shape.capsule.tail)-b).length,abs(c.shape.capsule.radius-r)))
assert max(errors)<1e-5,errors
report['export']={'colliders':len(actual),'groups':len(ext['colliderGroups']),'maximum_parenting_error':maximum,'maximum_roundtrip_error':max(errors)}
(out/'validation.json').write_text(json.dumps(report,indent=2))
print('VRM_COLLIDER_ROUNDTRIP_OK',json.dumps(report['export']),flush=True)
