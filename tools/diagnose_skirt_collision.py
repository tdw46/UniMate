"""BVT-only low-follow collider experiments in an isolated scene copy."""
import argparse,json,math,sys,time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Quaternion,Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_line
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_physics_preview import set_simulation,background_step
from avatar_colliders import segment_distance,spring_samples,plan_colliders
from avatar_vrm_colliders import add_capsule,add_group,call_operator
from avatar_apparel_weights import weights
from properties_hallway_rig import initialize
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--variant',default='baseline');p.add_argument('--full',action='store_true');p.add_argument('--bvt-runtime-only',action='store_true');args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
for repo in bpy.context.preferences.extensions.repos:
    if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')
bpy.ops.wm.open_mainfile(filepath=str(Path(args.source).resolve()))
if args.bvt_runtime_only:
    # Isolated tests can load the installed, unmodified solver independently
    # when unrelated BVT UI development prevents full addon registration.
    import importlib,types
    name='bl_ext.user_default.beyond_vrm_extension_suite'
    package=types.ModuleType(name)
    package.__path__=[str(Path.home()/'Documents/Blender/extensions/user_default/beyond_vrm_extension_suite')]
    sys.modules[name]=package
    solver=importlib.import_module(name+'.VRM_SpringSimulation')
    for cls in (solver.BVT_OT_SpringSimulationEngine,solver.BVT_OT_SetSpringSimulation,solver.BVT_OT_SetSpringLoopPhysics):bpy.utils.register_class(cls)
    solver.register_runtime()
else:
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite')
set_simulation(False)
rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator'));bpy.context.view_layer.objects.active=rig
if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
keep={rig,*rig.children_recursive}
for obj in list(bpy.data.objects):
    if obj not in keep:bpy.data.objects.remove(obj,do_unlink=True)
scene=bpy.context.scene
rig.animation_data_clear()
for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
settings=initialize(rig);sb=rig.data.vrm_addon_extension.spring_bone1
hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
legs=[getattr(hum,s+'_upper_leg').node.bone_name for s in ('left','right')]
leg_names={getattr(hum,s+'_'+n).node.bone_name for s in ('left','right') for n in ('upper_leg','lower_leg')}
meshes=[o for o in rig.children if o.type=='MESH']
height=max(b.head_local.z for b in rig.data.bones)-min(b.head_local.z for b in rig.data.bones)
margin=height*.0015
samples=spring_samples(rig,'skirt')
parts=[];garment=[]
for obj in meshes:
    obj.data.calc_loop_triangles();ws=[weights(obj,v.index) for v in obj.data.vertices]
    skirt={i for i,w in enumerate(ws) if sum(v for n,v in w.items() if n.startswith('Secondary_Skirt_'))>.05}
    skin={i for i,w in enumerate(ws) if sum(v for n,v in w.items() if n in leg_names)>.4}
    st=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in skirt for i in t.vertices)]
    lt=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in skin for i in t.vertices)]
    if st or lt:parts.append((obj,st,lt))
    tr=rig.matrix_world.inverted()@obj.matrix_world
    garment.extend((tr@obj.data.vertices[i].co,0.,'surface') for i in skirt)
base=[dict(bone=c.node.bone_name,offset=list(c.shape.capsule.offset),tail=list(c.shape.capsule.tail),radius=c.shape.capsule.radius) for c in sb.colliders if c.node.bone_name in leg_names]
full=[]
for name in sorted(leg_names):
    bone=rig.data.bones[name];a=Vector((0,0,0));b=Vector((0,bone.length,0))
    limit=min(segment_distance(p,bone.matrix_local@a,bone.matrix_local@b)-r-margin for p,r,_ in samples+garment)
    radius=min(max(s['radius'] for s in base if s['bone']==name),limit)
    if radius>height*.001:full.append(dict(bone=name,offset=list(a),tail=list(b),radius=radius))
if args.variant=='generated':
    from avatar_colliders import rebuild_colliders
    rebuild_colliders(rig,meshes,collider_roles=('skirt',))
    initialize(rig)
    settings.skirt_thickness=1.
if args.variant not in ('baseline','generated'):
    owned={g.uuid for g in sb.collider_groups if g.vrm_name=='Secondary_SkirtBody'}
    ids={r.collider_uuid for g in sb.collider_groups if g.uuid in owned for r in g.colliders}
    for i in reversed(range(len(sb.collider_groups))):
        if sb.collider_groups[i].uuid in owned:call_operator('remove_spring_bone1_collider_group',rig,collider_group_index=i)
    for i in reversed(range(len(sb.colliders))):
        if sb.colliders[i].uuid in ids:call_operator('remove_spring_bone1_collider',rig,collider_index=i)
    group=add_group(rig,'Secondary_SkirtBody')
    specs=full+(base if args.variant in ('combined','planes_combined') else [])
    if args.variant=='tapered':
        specs=[]
        for name in sorted(leg_names):
            bone=rig.data.bones[name]
            for index,original in enumerate(s for s in base if s['bone']==name):
                a=Vector((0,index/3*bone.length,0));b=Vector((0,(index+1)/3*bone.length,0))
                limit=min(segment_distance(p,bone.matrix_local@a,bone.matrix_local@b)-r-margin for p,r,_ in samples+garment)
                radius=min(original['radius'],limit)
                if radius>height*.001:specs.append(dict(bone=name,offset=list(a),tail=list(b),radius=radius))
    if args.variant=='per_chain':specs=[]
    for spec in specs:
        c=add_capsule(rig,spec);group.colliders.add().collider_uuid=c.uuid
    for spring in sb.springs:
        if spring.vrm_name.startswith('Secondary_Skirt_'):spring.collider_groups.add().collider_group_uuid=group.uuid
    if args.variant=='per_chain':
        for spring in sb.springs:
            if not spring.vrm_name.startswith('Secondary_Skirt_'):continue
            points=[(rig.data.bones[t.node.bone_name].head_local,h.hit_radius) for h,t in zip(spring.joints,spring.joints[1:])]
            g=add_group(rig,'Sector '+spring.vrm_name)
            for name in sorted(leg_names):
                bone=rig.data.bones[name];a=Vector((0,0,0));b=Vector((0,bone.length,0))
                limit=min(segment_distance(p,bone.matrix_local@a,bone.matrix_local@b)-r-margin for p,r in points)
                radius=min(max(s['radius'] for s in base if s['bone']==name),limit)
                if radius<=height*.001:continue
                c=add_capsule(rig,dict(bone=name,offset=list(a),tail=list(b),radius=radius))
                g.colliders.add().collider_uuid=c.uuid
            spring.collider_groups.clear();spring.collider_groups.add().collider_group_uuid=g.uuid
    if args.variant.startswith('planes'):
        for spring in sb.springs:
            if not spring.vrm_name.startswith('Secondary_Skirt_'):continue
            points=[rig.data.bones[j.node.bone_name].head_local for j in spring.joints]
            name=min(legs,key=lambda n:segment_distance(points[2],rig.data.bones[n].head_local,rig.data.bones[n].tail_local))
            bone=rig.data.bones[name];local=[bone.matrix_local.inverted()@p for p in points]
            normal=Vector((local[2].x,0,local[2].z)).normalized()
            distance=min(p.dot(normal)-j.hit_radius-margin for p,j in zip(local[1:],spring.joints))
            # One outward plane for this spring's angular sector, never six
            # incompatible outward half-spaces pretending to be a box.
            c=sb.add_collider(bpy.context,rig);c.node.bone_name=name;c.ui_collider_type='uiColliderTypePlane'
            c.reset_bpy_object(bpy.context,rig);bpy.context.view_layer.update()
            shape=c.extensions.vrmc_spring_bone_extended_collider.shape.plane
            shape.offset=normal*distance;bpy.context.view_layer.update();shape.normal=normal;bpy.context.view_layer.update()
            assert (Vector(shape.normal)-normal).length<1e-5
            assert (Vector(shape.offset)-normal*distance).length<1e-5
            g=add_group(rig,'Sector '+spring.vrm_name);g.colliders.add().collider_uuid=c.uuid
            spring.collider_groups.add().collider_group_uuid=g.uuid
    bpy.context.view_layer.update()
# Cache official shape data; world geometry follows each attachment pose bone.
colliders={}
for c in sb.colliders:
    ex=c.extensions.vrmc_spring_bone_extended_collider
    if ex.enabled and 'Plane' in ex.shape_type:
        shape=ex.shape.plane;colliders[c.uuid]=(c.node.bone_name,'plane',Vector(shape.offset),Vector(shape.normal),0.)
    elif c.shape_type=='Capsule':
        shape=c.shape.capsule;colliders[c.uuid]=(c.node.bone_name,'capsule',Vector(shape.offset),Vector(shape.tail),shape.radius)
groups={g.uuid:[colliders[r.collider_uuid] for r in g.colliders if r.collider_uuid in colliders] for g in sb.collider_groups}
pairs=[(spring,head,tail,[c for ref in spring.collider_groups for c in groups.get(ref.collider_group_uuid,[])]) for spring in sb.springs if spring.vrm_name.startswith('Secondary_Skirt_') for head,tail in zip(spring.joints,spring.joints[1:])]

def segment_segment_distance(a,b,c,d):
    result=min(segment_distance(a,c,d),segment_distance(b,c,d),segment_distance(c,a,b),segment_distance(d,a,b))
    closest=intersect_line_line(a,b,c,d)
    if closest:
        p,q=closest;ab=b-a;cd=d-c
        if 0<=(p-a).dot(ab)<=ab.length_squared and 0<=(q-c).dot(cd)<=cd.length_squared:
            result=min(result,(p-q).length)
    return result

def contacts(surface=False):
    maximum=0.;hits=0;worst=None;segment_max=0.
    matrices={n:rig.matrix_world@rig.pose.bones[n].matrix for n in leg_names}
    for spring,head,tail,cs in pairs:
        pos=Vector(tail.animation_state.current_world_translation)
        start=rig.matrix_world@rig.pose.bones[head.node.bone_name].head
        for name,kind,a,b,r in cs:
            matrix=matrices[name];a=matrix@a
            if kind=='capsule':
                b=matrix@b;penetration=r+head.hit_radius-segment_distance(pos,a,b)
                segment_max=max(segment_max,r-segment_segment_distance(start,pos,a,b))
            else:
                normal=(matrix.to_quaternion()@b).normalized();penetration=head.hit_radius-(pos-a).dot(normal)
            if penetration>height*1e-5:hits+=1
            if penetration>maximum:maximum=penetration;worst=[spring.vrm_name,head.node.bone_name,name,kind]
    triangles=None
    if surface:
        vertices=[];skirt=[];legs_tri=[]
        for obj,st,lt in parts:
            ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();n=len(vertices)
            vertices.extend(obj.matrix_world@v.co for v in mesh.vertices);ev.to_mesh_clear()
            skirt.extend(tuple(i+n for i in tri) for tri in st);legs_tri.extend(tuple(i+n for i in tri) for tri in lt)
        triangles=len(BVHTree.FromPolygons(vertices,skirt,all_triangles=True).overlap(BVHTree.FromPolygons(vertices,legs_tri,all_triangles=True)))
    return dict(maximum=maximum,hits=hits,worst=worst,segment_max=segment_max,triangles=triangles)

cases=[(x,z) for x,z in ((30,0),(-30,0),(0,30),(0,-30),(30,30),(30,-30),(-30,30),(-30,-30))]
if not args.full:cases=[(30,0),(-30,0),(0,30),(0,-30),(30,30),(-30,-30)]
results=[]
for follow in ((0.,.2) if args.full else (0.,)):
    settings.follow_groups['Skirt'].influence=follow
    for side in (range(2) if args.full else (0,)):
        for x,z in cases:
            set_simulation(False)
            for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
            bpy.context.view_layer.update();set_simulation(True)
            for _ in range(20):background_step(1/60)
            bpy.context.view_layer.update();rest=contacts(True)
            result=dict(follow=follow,side=side,x=x,z=z,rest=rest,maximum=0.,segment_max=0.,hit_frames=0,surfaces=[])
            for frame in range(91):
                t=frame/90;weight=math.sin(math.pi*t)**2
                pb=rig.pose.bones[legs[side]];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion((1,0,0),math.radians(x)*weight)@Quaternion((0,0,1),math.radians(z)*weight)
                bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
                value=contacts(frame in (22,45,68,90))
                if frame==45 and side==0 and follow==0 and x==-30 and z==-30:
                    (out/'sample_pose.json').write_text(json.dumps({p.name:[list(row) for row in p.matrix_basis] for p in rig.pose.bones}))
                if value['maximum']>result['maximum']:result.update(maximum=value['maximum'],worst=value['worst'],frame=frame)
                result['segment_max']=max(result['segment_max'],value['segment_max'])
                if value['hits']:result['hit_frames']+=1
                if value['triangles'] is not None:result['surfaces'].append(value['triangles'])
            results.append(result)
            print('CASE',args.variant,follow,side,x,z,json.dumps(result),flush=True)
    (out/'results.json').write_text(json.dumps(dict(variant=args.variant,full_capsules=full,cases=results),indent=2))
if args.variant=='generated':
    assert len(results)==(32 if args.full else 6)
    assert all(c['rest']['maximum']<height*1e-6 and c['segment_max']<height*1e-6 for c in results), results
set_simulation(False)
for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=str(out/'candidate.blend'))
print('SWEEP_COMPLETE',args.variant,flush=True)
