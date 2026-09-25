"""Headless, portable-VRM skirt fit: prepare, cache BVT poses, optimize, save.

Run with Blender --background --factory-startup --python this.py -- source out.
All expensive work is isolated from the user's interactive Blender process.
"""
import argparse,json,math,sys,time
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Quaternion,Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights
from avatar_physics_preview import set_simulation,background_step
from avatar_skirt_binding import rebind_skirt_strips,fit_rest_clearance
from avatar_contact_colliders import install_contact_colliders
from avatar_pose_contact import PoseContactFitter
from properties_hallway_rig import initialize
from avatar_skirt_fit_io import source_record,write_bundle


def parts_for(rig,meshes):
    human=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    legs={getattr(human,s+'_'+p).node.bone_name for s in ('left','right') for p in ('upper_leg','lower_leg')}
    parts=[]
    for obj in meshes:
        obj.data.calc_loop_triangles();ws=[weights(obj,v.index) for v in obj.data.vertices]
        skirt={i for i,w in enumerate(ws) if sum(v for n,v in w.items() if n.startswith('Secondary_Skirt_'))>.05}
        body={i for i,w in enumerate(ws) if i not in skirt and sum(v for n,v in w.items() if n in legs)>.4}
        st=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in skirt for i in t.vertices)]
        bt=[tuple(t.vertices) for t in obj.data.loop_triangles if all(i in body for i in t.vertices)]
        if st or bt:parts.append((obj,st,bt))
    return parts,sorted(legs)


def run(source,output,budget=.04,reuse=False,rig_name=None,rest_budget=.10):
    raise RuntimeError("Vertex-displacement fitting is disabled. Use fit_avatar_weights.py")
    if not bpy.app.background:raise RuntimeError('Pose fitting must run in an isolated background Blender process')
    if not math.isfinite(budget) or budget<=0:raise ValueError('Displacement budget must be positive and finite')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    def progress(stage,**data):
        value=dict(stage=stage,**data);(out/'progress.json').write_text(json.dumps(value));print(json.dumps(value),flush=True)
    for repo in getattr(bpy.context.preferences,'extensions',()).repos if hasattr(bpy.context.preferences,'extensions') else ():
        if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm' if hasattr(bpy.context.preferences,'extensions') else 'VRM_Addon_for_Blender')
    bpy.ops.wm.open_mainfile(filepath=str(Path(source).resolve()))
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite' if hasattr(bpy.context.preferences,'extensions') else 'beyond_vrm_extension_suite')
    candidates=[o for o in bpy.data.objects if o.type=='ARMATURE' and o.get('unimate_secondary_generator')
                and (rig_name is None or o.name==rig_name)]
    if len(candidates)!=1:raise ValueError('Select one generated rig with --rig')
    rig=candidates[0]
    bpy.context.window.scene=next(s for s in bpy.data.scenes if rig.name in s.objects)
    bpy.context.view_layer.objects.active=rig
    if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    set_simulation(False)
    keep={rig,*rig.children_recursive}
    for obj in list(bpy.data.objects):
        if obj not in keep:bpy.data.objects.remove(obj,do_unlink=True)
    meshes=[o for o in rig.children if o.type=='MESH']
    original=source_record(rig,meshes)
    expression_values={(o.name,k.name):k.value for o in meshes if o.data.shape_keys for k in o.data.shape_keys.key_blocks}
    for (name,key),value in expression_values.items():bpy.data.objects[name].data.shape_keys.key_blocks[key].value=0.
    rig.animation_data_clear()
    for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
    bpy.context.view_layer.update()
    settings=initialize(rig);follow=settings.follow_groups['Skirt'].influence
    parts,leg_names=parts_for(rig,meshes)
    if not any(st for _,st,_ in parts):raise ValueError('No generated skirt faces found')
    if not any(bt for _,_,bt in parts):raise ValueError('No leg-weighted body faces found')
    if not reuse:
        binding=rebind_skirt_strips(rig,meshes)
        rest_fit=fit_rest_clearance(rig,parts)
        colliders=install_contact_colliders(rig,meshes)
        settings.skirt_thickness=1.
        progress('prepared',binding=binding,rest_fit=rest_fit,colliders=colliders)
        (out/'source.json').write_text(json.dumps(original))
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'prepared.blend'))
    else:
        bpy.ops.wm.open_mainfile(filepath=str(out/'prepared.blend'))
        rig=bpy.data.objects[original['rig']]
        meshes=[o for o in rig.children if o.type=='MESH'];parts,leg_names=parts_for(rig,meshes);settings=initialize(rig)
    records=[];triangles=[];body_records=[];body_triangles=[]
    for obj,st,bt in parts:
        for faces,recs,ts in ((st,records,triangles),(bt,body_records,body_triangles)):
            ids=sorted({i for tri in faces for i in tri});lookup={v:len(recs)+j for j,v in enumerate(ids)}
            recs.extend((obj.name,i) for i in ids);ts.extend(tuple(lookup[i] for i in tri) for tri in faces)
    names=sorted({n for obj,i in records+body_records for n in weights(bpy.data.objects[obj],i)})
    w=np.array([[weights(bpy.data.objects[obj],i).get(n,0.) for n in names] for obj,i in records])
    body_w=np.array([[weights(bpy.data.objects[obj],i).get(n,0.) for n in names] for obj,i in body_records])
    body_rest=np.array([rig.matrix_world.inverted()@bpy.data.objects[obj].matrix_world@bpy.data.objects[obj].data.vertices[i].co for obj,i in body_records])
    inverse_rest=np.array([rig.data.bones[n].matrix_local.inverted() for n in names])
    rest=np.array([rig.matrix_world.inverted()@bpy.data.objects[obj].matrix_world@bpy.data.objects[obj].data.vertices[i].co for obj,i in records])
    weld_map={}
    for i,p in enumerate(rest):weld_map.setdefault(tuple(np.round(p,6)),[]).append(i)
    welds=[g for g in weld_map.values() if len(g)>1]
    if reuse:
        cache=np.load(out/'poses.npz')
        if 'body_matrices' not in cache:raise ValueError('Old pose cache lacks body skin matrices; rerun without --reuse')
        body_matrices=cache['body_matrices'];bases=cache['bases'];matrices=cache['matrices'];bodies=cache['bodies'];axes=cache['axes'];train=cache['train'];validation=cache['validation']
    else:
        bases=[];matrices=[];bodies=[];body_matrices=[];axes=[];train=[];validation=[]
        human=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
        upper=[human.left_upper_leg.node.bone_name,human.right_upper_leg.node.bone_name]
        motions=[]
        for f in (0.,.3):
            for side in range(2):
                for x,z,y in ((30,0,0),(-30,0,0),(0,30,0),(0,-30,0),(30,30,20),(-30,-30,-20)):
                    angles=[(0,0,0),(0,0,0)];angles[side]=(x,z,y)
                    motions.append((f,angles,90,False))
        for i in range(8):
            rng=np.random.default_rng(770+i)
            motions.append((follow,[tuple(rng.uniform(-28,28,3)) for _ in range(2)],48,True))
        original_location=rig.location.copy()
        def capture(held):
            coords={}
            for obj,_,_ in parts:
                evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
                coords[obj.name]=np.array([obj.matrix_world@v.co for v in mesh.vertices]);evaluated.to_mesh_clear()
            p=np.array([coords[obj][i] for obj,i in records])
            world=np.asarray(rig.matrix_world)
            transforms=np.einsum('ij,bjk,bkl->bil',world,np.array([rig.pose.bones[n].matrix for n in names]),inverse_rest)
            blended=np.einsum('vb,bij->vij',w,transforms)
            predicted=np.einsum('vij,vj->vi',blended[:,:3,:3],rest)+blended[:,:3,3]
            error=float(np.max(np.linalg.norm(predicted-p,axis=1)))
            if error>2e-5:raise ValueError(f'Unsupported skin/modifier deformation; affine cache error {error}')
            (validation if held else train).append(len(bases));bases.append(p);matrices.append(blended[:,:3,:3])
            bodies.append(np.array([coords[obj][i] for obj,i in body_records]))
            body_matrices.append(np.einsum('vb,bij->vij',body_w,transforms)[:,:3,:3])
            axes.append(np.array([(rig.matrix_world@rig.pose.bones[n].head,rig.matrix_world@rig.pose.bones[n].tail) for n in leg_names]))
        for mi,(f,angles,frames,held) in enumerate(motions):
            set_simulation(False)
            for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
            rig.location=original_location;settings.follow_groups['Skirt'].influence=f
            bpy.context.view_layer.update();set_simulation(True)
            for _ in range(20):background_step(1/60)
            if mi==0:capture(False)
            for frame in range(frames+1):
                t=frame/frames;weight=math.sin(math.pi*t)**2
                for n,(x,z,y) in zip(upper,angles):
                    pb=rig.pose.bones[n];pb.rotation_mode='QUATERNION';pb.rotation_quaternion=Quaternion((1,0,0),math.radians(x)*weight)@Quaternion((0,0,1),math.radians(z)*weight)@Quaternion((0,1,0),math.radians(y)*weight)
                if held:rig.location=original_location+Vector((.06*math.sin(math.pi*t),.025*math.sin(2*math.pi*t),.015*weight))
                bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
                if frame in (round(frames*.25),round(frames*.5),round(frames*.75)):capture(held)
            progress('sampling',motion=mi+1,total=len(motions))
        set_simulation(False);rig.location=original_location
        for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
        settings.follow_groups['Skirt'].influence=follow;bpy.context.view_layer.update()
        bases=np.array(bases);matrices=np.array(matrices);bodies=np.array(bodies);axes=np.array(axes)
        body_matrices=np.array(body_matrices)
        np.savez_compressed(out/'poses.npz',bases=bases,matrices=matrices,bodies=bodies,body_matrices=body_matrices,axes=axes,train=train,validation=validation)
    fitter=PoseContactFitter(np.array(triangles),bodies,bases,matrices,np.array(body_triangles),axes,budget=budget,welds=welds)
    delta,report=fitter.fit(train,validation,progress=lambda r:progress('optimizing',**r))
    for (name,i),d in zip(records,delta):
        obj=bpy.data.objects[name];local=(obj.matrix_world.inverted()@rig.matrix_world).to_3x3()@Vector(d)
        old=obj.data.vertices[i].co.copy()
        if obj.data.shape_keys:
            values=[(k,k.data[i].co.copy()) for k in obj.data.shape_keys.key_blocks]
            for k,p in values:k.data[i].co=p+local
        obj.data.vertices[i].co=old+local
    for obj in meshes:obj.data.update()
    # Rest integrity is a hard gate. A low body-contact score alone can hide
    # cloth self-folding, unweighted waist caps and nearby layer intersections.
    from avatar_rest_layers import repair_rig_layers,tree
    report['raw_fit_training_after']=report['training_after']
    report['raw_fit_validation_after']=report['validation_after']
    report['rest_layers']=repair_rig_layers(rig,json.loads((out/'source.json').read_text()),budget=rest_budget)
    final_rest=np.array([rig.matrix_world.inverted()@bpy.data.objects[obj].matrix_world@bpy.data.objects[obj].data.vertices[i].co for obj,i in records])
    final_body=np.array([rig.matrix_world.inverted()@bpy.data.objects[obj].matrix_world@bpy.data.objects[obj].data.vertices[i].co for obj,i in body_records])
    final_bodies=bodies+np.einsum('fvij,vj->fvi',body_matrices,final_body-body_rest)
    fitter.bodies=[tree(body,np.array(body_triangles)) for body in final_bodies]
    report['training_after']=fitter.intersections(final_rest-rest,train)
    report['validation_after']=fitter.intersections(final_rest-rest,validation)
    if sum(report['training_after'])>=sum(report['training_before']) or sum(report['validation_after'])>sum(report['validation_before']):
        raise ValueError('Rest-safe fit regressed motion coverage; no installation bundle was produced')
    rig['hallway_pose_fit']=1
    report['raw_optimizer_displacement']=float(np.max(np.linalg.norm(delta,axis=1)))
    report['maximum_displacement']=report['rest_layers']['maximum_displacement']
    report['train_poses']=len(train);report['held_out_poses']=len(validation)
    report['runtime_solver']='BVT only; fitting adds no playback code'
    (out/'report.json').write_text(json.dumps(report,indent=2))
    write_bundle(rig,json.loads((out/'source.json').read_text()),report,out/'fit.json')
    np.save(out/'delta.npy',delta)
    for (name,key),value in expression_values.items():bpy.data.objects[name].data.shape_keys.key_blocks[key].value=value
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'fitted.blend'))
    progress('complete',training_before=sum(report['training_before']),training_after=sum(report['training_after']),
             validation_before=sum(report['validation_before']),validation_after=sum(report['validation_after']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--budget',type=float,default=.04);p.add_argument('--reuse',action='store_true');p.add_argument('--rig');p.add_argument('--rest-budget',type=float,default=.10);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    run(a.source,a.output,a.budget,a.reuse,a.rig,a.rest_budget)
