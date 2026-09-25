"""Pose-sampled skirt weight fitting with byte-identical final mesh geometry.

Blender --background --factory-startup --python this.py -- source.blend output --rig NAME
No vertex, face, UV or shape-key writes. BVT alone supplies sampled physics.
"""
import argparse,json,math,sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Quaternion,Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights,assign
from avatar_physics_preview import set_simulation,background_step
from avatar_mesh_invariant import snapshot,verify
from avatar_weight_contact import WeightContactFitter
from avatar_skirt_fit_io import source_record,write_bundle
from properties_hallway_rig import initialize
from fit_avatar_skirt import parts_for


def run(source,output,rig_name,reuse=False):
    raise RuntimeError('Retired: skirts must stay spring-bound. Use native spring/contact generation instead')
    if not bpy.app.background:raise RuntimeError('Fit in an isolated background process')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    def progress(stage,**data):
        r=dict(stage=stage,**data);(out/'progress.json').write_text(json.dumps(r));print(json.dumps(r),flush=True)
    for repo in getattr(bpy.context.preferences,'extensions',()).repos if hasattr(bpy.context.preferences,'extensions') else ():
        if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm' if hasattr(bpy.context.preferences,'extensions') else 'VRM_Addon_for_Blender')
    bpy.ops.wm.open_mainfile(filepath=str(out/'prepared.blend') if reuse else str(Path(source).resolve()))
    rig=bpy.data.objects[rig_name];bpy.context.window.scene=next(s for s in bpy.data.scenes if rig.name in s.objects)
    bpy.context.view_layer.objects.active=rig
    if rig.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.beyond_vrm_extension_suite' if hasattr(bpy.context.preferences,'extensions') else 'beyond_vrm_extension_suite')
    set_simulation(False);keep={rig,*rig.children_recursive}
    for o in list(bpy.data.objects):
        if o not in keep:bpy.data.objects.remove(o,do_unlink=True)
    meshes=[o for o in rig.children if o.type=='MESH'];geometry=snapshot(meshes)
    if not rig.get('hallway_immutable_mesh'):
        from avatar_regenerate_secondary import regenerate
        (out/'regeneration.json').write_text(json.dumps(regenerate(rig,meshes),indent=2))
    settings=initialize(rig);follow=settings.follow_groups['Skirt'].influence
    rig.animation_data_clear()
    for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
    expressions={(o.name,k.name):k.value for o in meshes if o.data.shape_keys for k in o.data.shape_keys.key_blocks}
    for (n,k),v in expressions.items():bpy.data.objects[n].data.shape_keys.key_blocks[k].value=0.
    bpy.context.view_layer.update()
    if not reuse:
        (out/'source.json').write_text(json.dumps(source_record(rig,meshes)))
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'prepared.blend'))
    parts,leg_names=parts_for(rig,meshes);records=[];faces=[];body_records=[];body_faces=[]
    for obj,st,bt in parts:
        for ts,recs,dest in ((st,records,faces),(bt,body_records,body_faces)):
            ids=sorted({i for t in ts for i in t});lookup={i:len(recs)+j for j,i in enumerate(ids)}
            recs.extend((obj.name,i) for i in ids);dest.extend(tuple(lookup[i] for i in t) for t in ts)
    if not records or not body_records:raise ValueError('Need generated skirt and leg-weighted body faces')
    hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    upper=[hum.left_upper_leg.node.bone_name,hum.right_upper_leg.node.bone_name]
    names=sorted({n for obj,i in records for n in weights(bpy.data.objects[obj],i)} | set(upper) |
                 {b.name for b in rig.data.bones if b.name.startswith('Secondary_Skirt_') and b.use_deform})
    initial=np.array([[weights(bpy.data.objects[o],i).get(n,0.) for n in names] for o,i in records])
    rest=np.array([rig.matrix_world.inverted()@bpy.data.objects[o].matrix_world@bpy.data.objects[o].data.vertices[i].co for o,i in records])
    allowed=np.zeros(initial.shape,dtype=bool);secondary=np.array([n.startswith('Secondary_Skirt_') for n in names]);legs=np.array([n in upper for n in names])
    for v,(o,i) in enumerate(records):
        families={n.rsplit('_',1)[0] for n,w in weights(bpy.data.objects[o],i).items() if n.startswith('Secondary_Skirt_') and w>1e-6}
        allowed[v]=[initial[v,j]>0 or n=='Hips' or n.rsplit('_',1)[0] in families or n in upper for j,n in enumerate(names)]
    weld={}
    for i,p in enumerate(rest):weld.setdefault(tuple(np.round(p,6)),[]).append(i)
    welds=[g for g in weld.values() if len(g)>1 and all(np.array_equal(allowed[g[0]],allowed[i]) for i in g)]
    if reuse:
        cache=np.load(out/'poses.npz');positions=cache['positions'];bodies=cache['bodies'];axes=cache['axes'];train=cache['train'];held=cache['held']
    else:
        positions=[];bodies=[];axes=[];train=[];held=[];motions=[]
        for f in (0.,.4):
            for side in range(2):
                for x,z,y in ((30,0,0),(-30,0,0),(0,30,0),(0,-30,0),(30,30,20),(-30,-30,-20)):
                    angles=[(0,0,0),(0,0,0)];angles[side]=(x,z,y);motions.append((f,angles,90,False))
        for i in range(8):
            rng=np.random.default_rng(770+i);motions.append((follow,[tuple(rng.uniform(-28,28,3)) for _ in range(2)],48,True))
        inverse=np.array([rig.data.bones[n].matrix_local.inverted() for n in names]);location=rig.location.copy()
        def capture(validation):
            transforms=np.einsum('ij,bjk,bkl->bil',np.asarray(rig.matrix_world),np.array([rig.pose.bones[n].matrix for n in names]),inverse)
            points=np.einsum('kij,vj->vki',transforms[:,:3,:3],rest)+transforms[None,:,:3,3]
            coords={}
            for obj,_,_ in parts:
                ev=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh();coords[obj.name]=np.array([obj.matrix_world@v.co for v in mesh.vertices]);ev.to_mesh_clear()
            observed=np.array([coords[o][i] for o,i in records]);predicted=np.einsum('vkj,vk->vj',points,initial)
            if np.max(np.linalg.norm(observed-predicted,axis=1))>2e-5:raise ValueError('Unsupported modifier deformation for cached skinning')
            (held if validation else train).append(len(positions));positions.append(points)
            bodies.append(np.array([coords[o][i] for o,i in body_records]))
            axes.append(np.array([(rig.matrix_world@rig.pose.bones[n].head,rig.matrix_world@rig.pose.bones[n].tail) for n in leg_names]))
        for mi,(f,angles,frames,validation) in enumerate(motions):
            set_simulation(False)
            for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
            rig.location=location;settings.follow_groups['Skirt'].influence=f;bpy.context.view_layer.update();set_simulation(True)
            for _ in range(20):background_step(1/60)
            if mi==0:capture(False)
            for frame in range(frames+1):
                t=frame/frames;w=math.sin(math.pi*t)**2
                for n,(x,z,y) in zip(upper,angles):
                    p=rig.pose.bones[n];p.rotation_mode='QUATERNION';p.rotation_quaternion=Quaternion((1,0,0),math.radians(x)*w)@Quaternion((0,0,1),math.radians(z)*w)@Quaternion((0,1,0),math.radians(y)*w)
                if validation:rig.location=location+Vector((.06*math.sin(math.pi*t),.025*math.sin(2*math.pi*t),.015*w))
                bpy.context.view_layer.update();background_step(1/60);bpy.context.view_layer.update()
                if frame in (round(frames*.25),round(frames*.5),round(frames*.75)):capture(validation)
            progress('sampling',motion=mi+1,total=len(motions))
        set_simulation(False);rig.location=location
        for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
        settings.follow_groups['Skirt'].influence=follow;bpy.context.view_layer.update()
        positions=np.array(positions);bodies=np.array(bodies);axes=np.array(axes)
        np.savez_compressed(out/'poses.npz',positions=positions,bodies=bodies,axes=axes,train=train,held=held)
    fitter=WeightContactFitter(faces,positions,bodies,body_faces,axes,initial,allowed,welds=welds,secondary=secondary,legs=legs)
    hip_width=abs(rig.data.bones[upper[0]].head_local.x-rig.data.bones[upper[1]].head_local.x)
    fitted,report=fitter.fit_attachment(train,held,rest,names,upper,hip_width,progress=lambda r:progress('optimizing',**r))
    if sum(report['validation_after'])>sum(report['validation_before']):raise ValueError('Held-out motion worsened; weights were not installed')
    for (o,i),row in zip(records,fitted):assign(bpy.data.objects[o],[i],{n:float(w) for n,w in zip(names,row) if w>1e-8})
    report.update(mesh_signatures=verify(meshes,geometry),train_poses=len(train),held_out_poses=len(held),maximum_direct_leg_weight=float(np.max(fitted[:,legs].sum(axis=1))),runtime='BVT only',vertex_displacement=0.,removed_faces=0)
    (out/'report.json').write_text(json.dumps(report,indent=2));write_bundle(rig,json.loads((out/'source.json').read_text()),report,out/'fit.json')
    for (n,k),v in expressions.items():bpy.data.objects[n].data.shape_keys.key_blocks[k].value=v
    verify(meshes,geometry);bpy.ops.wm.save_as_mainfile(filepath=str(out/'fitted.blend'))
    progress('complete',training_before=sum(report['training_before']),training_after=sum(report['training_after']),validation_before=sum(report['validation_before']),validation_after=sum(report['validation_after']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('--rig',required=True);p.add_argument('--reuse',action='store_true');a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);run(a.source,a.output,a.rig,a.reuse)
