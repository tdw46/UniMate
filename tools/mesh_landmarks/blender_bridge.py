"""Main-thread Blender boundary. Prediction never sees original rig data."""
import hashlib
import json
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector


def geometry(meshes):
    vertices=[];faces=[]
    for obj in meshes:
        if obj.type!='MESH':raise ValueError('Select character meshes')
        offset=len(vertices)
        # Base geometry, not evaluated armature deformation, weights or names.
        vertices.extend(tuple(obj.matrix_world@v.co) for v in obj.data.vertices)
        obj.data.calc_loop_triangles()
        faces.extend(tuple(offset+i for i in t.vertices) for t in obj.data.loop_triangles)
    v=np.asarray(vertices,dtype=np.float32);f=np.asarray(faces,dtype=np.int32)
    if len(v)<100 or not len(f) or not np.isfinite(v).all():raise ValueError('Select a complete character mesh')
    return v,f


def write_input(meshes,path):
    v,f=geometry(meshes);np.savez(path,vertices=v,faces=f)
    return geometry_digest(v,f)


def geometry_digest(v,f):
    return hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest()


def refine_prediction(meshes, prediction):
    from .hand_refinement import refine
    vertices, faces = geometry(meshes)
    signature = geometry_digest(vertices, faces)
    if (prediction.get('hand_refinement_geometry_sha256') == signature
            and prediction.get('hand_refinement',{}).get('revision') == 4):
        return prediction
    result = refine(vertices, prediction)
    from .finger_surface import fit_digits
    audit = result['hand_refinement']
    audit['surface_fit'] = fit_digits(vertices, faces, result['joints'])
    from .arm_surface import fit_arm_pivots
    audit['arm_pivots'] = fit_arm_pivots(vertices, faces, result['joints'])
    audit['revision'] = 4
    result['hand_refinement_geometry_sha256'] = signature
    return result


def specifications(prediction):
    """Map the model's fixed humanoid topology; no source-rig fallback."""
    if prediction.get('source')!='mesh_geometry_only':raise ValueError('Expected mesh-only joint predictions')
    j=prediction['joints'];specs=[]
    mapping={'Hips':'Hips','Spine':'Spine','Spine1':'UpperSpine','Spine2':'Chest','Neck':'Neck','Head':'Head'}
    parents={'Hips':'Root','Spine':'Hips','UpperSpine':'Spine','Chest':'UpperSpine','Neck':'Chest','Head':'Neck'}
    for side,suffix in (('Left','.L'),('Right','.R')):
        for chain,start in (([('Shoulder','Clavicle'),('Arm','UpperArm'),('ForeArm','Forearm'),('Hand','Hand')],'Chest'),([('UpLeg','Thigh'),('Leg','Shin'),('Foot','Foot'),('ToeBase','Toe')],'Hips')):
            parent=start
            for raw,name in chain:
                name+=suffix;mapping[side+raw]=name;parents[name]=parent;parent=name
        for finger,name in [('Thumb','Thumb'),('Index','Index'),('Middle','Middle'),('Ring','Ring'),('Pinky','Little')]:
            parent='Hand'+suffix
            for n in range(1,4):
                mapped=f'{name}{n}{suffix}';mapping[f'{side}Hand{finger}{n}']=mapped;parents[mapped]=parent;parent=mapped
    if set(j)!=set(mapping):raise ValueError('Incomplete humanoid prediction')
    allpoints=np.array([p for item in j.values() for p in (item['head'],item['tail'])])
    if not np.isfinite(allpoints).all():raise ValueError('Invalid predicted joints')
    h=float(np.ptp(allpoints[:,2]));root=Vector(j['Hips']['head']);root.z=float(allpoints[:,2].min())
    specs.append(('Root',root,root+Vector((0,0,h*.07)),None))
    for raw,name in mapping.items():
        a=Vector(j[raw]['head']);b=Vector(j[raw]['tail'])
        if (b-a).length<h*1e-5:raise ValueError('Predicted joint is too short: '+name)
        specs.append((name,a,b,parents[name]))
    return specs


def build(context,meshes,prediction,builder,*,prefix='Character',secondary=True,finalize=None,binding=None,symmetrize=True,humanoid_only=True):
    import copy
    from avatar_mtoon_groups import rig_material
    prediction=refine_prediction(meshes,prediction)
    if symmetrize:
        from .rig_frames import symmetric_prediction
        prediction = symmetric_prediction(prediction)
    local_prediction=copy.deepcopy(prediction)
    v,_=geometry(meshes)
    origin=Vector(prediction['joints']['Hips']['head']);origin.z=float(v[:,2].min())
    # All body fitters work in a geometry-derived local frame, never rig space.
    for item in local_prediction['joints'].values():
        for key in ('head','tail'):item[key]=list(Vector(item[key])-origin)
    specs=specifications(local_prediction)
    original=context.window.scene;active=context.view_layer.objects.active
    selected=list(context.selected_objects)
    collections=('objects','meshes','armatures','materials','scenes','collections','texts','lights','cameras','worlds')
    before={name:set(getattr(bpy.data,name)) for name in collections}
    try:
        scene=bpy.data.scenes.new(prefix+' Auto Rig');scene.world=original.world.copy() if original.world else None
        context.window.scene=scene
        # Preserve background/lighting for comparison, while leaving all source data intact.
        for obj in original.objects:
            if obj.type in {'LIGHT','CAMERA'}:
                dup=obj.copy();dup.data=obj.data.copy();dup.parent=None;dup.matrix_world=obj.matrix_world.copy();dup.animation_data_clear()
                for constraint in list(dup.constraints):dup.constraints.remove(constraint)
                scene.collection.objects.link(dup)
                if obj==original.camera:scene.camera=dup
        copies=[];materials={}
        for obj in meshes:
            dup=obj.copy();dup.data=obj.data.copy();dup.animation_data_clear();dup.parent=None
            dup.matrix_parent_inverse=Matrix.Identity(4)
            dup.data.transform(Matrix.Translation(-origin)@obj.matrix_world,shape_keys=True);dup.matrix_world=Matrix.Identity(4)
            if dup.data.shape_keys:dup.data.shape_keys.animation_data_clear()
            for c in list(dup.constraints):dup.constraints.remove(c)
            for mod in list(dup.modifiers):dup.modifiers.remove(mod)
            dup.vertex_groups.clear()
            for key in list(dup.keys()):del dup[key]
            for slot in dup.material_slots:
                if slot.material:
                    mat=slot.material
                    if mat not in materials:
                        materials[mat] = rig_material(mat, prefix)
                    if materials[mat] != mat:
                        slot.material = materials[mat]
            scene.collection.objects.link(dup);copies.append(dup)
        rig,report=builder.rebuild(None,copies,specs=specs,symmetric_hands=symmetrize,correct_finger_geometry=True,symmetric_secondary=symmetrize and not humanoid_only,secondary=secondary,binding=binding)
        report['symmetry'] = dict(enabled=symmetrize, humanoid_only=humanoid_only)
        rig.matrix_world=Matrix.Translation(origin)
        rig.name=prefix+' Auto Rig';report['landmarks']=prediction
        if finalize is not None:finalize(rig,copies,report)
        return rig,copies,report
    except BaseException:
        if context.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        context.window.scene=original
        for name in collections:
            for item in set(getattr(bpy.data,name))-before[name]:
                if name=='objects':getattr(bpy.data,name).remove(item,do_unlink=True)
                else:
                    if hasattr(item,'use_fake_user'):item.use_fake_user=False
                    if name=='scenes' or item.users==0:getattr(bpy.data,name).remove(item)
        raise
    finally:
        for obj in original.view_layers[0].objects:
            obj.select_set(obj in selected,view_layer=original.view_layers[0])
        original.view_layers[0].objects.active=active
