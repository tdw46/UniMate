"""Fit an upright, roughly symmetric cropped bust from surface cross-sections.

No pretrained model or imported weights. Input must face Blender -Y with Z up
(as produced by the standard glTF importer). This fitter targets lowered arms
and cropped forearms, not arbitrary full-body poses.
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Quaternion
from mathutils.kdtree import KDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights, assign
from avatar_voxel_seams import correct_skin_seams
from avatar_atlas_landmark import center_atlas


def repaired_heat(meshes, rig):
    """Heat-only retry on a welded/capped copy; never voxelize the body."""
    height=max(v.co.z for o in meshes for v in o.data.vertices)-min(v.co.z for o in meshes for v in o.data.vertices)
    scale=10/height
    heat_rig=bpy.data.objects.new('Temporary heat skeleton',rig.data.copy())
    heat_rig.data.transform(Matrix.Scale(scale,4));bpy.context.scene.collection.objects.link(heat_rig)
    audit=[]
    try:
        for obj in meshes:
            data=obj.data.copy();bm=bmesh.new();bm.from_mesh(data)
            bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=height*1e-6)
            bmesh.ops.dissolve_degenerate(bm,dist=height*1e-8,edges=list(bm.edges))
            holes=bmesh.ops.holes_fill(bm,edges=[e for e in bm.edges if e.is_boundary],sides=0)
            filled=len(holes['faces']);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
            bm.to_mesh(data);bm.free();data.transform(Matrix.Scale(scale,4))
            temp=bpy.data.objects.new('Temporary closed heat mesh',data);bpy.context.scene.collection.objects.link(temp)
            try:
                bpy.ops.object.select_all(action='DESELECT');temp.select_set(True);heat_rig.select_set(True);bpy.context.view_layer.objects.active=heat_rig
                assert bpy.ops.object.parent_set(type='ARMATURE_AUTO')=={'FINISHED'}
                missing=sum(not weights(temp,v.index) for v in data.vertices)
                print('REPAIRED_HEAT',len(data.vertices),filled,missing,flush=True)
                assert missing==0,'Heat also failed on the welded/capped copy'
                lookup=KDTree(len(data.vertices))
                for v in data.vertices:lookup.insert(v.co,v.index)
                lookup.balance();obj.vertex_groups.clear()
                groups={g.name:obj.vertex_groups.new(name=g.name) for g in temp.vertex_groups}
                maximum=0.
                for v in obj.data.vertices:
                    _,i,d=lookup.find(v.co*scale);maximum=max(maximum,d/scale)
                    assert d/scale<height*2e-6,'Heat transfer has no coincident source vertex'
                    for n,w in weights(temp,i).items():groups[n].add([v.index],w,'REPLACE')
                audit.append({'mesh':obj.name,'heat_vertices':len(data.vertices),'temporary_cap_faces':filled,
                              'heat_unweighted_vertices':missing,'maximum_correspondence_distance':maximum,
                              'solver_scale':scale,'voxelized':False})
            finally:
                bpy.data.objects.remove(temp,do_unlink=True)
                if data.users==0:bpy.data.meshes.remove(data)
    finally:
        data=heat_rig.data;bpy.data.objects.remove(heat_rig,do_unlink=True)
        if data.users==0:bpy.data.armatures.remove(data)
    return audit


def fit_landmarks(points,atlas_audit=None):
    low,high=points.min(axis=0),points.max(axis=0);height=high[2]-low[2];cx=(low[0]+high[0])*.5
    def band(z):return points[abs(points[:,2]-z)<height*.004]
    profiles=[]
    for q in np.linspace(.56,.77,85):
        z=low[2]+height*q;p=band(z)
        if len(p):profiles.append((z,float(np.quantile(abs(p[:,0]-cx),.95))))
    mid=min(range(len(profiles)),key=lambda i:profiles[i][1]);narrow=profiles[mid][1]
    lower=next((i for i in range(mid,-1,-1) if profiles[i][1]>narrow*1.5),0)
    upper=next((i for i in range(mid,len(profiles)) if profiles[i][1]>narrow*1.35),len(profiles)-1)
    neck_z,head_z=profiles[lower][0],profiles[upper][0]
    def central(z):
        p=band(z);p=p[abs(p[:,0]-cx)<narrow*.65]
        return Vector((cx,float((np.quantile(p[:,1],.05)+np.quantile(p[:,1],.95))*.5),z))
    root=central(low[2]+height*.015);chest=central(low[2]+(neck_z-low[2])*.63)
    neck,head=central(neck_z),central(head_z)
    head,correction=center_atlas(points,neck,head)
    if atlas_audit is not None:atlas_audit.update(correction)
    specs=[('Root',root,root.lerp(chest,.3),None),('Spine',root.lerp(chest,.3),chest,'Root'),
           ('Chest',chest,neck,'Spine'),('Neck',neck,head,'Chest'),
           ('Head',head,central(high[2]-height*.045),'Neck')]
    shoulder_z=neck_z-height*.07
    shoulder_band=band(shoulder_z)
    for side,sign in [('L',1),('R',-1)]:
        # Trace disjoint forearm cross-sections, then fit their medial line.
        samples=[]
        for z in np.linspace(low[2]+height*.018,shoulder_z-height*.035,65):
            p=band(z);p=p[sign*(p[:,0]-cx)>narrow*2.1]
            if len(p)<15:continue
            xs=np.sort(sign*(p[:,0]-cx));gaps=np.where(np.diff(xs)>height*.006)[0]
            if len(gaps):p=p[sign*(p[:,0]-cx)>xs[gaps[-1]]+height*.003]
            else:
                # At merged torso/arm heights use the lateral tube's width
                # inferred from the separate forearm, excluding breast volume.
                if samples:
                    width=np.median([x[3] for x in samples])
                    p=p[sign*(p[:,0]-cx)>np.quantile(sign*(p[:,0]-cx),.98)-width]
            if len(p)<10:continue
            x=(np.quantile(p[:,0],.03)+np.quantile(p[:,0],.97))*.5
            y=(np.quantile(p[:,1],.03)+np.quantile(p[:,1],.97))*.5
            width=float(np.quantile(p[:,0],.97)-np.quantile(p[:,0],.03))
            samples.append((float(x),float(y),float(z),width))
        s=np.array(samples);assert len(s)>15,'Could not identify both lowered arm tubes'
        # The elbow is the change in direction of the arm centerline. Keep it
        # inside the central portion of the available shoulder-to-cut span.
        candidates=[]
        for k in range(6,len(s)-6):
            fraction=(shoulder_z-s[k,2])/(shoulder_z-low[2])
            if not .4<fraction<.75:continue
            error=0.
            for part in (s[:k+1],s[k:]):
                A=np.column_stack((part[:,2],np.ones(len(part))))
                coef=np.linalg.lstsq(A,part[:,:2],rcond=None)[0]
                error+=np.sum((A@coef-part[:,:2])**2)
            candidates.append((error,k))
        k=min(candidates)[1] if candidates else len(s)//2
        elbow=Vector(s[k,:3]);end=Vector(s[0,:3])
        p=shoulder_band[sign*(shoulder_band[:,0]-cx)>narrow*2.]
        arm_width=float(np.median(s[-8:,3]));outer=float(np.quantile(sign*(p[:,0]-cx),.97))
        x=cx+sign*(outer-arm_width*.5)
        near=p[abs(p[:,0]-x)<arm_width*.3]
        y=float((np.quantile(near[:,1],.05)+np.quantile(near[:,1],.95))*.5)
        shoulder=Vector((x,y,shoulder_z));clavicle=neck.lerp(shoulder,.2)
        specs += [('Clavicle.'+side,clavicle,shoulder,'Chest'),('UpperArm.'+side,shoulder,elbow,'Clavicle.'+side),
                  ('Forearm.'+side,elbow,end,'UpperArm.'+side)]
    return specs


def main():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    opts=parser.parse_args(args);source=Path(opts.input).resolve();out=Path(opts.output).resolve();out.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True);bpy.ops.import_scene.gltf(filepath=str(source))
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    for o in meshes:
        o.data.transform(o.matrix_world);o.matrix_world=Matrix.Identity(4);o.parent=None;o.vertex_groups.clear();o.modifiers.clear()
        o['binding_role']='body';o['binding_surface']='skin'
    for o in list(bpy.context.scene.objects):
        if o.type=='ARMATURE':bpy.data.objects.remove(o,do_unlink=True)
    points=np.array([v.co[:] for o in meshes for v in o.data.vertices]);atlas_audit={};specs=fit_landmarks(points,atlas_audit)
    rig=bpy.data.objects.new('Bust_Rig',bpy.data.armatures.new('Bust_Rig'));bpy.context.scene.collection.objects.link(rig)
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig;bpy.ops.object.mode_set(mode='EDIT')
    for name,a,b,parent in specs:
        bone=rig.data.edit_bones.new(name);bone.head=a;bone.tail=b
        if parent:bone.parent=rig.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT');rig.show_in_front=True
    for o in meshes:o.select_set(True)
    assert bpy.ops.object.parent_set(type='ARMATURE_AUTO')=={'FINISHED'}
    missing=sum(not weights(o,v.index) for o in meshes for v in o.data.vertices)
    print('HEAT_RESULT',missing,'unweighted vertices',flush=True)
    retry=repaired_heat(meshes,rig) if missing else []
    initial_missing=missing
    missing=sum(not weights(o,v.index) for o in meshes for v in o.data.vertices)
    assert missing==0, 'Automatic heat failed; do not substitute guessed weights'
    for o in meshes:
        bpy.ops.object.select_all(action='DESELECT');o.select_set(True)
        bpy.context.view_layer.objects.active=o
        bpy.ops.object.vertex_group_limit_total(limit=4);bpy.ops.object.vertex_group_normalize_all(lock_active=False)
    report={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'blender':bpy.app.version_string,'method':'geometric bust landmarks and ordinary Blender heat; no model inference',
            'atlas_correction':atlas_audit,'vertices':len(points),'triangles':sum(len(o.data.polygons) for o in meshes),'bones':len(specs),
            'unweighted_vertices':missing,'direct_heat_unweighted_vertices':initial_missing,'heat_retry':retry,
            'landmarks':{n:{'head':list(a),'tail':list(b),'parent':p} for n,a,b,p in specs}}
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'ordinary_heat.blend'))
    report['arm_boundary']={'applied':False,'method':'ordinary normalized heat; crease constraint disabled'}
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'heat_baseline.blend'))
    report['seams']=correct_skin_seams(meshes,rig,out/'local_seam_proxy.blend')
    scene=bpy.context.scene;scene.render.fps=30;scene.frame_start=0;scene.frame_end=239
    def turn(name,axis,degrees):
        basis=rig.data.bones[name].matrix_local.to_quaternion()
        return basis.inverted()@Quaternion(axis,math.radians(degrees))@basis
    for frame in range(240):
        phase=frame//80;t=(frame%80)/79;pulse=math.sin(2*math.pi*t)*math.sin(math.pi*t)
        for pb in rig.pose.bones:pb.rotation_mode='QUATERNION';pb.matrix_basis=Matrix.Identity(4)
        if phase==0:rig.pose.bones['Head'].rotation_quaternion=turn('Head',(0,0,1),40*pulse)@turn('Head',(1,0,0),15*math.sin(4*math.pi*t)*math.sin(math.pi*t))
        if phase==1:rig.pose.bones['Neck'].rotation_quaternion=turn('Neck',(0,1,0),18*pulse)@turn('Neck',(1,0,0),12*math.sin(4*math.pi*t)*math.sin(math.pi*t))
        if phase==2:
            amount=math.sin(math.pi*t)**2
            for side,sign in [('L',1),('R',-1)]:
                rig.pose.bones['UpperArm.'+side].rotation_quaternion=turn('UpperArm.'+side,(0,1,0),-sign*65*amount)
                rig.pose.bones['Forearm.'+side].rotation_quaternion=turn('Forearm.'+side,(1,0,0),-30*amount)
        for pb in rig.pose.bones:pb.keyframe_insert('rotation_quaternion',frame=frame)
    rig.animation_data.action.name='stitched-evaluation';scene.frame_set(0)
    bpy.ops.file.pack_all();scene['autorig_audit']=json.dumps(report)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'rigged.blend'))
    bpy.ops.export_scene.gltf(filepath=str(out/'rigged.glb'),export_format='GLB',export_animations=True,export_frame_range=True,export_force_sampling=True,export_skins=True)
    (out/'autorig_validation.json').write_text(json.dumps(report,indent=2));print('AUTORIG_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
