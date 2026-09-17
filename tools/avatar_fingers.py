"""VRM-guided fresh finger chains, spread binding and mesh/rest-pose baking.

Coordinates are canonical rig-local (X lateral, Z up). Source metadata supplies
joint locations only; source rigs and weights are deleted before the heat bind.
"""
import math
import bpy
import numpy as np
from mathutils import Matrix,Quaternion,Vector

DIGITS=('Thumb','Index','Middle','Ring','Little')


def source_finger_aliases(bones):
    result={}
    for side,suffix in [('left','L'),('right','R')]:
        for digit in DIGITS:
            parts=('Metacarpal','Proximal','Distal') if digit=='Thumb' and side+'ThumbMetacarpal' in bones else ('Proximal','Intermediate','Distal')
            for i,part in enumerate(parts,1):
                role=side+digit+part
                if role not in bones:raise ValueError('Finger landmark missing: '+role)
                result[role]=f'{digit}{i}.{suffix}'
    return result


def world_head(rig,name):return rig.matrix_world@rig.pose.bones[name].head


def hand_frame(rig,side,source=False):
    wrist=world_head(rig,('Wrist.' if source else 'Hand.')+side)
    forward=(world_head(rig,'Middle1.'+side)-wrist).normalized()
    across=world_head(rig,'Index1.'+side)-world_head(rig,'Little1.'+side)
    across=(across-forward*across.dot(forward)).normalized()
    normal=forward.cross(across).normalized()
    return wrist,forward,across,normal


def aim_pose(rig,name,direction,child=None):
    """Rotate a posed bone in armature space, retaining its current head."""
    pb=rig.pose.bones[name]
    # glTF bone display tails can be arbitrary; source chains use child heads.
    current=(rig.pose.bones[child].head-pb.head) if child else (pb.tail-pb.head)
    target=rig.matrix_world.to_3x3().inverted()@direction
    rotation=current.normalized().rotation_difference(target.normalized()).to_matrix().to_4x4()
    pb.matrix=Matrix.Translation(pb.head)@rotation@Matrix.Translation(-pb.head)@pb.matrix
    bpy.context.view_layer.update()


def spread_source_fingers(rig):
    audit=[]
    for side in ('L','R'):
        wrist,forward,across,normal=hand_frame(rig,side,True)
        before=[world_head(rig,f'{d}3.{side}') for d in DIGITS[1:]]
        # Anatomical fan angles in the measured palm plane, same on both sides.
        for digit,angle in zip(DIGITS,(55,21,7,-7,-21)):
            direction=forward*math.cos(math.radians(angle))+across*math.sin(math.radians(angle))
            aim_pose(rig,f'{digit}1.{side}',direction,f'{digit}2.{side}')
        after=[world_head(rig,f'{d}3.{side}') for d in DIGITS[1:]]
        width=(world_head(rig,'Index1.'+side)-world_head(rig,'Little1.'+side)).length
        gap=min((a-b).length for a,b in zip(after,after[1:]))
        assert gap>width*.25,'Source fingers failed to separate'
        audit.append({'side':side,'angles_degrees':dict(zip(DIGITS,(55,21,7,-7,-21))),
                      'distal_joint_gap_before':min((a-b).length for a,b in zip(before,before[1:])),
                      'distal_joint_gap_after':gap,'palm_width':width})
    return audit


def capture_finger_tips(rig,landmarks):
    for side in ('L','R'):
        for digit in DIGITS:
            name=f'{digit}3.{side}';pb=rig.pose.bones[name]
            # VRoid includes one terminal node for each distal joint. Refuse
            # ambiguous/missing nodes rather than using glTF's display tail.
            children=list(pb.children)
            if len(children)!=1:raise ValueError('Expected one terminal fingertip node: '+name)
            landmarks[name]['tail']=rig.matrix_world@children[0].head


def finger_specs(landmarks):
    result=[]
    for side in ('L','R'):
        for digit in DIGITS:
            for i in range(1,4):
                name=f'{digit}{i}.{side}';a=landmarks[name]['head']
                b=landmarks[f'{digit}{i+1}.{side}']['head'] if i<3 else landmarks[name]['tail']
                assert (b-a).length>1e-5
                result.append((name,a,b,f'{digit}{i-1}.{side}' if i>1 else 'Hand.'+side))
    return result


def apply_aligned_rest(meshes,rig):
    """Bake evaluated skin coordinates and posed bone matrices as one rest state."""
    thumb_before={s:[rig.data.bones['Hand.'+s].matrix_local.inverted()@rig.data.bones[f'Thumb{i}.{s}'].matrix_local for i in range(1,4)] for s in ('L','R')}
    for side,sign in [('L',1),('R',-1)]:
        direction=Vector((sign,0,0))
        for part in ('UpperArm','Forearm','Hand'):
            aim_pose(rig,part+'.'+side,direction)
        for digit in DIGITS[1:]:
            for i in range(1,4):aim_pose(rig,f'{digit}{i}.{side}',direction)
    bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
    records={};maximum_change=0.
    for obj in meshes:
        assert obj.data.shape_keys is None,'Rest bake requires baked shape keys'
        assert len(obj.modifiers)==1 and obj.modifiers[0].type=='ARMATURE','Unexpected modifier during rest bake'
        ev=obj.evaluated_get(deps);mesh=ev.to_mesh()
        coordinates=np.array([v.co[:] for v in mesh.vertices]);ev.to_mesh_clear()
        before=np.array([v.co[:] for v in obj.data.vertices])
        assert coordinates.shape==before.shape
        maximum_change=max(maximum_change,float(np.linalg.norm(coordinates-before,axis=1).max()))
        records[obj.name]=coordinates
    posed={pb.name:np.array(pb.matrix) for pb in rig.pose.bones}
    bpy.ops.object.select_all(action='DESELECT');rig.select_set(True);bpy.context.view_layer.objects.active=rig
    bpy.ops.object.mode_set(mode='POSE')
    assert bpy.ops.pose.armature_apply(selected=False)=={'FINISHED'}
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj in meshes:
        obj.data.vertices.foreach_set('co',records[obj.name].ravel());obj.data.update()
    bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
    maximum_error=0.
    for obj in meshes:
        ev=obj.evaluated_get(deps);mesh=ev.to_mesh()
        actual=np.array([v.co[:] for v in mesh.vertices]);ev.to_mesh_clear()
        maximum_error=max(maximum_error,float(np.linalg.norm(actual-records[obj.name],axis=1).max()))
    bone_error=max(float(np.abs(np.array(b.matrix_local)-posed[b.name]).max()) for b in rig.data.bones)
    thumb_error=max(float(np.abs(np.array(rig.data.bones['Hand.'+s].matrix_local.inverted()@rig.data.bones[f'Thumb{i}.{s}'].matrix_local)-np.array(thumb_before[s][i-1])).max()) for s in ('L','R') for i in range(1,4))
    alignment=min((rig.data.bones[f'{d}{i}.{s}'].tail_local-rig.data.bones[f'{d}{i}.{s}'].head_local).normalized().dot(Vector((sign,0,0))) for s,sign in [('L',1),('R',-1)] for d in DIGITS[1:] for i in range(1,4))
    assert maximum_error<1e-5 and bone_error<1e-5 and thumb_error<1e-5 and alignment>.99999
    assert maximum_change>.01
    return {'mesh_rest_vertices_updated':True,'maximum_rest_vertex_change':maximum_change,
            'maximum_evaluated_surface_jump':maximum_error,'maximum_rest_matrix_error':bone_error,
            'minimum_finger_alignment_dot':alignment,'thumb_relative_hand_rest_error':thumb_error,
            'thumb_alignment_changed':False,'passed':True}


def animate_hands(rig,frames=360):
    """Wrist flex/deviation, isolated digit curls, full fist, fan and opposition."""
    def turn(name,axis,degrees):
        basis=rig.data.bones[name].matrix_local.to_quaternion()
        return basis.inverted()@Quaternion(axis,math.radians(degrees))@basis
    for frame in range(frames):
        phase=frame//120;t=(frame%120)/119
        for pb in rig.pose.bones:pb.rotation_mode='QUATERNION';pb.matrix_basis=Matrix.Identity(4)
        for side,sign in [('L',1),('R',-1)]:
            forward=Vector((sign,0,0));dorsal=Vector((0,0,1));curl_axis=forward.cross(-dorsal)
            if phase==0:
                pulse=math.sin(2*math.pi*t)*math.sin(math.pi*t)
                rig.pose.bones['Hand.'+side].rotation_quaternion=turn('Hand.'+side,curl_axis,45*pulse)@turn('Hand.'+side,dorsal,sign*20*math.sin(4*math.pi*t)*math.sin(math.pi*t))
            if phase==1:
                digit_index=min(4,int(t*5));amount=math.sin(math.pi*((t*5)%1))**2
                digit=DIGITS[digit_index]
                for i,angle in enumerate((65,85,65),1):
                    name=f'{digit}{i}.{side}';rig.pose.bones[name].rotation_quaternion=turn(name,curl_axis,angle*amount)
            if phase==2:
                amount=math.sin(2*math.pi*t)**2
                if t<.5:
                    for digit in DIGITS[1:]:
                        for i,angle in enumerate((65,85,65),1):
                            name=f'{digit}{i}.{side}';rig.pose.bones[name].rotation_quaternion=turn(name,curl_axis,angle*amount)
                    rig.pose.bones['Thumb1.'+side].rotation_quaternion=turn('Thumb1.'+side,dorsal,sign*25*amount)@turn('Thumb1.'+side,curl_axis,25*amount)
                    for i in (2,3):rig.pose.bones[f'Thumb{i}.{side}'].rotation_quaternion=turn(f'Thumb{i}.{side}',curl_axis,40*amount)
                else:
                    for digit,angle in zip(DIGITS[1:],(-21,-7,7,21)):
                        name=f'{digit}1.{side}';rig.pose.bones[name].rotation_quaternion=turn(name,dorsal,sign*angle*amount)
                    rig.pose.bones['Thumb1.'+side].rotation_quaternion=turn('Thumb1.'+side,dorsal,-sign*18*amount)
        for pb in rig.pose.bones:pb.keyframe_insert('rotation_quaternion',frame=frame)
    rig.animation_data.action.name='evaluation'
