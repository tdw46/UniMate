"""Full-body source landmarks and a coordinated FK/analytic-leg-IK demo."""
import math
import bpy
from mathutils import Matrix,Vector,Quaternion
from avatar_fingers import aim_pose,DIGITS
FRAMES=720
PHASES=('HELLO / HEAD + NECK','TORSO / TWIST + REACH','LEGS / MARCH + ANKLES','FULL BODY / SQUAT + RISE','HANDS / CURL + SPREAD','FINALE / WAVE + WEIGHT SHIFT')


def lower_aliases():
 return {side+role:f'{alias}.{suffix}' for side,suffix in [('left','L'),('right','R')] for role,alias in [('UpperLeg','Thigh'),('LowerLeg','Shin'),('Foot','Foot'),('Toes','Toe')]}


def capture_toe_tips(rig,landmarks):
 for side in ('L','R'):
  pb=rig.pose.bones['Toe.'+side]
  if len(pb.children)!=1:raise ValueError('Expected a terminal toe node: '+side)
  landmarks['Toe.'+side]['tail']=rig.matrix_world@pb.children[0].head


def body_specs(landmarks):
 hip=landmarks['Hips']['head'];spine=landmarks['Abdomen']['head'];chest=landmarks['Torso']['head'];neck=landmarks['Neck']['head'];head=landmarks['Head']['head']
 root=Vector((hip.x,hip.y,0));size=(head-hip).length*.08
 result=[('Root',root,root+Vector((0,0,size)),None),('Hips',hip,spine,'Root'),('Spine',spine,chest,'Hips'),('Chest',chest,neck,'Spine'),('Neck',neck,head,'Chest'),('Head',head,landmarks['Head']['tail'],'Neck')]
 for side in ('L','R'):
  for name,next_name,parent in [('Thigh','Shin','Hips'),('Shin','Foot','Thigh.'+side),('Foot','Toe','Shin.'+side),('Toe',None,'Foot.'+side)]:
   a=landmarks[name+'.'+side]['head'];b=landmarks[next_name+'.'+side]['head'] if next_name else landmarks[name+'.'+side]['tail']
   result.append((name+'.'+side,a,b,parent))
 return result


def solve_leg(rig,side,target,foot_direction):
 thigh=rig.pose.bones['Thigh.'+side];hip=thigh.head.copy()
 a=rig.data.bones['Thigh.'+side].length;b=rig.data.bones['Shin.'+side].length
 delta=target-hip;distance=delta.length;axis=delta.normalized()
 assert distance<a+b+1e-4,(side,'unreachable ankle',distance,a+b)
 d=max(abs(a-b)+1e-6,min(distance,a+b-1e-7))
 along=(a*a-b*b+d*d)/(2*d)
 pole=Vector((0,-1,0));pole=(pole-axis*pole.dot(axis)).normalized()
 knee=hip+axis*along+pole*math.sqrt(max(0,a*a-along*along))
 aim_pose(rig,'Thigh.'+side,knee-hip)
 aim_pose(rig,'Shin.'+side,target-rig.pose.bones['Shin.'+side].head)
 aim_pose(rig,'Foot.'+side,foot_direction)
 return (rig.pose.bones['Foot.'+side].head-target).length


def animate_fullbody(rig,frames=FRAMES):
 h=rig.data.bones['Head'].tail_local.z
 ankles={s:rig.data.bones['Foot.'+s].head_local.copy() for s in ('L','R')}
 footdir={s:(rig.data.bones['Foot.'+s].tail_local-ankles[s]).normalized() for s in ('L','R')}
 def turn(name,axis,degrees):
  basis=rig.data.bones[name].matrix_local.to_quaternion()
  return basis.inverted()@Quaternion(axis,math.radians(degrees))@basis
 def rotate(name,axis,angle):rig.pose.bones[name].rotation_quaternion=turn(name,axis,angle)
 max_error=0.
 for frame in range(frames):
  phase=frame//120;t=(frame%120)/119;env=math.sin(math.pi*t)**2;wave=math.sin(math.tau*t)
  ramp=min(1.,frame/30,(frames-1-frame)/30);ramp=ramp*ramp*(3-2*ramp)
  for pb in rig.pose.bones:pb.rotation_mode='QUATERNION';pb.matrix_basis=Matrix.Identity(4)
  offset=Vector((0,0,-.015*h*ramp))
  if phase==2:offset.z-=.01*h*env*(.5+.5*math.cos(4*math.pi*t))
  if phase==3:offset+=Vector((0,.015*h*env,-.10*h*env))
  if phase==5:offset.x=.025*h*wave*env
  root=rig.pose.bones['Root'];root.location=rig.data.bones['Root'].matrix_local.to_quaternion().inverted()@offset
  rotate('Hips',(0,0,1),4*wave*env if phase in (2,5) else 0)
  rotate('Spine',(0,0,1),8*wave*env if phase in (1,5) else 0)
  chest=turn('Chest',(0,0,1),14*wave*env if phase==1 else 0)@turn('Chest',(1,0,0),-12*env if phase==3 else 0)
  rig.pose.bones['Chest'].rotation_quaternion=chest
  rotate('Neck',(0,1,0),9*wave*env if phase in (0,1,5) else 0)
  rig.pose.bones['Head'].rotation_quaternion=turn('Head',(0,0,1),28*wave*env if phase in (0,1) else 8*wave*env)@turn('Head',(1,0,0),8*math.sin(4*math.pi*t)*env)
  for side,sign in [('L',1),('R',-1)]:
   arm_down=72*ramp;reach=0.;elbow=8*ramp
   if phase==0 and side=='R':arm_down-=45*env;elbow+=75*env
   if phase==1:arm_down-=35*env;reach=35*env;elbow+=30*env
   if phase==2:reach=sign*20*math.sin(4*math.pi*t)*env;elbow+=12*env
   if phase==3:reach=75*env;elbow+=18*env
   if phase==4:arm_down-=40*env;reach=35*env;elbow+=40*env
   if phase==5:arm_down-=(45 if side=='R' else 18)*env;elbow+=(70 if side=='R' else 20)*env
   rig.pose.bones['UpperArm.'+side].rotation_quaternion=turn('UpperArm.'+side,(0,1,0),sign*arm_down)@turn('UpperArm.'+side,(0,0,1),-sign*reach)
   rotate('Clavicle.'+side,(0,1,0),-sign*4*env if phase in (0,4,5) else 0)
   rotate('Forearm.'+side,(0,0,1),-sign*elbow)
   curl_axis=Vector((0,sign,0))
   wrist=25*math.sin(6*math.pi*t)*env if phase in (0,5) and side=='R' else 12*wave*env if phase==4 else 0
   rotate('Hand.'+side,(0,0,1),sign*wrist)
   for digit in DIGITS:
    amount=.10*ramp
    if phase==4:amount=env*(.5+.5*math.sin(4*math.pi*t))
    elif phase==2:amount=.25*env
    elif phase in (0,5) and side=='R':amount=.05*ramp
    for i,angle in enumerate((50,70,45) if digit!='Thumb' else (15,25,30),1):rotate(f'{digit}{i}.{side}',curl_axis,angle*amount)
    if phase==4 and digit!='Thumb':
     spread=(DIGITS.index(digit)-2.5)*8*env*(.5-.5*math.sin(4*math.pi*t))
     pb=rig.pose.bones[f'{digit}1.{side}'];pb.rotation_quaternion=turn(pb.name,(0,0,1),sign*spread)@pb.rotation_quaternion
   bpy.context.view_layer.update()
   if ramp>0:
    target=ankles[side].copy();pitch=0.
    if phase==2:
     cycle=math.sin(4*math.pi*t+(0 if side=='L' else math.pi))
     target.y-=.065*h*cycle*env;target.z+=.065*h*max(0,cycle)*env;pitch=-12*max(0,cycle)*env
    if phase==5 and side=='L':target.z+=.025*h*env
    direction=Quaternion((1,0,0),math.radians(pitch))@footdir[side]
    max_error=max(max_error,solve_leg(rig,side,target,direction))
    rotate('Toe.'+side,(1,0,0),-8*max(0,cycle)*env if phase==2 else 0)
  for pb in rig.pose.bones:
   pb.keyframe_insert('rotation_quaternion',frame=frame)
   if pb.name=='Root':pb.keyframe_insert('location',frame=frame)
 rig.animation_data.action.name='evaluation'
 return {'frames':frames,'phases':PHASES,'leg_solver':'analytic two-segment IK baked to deform FK bones','maximum_ankle_target_error':max_error}
