"""Seed preview tubes/joints recovered from sub_9d854; research CPU output."""
import numpy as np
F=np.float32

def joint_coordinate(x):
 value=float(x)*100000.
 if np.isnan(value):return 0
 if value>=2**63:return 2**63-1
 if value<=-2**63:return -2**63
 return int(np.copysign(np.floor(abs(value)+.5),value))

def skeleton(vertices,heads,tails):
 v=np.asarray(vertices,'f4');h=np.asarray(heads,'f4');t=np.asarray(tails,'f4')
 d=v.max(axis=0)-v.min(axis=0);extent=F(np.sqrt(F(F(d[2]*d[2])+F(F(d[1]*d[1])+F(d[0]*d[0])))))
 radius=F(extent*F(.0016));verts=[];corners=[];starts=[]
 def poly(ids):starts.append(len(corners));corners.extend(ids)
 for head,tail in zip(h,t):
  direction=tail-head;length=F(np.sqrt(F(F(F(direction[0]*direction[0])+F(direction[1]*direction[1]))+F(direction[2]*direction[2]))))
  if length<F(1e-9):continue
  direction=direction/length;ref=np.array((0,0,1) if abs(direction[2])<F(.9) else (1,0,0),'f4')
  u=np.cross(direction,ref);norm=F(np.sqrt(F(F(u[2]*u[2])+F(F(u[0]*u[0])+F(u[1]*u[1])))));u=u/norm if norm>0 else u
  w=np.cross(direction,u);norm=F(np.sqrt(F(F(w[2]*w[2])+F(F(w[0]*w[0])+F(w[1]*w[1])))));w=w/norm if norm>0 else w
  offsets=[radius*(u+w*F(0)),radius*(u*F(.70710677)+w*F(.70710677)),radius*(w-u*F(4.371139e-8)),radius*(w*F(.70710677)-u*F(.70710677)),radius*(w*F(-8.742278e-8)-u),radius*(u*F(-.70710665)-w*F(.7071069)),radius*(u*F(1.1924881e-8)-w),radius*(u*F(.707107)-w*F(.70710653))]
  base=len(verts);verts.extend(head+x for x in offsets);verts.extend(tail+x for x in offsets)
  for i in range(8):j=(i+1)%8;poly([base+i,base+j,base+8+j,base+8+i])
 radius=F(extent*F(.0052));seen=set()
 triangles=[0,1,2,0,2,3,0,3,4,0,4,5,0,5,1,1,6,2,2,7,3,3,8,4,4,9,5,5,10,1,6,7,2,7,8,3,8,9,4,9,10,5,10,6,1,11,7,6,11,8,7,11,9,8,11,10,9,11,6,10]
 unit=np.array([(0,0,1),(.894,0,.447),(.276,.851,.447),(-.724,.526,.447),(-.724,-.526,.447),(.276,-.851,.447),(.724,.526,-.447),(-.276,.851,-.447),(-.894,0,-.447),(-.276,-.851,-.447),(.724,-.526,-.447),(0,0,-1)],'f4')
 for head,tail in zip(h,t):
  for point in (head,tail):
   key=tuple(joint_coordinate(x) for x in point)
   if key in seen:continue
   seen.add(key);base=len(verts);verts.extend(point+radius*unit)
   for i in range(0,60,3):poly([base+j for j in triangles[i:i+3]])
 return dict(verts=np.array(verts,'f4').reshape(-1,3),corners=np.array(corners,'i4'),loop_starts=np.array(starts,'i4'),colours=None)
