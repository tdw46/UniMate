"""Bounded mesh-aware chain-end preparation from sub_d0b6c/0x149f88."""
import ctypes
import numpy as np
F=np.float32
_lib=ctypes.CDLL(None)
for _name in ('sinf','cosf'):
 _f=getattr(_lib,_name);_f.argtypes=(ctypes.c_float,);_f.restype=ctypes.c_float

def exit_distance(bvh,vertices,triangles,origin,direction,length):
 distance=F(0);inside=1;epsilon=F(max(length,F(1e-6))*F(1e-5))
 for _ in range(32):
  hit=bvh._raycast((origin+direction*distance).tolist(),direction.tolist())
  if hit is None:return length
  distance=F(distance+F(hit[0]))
  if not distance<length:return length
  a,b,c=vertices[triangles[hit[1]]];normal=np.cross(b-a,c-a)
  dot=F(F(F(direction[0]*normal[0])+F(direction[1]*normal[1]))+F(direction[2]*normal[2]))
  inside+=1 if dot<=0 else -1
  if inside<1:return F(distance*F(.9))
  distance=F(distance+epsilon)
 return length

def no_hit_tails(runtime,vertices,triangles,heads,tails,mask):
 """Legacy research helper name retained; now includes intersecting rays."""
 bvh=runtime.MeshBvh(vertices,triangles);out=tails.copy()
 for i in np.flatnonzero(mask):
  delta=tails[i]-heads[i];sq=F(F(F(delta[0]*delta[0])+F(delta[1]*delta[1]))+F(delta[2]*delta[2]));length=F(np.sqrt(sq))
  if sq<=0 or not np.isfinite(length):continue
  direction=delta*F(F(1)/length);origin=heads[i]
  if bvh._raycast(origin.tolist(),direction.tolist()) is None:
   out[i]=origin+direction*F(length*F(.25));continue
  distance=exit_distance(bvh,vertices,triangles,origin,direction,length)
  if distance<F(length*F(.7)):
   ref=np.array((1,0,0) if abs(direction[0])<F(.9) else (0,1,0),'f4')
   u=np.cross(direction,ref);sq=F(F(u[2]*u[2])+F(F(u[0]*u[0])+F(u[1]*u[1])));norm=F(np.sqrt(sq))
   if sq>0 and np.isfinite(norm):u=u*F(F(1)/norm)
   else:u=np.array((0,1,0),'f4')
   w=np.cross(direction,u);chosen=direction.copy();best=distance
   for angle in (15,30,45,60):
    theta=F(F(angle)*F(.017453292));sn=F(_lib.sinf(theta));cs=F(_lib.cosf(theta));bias=F(F(cs*F(.5))+F(.5))
    for az in range(8):
     phi=F(F(F(az)*F(6.2831855))*F(.125));s=F(_lib.sinf(phi));c=F(_lib.cosf(phi))
     candidate=direction*cs+(u*F(sn*c)+w*F(sn*s))
     score=F(exit_distance(bvh,vertices,triangles,origin,candidate,length)*bias)
     if score>F(best*F(1.0001)):best=score;chosen=candidate
   direction=chosen;distance=exit_distance(bvh,vertices,triangles,origin,direction,length)
  out[i]=origin+direction*distance
 return out
