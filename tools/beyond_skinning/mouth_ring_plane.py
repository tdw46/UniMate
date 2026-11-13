"""Independent f64 two-axis power iteration used by mouth surface rings.

From 0x105fbc, with caller float32 points. This is a geometric stage, not the
complete ring-selection/weighting pass. No original module is loaded here.
"""
import math
import numpy as np

def plane(points, *, bounded=True):
 p=np.asarray(points,dtype='f4')
 if p.ndim!=2 or p.shape[1]!=3 or len(p)<1 or (bounded and (len(p)>4096 or not np.isfinite(p).all())):raise ValueError('invalid bounded ring points')
 n=len(p);center=[0.,0.,0.]
 for row in p:
  for k in range(3):center[k]+=float(row[k])/n
 a=b=c=d=e=f=0.
 for row in p:
  x,y,z=[float(row[k])-center[k] for k in range(3)]
  a+=x*x;b+=x*y;c+=x*z;d+=y*y;e+=y*z;f+=z*z
 x,y,z=0.0,0.0,0.238667185252719
 # Fixed native starts are populated from the image's read-only constants.
 x,y=INITIAL_A
 for _ in range(32):
  u=y*b+x*a+c*z;v=x*b+y*d+e*z;w=x*c+y*e+z*f
  norm=max(math.sqrt(u*u+v*v+w*w),1e-30);x,y,z=u/norm,v/norm,w/norm
 eigen=x*(y*b+x*a+c*z)+y*(x*b+y*d+e*z)+z*(x*c+y*e+z*f)
 px,py,pz=x*eigen,y*eigen,z*eigen
 u,v=INITIAL_B;w=.8081220356417687
 for _ in range(32):
  r=v*(b-px*y)+u*(a-px*x)+(c-px*z)*w
  s=u*(b-py*x)+v*(d-py*y)+(e-py*z)*w
  q=u*(c-x*pz)+v*(e-y*pz)+w*(f-pz*z)
  norm=max(math.sqrt(r*r+s*s+q*q),1e-30);u,v,w=r/norm,s/norm,q/norm
 return np.asarray([center,[x,y,z],[u,v,w]],'f8')

INITIAL_A=(0.79555728417573,0.556890098923011)
INITIAL_B=(0.3030457633656632,-0.5050762722761053)


def ring_order(points,axes=None):
 p=np.asarray(points,'f4');axes=plane(p) if axes is None else np.asarray(axes,'f8')
 if axes.shape!=(3,3) or not np.isfinite(axes).all():raise ValueError('invalid ring axes')
 def angle(i):
  x,y,z=[float(p[i,k])-axes[0,k] for k in range(3)]
  u,v=axes[1],axes[2]
  return math.atan2(x*v[0]+y*v[1]+z*v[2],x*u[0]+y*u[1]+z*u[2])
 return sorted(range(len(p)),key=angle)
