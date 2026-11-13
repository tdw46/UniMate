"""Bounded default facial seed assignment and minimum radii.

Nearest eligible endpoint rule is recovered from 0x90740. Mouth rings,
nondefault role tuning and face isolation are not implemented here.
"""
import numpy as np
F=np.float32

def endpoint_modes(face,heads,tails,base_modes,vertices=None):
 h=np.asarray(heads,'f4');t=np.asarray(tails,'f4');modes=list(base_modes)
 def distance(point):
  if vertices is None:raise ValueError('forehead endpoint selection requires mesh geometry')
  best=F(np.inf)
  for a in range(0,len(vertices),65536):
   d=np.asarray(vertices[a:a+65536],'f4')-point
   if len(d):best=min(best,F(np.min((d[:,0]*d[:,0]+d[:,1]*d[:,1])+d[:,2]*d[:,2])))
  return F(np.sqrt(best))
 for i,role,side in face['roles']:
  if role=='forehead':
   d=t[i]-h[i];length=F(np.sqrt(F(F(d[0]*d[0]+d[1]*d[1])+d[2]*d[2])))
   modes[i]=16 if distance(h[i])>F(F(length*F(.1))+F(distance(t[i])*F(3))) else 15
  elif role=='chin':modes[i]=16
  elif role in ('brow','cheek','lip_top','lip_bot','lip_corner'):modes[i]=15
  elif role.startswith('eyelid_'):modes[i]=15 if side=='x' or i in face.get('head_lids',()) else 16
  elif role=='jaw':modes[i]=18
 return modes

def sphere_radii(face,heads,tails,base_modes,scale,vertices=None):
 # Native unpaired corner fallback at 0x12ab0c uses ORIGINAL-space nearest
 # facial endpoint distance, then .3 and normalization scale in that order.
 if not any(role=='chin' for _,role,_ in face['roles']):return np.zeros(len(base_modes),'f4')
 h=np.asarray(heads,'f4');t=np.asarray(tails,'f4')
 modes=endpoint_modes(face,h,t,base_modes,vertices)
 # Corner radii are prepared before custom eyelid shapes replace endpoint
 # modes. Native complete eyelid groups use their original head endpoints here.
 for i,role,_ in face['roles']:
  if role.startswith('eyelid_') and face['shapes'][i]['kind']!='none':modes[i]=15
 points={i:h[i] if m==15 else t[i] for i,m in enumerate(modes) if m in (15,16)}
 radii=np.zeros(len(modes),'f4')
 for i,role,_ in face['roles']:
  if role!='chin':continue
  distances=[]
  for j,point in points.items():
   if i==j:continue
   d=points[i]-point;distances.append(F(np.sqrt(F(F(d[0]*d[0]+d[1]*d[1])+d[2]*d[2]))))
  nearest=min(distances,default=F(np.inf))
  if np.isfinite(nearest) and nearest>0:radii[i]=F(F(nearest*F(.3))*F(scale))
 return radii

def assign(face,heads,tails,base_modes,scale,vertices=None,*,prepared_endpoints=None):
 h=np.asarray(heads,'f4')*F(scale);t=np.asarray(tails,'f4')*F(scale);modes=endpoint_modes(face,heads,tails,base_modes,vertices)
 if prepared_endpoints is not None:
  h,t=prepared_endpoints
 ids=[i for i,m in enumerate(modes) if m in (15,16)]
 radii=np.zeros(len(modes),'f4');points={i:h[i] if modes[i]==15 else t[i] for i in ids}
 for i in ids:
  other=[j for j in ids if j!=i]
  if not other:continue
  delta=[points[i]-points[j] for j in other]
  distances=[F(np.sqrt(F(F(d[0]*d[0]+d[1]*d[1])+d[2]*d[2]))) for d in delta]
  radius=min((value for value in distances if np.isfinite(value)),default=F(0))
  radii[i]=F(radius*F(.6))
 for i,s in enumerate(face['shapes']):
  if s['kind']!='none':modes[i]=17
 return modes,radii
