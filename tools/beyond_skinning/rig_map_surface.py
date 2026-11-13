"""Independent active surface-ring stages from 0x173140, 0xda348 and 0x14c900.

Surface endpoint ownership, ring selection, connected components and arc ordering
are implemented without importing or calling the original extension.
"""
import math
import numpy as np
from .rig_map_native import mean, squared, eligible
F=np.float32

def dot(a,b):return F(F(a[0]*b[0]+a[1]*b[1])+a[2]*b[2])
def norm(v):
 n=np.sqrt(squared(v),dtype='f4')
 return np.zeros(3,'f4') if n<=F(1e-12) else np.divide(v,n,dtype='f4')
def context_norm(v):
 n=np.sqrt(squared(v),dtype='f4')
 return np.divide(v,n,dtype='f4') if n>F(1e-12) else np.zeros(3,'f4')
def cross(a,b):return np.array([F(a[1]*b[2]-a[2]*b[1]),F(a[2]*b[0]-a[0]*b[2]),F(a[0]*b[1]-a[1]*b[0])],'f4')
def endpoint(segment,vertices):
 """Strict nearest endpoint competition; tail must beat half the head distance."""
 choices=[]
 for point in segment:
  best=F(np.inf);index=None
  for i,v in enumerate(vertices):
   distance=np.sqrt(squared(point-v),dtype='f4')
   if distance<best:best,index=distance,i
  if index is None:return None
  choices.append((index,point.copy(),best))
 return choices[1] if choices[1][2]<F(choices[0][2]*F(.5)) else choices[0]

def ring(records,forward=None,right=None):
 if len(records)<4:return None
 from .mouth_ring_plane import plane
 p=np.array([p for _,p in records],'f4')
 axes=plane(p,bounded=False);center=axes[0].astype('f4')
 n=norm(np.array(forward,'f4')) if forward is not None else np.zeros(3,'f4')
 if not np.any(n):n=norm(cross(axes[1].astype('f4'),axes[2].astype('f4')))
 u=norm(cross(n,np.array([1,0,0] if abs(n[0])<F(.9) else [0,1,0],'f4')));v=cross(n,u)
 uu=uv=vv=0.
 for point in p:
  a=float(dot(point-center,u));b=float(dot(point-center,v));uu+=a*a;uv+=a*b;vv+=b*b
 angle=.5*math.atan2(2.*uv,uu-vv)
 major=norm(u*F(math.cos(angle))+v*F(math.sin(angle)));minor=cross(n,major)
 if right is not None and dot(major,right)<0:major=-major;minor=-minor
 ordered=sorted(range(len(p)),key=lambda i:float(np.arctan2(dot(p[i]-center,minor),dot(p[i]-center,major),dtype='f4')))
 projected=[dot(p[i]-center,major) for i in ordered]
 hi=lo=0
 for i in range(1,len(p)):
  if projected[i]>projected[hi]:hi=i
  if projected[i]<projected[lo]:lo=i
 if hi==lo:return None
 return dict(records=[records[i] for i in ordered],center=center,major=major,normal=n,hi=hi,lo=lo)

def arcs(r):
 n=len(r['records'])
 def walk(a,b):
  result=[];i=(a+1)%n
  while i!=b:result.append(i);i=(i+1)%n
  return result
 return walk(r['hi'],r['lo']),walk(r['lo'],r['hi'])

def geometry(names,parents,deform,s,head,groups):
 # Build the original eye-forward, left/right and head-up context.
 eyes={}
 for side in ('l','r'):
  ids=groups.get(('eye_main',side),[]) or groups.get(('eye',side),[])
  if ids:eyes[side]=ids[0] if groups.get(('eye_main',side)) else min(ids,key=lambda i:len(names[i]))
 forward=norm(sum((s[i,1]-s[i,0] for i in eyes.values()),np.zeros(3,'f4')))
 right=context_norm(s[eyes['l'],0]-s[eyes['r'],0]) if len(eyes)==2 else np.zeros(3,'f4')
 up=context_norm(s[head,1]-s[head,0]) if 0<=head<len(s) else np.zeros(3,'f4')
 derived=context_norm(cross(forward,right))
 if np.any(derived) and len(eyes)==2:
  facing=dot(derived,F(.5)*(s[eyes['l'],0]+s[eyes['r'],0])-mean(s[:,0]))
  if abs(facing)<=F(1e-12):derived=np.zeros(3,'f4')
  elif facing<0:derived=-derived
 if not np.any(up):up=derived
 if np.any(forward):
  perpendicular=context_norm(up-forward*dot(forward,up))
  up=perpendicular if np.any(perpendicular) else derived
 if not np.any(up):return None
 scale=np.sqrt(squared(s[eyes['l'],0]-s[eyes['r'],0]),dtype='f4') if len(eyes)==2 else np.sqrt(squared(s[next(iter(eyes.values())) if eyes else head,1]-s[next(iter(eyes.values())) if eyes else head,0]),dtype='f4')
 if not scale>F(1e-9):return None
 return dict(eyes=eyes,forward=forward if np.any(forward) else None,right=right if np.any(right) else None,up=up,scale=scale)

def surface_records(names,parents,deform,s,head,groups,surface):
 context=geometry(names,parents,deform,s,head,groups)
 if context is None:return []
 vertices=np.asarray(surface[0],'f4').reshape((-1,3));edges=surface[1]
 rows=[]
 def add(kind,side,ids=(),reason='',status='absent'):
  role='eyelid_corner' if kind.startswith('eyelid_corner') else kind
  rows.append((f'{kind}.{side}',kind,role,side,0,list(ids),status,reason))
 mask=eligible(names,parents,deform,head)
 excluded=set(i for (kind,side),ids in groups.items() if kind in ('head','eye','eye_main','eye_card') for i in ids)
 for side in ('l','r'):
  eye=context['eyes'].get(side)
  if eye is None:
   add('eyelid_top',side,reason='this side has no eye bone to hang an opening on')
   add('eyelid_bot',side,reason='this side has no eye bone');continue
  other=context['eyes'].get('r' if side=='l' else 'l');candidates=[]
  for i,allowed in enumerate(mask):
   if not allowed or i in excluded:continue
   ep=endpoint(s[i],vertices)
   if ep is None:continue
   point=ep[1];distance=np.sqrt(squared(point-s[eye,0]),dtype='f4')
   if distance<=F(context['scale']*F(.9)) and (other is None or not np.sqrt(squared(point-s[other,0]),dtype='f4')<=distance):candidates.append((i,point,distance,np.sqrt(squared((s[i,1] if np.array_equal(point,s[i,0]) else s[i,0])-s[eye,0]),dtype='f4')))
  if len(candidates)<=3:
   add('eyelid_top',side,reason=f'{len(candidates)} bone(s) round this eye, too few for a ring');continue
  chosen=[c for c in candidates if not c[2]*F(.35)<c[3]]
  if len(chosen)<4:
   chosen=[];score=F(np.inf)
   for seed in candidates:
    radius=seed[2];shell=[]
    for iteration in range(3):
     shell=[c for c in candidates if not F(radius*F(.1))<abs(F(c[2]-radius)) and not F(radius*F(.1))<abs(F(c[3]-radius))]
     if not shell:break
     total=F(-0.)
     for c in shell:total=F(total+F(c[2]+c[3]))
     new=F(total/F(2*len(shell)))
     converged=abs(F(new-radius))<=F(radius*F(.01));radius=new
     if converged:break
    if len(shell)<4:continue
    forward=context['forward']
    if forward is not None:
     u=norm(cross(forward,np.array([1,0,0] if abs(forward[0])<F(.9) else [0,1,0],'f4')));v=cross(forward,u)
     if not np.any(u) or not np.any(v):continue
     angles=sorted(F(np.arctan2(dot(c[1]-s[eye,0],v),dot(c[1]-s[eye,0],u))) for c in shell)
     gaps=[F(angles[(i+1)%len(angles)]-angle) for i,angle in enumerate(angles)]
     gaps=[g if not g<0 else F(g+F(2*math.pi)) for g in gaps]
     if F(max(gaps)/F(2*math.pi))>F(.6):continue
    distances=[d for c in shell for d in c[2:]];spread=F(F(max(distances)-min(distances))/radius)
    if not chosen or spread<score:chosen=shell;score=spread
  r=ring([(c[0],c[1]) for c in chosen],mean([c[1] for c in chosen])-s[eye,0] if chosen else None,context['right'])
  if r is None:
   add('eyelid_top',side,reason="the bones on this eye's outline do not make a ring");continue
  a,b=arcs(r)
  if not a or not b:
   add('eyelid_top',side,reason='the ring is its two corners and nothing between them');continue
  points=[p for _,p in r['records']]
  def height(arc):
   total=F(0)
   for j in arc:total=F(total+dot(points[j]-points[r['hi']],context['up']))
   return F(total/F(len(arc)))
  if height(a)<height(b):a,b=b,a
  inner=None
  if other is not None:
   direction=s[other,0]-s[eye,0]
   inner=r['hi'] if dot(points[r['hi']]-s[eye,0],direction)>=dot(points[r['lo']]-s[eye,0],direction) else r['lo']
   outer=r['lo'] if inner==r['hi'] else r['hi']
   add('eyelid_corner_inner',side,[r['records'][inner][0]],"the end of the eye's loop nearer the other eye",'geometry')
   add('eyelid_corner_outer',side,[r['records'][outer][0]],"the end of the eye's loop away from the other eye",'geometry')
  else:
   for kind in ('eyelid_corner_inner','eyelid_corner_outer'):add(kind,side,reason='no other eye to say which end of the loop is the inner corner')
  for kind,arc,word in [('eyelid_top',a,'upper'),('eyelid_bot',b,'lower')]:
   if inner is not None and (inner-arc[0])%len(points)<(arc[0]-inner)%len(points):arc=arc[::-1]
   add(kind,side,[r['records'][j][0] for j in arc],f"the {word} arc of the eye's loop",'geometry')

 mouth=sorted(set(i for (kind,side),ids in groups.items() if kind in ('lip_top','lip_bot','lip_corner') for i in ids))
 if len(mouth)<4:
  add('lip_top','x',reason=f'{len(mouth)} bone(s) round the mouth, too few for a ring');return rows
 records=[];anchors=[]
 for i in mouth:
  ep=endpoint(s[i],vertices)
  if ep is not None:records.append((i,ep[1]));anchors.append(ep[0])
 r=ring(records,context['forward'],context['right'])
 if r is None:return rows
 # Connected-component majority (last/highest component wins equal counts).
 adjacency=[[] for _ in vertices]
 for a,b in edges:
  if 0<=a<len(vertices) and 0<=b<len(vertices) and a!=b:adjacency[a].append(b);adjacency[b].append(a)
 components=[-1]*len(vertices);label=0
 for v in range(len(vertices)):
  if components[v]>=0:continue
  stack=[v];components[v]=label
  while stack:
   for nxt in adjacency[stack.pop()]:
    if components[nxt]<0:components[nxt]=label;stack.append(nxt)
  label+=1
 from collections import Counter
 counts=Counter(components[v] for v in anchors);dominant=max(counts,key=lambda c:(counts[c],c))
 width=abs(F(dot(r['records'][r['hi']][1]-r['center'],r['major'])-dot(r['records'][r['lo']][1]-r['center'],r['major'])))
 threshold=F(width*F(.15));dominant_points=vertices[np.array(components)==dominant]
 projections=[dot(point-r['center'],r['normal']) for _,point in records];low,high=min(projections),max(projections)
 kept=[]
 for record,anchor in zip(records,anchors):
  point=record[1]
  if components[anchor]!=dominant and not any(np.sqrt(squared(point-v),dtype='f4')<=threshold for v in dominant_points):continue
  projection=dot(point-r['center'],r['normal']);distance=max(F(low-projection),F(projection-high),F(0))
  if not F(distance/max(width,F(1e-9)))>F(.25):kept.append(record)
 r=ring(kept,context['forward'],context['right'])
 if r is None:return rows
 a,b=arcs(r)
 if not a or not b:
  add('lip_top','x',reason="the mouth's ring is its two corners alone");return rows
 points=[p for _,p in r['records']];center=r['center'];major=r['major'];up=context['up']
 def height(arc):
  total=F(0)
  for i in arc:total=F(total+dot(points[i]-points[r['lo']],up))
  return F(total/F(len(arc)))
 if height(a)<height(b):a,b=b,a
 width=abs(F(dot(points[r['hi']]-center,major)-dot(points[r['lo']]-center,major)));band=F(width*F(.03))
 if context['right'] is not None:
  for i in (r['hi'],r['lo']):add('lip_corner','r' if dot(points[i]-center,major)<0 else 'l',[r['records'][i][0]],"an extreme of the mouth's loop",'geometry')
 else:
  for side in ('l','r'):add('lip_corner',side,reason="no eyes to say which way round the mouth's loop runs")
 for kind,arc,word in [('lip_top',a,'upper'),('lip_bot',b,'lower')]:
  for side in ('l','r','x') if context['right'] is not None else ('x',):
   indices=[i for i in arc if context['right'] is None or (dot(points[i]-center,major)>band if side=='l' else dot(points[i]-center,major)<-band if side=='r' else abs(dot(points[i]-center,major))<=band)]
   # Native sort follows distance from the center band, not integer bone order.
   indices.sort(key=lambda i:abs(float(dot(points[i]-center,major))))
   if indices:add(kind,side,[r['records'][i][0] for i in indices],f"the {word} arc of the mouth's loop",'geometry')
 return rows
