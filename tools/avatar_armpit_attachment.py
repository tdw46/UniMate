"""Share upper cut-rim weights and diffuse the attachment into each bank.

The protected shoulder cap and distal arm separation remain outside this band.
Mesh coordinates, UVs, normals and topology are never changed here.
"""
import numpy as np,heapq
from mathutils.kdtree import KDTree
from avatar_apparel_weights import weights,assign
from avatar_surface_attachments import normalized

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
def close_attachment(rig,obj):
 if obj.get('hallway_armpit_attachment_revision')==1:return dict(already_applied=True)
 if not all(obj.data.attributes.get(name) for name in ('hallway_generated_fill','hallway_arm_bank')):return dict(vertices=0)
 if not all('UpperArm.'+side in rig.data.bones for side in ('L','R')):return dict(vertices=0)
 p=np.array([rig.matrix_world.inverted_safe()@obj.matrix_world@v.co for v in obj.data.vertices]);fill=np.array([v.value>0 for v in obj.data.attributes['hallway_generated_fill'].data]);banks=np.array([v.value for v in obj.data.attributes['hallway_arm_bank'].data]);edges=np.array([e.vertices[:] for e in obj.data.edges]);rim=set()
 for a,b in edges:
  if fill[a]!=fill[b]:rim.add(int(b if fill[a] else a))
 anchor={};domain=np.zeros(len(p),dtype=bool);pitch=min(rig.data.bones['UpperArm.'+s].length for s in ('L','R'));pairs=[]
 for side,sign in [('L',1),('R',-1)]:
  arm=rig.data.bones['UpperArm.'+side];h=np.array(arm.head_local);d=np.array(arm.tail_local)-h;L=arm.length
  if -d[2]<L*.55:continue
  t=(p-h)@d/(L*L);rad=np.linalg.norm(p-h-np.clip(t,0,1)[:,None]*d,axis=1);sel=(t>.12)&(t<.62)&(rad<L*.85);domain|=sel
  body=[i for i in rim if banks[i]==0 and sel[i]];hands=[i for i in rim if banks[i]==sign and sel[i]]
  if not body or not hands:continue
  tree=KDTree(len(body))
  for j,i in enumerate(body):tree.insert(p[i],i)
  tree.balance()
  for a in hands:
   _,b,dist=tree.find(p[a])
   if dist>L*.025:continue
   if _protected(weights(obj,a)) or _protected(weights(obj,b)):continue
   f=float(1-smooth((max(t[a],t[b])-.45)/.17));ra=weights(obj,a);rb=weights(obj,b);mean=normalized({n:(ra.get(n,0)+rb.get(n,0))*.5 for n in ra.keys()|rb.keys()})
   for i,r in ((a,ra),(b,rb)):anchor[i]=normalized({n:(1-f)*r.get(n,0)+f*mean.get(n,0) for n in r.keys()|mean.keys()})
   pairs.append((a,b))
 if not anchor:return {}
 graph=[[] for _ in p];dist=np.full(len(p),np.inf);queue=[]
 for a,b in edges:
  if not(domain[a] and domain[b]):continue
  length=float(np.linalg.norm(p[a]-p[b]));graph[a].append((int(b),length));graph[b].append((int(a),length))
 for i in anchor:dist[i]=0;heapq.heappush(queue,(0.,i))
 while queue:
  d,a=heapq.heappop(queue)
  if d>dist[a]:continue
  for b,length in graph[a]:
   dd=d+length
   if dd<dist[b] and dd<pitch*.12:dist[b]=dd;heapq.heappush(queue,(dd,b))
 selected=np.isfinite(dist);ids=np.flatnonzero(selected);lookup=np.full(len(p),-1);lookup[ids]=np.arange(len(ids));rows=[weights(obj,int(i)) for i in ids]
 names=sorted(set().union(*(r.keys() for r in rows)));w=np.array([[r.get(n,0) for n in names] for r in rows]);baseline=w.copy();moving=np.array([i not in anchor and dist[i]<pitch*.10 for i in ids]);protected=np.array([_protected(row) for row in rows]);moving[protected]=False
 for i,row in anchor.items():w[lookup[i]]=[row.get(n,0) for n in names]
 edge=edges[selected[edges].all(1)];edge=edge[(~protected[lookup[edge]]).all(1)];a=np.concatenate((lookup[edge[:,0]],lookup[edge[:,1]]));b=np.concatenate((lookup[edge[:,1]],lookup[edge[:,0]]));count=np.bincount(a,minlength=len(ids));factor=.6*(1-smooth(dist[ids]/(pitch*.10)))
 for _ in range(40):
  average=np.stack([np.bincount(a,weights=w[b,k],minlength=len(ids))/np.maximum(count,1) for k in range(len(names))],axis=1);w[moving]+=(average[moving]-w[moving])*factor[moving,None]
 for i,row,protect in zip(ids,w,protected):
  if not protect:assign(obj,[int(i)],normalized(dict(zip(names,map(float,row)))))
 obj['hallway_armpit_attachment_revision']=1
 return dict(pairs=len(pairs),vertices=len(ids),geometry_changed=False,max_influences=4)


def blend_repaired_armpits(rig,meshes):
 return [dict(object=obj.name,**close_attachment(rig,obj)) for obj in meshes if obj.data.attributes.get('hallway_arm_bank')]


def _protected(row):
 return row.get('Hips',0)>=.5 or any(n.startswith(('Secondary_','Hand.','Thumb','Index','Middle','Ring','Little','Thigh.','Shin.','Foot.')) and w>1e-5 for n,w in row.items())
