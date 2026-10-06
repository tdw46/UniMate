"""Conservative averaging at the upper-helper / main-skirt handoff.

Keep body weights exact and preserve the averaged total at each chain and
joint level when limiting to four influences. A first-moment-only reduction
can exchange helper weight for lower-joint weight and fold this short region.
"""
import numpy as np,itertools

def average_helper_handoff(rig,body,bones,values,edges,iterations=20,factor=.08):
 helpers={b.name:b['hallway_support_chain'].rsplit('_',1)[0] for b in rig.data.bones if b.get('hallway_skirt_support') and b.get('hallway_support_chain') and b['hallway_support_chain'] in bones}
 names=sorted(set(bones)|{n for row in body for n in row})
 lookup={n:i for i,n in enumerate(names)}
 bone_columns={n:i for i,n in enumerate(bones)}
 seed=np.array([[row.get(n,0.) if n not in bone_columns else values[i,bone_columns[n]] for n in names] for i,row in enumerate(body)])
 if not len(seed):return body,values
 if not np.isfinite(seed).all() or (seed<0).any() or not 0<=factor<=1 or iterations<0:
  raise ValueError('Invalid helper interpolation weights')
 chains=[helpers.get(n,n.rsplit('_',1)[0] if n in bones else None) for n in names]
 levels=[-1 if n in helpers else int(n.rsplit('_',1)[1]) if n in bones else None for n in names]
 cloth=np.array([c is not None for c in chains]);h=np.array([n in helpers for n in names])
 eligible=seed[:,h].sum(axis=1)>1e-8
 # Include one graph ring around the handoff, with the remaining cloth
 # providing pinned boundary samples. This graph is already virtually welded.
 for a,b in edges:
  if seed[a,h].sum()>1e-8:eligible[b]=True
  if seed[b,h].sum()>1e-8:eligible[a]=True
 allowed=seed>0;movable=np.zeros(len(seed),dtype=bool);matrices={}
 for i in np.flatnonzero(eligible):
  cs={chains[k] for k in np.flatnonzero(seed[i]>1e-8) if cloth[k]}
  if not cs or len(cs)>2:continue
  lv={levels[k] for k in np.flatnonzero(seed[i]>1e-8) if cloth[k]}|{-1,0}
  if max(lv)>1:continue
  cols=tuple(k for k,n in enumerate(names) if (cloth[k] and chains[k] in cs and levels[k] in lv) or (not cloth[k] and seed[i,k]>0))
  rows=[[float(chains[k]==c) for k in cols] for c in sorted(cs)]
  rows += [[float(levels[k]==l) for k in cols] for l in sorted(lv)[:-1]]
  rows += [[float(k==j) for k in cols] for j in cols if not cloth[j]]
  a=np.array(rows)
  # Four weights cannot preserve more than four independent quantities.
  # Retain the existing mixed bodice row in that case rather than sacrificing
  # attachment weights or silently exceeding the export influence budget.
  if np.linalg.matrix_rank(a)>4:continue
  allowed[i]=False;allowed[i,list(cols)]=True;movable[i]=True;matrices[i]=(cols,a)
 e=np.array(edges,dtype=int).reshape(-1,2);target=np.r_[np.arange(len(seed)),e[:,0],e[:,1]];source=np.r_[np.arange(len(seed)),e[:,1],e[:,0]];degree=np.bincount(target,minlength=len(seed))[:,None]
 w=seed.copy();bodymass=seed[:,~cloth].sum(axis=1)
 for _ in range(iterations):
  avg=np.zeros_like(w);np.add.at(avg,target,w[source]);w=(1-factor)*w+factor*avg/degree;w*=allowed
  w[:,~cloth]=seed[:,~cloth]
  w[:,cloth]*=((1-bodymass)/np.maximum(w[:,cloth].sum(axis=1),1e-30))[:,None]
  w[~movable]=seed[~movable]
 result=seed.copy();cache={}
 for i in np.flatnonzero(movable):
  cols,a=matrices[i];key=cols
  if key not in cache:
   rank=np.linalg.matrix_rank(a);bases=[]
   mandatory={j for j,k in enumerate(cols) if not cloth[k]}
   for sub in itertools.combinations(range(len(cols)),min(4,len(cols))):
    if not mandatory.issubset(sub):continue
    aa=a[:,sub]
    if np.linalg.matrix_rank(aa)==rank:bases.append((np.array(sub),np.linalg.pinv(aa)))
   cache[key]=bases
  dense=w[i,list(cols)];b=a@dense;best=None;cost=float('inf')
  for sub,pinv in cache[key]:
   v=dense[sub]+pinv@(b-a[:,sub]@dense[sub])
   if v.min() < -1e-9:continue
   candidate=np.zeros(len(cols));candidate[sub]=np.maximum(v,0)
   err=np.max(np.abs(a@candidate-b))
   if err>1e-7:continue
   diff=float(np.sum((candidate-dense)**2))
   if diff<cost:best=candidate;cost=diff
  if best is None:
   raise ValueError('No nonnegative four-weight helper interpolation')
  result[i]=0;result[i,list(cols)]=best
  result[i,~cloth]=seed[i,~cloth]
 return [{n:float(v) for n,v in zip(names,row) if n not in bones and v>1e-10} for row in result],np.array([[row[lookup[n]] for n in bones] for row in result])
