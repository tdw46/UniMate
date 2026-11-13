"""Independent region writeback for bounded sparse deform rows.

Selection belongs to the planner. This combines supplied label skins in source
order, clears only vertices actually written, and preserves non-deform groups.
No normalization or top-k truncation follows the original per-group writes.
"""
import numpy as np

def apply_regions(weights,bones,names,offsets,regions,*,deform_names=None):
 w=np.asarray(weights,'f4');b=np.asarray(bones,'i4');offsets=np.asarray(offsets,'i8')
 if w.ndim!=2 or w.shape!=b.shape or not np.isfinite(w).all() or np.any(w<0) or np.any(w>1) or len(set(names))!=len(names):raise ValueError('invalid region input rows')
 if np.any((b < -1)|(b>=len(names))) or len(offsets)<2 or offsets[0]!=0 or offsets[-1]!=len(w) or np.any(np.diff(offsets)<0):raise ValueError('invalid region input indices')
 lookup={name:i for i,name in enumerate(names)};deform=set(names) if deform_names is None else set(deform_names)
 if not deform.issubset(lookup):raise ValueError('unknown deform group')
 # Validate before changing caller data. Sparse output can grow beyond top-k.
 prepared=[]
 for mi,touched,counts,labels in regions:
  if not 0<=mi<len(offsets)-1:raise ValueError('invalid region mesh')
  size=int(offsets[mi+1]-offsets[mi]);items=[]
  for label,indices,skin,replace in labels:
   if skin is None:raise ValueError('missing region skin')
   sn,sw,si,_=skin;ii=np.asarray(indices,'i8');sw=np.asarray(sw,'f4');si=np.asarray(si,'i4')
   if ii.ndim!=1 or sw.ndim!=2 or si.shape!=sw.shape or len(sw)!=len(ii) or np.any(ii<0) or np.any(ii>=size):raise ValueError('invalid region skin dimensions')
   if any(name not in lookup for name in sn) or len(set(sn))!=len(sn) or not np.isfinite(sw).all() or np.any(sw<0) or np.any(sw>1) or np.any(si < -1) or np.any(si>=len(sn)):raise ValueError('invalid region skin values')
   items.append((ii+offsets[mi],sn,sw,si,bool(replace)))
  prepared.append(items)
 outw=w.copy();outb=b.copy()
 for items in prepared:
  flat=[int(i) for ii,*_ in items for i in ii];written=set(flat)
  # Retain the already-qualified, byte-stable disjoint path when all groups
  # are deform groups and there are no duplicate destination indices.
  unique_slots=all(len(set(map(int,row[row>=0])))==int(np.count_nonzero(row>=0)) for _,_,_,si,_ in items for row in si)
  if len(flat)==len(written) and deform==set(names) and unique_slots:
   for ii,sn,sw,si,replace in items:
    width=max(outw.shape[1],sw.shape[1])
    if width>outw.shape[1]:
     outw=np.pad(outw,((0,0),(0,width-outw.shape[1])));outb=np.pad(outb,((0,0),(0,width-outb.shape[1])),constant_values=-1)
    outw[ii]=0;outb[ii]=-1;outw[ii,:sw.shape[1]]=sw
    ids=np.full(si.shape,-1,'i4');valid=si>=0;ids[valid]=np.array([lookup[name] for name in sn],'i4')[si[valid]];outb[ii,:si.shape[1]]=ids
   continue
  rows={i:{int(bone):np.float32(weight) for bone,weight in zip(outb[i],outw[i]) if bone>=0 and weight>0 and names[bone] not in deform} for i in written}
  for ii,sn,sw,si,replace in items:
   # Match group batching and distinct-weight ordering, including repeated
   # vertex IDs and repeated bone IDs in multiple columns of one skin.
   for col in range(sw.shape[1]):
    for local,name in enumerate(sn):
     mask=(si[:,col]==local)&(sw[:,col]>0);values=sw[mask,col];dest=ii[mask];bone=lookup[name]
     for value in np.unique(values):
      for index in dest[values==value]:
       row=rows[int(index)];row[bone]=value if replace else np.minimum(np.float32(1),np.float32(row.get(bone,np.float32(0))+value))
  width=max([outw.shape[1],*[len(row) for row in rows.values()]])
  if width>outw.shape[1]:
   outw=np.pad(outw,((0,0),(0,width-outw.shape[1])));outb=np.pad(outb,((0,0),(0,width-outb.shape[1])),constant_values=-1)
  for i,row in rows.items():
   outw[i]=0;outb[i]=-1
   for col,(bone,weight) in enumerate(row.items()):outw[i,col]=weight;outb[i,col]=bone
 return outw,outb
