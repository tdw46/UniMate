"""Bounded leaf-fan geometric detection with one-level auxiliary hubs.

Recovered 0x15af88: exclude cones, collect leaf chains, require four/five
multi-bone chains, reject divergent average absolute direction alignment.
"""
import numpy as np
from .rig_mapping import validate

def detect_fingers_geometric(heads,tails,parents,cone_mask):
    h,t=(np.asarray(x,'f4').astype('f8') for x in (heads,tails));p=list(map(int,parents));m=list(map(bool,cone_mask))
    if len(p)>256:raise ValueError('Geometric finger research skeleton bound exceeded')
    validate(['']*len(p),p,m)
    if h.shape!=(len(p),3) or t.shape!=h.shape or len(m)!=len(p) or not np.isfinite(h).all() or not np.isfinite(t).all():raise ValueError('Invalid finite finger geometry')
    return _detect_geometric(h,t,p,m)

def _detect_geometric(h,t,p,m,checked=list):
    children=checked([[] for _ in p])
    for i,par in enumerate(p):
        if not m[i] and par>=0:children[par].append(i)
    groups={};claimed=set()
    for leaf in range(len(p)):
        if m[leaf] or children[leaf]:continue
        chain=[leaf];extra=[];at=leaf
        while p[at]>=0 and not m[p[at]]:
            par=p[at];siblings=[x for x in children[par] if x!=at]
            if siblings:
                if len(siblings)==1 and children[at] and not children[siblings[0]]:
                    a=t[siblings[0]]-h[siblings[0]];b=t[par]-h[par]
                    na=np.sqrt((a[0]*a[0]+a[1]*a[1])+a[2]*a[2]);nb=np.sqrt((b[0]*b[0]+b[1]*b[1])+b[2]*b[2])
                    # ARM b.le includes unordered values; decompiled !(x <= y) is misleading.
                    a=a/na if na>1e-9 else np.zeros(3);b=b/nb if nb>1e-9 else np.zeros(3)
                    # ARM b.lt rejects unordered dot products.
                    if abs((a[0]*b[0]+a[1]*b[1])+a[2]*b[2])>=.7:
                        extra.append(siblings[0]);claimed.add(siblings[0]);chain.append(par);at=par;continue
                break
            chain.append(par);at=par
        groups.setdefault(p[chain[-1]],[]).append((chain,extra))
    output=[];used=set();ranked=[]
    candidates=set(groups)|{p[i] for i in groups if i>=0}
    def depth(i):
        count=0
        while i>=0:count+=1;i=p[i]
        return count
    for hub in sorted(i for i in candidates if i>=0):
        keys=[i for i in [hub]+children[hub] if i in groups]
        records=[record for i in keys for record in groups[i] if record[0][0] not in claimed]
        valid=[(c,extra) for c,extra in records if len(c)>=2]
        if len(valid) not in (4,5):continue
        covered={i for c,extra in records for i in c+extra}|(set(keys)-{hub})
        desc=set();todo=list(children[hub])
        while todo:
            i=todo.pop()
            if i in desc:continue
            desc.add(i);todo.extend(children[i])
        if desc!=covered:continue
        dirs=[]
        for c,_ in valid:
            d=t[c[0]]-h[c[-1]];length=np.sqrt((d[0]*d[0]+d[1]*d[1])+d[2]*d[2]);dirs.append(d/length if length>1e-9 else np.zeros(3))
        alignment=0.
        for a in dirs:
            for b in dirs:alignment+=abs((a[0]*b[0]+a[1]*b[1])+a[2]*b[2])
        if (alignment-len(dirs))/(len(dirs)*(len(dirs)-1))<.5:continue
        accepted=[sorted(c+extra) for c,extra in valid]
        owned={i for c in accepted for i in c}
        # Native tuple sort: more chains, fewer gathered hubs, deeper root.
        ranked.append(((-len(valid),len(keys),-depth(hub)),accepted,owned))
    for _,accepted,owned in sorted(ranked,key=lambda row:row[0]):
        # A competing candidate is rejected as a whole on any shared bone.
        if owned&used:continue
        output.extend(accepted);used.update(owned)
    return sorted(output,key=lambda c:c[0])

def detect_fingers(names,heads,tails,parents,cone_mask,deform=None):
    """Name precedence, geometric completion, then disjoint geometric additions."""
    from .rig_mapping import detect_fingers_by_name
    named,family=detect_fingers_by_name(names,parents,deform)
    geometric=detect_fingers_geometric(heads,tails,parents,cone_mask)
    return _combine_fingers(named,family,geometric)

def _combine_fingers(named,family,geometric):
    if not named:return geometric,'geometric'
    completed=0;result=[]
    for chain in named:
        best=next((c for c in geometric if len(c)>len(chain) and set(chain)<=set(c)),None)
        result.append(best if best is not None else chain)
        completed+=best is not None
    owned={i for c in result for i in c};extra=[c for c in geometric if not owned.intersection(c)]
    how=f'name ({family})'
    if completed:how+=f', {completed} completed by geometry'
    if extra:how+=f' + {len(extra)} geometric'
    return sorted(result+extra,key=lambda c:c[0]),how


def _public_geometric_fingers(heads,tails,parents,mask):
    from . import _bvt_solver as runtime
    if not len(heads):return []
    class Checked(list):
        def __getitem__(self,index):
            if index<0 or index>=len(self):raise runtime.PanicException(f'index out of bounds: the len is {len(self)} but the index is {index}')
            return super().__getitem__(index)
    h=Checked(np.asarray(heads,'f8'));t=Checked(np.asarray(tails,'f8'))
    return _detect_geometric(h,t,Checked(parents),Checked(mask),Checked)


def _public_combined_fingers(names,heads,tails,parents,mask,deform):
    from .rig_mapping import _public_named_fingers
    named,family=_public_named_fingers(names,parents,deform)
    geometric=_public_geometric_fingers(heads,tails,parents,mask)
    return _combine_fingers(named,family,geometric)
