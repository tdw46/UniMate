"""Public bind metadata versus solve geometry. Original binary never consulted."""
import numpy as np
from .rig_mapping import normalize

def resolve(runtime,names,h,t,p,full_names,full_parent,full_deform,segments,face,unsupported):
    nf=len(full_names)
    if len(full_deform)<nf:
        raise runtime.PanicException(f'index out of bounds: the len is {len(full_deform)} but the index is {len(full_deform)}')
    # Metadata traverses the named/deforming context independently of the
    # solve arrays. Extra unnamed entries and unused segment rows are inert.
    class Parents(list):
        def __getitem__(self,index):
            if isinstance(index,int) and (index<0 or index>=len(self)):
                raise runtime.PanicException(f'index out of bounds: the len is {len(self)} but the index is {index}')
            return super().__getitem__(index)
    parents=Parents(map(int,full_parent[:nf]))
    # Bound cycles/invalid links without eagerly dereferencing missing rows.
    for start in range(len(parents)):
        seen=set();i=start
        while 0<=i<len(parents):
            if i in seen:unsupported('cyclic full context')
            seen.add(i);i=parents[i]
        if i < -1 or i>=nf:unsupported('full context parent bounds')
    from .spine_mapping import detect_spine_owned
    torso=detect_spine_owned(list(full_names),list(map(normalize,full_names)),list(parents),list(map(bool,full_deform[:nf])),None)
    if torso['family']:
        for i in range(nf):parents[i]
    fp=np.asarray(list(parents)+[-1]*(nf-len(parents)),dtype='i8')
    if not torso["verified"]:
        torso=dict(torso,family="",label="",pelvis=-1,head=-1,spine=[],neck=[])
    lookup={}
    for i,name in enumerate(names):lookup.setdefault(name,i)
    reverse={i:lookup[name] for i,name in enumerate(full_names) if name in lookup}
    report=dict(torso)
    report["pelvis_name"]=full_names[torso["pelvis"]] if torso["pelvis"]>=0 else ""
    report["spine_names"]=[full_names[i] for i in torso["spine"]]
    for key in ('pelvis','head'):report[key]=reverse.get(torso[key],-1)
    for key in ('spine','neck'):report[key]=[reverse[i] for i in torso[key] if i in reverse]
    roles,family=runtime.detect_face(list(full_names),fp.tolist(),full_deform[:nf].tolist(),torso['head']) if face else ([], '')
    # Native full-context mouth grouping reads lower then upper segment rows
    # before the solve uses its own posed geometry. Preserve those bounds and
    # the pre-cache failure, without demanding unused body/terminal rows.
    active_roles=[row for row in roles if row[0] in reverse]
    if {'lip_top','lip_bot'} <= {role for _,role,_ in active_roles}:
        for i,role,_ in sorted(active_roles,key=lambda row:row[1]):
            if role not in ('lip_top','lip_bot','lip_corner'):continue
            if i>=len(segments):
                raise runtime.PanicException(f'index out of bounds: the len is {len(segments)} but the index is {i}')
            if segments.shape[1:]!=(2,3):unsupported('mouth context segment columns')
    roles=list(dict.fromkeys((reverse[i],role,side) for i,role,side in active_roles))
    if not roles:family=""
    context=None
    # A genuine superset retains auxiliary face geometry. When every context
    # name is solved, the live solve positions own those shapes; segment data
    # is not a second, conflicting pose to enforce.
    if face and roles and any(name not in lookup for name in full_names if name):
        # The active solve supplies its own endpoints. Unsolved hierarchy
        # entries still own their names/parents but need no fabricated pivot.
        count=min(nf,len(segments))
        ch=np.full((nf,3),np.nan,'f4');ct=ch.copy()
        ch[:count]=segments[:count,0];ct[:count]=segments[:count,1]
        for i,j in reverse.items():ch[i]=h[j];ct[i]=t[j]
        cn=list(full_names);cd=np.asarray(full_deform[:nf],bool).tolist()
        # Extra solve columns participate in geometry, but cannot acquire a
        # semantic role from a context entry that does not exist.
        missing=[j for j,name in enumerate(names) if name not in cn]
        if missing:
            cn.extend(names[j] for j in missing)
            ch=np.concatenate([ch,h[missing]]);ct=np.concatenate([ct,t[missing]])
            cd.extend(False for _ in missing)
            fp=np.concatenate([fp,np.full(len(missing),-1,'i8')])
        context=(cn,ch,ct,fp,cd)
    return report,roles,family,context
