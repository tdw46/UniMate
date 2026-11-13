"""Independent rig-name driven, eyelid shape construction.

Default tuning only. Names/hierarchy and caller geometry are the sole inputs.
This is a bounded face-family slice, not the complete humanoid rig map. Other
face roles and unsupported corner configurations fail closed rather than using approximate records.
"""
import numpy as np


def construct(runtime, names, heads, tails, parents, deform=None, *, head=-1, context_only=False,active_ids=None,role_override=None,allow_nonfinite=False):
    h=np.ascontiguousarray(heads,'f4');t=np.ascontiguousarray(tails,'f4')
    n=len(names);p=np.ascontiguousarray(parents,'i8')
    if h.shape!=(n,3) or t.shape!=h.shape or p.shape!=(n,):
        raise ValueError('invalid bounded face skeleton')
    if (p>=n).any() or (p < -1).any() or head>=n or head < -1:
        raise ValueError('face skeleton outside research bounds')
    # Reject cycles before any name-based native-compatible parent traversal.
    for i in range(n):
        seen=set();j=i
        while j>=0:
            if j in seen:raise ValueError('cyclic face hierarchy')
            seen.add(j);j=int(p[j])
    roles,family=runtime.detect_face(list(names),p.tolist(),deform,head) if role_override is None else role_override
    if active_ids is not None:roles=[row for row in roles if row[0] in active_ids]
    if not allow_nonfinite and (not np.isfinite(h).all() or not np.isfinite(t).all()):
        raise ValueError('invalid bounded face skeleton')
    supported={'eye','eyelid_top','eyelid_bot','eyelid_corner','brow','cheek','nose','jaw','tongue','ear','skull','forehead','chin'}
    partial_mouth=not ({'lip_top','lip_bot'} <= {role for _,role,_ in roles})
    if partial_mouth:supported.update(('lip_top','lip_bot','lip_corner'))
    if context_only:supported.update(('lip_top','lip_bot','lip_corner','teeth_top','teeth_bot'))
    if any(role not in supported or side not in ('l','r','x') for _,role,side in roles):
        raise ValueError('unknown automatic facial role or side')
    groups={side:[i for role in ('eyelid_top','eyelid_bot') for i,r,s in roles if r==role and s==side] for side in ('l','r')}
    corners={side:[i for i,r,s in roles if r=='eyelid_corner' and s==side] for side in ('l','r')}
    from .rig_map_native import corners as partition,mean as point_mean,squared
    records={}
    for i,role,side in roles:records.setdefault((role,side),[]).append(i)
    partitions=partition(records,np.stack([h,t],1))
    for side,ids in corners.items():
        if not ids:continue
        # Inner then outer: native map splits each side by tail proximity to X=0.
        inner,outer,_=partitions[side]
        ids[:]=inner+outer
        groups[side].extend(ids)
    role_by_id={i:r for i,r,_ in roles}
    shapes=[{'kind':'none'} for _ in range(n)];head_lids=[]
    for side,ids in groups.items():
        if not ids:continue
        if len(ids)<4 or not any(role_by_id[i]=='eyelid_top' for i in ids) or not any(role_by_id[i]=='eyelid_bot' for i in ids):
            head_lids.extend(ids)
            continue
        other=groups['r' if side=='l' else 'l']
        eyes=[i for i,r,s in roles if r=='eye' and s==side]
        axis_hint=None
        inner,outer,_=partitions.get(side,([],[],False))
        if inner and outer:
            delta=point_mean(t[outer])-point_mean(t[inner]);length=np.float32(np.sqrt(squared(delta)))
            axis_hint=(np.array([1,0,0],'f4') if length<=np.float32(1e-12) else delta/length).tolist()
        try:
            values=runtime.eyelid_group_shapes(h[ids].tolist(),t[ids].tolist(),[role_by_id[i]=='eyelid_top' for i in ids],h[eyes].tolist(),opposite_tails=t[other].tolist() if other else None,axis_hint=axis_hint)
        except ValueError as exc:
            if str(exc)!='degenerate eyelid ring':raise
            continue
        if corners[side]:
            F=np.float32
            first=values[0];center=np.asarray(first['center'],'f4');axis=np.asarray(first['axis'],'f4');lateral=np.asarray(first['lateral'],'f4')
            dot=lambda a,b:F(F(a[0]*b[0]+a[1]*b[1])+a[2]*b[2])
            def ordinate(point):
                d=point-center;y=dot(lateral,d)
                if 'sphere' not in first:return y
                x=dot(axis,d);radial=F(np.hypot(x,y));radius=F(first['sphere'][0])
                if radial<=F(1e-9):return F(0)
                z=F(F(d[0]*F(0)+d[1])+d[2]*F(0))
                return F(F(y/radial)*F(radius*F(np.arctan2(radial,F(-z)))))
            seam=F(-0.)
            for i in corners[side]:seam=F(seam+ordinate(t[i]))
            seam=F(seam/F(len(corners[side])))
            mean=np.zeros(3,'f4')
            for i in ids:mean+=t[i]/F(len(ids))
            xs=[dot(axis,t[i]-mean) for i in ids];width=max(F(max(xs)-min(xs)),F(1e-6))
            for i,value in zip(ids,values):
                params=np.asarray(value['params'],'f4')
                params[5]=seam;params[8]=max(F(0),F(params[8]-seam));params[9]=max(F(0),F(params[9]+seam))
                params[4]=F(-1 if ordinate(t[i])<seam else 1)
                if i in corners[side]:params[4]=0;params[7]=F(width*F(.3));params[10]=0;params[11]=F(.4)
                value['params']=params.tolist()
        for i,value in zip(ids,values):shapes[i]=value
    return dict(shapes=shapes,roles=roles,family=family,groups=groups,head_lids=head_lids,complete_rig_shapes=False)


def normalized_shapes(shapes, scale):
    """Scale geometric fields after construction; retain angular/unitless fields."""
    import copy
    out=copy.deepcopy(shapes);s=np.float32(scale)
    for shape in out:
        if shape['kind']=='none':continue
        if shape['kind']!='profile':raise ValueError('unrecovered normalized shape kind')
        for key in ('center','fallback_center','radii','heights','lengths','sphere'):
            if key in shape:shape[key]=(np.asarray(shape[key],'f4')*s).tolist()
        if 'plane' in shape:
            plane=np.asarray(shape['plane'],'f4');plane[:3]*=s;shape['plane']=plane.tolist()
        params=np.asarray(shape['params'],'f4');params[[0,1,2,3,5,6,7,8,9,12]]*=s;shape['params']=params.tolist()
    return out


def project_context(runtime, context, names, heads, tails, *, allow_nonfinite=False):
    """Construct from the full rig, then remap records into the solve columns.

    The planner retains all part bones; solve shapes use only active columns.
    Reserved eyes must not impose the eye-distance cap on eyelid depth.
    Context is caller geometry, not an oracle plan or previous skin weights.
    """
    n,h,t,p=context[:4]
    deform=context[4] if len(context)==5 else None
    h=np.asarray(h,'f4');t=np.asarray(t,'f4')
    head=-1
    if deform is not None:
        from .rig_mapping import detect_spine
        head=detect_spine(n,p,deform)['head']
    roles,family=runtime.detect_face(list(n),np.asarray(p).tolist(),deform,head)
    # A context occurrence without a facial role must not hide a later,
    # eligible occurrence with the same name (for example a root-level jaw).
    lookup={}
    for i,_,_ in roles:lookup.setdefault(n[i],i)
    for i,name in enumerate(n):lookup.setdefault(name,i)
    if any(name not in lookup for name in names):raise ValueError('solve bone missing from full context')
    ids=[lookup[name] for name in names]
    full=construct(runtime,n,h,t,p,deform=deform,head=head,context_only=True,active_ids=set(ids),role_override=(roles,family),allow_nonfinite=allow_nonfinite);remap={}
    for new,old in enumerate(ids):remap.setdefault(old,new)
    return dict(shapes=[full['shapes'][i] for i in ids],roles=[(remap[i],role,side) for i,role,side in full['roles'] if i in remap],
                family=full['family'],groups={side:[remap[i] for i in members if i in remap] for side,members in full['groups'].items()},
                head_lids=[remap[i] for i in full['head_lids'] if i in remap],complete_rig_shapes=False,full_context_count=len(n))


def reported_roles(roles,heads,tails):
    from .rig_map_native import corners as partition
    groups={}
    for i,role,side in roles:groups.setdefault((role,side),[]).append(i)
    parts=partition(groups,np.stack([heads,tails],1));kinds={}
    for inner,outer,_ in parts.values():
        kinds.update((i,'eyelid_corner_inner') for i in inner)
        kinds.update((i,'eyelid_corner_outer') for i in outer)
    return [(i,kinds.get(i,role) if role=='eyelid_corner' else role,side) for i,role,side in roles]
