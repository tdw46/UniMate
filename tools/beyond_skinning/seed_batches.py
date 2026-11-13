"""Bounded public seed transport; full-cloud decisions remain global per bone."""
import numpy as np
PAIR_BLOCK=4_000_000
def _storage(points,bones):
    # Outer bind admission also includes its actual graph and phase resources.
    # This local check bounds owned masks plus block scratch independently.
    from .field_memory import require
    return require(points,bones,compact_phase='seeding')

def _layout(points,heads):
    n=len(points);b=len(heads)
    _storage(n,b)
    if n>PAIR_BLOCK:
        raise ValueError('public seed mask storage/work budget exceeded')
    return n,b,max(1,PAIR_BLOCK//max(n,1))

def profile_masks(runtime,points,heads,tails,rh,rt,rm):
    n,b,width=_layout(points,heads);out=np.empty((b,n),'u1')
    for a in range(0,b,width):
        runtime.check_cancel();z=min(a+width,b)
        h,t=heads[a:z],tails[a:z]
        finite=np.isfinite(h).all(axis=1)&np.isfinite(t).all(axis=1)
        if finite.all():
            rows=runtime.seed_profile_masks(points,h,t,rh[a:z],rt[a:z],rm[a:z])
            out[a:z]=np.frombuffer(b''.join(rows),'u1').reshape(z-a,n)
        else:
            # Projection against a nonfinite segment cannot pass the native
            # ordered radius comparison. Keep those mask rows empty.
            block=out[a:z]
            block.fill(0)
            if finite.any():
                radii=[np.asarray(radius[a:z])[finite].tolist() for radius in (rh,rt,rm)]
                rows=runtime.seed_profile_masks(points,h[finite],t[finite],*radii)
                block[finite]=np.frombuffer(b''.join(rows),'u1').reshape(-1,n)
    return out

def nearest_bones(runtime, points, heads, tails):
    finite=np.isfinite(heads).all(axis=1)&np.isfinite(tails).all(axis=1)
    if finite.all():
        return runtime.nearest_seed_bones(points,heads,tails)
    ids=np.flatnonzero(finite)
    if not len(ids):
        return [0]*len(points)
    return ids[runtime.nearest_seed_bones(points,heads[ids],tails[ids])].tolist()

def constrained_masks(runtime,points,heads,tails,rh,rt,rm,kinds,labels,sphere_radii,**kw):
    n,b,width=_layout(points,heads);out=np.empty((b,n),'u1')
    finite=np.isfinite(heads).all(axis=1)&np.isfinite(tails).all(axis=1)
    if not finite.all():
        ids=np.flatnonzero(finite)
        out.fill(0)
        if len(ids):
            def select(rows):
                return [rows[i] for i in ids]
            remap=np.full(b,-1,'i4')
            remap[ids]=np.arange(len(ids))
            filtered=dict(kw)
            for key in ('centers','oriented','base_radii','custom_shapes'):
                if filtered.get(key) is not None:
                    filtered[key]=select(filtered[key])
            out[ids]=constrained_masks(runtime,points,heads[ids],tails[ids],
                select(rh),select(rt),select(rm),select(kinds),remap[labels].tolist(),
                select(sphere_radii),**filtered)
        for i in np.flatnonzero(~finite):
            if kinds[i] in (2,18):
                out[i]=np.equal(labels,i)
            if kinds[i] not in (15,16):
                continue
            center=(heads if kinds[i]==15 else tails)[i]
            if kw.get('centers') is not None:
                center=np.asarray(kw['centers'][i],'f4')
            if not np.isfinite(center).all():
                continue
            # Endpoint spheres read only their selected center. Supply that
            # same point at both ends, excluding this row from segment search.
            endpoint=center.reshape(1,3)
            base=kw['base_radii'][i] if kw.get('base_radii') is not None else 0.
            out[i]=constrained_masks(runtime,points,endpoint,endpoint,[0.],[0.],[0.],
                [kinds[i]],[0]*n,[sphere_radii[i]],base_radii=[base],
                cell_size=kw.get('cell_size',.1),point_radii=kw.get('point_radii'))[0]
        return out
    defaults=dict(centers=None,cell_size=.1,reference_length=0.,oriented=None,base_radii=None,point_radii=None,axial_padding=0.,body_scale=1.,separation=1.,custom_shapes=None)
    if set(kw)-set(defaults):raise TypeError('unknown constrained seed option')
    defaults.update(kw)
    args=(points,heads,tails,rh,rt,rm,kinds,labels,sphere_radii,*defaults.values())
    prepare=getattr(runtime,'_PreparedSeedMasks',None)
    prepared=None
    for a in range(0,b,width):
        runtime.check_cancel();z=min(a+width,b)
        out[a:z]=(runtime.constrained_seed_mask_block(*args,a,z) if prepared is None else prepared.block(a,z))
        # Defer the immutable snapshot until existing first-block validation
        # succeeds; reuse only within this call, never across binds or caches.
        if a==0 and z<b and prepare is not None:prepared=prepare(*args)
    return out

def custom_field(runtime,points,masks,priorities,shapes):
    n=len(points);b=len(priorities)
    if masks.shape!=(b,n):raise ValueError('public custom seed work budget exceeded')
    _storage(n,b)
    width=max(1,PAIR_BLOCK//max(b,1));fixed=np.empty(n,'u1');labels=np.empty(n,'u4');touched=protected=0
    for a in range(0,n,width):
        runtime.check_cancel();z=min(a+width,n)
        if 'array_output' in (getattr(runtime.custom_seed_field,'__text_signature__','') or ''):
            ff,ll,tt,pp=runtime.custom_seed_field(points[a:z],masks[:,a:z],priorities,shapes,array_output=True)
        else:
            _,_,(ff,ll),tt,pp=runtime.custom_seed_field(points[a:z],masks[:,a:z],priorities,shapes)
        fixed[a:z]=np.frombuffer(ff,'u1');labels[a:z]=ll
        touched+=int(np.frombuffer(tt,'u1').sum(dtype=np.uint64));protected+=int(np.frombuffer(pp,'u1').sum(dtype=np.uint64))
    return fixed,labels,int(touched),int(protected)
