"""Independent default bone preparation with optional bucket shapes disabled.

All input geometry is supplied by the caller. No original module is imported.
The separately configured bucket constructor and tuning overrides remain open.
"""
import numpy as np


def prepare(runtime, verts, faces, heads, tails, parents, points, cell, *, bounded=True):
    h=np.ascontiguousarray(heads,'f4');t=np.ascontiguousarray(tails,'f4');p=np.ascontiguousarray(parents,'i8')
    if (bounded and len(h)*112>20_000_000) or not np.isfinite(cell) or cell<=0:
        raise ValueError('invalid base radius skeleton')
    # Bound rays, owned index storage/build work and query count before sampling.
    if bounded and len(h) and len(points) and (len(h)*10>20_000_000 or (len(h)+len(points))*96>128*1024*1024 or len(points)*len(points).bit_length()>20_000_000):
        raise ValueError('invalid bounded radius coverage inputs')
    finite=np.isfinite(h).all(axis=1)&np.isfinite(t).all(axis=1)
    if finite.all():
        ends,m=runtime.bone_base_radii(h.tolist(),t.tolist(),verts,faces,float(cell),bounded=bounded)
        ends=np.asarray(ends,'f4').reshape(-1,2);m=np.asarray(m,'f4')
    else:
        # Nonfinite ray frames have no valid hits in the original. Keep its
        # voxel-derived floor; finite neighbors still use the normal sampler.
        ends=np.full((len(h),2),float(cell)*.75,dtype='f4')
        m=np.full(len(h),float(cell)*.75,dtype='f4')
        if finite.any():
            e,q=runtime.bone_base_radii(h[finite].tolist(),t[finite].tolist(),verts,faces,float(cell),bounded=bounded)
            ends[finite]=e;m[finite]=q
    cone=runtime.core_chain_mask(h,t,p,m,1.8)
    for i in range(len(h)):
        if not cone[i]:
            d=t[i]-h[i];length=np.float32(np.sqrt(np.float32(np.float32(d[0]*d[0]+d[1]*d[1])+d[2]*d[2])))
            cap=float(length)*.5
            if cap>0:ends[i]=np.minimum(ends[i].astype('f8'),cap).astype('f4')
        a,b=map(float,ends[i])
        if a>=1e-6 and b>=1e-6:
            if b*3<a:ends[i,0]=b*3
            elif a*3<b:ends[i,1]=a*3
        if not cone[i]:m[i]=np.float32(np.float32(ends[i,0]+ends[i,1])*np.float32(.5))
    if len(h) and len(points):
        if finite.all():
            ends,m=runtime.fit_cloud_radii(np.asarray(points,'f4').tolist(),h.tolist(),t.tolist(),ends.tolist(),m.tolist(),float(cell),bounded=bounded)
        elif finite.any():
            # Fit finite neighbors through the unchanged native sampler.
            e,q=runtime.fit_cloud_radii(np.asarray(points,'f4').tolist(),h[finite].tolist(),t[finite].tolist(),ends[finite].tolist(),m[finite].tolist(),float(cell),bounded=bounded)
            ends[finite]=e;m[finite]=q
        if not finite.all():
            # With a nonempty cloud, nonfinite queries have infinite nearest
            # distance and take the original maximum cell expansion. An empty
            # cloud bypasses this entire fitting stage and retains the floor.
            delta=float(cell)*(max(1,int((1.8*.45)/float(cell)))+1)
            ends[~finite]=(ends[~finite].astype('f8')+delta).astype('f4')
            m[~finite]=(m[~finite].astype('f8')+delta).astype('f4')
    return dict(heads=h,tails=t,radii=np.asarray(ends,'f4'),r_mid=np.asarray(m,'f4'),cone_mask=cone,bucket=None,complete_native_preparation=False)
