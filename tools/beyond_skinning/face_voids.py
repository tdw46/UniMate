"""Bounded automatic face boxes, serial refinement markers and cloud filtering."""
import numpy as np

def filter_voxels(indices,boxes):
    xyz=np.asarray(indices,dtype='i8').reshape(-1,3)
    raw=np.asarray(boxes)
    if raw.size==0:return xyz.tolist()
    if raw.ndim!=2 or raw.shape[1]!=7 or len(raw)>64 or not np.issubdtype(raw.dtype,np.integer) or np.any(raw[:,0]!=0):raise ValueError('only prepared level-zero integer void boxes are qualified')
    keep=np.ones(len(xyz),bool)
    for box in raw:
        lo=np.minimum(box[1:4],box[4:7]);hi=np.maximum(box[1:4],box[4:7])
        keep &= ~np.all((xyz>=lo)&(xyz<=hi),axis=1)
    return xyz[keep].tolist()


def face_groups(roles,heads,tails):
    """Reference 0x90aa4: interleaved heads/tails in solve-bone order."""
    h=np.asarray(heads,'f4');t=np.asarray(tails,'f4')
    if h.shape!=t.shape or h.ndim!=2 or h.shape[1]!=3 or not np.isfinite(h).all() or not np.isfinite(t).all():raise ValueError('invalid bounded face void geometry')
    ordered=sorted(roles,key=lambda row:row[0])
    def points(kinds,side=None):
        ids=[i for i,role,s in ordered if role in kinds and (side is None or s==side)]
        return np.stack([h[ids],t[ids]],axis=1).reshape(-1,3)
    out=[]
    mouth=points(('lip_corner',))
    if len(mouth)<2:mouth=points(('lip_top','lip_bot'))
    if len(mouth)>=2:out.append((mouth,mouth))
    for side in ('l','r'):
        eyes=points(('eyelid_top','eyelid_bot','eyelid_corner'),side)
        corners=points(('eyelid_corner',),side)
        if len(eyes)>=2:out.append((eyes,corners if len(corners) else eyes))
    return out


def plan_boxes(groups,origin,bounds_max,cell,*,level=0,multiplier=2,bounded=True):
    """Bounded level-tagged automatic face planes; inclusive voxel endpoints.

    ARM64 fcvtms floors both interval endpoints and mean Z; fcvtps rounds the
    whole orthogonal span upward. Decompiler integer casts lost this distinction.
    Equal XY widths choose X. Cell/origin are the existing coarse-grid values.
    """
    if bounded and (not 0<=level<=3 or not 1<=multiplier<=4):raise ValueError('face void level outside research bounds')
    F=np.float32;cell=F(F(cell)/F(multiplier**level));lo=np.asarray(origin,'f4');hi=np.asarray(bounds_max,'f4');cell=F(cell)
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([*lo,*hi,cell]).all() or cell<=0 or np.any(hi<lo) or (bounded and len(groups)>3):raise ValueError('invalid face void grid')
    boxes=[]
    for pts,mean_pts in groups:
        pts=np.asarray(pts,'f4');mean_pts=np.asarray(mean_pts,'f4')
        if pts.ndim!=2 or pts.shape[1]!=3 or len(pts)<2 or (bounded and len(pts)>512) or mean_pts.ndim!=2 or mean_pts.shape[1]!=3 or len(mean_pts)<1 or (bounded and len(mean_pts)>512) or not np.isfinite(pts).all() or not np.isfinite(mean_pts).all():raise ValueError('invalid face void group')
        low=pts[:,:2].min(axis=0);high=pts[:,:2].max(axis=0);width=high-low
        axis=int(width[0]<width[1]);other=1-axis
        a=[0,0,0];b=[0,0,0]
        a[axis]=int(np.floor(F(F(low[axis]-lo[axis])/cell)))
        b[axis]=int(np.floor(F(F(high[axis]-lo[axis])/cell)))
        b[other]=int(np.ceil(F(F(hi[other]-lo[other])/cell)))
        total=F(-0.)
        for point in mean_pts:total=F(total+point[2])
        a[2]=b[2]=int(np.floor(F(F(F(total/F(len(mean_pts)))-lo[2])/cell)))
        boxes.append([level,*a,*b])
    return boxes


def checked_boxes(boxes,*,bounded=True):
    raw=np.asarray([] if boxes is None else boxes)
    if not raw.size:return np.empty((0,7),'i8')
    if raw.ndim!=2 or raw.shape[1]!=7 or not np.issubdtype(raw.dtype,np.integer):raise ValueError('invalid integer box records')
    if np.any(raw[:,0]<0) or np.any(raw[:,0]>4294967295) or np.any(raw[:,1:]< -2147483648) or np.any(raw[:,1:]>2147483647):raise ValueError('box record outside native integer range')
    if bounded and (len(raw)>64 or np.any(raw[:,0]>3) or np.any(np.abs(raw[:,1:])>100000)):raise ValueError('void box research bound exceeded')
    raw=raw.astype('i8');low=np.minimum(raw[:,1:4],raw[:,4:7]);high=np.maximum(raw[:,1:4],raw[:,4:7]);raw[:,1:4]=low;raw[:,4:7]=high
    return raw


def force_markers(runtime,markers,boxes,multiplier,*,bounded=True,max_entries=None):
    """Native 0xaaaf4: pad at source level, expand inclusive parent markers."""
    from itertools import product
    from math import prod
    initial = sum(map(len, markers))
    if max_entries is not None and initial > max_entries:
        raise ValueError('adaptive forced markers exceed research working-set budget')
    out=[list(map(list,rows)) for rows in markers];seen=[set(map(tuple,rows)) for rows in out]
    visits=0
    for row in checked_boxes(boxes,bounded=bounded):
        # Native pads in signed 32-bit arithmetic before rescaling. Do not
        # normalize the wrapped endpoints: an inverted interval emits nothing.
        wrap=lambda value: (int(value)+2147483648)%4294967296-2147483648
        padded=[int(row[0]),*(wrap(v-1) for v in row[1:4]),*(wrap(v+1) for v in row[4:7])]
        for level in range(min(len(out),int(row[0]))):
            bounds=runtime.rescale_refinement_box(padded,level,multiplier,bounded=bounded)
            count=prod(max(0, int(bounds[k+3])-int(bounds[k])+1) for k in range(3))
            visits+=count
            if max_entries is not None and initial + visits > max_entries:
                raise ValueError('adaptive forced markers exceed research working-set budget')
            if bounded and visits>200000:raise ValueError('void marker workload exceeded')
            for xyz in product(*(range(bounds[k],bounds[k+3]+1) for k in range(3))):
                if xyz not in seen[level]:out[level].append(list(xyz));seen[level].add(xyz)
    return out


def cloud_keep(runtime,positions,levels,origin,coarse_cell,multiplier,boxes,*,bounded=True):
    """Native post-assembly filtering at each point's own level, before ghosts."""
    p=np.asarray(positions,'f4');levels=np.asarray(levels);origin=np.asarray(origin,'f4');raw=checked_boxes(boxes,bounded=bounded)
    keep=np.ones(len(p),bool)
    if not len(raw):return keep
    for level in np.unique(levels):
        ids=np.flatnonzero(levels==level);cell=np.float32(np.float32(coarse_cell)/np.float32(multiplier**int(level)))
        # ARM64 fcvtms w: floor, saturate to i32, and map NaN to zero.
        with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
            mapped=np.floor((p[ids]-origin)/cell).astype('f8')
        xyz=np.clip(np.nan_to_num(mapped,nan=0.,posinf=2147483647.,neginf=-2147483648.),-2147483648.,2147483647.).astype('i8')
        for row in raw:
            bound=runtime.rescale_refinement_box(row.tolist(),int(level),multiplier,bounded=bounded)
            keep[ids]&=~np.all((xyz>=bound[:3])&(xyz<=bound[3:]),axis=1)
    return keep


def public_boxes(boxes,*,bounded=True):
    """Public tuples -> normalized records; retain order and duplicates in key."""
    if boxes is None:return []
    rows=[]
    for level,low,high in boxes:
        if len(low)!=3 or len(high)!=3:raise ValueError('box endpoints require three coordinates')
        rows.append([level,*low,*high])
    return checked_boxes(rows,bounded=bounded).tolist()
