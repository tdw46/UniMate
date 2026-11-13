"""Owned, one-slot research session. No original fallback.

Bounded serial clouds, normalized void/refinement boxes, radius preparation and
public bucket metadata/cache ownership are implemented. Direct extraction/cache controls are qualified separately from
unrestricted cloud generation and the full derived-cache API.
"""
import numpy as np
from threading import RLock
from functools import wraps
from .adaptive_detector import detect
from .bone_preparation import prepare
from .cloud_resources import CloudResources


def _near(a,b,tol):
    if a.shape!=b.shape:return False
    # Native max-number reduction ignores unordered differences (including inf-inf).
    with np.errstate(invalid='ignore',over='ignore'):
        distance=np.fmax.reduce(np.abs(a-b).reshape(-1),initial=np.float32(0))
    return float(distance)<float(np.float32(tol))


def _threshold_key(value):
    # Native f32 multiply followed by saturating signed-i64 conversion.
    with np.errstate(over='ignore',invalid='ignore'):
        scaled=np.float32(np.float32(value)*np.float32(1e6))
    if np.isnan(scaled):return 0
    if scaled>=2**63:return 2**63-1
    if scaled<=-2**63:return -2**63
    return int(scaled)


def _clone(d):
    return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in d.items()}


def _cache_locked(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self._lock:return method(self,*args,**kwargs)
    return call


class ResearchBindSession:
    def __init__(self,runtime,*,interior_route="dense",bounded_preparation=True,bounded_cloud=True):
        if interior_route not in ("dense","winding"):raise ValueError("unknown interior route")
        self._lock=RLock()
        self.interior_route=interior_route
        self.runtime=runtime
        # Explicit harness policy; the recovered public adapter selects False.
        self.bounded_preparation=bool(bounded_preparation)
        self.bounded_cloud=bool(bounded_cloud)
        self.clear()

    @_cache_locked
    def clear(self):
        self._cloud=None
        self._bones=None
        self.generation=0

    def cloud_for(self,verts,faces,bbox_min,bbox_max,coarse_res,refine_multiplier,max_depth,
                  fill_interior,winding_threshold,ghost_cull,probe_multiplier=None,
                  void_boxes=None,refine_boxes=None,*,_grid_memory_admission=None):
        from .face_voids import public_boxes
        voids=public_boxes(void_boxes,bounded=self.bounded_cloud);refines=public_boxes(refine_boxes,bounded=self.bounded_cloud)
        v=np.array(verts,dtype='f4',copy=True,order='C');f=np.array(faces,dtype='i4',copy=True,order='C')
        # Native default exponent saturates at eight, without constructing huge integers.
        probe=probe_multiplier
        if probe is None:
            probe=1
            for _ in range(min(max_depth,4)):
                probe=min(8,probe*refine_multiplier)
            probe=max(1,probe)
        self.runtime.validate_cloud_geometry(v,f,bbox_min,bbox_max,coarse_res,refine_multiplier,probe)
        key=(self.interior_route,int(coarse_res),int(refine_multiplier),int(max_depth),bool(fill_interior),_threshold_key(winding_threshold),bool(ghost_cull),int(probe),tuple(map(tuple,voids)),tuple(map(tuple,refines)))
        with self._lock:
            old=self._cloud
            hit=old is not None and old['key']==key and np.array_equal(f,old['faces']) and _near(v,old['verts'],1e-5)
            if hit:
                return old['cloud'],old['bvh'],dict(old['summary'],levels=list(old['summary']['levels']),hit=True)
        # Original 0xa93a8 unlocks before construction and republishes under lock.
        # Every miss owns its inputs and result; completion order owns the cache.
        if not hit:
            self.runtime.check_cancel()
            cloud=detect(self.runtime,v,f,[bbox_min,bbox_max],resolution=coarse_res,multiplier=refine_multiplier,max_depth=max_depth,probe_multiplier=probe,fill_interior=fill_interior,ghost=ghost_cull,interior_route=self.interior_route,prepared_void_boxes=voids,prepared_refine_boxes=refines,winding_threshold=winding_threshold,bounded=self.bounded_cloud,grid_memory_admission=_grid_memory_admission)
            order=self.runtime.morton_order(cloud['cloud_positions'])
            positions=cloud['cloud_positions'][order];levels=cloud['cloud_levels'][order];radii=cloud['cloud_radii'][order]
            obj=self.runtime.VoxelCloud(positions,levels,radii);bvh=self.runtime.MeshBvh(v,f)
            summary=dict(voxel_size=cloud['voxel_size'],ghost_culled=sum(cloud['ghost_counts']),levels=[(x['depth'],x['voxel_size'],len(x['voxel_centers']),len(x['inside_centers'])) for x in cloud['levels']])
            with self._lock:
                self._cloud=dict(key=key,verts=v,faces=f,cloud=obj,bvh=bvh,summary=summary,points=positions,resources=CloudResources(self.runtime,positions,levels,radii))
                self.generation+=1;self._bones=None;old=self._cloud
        return old['cloud'],old['bvh'],dict(old['summary'],levels=list(old['summary']['levels']),hit=hit)

    def bones_for(self,heads,tails,parent_idx,shape_cfg,*,_bucket=None):
        h=np.array(heads,'f4',copy=True);t=np.array(tails,'f4',copy=True);p=np.array(parent_idx,'i8',copy=True)
        with self._lock:
            if self._cloud is None:raise RuntimeError('bones_for before cloud_for: the session has no cloud')
            c=self._cloud;generation=self.generation;old=self._bones
            hit=old is not None and old['generation']==generation and old['cfg']==shape_cfg and np.array_equal(p,old['parents']) and _near(h,old['heads'],1e-6) and _near(t,old['tails'],1e-6)
            if hit:return dict(_clone(old['prep']),hit=True)
        # Own this generation while preparation runs without the cache lock.
        # Concurrent misses each publish their own result; the captured generation
        # prevents a late old-cloud result from hitting the replacement cloud.
        prep=prepare(self.runtime,c['verts'],c['faces'],h,t,p,c['points'],c['summary']['voxel_size'],bounded=self.bounded_preparation)
        prep.pop('complete_native_preparation',None)
        prep['cone_mask']=np.asarray(prep['cone_mask'],bool)
        prep['bucket']=None if _bucket is None else np.frombuffer(_bucket,dtype='u1').copy()
        with self._lock:
            self._bones=dict(heads=h,tails=t,parents=p,cfg=shape_cfg,prep=prep,generation=generation)
        return dict(_clone(prep),hit=False)

    @_cache_locked
    def resources(self):
        if self._cloud is None:raise ValueError('cloud_for must be called before resource access')
        return self._cloud['resources']
