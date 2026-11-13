"""Independent bounded serial detector composition; no original imports.

Dense fill and bounded serial winding classification are implemented. Winding
queries use the recovered GPU shader when available, with the qualified CPU
fallback; universal equivalence near classification thresholds is not claimed. Level-tagged void boxes force serial refinement and filter assembled points.
Parallel marker/cluster scheduling remains unqualified.
"""
import numpy as np


MAX_GRID_CELLS=1_000_000
MAX_ADAPTIVE_WORK=16_000_000

class WorkBudget:
    """Cumulative cell, probe-step and interior-candidate bounds for one build.

    These are conservative work units, not byte counts or complete CPU accounting.
    """
    def __init__(self,*,bounded=True):self.used=0;self.bounded=bool(bounded)
    def spend(self,stage,amount):
        amount=int(amount)
        if amount<0 or (self.bounded and amount>MAX_ADAPTIVE_WORK-self.used):
            raise ValueError('adaptive work budget exceeded before '+stage)
        self.used+=amount


def cluster_grid_bounds(origin,coarse_cell,parent_cell,markers):
    # Mirror the native float32 marker bounds and coarse-cell padding.
    m=np.asarray(markers,'i4');origin=np.asarray(origin,'f4');p=np.float32(parent_cell)
    low=origin+p*m.min(axis=0).astype('f4')
    high=origin+p*((m.max(axis=0)+np.int32(1)).astype('f4'))
    inverse=np.float32(1)/p
    low=origin+p*np.floor((low-origin)*inverse)
    high=origin+p*np.ceil((high-origin)*inverse)
    return low-np.float32(coarse_cell),high+np.float32(coarse_cell)


def cluster_grid_budget(origin,coarse_cell,parent_cell,fine_cell,markers,*,bounded=True):
    return grid_budget(cluster_grid_bounds(origin,coarse_cell,parent_cell,markers),fine_cell,bounded=bounded)


def grid_budget(bounds,cell,*,bounded=True):
    """Admission by actual grid dimensions, before native allocation or probing."""
    lo=np.asarray(bounds[0],'f4');hi=np.asarray(bounds[1],'f4');cell=np.float32(cell)
    if not bounded:
        from .public_api import _runtime
        return _runtime().grid_admission(lo.tolist(),hi.tolist(),float(cell),bounded=False)
    with np.errstate(invalid='ignore',over='ignore',divide='ignore'):
        extent=hi-lo
        # Native float max selects one for NaN (e.g. infinity / infinity).
        dims=np.fmax(np.ceil(extent/cell),np.float32(1))
    if not cell>0 or (bounded and not np.isfinite(cell)) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or not np.isfinite(dims).all():
        raise ValueError('nonfinite grid bounds or cell outside research qualification')
    cells=1
    for d in dims:
        cells*=int(d)
        if bounded and cells>MAX_GRID_CELLS:raise ValueError('cloud grid allocation exceeds research cell budget')
    return cells


def require_refinement_bytes(needed, available):
    if needed > available:
        raise ValueError(f'adaptive refinement working-set research budget exceeded: {needed} bytes > {available} bytes')


def check_marker_storage(depths, max_depth, multiplier, available):
    # Each active probe inserts a cube of padded markers per level. Check its
    # upper bound before the native hash sets and returned Python rows exist.
    remaining = available - 512*max_depth
    require_refinement_bytes(0, remaining)
    levels = np.asarray(depths, dtype='u4')
    used = 0
    padding = 1
    for level in range(min(max_depth, int(levels.max(initial=0)))):
        active = int(np.count_nonzero(levels > level))
        used += 512*active*(2*padding+1)**3
        require_refinement_bytes(used, remaining)
        padding *= multiplier


def detect(runtime, verts, faces, bounds, *, resolution=8, multiplier=2,
           max_depth=1, probe_multiplier=4, fill_interior=True, ghost=True,
           interior_route="dense", prepared_void_boxes=None, face_void_groups=None, prepared_refine_boxes=None, winding_threshold=.5,bounded=True,diagnostics=False,ghost_mesh=None,grid_memory_admission=None):
    if interior_route not in ("dense", "winding"):raise ValueError("unknown interior classification route")
    if resolution<1 or multiplier<1 or probe_multiplier<1 or max_depth<0 or (bounded and max_depth and (max_depth>4 or multiplier>16 or probe_multiplier>5000)):
        raise ValueError('detector research workload exceeds bound')
    v=np.ascontiguousarray(verts,'f4');f=np.ascontiguousarray(faces,'i4')
    lo=np.asarray(bounds[0],'f4');hi=np.asarray(bounds[1],'f4')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        coarse_cell=np.float32(np.max(hi-lo)/np.float32(resolution))
    work=WorkBudget(bounded=bounded);coarse_cells=grid_budget([lo,hi],coarse_cell,bounded=bounded)
    work.spend('coarse grid',coarse_cells)
    if max_depth:work.spend('probe grid',grid_budget([lo,hi],np.float32(coarse_cell/np.float32(probe_multiplier)),bounded=bounded))
    plan = grid_memory_admission() if grid_memory_admission is not None else None
    coarse_options = {'diagnostics': True} if diagnostics else {}
    if plan is not None and 'coarse_point_limit' in plan:
        coarse_options['max_output_points'] = plan['coarse_point_limit']
    c=runtime.coarse_cloud(v,f,lo.tolist(),hi.tolist(),resolution,fill_interior,winding_threshold,bounded=bounded,**coarse_options)
    refinement_bytes = None
    if plan is not None and 'refinement_bytes' in plan:
        coarse_points = len(c['voxel_indices']) + len(c['inside_voxel_indices'])
        refinement_bytes = plan['refinement_bytes'] - 1024*coarse_points
    if face_void_groups is not None:
        if prepared_void_boxes is not None:raise ValueError('combined automatic and prepared boxes unsupported')
        from .face_voids import plan_boxes
        prepared_void_boxes=plan_boxes(face_void_groups,lo,hi,c['voxel_size'],level=max_depth,multiplier=multiplier)
    cell=np.float32(c['voxel_size'])
    # Native probe voxelization recomputes spacing from the full extent and
    # combined resolution. Dividing the rounded coarse cell again changes
    # triangle ownership at non-power-of-two multipliers.
    pc=np.float32(np.max(hi-lo)/np.float32(resolution*probe_multiplier))
    array_assembly = getattr(runtime, '_assemble_adaptive_arrays', None) if refinement_bytes is None else None
    surfaces=[];interiors=[];cells=[];levels=[];xyz=np.empty((0,3),'i4');normals=np.empty((0,3),'f4');depths=[];first_groups=[]
    if max_depth:
        if refinement_bytes is not None:
            require_refinement_bytes(512*max_depth, refinement_bytes)
        # Owned array transport preserves the legacy helper result by default.
        probe_options = {'array_output': True} if 'array_output' in (getattr(runtime.probe_surface, '__text_signature__', None) or '') else {}
        probe=runtime.probe_surface(v,f,lo.tolist(),hi.tolist(),float(pc),bounded=bounded,**probe_options)
        xyz=np.asarray(probe['indices'],'i4').reshape(-1,3);normals=np.asarray(probe['normals'],'f4').reshape(-1,3)
        work.spend('refinement ray steps',len(xyz)*(4*probe_multiplier+1))
        dist,target=runtime.refinement_probe(xyz.astype('i8'),normals,probe['dimensions'],2*probe_multiplier,bounded=bounded)
        depths=runtime.refinement_depths(dist,target,float(cell),float(pc),multiplier,max_depth,bounded=bounded)
        if refinement_bytes is not None:
            check_marker_storage(depths, max_depth, multiplier, refinement_bytes)
        markers=runtime.refinement_markers(xyz,depths,lo.tolist(),float(pc),float(cell),multiplier,max_depth,1,True,bounded=bounded)
        if prepared_void_boxes is not None or prepared_refine_boxes is not None:
            from .face_voids import force_markers
            options = {}
            if refinement_bytes is not None:
                options['max_entries'] = (refinement_bytes-512*max_depth)//1024
            markers=force_markers(runtime,markers,list(prepared_void_boxes or [])+list(prepared_refine_boxes or []),multiplier,bounded=bounded,**options)
        if refinement_bytes is not None:
            refinement_bytes -= 512*(max_depth + sum(map(len, markers)))
            require_refinement_bytes(0, refinement_bytes)
        if interior_route == 'dense':
            dims=c['resolution'];outside=np.ones(int(np.prod(dims)),'u1')
            for x,y,z in c['voxel_indices']+c['inside_voxel_indices']:outside[x+(y+z*dims[1])*dims[0]]=0
        # Diagnostic memory budgets retain the prior transient-tree path.
        # Ordinary detection owns one immutable tree across its cluster queries;
        # release it before cloud assembly, ghost culling, or cache publication.
        prepare_winding = getattr(runtime, '_WindingQuery', None) if refinement_bytes is None else None
        winding = None
        # The native detector uses GPU winding independently of the diffusion
        # switch. The recovered shader returns normalized winding numbers.
        # Diagnostic budgets retain their previously qualified CPU ownership.
        prepare_gpu_winding = getattr(runtime, '_PreparedGpuWinding', None) if refinement_bytes is None else None
        gpu_winding = None
        prepare_surface = getattr(runtime, '_SurfaceQuery', None) if refinement_bytes is None else None
        prepared_surface = None
        fine=cell
        voids=[]
        active_levels=next((i for i,mark in enumerate(markers) if not mark),len(markers))
        for j,mark in enumerate(markers):
            groups=runtime.refinement_cluster_groups(mark,bounded=bounded)
            if not groups:break
            parent=np.float32(cell/np.float32(multiplier**j));fine=np.float32(fine/multiplier)
            surface=[];inside=[];boxes=[];surface_owners=[];inside_owners=[]
            if j==0:first_groups=groups
            for group in groups:
                bounds = cluster_grid_bounds(lo, cell, parent, group)
                fine_cells = grid_budget(bounds, fine, bounded=bounded)
                if refinement_bytes is not None:
                    from .field_memory import surface_box_coverage
                    raster = surface_box_coverage((v,f), *bounds, fine, visits_only=True)
                    if raster is None:
                        raise ValueError('adaptive cluster raster exceeds research working-set qualification')
                    surface_count = raster['occupied_upper_bound']
                    raster_bytes = 512*surface_count + 16*raster['triangle_box_visits']
                    candidates = len(group)*multiplier**3
                    flood_bytes = 16*fine_cells + 10*coarse_cells if interior_route == "dense" else 0
                    require_refinement_bytes(raster_bytes + flood_bytes + 1024*(surface_count+candidates+1), refinement_bytes)
                work.spend('adaptive cluster grids',coarse_cells+fine_cells)
                work.spend('adaptive interior candidates',len(group)*multiplier**3)
                surface_args=(float(parent),float(fine),group,multiplier,voids if fill_interior and j==active_levels-1 else [],outside.tolist() if interior_route=="dense" else None,None,3 if len(groups)==1 else 1)
                if prepare_surface is not None:
                    if prepared_surface is None:prepared_surface=prepare_surface(v,f,lo.tolist(),hi.tolist(),float(cell),bounded=bounded)
                    result=prepared_surface.query(*surface_args)
                else:
                    result=runtime.adaptive_surface_points(v,f,lo.tolist(),hi.tolist(),float(cell),*surface_args,bounded=bounded)
                if diagnostics:surface_owners.extend([len(boxes)]*len(result['surface_points']))
                if array_assembly is not None:
                    surface.append(np.asarray(result['surface_points'],'f4').reshape(-1,3))
                else:
                    surface.extend(result['surface_points'].tolist() if isinstance(result['surface_points'],np.ndarray) else result['surface_points'])
                candidate=np.asarray(result['inside_points'],'f4').reshape(-1,3)
                if fill_interior and interior_route=="winding" and len(candidate):
                    raw_angles = None
                    if prepare_gpu_winding is not None:
                        try:
                            if gpu_winding is None:
                                gpu_winding = prepare_gpu_winding(v, f)
                            raw_angles = gpu_winding.query(candidate)
                        except ValueError as exc:
                            # Native context discovery may be unavailable even
                            # in a GPU-enabled build; retain CPU classification.
                            # Do not swallow allocation, validation or device errors.
                            if str(exc) != 'No supported compute adapter':
                                raise
                            gpu_winding = None
                            prepare_gpu_winding = None
                    if raw_angles is not None:
                        angles = np.frombuffer(raw_angles, 'f4')
                    else:
                        if prepare_winding is not None:
                            if winding is None:
                                winding = prepare_winding(v, f)
                            raw_angles = winding.query(candidate)
                        else:
                            raw_angles = runtime.winding_solid_angles_cpu(v, f, candidate)
                        angles = np.frombuffer(raw_angles, 'f4') / np.float32(12.566371)
                    del raw_angles
                    # Native fine-level classification stays at .5; the public threshold affects coarse fill.
                    candidate=candidate[angles>=np.float32(.5)]
                if fill_interior:
                    if array_assembly is not None:inside.append(candidate)
                    else:inside.extend(candidate.tolist())
                    if diagnostics:inside_owners.extend([len(boxes)]*len(candidate))
                boxes.append(result['detection_bounds'])
                if refinement_bytes is not None:
                    # Completed groups keep only their output rows and box.
                    refinement_bytes -= 1024*(len(result['surface_points']) + (len(candidate) if fill_interior else 0) + 1)
                    require_refinement_bytes(0, refinement_bytes)
                    del result, candidate
            if array_assembly is not None:
                surface=np.concatenate(surface,axis=0) if surface else np.empty((0,3),'f4')
                inside=np.concatenate(inside,axis=0) if inside else np.empty((0,3),'f4')
            if fill_interior and active_levels>1 and j==active_levels-2:
                voids=np.floor((np.asarray(inside,'f4').reshape(-1,3)-lo)*np.float32(1.0/fine)).astype('i4').tolist()
            surfaces.append(surface);interiors.append(inside);cells.append(float(fine))
            levels.append(dict(depth=j+1,voxel_centers=np.asarray(surface,'f4').reshape(-1,3),inside_centers=np.asarray(inside,'f4').reshape(-1,3),detection_bboxes=np.asarray(boxes,'f4').reshape(-1,6),voxel_size=float(fine)))
            if diagnostics:levels[-1].update(voxel_count=len(surface),inside_count=len(inside),voxel_parent_clusters=np.asarray(surface_owners,'i4'),inside_parent_clusters=np.asarray(inside_owners,'i4'))
        del winding, gpu_winding, prepared_surface
    out=(array_assembly or runtime.assemble_adaptive_cloud)(c['voxel_indices'],c['inside_voxel_indices'],lo.tolist(),float(cell),surfaces,interiors,cells,bounded=bounded)
    p=np.asarray(out['cloud_positions'],'f4').reshape(-1,3);lev=np.frombuffer(out['cloud_levels'],'u1');r=np.asarray(out['cloud_radii'],'f4');counts=[]
    if prepared_void_boxes is not None:
        from .face_voids import cloud_keep
        keep=cloud_keep(runtime,p,lev,lo,cell,multiplier,prepared_void_boxes,bounded=bounded)
        p=p[keep];lev=lev[keep];r=r[keep]
    if ghost and len(p):
        gv,gf=(v,f) if ghost_mesh is None else ghost_mesh
        keep,counts=runtime.ghost_cull(gv,gf,p,lev.tolist(),lo.tolist(),hi.tolist(),float(cell),multiplier,bounded=bounded)
        keep=np.frombuffer(keep,'u1').astype(bool);p=p[keep];lev=lev[keep];r=r[keep]
    runtime.check_cancel()
    result=dict(refine_boxes=prepared_refine_boxes,void_boxes=prepared_void_boxes,cloud_positions=p,cloud_levels=lev,cloud_radii=r,levels=levels,voxel_size=float(cell),ghost_counts=counts,complete_native_detector=False)

    if diagnostics:result.update(coarse=c,l4_voxel_indices=xyz,l4_voxel_depths=np.asarray(depths,'u4'),l4_voxel_normals=normals,first_groups=first_groups)
    return result
