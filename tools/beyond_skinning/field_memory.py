"""Research working-set admission, independent of original allocation failures.

Bounds retained dense levels and overlapping native/array temporaries. This is
not a process/GPU memory cap: source geometry, interpreter and preexisting cloud
caches already exist. Public binds set no independent estimated-byte cap; the
original host hint belongs only to its native per-grid guard. Explicit diagnostic
budgets remain available and never describe free memory or allocation success.
"""
from contextvars import ContextVar
import operator
import os
import sys
from functools import lru_cache
DEFAULT_BUDGET_BYTES = 256 * 1024 * 1024

# Variables are local to the current bind, including nested callback binds.
# None denotes native public admission. The diagnostic default does not apply
# inside that scope; tokens restore the caller after errors and nested binds.
bind_phase_budget = ContextVar('bind_phase_budget', default=DEFAULT_BUDGET_BYTES)

def reported_phase_budget(host_bytes):
    return max(DEFAULT_BUDGET_BYTES, host_bytes // 8)

@lru_cache(maxsize=1)
def physical_host_bytes():
    """Installed host RAM for independent scratch admission, in bytes."""
    try:
        if os.name == 'nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                            *[(name, ctypes.c_uint64) for name in
                              ('total_phys', 'avail_phys', 'total_page', 'avail_page',
                               'total_virtual', 'avail_virtual', 'avail_extended')]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total_phys
            return 0
        return max(0, os.sysconf('SC_PHYS_PAGES')) * max(0, os.sysconf('SC_PAGE_SIZE'))
    except (AttributeError, OSError, ValueError):
        return 0

def sample_map_storage(vertices,sample_k,*,prior_bytes=0,borrowed_samples=False):
    """Map ownership plus one native/Python conversion block, not a flat cap."""
    v,k,prior=map(operator.index,(vertices,sample_k,prior_bytes))
    if min(v,k,prior)<0:raise ValueError('negative sample map dimension')
    # The internal map owns one uint32 ID and one float32 weight per entry.
    # Conversion and sentinel normalization both fit in one query block.
    # External callers receive writable copies under the existing allowance.
    block=min(v,1024,max(1,65536//max(k,1)))
    bytes_per_entry = 8 if borrowed_samples else 24
    return dict(map_arrays=bytes_per_entry*v*k,prior_map=prior,transport=block*(88*k+160))


def require_sample_map(vertices,sample_k,*,prior_bytes=0,budget_bytes=None,borrowed_samples=False):
    budget_bytes = bind_phase_budget.get() if budget_bytes is None else operator.index(budget_bytes)
    parts=sample_map_storage(vertices,sample_k,prior_bytes=prior_bytes,borrowed_samples=borrowed_samples)
    parts['fixed_reserve']=1024*1024
    estimated=(sum(parts.values())*5+3)//4
    if budget_bytes is not None and estimated>budget_bytes:
        raise ValueError(f'sample map working-set research budget exceeded: {estimated} bytes > {budget_bytes} bytes')
    return dict(components=parts,estimated_bytes=estimated)

def estimate(points, bones, *, vertices=0, levels=1, edges=0, sample_k=0, top_k=4, rolling=False, compact_phase=None, stage_points=0, stage_edges=0, prior_bytes=0, shared_graph=False, solver_field_released=False, dense_phase=None,borrowed_samples=False,owned_dense_graph=False,cached_samples=False,retained_owner_bytes=None,stream_direct=False,packed_samples=False):
    dims = [operator.index(x) for x in (points,bones,vertices,levels,edges,sample_k,top_k)]
    if any(x < 0 for x in dims) or levels < 1:
        raise ValueError('negative field dimension or empty level count')
    points,bones,vertices,levels,edges,sample_k,top_k=dims
    if dense_phase is not None and (not rolling or compact_phase is not None or dense_phase not in ("solve", "sampling")):
        raise ValueError("invalid dense allocation phase")
    if solver_field_released and not (compact_phase == "sampling" or dense_phase == "sampling"):
        raise ValueError("released solver field applies only to sampling")
    if borrowed_samples and not (compact_phase=="sampling" or dense_phase=="sampling"):
        raise ValueError("borrowed sample maps apply only to sampling")
    if owned_dense_graph and (not rolling or compact_phase is not None or not 0<bones<=16 or levels<=1):
        raise ValueError('owned dense graph requires rolling adaptive dense fields')
    if cached_samples and not (compact_phase=="sampling" and borrowed_samples and solver_field_released):
        raise ValueError("cached sample lease requires released compact sampling")
    if stream_direct and not (dense_phase == "sampling" and borrowed_samples and solver_field_released):
        raise ValueError("direct sampling requires a released public dense field")
    if packed_samples and not (compact_phase == "sampling" and borrowed_samples and solver_field_released):
        raise ValueError("packed sampling requires a released public compact field")
    retained=0 if retained_owner_bytes is None else operator.index(retained_owner_bytes)
    if retained<0:raise ValueError('negative retained owner storage')
    def ownership(result):
        if retained_owner_bytes is not None:
            # This phase allowance excludes preexisting cache allocations.
            # Report actual additional shared owners separately; do not silently
            # reinterpret its established policy as an aggregate/RSS cap.
            result['retained_owner_bytes']=retained
            result['estimated_with_retained_bytes']=result['estimated_bytes']+retained
        return result
    entries=points*bones
    if compact_phase is not None:
        return ownership(compact_estimate(points,bones,vertices=vertices,edges=edges,
            sample_k=sample_k,top_k=top_k,phase=compact_phase,
            stage_points=stage_points,stage_edges=stage_edges,
            prior_bytes=prior_bytes,shared_graph=shared_graph,solver_field_released=solver_field_released,borrowed_samples=borrowed_samples,cached_samples=cached_samples,packed_samples=packed_samples))
    # Rolling public solves discard completed level fields/maps; at most two
    # graph generations, one sparse state and five dense equivalents coexist.
    # Private diagnostic callers retain the original all-level estimate.
    sparse_levels=(min(1,levels-1) if rolling else max(0,levels-1)) if bones>16 else 0
    parts=dict(
        # Retained stage fields plus seeds, Jacobi/checkpoint, interpolation,
        # cleanup Vec rows and owned array conversion. A coarse-only solve
        # has no retained/interpolated fine-level states; four temporaries
        # cover seeds plus the three Jacobi/checkpoint buffers.
        dense_fields=4*entries*(5 if rolling else levels+(4 if levels==1 else 8)),
        seed_masks=3*entries,
        # CSR graphs, Python argument staging, coarsened graphs and sparse maps.
        graph_and_maps=edges*(64*(min(2,levels) if rolling else levels)+16*sparse_levels),
        sparse_states=points*sparse_levels*16*18,
        point_state=points*256,
        sampling=vertices*(8*bones+8*sample_k+128*min(top_k,bones,32)),
        fixed_reserve=1024*1024)
    if owned_dense_graph:
        # Array-returning coarse graphs/maps and direct sequence cleanup avoid
        # whole-edge Python scalar lists. Count two CSR generations plus native
        # extraction, row/diagonal/mapping and conservative vector growth.
        parts['graph_and_maps']=24*edges+32*points
    if rolling:
        # Public final reduction owns contiguous output and bounds dense sampled
        # rows. Count map publication/copies, orphan arrays/adjacency, and one
        # block's native row transport instead of whole-mesh Python float lists.
        top=min(top_k,bones,32)
        block=min(vertices,1024,max(1,65536//max(bones,1)))
        vertex_output=vertices*((16 if borrowed_samples else 24)*sample_k+16*top+128)+16*edges+block*(8*bones+16*top+64)+max(0,prior_bytes-8*vertices*sample_k)
        map_build=sum(sample_map_storage(vertices,sample_k,prior_bytes=prior_bytes,borrowed_samples=borrowed_samples).values())
        parts['sampling']=max(map_build,vertex_output)
        if stream_direct:
            from .sample_batches import direct_block_rows
            rows = min(vertices, direct_block_rows(sample_k, bones))
            # Public direct queries retain only reduced outputs and one map
            # block. Charge native conversion and dense reduction conservatively.
            block_storage = sum(sample_map_storage(rows, sample_k, borrowed_samples=True).values())
            parts['sampling'] = (prior_bytes + vertices * (16 * top_k + 128)
                                 + 16 * edges + block_storage
                                 + rows * (8 * bones + 16 * top_k + 64))
        # Completed solve scratch is released before vertex sampling. Sampling
        # retains only the final field, cleaned field and original graph; its
        # output/map allowance must not be added to expired Jacobi states.
        common=parts['seed_masks']+parts['point_state']+parts['fixed_reserve']
        phases=dict(solve=common+parts['dense_fields']+parts['graph_and_maps']+parts['sparse_states'],
                    sampling=common+(4 if solver_field_released else 8)*entries+(16 if owned_dense_graph else 64)*edges+parts['sampling'])
        # Each qualified public phase checks its own live storage. Cleanup
        # remains covered by the unchanged conservative solve allowance.
        raw=phases[dense_phase] if dense_phase is not None else max(phases.values())
        return ownership(dict(entries=entries,components=parts,phase_bytes=phases,estimated_bytes=(raw*5+3)//4))
    raw=sum(parts.values())
    return ownership(dict(entries=entries,components=parts,estimated_bytes=(raw*5+3)//4))

def require(points,bones,*,budget_bytes=None,**kwargs):
    budget=bind_phase_budget.get() if budget_bytes is None else operator.index(budget_bytes)
    if budget is not None and budget<0:raise ValueError('negative field memory budget')
    result=estimate(points,bones,**kwargs)
    if budget is not None and result['estimated_bytes']>budget:
        raise ValueError(f"field working-set research budget exceeded: {result['estimated_bytes']} bytes > {budget} bytes")
    return result

def require_grid(bounds, resolution, *, probe_multiplier=1, adaptive=False,
                 vertices=0, triangles=0, budget_bytes=None, geometry=None, limit_coarse_output=False):
    """Preflight finite ordinary grids before voxelization.

    Accounts conservatively for cell/raster/probe staging. Invalid/degenerate
    bounds remain owned by the existing native validation route. A bounded triangle-box bitmap may refine an over-budget estimate. Actual
    adaptive cluster and final point/edge resources are checked separately.
    """
    budget_bytes = bind_phase_budget.get() if budget_bytes is None else operator.index(budget_bytes)
    import numpy as np
    lo=np.asarray(bounds[0], 'f4');hi=np.asarray(bounds[1], 'f4')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        extent=hi-lo;cell=np.float32(np.max(extent)/np.float32(resolution))
        if not np.isfinite(extent).all() or not np.isfinite(cell) or cell<=0:
            return None
        def cells(size):
            dims=np.fmax(np.ceil(extent/size),np.float32(1))
            if not np.isfinite(dims).all():return None
            total=1
            for d in dims:total*=int(d)
            return total
        coarse=cells(cell)
        probes=cells(np.float32(cell/np.float32(probe_multiplier))) if adaptive else 0
    if coarse is None or probes is None:return None
    # Python/native cell records, raster marks, normals and temporary indices.
    raw=64*coarse+64*probes+vertices*32+triangles*64+1024*1024
    estimated=(raw*5+3)//4
    coverage=None
    if budget_bytes is None:
        # The caller already performs native grid admission. Do not manufacture
        # output/marker limits from a diagnostic peak estimate on public binds.
        return dict(coarse_cells=coarse,probe_cells=probes,estimated_bytes=estimated)
    if estimated>budget_bytes and adaptive and geometry is not None:
        coverage=surface_box_coverage(geometry,lo,hi,np.float32(cell/np.float32(probe_multiplier)))
        if coverage is not None:
            # Surface raster/BTree, Python coordinate/normal results, ndarray
            # conversion and refinement maps scale with occupied surface cells,
            # not the whole address grid. Face references remain separate.
            raw=64*coarse+512*coverage['occupied_upper_bound']+4*coverage['triangle_box_visits']+vertices*32+triangles*64+1024*1024
            estimated=(raw*5+3)//4
    if estimated > budget_bytes and adaptive and limit_coarse_output and geometry is not None:
        available = budget_bytes*4//5
        fixed = 16*coarse + 64*vertices + 512*triangles + 1024*1024
        if fixed < available:
            # Match the detector's f32 spacing for the combined resolution.
            probe_cell = np.float32(np.max(extent)/np.float32(resolution*probe_multiplier))
            coarse_surface = surface_box_coverage(geometry, lo, hi, cell, visits_only=True)
            fine_surface = surface_box_coverage(geometry, lo, hi, probe_cell, visits_only=True)
            if coarse_surface is not None and fine_surface is not None:
                for raster in (coarse_surface, fine_surface):
                    fixed += 512*raster['occupied_upper_bound'] + 16*raster['triangle_box_visits']
                remaining = available - fixed
                point_limit = remaining//1024
                if point_limit >= coarse_surface['occupied_upper_bound']:
                    return dict(coarse_cells=coarse, probe_cells=cells(probe_cell),
                                surface_coverage=fine_surface, reserved_bytes=fixed,
                                coarse_point_limit=point_limit, refinement_bytes=remaining,
                                estimated_bytes=(5*(fixed+remaining)+3)//4)
    if estimated>budget_bytes and adaptive and coverage is None and limit_coarse_output and geometry is not None:
        # If the byte-per-cell fallback was too large, the same triangle boxes
        # still admit exact union counting with one bit per addressable cell.
        probe_cell=np.float32(cell/np.float32(probe_multiplier))
        packed=surface_box_coverage(geometry,lo,hi,probe_cell,packed=True)
        if packed is not None:
            raw=64*coarse+512*packed['occupied_upper_bound']+4*packed['triangle_box_visits']+vertices*32+triangles*64+1024*1024
            estimated=(raw*5+3)//4
            coverage=packed
    # A coarse-only build owns one winding tile per group of cells and
    # variable-size surface/interior output. Both fills classify interiors.
    # Use this plan only when the native emitter enforces its point allowance.
    if estimated > budget_bytes and limit_coarse_output and not adaptive and geometry is not None:
        fixed = 16*coarse + 64*vertices + 512*triangles + 1024*1024
        available = budget_bytes*4//5
        if fixed < available and coarse <= 16_000_000:
            coverage = surface_box_coverage(geometry, lo, hi, cell)
            if coverage is not None:
                fixed += 512*coverage['occupied_upper_bound'] + 16*coverage['triangle_box_visits']
                point_limit = (available - fixed)//1024
                if point_limit >= coverage['occupied_upper_bound']:
                    return dict(coarse_cells=coarse, probe_cells=0, surface_coverage=coverage,
                                reserved_bytes=fixed, coarse_point_limit=point_limit,
                                estimated_bytes=(5*(fixed+1024*point_limit)+3)//4)
    if estimated>budget_bytes:
        raise ValueError(f'grid working-set research budget exceeded: {estimated} bytes > {budget_bytes} bytes')
    result=dict(coarse_cells=coarse,probe_cells=probes,estimated_bytes=estimated)
    if coverage is not None:result['surface_coverage']=coverage
    return result


def surface_box_coverage(geometry,lo,hi,cell,*,visits_only=False,packed=False):
    """Conservative union of raster triangle AABBs, without doing rasterization.

    Small bounded scratch only. Refuse the optimization for invalid geometry or
    large address/work domains; callers retain their original conservative gate.
    No source mesh mutation, cache publication, or native validation override.
    """
    import numpy as np
    v,f=geometry
    v=np.asarray(v);f=np.asarray(f)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu':return None
    # Source vertices are already caller-owned. Bound the actual referenced
    # triangle scratch instead of refusing a mesh based on its vertex count.
    if len(f)>1000000:return None
    if not np.isfinite(cell) or cell<=0:return None
    dims=np.fmax(np.ceil((hi-lo)/cell),np.float32(1))
    if not np.isfinite(dims).all():return None
    total=1
    for x in dims:total*=int(x)
    if total > (0xffffffff if visits_only or packed else 16_000_000):return None
    if packed:
        nx,ny,nz=map(int,dims)
        byte_width=(nx+7)//8
        if byte_width*ny*nz>16_000_000:return None
    dims=dims.astype('i8');top=lo+dims.astype('f4')*cell
    def boxes(start):
        faces=f[start:start+4096]
        if np.any(faces<0) or np.any(faces>=len(v)):return None
        tri=np.asarray(v[faces],'f4')
        if not np.isfinite(tri).all():return None
        low=tri.min(axis=1);high=tri.max(axis=1);del tri
        keep=np.all((high>=lo)&(low<=top),axis=1)
        low=low[keep];high=high[keep]
        with np.errstate(over='ignore',invalid='ignore'):
            low=np.clip(np.floor((low-lo)/cell),0,dims-1)
            high=np.clip(np.floor((high-lo)/cell),0,dims-1)
        if not np.isfinite(low).all() or not np.isfinite(high).all():return None
        return low.astype('i8'),high.astype('i8')
    # Validate/count before bitmap allocation. Recompute the bounded boxes on
    # the second pass rather than retaining an all-triangle temporary.
    visits=0
    for start in range(0,len(f),4096):
        pair=boxes(start)
        if pair is None:return None
        low,high=pair
        visits+=sum(map(int,np.prod(high-low+1,axis=1)))
        if visits>16_000_000:return None
    if visits_only:
        # Every occupied cell is visited by at least one triangle. Summing
        # clipped boxes can overcount overlaps but uses no grid-sized bitmap.
        return dict(occupied_upper_bound=min(total, visits), triangle_box_visits=visits, scratch_grid_bytes=0)
    if packed:
        cells=np.zeros((nz,ny,byte_width),'u1')
        for start in range(0,len(f),4096):
            pair=boxes(start)
            if pair is None:return None
            for (x0,y0,z0),(x1,y1,z1) in zip(*pair):
                region=cells[z0:z1+1,y0:y1+1]
                left,right=int(x0)//8,int(x1)//8
                first=(255<<(int(x0)%8))&255
                last=(1<<((int(x1)%8)+1))-1
                if left==right:
                    region[:,:,left] |= first&last
                else:
                    region[:,:,left] |= first
                    region[:,:,left+1:right]=255
                    region[:,:,right] |= last
        bits=np.array([i.bit_count() for i in range(256)],'u1')
        flat=cells.reshape(-1)
        occupied=sum(int(bits[flat[i:i+65536]].sum()) for i in range(0,len(flat),65536))
        return dict(occupied_upper_bound=occupied,triangle_box_visits=visits,scratch_grid_bytes=cells.nbytes)
    cells=np.zeros(tuple(dims[::-1]),'u1')
    for start in range(0,len(f),4096):
        pair=boxes(start)
        if pair is None:return None
        low,high=pair
        for a,b in zip(low,high):cells[a[2]:b[2]+1,a[1]:b[1]+1,a[0]:b[0]+1]=1
    return dict(occupied_upper_bound=int(np.count_nonzero(cells)),triangle_box_visits=visits,scratch_grid_bytes=total)



def compact_estimate(points,bones,*,phase,vertices=0,edges=0,sample_k=0,top_k=4,
                     stage_points=0,stage_edges=0,prior_bytes=0,shared_graph=False,solver_field_released=False,borrowed_samples=False,cached_samples=False,packed_samples=False):
    """Allocation/lifetime ledger for public compact refinement only.

    256 bytes/point reserves source/cloud copies, seed vectors, mapping, sample
    grid/occupancy and Python scalar staging. Original CSR is retained throughout.
    Refined field payload is 16*(u16+f32) per point, not a dense bone matrix.
    This is a research allocation admission estimate, not a process RSS ceiling.
    """
    vals=[operator.index(x) for x in (points,bones,vertices,edges,sample_k,top_k,stage_points,stage_edges,prior_bytes)]
    if any(x<0 for x in vals):raise ValueError('negative compact phase dimension')
    if solver_field_released and phase != "sampling":
        raise ValueError("released solver field applies only to compact sampling")
    if borrowed_samples and phase != "sampling":
        raise ValueError("borrowed sample maps apply only to sampling")
    n,b,v,e,k,top,s,se,prior=vals
    if cached_samples and not (phase=="sampling" and borrowed_samples and solver_field_released and (packed_samples or prior==8*v*k)):
        raise ValueError("cached sample lease size or phase differs")
    common=256*n+8*(n+e)+4+1024*1024
    parts=dict(point_and_resource_reserve=256*n,retained_graph=8*(n+e)+4,fixed_reserve=1024*1024)
    if phase=='seeding':
        extra=n*b+3*min(n*b,4_000_000)+32*n
    elif phase=='coarsening':
        # Native helper additionally checks actual region degrees before edge
        # allocations. Its budget is whatever remains in the per-call allowance.
        extra=prior+224*n
    elif phase in ('dense','sparse'):
        sg=0 if shared_graph else 8*(s+se)+4
        if phase=='dense':extra=prior+sg+16*s*b+48*s+4*n
        else:
            # Groups, initial + four Jacobi/checkpoint float arrays, 4-column
            # interlevel map, stage representatives/seed vectors and edge slots.
            extra=prior+sg+352*s+32*s+(12*s if shared_graph else 32*s)+16*se
    elif phase=='cleanup':
        # Retain final compact field, mutable values copy, one decoded column,
        # union/find/mass scratch and cleaned output, conservatively overlapped.
        extra=96*n+64*n+112*n+6*n*min(top,b,32)
        # Wider rigs extend the formerly fixed 256-bit presence map and decoded
        # row scratch; count this incremental memory before native allocation.
        extra+=((n+255)//256)*max(0,(b+63)//64-4)*8+16*max(0,b-256)
    elif phase == 'sampling' and packed_samples:
        from .sample_batches import direct_block_rows
        rows = direct_block_rows(k, 1)
        block = min(rows, v)
        # Reserve the output, prior lease, codec workspace, native query rows,
        # decoded buffers and per-block headers before any cache growth.
        extra = (6*n*min(top,b,32) + v*(16*top+128) + 16*e + prior
                 + block*(8*b+16*top+64+32*k) + 1024*1024
                 + 512*((v+rows-1)//rows))
        if not cached_samples:
            extra += sum(sample_map_storage(block,k,borrowed_samples=True).values())
    elif phase=='sampling':
        # Public bind drops every diagnostic field owner after cleanup. Only
        # that route may exclude the expired 16-column (u16, f32) field.
        # Diagnostic/default callers still retain and account for both fields.
        retained=(0 if solver_field_released else 96*n)+6*n*min(top,b,32)
        # Separate map conversion (native rows + Python boxes + ndarray copies)
        # from sampled dense vertex rows, sparse reduction and orphan repair.
        # Native nested rows are converted only one bounded query block at a
        # time. Full cache/output arrays remain owned and counted. Preserve the
        # prior cached map until successful replacement, including k changes.
        block=min(v,1024,max(1,65536//max(k,1)))
        map_build=sum(sample_map_storage(v,k,prior_bytes=prior,borrowed_samples=borrowed_samples).values())
        sample_block=min(v,1024,max(1,65536//max(b,1)))
        # Internal readers and the newly published cache share one pair of
        # arrays. Only an actually existing prior generation can add a second
        # map; max(new_map, prior) invented that owner on cold/width-growth
        # misses. Keep prior storage conservatively through output reduction.
        map_owners=8*v*k+prior if borrowed_samples else 16*v*k+max(8*v*k,prior)
        vertex_output=v*(16*min(top,b,32)+128)+16*e+sample_block*(8*b+16*min(top,b,32)+64)+map_owners
        if cached_samples:
            # Strong owners are already acquired from the exact immutable cache
            # key. No map build, native row conversion or map copy happens.
            vertex_output=v*(16*min(top,b,32)+128)+16*e+sample_block*(8*b+16*min(top,b,32)+64)+prior
            extra=retained+vertex_output
        else:extra=retained+max(map_build,vertex_output)
    else:raise ValueError('unknown compact allocation phase')
    parts['phase_allocations']=extra
    raw=common+extra
    return dict(entries=n*b,components=parts,phase=phase,phase_bytes={phase:raw},estimated_bytes=(raw*5+3)//4)


def coarsen_budget(points,bones,*,edges,prior_bytes=0,retained_owner_bytes=None):
    # Use the same per-call budget and safety margin as require(); the native
    # helper bounds both its pre-mapping and per-region/output allocations.
    row=require(points,bones,edges=edges,prior_bytes=prior_bytes,compact_phase='coarsening',retained_owner_bytes=retained_owner_bytes)
    common=sum(row['components'][k] for k in ('point_and_resource_reserve','retained_graph','fixed_reserve'))
    budget = bind_phase_budget.get()
    # Native coarsening accepts a size_t scratch ceiling; the addressable-size
    # sentinel leaves checked native arithmetic/allocation authoritative.
    if budget is None:
        return sys.maxsize
    return max(0,budget*4//5-common-prior_bytes)
