"""Independent public-signature CPU bridge. Research only; not full API parity.

Regular-shape CPU, bounded named-body and named-face CPU routes are supported.
No source-module fallback. Unsupported modes reject before changing the session.
"""
from pathlib import Path
import importlib.util
import importlib.machinery
import sys,os,hashlib,operator,time
import numpy as np
from .session import ResearchBindSession
from .adaptive_bind import bind_mesh

class UnsupportedBindOption(NotImplementedError):
    """Valid original mode not yet supported by the independent bridge."""

_RUNTIME=None

def _runtime():
    from .loader import runtime
    return runtime()

probe_multiplier = _runtime().probe_multiplier
auto_bone_mid_profile = _runtime().auto_bone_mid_profile
morton_order = _runtime().morton_order
scoped_reskin = _runtime().scoped_reskin
rest_space_keep = _runtime().rest_space_keep
core_chain_mask = _runtime().core_chain_mask
deform_subset = _runtime().deform_subset
join_meshes = _runtime().join_meshes
BindSession = _runtime().BindSession
set_bind_env = _runtime().set_bind_env

def _unsupported(name):raise UnsupportedBindOption(f'{name}: outside the qualified public CPU route')

def _array(value,name,dtype,shape,*,finite=True):
    if not isinstance(value,np.ndarray) or value.dtype!=np.dtype(dtype):_unsupported(name+' dtype')
    if value.ndim!=len(shape) or any(n is not None and value.shape[i]!=n for i,n in enumerate(shape)):_unsupported(name+' shape')
    if not value.flags.c_contiguous or (finite and not np.isfinite(value).all()):_unsupported(name+' layout/finite values')
    return value

def _record_columns(array, width):
    """Native row extraction reads the first width values of every record."""
    if array.shape[1] == width:
        return array
    if not len(array):
        return np.empty((0,width),dtype=array.dtype)
    if array.shape[1] < width:
        raise _runtime().PanicException('ndarray: index out of bounds')
    return np.ascontiguousarray(array[:,:width])


def _bind_impl(*args,**kwargs):
    from .field_memory import bind_phase_budget
    # Public admission follows the original per-grid host hint and native size
    # checks. Estimated scratch is telemetry, not a fraction-of-physical-RAM
    # refusal. Private diagnostics can still supply explicit bounded budgets.
    token = bind_phase_budget.set(None)
    try:return _bind_impl_inner(*args,**kwargs)
    except ValueError as exc:
        if str(exc).startswith('Bbox extent must be strictly positive, got '):
            raise ValueError('[invalid_grid] '+str(exc)) from None
        raise
    finally:
        bind_phase_budget.reset(token)


def _bind_impl_inner(session,verts,faces,edges,bone_names,heads,tails,parent_idx,full_names,full_parent_idx,full_deform,full_segments,coarse_res,top_k,max_depth,refine_multiplier,fill_interior,jacobi_iters,auto_iters,auto_iters_cap,stop_moving,stop_moving_fine,quality,sample_k,fmg_iters,refine_iters_ratio,edge_cull,bleed_threshold,bucket_shape,winding_threshold,ghost_cull,sample_kernel,sample_volume,finger_isolate,finger_isolate_thresh,finger_isolate_smooth,spine_shapes,spine_from_names,trunk_seed_edge,trunk_seed_lat,trunk_seed_lat_pow,use_gpu,check_coverage,min_coverage_frac,progress=None,void_boxes=None,face=False,face_voids=False,face_refine_eyes=True,face_split=False,face_smooth=0,face_weight_smooth=0.0,face_weight_smooth_iters=0,meshes=None,viz=None,viz_bone=None,chain_ends=None,*,_owner):
    if type(session) is not BindSession:raise TypeError('session must be an independent BindSession')
    if viz not in (None,'cloud','field','seeds'):_runtime().invalid_viz(viz)
    # Native typed extraction and cancellation have already completed. These
    # semantic checks precede cache access and bounded research policy checks.
    if edges is not None:
        edges=_record_columns(edges,2)
    heads=_record_columns(heads,3)
    tails=_record_columns(tails,3)
    if full_segments.shape[1:]!=(2,3):
        if not len(full_segments):
            full_segments=np.empty((0,2,3),dtype=full_segments.dtype)
        elif full_segments.shape[1]<2 or full_segments.shape[2]<3:
            raise _runtime().PanicException('ndarray: index out of bounds')
        else:
            full_segments=np.ascontiguousarray(full_segments[:,:2,:3])
    n=len(bone_names)
    if spine_from_names and spine_shapes is not None and len(spine_shapes)<3:
        # Native named-torso preparation appends two defaults, then reads
        # all five options before grid validation or capsule-off dispatch.
        torso_check=_runtime().detect_spine(full_names,full_parent_idx,full_deform)
        if torso_check['verified']:
            count=len(spine_shapes)+2
            raise _runtime().PanicException(f'index out of bounds: the len is {count} but the index is {count}')
    if chain_ends is not None and len(chain_ends) not in (0,n):
        raise ValueError('chain_ends must have one flag per bone')
    if ((meshes is None and any(x is None for x in (verts,faces,edges))) or
        (meshes is not None and any(x is not None for x in (verts,faces,edges)))):
        raise ValueError('pass either verts, faces and edges, or meshes (and None for the three)')
    if meshes == []:
        raise ValueError('[invalid_mesh] empty vertex array')
    joined=None
    if meshes is not None:
        joined=_join_inputs(meshes)
        verts,faces,edges=(joined[key] for key in ('verts','faces','edges'))
    from .bind_mesh_validation import validate_mesh
    validate_mesh(verts,faces)
    if len(heads)!=n or len(tails)!=n or len(parent_idx)!=n:
        raise ValueError('[invalid_armature] bone arrays must share length')
    if not 8<=coarse_res<=2048:
        raise ValueError(f'[invalid_grid] coarse_res out of range [8, 2048]: {coarse_res}')
    if refine_multiplier==0:
        raise ValueError('[invalid_mesh] resolution, refine_multiplier and probe_l4_multiplier must be > 0')
    # Validate option admission before cached resources or callbacks can change.
    # Explicit GPU requests preserve native NoAdapter failure rather than silently changing solver.
    if viz_bone is not None and not isinstance(viz_bone,str):_unsupported('viz_bone type')
    prepared_void_boxes=None
    if void_boxes is not None:
        # The compiled public entry has already extracted native u32/i32
        # records. Public calls must not inherit diagnostic box-count limits.
        from .face_voids import public_boxes
        prepared_void_boxes=public_boxes(void_boxes,bounded=False)
    from .bucket_config import parse as parse_bucket
    bucket_key,bucket_modes=parse_bucket(bucket_shape)
    bucket_options=dict(bucket_key=bucket_key,bucket_modes=bucket_modes,public_scalar_policy=True,winding_threshold=winding_threshold)
    if use_gpu:bucket_options['use_gpu']=True
    if not face:
        face_voids=False;face_split=False;face_smooth=0
    if not finger_isolate:
        finger_isolate_thresh=.15;finger_isolate_smooth=0
    if not face or not face_weight_smooth>0 or face_weight_smooth_iters==0:
        face_weight_smooth=0.;face_weight_smooth_iters=0
    # These fractions govern ordered comparisons; NaN disables the moving
    # fraction stop, equivalent to zero with the independent max-delta check.
    stop_moving=float(np.fmin(1.,np.fmax(0.,stop_moving)))
    stop_moving_fine=float(np.fmin(1.,np.fmax(0.,stop_moving_fine)))
    refine_iters_ratio=max(1,refine_iters_ratio)
    # Beyond the overlap sphere all graph edges have already been rejected.
    edge_cull=0. if np.isnan(edge_cull) or edge_cull<=0 else min(edge_cull,1.)
    if not isinstance(face_smooth,int) or not 0<=face_smooth<=4294967295:_unsupported('face_smooth')
    # Native quality text does not replace the explicit CPU stopping fractions.
    # Native accepts arbitrary quality text; explicit stopping settings remain authoritative.
    kernels={sample_kernel:0 if sample_kernel.isascii() and sample_kernel.lower()=='invd' else 3}
    for name,value in [('face',face),('face_voids',face_voids),('face_split',face_split),('spine_from_names',spine_from_names),('fill_interior',fill_interior),('auto_iters',auto_iters),('ghost_cull',ghost_cull),('sample_volume',sample_volume),('finger_isolate',finger_isolate),('check_coverage',check_coverage),('face_refine_eyes',face_refine_eyes)]:
        if type(value) is not bool:_unsupported(name+' type')
    ints={}
    for name,value,lo,hi in [('coarse_res',coarse_res,8,2048),('top_k',max(1,top_k),1,np.iinfo(np.uintp).max),('max_depth',max_depth,0,4294967295),('refine_multiplier',refine_multiplier,1,4294967295),('jacobi_iters',jacobi_iters,0,4294967295),('auto_iters_cap',auto_iters_cap,0,4294967295),('sample_k',sample_k,0,np.iinfo(np.uintp).max),('refine_iters_ratio',refine_iters_ratio,1,4294967295),('finger_isolate_smooth',finger_isolate_smooth,0,4294967295),('face_weight_smooth_iters',face_weight_smooth_iters,0,4294967295)]:
        try:x=operator.index(value)
        except TypeError:_unsupported(name+' integer')
        if not lo<=x<=hi:_unsupported(name+' range')
        ints[name]=x
    if fmg_iters is not None and (not isinstance(fmg_iters,int) or not 0<=fmg_iters<=4294967295):_unsupported('fmg_iters')
    # Match the public f32 extraction before deciding whether smoothing is active.
    face_weight_smooth=float(np.float32(face_weight_smooth))
    n=len(bone_names)
    if chain_ends is not None:
        if not isinstance(chain_ends,np.ndarray) or chain_ends.dtype!=np.dtype('bool') or chain_ends.ndim!=1:
            raise TypeError(f"argument 'chain_ends': '{type(chain_ends).__name__}' object cannot be converted to 'PyArray<T, D>'")
        if len(chain_ends) not in (0,n):raise ValueError('chain_ends must have one flag per bone')
        if not chain_ends.flags.c_contiguous:_unsupported('chain_ends layout')

    if not isinstance(bone_names,(list,tuple)) or not all(isinstance(s,str) for s in bone_names):_unsupported('bone_names')
    v=_array(verts,'verts','f4',(None,3),finite=False);f=_array(faces,'faces','i4',(None,3));e=edges
    # Sampling/output allocations are admitted by the phase memory ledger.
    h=heads;t=tails;p=_array(parent_idx,'parent_idx','i8',(n,))
    from .bind_context import resolve
    torso,roles,face_family,context=resolve(_runtime(),bone_names,h,t,p,full_names,full_parent_idx,full_deform,full_segments,face,_unsupported)
    spine_from_names=spine_from_names and torso["verified"]
    if chain_ends is not None and chain_ends.any():
        from .chain_ends import no_hit_tails
        t=no_hit_tails(_runtime(),v,f,h,t,chain_ends)
        if context is not None:
            ctx_ids={name:i for i,name in enumerate(context[0])}
            for i,name in enumerate(bone_names):context[2][ctx_ids[name]]=t[i]
    from .rig_mapping import validate
    validate(bone_names,p,None)
    policy=dict(public_scalar_policy=True,jacobi_iters=ints['jacobi_iters'],auto_iters=auto_iters,auto_iters_cap=ints['auto_iters_cap'],fmg_iters=fmg_iters,refine_iters_ratio=ints['refine_iters_ratio'],stop_moving=stop_moving,stop_moving_fine=stop_moving_fine)
    from .bind_policy import schedule
    schedule(1,**policy) # Validate scalars without allocating inactive requested levels.
    times={}
    def on_stage(name):times[name]=time.perf_counter()
    if viz is not None:
        visual_labels=None
        visual_skeleton=(v,h,t)
        if face and {'lip_top','lip_bot'} <= {role for _,role,_ in roles}:
            lipids={i for i,role,_ in roles if role in ('lip_top','lip_bot','lip_corner')}
            keep=[i for i in range(n) if i not in lipids];lookup={old:new for new,old in enumerate(keep)}
            visual_labels=keep
            roles=[(lookup[i],role,side) for i,role,side in roles if i in lookup]
            bone_names=[bone_names[i] for i in keep];h=h[keep];t=t[keep];p=np.array([lookup.get(int(p[i]),-1) for i in keep],'i8')
        selected=bone_names.index(viz_bone) if viz_bone in bone_names else None
        return bind_mesh(_runtime(),v,f,e,h,t,p,resolution=ints['coarse_res'],max_depth=ints['max_depth'],multiplier=ints['refine_multiplier'],top_k=ints['top_k'],sample_k=ints['sample_k'],kernel=kernels[sample_kernel],volume=sample_volume,bleed=bleed_threshold,fill_interior=fill_interior,ghost_cull=ghost_cull,edge_cull=edge_cull,iteration_policy=policy,session=_owner,on_stage=on_stage,progress=progress,prepared_void_boxes=prepared_void_boxes,**bucket_options,body_names=bone_names if bucket_modes is not None and not face else None,torso_names=bone_names if bucket_modes is None else None,rig_names=bone_names if face else None,full_rig_context=context if face else None,role_override=(roles,face_family) if face and context is None else None,face_voids=face_voids,face_refine_eyes=face and face_refine_eyes,spine_shapes=spine_shapes,trunk_seed_edge=trunk_seed_edge,trunk_seed_lat=trunk_seed_lat,trunk_seed_lat_pow=trunk_seed_lat_pow,spine_from_names=spine_from_names,retained_torso_context=spine_from_names and torso["verified"],visualization=viz,viz_bone=selected,viz_label_map=visual_labels,viz_skeleton=visual_skeleton)
    if face:
        solve_options=dict(retain_diagnostics=False,**bucket_options,spine_shapes=spine_shapes,trunk_seed_edge=trunk_seed_edge,trunk_seed_lat=trunk_seed_lat,trunk_seed_lat_pow=trunk_seed_lat_pow,multiplier=ints['refine_multiplier'],top_k=ints['top_k'],sample_k=ints['sample_k'],kernel=kernels[sample_kernel],volume=sample_volume,bleed=bleed_threshold,fill_interior=fill_interior,ghost_cull=ghost_cull,edge_cull=edge_cull,iteration_policy=policy,spine_from_names=spine_from_names,retained_torso_context=spine_from_names and torso["verified"],check_coverage=check_coverage,min_coverage_frac=min_coverage_frac)
        from .mouth_bind import bind as mouth_bind
        face_result=mouth_bind(_runtime(),v,f,e,bone_names,h,t,p,resolution=ints['coarse_res'],max_depth=ints['max_depth'],iterations=ints['jacobi_iters'],face_smooth=face_smooth,face_split=face_split,prepared_void_boxes=prepared_void_boxes,face_weight_smooth=face_weight_smooth,face_weight_smooth_iters=ints['face_weight_smooth_iters'],face_voids=face_voids,face_refine_eyes=face_refine_eyes,session=_owner,on_stage=on_stage,progress=progress,full_rig_context=context,role_override=(roles,face_family),finger_isolate=finger_isolate,finger_isolate_thresh=finger_isolate_thresh,finger_isolate_smooth=ints['finger_isolate_smooth'],solver_options=solve_options)
        result=face_result['base'];result['vertex_weights']=face_result['weights'];result['vertex_bones']=face_result['bones']
    else:
     result=bind_mesh(_runtime(),v,f,e,h,t,p,retain_diagnostics=False,resolution=ints['coarse_res'],max_depth=ints['max_depth'],multiplier=ints['refine_multiplier'],top_k=ints['top_k'],sample_k=ints['sample_k'],kernel=kernels[sample_kernel],volume=sample_volume,bleed=bleed_threshold,fill_interior=fill_interior,ghost_cull=ghost_cull,edge_cull=edge_cull,iteration_policy=policy,session=_owner,on_stage=on_stage,progress=progress,prepared_void_boxes=prepared_void_boxes,**bucket_options,body_names=bone_names if bucket_modes is not None else None,torso_names=bone_names if bucket_modes is None else None,spine_shapes=spine_shapes,trunk_seed_edge=trunk_seed_edge,trunk_seed_lat=trunk_seed_lat,trunk_seed_lat_pow=trunk_seed_lat_pow,spine_from_names=spine_from_names,retained_torso_context=spine_from_names and torso["verified"],check_coverage=check_coverage,min_coverage_frac=min_coverage_frac,finger_isolate=finger_isolate,finger_isolate_thresh=finger_isolate_thresh,finger_isolate_smooth=ints['finger_isolate_smooth'],finger_names=bone_names)
    end=time.perf_counter();stages=result['stages'];fine=stages[-1];coarse=stages[0]
    from .rig_face_shapes import reported_roles
    report_roles=list(dict.fromkeys(reported_roles(roles,h,t))) if face else roles
    public=dict(vertex_weights=result['vertex_weights'],vertex_bones=result['vertex_bones'],n_points=len(result['points']),ghost_culled=result['ghost_culled'] if ghost_cull else -1,iters_coarse=coarse['iterations'] if len(stages)>1 else 0,iters_fine=fine['iterations'],iters_cap=fine['policy']['budget'],iters_coarse_cap=coarse['policy']['budget'],rig_family=torso['label'],pelvis_bone=torso['pelvis_name'],spine_bones=torso['spine_names'],face_bones=[(bone_names[i],role,side) for i,role,side in report_roles],face_family=face_family,time_detect_ms=(times['seeding']-times['voxelizing'])*1000,time_raster_ms=(times['solving']-times['seeding'])*1000,time_solve_ms=(times['finishing']-times['solving'])*1000,time_sample_ms=(end-times['finishing'])*1000,cloud_hit=result['cloud_hit'],bones_hit=result['bones_hit'])

    if joined is not None:
        public['meshes'],public['meshes_fill_source']=_mesh_outputs(joined['base_to_body'],result['vertex_weights'],result['vertex_bones'],result['orphan_sources'])
        public['time_sample_ms']=(time.perf_counter()-times['finishing'])*1000
    return public


def _join_inputs(meshes):
    if not isinstance(meshes,(tuple,list)) or not len(meshes):_unsupported('meshes count')
    from .bind_mesh_validation import validate_part
    for index,record in enumerate(meshes):
        if not isinstance(record,(tuple,list)) or len(record)!=5:_unsupported('mesh record')
        v,f,b,m,first=record
        validate_part(index,v,f,b,m)
    return _runtime().join_meshes(meshes)


def _mesh_outputs(mappings,weights,bones,sources):
    outputs=[];fills=[]
    for mapping in mappings:
        if mapping is None:outputs.append(None);fills.append(None);continue
        mapping=np.asarray(mapping,'i8');inverse={}
        for i,vertex in enumerate(mapping):inverse[int(vertex)]=i # Last base alias owns copied donors.
        fill=np.empty(len(mapping),'i8')
        for i,vertex in enumerate(mapping):
            donor=int(sources[vertex])
            fill[i]=i if donor==vertex else -1 if donor<0 else inverse.get(donor,-2)
        outputs.append((weights[mapping].copy(),bones[mapping].copy()));fills.append(fill)
    return outputs,fills


def plan_face_parts(names,parent_idx,deform,segments,meshes,keep=None):
    """Independent bounded whole-part/region planning in the original signature.

    Returns ownership and base-vertex skins; does not apply them to a main bind.
    """
    n=len(names)
    if not isinstance(names,(list,tuple)) or not all(isinstance(s,str) for s in names) or not 1<=n<=256 or len(set(names))!=n:_unsupported('part names')
    p=_array(parent_idx,'parent_idx','i8',(n,));d=_array(deform,'deform','bool',(n,));s=_array(segments,'segments','f4',(n,2,3))
    from .rig_mapping import validate
    validate(names,p,d)
    if keep is not None and (not isinstance(keep,(list,tuple)) or not all(isinstance(x,str) for x in keep)):_unsupported('part keep')
    if not isinstance(meshes,(list,tuple)) or len(meshes)>256:_unsupported('part meshes')
    from .mesh_correspondence import plan_inputs
    result=plan_inputs(_runtime(),meshes,list(names),s[:,0].copy(),s[:,1].copy(),p,keep,deform=d)
    return {key:result[key] for key in ('parts','regions','part_bones')}

# Re-export the compiled no-argument controls directly. Python forwarding
# wrappers would change their signatures and invalid-call TypeError contract.
request_cancel = _runtime().request_cancel
clear_cancel = _runtime().clear_cancel
cancel_message = _runtime().cancel_message

# Typed public extraction precedes cache ownership and Python research admission.
bind = _runtime().bind

# Standalone typed public detector; internal bounded helpers remain separate.
detect_fingers_by_name = _runtime().detect_fingers_by_name

detect_fingers_geometric = _runtime().detect_fingers_geometric

detect_fingers = _runtime().detect_fingers

detect_spine = _runtime().detect_spine

# Public NumPy signature; the legacy Vec-based bind helper remains isolated.
detect_face = _runtime().detect_face_public

full_skeleton = _runtime().full_skeleton

# Unified mapper; unrecovered geometry routes raise explicitly (research-only).
rig_map = _runtime().rig_map

# Typed, owned standalone face planner; private bounded helpers remain available.
plan_face_parts = _runtime().plan_face_parts

# Standalone working-frame helper with the original typed public contract.
bind_grid = _runtime().bind_grid

# Standalone compiled smoothing, with owned outputs and native random maps.
smooth_weights = _runtime().smooth_weights

# Stateless public side-name helpers.
flip_side = _runtime().flip_side
blender_flips = _runtime().blender_flips

# Standalone hierarchy-aware visualization hue helper.
bone_hues = _runtime().bone_hues

# Standalone public anatomical name classifier.
is_thigh_name = _runtime().is_thigh_name

# Frozen-target compatibility metadata; independent research identity is separate.
gpu_limits = _runtime().gpu_limits
gpu_budget_check = _runtime().gpu_budget_check
version = _runtime().version
tuning_values = _runtime().tuning_values

face_seam_labels = _runtime().face_seam_labels

detect_refine_faceprobe = _runtime().detect_refine_faceprobe
