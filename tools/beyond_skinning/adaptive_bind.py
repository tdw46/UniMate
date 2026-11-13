"""Independent bounded adaptive CPU orchestration. Research only.

No original-module imports. Regular seeds, bounded automatic hierarchy/torso
assignment, and default eyelid/corner/brow/cheek/jaw routes are independently
composed. Verified named torso/head plus skull bands now compose through final
weights on bounded fresh controls. Full rig mapping, GPU/public dispatch and complete cache lifetime
qualification remain open. Bounded face/finger isolation composes here before
orphan repair; mouth rings and surface tuning compose through mouth_bind.
"""
import numpy as np
import copy
from .adaptive_sparse import refine
from .compact_field import CompactField


def _owned_dense_graph(runtime,bones,levels,array_transport):
    return bool(array_transport and 0<bones<=16 and len(np.unique(levels))>1
                and all(callable(getattr(runtime,name,None)) for name in
                        ('coarsen_level_array','build_interlevel_arrays','clean_dense_array')))


def solve_levels(runtime,points,levels,radii,graph,fixed,labels,values,bones,*,iterations=4,fine_iterations=1,iteration_policy=None,array_transport=False,use_gpu=False,gpu_host_field_bytes=None,retained_dense_seeds=False,retained_samples=None):
    retained_samples=retained_samples or (lambda:None)
    if use_gpu:
        from .gpu_solver import dense
        dense_solve=lambda *a,**kw:dense(runtime,*a,**kw)
    else:dense_solve=runtime.solve_dense_cpu
    compact_route=array_transport and bones>16 and len(np.unique(levels))>1
    owned_dense=_owned_dense_graph(runtime,bones,levels,array_transport)
    if array_transport:
        from .field_memory import require,coarsen_budget
        if not compact_route:require(len(points),bones,levels=len(np.unique(levels)),edges=len(graph['columns']),rolling=True,owned_dense_graph=owned_dense,retained_owner_bytes=retained_samples())
    elif len(points)*bones>1_000_000:raise ValueError('dense bone-field research budget exceeded')
    unique=np.unique(levels);stages=[];prior=None;previous=None
    policies=None
    if iteration_policy is not None:
        from .bind_policy import schedule
        policies=schedule(len(unique),sparse_refinement=bones>16,bone_count=bones,**iteration_policy)
    # Native diagnostics count selected logical allocations cumulatively,
    # including completed stages; this is neither RSS nor retained memory.
    accounted=gpu_host_field_bytes
    for index,level in enumerate(unique):
        policy=policies[index] if policies else dict(budget=iterations if index==0 else fine_iterations,stop=None)
        budget,stop=policy['budget'],policy['stop']
        prior_bytes=(prior.groups.nbytes+prior.values.nbytes if isinstance(prior,CompactField) else prior.nbytes if prior is not None else 0)
        if index<len(unique)-1:
            coarsen=runtime.coarsen_level_array if compact_route or owned_dense else runtime.coarsen_level_graph
            options=dict(scratch_budget=coarsen_budget(len(points),bones,edges=len(graph['columns']),prior_bytes=prior_bytes,retained_owner_bytes=retained_samples())) if compact_route or owned_dense else {}
            stage=coarsen(points,radii,float(radii[levels==level].max()),graph['rows'],graph['columns'],graph['weights'],fixed,labels,values,bones,None,int(level),**options)
        else:
            stage=dict(representatives=np.arange(len(points),dtype='u8'),mapping=np.arange(len(points),dtype='u4'),rows=graph['rows'],columns=graph['columns'],weights=graph['weights'],diagonal=graph['diagonal'],fixed=fixed,labels=labels,values=values)
        sg={key:stage[key] for key in ('rows','columns','weights')};sg['dinv']=stage['diagonal']
        if compact_route:
            require(len(points),bones,edges=len(graph['columns']),compact_phase='dense' if prior is None else 'sparse',stage_points=len(stage['fixed']),stage_edges=len(stage['columns']),prior_bytes=prior_bytes,shared_graph=index==len(unique)-1,retained_owner_bytes=retained_samples())
        if prior is None or bones<=16:
            seeds=np.zeros((len(stage['fixed']),bones),'f4');valid=stage['labels']!=0xffffffff;seeds[np.flatnonzero(valid),stage['labels'][valid]]=stage['values'][valid]
        if prior is None:
            gpu_options=dict(host_field_bytes=accounted) if use_gpu else {}
            field,done=dense_solve(sg['rows'],sg['columns'],sg['weights'],sg['dinv'],stage['fixed'],seeds,budget,stop=stop,**gpu_options)
            stage.update(field=field,iterations=done)
        else:
            interlevel=runtime.build_interlevel_arrays if compact_route or owned_dense else runtime.build_interlevel_map
            options=dict(scratch_budget=coarsen_budget(len(points),bones,edges=len(graph['columns']),prior_bytes=prior_bytes,retained_owner_bytes=retained_samples())) if compact_route or owned_dense else {}
            ids,w=interlevel(graph['rows'],graph['columns'],graph['weights'],previous['mapping'],stage['representatives'],**options)
            ids=np.asarray(ids,'u4');w=np.asarray(w,'f4');stage.update(sample_ids=ids,sample_weights=w)
            if bones>16:
                sparse=refine(runtime,sg,ids,w,prior,stage['fixed'],stage['labels'],iterations=budget,seed_values=stage['values'],stop=stop,array_transport=array_transport,use_gpu=use_gpu,host_field_bytes=accounted,retain_edge_map=not array_transport)
                if array_transport:field=CompactField(sparse['groups'],sparse['field'],bones)
                else:
                    field=np.zeros((len(stage['fixed']),bones),'f4');ii,jj=np.where(sparse['groups']!=65535);field[ii,sparse['groups'][ii,jj]]=sparse['field'][ii,jj]
                stage.update(sparse=sparse,field=field,iterations=sparse['iterations'])
            else:
                initial=runtime.interpolate_level_array(ids,w,prior) if array_transport else np.asarray(runtime.interpolate_level_field(ids,w,prior),'f4')
                # Native finest dense level retains custom/trunk fields and
                # dispatches CPU f32 even when the coarsened solve used GPU.
                if accounted is not None:accounted+=4*len(stage['fixed'])*bones
                refined_solve=runtime.solve_dense_cpu if retained_dense_seeds and index==len(unique)-1 else dense_solve
                gpu_options=dict(host_field_bytes=accounted) if use_gpu and refined_solve is dense_solve else {}
                field,done=refined_solve(sg['rows'],sg['columns'],sg['weights'],sg['dinv'],stage['fixed'],seeds,budget,initial,stop=stop,**gpu_options)
                stage.update(field=field,iterations=done)
        if accounted is not None:
            n=len(stage['fixed'])
            if index==0:accounted+=4*n*bones # dense returned field
            elif bones>16:
                accounted+=16*len(sg['columns']) # sparse edge map
                if index<len(unique)-1:accounted+=4*n*bones # intermediate expansion
            else:
                if use_gpu and not (retained_dense_seeds and index==len(unique)-1):accounted+=2*n*((bones+3)&~3)
                accounted+=4*n*bones
        stage['policy']=policy
        if array_transport:
            # Only the next level consumes the prior field and mapping. Public
            # output exposes iteration metadata, not diagnostic stage arrays.
            stage={key:stage[key] for key in ('mapping','field','iterations','policy')}
            if previous is not None:
                previous.pop('field',None);previous.pop('mapping',None)
            seeds=None;sg=None
            if index>0:
                ids=None;w=None
                # Dense interpolation belongs only to the completed solve.
                # Release it before the next coarsening/solve allocation.
                if bones<=16:initial=None
            if bones>16 and index>0:sparse=None
        prior=stage['field'];previous=stage;stages.append(stage)
    if array_transport:stages[-1].pop('mapping',None)
    return stages


def bind_mesh(runtime,verts,faces,edges,heads,tails,parents,*,resolution=16,max_depth=2,multiplier=2,iterations=4,fine_iterations=1,top_k=4,bleed=.05,sample_k=4,kernel=3,volume=True,session=None,rig_names=None,base_seed_modes=None,body_names=None,spine_shapes=None,spine_from_names=True,full_rig_context=None,role_override=None,fill_interior=True,ghost_cull=True,on_stage=None,edge_cull=.85,iteration_policy=None,check_coverage=False,min_coverage_frac=.01,progress=None,progress_estimates=None,prepared_void_boxes=None,face_voids=False,face_refine_eyes=False,finger_isolate=False,finger_isolate_thresh=.15,finger_isolate_smooth=2,finger_names=None,torso_names=None,trunk_seed_edge=0.,trunk_seed_lat=1.,trunk_seed_lat_pow=1.,public_scalar_policy=False,winding_threshold=.5,bucket_key=None,bucket_modes=(2,14,6,14,6),visualization=None,viz_bone=None,viz_label_map=None,viz_skeleton=None,retain_diagnostics=True,use_gpu=False,retained_torso_context=False):
    if visualization is not None and (visualization not in ('cloud','field','seeds') or session is None):
        raise ValueError('visualization requires a public research session and a qualified mode')
    finger_chains=None
    if finger_isolate:
        if not public_scalar_policy and (not np.isfinite(finger_isolate_thresh) or not 0<=finger_isolate_smooth<=8):
            raise ValueError('finger isolation options outside research bounds')
    # Stage durations are the recovered workload estimates, independent of cache hits.
    sink=None
    if progress is not None:
        from .progress_contract import ProgressSink
        sink=ProgressSink(progress,progress_estimates,triangles=len(faces),bones=len(heads))
    if face_refine_eyes and rig_names is None:raise ValueError("automatic eye refinement requires face names")
    if face_voids and rig_names is None:raise ValueError("automatic voids require face names")
    runtime.check_cancel()
    def stage(name,*,points=0):
        if sink is not None:sink.emit(name,points=points)
        if on_stage is not None:on_stage(name)
    from .adaptive_detector import detect
    from .bone_preparation import prepare
    v=np.ascontiguousarray(verts,'f4');f=np.ascontiguousarray(faces,'i4');h=np.ascontiguousarray(heads,'f4');t=np.ascontiguousarray(tails,'f4');b=len(h);top_k=min(top_k,b,32) if public_scalar_policy else min(top_k,b)
    iteration_limit=256 if iteration_policy is not None else 32
    if not 8<=resolution<=(2048 if public_scalar_policy else 24) or (not public_scalar_policy and not 1<=b<=256) or not 0<=iterations<=iteration_limit or not 0<=fine_iterations<=8:raise ValueError('adaptive research workload bound exceeded')
    if not public_scalar_policy and b>32 and ((max_depth>0 and (max_depth>2 or resolution>16)) or len(v)*b>8_000_000):raise ValueError('large-rig adaptive workload bound exceeded')
    if full_rig_context is not None and rig_names is None:raise ValueError("full context requires named solve columns")
    face=None;body=None;skull=None
    if body_names is not None:
        if rig_names is not None:raise ValueError('choose one body/face mapping route')
        if len(body_names)!=b:raise ValueError('body names must match bones')
        from .body_seed_modes import assign
        body=assign(body_names,parents,spine_shapes=spine_shapes,spine_from_names=spine_from_names,bucket_modes=bucket_modes)
        if any(m not in range(19) for m in body["modes"]):raise ValueError("body bind override outside qualified subset")
    if rig_names is not None:
        from .rig_face_shapes import construct, normalized_shapes
        explicit_base_modes=base_seed_modes is not None
        if len(rig_names)!=b:raise ValueError('face names must match bones')
        if base_seed_modes is None:
            from .body_seed_modes import assign
            mapped=assign(rig_names,parents,spine_shapes=spine_shapes,spine_from_names=spine_from_names,bucket_modes=bucket_modes)
            base_seed_modes=mapped['modes']
            if mapped['torso'] and mapped['torso']['verified']:
                body=mapped
                head_id=body['torso']['head']
                if bucket_modes is not None and spine_shapes is None and head_id>=0 and head_id in parents and (role_override is None or role_override[0]):base_seed_modes[head_id]=bucket_modes[1]
        if len(base_seed_modes)!=b or any(x not in range(19) for x in base_seed_modes):raise ValueError('bounded face integration currently supports qualified torso base seed modes')
        if full_rig_context is None:face=construct(runtime,rig_names,h,t,parents,role_override=role_override,allow_nonfinite=public_scalar_policy)
        else:
            from .rig_face_shapes import project_context
            face=project_context(runtime,full_rig_context,rig_names,h,t,allow_nonfinite=public_scalar_policy)
        if not public_scalar_policy and b>32 and face['roles']:raise ValueError('large-rig facial binding remains unqualified')
        original_heads=h.copy();original_tails=t.copy()
        skull_ids=[i for i,role,_ in face['roles'] if role=='skull']
        if skull_ids:
            # Seed-mode preferences do not disable semantic head/skull geometry.
            from .rig_mapping import detect_spine
            face_torso=body['torso'] if body is not None else detect_spine(rig_names,parents,None)
            head_id=face_torso['head'] if face_torso['verified'] else -1
            if head_id<0:
                # Recovered prepare-main fallback: all selected skull segments
                # must share one nonnegative solve parent. No common parent
                # means no band reassignment, not a rejected bind.
                common=int(parents[skull_ids[0]])
                if common>=0 and all(int(parents[i])==common for i in skull_ids):head_id=common
            if head_id>=0:
                from .skull_bands import construct as skull_construct
                jaw_ids=[i for i,role,_ in face['roles'] if role=='jaw']
                skull=dict(head=head_id,bands=skull_construct(h,t,head_id,skull_ids,jaw_ids[0] if jaw_ids else None))
    elif base_seed_modes is not None:raise ValueError('base seed modes require a rig-face construction request')
    face_vertices=v if face is not None else None
    grid=runtime.bind_grid(v,resolution,np.ascontiguousarray(edges,'i8'),np.stack([h,t],axis=1));scale=np.float32(grid['scale'])
    if abs(grid['scale']-1)>1e-6:v=v*scale;h=h*scale;t=t*scale
    void_groups=None
    if face_voids:
        from .face_voids import face_groups
        void_groups=face_groups(face["roles"],h,t)
    if face_voids:
        from .face_voids import plan_boxes
        lo=np.asarray(grid['bbox_min'],'f4');hi=np.asarray(grid['bbox_max'],'f4')
        cell=np.float32(np.max(hi-lo)/np.float32(resolution))
        prepared_void_boxes=list(prepared_void_boxes or [])+plan_boxes(void_groups,lo,hi,cell,level=max_depth,multiplier=multiplier,bounded=not public_scalar_policy)
        void_groups=None # Boxes are already planned for both session and direct paths.
    prepared_refine_boxes=None
    if face_refine_eyes:
        from .eye_refinement import plan as plan_eyes
        lo=np.asarray(grid['bbox_min'],'f4');hi=np.asarray(grid['bbox_max'],'f4')
        cell=np.float32(np.max(hi-lo)/np.float32(resolution))
        prepared_refine_boxes=plan_eyes(face['roles'],h,t,lo,hi,cell,level=max_depth,multiplier=multiplier,bounded=not public_scalar_policy)
    grid_memory_admission=None
    if public_scalar_policy:
        # Admission protects a new allocation, not reuse of an owned cloud.
        # Run after native geometry/RAM errors and the voxelizing callback;
        # session misses invoke it before constructing any grid storage.
        from .field_memory import require_grid
        def grid_memory_admission():
            return require_grid((grid['bbox_min'],grid['bbox_max']),resolution,
                         probe_multiplier=runtime.probe_multiplier(max_depth,multiplier),
                         adaptive=bool(max_depth),vertices=len(v),triangles=len(f),geometry=(v,f),
                         limit_coarse_output='max_output_points' in (getattr(runtime.coarse_cloud,'__text_signature__',None) or ''))
    cloud_hit=False;bone_hit=False
    stage("voxelizing")
    if session is None:
        runtime.check_cancel()
        cloud=detect(runtime,v,f,[grid['bbox_min'],grid['bbox_max']],resolution=resolution,max_depth=max_depth,multiplier=multiplier,probe_multiplier=runtime.probe_multiplier(max_depth,multiplier),interior_route="winding",fill_interior=fill_interior,ghost=ghost_cull,prepared_void_boxes=prepared_void_boxes,face_void_groups=void_groups,prepared_refine_boxes=prepared_refine_boxes,winding_threshold=winding_threshold,grid_memory_admission=grid_memory_admission)
        voxel_size=cloud['voxel_size']
        p=cloud['cloud_positions'];level=cloud['cloud_levels'];r=cloud['cloud_radii'];stage('seeding');prep=prepare(runtime,v,f,h,t,parents,p,cloud['voxel_size'])
        order=runtime.morton_order(p);p=p[order];level=level[order];r=r[order]
        from .cloud_resources import CloudResources
        resources=CloudResources(runtime,p,level,r)
    else:
        if session.runtime is not runtime or session.interior_route!="winding":raise ValueError('adaptive bind requires a matching winding-route research session')
        with session._lock:
            obj,_,summary=session.cloud_for(v,f,grid['bbox_min'],grid['bbox_max'],resolution,multiplier,max_depth,fill_interior,winding_threshold,ghost_cull,probe_multiplier=runtime.probe_multiplier(max_depth,multiplier),void_boxes=None if prepared_void_boxes is None else [(x[0],tuple(x[1:4]),tuple(x[4:7])) for x in prepared_void_boxes],refine_boxes=None if prepared_refine_boxes is None else [(x[0],tuple(x[1:4]),tuple(x[4:7])) for x in prepared_refine_boxes],_grid_memory_admission=grid_memory_admission)
            resources=session.resources()
        voxel_size=summary['voxel_size']
        stage('seeding')
        if visualization=='cloud':
            from .visualization import display
            return display('cloud',obj.positions(),obj.levels(),voxel_size,multiplier,scale,summary['hit'])
        try:prep=session.bones_for(h,t,parents,bucket_key if bucket_key is not None else ('{}' if body_names is not None or rig_names is not None else 'off'),_bucket=None if bucket_modes is None else bytes(bucket_modes))
        except RuntimeError as exc:
            if str(exc)=='bones_for before cloud_for: the session has no cloud':raise RuntimeError('[core] '+str(exc)) from None
            raise
        cloud_hit=summary['hit'];bone_hit=prep['hit'];p=obj.positions();level=obj.levels();r=obj.radii()
    if bucket_key is not None and (body_names is not None or rig_names is not None):
        from .body_seed_modes import assign
        mapped=assign(body_names if body_names is not None else rig_names,parents,spine_shapes=spine_shapes,spine_from_names=spine_from_names,bucket_modes=bucket_modes,prepared=(prep['heads'],prep['tails'],prep['r_mid']))
        if face is None:body=mapped
        else:
            base_seed_modes=mapped['modes']
            if mapped['torso'] and mapped['torso']['verified']:body=mapped
    if not len(p):runtime.fail_empty_cloud()
    retained_samples=getattr(resources,'live_sample_bytes',None)
    if not callable(retained_samples):retained_samples=resources.sample_cache_bytes
    compact_route=b>16 and len(np.unique(level))>1
    owned_dense=_owned_dense_graph(runtime,b,level,public_scalar_policy)
    direct_samples=public_scalar_policy and not compact_route
    sample_width=(min(max(sample_k,1),512) if compact_route else min(sample_k,len(p))) if public_scalar_policy else sample_k
    if public_scalar_policy:
        from .field_memory import require
        memory_args=dict(vertices=len(v),levels=len(np.unique(level)),sample_k=sample_width,top_k=top_k,rolling=True,compact_phase='seeding' if compact_route else None)
        if owned_dense:memory_args['owned_dense_graph']=True
        if not compact_route and not retain_diagnostics:memory_args['dense_phase']='solve'
        require(len(p),b,retained_owner_bytes=retained_samples(),**memory_args) # before graph, masks or dense fields
    elif len(p)*b>1_000_000:raise ValueError('dense bone-field research budget exceeded')
    graph_builds=resources.build_counts['graph']
    graph=resources.graph(cull=edge_cull,copy_output=False)
    graph_accounted_bytes=8*len(graph['columns']) if resources.build_counts['graph']!=graph_builds else 0
    if public_scalar_policy:require(len(p),b,edges=len(graph['columns']),retained_owner_bytes=retained_samples(),**memory_args)

    segment=np.asarray(t,'f4')-np.asarray(h,'f4')
    lengths=np.sqrt((segment[:,0]*segment[:,0]+segment[:,1]*segment[:,1])+segment[:,2]*segment[:,2])
    reference_length=float(np.min(lengths[lengths>np.float32(1e-9)])) if np.any(lengths>np.float32(1e-9)) else 0.
    endpoint_seed_modes=None
    if bucket_key is not None and bucket_modes is not None:
        # Endpoint-radius preparation precedes hierarchy selection: classify
        # each segment as a terminal, then apply semantic facial overrides.
        from .body_seed_modes import assign
        endpoint_seed_modes=assign([f'endpoint_{i}' for i in range(b)],[-1]*b,spine_from_names=False,bucket_modes=bucket_modes,prepared=(prep['heads'],prep['tails'],prep['r_mid']))['modes']
    # Named overrides participate in endpoint-radius preparation before
    # facial shapes, while 255 preserves the selected hierarchy mode.
    if endpoint_seed_modes is not None and body is not None:
        for i,mode in enumerate(body.get('override_modes',())):
            if mode!=255:endpoint_seed_modes[i]=0 if mode>18 else 5 if mode==10 else mode
    from .seed_batches import constrained_masks,profile_masks,custom_field,nearest_bones
    rr=[prep['radii'][:,0].tolist(),prep['radii'][:,1].tolist(),prep['r_mid'].tolist()]
    try:
        if body is not None and face is None:
            modes=body['modes'];nearest=nearest_bones(runtime,p,prep['heads'],prep['tails']) if public_scalar_policy else runtime.nearest_seed_bones(p,prep['heads'],prep['tails'])
            from .face_seed_modes import assign as endpoint_modes
            _,base_radii=endpoint_modes(dict(roles=[],shapes=[dict(kind='none')]*b),heads,tails,endpoint_seed_modes if endpoint_seed_modes is not None else modes,scale)
            masks=constrained_masks(runtime,p,prep['heads'],prep['tails'],*rr,modes,nearest,[0.]*b,base_radii=base_radii.tolist(),reference_length=reference_length,cell_size=float(voxel_size),body_scale=float(np.ptp(p,axis=0).max()),separation=1.5,axial_padding=.25,point_radii=r.tolist()) if public_scalar_policy else np.frombuffer(b''.join(runtime.constrained_seed_masks(p,prep['heads'],prep['tails'],*rr,modes,nearest,[0.]*b,base_radii=base_radii.tolist(),reference_length=reference_length,cell_size=float(voxel_size),body_scale=float(np.ptp(p,axis=0).max()),separation=1.5,axial_padding=.25,point_radii=r.tolist())),'u1').reshape(b,len(p)).copy()
        elif face is None or bucket_modes is None:
            masks=profile_masks(runtime,p,prep['heads'],prep['tails'],*rr) if public_scalar_policy else np.frombuffer(b''.join(runtime.seed_profile_masks(p,prep['heads'],prep['tails'],*rr)),'u1').reshape(b,len(p)).copy()
        else:
            shapes=normalized_shapes(face['shapes'],scale)
            from .face_seed_modes import assign as face_modes,sphere_radii as face_sphere_radii
            cached_endpoints=None
            if bone_hit and (not np.isfinite(prep['heads']).all() or not np.isfinite(prep['tails']).all()):
                cached_endpoints=prep['heads'],prep['tails']
            modes,base_radii=face_modes(face,original_heads,original_tails,base_seed_modes,scale,face_vertices,prepared_endpoints=cached_endpoints)
            if bucket_key is not None:
                _,base_radii=face_modes(face,original_heads,original_tails,endpoint_seed_modes,scale,face_vertices,prepared_endpoints=cached_endpoints)
            spheres=face_sphere_radii(face,original_heads,original_tails,base_seed_modes,scale,face_vertices).tolist()
            centers_kw={}
            if cached_endpoints is not None and any(spheres):
                # Prepared endpoints own the cached base radii. Explicit chin
                # spheres retain their centers from the current facial plan.
                centers=np.where((np.asarray(modes)==16)[:,None],prep['tails'],prep['heads']).copy()
                for i,radius in enumerate(spheres):
                    if radius>0:centers[i]=original_tails[i]*scale
                centers_kw=dict(centers=centers.tolist())
            nearest=nearest_bones(runtime,p,prep['heads'],prep['tails']) if public_scalar_policy else runtime.nearest_seed_bones(p,prep['heads'],prep['tails'])
            masks=constrained_masks(runtime,p,prep['heads'],prep['tails'],*rr,modes,nearest,spheres,**centers_kw,reference_length=reference_length,cell_size=float(voxel_size),point_radii=r.tolist(),custom_shapes=shapes,base_radii=base_radii.tolist(),separation=1.5,axial_padding=.25,body_scale=float(np.ptp(p,axis=0).max())) if public_scalar_policy else np.frombuffer(b''.join(runtime.constrained_seed_masks(p,prep['heads'],prep['tails'],*rr,modes,nearest,spheres,reference_length=reference_length,cell_size=float(voxel_size),point_radii=r.tolist(),custom_shapes=shapes,base_radii=base_radii.tolist(),separation=1.5,axial_padding=.25,body_scale=float(np.ptp(p,axis=0).max()))),'u1').reshape(b,len(p)).copy()
        step=np.float32(max(np.ptp(p,axis=0).max(),np.float32(1e-6))*np.float32(.01))
        if public_scalar_policy:
            from .seed_clearance import public_clearance
            clearance=public_clearance(runtime,p,prep['heads'],prep['tails'],step)
        else:
            clearance=np.frombuffer(runtime.seed_bone_clearance(p,prep['heads'],prep['tails'],step),'u1')
        if 0<clearance.sum()<b:masks[clearance!=0]=0
        # Even capsule-off uses the prepared mid radius for overlapping seed
        # priority. Terminal expansion makes bone length an inequivalent proxy.
        priority=prep['r_mid']
        if face is None:
            fixed,labels,values=runtime.regular_seed_field(masks,priority.tolist());fixed=np.frombuffer(fixed,'u1');labels=np.asarray(labels,'u4');values=np.asarray(values,'f4')
        else:
            if bucket_modes is None:
                shapes=normalized_shapes(face['shapes'],scale)
                modes=list(base_seed_modes);base_radii=np.zeros(b,'f4')
            priority=prep['r_mid']
            if public_scalar_policy:fixed,labels,touched_count,protected_count=custom_field(runtime,p,masks,priority.tolist(),shapes)
            else:
                if 'array_output' in (getattr(runtime.custom_seed_field,'__text_signature__','') or ''):
                    fixed,labels,touched,protected=runtime.custom_seed_field(p,masks,priority.tolist(),shapes,array_output=True)
                else:
                    _,_,(fixed,labels),touched,protected=runtime.custom_seed_field(p,masks,priority.tolist(),shapes)
                fixed=np.frombuffer(fixed,'u1');labels=np.asarray(labels,'u4');touched_count=int(np.frombuffer(touched,'u1').sum(dtype=np.uint64));protected_count=int(np.frombuffer(protected,'u1').sum(dtype=np.uint64))
            values=np.ones(len(p),'f4')
            from .face_corner_seeds import records, strengths
            soft_records=records(face,original_heads,scale)
            values=strengths(p,labels,soft_records)
            face.update(soft_records=soft_records.tolist(),shapes=shapes,modes=modes,touched=touched_count,protected=protected_count,explicit_base_modes=explicit_base_modes,base_radii=base_radii.tolist())
        if public_scalar_policy:masks=None # Seed assignments now own all needed values.
        trunk=body['trunk'] if body is not None else None
        if body is None and torso_names is not None and spine_from_names:
            from .body_seed_modes import assign
            trunk=assign(torso_names,parents,spine_from_names=True)['trunk']
        if trunk is not None:
            values,axis=runtime.seed_strengths(p,prep['heads'],prep['tails'],labels.tolist(),trunk,trunk_seed_edge,trunk_seed_lat,trunk_seed_lat_pow,bounded=not public_scalar_policy)
            values=np.asarray(values,'f4')
            if body is not None:body['fade_axis']=axis
        if skull is not None:
            from .skull_bands import reassign
            labels=reassign(p,labels,skull['head'],prep['heads'][skull['head']],prep['tails'][skull['head']],skull['bands'])
    except BaseException as exc:
        # Original bind announces solving before reporting seeding cancellation.
        # Block checkpoints may stop sooner, but retain that public event order.
        if type(exc).__name__=='PanicException' and str(exc)==runtime.cancel_message():
            stage('solving',points=len(p))
        raise
    stage("solving",points=len(p));runtime.check_cancel()
    if visualization=='seeds':
        from .visualization import display
        from .seed_preview import skeleton
        inv=np.float32(1)/scale
        return display('seeds',p,level,voxel_size,multiplier,scale,cloud_hit,label_map=viz_label_map,seed_labels=labels,skeleton=skeleton(*(viz_skeleton if viz_skeleton is not None else (v*inv,h*inv,t*inv))))
    # Native direct solves with retained trunk/custom seed fields select CPU f32,
    # even with GPU requested. Coarsened levels and sparse refinement use GPU;
    # the final dense level retains those seed fields and dispatches CPU f32.
    dense_seed_context=retained_torso_context or trunk is not None or (face is not None and any(s["kind"]!="none" for s in face["shapes"]))
    solver_gpu=use_gpu and not (dense_seed_context and len(np.unique(level))==1)
    if public_scalar_policy and b==0:
        if solver_gpu:
            raise runtime.PanicException('attempt to divide by zero')
        runtime.fail_empty_cloud()
    stages=solve_levels(runtime,p,level,r,graph,fixed,labels,values,b,iterations=iterations,fine_iterations=fine_iterations,iteration_policy=iteration_policy,array_transport=public_scalar_policy,use_gpu=solver_gpu,gpu_host_field_bytes=(len(p)*b if bucket_modes is None else 0)+graph_accounted_bytes,retained_dense_seeds=dense_seed_context,retained_samples=retained_samples)
    field=stages[-1]['field']
    if public_scalar_policy and not retain_diagnostics:
        # All seed assignments have been consumed; keep diagnostics only when
        # requested. Sampling/cleanup need the field and graph, not these arrays.
        fixed=None;labels=None;values=None
    if visualization!='field':stage('finishing')
    if isinstance(field,CompactField):
        require(len(p),b,edges=len(graph['columns']),top_k=top_k,compact_phase='cleanup',retained_owner_bytes=retained_samples())
        clean=field.clean(runtime,graph,r,bleed,top_k)
    elif owned_dense:
        if public_scalar_policy:require(len(p),b,edges=len(graph['columns']),retained_owner_bytes=retained_samples(),**memory_args)
        # PyO3 still owns its native Vec arguments, but no full Python scalar
        # list overlaps with those copies. Arithmetic stays in the same kernel.
        clean=runtime.clean_dense_array(graph['rows'],graph['columns'],r,field,bleed,top_k)
    else:clean=runtime.clean_dense_array(graph['rows'].tolist(),graph['columns'].tolist(),r.tolist(),field,bleed,top_k) if public_scalar_policy else np.asarray(runtime.clean_dense_field(graph['rows'].tolist(),graph['columns'].tolist(),r.tolist(),field,bleed,top_k,bounded=not public_scalar_policy),'f4')
    if visualization=='field':
        if isinstance(clean,CompactField):
            require(len(p),b,edges=len(graph['columns']),levels=len(np.unique(level)),rolling=True,retained_owner_bytes=retained_samples())
        from .visualization import display
        return display('field',p,level,voxel_size,multiplier,scale,cloud_hit,clean.dense() if isinstance(clean,CompactField) else clean,viz_bone,viz_label_map)
    if not retain_diagnostics:
        # Neither public output nor post-face work consumes solver diagnostics.
        # Drop the old field before sample maps and vertex fields coexist.
        for solved in stages:solved.pop('field',None)
        field=None
    borrow_samples=public_scalar_policy and not retain_diagnostics and getattr(resources,'borrowed_sample_maps',False)
    direct_reducer = getattr(resources, 'sample_direct_reduced', None)
    stream_direct = direct_samples and borrow_samples and sample_width > 0 and callable(direct_reducer)
    cached_samples=None
    lookup=getattr(resources,'cached_sample_map',None)
    if borrow_samples and compact_route and sample_width and callable(lookup):
        cached_samples=lookup(v,sample_width,kernel,volume,cache_k=sample_k)
    snapshot=getattr(resources,'sample_owner_snapshot',None)
    if callable(snapshot):
        memory_snapshot=snapshot(cached_samples)
        admitted_map_bytes=memory_snapshot['included_bytes'];other_map_bytes=memory_snapshot['other_bytes']
    else:
        from .sample_batches import sample_owner_bytes
        admitted_map_bytes=sample_owner_bytes(cached_samples) if cached_samples is not None else resources.sample_cache_bytes()
        other_map_bytes=max(0,retained_samples()-admitted_map_bytes)
    packed_budget = None
    if isinstance(clean,CompactField):
        from .sample_batches import PackedSampleMap
        from .field_memory import estimate, DEFAULT_BUDGET_BYTES, bind_phase_budget
        options = dict(edges=len(graph['columns']), vertices=len(v), sample_k=sample_width,
                       top_k=top_k, prior_bytes=admitted_map_bytes, compact_phase='sampling',
                       solver_field_released=not retain_diagnostics, borrowed_samples=borrow_samples,
                       cached_samples=cached_samples is not None, retained_owner_bytes=other_map_bytes)
        packed = isinstance(cached_samples, PackedSampleMap)
        if cached_samples is None and borrow_samples and callable(getattr(resources, 'sample_packed_reduced', None)):
            packed = estimate(len(p), b, **options)['estimated_bytes'] > DEFAULT_BUDGET_BYTES
        if packed:
            options['packed_samples'] = True
            if cached_samples is not None and cached_samples.replay is not None:
                options['cached_samples'] = False  # Reserve native query scratch for replayed blocks.
        admission = require(len(p), b, **options)
        if packed:
            packed_budget = DEFAULT_BUDGET_BYTES*4//5 - admission['phase_bytes']['sampling']
            if bind_phase_budget.get() is None:
                # Keep lossless packed/replayed maps as a memory optimization,
                # not a second fixed public bind cap. Exact query coordinates
                # always provide a finite replay representation if packing grows.
                packed_budget = max(v.nbytes, packed_budget)
    elif public_scalar_policy:
        require(len(p),b,edges=len(graph['columns']),vertices=len(v),levels=len(np.unique(level)),sample_k=sample_width,top_k=top_k,prior_bytes=admitted_map_bytes,retained_owner_bytes=other_map_bytes,rolling=True,dense_phase='sampling' if not retain_diagnostics else None,solver_field_released=not retain_diagnostics,borrowed_samples=borrow_samples,owned_dense_graph=owned_dense,stream_direct=stream_direct)
    resources.grid(copy_output=False);resources._sampling_occupancy()
    # Native bind starts with the sample-grid cell, not twice the largest
    # adaptive point radius. The initial candidate region affects filtering.
    if packed_budget is not None:
        sparse_weights, sparse_bones = resources.sample_packed_reduced(
            v, sample_width, sample_k, kernel, volume, clean, b, top_k, cached_samples, packed_budget, allow_query_blocks=True)
        clean = None
    elif stream_direct:
        sparse_weights, sparse_bones = direct_reducer(v, sample_width, sample_k, kernel, volume, clean, b, top_k)
        clean = None
    else:
        if cached_samples is not None:
            ids,w=cached_samples
        elif sample_width==0:
            ids=np.empty((len(v),0),'u4');w=np.empty((len(v),0),'f4')
        else:
            # Native sparse adaptive maps clamp search/storage to 1..512 but key
            # the retained slot by the original request. Dense fields sample
            # directly with the full request, including adaptive dense fields.
            ids,w=resources.sample_map(v,sample_width,kernel,volume,retain=compact_route,requested_k=sample_k if direct_samples else None,cache_k=sample_k if compact_route else None,**(dict(copy_output=False) if borrow_samples else {}))
        ids=np.asarray(ids,'u4');w=np.asarray(w,'f4')
        # Public bind allocates zero-filled map slots; the low-level sampler leaves
        # unused slots untouched (our low-level owned buffer uses u32::MAX).
        # Borrowed maps were normalized before immutable publication. Never write
        # through that alias; legacy/diagnostic writable maps keep normalization.
        if not borrow_samples:ids[ids==0xffffffff]=0
        if public_scalar_policy and not retain_diagnostics:
            from .sample_batches import sample_reduced
            sparse_weights,sparse_bones=sample_reduced(runtime,clean,ids,w,b,top_k)
            clean=None;ids=None;w=None
        else:
            sampled=clean.sample(runtime,ids,w) if isinstance(clean,CompactField) else runtime.accumulate_sample_array(ids,w,clean) if public_scalar_policy else np.asarray(runtime.accumulate_sample_fields(ids,w,clean),'f4')
            reduced=[runtime.reduce_top_weights(row.tolist(),top_k,0.) for row in sampled]
            if not retain_diagnostics:sampled=None;clean=None;ids=None;w=None
            sparse_weights=np.asarray([x[1] for x in reduced],'f4');sparse_bones=np.asarray([x[0] for x in reduced],'i4')
    finger_summary=None
    if finger_isolate:
        from .finger_geometry import detect_fingers
        finger_names=finger_names if finger_names is not None else rig_names if rig_names is not None else body_names if body_names is not None else ['']*b
        if len(finger_names)!=b:raise ValueError('finger names must match solve columns')
        finger_detector=runtime.detect_fingers if public_scalar_policy else detect_fingers
        finger_chains,finger_how=finger_detector(finger_names,prep['heads'],prep['tails'],parents,prep['cone_mask'])
    if finger_chains:
        # Keep isolation before orphan repair and the caller's mouth processing.
        from .finger_isolation import isolate_named
        sparse_weights,sparse_bones,finger_summary=isolate_named(v,f,prep['heads'],prep['tails'],finger_chains,sparse_weights,sparse_bones,finger_isolate_thresh,finger_isolate_smooth,bounded=not public_scalar_policy)
        finger_summary['detection']=finger_how
    repaired=runtime.fill_orphan_weights(sparse_weights,sparse_bones,np.ascontiguousarray(edges,'i8'),v,allow_nonfinite_positions=True)
    coverage_count=None
    if check_coverage:
        from .progress_contract import check_coverage as validate_coverage
        # Coverage measures the solve, before orphan filling borrows neighbors.
        coverage_count=validate_coverage(sparse_weights,min_coverage_frac)
    if not retain_diagnostics:
        return dict(ghost_culled=summary['ghost_culled'] if session is not None else sum(cloud['ghost_counts']),points=p,stages=stages,vertex_bones=repaired['bones'],vertex_weights=repaired['weights'],orphan_sources=np.asarray(repaired['sources'],'i8'),cloud_hit=cloud_hit,bones_hit=bone_hit)
    return dict(ghost_culled=summary['ghost_culled'] if session is not None else sum(cloud['ghost_counts']),finger_isolation=finger_summary,refine_boxes=prepared_refine_boxes,void_boxes=cloud["void_boxes"] if session is None else prepared_void_boxes,coverage_count=coverage_count,progress_callback_errors=sink.callback_errors if sink else 0,body=body,face=face,skull=skull,masks=masks,points=p,levels=level,radii=r,graph=copy.deepcopy(graph),fixed=fixed,labels=labels,values=values,stages=stages,field=field,clean=clean,ids=ids,sample_weights=w,sampled=sampled,vertex_bones=repaired['bones'],vertex_weights=repaired['weights'],orphan_sources=np.asarray(repaired['sources'],'i8'),cloud_hit=cloud_hit,bones_hit=bone_hit,resource_builds=resources.build_counts.copy(),complete_native_bind=False)
