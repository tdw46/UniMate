"""Small independent default mouth binding composition; no original-module fallback."""
import numpy as np
import sys
from .adaptive_bind import bind_mesh
from .mouth_project import project,distance,F

def mesh_graph(vertices,edges,*,bounded=True):
 v=np.asarray(vertices,'f4')
 if not bounded:
  from .face_memory import require
  require(len(v),2*len(edges))
 e=np.asarray(edges,'i8')
 native=getattr(sys.modules.get(__package__+'._bvt_solver'),'mouth_mesh_graph',None)
 if native is not None and v.ndim==2 and v.shape[1]==3 and e.ndim==2 and e.shape[1]==2 and v.flags.aligned and e.flags.aligned:
  r,c,w=native(v,e)
  return dict(rows=r,columns=c,weights=w)
 adj=[set() for _ in v]
 for a,b in e:
  if not 0<=a<len(v) or not 0<=b<len(v):raise ValueError('edge outside vertices')
  if a!=b:adj[a].add(int(b));adj[b].add(int(a))
 r=[0];c=[];w=[]
 for i,near in enumerate(adj):
  for j in sorted(near):c.append(j);w.append(float(distance(v[i],v[j])))
  r.append(len(c))
 return dict(rows=r,columns=c,weights=w)

def parameters(tails):
 t=np.asarray(tails,'f4');span=F(np.ptp(t,axis=0).max())
 if not np.isfinite(span) or span<=0:raise ValueError('degenerate mouth ring')
 values=[F(span*F(x)) for x in (.15,.21,1.,.07,.21,.35)]
 # Native corner spread scales the already-rounded ordinary spread.
 return values+[F(values[5]*F(.8)),F(2.2),3]

def smooth_face_weights(out,roles,edges,factor,passes,hash_seed=None,random_states=None,*,bounded=True):
 from .face_weight_smoothing import accumulate_ordered,finalize_tables
 # Native owner filter (0x462c): skin roles only, not rigid eye/teeth or tongue/ear.
 from .rig_map_public import SKIN
 owners=[i for i,role,_ in roles if role in SKIN or role=='eyelid_corner']
 mask=np.any((out['weights']>0)&(out['bones']>=0)&np.isin(out['bones'],owners),axis=1)
 tables=accumulate_ordered(out['weights'],out['bones'],edges,mask,factor,passes,hash_seed,random_states=random_states,bounded=bounded)
 out['weights'],out['bones']=finalize_tables(out['weights'],out['bones'],tables,mask,min(4,out['weights'].shape[1]))
 out['weight_smooth_mask']=mask

def bind(runtime,v,f,e,names,h,t,p,*,resolution=16,max_depth=0,iterations=4,full_rig_context=None,role_override=None,face_smooth=0,split_allowed=None,face_split=False,face_weight_smooth=0.,face_weight_smooth_iters=0,weight_hash_seed=None,weight_random_states=None,prepared_void_boxes=None,face_voids=False,session=None,face_refine_eyes=False,finger_isolate=False,finger_isolate_thresh=.15,finger_isolate_smooth=2,on_stage=None,solver_options=None,progress=None):
 bounded=not bool(solver_options and solver_options.get('public_scalar_policy'))
 if (bounded and not 0<=face_weight_smooth<=1) or not isinstance(face_weight_smooth_iters,(int,np.integer)) or face_weight_smooth_iters<0 or (bounded and face_weight_smooth_iters>16):raise ValueError('face weight tuning research bounds')
 if not isinstance(face_smooth,(int,np.integer)) or face_smooth<0 or (bounded and face_smooth>16):raise ValueError('face smoothing research bounds')
 solve_options=dict(sample_k=4,progress=progress)
 if solver_options is not None:
  allowed={'retained_torso_context','use_gpu','retain_diagnostics','public_scalar_policy','winding_threshold','bucket_key','bucket_modes','spine_shapes','trunk_seed_edge','trunk_seed_lat','trunk_seed_lat_pow','multiplier','top_k','sample_k','kernel','volume','bleed','fill_interior','ghost_cull','edge_cull','iteration_policy','spine_from_names','check_coverage','min_coverage_frac'}
  if not isinstance(solver_options,dict) or set(solver_options)-allowed:raise ValueError('unsupported facial solve option')
  solve_options.update(solver_options)
 h=np.asarray(h,'f4');t=np.asarray(t,'f4');p=np.asarray(p,'i8');v=np.asarray(v,'f4')
 from .rig_mapping import validate
 validate(names,p,None)
 if h.shape!=(len(names),3) or t.shape!=h.shape or (bounded and (not np.isfinite(h).all() or not np.isfinite(t).all())):raise ValueError('invalid mouth skeleton')
 roles,_=runtime.detect_face(names,p.tolist(),None,-1) if role_override is None else role_override
 lips=sorted((i,1 if role=='lip_top' else 2 if role=='lip_bot' else 0) for i,role,_ in roles if role in ('lip_top','lip_bot','lip_corner'))
 if role_override is not None:
  order={i:j for j,(i,_,_) in enumerate(role_override[0])}
  lips.sort(key=lambda row:order[row[0]])
 if full_rig_context is not None:
  # Preserve full-skeleton mouth traversal independently of solve-column order.
  order={}
  for i,name in enumerate(full_rig_context[0]):order.setdefault(name,i)
  lips.sort(key=lambda row:order[names[row[0]]])
 if bounded and len(lips)>16:raise ValueError('mouth ring outside qualified research size')
 if not {1,2} <= {flag for _,flag in lips}:
  if split_allowed is not None:raise ValueError('explicit partial-mouth split masks remain unqualified')
  base=bind_mesh(runtime,v,f,e,h,t,p,resolution=resolution,max_depth=max_depth,iterations=iterations,**solve_options,rig_names=names,full_rig_context=full_rig_context,role_override=role_override,prepared_void_boxes=prepared_void_boxes,face_voids=face_voids,session=session,face_refine_eyes=face_refine_eyes,finger_isolate=finger_isolate,finger_isolate_thresh=finger_isolate_thresh,finger_isolate_smooth=finger_isolate_smooth,on_stage=on_stage)
  out=dict(weights=base['vertex_weights'].copy(),bones=base['vertex_bones'].copy(),base=base,lip_ids=[i for i,_ in lips],mouth_route='voxel-partial',complete_native_bind=False)
  # Native post-face work needs a retained mouth surface graph. A standalone
  # weight-smoothing request cannot build it, nor can a scope without lip bones.
  if face_smooth and lips:
   graph=mesh_graph(v,e,bounded=bounded);out['graph']=graph
   if face_smooth:
    from .face_tuning import mouth_mask,smooth_surface
    mask=mouth_mask(v,h,t,out['lip_ids'],graph,allow_nonfinite_vertices=True,bounded=bounded)
    out['weights'],out['bones']=smooth_surface(out['weights'],out['bones'],graph,mask,passes=face_smooth,bounded=bounded)
    out['smooth_mask']=mask
   if face_weight_smooth>0 and face_weight_smooth_iters>0:
    smooth_face_weights(out,roles,e,face_weight_smooth,face_weight_smooth_iters,weight_hash_seed,weight_random_states,bounded=bounded)
  return out
 lipids=[i for i,_ in lips];keep=[i for i in range(len(names)) if i not in lipids];lookup={old:new for new,old in enumerate(keep)}
 parents=np.array([lookup.get(int(p[i]),-1) for i in keep],'i8')
 base=bind_mesh(runtime,v,f,e,h[keep],t[keep],parents,resolution=resolution,max_depth=max_depth,iterations=iterations,**solve_options,rig_names=[names[i] for i in keep],full_rig_context=full_rig_context,role_override=([(lookup[i],role,side) for i,role,side in role_override[0] if i in lookup],role_override[1]) if role_override is not None else None,prepared_void_boxes=prepared_void_boxes,face_voids=face_voids,session=session,face_refine_eyes=face_refine_eyes,finger_isolate=finger_isolate,finger_isolate_thresh=finger_isolate_thresh,finger_isolate_smooth=finger_isolate_smooth,on_stage=on_stage)
 ids=base['vertex_bones'].copy();valid=ids>=0;ids[valid]=np.asarray(keep,'i4')[ids[valid]]
 graph=mesh_graph(v,e,bounded=bounded);params=parameters(h[lipids])
 split_categories=None
 if face_split or split_allowed is not None:
  from .face_split import mouth_record,categories
  record=mouth_record(runtime,names,h,t,p)
  if face_split and split_allowed is None:
   from .split_ownership import mouth_inputs,select
   eyes,shape,threshold=mouth_inputs(runtime,names,h,t,p,record)
   split_allowed=select(v,e,eyes,shape,threshold,allow_nonfinite_vertices=True,bounded=bounded)
  if split_allowed is None:split_allowed=np.ones(len(v),'u1')
  split_categories=categories(runtime,v,f,h,t,graph,record,split_allowed,allow_nonfinite_vertices=True,bounded=bounded)
 out=project(v,h,lipids,[flag for _,flag in lips],params,graph,base['vertex_weights'],ids,categories=split_categories,allow_nonfinite_vertices=True,bounded=bounded)
 if split_categories is not None:out.update(split_categories=split_categories,split_allowed=np.asarray(split_allowed,'u1'))
 if face_smooth:
  from .face_tuning import mouth_mask,smooth_surface
  mask=mouth_mask(v,h,t,lipids,graph,allow_nonfinite_vertices=True,bounded=bounded)
  out['weights'],out['bones']=smooth_surface(out['weights'],out['bones'],graph,mask,passes=face_smooth,bounded=bounded)
  out['smooth_mask']=mask
 if face_weight_smooth>0 and face_weight_smooth_iters>0:
  smooth_face_weights(out,roles,e,face_weight_smooth,face_weight_smooth_iters,weight_hash_seed,weight_random_states,bounded=bounded)
 if split_categories is not None:
  from .face_split import constrain_weights
  out['weights'],out['bones']=constrain_weights(v,h,record,split_categories,out['weights'],out['bones'])
 out.update(base=base,params=list(map(float,params)),graph=graph,lip_ids=lipids,complete_native_bind=False)
 return out
