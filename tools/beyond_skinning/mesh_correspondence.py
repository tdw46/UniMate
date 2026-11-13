"""Bounded public face-plan input correspondence; no guessed nearest remapping."""
import numpy as np

def world_vertices(vertices,matrix,*,selection=False,bounded=True):
 v=np.asarray(vertices,'f4');m=np.asarray(matrix,'f8')
 if v.ndim!=2 or v.shape[1]!=3 or (bounded and not 0<len(v)<=4096) or m.shape!=(4,4) or (bounded and not np.isfinite(v).all()) or (bounded and not np.isfinite(m).all()):raise ValueError('invalid base/world geometry')
 # 0x1189a0 uses f32 matrix arithmetic for fallback selection; 0x1117dc
 # uses f64 arithmetic and rounds once for final base-space skin positions.
 dt=np.float32 if selection else np.float64;x=v.astype(dt);a=m.astype(dt)
 return np.stack([((x[:,0]*a[j,0]+x[:,1]*a[j,1])+x[:,2]*a[j,2])+a[j,3] for j in range(3)],1).astype('f4')

def plan_inputs(runtime,inputs,n,h,t,p,keep=None,*,deform=None,rig_context=None):
 from .full_face_parts import plan_parts
 bounded=rig_context is None
 selected=[];whole=[];skin=[]
 for name,evaluated,base,matrix,edges,triangles in inputs:
  v=np.asarray(evaluated,'f4');bv=np.asarray(base,'f4');e=np.asarray(edges,'i8');f=np.asarray(triangles,'i8')
  if v.ndim!=2 or v.shape[1]!=3 or (bounded and not 0<len(v)<=4096) or (bounded and not np.isfinite(v).all()):raise ValueError('invalid evaluated geometry')
  if not bounded:f=f.astype('u4').astype('i8')
  sw=world_vertices(bv,matrix,bounded=bounded)
  if bounded and (e.ndim!=2 or e.shape[1]!=2 or np.any(e<0) or np.any(e>=len(bv)) or f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(v))):raise ValueError('invalid base topology')
  if bounded and len(v)!=len(bv) and len(f):raise ValueError('changed-count triangles remain unqualified')
  cv=v if len(v)==len(bv) else world_vertices(bv,matrix,selection=True,bounded=bounded)
  selected.append((name,cv,f,e));whole.append((name,v,f,e));skin.append(sw)
 return plan_parts(runtime,selected,n,h,t,p,keep,skin_vertices=skin,whole_meshes=whole,deform=deform,rig_context=rig_context)

def bind_inputs(runtime,inputs,evaluated_faces,n,h,t,p,keep=None,*,max_depth=0):
 """Bounded default full binding with explicit evaluated solve triangles.

 Face planning and solve sampling have different coordinate contracts. Reuse
 the separately qualified join_meshes correspondence for body-weight sampling;
 apply independently computed part skins afterward, on base vertex IDs.
 """
 from .mouth_bind import bind as solve
 from .region_writeback import apply_regions
 if len(inputs)!=len(evaluated_faces):raise ValueError('missing evaluated topology')
 plan=plan_inputs(runtime,inputs,n,h,t,p,keep)
 joined=runtime.join_meshes([(np.asarray(v,'f4'),np.asarray(f,'i4'),np.asarray(base,'f4'),np.asarray(matrix,'f4'),False) for (_,v,base,matrix,_,_),f in zip(inputs,evaluated_faces)])
 ids=[i for i,name in enumerate(n) if name not in plan['part_bones']];lookup={old:new for new,old in enumerate(ids)}
 if any(p[i]>=0 and p[i] not in lookup for i in ids):raise ValueError('retained child of reserved part')
 pp=np.array([-1 if p[i]<0 else lookup[p[i]] for i in ids],'i8')
 body=solve(runtime,joined['verts'],joined['faces'],joined['edges'],[n[i] for i in ids],h[ids],t[ids],pp,max_depth=max_depth,full_rig_context=(n,h,t,p))
 body_bones=body['bones'].copy();valid=body_bones>=0;body_bones[valid]=np.array(ids,'i4')[body_bones[valid]]
 mapping=np.concatenate(joined['base_to_body']).astype('i8');weights=body['weights'][mapping].copy();bones=body_bones[mapping].copy();offsets=np.cumsum([0]+[len(m[2]) for m in inputs])
 for mi,label,score,skin in plan['parts']:
  if skin is None:raise ValueError('empty part skin outside bounds')
  sn,sw,si,_=skin;sl=slice(offsets[mi],offsets[mi+1]);weights[sl]=0;bones[sl]=-1
  weights[sl,:sw.shape[1]]=sw;bones[sl,:si.shape[1]]=np.array([n.index(name) for name in sn],'i4')[si]
 weights,bones=apply_regions(weights,bones,n,offsets,plan['regions'])
 return dict(weights=weights,bones=bones,base_to_body=joined['base_to_body'],plan=plan,complete_native_bind=False)
