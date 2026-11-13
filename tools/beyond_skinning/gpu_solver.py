"""Bounded fresh-input GPU transport for the recovered dense/sparse kernels.

Graph/seed preparation is independent; no captured original buffers are used.
The retained research runtime owns GPU submission, completion and cancellation.
"""
import numpy as np
import struct
from .gpu_memory import require

def device_admission(runtime):
 """Use the chosen cached device, never a second adapter's advertised limits."""
 query=getattr(runtime,'gpu_device_limits',None)
 if query is None:
  # Older hash-qualified libraries retain adapter probing and safe bounds.
  if runtime.gpu_limits() is None:raise runtime.PanicException('wgpu context init failed: NoAdapter')
  return {}
 try:buffer_bytes,storage_bytes=query()
 except ValueError as exc:
  if str(exc)=='No supported compute adapter':
   raise runtime.PanicException('wgpu context init failed: NoAdapter') from None
  raise
 return dict(buffer_bytes=buffer_bytes,storage_bytes=storage_bytes,device_working_bytes=None)

def bits(values):
 raw=np.packbits(np.asarray(values,dtype='u1').reshape(-1),bitorder='little').tobytes()
 return raw+b'\0'*((-len(raw))%4)

def half(values):
 raw=np.asarray(values,dtype='<f2').tobytes()
 return raw+b'\0'*((-len(raw))%4)

def execute(runtime,kind,graph,mask,seeds,initial,iterations,stop,seed_words=0,mask_words=0,host_field_bytes=None,packed_sparse=False):
 n,k=initial.shape
 require(kind,n,k,len(graph['columns']),iterations,automatic=stop is not None,borrowed_inputs=bool(getattr(runtime,'immutable_replay_inputs',False)),staged_readback=bool(getattr(runtime,'staged_auto_readback',False)),fixed_uniforms=bool(getattr(runtime,'generated_fixed_uniforms',False)),packed_sparse=packed_sparse,**device_admission(runtime))
 runtime.check_cancel()
 gx=min(32768,(n*(k//(4 if kind=='dense' else 2))+255)//256)
 arrays=[np.asarray(graph[x],dtype=d).tobytes() for x,d in [('rows','<u4'),('columns','<u4'),('weights','<f4'),('dinv','<f4')]]
 if len(graph['columns'])==0:
  from .gpu_errors import empty_columns
  if kind=='dense':empty_columns(runtime,host_field_bytes)
  else:empty_columns(runtime,host_field_bytes,kind=kind)
 if iterations==0:return initial.astype('f4'),0
 arrays += [mask,seeds]
 fields=(k,n,gx,0) if kind=='dense' else (n,k,gx,mask_words)
 uniform=struct.pack('<4If3I',*fields,1.,seed_words,0,0)
 from .gpu_errors import invoke
 if stop is not None:
  raw,done,_=invoke(runtime,host_field_bytes,runtime.replay_gpu_auto,kind,arrays+[uniform],half(initial),iterations,*stop)
 elif getattr(runtime,'generated_fixed_uniforms',False):
  raw=invoke(runtime,host_field_bytes,runtime.replay_gpu,kind,arrays+[uniform],half(initial),iterations,True);done=iterations
 else:
  uniforms=bytearray(iterations*256);omega=np.float32(1)
  for i in range(iterations):
   omega=np.float32(1) if i==0 else np.float32(1.9609766) if i==1 else np.float32(1)/np.float32(np.float32(np.float32(omega*np.float32(-.98010004))*np.float32(.25))+np.float32(1))
   struct.pack_into('<4If3I',uniforms,i*256,*fields,float(omega),seed_words,0,0)
  raw=invoke(runtime,host_field_bytes,runtime.replay_gpu,kind,arrays+[bytes(uniforms)],half(initial),iterations);done=iterations
 return np.frombuffer(raw,'<f2').astype('f4').reshape(n,k),done

def dense_inputs(fixed,seeds,initial,iterations):
 """Pack the original padded f32 rows in bounded batches; upload f16 once.

 Two-row alignment keeps packed bit offsets byte-aligned even at four slots.
 Preserve f32 summation over padded rows and f32-before-f16 rounding exactly.
 Zero-iteration calls still return the original f32 interpolation values.
 """
 n,b=seeds.shape;k=(b+3)&~3
 block=max(2,(65536//k//2)*2)
 active=(np.asarray(fixed)&1)!=0
 field=np.empty((n,k),'f4' if iterations==0 else '<f2')
 mask=bytearray(((n*k+31)//32)*4);seed=bytearray(len(mask))
 values=np.empty(n,'f4')
 for start in range(0,n,block):
  end=min(start+block,n);count=end-start;selected=active[start:end]
  sv=np.zeros((count,k),'f4');sv[:,:b]=seeds[start:end]
  values[start:end]=np.sum(sv,axis=1,dtype='f4')
  take=np.zeros((count,k),bool);take[:,:b]=selected[:,None]
  offset=start*k//8
  raw=np.packbits(take.reshape(-1),bitorder='little').tobytes()
  mask[offset:offset+len(raw)]=raw
  raw=np.packbits((sv!=0).reshape(-1),bitorder='little').tobytes()
  seed[offset:offset+len(raw)]=raw
  del take,raw
  upload=np.zeros((count,k),'f4')
  if initial is not None:
   upload[:,:b]=initial[start:end]
   upload[selected]=sv[selected]
  field[start:end]=upload
  del sv,upload
 seed_words=0
 if np.any(values[active]!=1):
  seed_words=len(seed)//4;seed.extend(half(values))
 return field,bytes(mask),bytes(seed),seed_words

def dense(runtime,rows,columns,weights,dinv,fixed,seeds,iterations,initial=None,stop=None,host_field_bytes=None):
 n,b=seeds.shape;k=(b+3)&~3
 # Native context acquisition precedes transport/uniform allocation.
 require('dense',n,k,len(columns),iterations,automatic=stop is not None,borrowed_inputs=bool(getattr(runtime,'immutable_replay_inputs',False)),staged_readback=bool(getattr(runtime,'staged_auto_readback',False)),fixed_uniforms=bool(getattr(runtime,'generated_fixed_uniforms',False)),**device_admission(runtime))
 graph=dict(rows=rows,columns=columns,weights=weights,dinv=dinv)
 field,mask,seed,seed_words=dense_inputs(fixed,seeds,initial,iterations)
 if initial is not None and host_field_bytes is not None:
  host_field_bytes+=2*n*k # original allocation-error accounting, not our storage
 out,done=execute(runtime,'dense',graph,mask,seed,field,iterations,stop,seed_words=seed_words,host_field_bytes=host_field_bytes)
 return out[:,:b].copy(),done

def sparse_inputs(groups,initial,fixed,seed_indices,seed_values):
 """Same f32 seed replacement and bit layout, bounded batches then f16 upload."""
 n,k=initial.shape;block=max(8,(65536//k//8)*8)
 field=np.empty((n,k),'<f2')
 active_bits=bytearray(((n+31)//32)*4);one_bits=bytearray(((n*k+31)//32)*4)
 vals=np.ones(n,'f4') if seed_values is None else np.asarray(seed_values,'f4')
 for start in range(0,n,block):
  end=min(n,start+block);active=(np.asarray(fixed[start:end])&1)!=0
  one=(groups[start:end]!=65535)&(groups[start:end]==np.asarray(seed_indices[start:end])[:,None])
  batch=np.asarray(initial[start:end],'f4').copy()
  batch[active]=(one*vals[start:end,None])[active]
  field[start:end]=batch
  raw=np.packbits(active,bitorder='little').tobytes();offset=start//8;active_bits[offset:offset+len(raw)]=raw
  raw=np.packbits(one.reshape(-1),bitorder='little').tobytes();offset=start*k//8;one_bits[offset:offset+len(raw)]=raw
 mask_words=len(active_bits)//4;seed_words=mask_words+len(one_bits)//4
 return field,bytes(active_bits)+bytes(one_bits)+half(vals),mask_words,seed_words

def sparse(runtime,graph,edge,groups,initial,fixed,seed_indices,iterations,seed_values=None,stop=None,host_field_bytes=None):
 n,k=initial.shape;assert k==16
 # Native context acquisition precedes transport/uniform allocation.
 packed=type(edge) is bytes and iterations>0 and bool(getattr(runtime,'immutable_replay_inputs',False))
 require('sparse',n,k,len(graph['columns']),iterations,automatic=stop is not None,borrowed_inputs=bool(getattr(runtime,'immutable_replay_inputs',False)),staged_readback=bool(getattr(runtime,'staged_auto_readback',False)),fixed_uniforms=bool(getattr(runtime,'generated_fixed_uniforms',False)),packed_sparse=packed,**device_admission(runtime))
 if packed:
  field,mask,mask_words,seed_words=sparse_inputs(groups,initial,fixed,seed_indices,seed_values)
 else:
  active=(np.asarray(fixed)&1)!=0
  one=(groups!=65535)&(groups==np.asarray(seed_indices)[:,None])
  vals=np.ones(n,'f4') if seed_values is None else np.asarray(seed_values,'f4')
  field=initial.copy();field[active]=(one*vals[:,None])[active]
  mask=bits(active);mask_words=len(mask)//4;mask+=bits(one);seed_words=len(mask)//4;mask+=half(vals)
 edge_bytes=edge if type(edge) is bytes else np.asarray(edge,'u1').tobytes()
 out,done=execute(runtime,'sparse',graph,mask,edge_bytes,field,iterations,stop,seed_words=seed_words,mask_words=mask_words,host_field_bytes=host_field_bytes,packed_sparse=packed)
 return out,done
