"""One job per isolated process. No Blender imports or original solver fallback."""
from pathlib import Path
import json,os,sys,time,traceback

def main():
    import numpy as np
    from beyond_skinning import public_api as api
    job=Path(sys.argv[1]);request=json.loads((job/'binding.json').read_text())
    settings=json.loads(Path(__file__).with_name('defaults.json').read_text())
    settings['use_gpu']=request['backend']=='gpu'
    settings['coarse_res']=request.get('resolution',92)
    if not 16<=settings['coarse_res']<=128:raise ValueError('Invalid binding resolution')
    with np.load(job/'binding-input.npz',allow_pickle=False) as data:
        v,f,e,h,t,p,d=(data[k] for k in ('vertices','triangles','edges','heads','tails','parents','deform'))
    from beyond_skinning.field_memory import physical_host_bytes
    api.set_bind_env(physical_host_bytes(), settings['coarse_res'], settings['max_depth'])
    names=request['bone_names'];started=time.monotonic()
    ids,parents=api.deform_subset(p,d,np.linalg.norm(t-h,axis=1).astype('f4'),names)
    ids=np.asarray(ids,'i8');parents=np.asarray(parents,'i8')
    if not len(ids):raise ValueError('No deforming bones found')
    result=api.bind(api.BindSession(),v,f,e,[names[i] for i in ids],h[ids],t[ids],parents,names,p,d,np.stack([h,t],1),**settings)
    w=result['vertex_weights'];b=result['vertex_bones']
    # Native rows address the selected deform skeleton, not full-rig indices.
    if w.shape!=b.shape or len(w)!=len(v) or not np.isfinite(w).all() or np.any(w<0) or np.any(b>=len(ids)) or np.any((b<0)&(w>0)):
        raise RuntimeError('Solver returned invalid influence rows')
    if not np.allclose(w.sum(axis=1),1.,rtol=0,atol=1e-5):raise RuntimeError('Solver returned unweighted vertices')
    np.savez(job/'binding-output.npz',weights=w,bones=b)
    report=dict(engine=request['brand'],backend=request['backend'],seconds=time.monotonic()-started,bone_names=[names[i] for i in ids],vertices=len(v),settings=settings,preserve_volume=False,source_weights_used=False)
    if settings['use_gpu']:report['adapter']=api.gpu_limits()[:2]
    (job/'binding-result.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    # Only the worker owns this sys.path entry. Blender never imports the runtime.
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    try:main()
    except BaseException as error:
        traceback.print_exc()
        if len(sys.argv)>1:(Path(sys.argv[1])/'binding-error.txt').write_text(str(error)[-4000:])
        raise
