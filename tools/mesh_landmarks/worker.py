"""Isolated install/download/inference entry point. Never imported into Blender."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import venv

PACKAGE=Path(__file__).resolve().parent
MANIFEST=json.loads((PACKAGE/'manifest.json').read_text())
# Preserve the existing addon's older Python range without source builds.
LEGACY_INTEL_MAC=sys.platform=='darwin' and platform.machine()=='x86_64'
REQUIREMENTS=[
    'torch==2.2.2' if LEGACY_INTEL_MAC else 'torch==2.6.0',
    'numpy==1.26.4' if LEGACY_INTEL_MAC else 'numpy==2.2.6',
    'einops==0.8.1','trimesh==4.6.13',
    'scipy==1.15.3' if sys.version_info<(3,11) else 'scipy==1.16.3',
    'shapely==2.1.2','certifi==2025.11.12']


def python_path(root):
    return Path(root)/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def verified_weights(root):
    for name,meta in MANIFEST['weights'].items():
        p=Path(root)/'weights'/name
        if not p.is_file() or p.stat().st_size!=meta['bytes']:return False
        with p.open('rb') as f:
            checksum=hashlib.sha256()
            while chunk:=f.read(1024*1024):checksum.update(chunk)
            digest=checksum.hexdigest()
        if digest!=meta['sha256']:return False
    return True


def claim_lock(path):
    """Recover a crashed worker's lock without interrupting another process."""
    for attempt in range(2):
        try:
            fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.write(fd,str(os.getpid()).encode());os.close(fd);return
        except FileExistsError:
            text=path.read_text();pid=int(text)
            if os.name=='nt':
                import ctypes
                kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                kernel.OpenProcess.restype=ctypes.c_void_p
                handle=kernel.OpenProcess(0x100000,False,pid)
                alive=bool(handle)
                if handle:
                    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
                    alive=kernel.WaitForSingleObject(handle,0)==258
                    kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(handle)
                elif ctypes.get_last_error()==5:alive=True
            else:
                try:os.kill(pid,0);alive=True
                except ProcessLookupError:alive=False
                except PermissionError:alive=True
            if alive:raise RuntimeError('Another rigging worker is using this runtime')
            if path.read_text()==text:path.unlink()
    raise RuntimeError('Could not acquire the rigging runtime')


def run(args):
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    state_path=root/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.action=='setup':
        # A separate venv preserves Blender's bundled NumPy and Python environment.
        state['ready']=False;write_json(state_path,state)
        venv.EnvBuilder(with_pip=True).create(root/'.venv')
        python=python_path(root)
        env=dict(os.environ,PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        base=[str(python),'-m','pip','install','--only-binary=:all:','--no-cache-dir']
        # CPU wheels on Windows/Linux avoid downloading CUDA libraries by default.
        torch_index=[] if sys.platform=='darwin' else ['--index-url','https://download.pytorch.org/whl/cpu']
        subprocess.run(base+torch_index+[REQUIREMENTS[0]],env=env,check=True)
        subprocess.run(base+REQUIREMENTS[1:],env=env,check=True)
        subprocess.run([str(python),str(__file__),'verify','--root',str(root),'--lock-owner-pid',str(os.getpid())],env=env,check=True)
        return dict(ready=True)
    if args.action=='prepare':
        import ssl,certifi
        tls=ssl.create_default_context(cafile=certifi.where())
        weights=root/'weights';weights.mkdir(exist_ok=True)
        state['models_ready']=False;write_json(state_path,state)
        for name,meta in MANIFEST['weights'].items():
            target=weights/name
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==meta['sha256']:continue
            url=f"https://huggingface.co/{MANIFEST['model_repository']}/resolve/{MANIFEST['model_revision']}/output/best/new/{name}"
            partial=target.with_suffix('.partial')
            try:
                with urllib.request.urlopen(url,timeout=90,context=tls) as response,partial.open('wb') as output:
                    digest=hashlib.sha256();total=0
                    while chunk:=response.read(1024*1024):
                        total+=len(chunk);digest.update(chunk);output.write(chunk)
                        print(f'Downloading {name}: {total*100//meta["bytes"]}%',flush=True)
                if total!=meta['bytes'] or digest.hexdigest()!=meta['sha256']:raise ValueError('Model checksum mismatch')
                partial.replace(target)
            finally:partial.unlink(missing_ok=True)
        state.update(models_ready=True,model_revision=MANIFEST['model_revision']);write_json(state_path,state)
        return dict(models_ready=True)
    sys.path.insert(0,str(PACKAGE.parent))
    import numpy as np,torch,trimesh
    from mesh_landmarks.inference import predict
    if args.action=='verify':
        from scipy.spatial import cKDTree
        import shapely
        from mesh_landmarks.model import PCAE
        from mesh_landmarks.skeleton import tree
        from mesh_landmarks.portable_ops import fps
        assert len(tree())==52
        assert len(fps(torch.zeros(2,3),torch.zeros(2,dtype=torch.long),.5))==1
        cKDTree(np.zeros((2,3))).query([0,0,0])
        state.update(ready=True,python=sys.version.split()[0],torch=torch.__version__,schema=1,
            requirements=REQUIREMENTS,devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])+(['cuda'] if torch.cuda.is_available() else []))
        write_json(state_path,state);return state
    if not verified_weights(root):raise RuntimeError('Download or repair the rigging model before generation')
    with np.load(args.input,allow_pickle=False) as data:
        if set(data.files)!={'vertices','faces'}:raise ValueError('Inference accepts only vertices and faces')
        result=predict(data['vertices'],data['faces'],root/'weights',args.device,lambda s:print(s,flush=True))
    result['input_sha256']=hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    result['model_revision']=MANIFEST['model_revision']
    write_json(args.output,result);return dict(finished=True,device=result['device'],seconds=result['seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','prepare','verify','infer'])
    p.add_argument('--root',required=True);p.add_argument('--lock-owner-pid',type=int);p.add_argument('--input');p.add_argument('--output');p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');args=p.parse_args()
    # One writer/inference per runtime; cannot overwrite a model used by another job.
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    lock=root/'worker.lock';owns=False
    try:
        if args.lock_owner_pid is not None:
            if args.action!='verify' or os.getppid()!=args.lock_owner_pid or lock.read_text()!=str(args.lock_owner_pid):
                raise RuntimeError('Invalid runtime verification owner')
        else:
            claim_lock(lock);owns=True
        print(json.dumps(run(args)),flush=True)
    finally:
        if owns:lock.unlink(missing_ok=True)
