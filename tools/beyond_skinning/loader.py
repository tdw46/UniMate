"""Load only the bundled, hash-pinned native ABI in an isolated worker."""
from pathlib import Path
import hashlib,importlib.util,importlib.machinery,json,platform,sys,os
_RUNTIME=None
_SELECTION=None

def backend_info():
    runtime()
    return dict(_SELECTION)

def _probe_gpu(module):
    """Probe the actual compute device, only in the isolated binding worker."""
    limits=module.gpu_limits()
    if limits is None or limits[1] not in ('Metal','Vulkan'):
        return None
    module.gpu_device_limits()
    return list(limits[:2])

def _load_binary(root,entry):
    binary=root/entry['path']
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Solver package verification failed. Reinstall the add-on.')
    name=__package__+'._bvt_solver'
    loader=importlib.machinery.ExtensionFileLoader(name,str(binary));spec=importlib.util.spec_from_file_location(name,binary,loader=loader)
    module=importlib.util.module_from_spec(spec);loader.exec_module(module);sys.modules[name]=module
    return module

def runtime():
    global _RUNTIME,_SELECTION
    if _RUNTIME is not None:return _RUNTIME
    if sys.version_info < (3,11):raise RuntimeError('The solver requires Python 3.11 or newer. Choose Blender weighting on this Blender version.')
    root=Path(__file__).resolve().parent
    machine=platform.machine().lower();machine={'aarch64':'arm64','x86_64':'amd64'}.get(machine,machine)
    key=platform.system().lower()+'-'+machine
    requested=os.environ.get('BEYOND_SKINNING_BACKEND','auto')
    if requested not in ('auto','cpu','gpu'):raise ValueError('Unknown skinning backend')
    manifest=json.loads((root/'manifest.json').read_text())
    binaries=manifest['binaries'].get(key,{})
    mode=('gpu' if 'gpu' in binaries else 'cpu') if requested=='auto' else requested
    entry=binaries.get(mode)
    if not entry:raise RuntimeError('Solver unavailable on this platform. Choose Blender weighting.')
    module=_load_binary(root,entry)
    selection=dict(requested=requested,backend=mode,adapter=None,fallback_reason=None)
    if requested=='auto':
        if mode=='gpu':
            # The GPU library also implements the CPU path. If device creation
            # fails, use that path without loading a second native module.
            errors=(RuntimeError,ValueError)+tuple(t for t in (
                getattr(module,'PanicException',None),getattr(module,'GpuDeviceError',None)
            ) if isinstance(t,type) and issubclass(t,BaseException))
            try:selection['adapter']=_probe_gpu(module)
            except errors as error:
                selection['fallback_reason']=str(error)
            if selection['adapter'] is None:
                selection['backend']='cpu'
                selection['fallback_reason']=selection['fallback_reason'] or 'No supported Metal or Vulkan compute device'
        else:selection['fallback_reason']='No bundled GPU runtime for this platform'
    _SELECTION=selection
    _RUNTIME=module;return module
