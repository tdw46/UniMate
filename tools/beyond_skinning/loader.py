"""Load only the bundled, hash-pinned native ABI in an isolated worker."""
from pathlib import Path
import hashlib,importlib.util,importlib.machinery,json,platform,sys,os
_RUNTIME=None

def runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    if sys.version_info < (3,11):raise RuntimeError('The solver requires Python 3.11 or newer. Choose Blender weighting on this Blender version.')
    root=Path(__file__).resolve().parent
    machine=platform.machine().lower();machine={'aarch64':'arm64','x86_64':'amd64'}.get(machine,machine)
    key=platform.system().lower()+'-'+machine
    mode=os.environ.get('BEYOND_SKINNING_BACKEND','cpu')
    if mode not in ('cpu','gpu'):raise ValueError('Unknown skinning backend')
    manifest=json.loads((root/'manifest.json').read_text())
    entry=manifest['binaries'].get(key,{}).get(mode)
    if not entry:raise RuntimeError('Solver unavailable on this platform. Choose Blender weighting.')
    binary=root/entry['path']
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Solver package verification failed. Reinstall the add-on.')
    name=__package__+'._bvt_solver'
    loader=importlib.machinery.ExtensionFileLoader(name,str(binary));spec=importlib.util.spec_from_file_location(name,binary,loader=loader)
    module=importlib.util.module_from_spec(spec);loader.exec_module(module);sys.modules[name]=module
    _RUNTIME=module;return module
