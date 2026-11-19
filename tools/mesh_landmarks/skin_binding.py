"""Main-thread export and transactional application for the independent solver."""
from pathlib import Path
import json,platform,sys
import bpy
import numpy as np
from mathutils import Matrix,Vector
from . import blender_bridge


def package_root():
    for parent in Path(__file__).resolve().parents[:3]:
        root=parent/'beyond_skinning'
        if (root/'worker.py').is_file():return root
    raise RuntimeError('The skinning solver package is missing. Reinstall the add-on or choose Blender weighting.')


def available():
    try:
        root=package_root();machine=platform.machine().lower();machine={'aarch64':'arm64','x86_64':'amd64'}.get(machine,machine)
        return sys.version_info>=(3,11) and (root/'manifest.json').is_file() and (root/'native'/(platform.system().lower()+'-'+machine)/('cpu.pyd' if sys.platform=='win32' else 'cpu.so')).is_file()
    except RuntimeError:return False


def python_executable():
    # sys.executable may refer to Blender rather than its Python interpreter.
    exe=Path(sys.executable)
    if exe.name.lower().startswith('python'):return exe
    root=Path(sys.prefix)
    choices=[root/'bin'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor)),root/'bin/python.exe',root/'python.exe']
    for candidate in choices:
        if candidate.is_file():return candidate
    raise RuntimeError('Blender Python is missing; repair the Blender installation.')


def prepare(meshes,prediction,job,options):
    import copy
    if not available():raise RuntimeError('Solver unavailable on this platform. Choose Blender weighting.')
    prediction=blender_bridge.refine_prediction(meshes,prediction)
    if options.get('symmetrize', True):
        from .rig_frames import symmetric_prediction
        prediction = symmetric_prediction(prediction)
    local=copy.deepcopy(prediction);world,_=blender_bridge.geometry(meshes)
    origin=Vector(prediction['joints']['Hips']['head']);origin.z=float(world[:,2].min())
    for joint in local['joints'].values():
        for key in ('head','tail'):joint[key]=list(Vector(joint[key])-origin)
    specs=blender_bridge.specifications(local);names=[s[0] for s in specs]
    verts=[];faces=[];edges=[];offset=0;counts=[]
    for obj in meshes:
        matrix=Matrix.Translation(-origin)@obj.matrix_world
        verts.extend(tuple(matrix@v.co) for v in obj.data.vertices)
        obj.data.calc_loop_triangles()
        faces.extend(tuple(offset+i for i in t.vertices) for t in obj.data.loop_triangles)
        edges.extend(tuple(offset+i for i in e.vertices) for e in obj.data.edges)
        counts.append(len(obj.data.vertices));offset+=counts[-1]
    job=Path(job)
    np.savez(job/'binding-input.npz',vertices=np.asarray(verts,'f4'),triangles=np.asarray(faces,'i4').reshape(-1,3),edges=np.asarray(edges,'i8').reshape(-1,2),heads=np.asarray([s[1] for s in specs],'f4'),tails=np.asarray([s[2] for s in specs],'f4'),parents=np.asarray([names.index(s[3]) if s[3] else -1 for s in specs],'i8'),deform=np.asarray([n!='Root' for n in names],bool))
    request=dict(bone_names=names,counts=counts,backend=options['backend'],brand=options['brand'])
    (job/'binding.json').write_text(json.dumps(request))
    return [str(python_executable()),'-I',str(package_root()/'worker.py'),str(job)]


def load(job):
    job=Path(job);report=json.loads((job/'binding-result.json').read_text());request=json.loads((job/'binding.json').read_text())
    with np.load(job/'binding-output.npz',allow_pickle=False) as data:w,b=data['weights'].copy(),data['bones'].copy()
    if w.shape!=b.shape or len(w)!=sum(request['counts']) or not np.isfinite(w).all() or np.any(w<0) or np.any(b>=len(report['bone_names'])) or np.any((b<0)&(w>0)) or not np.allclose(w.sum(1),1.,rtol=0,atol=1e-5):raise ValueError('Invalid solver output; the original character was preserved.')
    return dict(weights=w,bones=b,counts=request['counts'],report=report)


def apply(rig,meshes,binding,heads):
    """Only generated copies are mutated. Caller owns scene-level rollback."""
    names=binding['report']['bone_names'];w=binding['weights'];b=binding['bones']
    if [len(o.data.vertices) for o in meshes]!=binding['counts'] or any(n not in rig.data.bones for n in names):raise ValueError('Generated geometry or skeleton changed during binding')
    offset=0;reports=[]
    for obj in meshes:
        count=len(obj.data.vertices);rows=w[offset:offset+count];ids=b[offset:offset+count];offset+=count
        obj.vertex_groups.clear()
        groups=[obj.vertex_groups.new(name=n) for n in names]
        head=rig.data.bones.get('Head');head_group=obj.vertex_groups.get('Head') if head else None
        if head and head_group is None:head_group=obj.vertex_groups.new(name='Head')
        owned=set(heads.get(obj.name,()))
        for i,(weights,bones) in enumerate(zip(rows,ids)):
            if i in owned and head_group is not None:head_group.add([i],1.,'REPLACE');continue
            for weight,bone in zip(weights,bones):
                if weight>0:groups[int(bone)].add([i],float(weight),'REPLACE')
        mod=obj.modifiers.new('Skin deformation','ARMATURE');mod.object=rig
        if hasattr(mod,'use_deform_preserve_volume'):mod.use_deform_preserve_volume=False
        mod.show_in_editmode=True;mod.show_on_cage=True;obj.parent=rig
        reports.append(dict(object=obj.name,vertices=count,cranial_vertices=len(owned)))
    return dict(binding['report'],objects=reports,cleanup='Cranial ownership and existing secondary rig cleanup',source_weights_used=False,preserve_volume=False)
