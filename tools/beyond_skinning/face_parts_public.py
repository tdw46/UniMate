"""Public face-plan routing after typed native extraction; no oracle fallback."""
import numpy as np

def assemble(names,parents,deform,segments,meshes,keep):
 from . import _bvt_solver as runtime
 empty=dict(parts=[],regions=[],part_bones=[])
 if not names:return empty
 d=np.frombuffer(deform,'u1').copy().view(bool)
 context=runtime.rig_map(names,parents,d,segments,mesh_names=[m[0] for m in meshes],scope='detected')
 if not context['face_family'] or not np.any(d):return empty
 # Active geometry continues through the independently recovered planner.
 # Broader family/shape assembly is deliberately not inferred from names here.
 from .mesh_correspondence import plan_inputs
 result=plan_inputs(runtime,meshes,names,segments[:,0].copy(),segments[:,1].copy(),parents,keep,deform=d,rig_context=context)
 return {key:result[key] for key in ('parts','regions','part_bones')}
