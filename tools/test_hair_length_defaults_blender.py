"""Run in Blender: generated short/long profiles and legacy panel semantics."""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import bpy
from mathutils import Vector
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('defaults',root/'avatar_rig_defaults.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
# Execute the actual adapter without registering the entire renderer.
tree=ast.parse((root/'properties_hallway_rig.py').read_text())
ns={'springs':lambda rig,prefix:rig.springs,'non_root_joint_roles':lambda rig,chains:[[i>0 for i in range(len(s.joints))] for s in chains]}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_spring'],type_ignores=[]),str(root/'properties_hallway_rig.py'),'exec'),ns)
class Joint(dict):
 def __init__(self,name):super().__init__();self.node=SimpleNamespace(bone_name=name);self.stiffness=.3;self.drag_force=.3;self.gravity_power=.0105
class Spring(dict):
 def __init__(self):super().__init__();self.joints=[Joint('a'),Joint('b'),Joint('c')]
def fixture(length,height=1.):
 s=Spring();r=SimpleNamespace(data=SimpleNamespace(bones={n:SimpleNamespace(head_local=Vector((0,0,length*i/2))) for i,n in enumerate('abc')}),springs=[s]);return r,s
points=[d.hair_length_defaults(i/1000) for i in range(401)]
assert all(a['stiffness']>=b['stiffness'] and a['drag']>=b['drag'] for a,b in zip(points,points[1:]))
assert d.hair_length_defaults(.02)==d.hair_length_defaults(.075)
assert math.isclose(d.hair_length_defaults(.23)['stiffness'],.3)
assert all(math.isclose(p['gravity']/p['stiffness'],.035) for p in points)
for invalid in (-1,float('nan'),float('inf')):
 try:d.hair_length_defaults(invalid)
 except ValueError:pass
 else:raise AssertionError('Invalid length accepted')
group=SimpleNamespace(prefix='Hair',stiffness=.3,non_root_stiffness=1.,drag=.3,gravity=.0105)
for length in (.02,.075,.12,.2,.23,.4):
 values=[]
 for scale in (.01,1,100):
  rig,s=fixture(length*scale);d.configure_hair_spring(s,rig,scale)
  ns['apply_spring'](rig,group);values.append([[j.stiffness,j.drag_force,j.gravity_power] for j in s.joints])
  before=[v[:] for v in values[-1]];ns['apply_spring'](rig,group)
  assert before==[[j.stiffness,j.drag_force,j.gravity_power] for j in s.joints]
  group.stiffness=0.;ns['apply_spring'](rig,group,'stiffness');group.stiffness=.3;ns['apply_spring'](rig,group,'stiffness')
  assert before==[[j.stiffness,j.drag_force,j.gravity_power] for j in s.joints]
 for rows in values[1:]:
  assert all(math.isclose(x,y,rel_tol=1e-6) for a,b in zip(values[0],rows) for x,y in zip(a,b))
rig,s=fixture(.1);ns['apply_spring'](rig,group)
assert all(j.drag_force==.3 and j.gravity_power==.0105 for j in s.joints)
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps({'passed':True,'version':bpy.app.version_string,'samples':401,'scale_cases':18,'legacy_unchanged':True}))
