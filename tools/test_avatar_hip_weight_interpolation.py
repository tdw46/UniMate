"""Four-slot upper handoff preserves body, chain and helper-level totals."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_hip_weight_interpolation import average_helper_handoff
class Bone(dict):
 def __init__(self,name,root):
  super().__init__(hallway_skirt_support=2,hallway_support_chain=root);self.name=name
rig=SimpleNamespace(data=SimpleNamespace(bones=[Bone('Support_A','Skirt_A_00'),Bone('Support_B','Skirt_B_00')]))
bones=['Skirt_A_00','Skirt_B_00'];count=41
x=np.arange(count);share=.5+.15*np.sin(x*1.8);level=.4+.12*np.sin(x*1.3)
# Each row has body plus a two-by-two helper/main patch. The sparse result
# must preserve the averaged patch marginals while reducing five slots to four.
seed=np.stack((np.full(count,.1),.9*share*level,.9*(1-share)*level,.9*share*(1-level),.9*(1-share)*(1-level)),axis=1)
body=[dict(Hips=row[0],Support_A=row[1],Support_B=row[2]) for row in seed]
edges=list(zip(range(count-1),range(1,count)))
b,w=average_helper_handoff(rig,body,bones,seed[:,3:],edges)
r=np.array([[row.get('Hips',0),row.get('Support_A',0),row.get('Support_B',0),v[0],v[1]] for row,v in zip(b,w)])
e=seed.copy()
for _ in range(20):
 average=e.copy();degree=np.ones(count)
 for a,c in edges:average[a]+=e[c];average[c]+=e[a];degree[a]+=1;degree[c]+=1
 e=.92*e+.08*average/degree[:,None]
assert np.isfinite(r).all() and (r>=0).all()
assert np.count_nonzero(r,axis=1).max()<=4
assert np.allclose(r.sum(axis=1),1.) and np.array_equal(r[:,0],seed[:,0])
for cols in ([1,3],[2,4],[1,2],[3,4]):assert np.allclose(r[:,cols].sum(axis=1),e[:,cols].sum(axis=1),atol=1e-8)
assert np.linalg.norm(np.diff(r[:,1:3].sum(axis=1),n=2)) < .6*np.linalg.norm(np.diff(seed[:,1:3].sum(axis=1),n=2))
b2,w2=average_helper_handoff(rig,body,bones,seed[:,3:],edges)
assert b==b2 and np.array_equal(w,w2)
print('UPPER_HANDOFF_INTERPOLATION_OK')
