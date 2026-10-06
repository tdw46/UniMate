"""Numerical tests for averaged four-weight patches and virtual seam welding."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_dress_boundary_weights import average_local_weights,project_chain_moments
from avatar_weight_seams import coincident_groups
bones=[f'chain_{c}_{j:02d}' for c in range(3) for j in (2,3)]
rng=np.random.default_rng(1847)
w=rng.dirichlet(np.full(6,.5),size=1000)
movable=np.ones(len(w),dtype=bool);movable[-20:]=False
r,report=project_chain_moments(w,bones,movable)
assert np.isfinite(r).all() and (r>=0).all()
assert np.count_nonzero(r[movable],axis=1).max()<=4
assert np.allclose(r.reshape(-1,3,2).sum(axis=2),w.reshape(-1,3,2).sum(axis=2))
assert np.allclose(r@np.array([2,3]*3),w@np.array([2,3]*3))
assert np.array_equal(r[~movable],w[~movable])
# A noisy two-chain field has an exact four-slot representation, so this
# isolates the average filter from the projection and verifies peak damping.
x=np.linspace(0,1,101);share=np.clip(x+.15*np.sin(np.arange(len(x))*2),0,1)
w=np.stack(((1-share)*.3,(1-share)*.7,share*.3,share*.7),axis=1)
mask=np.ones(len(w),dtype=bool);mask[[0,-1]]=False
edges=list(zip(range(len(w)-1),range(1,len(w))))
bones=['A_02','A_03','B_02','B_03']
r,_=average_local_weights(w,bones,edges,mask,np.ones(w.shape,dtype=bool))
assert np.allclose(r.sum(axis=1),1.) and np.count_nonzero(r,axis=1).max()<=4
assert np.array_equal(r[~mask],w[~mask])
assert np.linalg.norm(np.diff(r[:,2:].sum(axis=1),n=2)) < .25*np.linalg.norm(np.diff(share,n=2))
upper,_=average_local_weights(w,[n.replace('_02','_00').replace('_03','_01') for n in bones],edges,mask,np.ones(w.shape,dtype=bool))
assert np.array_equal(upper,r)
assert coincident_groups([[0,0,0],[1,0,0],[0,0,0],[1,0,0],[.5,0,0]])==[[0,2],[1,3],[4]]
assert coincident_groups([[0,0,0],[.0001,0,0],[1,0,0]])==[[0],[1],[2]]
assert coincident_groups([])==[]
print('DRESS_AVERAGE_AND_SEAMS_OK',report)
