"""Area-uniform rig-local surface samples without modifying mesh geometry."""
import numpy as np


def sample_surface(meshes,count=250000):
    triangles=[]
    for obj in meshes:
        obj.data.calc_loop_triangles()
        vertices=np.array([v.co[:] for v in obj.data.vertices])
        indices=np.array([t.vertices[:] for t in obj.data.loop_triangles],dtype=int)
        if len(indices):triangles.append(vertices[indices])
    if not triangles:return np.empty((0,3))
    triangles=np.concatenate(triangles)
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    valid=area>0;triangles=triangles[valid];area=area[valid]
    if not len(area):return np.empty((0,3))
    rng=np.random.default_rng(0)
    selected=triangles[rng.choice(len(triangles),count,p=area/area.sum())]
    u,v=rng.random((2,count));root=np.sqrt(u)
    return ((1-root)[:,None]*selected[:,0]+(root*(1-v))[:,None]*selected[:,1]+(root*v)[:,None]*selected[:,2])
