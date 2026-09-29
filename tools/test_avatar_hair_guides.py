"""Run in factory-startup Blender; regression for cap-connected hair gaps."""
from pathlib import Path
import sys
import bpy
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar_hair_guides import hanging_branches, branch_line


def run():
    if not bpy.app.background:
        raise RuntimeError('Use isolated background Blender for guide fixtures')
    # Two long locks connect across their scalp. A third lock flares almost
    # horizontally. Its angle must not cause rejection or pull the other locks.
    points = np.array([(-1,0,2), (-1,0,1), (-1,0,0),
                       (1,0,2), (1,0,1), (1,0,0),
                       (3,0,2), (4,0,1.45), (5,0,1.35)], dtype=float)
    edges = [(0,1),(1,2),(3,4),(4,5),(0,3),(3,6),(6,7),(7,8)]
    mesh = bpy.data.meshes.new('Cap branch regression')
    try:
        mesh.from_pydata(points.tolist(), edges, [])
        branches = hanging_branches(mesh, range(len(points)), points, 1.5, .1)
        assert len(branches) == 3
        assert {tuple(b['ids']) for b in branches} == {(1,2),(4,5),(7,8)}
        for branch in branches:
            line = branch_line(branch, 4)
            assert len(line) == 5
            assert abs((branch['root']-line[0]).length-.1) < 1e-6
            x = [p.x for p in line]
            if branch['ids'][0] == 1:
                assert max(abs(v+1) for v in x) < 1e-6
            elif branch['ids'][0] == 4:
                assert max(abs(v-1) for v in x) < 1e-6
            else:
                assert max(x)-min(x) > 1.0  # Legitimate horizontal flare survives.
                assert all(a <= b for a,b in zip(x,x[1:]))
        # Coordinates are stored as float32 in Blender; compare at that precision.
        assert np.array_equal(points.astype(np.float32), np.array([v.co[:] for v in mesh.vertices],dtype=np.float32))
        print('HAIR_GUIDE_REGRESSION_OK: cap branches isolated; sideways strand preserved; mesh unchanged')
    finally:
        bpy.data.meshes.remove(mesh)


if __name__ == '__main__':
    run()
