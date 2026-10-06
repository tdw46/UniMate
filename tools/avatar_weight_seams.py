"""Virtual rest-position welding for skin-weight filters; never edit topology."""
import numpy as np
from mathutils.kdtree import KDTree


def coincident_groups(positions, relative_tolerance=1e-6):
    """Return representative row indices, including each singleton.

    Callers must scope positions to one garment family. A bounded cluster uses
    distance to its representative, preventing transitive near-point chains
    from welding a broad surface band. Coordinates are rest-space only.
    """
    p = np.asarray(positions, dtype=float).reshape((-1,3))
    if not np.isfinite(p).all() or relative_tolerance < 0:
        raise ValueError('Invalid virtual weld coordinates')
    if not len(p):return []
    tolerance = max(float(np.linalg.norm(np.ptp(p,axis=0)))*relative_tolerance,1e-9)
    tree = KDTree(len(p))
    for i,point in enumerate(p):tree.insert(point,i)
    tree.balance()
    used=set();groups=[]
    for i,point in enumerate(p):
        if i in used:continue
        group=sorted(j for _,j,_ in tree.find_range(point,tolerance) if j not in used)
        used.update(group);groups.append(group)
    return groups
