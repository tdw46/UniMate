"""Scoped secondary-region connectivity. Never weld or edit the source mesh.

Only caller-admitted vertices participate. Hand/foot domains are protected before
admission, and this graph is never passed into humanoid/finger skinning.
"""
from types import SimpleNamespace
import numpy as np
from mathutils.kdtree import KDTree


def connected_graph(mesh, points, allowed, tolerance):
    ids=sorted(set(allowed));parent={i:i for i in ids};edges=[];links=[]
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):
        x,y=root(a),root(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    for edge in mesh.edges:
        a,b=edge.vertices
        if a in parent and b in parent:edges.append((a,b));join(a,b)
    original={i:root(i) for i in ids}
    tree=KDTree(len(ids))
    for i in ids:tree.insert(points[i],i)
    tree.balance()
    pairs=set(edges)
    # Keep local seam correspondences even after components are connected.
    # A spanning-tree-only weld loses those correspondences when a later scalp
    # or height cut removes its sole bridge, stranding lower seam islands.
    for i in ids:
        targets=set()
        for _,j,_ in sorted(tree.find_range(points[i],tolerance),key=lambda v:(v[2],v[1])):
            if original[j]==original[i] or original[j] in targets:continue
            targets.add(original[j]);pair=(min(i,j),max(i,j))
            if pair not in pairs:
                pairs.add(pair);join(i,j);edges.append(pair);links.append(pair)
    groups={}
    for i in ids:groups.setdefault(root(i),[]).append(i)
    graph=SimpleNamespace(edges=[SimpleNamespace(vertices=e) for e in edges])
    return graph,list(groups.values()),links


def protected_extremities(rig,points,height):
    """Exclude hands/digits and feet independent of material or source weights."""
    result=np.zeros(len(points),dtype=bool)
    for side in ('L','R'):
        for name,radius in [('Hand.'+side,.045),('Foot.'+side,.04)]:
            bone=rig.data.bones.get(name)
            if bone is None:continue
            a=np.asarray(bone.head_local);b=np.asarray(bone.tail_local);d=b-a
            t=np.clip((points-a)@d/max(float(d@d),1e-16),0,2.5)
            result|=np.linalg.norm(points-(a+t[:,None]*d),axis=1)<height*radius
    return result
