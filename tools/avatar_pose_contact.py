"""Offline pose contact fitting. No runtime physics, handlers or export drivers.

Cached skin matrices map rest displacements to every sampled pose. The bounded
projection optimizer uses NumPy bundled with Blender; no SciPy installation.
"""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def sample_surface(triangles):
    """Unique vertices, edge midpoints and face centers, with barycentric rows."""
    vertices=sorted({int(i) for t in triangles for i in t})
    edges=sorted({tuple(sorted((int(t[i]),int(t[(i+1)%3])))) for t in triangles for i in range(3)})
    rows=[(i,i,i) for i in vertices]+[(a,b,b) for a,b in edges]+[tuple(t) for t in triangles]
    weights=[(1.,0.,0.)]*len(vertices)+[(.5,.5,0.)]*len(edges)+[(1/3,)*3]*len(triangles)
    return np.array(rows,dtype=np.int32),np.array(weights)


class PoseContactFitter:
    def __init__(self, triangles, poses, bases, matrices, body_triangles, axes,
                 clearance=.002, budget=.04, welds=()):
        self.triangles=triangles
        self.bases=bases
        self.matrices=matrices
        self.axes=axes
        self.clearance=clearance
        self.budget=budget
        self.rows,self.weights=sample_surface(triangles)
        self.bodies=[BVHTree.FromPolygons([Vector(p) for p in body],body_triangles,all_triangles=True) for body in poses]
        self.vertex_count=bases.shape[1]
        self.welds=welds

    def positions(self, delta, frame):
        return self.bases[frame]+np.einsum('vij,vj->vi',self.matrices[frame],delta)

    def intersections(self, delta, frames):
        counts=[]
        for frame in frames:
            tree=BVHTree.FromPolygons([Vector(p) for p in self.positions(delta,frame)],self.triangles,all_triangles=True)
            counts.append(len(tree.overlap(self.bodies[frame])))
        return counts

    def constraints(self, delta, frames):
        rows=[];coefficients=[];bounds=[]
        for frame in frames:
            positions=self.positions(delta,frame)
            points=np.einsum('si,sij->sj',self.weights,positions[self.rows])
            a=self.axes[frame,:,0];ab=self.axes[frame,:,1]-a
            t=np.clip(np.einsum('slj,lj->sl',points[:,None]-a,ab)/np.maximum(np.sum(ab*ab,axis=1),1e-12),0,1)
            centers=a[None]+t[:,:,None]*ab[None]
            distance=np.linalg.norm(points[:,None]-centers,axis=2)
            nearest=np.argmin(distance,axis=1)
            for si,p in enumerate(points):
                origin=centers[si,nearest[si]];radial=p-origin;length=np.linalg.norm(radial)
                if length<1e-8 or length>.3:continue
                normal=radial/length
                start=Vector(origin);direction=Vector(normal);outer=None
                # Pick the first outward exit. Unlike a nearest-surface normal,
                # this distinguishes the inside of a limb from the gap between legs.
                for _ in range(16):
                    hit,n,_,_=self.bodies[frame].ray_cast(start,direction,.3)
                    if hit is None:break
                    if n.dot(direction)>.05:
                        outer=np.asarray(hit);break
                    start=hit+direction*1e-5
                if outer is None:continue
                deficit=float(np.dot(outer-p,normal)+self.clearance)
                if deficit<-.003:continue
                ids=self.rows[si]
                grad=np.einsum('j,kji,k->ki',normal,self.matrices[frame,ids],self.weights[si])
                bound=deficit+float(np.sum(grad*delta[ids]))
                # Repeated vertex indices in vertex/edge samples must be merged.
                for j in range(3):
                    for k in range(j):
                        if ids[j]==ids[k]:grad[k]+=grad[j];grad[j]=0.
                rows.append(ids);coefficients.append(grad);bounds.append(bound)
        return np.asarray(rows,dtype=np.int32),np.asarray(coefficients),np.asarray(bounds)

    def fit(self, train, validation, iterations=8, progress=None):
        delta=np.zeros((self.vertex_count,3));best=delta.copy()
        initial=self.intersections(delta,train)
        best_score=sum(initial);history=[]
        constraints=[]
        for iteration in range(iterations):
            ids,grad,bounds=self.constraints(delta,train)
            if not len(ids):break
            constraints.append((ids,grad,bounds))
            # Recent contact planes follow the actual curved body. Keeping all
            # historic tangent planes would incorrectly inflate concave garments.
            active=constraints[-2:]
            for _ in range(30):
                for rows,coeff,rhs in active:
                    for indices,g,b in zip(rows,coeff,rhs):
                        deficit=b-float(np.sum(g*delta[indices]))
                        if deficit<=0:continue
                        norm=float(np.sum(g*g))
                        if norm<1e-14:continue
                        step=.75*deficit/norm
                        for index,gradient in zip(indices,g):
                            delta[index]+=gradient*step
                            # Bound shape changes, especially vertical hem/waist drift.
                            delta[index,2]=np.clip(delta[index,2],-self.budget*.25,self.budget*.25)
                            length=np.linalg.norm(delta[index])
                            if length>self.budget:delta[index]*=self.budget/length
                for group in self.welds:delta[group]=np.mean(delta[group],axis=0)
            counts=self.intersections(delta,train)
            score=sum(counts)
            report=dict(iteration=iteration,contacts=len(ids),intersections=score,
                        maximum_displacement=float(np.max(np.linalg.norm(delta,axis=1))))
            history.append(report)
            if progress:progress(report)
            if score<best_score:best_score=score;best=delta.copy()
            if score==0:break
        return best,dict(training_before=initial,training_after=self.intersections(best,train),
                         validation_before=self.intersections(np.zeros_like(best),validation),
                         validation_after=self.intersections(best,validation),history=history,
                         budget=self.budget,clearance=self.clearance)
