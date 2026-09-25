"""Offline contact fitting in skin-weight space; rest vertices are immutable."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
class WeightContactFitter:
    def __init__(self,faces,positions,bodies,body_faces,axes,initial,allowed,clearance=.001,welds=(),secondary=None,legs=None):
        self.faces=np.asarray(faces);self.positions=np.asarray(positions)
        self.bodies=[BVHTree.FromPolygons([Vector(p) for p in b],body_faces,all_triangles=True) for b in bodies]
        self.initial=np.asarray(initial)

    def posed(self,w,f):return np.einsum('vkj,vk->vj',self.positions[f],w)

    def intersections(self,w,frames):
        return [len(BVHTree.FromPolygons([Vector(p) for p in self.posed(w,f)],self.faces,all_triangles=True).overlap(self.bodies[f])) for f in frames]

    def fit_attachment(self,train,held,rest,bone_names,upper,hip_width,progress=None):
        """Small deterministic search; smooth attachment changes retain physics.

        Only training triangle contacts choose parameters. Existing spring/hips
        weights keep at least 20% of their contribution. Maintain >5% spring
        membership so auditing cannot accidentally drop fitted garment faces.
        """
        before=self.intersections(self.initial,train);best=self.initial.copy();score=sum(before);choice={};history=[]
        spring=np.array([n.startswith('Secondary_Skirt_') for n in bone_names]);mass=self.initial[:,spring].sum(axis=1)
        height=np.ptp(rest[:,2]);lower=np.clip((rest[:,2].max()-rest[:,2])/max(height,1e-8),0,1)
        for width_fraction in (.1,.3,.6):
            width=max(hip_width*width_fraction,1e-6)
            t=np.clip((rest[:,0]/width+1)/2,0,1);t=t*t*(3-2*t)
            anchor=np.zeros_like(best);anchor[:,bone_names.index(upper[0])]=t;anchor[:,bone_names.index(upper[1])]=1-t
            for profile in ('uniform','lower','tapered'):
                taper=np.ones(len(rest)) if profile=='uniform' else lower if profile=='lower' else .5+.5*lower
                for strength in (.2,.4,.6,.8):
                    alpha=np.minimum(strength*taper,np.maximum(0.,1-.06/np.maximum(mass,1e-8)))
                    trial=self.initial*(1-alpha[:,None])+anchor*alpha[:,None]
                    counts=self.intersections(trial,train);value=sum(counts)
                    item=dict(width_fraction=width_fraction,profile=profile,strength=strength,training_pairs=value)
                    history.append(item)
                    if progress:progress(item)
                    if value<score:score=value;best=trial;choice=item
        return best,dict(training_before=before,training_after=self.intersections(best,train),
            validation_before=self.intersections(self.initial,held),validation_after=self.intersections(best,held),
            choice=choice,history=history,geometry_changed=False,minimum_original_binding_fraction=.2)
