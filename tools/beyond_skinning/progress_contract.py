"""Public stage fractions and isolated callback delivery.

Stage seconds follow the original float32 workload heuristic, not measured
wall time. Legacy prepared estimates remain supported by internal research callers.
"""
import numpy as np
STAGES=('voxelizing','seeding','solving','finishing')
FRACTIONS=tuple(float(np.float32(x)) for x in (0.,.2,.3,.95,1.))
class ProgressSink:
    def __init__(self,callback,estimates=None,*,triangles=0,bones=0):
        self.callback=callback
        if estimates is not None and len(estimates)!=4:raise ValueError('four prepared progress estimates required')
        self.estimates=None if estimates is None else tuple(float(np.float32(x)) for x in estimates)
        self.triangles=triangles
        self.bones=bones
        self.callback_errors=0
    def emit(self,name,*,points=0):
        i=STAGES.index(name)
        seconds=self.estimates[i] if self.estimates is not None else estimate(name,self.triangles,self.bones,points)
        try:self.callback(FRACTIONS[i],name,FRACTIONS[i+1],seconds)
        except BaseException:self.callback_errors+=1 # Original PyErr handling swallows every callback exception.

def estimate(stage,triangles,bones,points):
    """Recovered ARM64 operations at 12964c, 12cb98, 12d9d8, 12eae0.

    Separate float32 arithmetic is intentional: no fused multiply/add or f64
    intermediate. The solve estimate uses final cloud points and solve bones;
    iteration count, quality and cache hits do not alter this native heuristic.
    """
    F=np.float32
    if stage=='voxelizing':return float(F(F(F(triangles)*F(6e-7))+F(.3)))
    if stage=='seeding':return float(F(.12))
    if stage=='solving':return float(F(F(F(F(bones)*F(1.2e-5))+F(.0004))*F(points)/F(1000)))
    if stage=='finishing':return float(F(.15))
    raise ValueError('unknown progress stage')

def check_coverage(weights,minimum=.01):
    """Post-writeback coverage gate; sparse/partial thresholds remain bounded."""
    w=np.asarray(weights,'f4')
    count=int(np.count_nonzero(np.any(w>0,axis=1)))
    if len(w) and count/len(w)<minimum:
        raise ValueError(f"[invalid_armature] The armature does not deform this mesh: the bind reached only {count} of {len(w)} vertices. The rig and mesh are usually in different scales, which means a glTF/FBX import malformed the rest skeleton. Re-import with bone_heuristic='TEMPERANCE' and 'Guess Original Bind Pose' enabled, or match the armature scale to the mesh, then bind.")
    return count
