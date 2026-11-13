"""Owned sparse adaptive fields; dense expansion is diagnostic/visualization only."""
import numpy as np
class CompactField:
    def __init__(self,groups,values,bones):
        self.groups=groups;self.values=values;self.bones=bones
        self.shape=(len(groups),bones)
    def dense(self):
        out=np.zeros(self.shape,'f4');i,j=np.where(self.groups!=65535)
        out[i,self.groups[i,j]]=self.values[i,j]
        return out
    def tobytes(self):
        # Historical research controls explicitly compare dense diagnostic bytes.
        return self.dense().tobytes()
    def clean(self,runtime,graph,radii,threshold,top_k):
        g,v=runtime.clean_compact_arrays(graph['rows'],graph['columns'],radii,self.groups,self.values,self.bones,threshold,top_k)
        return CompactField(g,v,self.bones)
    def sample(self,runtime,ids,weights):
        return runtime.sample_compact_array(ids,weights,self.groups,self.values,self.bones)
