"""Independent public detector diagnostics, assembled from recovered stages."""
import numpy as np
from .adaptive_detector import detect as detector

def detect(v,f,lo,hi,resolution,multiplier,depth,probe,fill,threshold,ghost,mesh):
 from . import _bvt_solver as rt
 c=detector(rt,v,f,[lo,hi],resolution=resolution,multiplier=multiplier,max_depth=depth,probe_multiplier=probe,fill_interior=fill,ghost=ghost,winding_threshold=threshold,interior_route="winding",bounded=False,diagnostics=True,ghost_mesh=mesh)
 coarse=c['coarse'];out={k:coarse[k] for k in ('resolution','voxel_size','bbox_min','surface_voxels','avg_faces_per_voxel','max_faces_per_voxel')}
 for k in ('voxel_indices','inside_voxel_indices'):out[k]=np.asarray(coarse[k],'i4').reshape(-1,3)
 out['inside_voxel_count']=len(out['inside_voxel_indices']);out['ghost_culled']=sum(c['ghost_counts'])
 for k in ('cloud_positions','cloud_levels','cloud_radii'):out[k]=c[k]
 # Legacy overlap debug fields are empty in the face-probe route.
 out['overlap_indices']=np.empty((0,3),'i4');out['overlap_clusters']=np.empty(0,'i4');out['cluster_bboxes']=np.empty((0,6),'i4')
 out['avg_faces_per_voxel']=float(np.float32(out['avg_faces_per_voxel']))
 out['levels']=c['levels']
 for k in ('l4_voxel_indices','l4_voxel_depths','l4_voxel_normals'):out[k]=c[k]
 return out
