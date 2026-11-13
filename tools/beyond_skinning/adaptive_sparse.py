"""Independent sparse refinement stage; caller owns cloud/seed/level policies.

Prepared inputs are explicit. No native module imports or fallback; this is not
an independently orchestrated complete adaptive bind.
"""
import numpy as np
from .compact_field import CompactField


def refine(runtime,graph,sample_ids,sample_weights,prior_field,fixed,seed_indices,*,
           iterations=1,seed_values=None,slots=16,stop=None,array_transport=False,use_gpu=False,host_field_bytes=None,retain_edge_map=True):
    if array_transport and isinstance(prior_field,CompactField):
        groups,initial=runtime.interpolate_compact_arrays(sample_ids,sample_weights,prior_field.groups,prior_field.values,prior_field.bones,np.asarray(seed_indices,'u4'),slots)
    elif array_transport:
        groups,initial=runtime.interpolate_sparse_arrays(sample_ids,sample_weights,prior_field,np.asarray(seed_indices,'u4'),slots)
    else:
        dense=np.asarray(runtime.interpolate_level_field(sample_ids,sample_weights,prior_field),'f4')
        if len(dense)!=len(seed_indices):raise ValueError('inconsistent refinement rows')
        pairs=[runtime.sparse_seed_slots(row.tolist(),slots,int(seed)) for row,seed in zip(dense,seed_indices)]
        groups=np.asarray([x[0] for x in pairs],'u2');initial=np.asarray([x[1] for x in pairs],'f4')
    edge=runtime.build_sparse_edge_array(graph['rows'],graph['columns'],groups) if array_transport else np.frombuffer(runtime.build_sparse_edge_map(graph['rows'],graph['columns'],groups),'u1')
    packed_edge=use_gpu and array_transport and not retain_edge_map and bool(getattr(runtime,"immutable_replay_inputs",False))
    if packed_edge:
        # Complete conversion before replay; do not retain both the mutable
        # edge array and its immutable transport payload throughout the solve.
        edge=edge.tobytes()
    if use_gpu:
        from .gpu_solver import sparse
        field,done=sparse(runtime,graph,edge,groups,initial,fixed,seed_indices,iterations,seed_values,stop=stop,host_field_bytes=None if host_field_bytes is None else host_field_bytes+16*len(graph['columns']))
    else:field,done=runtime.solve_sparse_cpu(graph['rows'],graph['columns'],graph['weights'],graph['dinv'],edge,groups,initial,fixed,seed_indices,iterations,seed_values,stop=stop)
    return dict(groups=groups,initial=initial,edge_map=None if packed_edge else edge,field=field,iterations=done,complete_native_bind=False)
