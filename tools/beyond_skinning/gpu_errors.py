"""Recovered allocation diagnostic for the qualified dense/sparse GPU stages, including facial contexts.

The original's tracked host fields are cumulative logical allocations, not RSS
or available VRAM. Callers must supply recovered accounting for their route.
Profile-mask preparation contributes points*bones bytes; constrained shapes
contribute no initial profile mask. The caller supplies graph-build, completed
field, interpolation/upload and sparse edge-map contributions for its stage.
These logical counters do not describe actual memory usage. Unknown contexts
remain explicit; arbitrary runtime/device errors are not relabeled here.
"""
def empty_columns(runtime, host_field_bytes, *, kind='dense'):
    if host_field_bytes is None:
        raise ValueError('GPU integration: refined/facial empty-edge accounting remains unqualified')
    detail="Validation Error\n\nCaused by:\n  In Device::create_bind_group, label = 'vbdk_bg'\n    Buffer with 'vbdk_col_idx' label binding size is zero\n"
    if kind=='sparse':detail="Validation Error\n\nCaused by:\n  In Device::create_bind_group\n    Buffer with '' label binding size is zero\n"
    device_failure(runtime,host_field_bytes,detail)

def device_failure(runtime,host_field_bytes,detail):
    if host_field_bytes is None:raise ValueError('GPU device failure accounting remains unqualified')
    _,base,depth,_=runtime.bind_env_snapshot()
    defaults=f'The defaults (Base Resolution {base}, Depth {depth})' if base and depth else 'The defaults'
    raise runtime.PanicException(
        f'Auto Skinning stopped: the GPU refused an allocation ({detail}). '
        'At this voxel resolution the solve needs more video memory than the card has free '
        f'(the host side had already allocated {host_field_bytes / 1073741824.0:.2f} GiB of fields).\n\n'
        f'You almost certainly do not need it this high. {defaults} cover nearly every model. '
        'A finer voxel grid only helps when weights bleed across a small gap in the mesh, between the thighs for instance.\n\n'
        'Lower Base Resolution or Depth and run again. Nothing has been changed on your mesh or your rig.')

def invoke(runtime,host_field_bytes,operation,*args):
    """Only the compiled device callback can produce this typed failure.

    ValueError, MemoryError, cancellation and unrelated Rust panics retain their
    original identity; old research binaries without this capability still work.
    """
    try:return operation(*args)
    except getattr(runtime,'GpuDeviceError',()) as error:
        device_failure(runtime,host_field_bytes,str(error))
