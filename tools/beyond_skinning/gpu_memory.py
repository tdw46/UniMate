"""Allocation-free admission for the current independent GPU transport.

Defaults preserve the bounded diagnostic/older-runtime route. The public solver
supplies the actual cached device's limits when its runtime exposes them. The
original requests adapter buffer maxima; those are capability limits, never a
free-VRAM measurement. Host estimates are telemetry during public binds; diagnostic budgets are explicit.
"""
import operator

STORAGE_BYTES = 128 * 1024 * 1024
BUFFER_BYTES = 256 * 1024 * 1024
WORKING_BYTES = 256 * 1024 * 1024
U32_MAX = (1 << 32) - 1

def plan(kind, points, slots, edges, iterations, *, automatic=False, borrowed_inputs=False, staged_readback=False,fixed_uniforms=False,packed_sparse=False):
    n,k,e,it = map(operator.index,(points,slots,edges,iterations))
    if kind not in ('dense','sparse') or n<=0 or k<=0 or e<0 or it<0:
        raise ValueError('invalid GPU transport dimensions')
    if (kind=='dense' and k%4) or (kind=='sparse' and k!=16):
        raise ValueError('invalid GPU transport slot width')
    if packed_sparse and (kind!="sparse" or not borrowed_inputs or it==0):
        raise ValueError("packed sparse admission requires borrowed nonzero sparse replay")
    cells=n*k
    if max(n,k,e,it,cells)>U32_MAX:
        raise ValueError('GPU transport exceeds 32-bit addressing')
    groups=(cells//(4 if kind=='dense' else 2)+255)//256
    gx=min(groups,32768)
    gy=(groups+gx-1)//gx
    word_bytes=lambda count:((count+31)//32)*4
    half_bytes=((n+1)//2)*4
    graph=dict(rows=4*(n+1),columns=4*e,weights=4*e,diagonal=4*n)
    if kind=='dense':
        packed=dict(mask=word_bytes(cells),seeds=word_bytes(cells)+half_bytes)
    else:
        packed=dict(mask=word_bytes(n)+word_bytes(cells)+half_bytes,edge_map=16*e)
    field_bytes=2*cells
    storage=dict(graph,**packed,field=field_bytes)
    uniforms=256*(min(it,32) if automatic else min(it,256) if fixed_uniforms else it)
    static=sum(graph.values())+sum(packed.values())
    # Conservative overlapping arrays, bool/half packing, PyO3 Vec copies,
    # returned f16/f32 conversion and fixed uniform staging. Packed native CSR
    # validation borrows input bytes, removing one owned graph copy.
    # Dense preparation owns a half upload, not padded full-mesh f32 seeds
    # and masks. Keep 20 bytes/cell for caller fields, packed/native transport,
    # readback and output overlap, plus explicit bounded f32 packing scratch.
    packing_cells=min(n,max(2,(65536//k//2)*2))*k if kind=='dense' else 0
    host=((22 if it==0 else 20) if kind=='dense' else 24)*cells+12*packing_cells+5*static-sum(graph.values())+3*uniforms+16*n+1024*1024
    # Public Python transport supplies exact bytes. Only a capability-qualified
    # bridge can omit their former PyO3 payload copies; older runtimes retain
    # the complete estimate. Automatic input carries one 32-byte record.
    borrowed_payload=(static+(32 if automatic or fixed_uniforms else uniforms)+field_bytes) if borrowed_inputs and it else 0
    host-=borrowed_payload
    if packed_sparse:
        # One immutable host edge map; graph arrays plus their immutable bytes.
        # 16 bytes/cell covers caller groups/initial, half upload/transport,
        # native return and f32 output (at most 14 simultaneous bytes/cell).
        # Mapped device construction uses the device storage counted below.
        packing_cells=min(n,max(8,(65536//k//8)*8))*k
        host=16*cells+2*sum(graph.values())+sum(packed.values())+16*n+12*packing_cells+3*uniforms+1024*1024
    # Static inputs, ping-pong, readback and optional convergence checkpoint.
    # Automatic final readback may follow the completed convergence checkpoint.
    # That route retires spare/previous fields first: peak has three field-sized
    # buffers rather than four. Fixed and older runtimes retain their allowance.
    field_count=3 if automatic and staged_readback and it else (4 if automatic else 3)
    device=static+uniforms+field_bytes*field_count+1024
    return dict(cells=cells,grid_x=gx,grid_y=gy,storage=storage,uniform_bytes=uniforms,
                host_bytes=host,device_bytes=device,borrowed_payload_bytes=borrowed_payload,device_field_count=field_count,estimated_bytes=host+device)

def require(kind, points, slots, edges, iterations, *, automatic=False,borrowed_inputs=False,staged_readback=False,fixed_uniforms=False,packed_sparse=False,
            storage_bytes=STORAGE_BYTES,buffer_bytes=BUFFER_BYTES,
            budget_bytes=None,device_working_bytes=WORKING_BYTES):
    p=plan(kind,points,slots,edges,iterations,automatic=automatic,borrowed_inputs=borrowed_inputs,staged_readback=staged_readback,fixed_uniforms=fixed_uniforms,packed_sparse=packed_sparse)
    for name,size in p['storage'].items():
        if size>min(storage_bytes,buffer_bytes):
            raise ValueError(f'GPU transport {name} exceeds requested device buffer limit')
    if p['uniform_bytes']>buffer_bytes:
        raise ValueError('GPU transport uniforms exceed requested device buffer limit')
    if budget_bytes is None:
        from .field_memory import bind_phase_budget
        budget_bytes = bind_phase_budget.get()
        # Unknown/older runtimes retain their bounded device policy. The
        # capability-qualified runtime uses native buffer checks plus host
        # admission and actual device errors, not an invented total-VRAM cap.
        if device_working_bytes is not None and p['device_bytes'] > device_working_bytes:
            raise ValueError(f"GPU transport device working-set research budget exceeded: {p['device_bytes']} bytes > {device_working_bytes} bytes")
    if budget_bytes is not None and p['estimated_bytes']>budget_bytes:
        raise ValueError(f"GPU transport working-set research budget exceeded: {p['estimated_bytes']} bytes > {budget_bytes} bytes")
    return p
