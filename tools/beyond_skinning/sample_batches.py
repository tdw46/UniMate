"""Bound independent query transport around the unchanged recovered sampler."""
import operator
import zlib
import numpy as np
BLOCK_ENTRIES=65536
MAX_BLOCK_ROWS=1024

def direct_block_rows(sample_width, bones):
    columns = max(sample_width, bones, 1)
    return min(MAX_BLOCK_ROWS, max(1, BLOCK_ENTRIES // columns))


def sample_map_arrays(runtime,queries,points,radii,grid,k,kernel,volume,occupancy,*,prior_bytes=0,requested_k=None,borrowed_samples=False,on_allocate=None):
    k=operator.index(k);n=len(queries)
    if k<1:raise ValueError('sample map output research budget exceeded')
    from .field_memory import require_sample_map
    require_sample_map(n,k,prior_bytes=prior_bytes,borrowed_samples=borrowed_samples)
    ids=np.empty((n,k),'u4')
    if on_allocate is not None:on_allocate((ids,))
    weights=np.empty((n,k),'f4')
    if on_allocate is not None:on_allocate((weights,))
    width=min(MAX_BLOCK_ROWS,max(1,BLOCK_ENTRIES//k))
    transport = {}
    if 'cell_coordinates' in grid:
        transport['cell_coordinates'] = grid['cell_coordinates']
    if "array_output" in (getattr(runtime.sample_from_grid, "__text_signature__", "") or ""):
        transport["array_output"] = True
    prepare=getattr(runtime,'_PreparedOccupancy',None)
    prepared=None
    prepare_grid=getattr(runtime,'_PreparedSampleGrid',None)
    prepared_grid=None
    # Snapshot only a bounded ordinary grid; larger/unknown storage retains
    # the original validated block route instead of growing retained scratch.
    buffers=[points,radii,grid['rows'],grid['columns']]
    if 'cell_coordinates' in grid:buffers.append(grid['cell_coordinates'])
    snapshot_bytes=sum(getattr(a,'nbytes',8*1024*1024+1) for a in buffers)
    use_grid=prepare_grid is not None and snapshot_bytes<=min(8*1024*1024,128*len(points))
    for a in range(0,max(n,1),width):
        if prepared is not None:runtime.check_cancel()
        z=min(n,a+width)
        if prepared_grid is not None:
            ii,ww=prepared_grid.query(queries[a:z],k,kernel,volume,float(grid['cell']),requested_k=requested_k,prepared_occupancy=prepared)
        else:
            ii,ww=runtime.sample_from_grid(queries[a:z],points,radii,grid['rows'],grid['columns'],grid['origin'],grid['dimensions'],grid['cell'],k,kernel,volume,float(grid['cell']),occupancy if prepared is None else None,allow_nonfinite_queries=True,**transport,**({} if requested_k is None else dict(requested_k=requested_k)))
        if z > a:
            block_ids = ids[a:z]
            block_ids[:] = ii
            weights[a:z] = ww
        del ii, ww
        if z > a:
            block_ids[block_ids == 0xffffffff] = 0
        # Preserve the first block's existing extraction/validation order. No
        # global/cloud cache owner: release this snapshot when this call exits.
        if a==0 and z<n and occupancy is not None and prepare is not None:
            prepared=prepare(occupancy)
            transport['prepared_occupancy']=prepared
        if a==0 and z<n and use_grid and (occupancy is None or prepared is not None):
            prepared_grid=prepare_grid(points,radii,grid['rows'],grid['columns'],grid['origin'],grid['dimensions'],grid['cell'],cell_coordinates=grid.get('cell_coordinates'))
    return ids,weights


def sample_reduced(runtime,field,ids,weights,bones,top_k):
    """Bound sampled dense rows and emit final owned weights without scalar lists."""
    from .compact_field import CompactField
    fused=getattr(runtime,'sample_compact_reduced',None)
    if isinstance(field,CompactField) and callable(fused) and len(ids)>0 and top_k>0:
        ii,ww=fused(ids,weights,field.groups,field.values,bones,top_k)
        return ww,ii
    n=len(ids);out_bones=np.empty((n,top_k),'i4');out_weights=np.empty((n,top_k),'f4')
    width=min(MAX_BLOCK_ROWS,max(1,BLOCK_ENTRIES//max(bones,1)))
    for a in range(0,n,width):
        z=min(n,a+width)
        sampled=field.sample(runtime,ids[a:z],weights[a:z]) if isinstance(field,CompactField) else runtime.accumulate_sample_array(ids[a:z],weights[a:z],field)
        ii,ww=runtime.reduce_sample_array(sampled,top_k)
        out_bones[a:z]=ii;out_weights[a:z]=ww
        del sampled,ii,ww
    return out_weights,out_bones

class PackedSampleMap:
    """Lossless blocks of normalized ID and weight rows from one cloud."""
    def __init__(self, shape, rows, blocks, replay=None):
        self.shape = shape
        self.rows = rows
        self.blocks = tuple(blocks)
        self.replay = replay
        self.nbytes = sum(block.nbytes for block in self.blocks)

    @staticmethod
    def encode(ids, factors, inverse):
        highest = int(ids.max(initial=0))
        id_bytes = 1 if highest < 256 else 2 if highest < 65536 else 4
        compact_ids = ids.astype(f'u{id_bytes}', copy=False)
        # Delta-code unsigned weight bits along the neighbor axis. Modular
        # integer addition reverses this even for signed zeros and NaN payloads.
        bits = factors.view('u4')
        delta = bits.copy()
        np.subtract(bits[:, 1:], bits[:, :-1], out=delta[:, 1:])
        prefix = len(ids).to_bytes(4, 'little') + bytes([id_bytes])
        if inverse is not None:
            prefix += inverse.tobytes()
        # Group equally significant bytes to expose repeated high bytes to zlib.
        raw = b''.join((prefix, compact_ids.view('u1').reshape(-1,id_bytes).T.tobytes(),
                        delta.view('u1').reshape(-1,4).T.tobytes()))
        return np.frombuffer(zlib.compress(raw, level=1), dtype='u1')

    def decoded(self):
        n, width = self.shape
        for index, block in enumerate(self.blocks):
            start = index * self.rows
            stop = min(n, start + self.rows)
            if block.ndim == 2:
                queries, inverse = distinct_queries(block)
                ids, factors = self.replay(queries)
                yield slice(start, stop), ids, factors, inverse
                continue
            raw = zlib.decompress(block)
            unique = int.from_bytes(raw[:4], 'little')
            inverse = None
            id_bytes = raw[4]
            offset = 5
            if unique < stop - start:
                inverse = np.frombuffer(raw, 'u2', count=stop-start, offset=offset)
                offset += inverse.nbytes
            count = unique * width
            planes = np.frombuffer(raw, 'u1', count=count*id_bytes, offset=offset).reshape(id_bytes,count)
            ids = planes.T.copy().view(f'u{id_bytes}').reshape(unique,width).astype('u4',copy=False)
            planes = np.frombuffer(raw, 'u1', offset=offset+count*id_bytes).reshape(4,count)
            bits = planes.T.copy().view('u4').reshape(unique,width)
            np.cumsum(bits, dtype='u4', axis=1, out=bits)
            factors = bits.view('f4')
            yield slice(start, stop), ids, factors, inverse

    def materialize(self, writable):
        from .field_memory import require_sample_map
        require_sample_map(*self.shape, prior_bytes=self.nbytes)
        ids = np.empty(self.shape, 'u4')
        factors = np.empty(self.shape, 'f4')
        for block, ii, ww, inverse in self.decoded():
            ids[block] = ii if inverse is None else ii[inverse]
            factors[block] = ww if inverse is None else ww[inverse]
        ids.flags.writeable = factors.flags.writeable = writable
        return ids, factors

def sample_owner_bytes(samples):
    if samples is None:
        return 0
    if isinstance(samples, PackedSampleMap):
        return samples.nbytes
    return sum(array.nbytes for array in samples)

def distinct_queries(queries):
    # Group only bit-identical normalized positions within this bounded block.
    # Sampling is independent for each query, including signed zeros and NaNs.
    packed = np.ascontiguousarray(queries)
    keys = packed.view(np.dtype((np.void, packed.dtype.itemsize*packed.shape[1]))).ravel()
    _, indices, inverse = np.unique(keys, return_index=True, return_inverse=True)
    if len(indices) == len(keys):
        return queries, None
    return packed[indices], inverse.reshape(-1).astype('u2')
