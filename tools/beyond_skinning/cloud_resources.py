"""Research lazy resources owned by one immutable cloud generation.

Graph cache keys are the three exact f32 bit patterns (0x146f4). Sample grid
and occupancy have no tuning key. Public callers receive owned copies; internal
bind consumers borrow these values read-only. Threaded original cache ownership,
generalized query-map modes, sparse occupancy construction and cancellation remain open.
"""
import copy
import weakref
from functools import partial
from threading import RLock
import numpy as np
from .sample_batches import PackedSampleMap, sample_owner_bytes
class CloudResources:
    # Internal binding may borrow normalized immutable maps; public copies stay default.
    borrowed_sample_maps=True
    def __init__(self,runtime,points,levels,radii):
        self._large_sampling_levels=None
        self.runtime=runtime
        self.points=np.array(points,'f4',copy=True);self.levels=np.array(levels,'u1',copy=True);self.radii=np.array(radii,'f4',copy=True)
        for a in (self.points,self.levels,self.radii):a.flags.writeable=False
        self._graph_key=None;self._graph=None;self._grid=None;self._occupancy=None
        self.build_counts=dict(graph=0,grid=0,occupancy=0)
        self._sample_lock=RLock();self._sample_key=None;self._samples=None;self.sample_map_builds=0
        self._sample_owners=weakref.WeakValueDictionary()
    def graph(self,multiplier=10.,power=-3.,cull=.85,*,copy_output=True):
        parameters=np.array([multiplier,power,cull],'f4')
        if not np.isfinite(parameters).all():raise ValueError('nonfinite graph cache parameters')
        key=parameters.tobytes()
        if key!=self._graph_key:
            graph=self.runtime.build_graph(self.points,self.levels,self.radii,*map(float,parameters))
            self._graph=graph;self._graph_key=key;self.build_counts['graph']+=1
        return copy.deepcopy(self._graph) if copy_output else self._graph
    def grid(self,*,copy_output=True):
        if self._grid is None:
            self._grid=self.runtime.build_sample_grid(self.points,self.radii,allow_sparse=True);self.build_counts['grid']+=1
        return copy.deepcopy(self._grid) if copy_output else self._grid
    def occupancy(self,*,copy_output=True):
        if self._occupancy is None:
            self._occupancy=self.runtime.build_occupancy(self.points,self.levels,self.radii);self.build_counts['occupancy']+=1
        return copy.deepcopy(self._occupancy) if copy_output else self._occupancy

    def sample_cache_bytes(self):
        with self._sample_lock:
            return sample_owner_bytes(self._samples)

    def live_sample_bytes(self):
        """Count distinct shared-map allocations, including retired borrowers.

        Weak references observe ownership without extending a map's lifetime.
        Public writable copies are caller-owned and outside this cache ledger.
        """
        with self._sample_lock:
            owners = tuple(self._sample_owners.values())
            return sum(owner.nbytes for owner in owners)

    def _track_sample_owners(self,arrays):
        # Allocation tracking is separate from cache publication. Failed and
        # in-flight builds never become cache hits; weak owners reflect the
        # actual lifetime of their arrays, including retained error tracebacks.
        with self._sample_lock:
            for array in arrays:
                self._sample_owners[id(array)] = array

    def sample_owner_snapshot(self,samples=None):
        """One coherent snapshot, not a reservation against later allocations."""
        with self._sample_lock:
            included=samples if samples is not None else self._samples
            prior=sample_owner_bytes(included)
            live=self.live_sample_bytes()
            return dict(included_bytes=prior,other_bytes=max(0,live-prior),live_bytes=live)

    @staticmethod
    def _map_key(queries,k,kernel,volume,requested_k=None,cache_k=None):
        key=(len(queries),int(k),int(kernel),bool(volume))
        if requested_k is not None:key=(*key,int(requested_k))
        if cache_k is not None:key=(*key,int(cache_k))
        return key

    def cached_sample_map(self,queries,k,kernel,volume,*,cache_k=None):
        """Strong immutable owners for an exact internal retained-map hit.

        Holding this tuple keeps the selected generation alive if the slot is
        replaced. No work or cache publication occurs on a miss.
        """
        key=self._map_key(queries,k,kernel,volume,cache_k=cache_k)
        with self._sample_lock:
            if self._samples is not None and self._sample_key==key:
                return self._samples
        return None

    @staticmethod
    def _dense_sampling_record(row):
        return (row['origin'],row['inverse'],row['cell'],row['solid_minimum'],
                row['solid_dimensions'],row['solid_bits'],row['dilated_minimum'],
                row['dilated_dimensions'],row['dilated_bits'])

    def _sampling_occupancy(self):
        if self._large_sampling_levels is not None:
            return self._large_sampling_levels
        try:
            return [self._dense_sampling_record(row) for row in self.occupancy(copy_output=False)]
        except ValueError as exc:
            if str(exc)!='Sparse or oversized occupancy is not reconstructed':
                raise
        result=[]
        # Preserve encounter order. Each level owns one solid set and the
        # same 27-neighbor dilation; the sparse reader evaluates it on demand.
        levels,first=np.unique(self.levels,return_index=True)
        for level in levels[np.argsort(first)]:
            self.runtime.check_cancel()
            mask=self.levels==level
            points=self.points[mask]
            radii=self.radii[mask]
            cell=(radii[0]+radii[0])/np.float32(1.7320508)
            inverse=np.float32(1)/cell
            center=points[0]
            grid=(points-center)*inverse
            absolute=np.floor(np.abs(grid.astype('f8'))+.5)
            coords=np.clip(np.copysign(absolute,grid),-2**31,2**31-1).astype('i4')
            span=np.ptp(coords.astype('i8'),axis=0)+3
            if int(span[0])*int(span[1])*int(span[2])<=16_000_000:
                dense=self.runtime.build_occupancy(points,self.levels[mask],radii)
                result.append(self._dense_sampling_record(dense[0]))
            else:
                origin=center-cell*np.float32(.5)
                result.append((origin.tolist(),float(inverse),float(cell),(1,coords.tolist()),None))
        self._large_sampling_levels=result
        self.build_counts['occupancy']+=1
        return result

    def sample_direct_reduced(self, queries, k, requested_k, kernel, volume, field, bones, top_k):
        """Consume each direct query block before building the next one."""
        from .sample_batches import direct_block_rows, sample_map_arrays

        grid = self.grid(copy_output=False)
        occupancy = self._sampling_occupancy()
        prior = self.sample_cache_bytes()
        rows = direct_block_rows(k, bones)
        weights = np.empty((len(queries), top_k), 'f4')
        indices = np.empty((len(queries), top_k), 'i4')
        for start in range(0, max(1, len(queries)), rows):
            block = slice(start, start + rows)
            ids, factors = sample_map_arrays(
                self.runtime, queries[block], self.points, self.radii, grid, k,
                kernel, volume, occupancy, prior_bytes=prior, requested_k=requested_k,
                borrowed_samples=True, on_allocate=self._track_sample_owners)
            if len(ids):
                sampled = self.runtime.accumulate_sample_array(ids, factors, field)
                indices[block], weights[block] = self.runtime.reduce_sample_array(sampled, top_k)
                del sampled
            del ids, factors
        with self._sample_lock:
            self.sample_map_builds += 1
        return weights, indices

    def sample_packed_reduced(self, queries, k, cache_k, kernel, volume, field,
                              bones, top_k, lease, payload_budget, *, allow_query_blocks=False):
        from .sample_batches import direct_block_rows, distinct_queries, sample_map_arrays, sample_reduced
        weights = np.empty((len(queries), top_k), 'f4')
        indices = np.empty((len(queries), top_k), 'i4')
        def reduce_block(block, ids, factors, inverse):
            ww, ii = sample_reduced(self.runtime, field, ids, factors, bones, top_k)
            weights[block] = ww if inverse is None else ww[inverse]
            indices[block] = ii if inverse is None else ii[inverse]

        if lease is not None:
            for block, ids, factors, inverse in lease.decoded():
                reduce_block(block, ids, factors, inverse)
            return weights, indices

        # Cache keys permit a new rig to reuse the same spatial map.
        rows = direct_block_rows(k, 1)
        grid = self.grid(copy_output=False)
        occupancy = self._sampling_occupancy()
        blocks = []
        used = 0
        replay = None
        # Keep enough space for exact f32 coordinates of any remaining rows.
        # A warm lease can resample those coordinates against its own cloud.
        can_replay = allow_query_blocks and queries.nbytes <= payload_budget
        owner = weakref.WeakMethod(self._track_sample_owners)
        def track_replay(arrays):
            # A retained lease can outlive the cache that published it.
            observe = owner()
            if observe is not None:
                observe(arrays)
        for start in range(0, len(queries), rows):
            block = slice(start, start + rows)
            distinct, inverse = distinct_queries(queries[block])
            ids, factors = sample_map_arrays(
                self.runtime, distinct, self.points, self.radii, grid, k, kernel,
                volume, occupancy, borrowed_samples=True, on_allocate=self._track_sample_owners)
            if replay is None:
                packed = PackedSampleMap.encode(ids, factors, inverse)
                end = min(start + rows, len(queries))
                tail_bytes = (len(queries)-end)*queries.itemsize*queries.shape[1]
                if can_replay and used + packed.nbytes + tail_bytes > payload_budget:
                    replay = partial(sample_map_arrays, self.runtime, points=self.points,
                                     radii=self.radii, grid=grid, k=k, kernel=kernel,
                                     volume=volume, occupancy=occupancy, borrowed_samples=True,
                                     on_allocate=track_replay)
            if replay is not None:
                packed = queries[block].copy(order='C')
                packed.flags.writeable = False
            used += packed.nbytes
            if used > payload_budget:
                raise ValueError('retained sample map working-set research budget exceeded')
            blocks.append(packed)
            self._track_sample_owners((packed,))
            reduce_block(block, ids, factors, inverse)
            del ids, factors

        samples = PackedSampleMap((len(queries), k), rows, blocks, replay)
        key = self._map_key(queries, k, kernel, volume, cache_k=cache_k)
        with self._sample_lock:
            self._sample_key, self._samples = key, samples
            self.sample_map_builds += 1
        return weights, indices

    def sample_map(self,queries,k,kernel,volume,*,retain=True,requested_k=None,cache_k=None,copy_output=True):
        """One native map slot per immutable cloud, keyed by count and options.

        Query positions are admitted by the owning cloud cache, including its
        NaN-ignoring tolerance. They are deliberately not a second map key.
        """
        key=self._map_key(queries,k,kernel,volume,requested_k,cache_k)
        with self._sample_lock:
            hit = self._samples if retain and key == self._sample_key else None
        if isinstance(hit, PackedSampleMap):
            return hit.materialize(copy_output)
        if hit is not None:
            return tuple(a.copy() for a in hit) if copy_output else hit
        # Queries are independent in the recovered 0x5d370 sampling loop.
        # Keep native arithmetic, search radius and occupancy filtering exact;
        # bound only the nested Python scalar transport, never cloud context.
        from .sample_batches import sample_map_arrays
        grid=self.grid(copy_output=False)
        occupancy=self._sampling_occupancy()
        ids,weights=sample_map_arrays(self.runtime,queries,self.points,self.radii,grid,k,kernel,volume,occupancy,prior_bytes=self.sample_cache_bytes(),requested_k=requested_k,borrowed_samples=not copy_output,on_allocate=self._track_sample_owners)
        ids.flags.writeable=False;weights.flags.writeable=False
        with self._sample_lock:
            if retain:self._sample_key=key;self._samples=(ids,weights)
            self.sample_map_builds+=1
        return (ids.copy(),weights.copy()) if copy_output else (ids,weights)
