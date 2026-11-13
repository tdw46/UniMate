"""Public clearance batching over independent bones; unchanged native predicate."""
import numpy as np

def _large_clearance(runtime, points, heads, tails, step):
    """Skip entire point blocks whose f32 distance lower bound is outside the capsule."""
    # In each axis the native projected displacement d*f lies between 0 and
    # d, including rounding. Bound q = point-head first, using the same f32
    # subtraction and square/sum order as the native predicate.
    step = np.float32(step)
    if not np.isfinite(step) or step <= 0 or not np.isfinite(heads).all() or not np.isfinite(tails).all():
        raise ValueError('invalid or oversized research clearance inputs')
    clear = np.ones(len(heads), 'u1')
    block_size = 256
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        direction = tails-heads
        threshold = step*step
        for offset in range(0,len(points),65536):
            runtime.check_cancel()
            chunk = points[offset:offset+65536]
            if not np.isfinite(chunk).all():
                raise ValueError('invalid or oversized research clearance inputs')
            # Spatially ordered small boxes. Keep transient indices and points
            # bounded independently of the number of rig bones or mesh vertices.
            ordered = chunk[np.argsort(chunk[:,2],kind='stable')]
            starts = np.arange(0,len(ordered),block_size)
            low = np.minimum.reduceat(ordered,starts,axis=0)
            high = np.maximum.reduceat(ordered,starts,axis=0)
            for i in np.flatnonzero(clear):
                if i%32==0:
                    runtime.check_cancel()
                delta = direction[i]
                below = np.minimum(np.float32(0),delta)-(high-heads[i])
                above = (low-heads[i])-np.maximum(np.float32(0),delta)
                distance = np.maximum(np.maximum(above,below),np.float32(0))
                square = distance*distance
                lower_bound = square[:,2]+(square[:,0]+square[:,1])
                # An indeterminate bound cannot discard a block.
                for start in starts[~(lower_bound>threshold)]:
                    if not runtime.seed_bone_clearance(ordered[start:start+block_size],
                                                       heads[i:i+1],tails[i:i+1],step)[0]:
                        clear[i]=0
                        break
    return clear

def public_clearance(runtime,points,heads,tails,step):
    # Every bone sees the complete cloud, in native point order. Bound both
    # per-call work and total work; this is independent of mask storage caps.
    finite=np.isfinite(heads).all(axis=1)&np.isfinite(tails).all(axis=1)
    if not finite.all():
        clear=np.ones(len(heads),'u1')
        if finite.any():
            clear[finite]=public_clearance(runtime,points,heads[finite],tails[finite],step)
        return clear
    n=len(points);bones=len(heads)
    if n*bones>64*1024*1024 or n>4_000_000:
        return _large_clearance(runtime,points,heads,tails,step)
    if not n or not bones or n*bones<=4_000_000:
        return np.frombuffer(runtime.seed_bone_clearance(points,heads,tails,step),'u1')
    width=max(1,4_000_000//n)
    result=np.empty(bones,'u1')
    for start in range(0,bones,width):
        runtime.check_cancel()
        stop=min(start+width,bones)
        result[start:stop]=np.frombuffer(runtime.seed_bone_clearance(points,heads[start:stop],tails[start:stop],step),'u1')
    return result
