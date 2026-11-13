"""Conservative finite-pass support only after global/component admission refuses.

No arithmetic weights, hash tables or entropy are touched. Never removes a live
ID: keeping self support even at factor1 and ignoring pruning/underflow ensures
the final union bounds every initial/intermediate table, including temporary keys.
"""
import operator
import numpy as np


def finite_pass_entries(weights,bones,adj,selected,passes,live,component_entries,*,work_limit=8_000_000):
    n=len(weights);passes=operator.index(passes);live=operator.index(live)
    if passes<0 or live<0:raise ValueError('negative support planning dimension')
    # Two row lists and conservatively sized Python integer bitsets, one ID->bit
    # dictionary, and per-row temporary integers. This fits within the existing
    # admitted1024*n+576*live map/planning reserve, or this optimization declines.
    integer_bytes=64+8*((live+29)//30)
    scratch=2*n*(8+integer_bytes)+128*live+4*integer_bytes+256
    if scratch>1024*n+576*live:return None
    if n*max(1,weights.shape[1])>work_limit:return None
    active_ids=np.unique(bones[(bones>=0)&(weights>0)])
    if len(active_ids)!=live:raise ValueError('support planning influence mismatch')
    bit_index={int(b):i for i,b in enumerate(active_ids)}
    bits=[]
    for ws,bs in zip(weights,bones):
        value=0
        for w,b in zip(ws,bs):
            if b>=0 and w>0:value|=1<<bit_index[int(b)]
        bits.append(value)
    total=sum(value.bit_count() for value in bits)
    if total>=component_entries:return None
    spent=n*max(1,weights.shape[1])
    for _ in range(passes):
        work=n+sum(len(near) for i,near in enumerate(adj) if selected[i])
        if spent+work>work_limit:return None
        spent+=work;old=bits;bits=old.copy();changed=False
        for i,near in enumerate(adj):
            if not selected[i]:continue
            value=old[i]
            for j in near:value|=old[j]
            if value!=old[i]:changed=True;total+=value.bit_count()-old[i].bit_count();bits[i]=value
        if total>=component_entries:return None
        if not changed:break
    return total
