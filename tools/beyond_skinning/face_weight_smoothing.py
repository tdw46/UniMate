"""Independent bounded face-weight smoothing with per-thread OS random state.

Explicit seeds and entropy owners remain available for reproducible controls.
The prepared-order finalizer is for component diagnostics. No original calls;
global native wrapper initialization and unrestricted public API remain open.
"""
import numpy as np
import sys
F=np.float32

def accumulate(weights,bones,edges,selected,factor,passes):
    w=np.asarray(weights,'f4');ids=np.asarray(bones,'i4');mask=np.asarray(selected,bool);factor=F(factor)
    if w.ndim!=2 or w.shape!=ids.shape or mask.shape!=(len(w),) or len(w)>4096 or not np.isfinite(w).all() or not 0<=factor<=1 or not 0<=passes<=16:raise ValueError('weight smoothing research bounds')
    rows=[]
    for ws,bs in zip(w,ids):
        row={}
        for value,b in zip(ws,bs):
            if b>=0 and value>0:row[int(b)]=F(row.get(int(b),F(0))+value)
        rows.append(row)
    adj=[[] for _ in rows]
    for a,b in np.asarray(edges,'i8').reshape(-1,2):
        if 0<=a<len(rows) and 0<=b<len(rows) and a!=b:adj[a].append(int(b));adj[b].append(int(a))
    for _ in range(passes):
        old=rows;rows=[r.copy() for r in old]
        for i,near in enumerate(adj):
            if not mask[i] or factor<=0 or not near:continue
            row={b:F(F(1-factor)*value) for b,value in old[i].items()};inv=F(F(1)/F(len(near)))
            for j in near:
                for b,value in old[j].items():row[b]=F(row.get(b,F(0))+F(inv*F(factor*value)))
            rows[i]={b:value for b,value in row.items() if value>0}
    return rows

def finalize(weights,bones,rows,selected,top_k,orders):
    """Normalize in explicit hash iteration order, then weight/bone-descending sort and truncate."""
    w=np.asarray(weights,'f4').copy();ids=np.asarray(bones,'i4').copy();k=w.shape[1]
    if len(rows)!=len(w) or len(orders)!=len(w) or not 0<=top_k<=k:raise ValueError('invalid prepared weight rows')
    for i,row in enumerate(rows):
        if not selected[i]:continue
        order=orders[i]
        if len(order)!=len(row) or set(order)!=set(row):raise ValueError('prepared traversal does not match computed keys')
        total=F(-0.)
        for b in order:total=F(total+row[b])
        pairs=[(b,F(row[b]/total) if total>0 else row[b]) for b in order]
        # Native partial_cmp treats unordered weights as tied, then uses
        # descending bone ID. Python tuple ordering does not do that for NaN.
        from functools import cmp_to_key
        def compare(a,b):
            if a[1]>b[1]:return -1
            if a[1]<b[1]:return 1
            return (a[0]<b[0])-(a[0]>b[0])
        pairs.sort(key=cmp_to_key(compare));pairs=pairs[:top_k]
        w[i]=0;ids[i]=-1
        for j,(b,value) in enumerate(pairs):ids[i,j]=b;w[i,j]=value
    return w,ids

def accumulate_ordered(weights,bones,edges,selected,factor,passes,hash_seed=None,*,random_states=None,bounded=True):
    """Reconstruct tables using owned entropy or an explicit test seed, never orders."""
    from .weight_hash_order import Table,M
    w=np.asarray(weights,'f4');ids=np.asarray(bones,'i4');mask=np.asarray(selected,bool);factor=F(factor)
    if w.ndim!=2 or w.shape!=ids.shape or mask.shape!=(len(w),) or (bounded and len(w)>4096) or not np.isfinite(w).all() or passes<0 or (bounded and (not 0<=factor<=1 or passes>16)):raise ValueError('weight smoothing research bounds')
    needs_component_bound=False
    if not bounded:
        from .face_memory import require
        live=int(np.unique(ids[(ids>=0)&(w>0)]).size)
        try:
            require(len(w),2*len(edges),slots=w.shape[1],influences=live,hash_tables=True)
        except ValueError as exc:
            if not str(exc).startswith('facial working-set research budget exceeded:'):raise
            # Only analyze support when the global all-bones/all-rows bound
            # refuses. Admit graph/planning scratch before constructing it.
            # Reserve per-vertex planning plus one component-wide owner set.
            # The existing table allowance exceeds those set/stack payloads.
            require(len(w),2*len(edges),slots=w.shape[1],influences=live,hash_tables=True,map_entries=live)
            needs_component_bound=True
    edge_rows=np.asarray(edges,'i8').reshape(-1,2)
    native_adj=getattr(sys.modules.get(__package__+'._bvt_solver'),'weight_adjacency',None)
    if native_adj is not None and edge_rows.flags.aligned:
        adj=native_adj(len(w),edge_rows)
    else:
        adj=[[] for _ in range(len(w))]
        for a,b in edge_rows:
            if 0<=a<len(w) and 0<=b<len(w) and a!=b:adj[a].append(int(b));adj[b].append(int(a))

    if needs_component_bound:
        entries=component_map_entries(w,ids,adj,mask)
        try:
            require(len(w),2*len(edges),slots=w.shape[1],influences=live,hash_tables=True,map_entries=entries)
        except ValueError as exc:
            if not str(exc).startswith('facial working-set research budget exceeded:'):raise
            # Connected components are an infinite-pass bound. Only after it
            # refuses, conservatively propagate support for the actual passes.
            # Planner scratch fits the already-admitted planning reservation;
            # a planner refusal keeps the original component-policy error.
            from .face_support import finite_pass_entries
            tighter=finite_pass_entries(w,ids,adj,mask,passes if not factor<=0 else 0,live,entries)
            if tighter is None:raise
            require(len(w),2*len(edges),slots=w.shape[1],influences=live,hash_tables=True,map_entries=tighter)

    from .random_state import DEFAULT_STATES
    if hash_seed is not None and random_states is not None:raise ValueError('choose explicit seed or random-state owner')
    if hash_seed is None:
        owner=DEFAULT_STATES if random_states is None else random_states
        def new():return Table(*owner.new())
    else:
        key0,key1=map(int,hash_seed)
        if not 0<=key0<=M or not 0<=key1<=M:raise ValueError('invalid random state')
        def new():
            nonlocal key0
            table=Table(key0,key1);key0=(key0+1)&M;return table
    native_initial=getattr(sys.modules.get(__package__+'._bvt_solver'),'weight_initial_row',None)
    # Unaligned/unusual byte-stride arrays retain NumPy's original scalar path.
    if not w.flags.aligned or not ids.flags.aligned:native_initial=None
    rows=[]
    for ws,bs in zip(w,ids):
        table=new()
        if native_initial is not None:
            slots,hashes=native_initial(ws,bs,*table.keys)
            table.slots=[None if entry is None else (entry[0],F(entry[1])) for entry in slots]
            table._hashes=hashes;table.count=sum(p is not None for p in table.slots)
            rows.append(table);continue
        values={}
        for value,b in zip(ws,bs):
            if b>=0 and value>0:
                b=int(b);values[b]=F(values.get(b,F(0))+value);table.set(b,values[b])
        rows.append(table)
    native_row=getattr(sys.modules.get(__package__+'._bvt_solver'),'weight_diffuse_row',None)
    for _ in range(passes):
        old=rows;rows=old.copy()
        for i,near in enumerate(adj):
            if not mask[i] or factor<=0 or not near:continue
            table=new()
            if native_row is not None:
                slots,hashes=native_row(old,near,i,float(factor),*table.keys)
                # Python rows keep f32 scalar values on every supported NumPy ABI.
                table.slots=[None if entry is None else (entry[0],F(entry[1])) for entry in slots]
                table._hashes=hashes;table.count=sum(p is not None for p in table.slots)
                rows[i]=table;continue
            values={};inv=F(F(1)/F(len(near)))
            for b,value in old[i].items():values[b]=F(F(1-factor)*value);table.set(b,values[b])
            for j in near:
                for b,value in old[j].items():
                    first=b not in values
                    values[b]=F(values.get(b,F(0))+F(inv*F(factor*value)))
                    if first:table.set(b,values[b])
            # The new table is private until this pass completes. Only first
            # insertion changes bucket/growth order; publish final sums once.
            table.slots=[(p[0],values[p[0]]) if p is not None else None for p in table.slots]
            table.prune();rows[i]=table
    return rows


def component_map_entries(weights,bones,adj,selected):
    """Conservative support bound; diffusion cannot cross graph components.

    Unselected rows keep their initial tables. Selected rows may acquire every
    positive source ID in their component, independent of pass count/factor.
    No entropy, weights, neighbor order or table construction is changed.
    """
    native=getattr(sys.modules.get(__package__+'._bvt_solver'),'weight_component_entries',None)
    if native is not None and all(isinstance(a,np.ndarray) and a.flags.aligned and a.dtype==np.dtype(dtype) for a,dtype in ((weights,'f4'),(bones,'i4'),(selected,'?'))):
        return native(weights,bones,adj,selected)
    visited=np.zeros(len(weights),bool);total=0
    for start in range(len(weights)):
        if visited[start]:continue
        visited[start]=True;stack=[start];owners=set();chosen=0;unchanged=0
        while stack:
            i=stack.pop();local=set(int(b) for b,w in zip(bones[i],weights[i]) if b>=0 and w>0)
            owners.update(local)
            if selected[i]:chosen+=1
            else:unchanged+=len(local)
            for j in adj[i]:
                if not visited[j]:visited[j]=True;stack.append(j)
        total+=unchanged+chosen*len(owners)
    return total


def _nonfinite_pairs(pairs,top_k):
    # Same ordered f32 normalization and Python comparator as finalize().
    from functools import cmp_to_key
    pairs=[(b,F(v)) for b,v in pairs];total=F(-0.)
    for _,v in pairs:total=F(total+v)
    pairs=[(b,F(v/total) if total>0 else v) for b,v in pairs]
    def compare(a,b):
        if a[1]>b[1]:return -1
        if a[1]<b[1]:return 1
        return (a[0]<b[0])-(a[0]>b[0])
    pairs.sort(key=cmp_to_key(compare));return pairs[:top_k]

def finalize_tables(weights,bones,tables,selected,top_k):
    """Consume owned table slots directly, avoiding dict/order copies per row."""
    native=getattr(sys.modules.get(__package__+'._bvt_solver'),'weight_finalize_tables',None)
    w=np.asarray(weights,'f4');ids=np.asarray(bones,'i4');mask=np.asarray(selected,bool)
    if native is not None and all(a.flags.aligned for a in (w,ids,mask)):
        return native(w,ids,tables,mask,top_k,_nonfinite_pairs)
    return finalize(w,ids,[dict(t.items()) for t in tables],mask,top_k,[[p[0] for p in t.items()] for t in tables])
