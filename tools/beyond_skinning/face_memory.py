"""Conservative facial scratch admission; not original allocator-error parity.

No fixed vertex/pass/work count caps. An optional diagnostic memory budget covers
Python graph/heap/table staging and overlapping owned arrays, excluding existing
solver caches and interpreter state. Call before constructing the owned scratch.
"""
import operator
from .field_memory import DEFAULT_BUDGET_BYTES, bind_phase_budget

def require(vertices, directed_edges, *, slots=0, influences=0, fields=0,
            hash_tables=False, budget_bytes=None, map_entries=None):
    n,e,k,b,f=map(operator.index,(vertices,directed_edges,slots,influences,fields))
    if min(n,e,k,b,f)<0:raise ValueError('negative facial working-set dimension')
    entries=n*b if map_entries is None else operator.index(map_entries)
    if entries<0 or (map_entries is not None and (not hash_tables or entries>n*b)):
        raise ValueError('invalid facial map entry bound')
    # Set/list graph construction, retained CSR + conversion, and a full
    # distance heap can overlap. Row/output copies and retained ring distance
    # fields are separate. Hash diffusion retains old/new maps plus finalizer
    # dictionaries/order lists. Default bounds every row by all live influence IDs;
    # callers may supply a verified smaller support count after planning admission.
    parts=dict(graph=256*n+128*e,paths=64*n+96*e,
               output=48*n*k,fields=4*n*f,
               maps=1024*n+384*entries if hash_tables else 0,
               hash_cache=192*entries if hash_tables else 0,
               reserve=1024*1024)
    estimate=(sum(parts.values())*5+3)//4
    budget_bytes = bind_phase_budget.get() if budget_bytes is None else operator.index(budget_bytes)
    if budget_bytes is not None and estimate>budget_bytes:
        raise ValueError(f'facial working-set research budget exceeded: {estimate} bytes > {budget_bytes} bytes')
    return dict(estimated_bytes=estimate,components=parts)
