"""Bounded float32 surface geodesic used to connect projected mouth points.

Native 0xe0108: stable distance-only binary heap, right child wins heap-pop ties.
Not the complete mouth weighting pass.
"""
import numpy as np
F=np.float32

def shortest_path(rows,columns,weights,start,end,limit,blocked=None,*,allow_nonfinite_edges=False,bounded=True):
 r=np.asarray(rows,'i8');c=np.asarray(columns,'i8');w=np.asarray(weights,'f4');n=len(r)-1
 if n<0 or (bounded and n>10000) or r.ndim!=1 or c.ndim!=1 or w.shape!=c.shape or r[0]!=0 or r[-1]!=len(c) or np.any(np.diff(r)<0) or np.any(c<0) or np.any(c>=n) or (not allow_nonfinite_edges and not np.isfinite(w).all()) or np.any(w<0):raise ValueError('invalid bounded surface graph')
 mask=np.zeros(n,'u1') if blocked is None else np.asarray(blocked,'u1')
 if mask.shape!=(n,):raise ValueError('blocked mask length mismatch')
 if not 0<=start<n or not 0<=end<n:return None
 if mask[start]&1 or mask[end]&1:return None
 if not bounded:
  from .face_memory import require
  require(n,len(c))
 dist=np.full(n,np.inf,'f4');parent=np.full(n,-1,'i8');dist[start]=0;heap=[(start,F(0))];limit=F(limit)
 def push(item):
  i=len(heap);heap.append(item)
  while i:
   j=(i-1)//2
   if heap[j][1]<=item[1]:break
   heap[i]=heap[j];i=j
  heap[i]=item
 def pop():
  result=heap[0];last=heap.pop()
  if heap:
   i=0;nheap=len(heap)
   while 2*i+1<nheap:
    j=2*i+1
    if j+1<nheap and heap[j+1][1]<=heap[j][1]:j+=1
    heap[i]=heap[j];i=j
   while i:
    j=(i-1)//2
    if heap[j][1]<=last[1]:break
    heap[i]=heap[j];i=j
   heap[i]=last
  return result
 while heap:
  i,d=pop()
  if d>dist[i]:continue
  if i==end:break
  for k in range(r[i],r[i+1]):
   j=c[k]
   if mask[j]&1:continue
   value=F(d+w[k])
   if value<=limit and value<dist[j]:dist[j]=value;parent[j]=i;push((int(j),value))
 if not np.isfinite(dist[end]):return None
 path=[int(end)]
 while path[-1]!=start:path.append(int(parent[path[-1]]))
 return path[::-1]


def surface_distances(rows,columns,weights,seeds,limit,*,allow_nonfinite_edges=False,bounded=True):
 """Limited multi-source distance field with initial projection offsets (0xdfc04)."""
 import heapq
 r=np.asarray(rows,'i8');c=np.asarray(columns,'i8');w=np.asarray(weights,'f4');n=len(r)-1
 if n<0 or (bounded and n>10000) or r.ndim!=1 or c.ndim!=1 or w.shape!=c.shape or r[0]!=0 or r[-1]!=len(c) or np.any(np.diff(r)<0) or np.any(c<0) or np.any(c>=n) or (not allow_nonfinite_edges and not np.isfinite(w).all()) or np.any(w<0):raise ValueError('invalid bounded surface graph')
 if not bounded:
  from .face_memory import require
  require(n,len(c))
 dist=np.full(n,np.inf,'f4');heap=[];limit=F(limit)
 for i,d in seeds:
  d=F(d)
  if 0<=i<n and d<=limit and d<dist[i]:dist[i]=d;heapq.heappush(heap,(float(d),int(i)))
 while heap:
  d,i=heapq.heappop(heap);d=F(d)
  if d>dist[i]:continue
  for k in range(r[i],r[i+1]):
   j=c[k];value=F(d+w[k])
   if value<=limit and value<dist[j]:dist[j]=value;heapq.heappush(heap,(float(value),int(j)))
 return dist
