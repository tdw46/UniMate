import math
import numpy as np
import torch
from torch import nn
class DropPath(nn.Identity):
 def __init__(self,drop_prob=0.): super().__init__()
 def forward(self,x):
  if self.training: raise RuntimeError('This portable backend is inference-only')
  return x

def fps(pos,batch,ratio,random_start=True):
 # Greedy Euclidean FPS; same operation as torch_cluster, portable NumPy CPU.
 points=pos.detach().cpu().numpy(); batches=batch.detach().cpu().numpy(); out=[]
 for b in np.unique(batches):
  ids=np.flatnonzero(batches==b);p=points[ids];k=math.ceil(len(p)*ratio)
  distances=np.full(len(p),np.inf);i=int(torch.randint(len(p),(1,))) if random_start else 0
  for _ in range(k):
   out.append(ids[i]);distances=np.minimum(distances,np.sum((p-p[i])**2,axis=1));i=int(np.argmax(distances))
 return torch.tensor(out,dtype=torch.long,device=pos.device)
