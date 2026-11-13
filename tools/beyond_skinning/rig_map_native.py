"""Recovered ARP vocabulary and geometric side admission for public rig_map.

Literal vocabulary from 0x57c090; validation from 0xe4c20; corner partition
from 0xe309c/0x1798ac. No dependency on the original extension.
"""
import re
from collections import defaultdict
import numpy as np

# (kind, paired, auxiliary, patterns). Pattern order is significant.
PATTERNS=(
 ('skull',False,False,('c_skull_#.x',)),
 ('jaw',False,False,('jawbone.x','c_jawbone.x')),
 ('chin',False,False,('c_chin_#.x',)),('nose',False,False,('c_nose_#.x',)),
 ('tongue',False,False,('tong_#.x',)),
 ('teeth_top',False,False,('c_teeth_top.x',)),('teeth_top',True,False,('c_teeth_top.$',)),
 ('teeth_bot',False,False,('c_teeth_bot.x',)),('teeth_bot',True,False,('c_teeth_bot.$',)),
 ('lip_top',False,False,('c_lips_top.x',)),('lip_top',True,False,('c_lips_top.$','c_lips_top_#.$')),
 ('lip_bot',False,False,('c_lips_bot.x',)),('lip_bot',True,False,('c_lips_bot.$','c_lips_bot_#.$')),
 ('lip_corner',True,False,('c_lips_corner_mini.$','c_lips_smile.$')),
 ('eyelid_top',True,False,('c_eyelid_top_#.$','c_eyelid_twk_top.$')),
 ('eyelid_bot',True,False,('c_eyelid_bot_#.$','c_eyelid_twk_bot.$')),
 ('eyelid_tweak',True,True,('c_eyelid_twk_top.$','c_eyelid_twk_bot.$')),
 ('eyelid_corner',True,False,('c_eyelid_corner_#.$',)),
 ('eye',True,False,('c_eye.$','c_eye_offset.$','c_eye_ref_track.$')),
 ('eye_main',True,True,('c_eye.$',)),('eye_card',True,True,('c_eye_ref_track.$',)),
 ('brow',True,False,('c_eyebrow_#.$','c_eyebrow_#_end.$','c_eyebrow_full.$')),
 ('cheek',True,False,('c_cheek_smile.$','c_cheek_inflate.$')),('ear',True,False,('c_ear_#.$',)),
)

def _tokens(name):
 name=name.split(':',1)[-1];out=[];word='';previous=''
 for c in name:
  if c.isascii() and c.isalnum():
   if (previous.isascii() and ((previous.islower() and (c.isupper() or c.isdigit())) or (previous.isupper() and c.isdigit()) or (previous.isdigit() and c.isalpha()))):
    if word:out.append(word.lower());word=''
   word+=c
  else:
   if word:out.append(word.lower());word=''
   if c.isalpha():out.append(c.lower())
  previous=c
 if word:out.append(word.lower())
 return out

def eligible(names,parents,deform,head):
 from . import _bvt_solver as rt
 raw=np.asarray(deform).view('u1');mask=[]
 for i in range(len(names)):
  if head<0:mask.append(bool(int(raw[i])&1));continue
  if raw[i]!=1 or i==head:mask.append(False);continue
  current=i;allowed=False
  for _ in range(len(parents)):
   current=int(parents[current])
   if current<0:break
   if current==head:allowed=True;break
   if current>=len(names):raise rt.PanicException(f'index out of bounds: the len is {len(names)} but the index is {current}')
   if any(t in ('head','face','facial','skull','頭') for t in _tokens(names[current])):allowed=True;break
  mask.append(allowed)
 return mask

def mean(points):
 out=np.zeros(3,'f4')
 for point in points:out=np.add(out,np.divide(point,np.float32(len(points)),dtype='f4'),dtype='f4')
 return out

def squared(v):
 return np.float32(np.float32(v[0]*v[0]+v[1]*v[1])+v[2]*v[2])

def dictionary(names,parents,deform,segments,head):
 mask=eligible(names,parents,deform,head)
 normalized=[n.split(':',1)[-1].strip().lower() if m else '' for n,m in zip(names,mask)]
 groups=defaultdict(list)
 for kind,paired,aux,patterns in PATTERNS:
  for side in ('l','r') if paired else ('x',):
   expressions=[re.compile(re.escape(p.replace('$',side)).replace(r'\#','[0-9]+')+r'\Z') for p in patterns]
   ids=[i for i,n in enumerate(normalized) if any(e.fullmatch(n) for e in expressions)]
   # Unnumbered patterns have rank zero; numbered patterns sort by their capture.
   if any('#' in p for p in patterns):
    def rank(i):
     for pattern in patterns:
      pattern=pattern.replace('$',side)
      match=re.fullmatch(re.escape(pattern).replace(r'\#','([0-9]+)'),normalized[i])
      if match:return (int(match[1]) if '#' in pattern else 0,i)
    ids=[i for i in ids if rank(i)[0] <= 0x7fffffffffffffff]
    ids.sort(key=rank)
   groups[kind,side].extend(ids)
 groups={k:v for k,v in groups.items() if v}
 rejected=set()
 for kind in dict.fromkeys(k[0] for k in groups):
  sides={sd:ids for (k,sd),ids in groups.items() if k==kind}
  if kind in ('eyelid_top','eyelid_bot') and any(len(ids)<2 for ids in sides.values()):rejected.add(kind);continue
  if not ('l' in sides or 'r' in sides):continue
  left,right=sides.get('l',[]),sides.get('r',[])
  if not left or len(left)!=len(right):rejected.add(kind);continue
  a,b=segments[left,1],segments[right,1];ca,cb=mean(a),mean(b)
  points=np.concatenate((a,b));extent=np.max(np.max(points,axis=0)-np.min(points,axis=0))
  distance=np.sqrt(squared(ca-cb),dtype='f4')
  if distance<=np.float32(extent*np.float32(.05)):rejected.add(kind);continue
  if any(squared(pt-cb)<squared(pt-ca) for pt in a) or any(squared(pt-ca)<squared(pt-cb) for pt in b):rejected.add(kind)
 if rejected.intersection(('eyelid_top','eyelid_bot')):rejected.add('eyelid_corner')
 return {k:ids for k,ids in groups.items() if k[0] not in rejected}

def corners(groups,segments):
 centers={}
 for side in ('l','r','x'):
  ids=groups.get(('eyelid_top',side),[])+groups.get(('eyelid_bot',side),[])+groups.get(('eyelid_corner',side),[])
  if ids:centers[side]=mean(segments[ids,1])
 result={}
 for (kind,side),ids in groups.items():
  if kind!='eyelid_corner':continue
  opposite={'l':'r','r':'l'}.get(side)
  geometric=side in centers and opposite in centers
  if geometric:
   center=centers[side];direction=centers[opposite]-center
   vals=[]
   for i in ids:
    v=segments[i,1]-center
    vals.append(np.float32(np.float32(v[0]*direction[0]+v[1]*direction[1])+v[2]*direction[2]))
   threshold=np.float32((min(vals)+max(vals))*np.float32(.5)) if len(vals)>1 else np.float32(0)
   inner=[i for i,v in zip(ids,vals) if not v<threshold];outer=[i for i,v in zip(ids,vals) if v<threshold]
  else:inner,outer=ids[:1],ids[1:]
  result[side]=(sorted(inner),sorted(outer),geometric)
 return result

def mesh_parts(names):
 result=[]
 for i,name in enumerate(names or ()):
  name=name.lower()
  for label,words in (('teeth',('teeth','tooth','denture','歯')),('body',('brow','lash','lip','hair','skin','face','head','body','mouth')),('eye_side',('brow','lash','hair'))):
   if any(w in name for w in words):result.append((i,label))
 return result
