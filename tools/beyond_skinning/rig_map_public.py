"""Independent rig-map assembly (0xe4c20), research-only supported routes.

Typed extraction and ownership are in rig_map_entry.rs. Native ARP patterns and
mesh-part routing are recovered. Active eye/mouth geometry is assembled by rig_map_surface.
"""
from collections import defaultdict
import numpy as np

ALL='pelvis spine neck head skull forehead brow eye eye_main eye_card eyelid_top eyelid_bot eyelid_corner_inner eyelid_corner_outer eyelid_tweak eyelid_glue cheek nose lip_top lip_bot lip_corner jaw chin teeth_top teeth_bot tongue ear finger'.split()
FACE='skull forehead brow eye eyelid_top eyelid_bot eyelid_corner_inner eyelid_corner_outer cheek nose lip_top lip_bot lip_corner jaw chin teeth_top teeth_bot tongue ear'.split()
SKIN='forehead brow eyelid_top eyelid_bot eyelid_corner_inner eyelid_corner_outer cheek nose lip_top lip_bot lip_corner jaw chin'.split()
class UnsupportedRigMapOption(NotImplementedError):pass

def assemble(names,parents,deform,segments,mesh_names,scope,head,surface):
 from . import _bvt_solver as rt
 result=dict(elements=[],geometry=[],parts=[],kinds=dict(all=list(ALL),face=list(FACE),skin=list(SKIN),rigid=['eye','teeth_top','teeth_bot']),family='',label='',reason='',verified=False,face_family='')

 if not names:return result
 p=np.array(parents,'i8');d=np.frombuffer(deform,'u1').copy().view(bool);s=np.array(segments,'f4').reshape((-1,2,3))
 spine=rt.detect_spine(names,p,d,s)
 result.update((k,spine[k]) for k in ('family','label','reason','verified'))
 mode=2 if head is not None else {'verified':0,'detected':1,'off':3}[scope]
 h=head if mode==2 else spine['head'] if mode==1 or spine['verified'] else -1
 from .rig_map_native import dictionary, corners, mesh_parts
 result['parts']=mesh_parts(mesh_names)
 native=dictionary(names,p,d,s,h) if mode!=3 and spine['family']=='arp' else {}
 # The original accepts only the ARP family from its legacy face detector.
 roles,family=rt.detect_face_public(names,p,d,int(h))
 if mode==3 or family!='arp':roles=[];family=''
 result['face_family']=family
 groups=defaultdict(list)
 for i,role,side in roles:groups[(role,side)].append(i)
 groups.update(native)
 partitions=corners(groups,s)
 elements=result['elements']
 def add(kind,side,ids,status='legacy',reason='detect_face',role=None,index=0):
  ident=f'{kind}.{side}'+(f'.{index}' if kind=='finger' else '')
  elements.append((ident,kind,role or kind,side,index,list(ids),status,reason))
 for kind in ('pelvis','spine','neck'):
  ids=([spine['pelvis']] if spine['pelvis']>=0 else []) if kind=='pelvis' else spine[kind]
  if not spine['verified']:add(kind,'x',[],'absent','the spine is not verified: '+spine['reason'])
  elif ids:add(kind,'x',ids,'legacy','detect_spine')
  else:add(kind,'x',[],'absent','no bone identified')
 inferred=False
 jaws=[i for i,role,_ in roles if role=='jaw']
 if h<0 and mode==1 and jaws:
  # Native ancestor mask skips face roles; unclassified head ancestors remain eligible.
  by_index={i:role for i,role,_ in roles}
  current=jaws[0];seen=set()
  while current>=0:
   if current in seen:raise UnsupportedRigMapOption('cyclic jaw ancestry has not been qualified')
   seen.add(current)
   if current>=len(p):raise rt.PanicException(f'index out of bounds: the len is {len(p)} but the index is {current}')
   current=int(p[current])
   if current<0:break
   if current>=len(names):raise rt.PanicException(f'index out of bounds: the len is {len(names)} but the index is {current}')
   if d[current] and by_index.get(current) not in set(FACE):h=current;inferred=True;break
 if 0<=h<len(names):
  reason='the first deforming bone above the jaw' if inferred else 'the head the caller gave' if mode==2 else 'detect_spine'
  add('head','x',[h],'legacy',reason)
 else:add('head','x',[],'absent','no head bone'+(': '+spine['reason'] if spine['reason'] else ''))
 for (role,side),ids in groups.items():
  status,reason=('dictionary','arp') if (role,side) in native else ('legacy','detect_face')
  if role=='eyelid_corner':
   inner,outer,geometric=partitions.get(side,(ids[:1],ids[1:],False))
   for kind,part in [('eyelid_corner_inner',inner),('eyelid_corner_outer',outer)]:
    add(kind,side,part,status if part else 'absent',(reason if geometric else reason+', no other eye to measure against: the order the pass gave') if part else 'the lid has no corner bone on this half of the eye',role='eyelid_corner')
  else:add(role,side,ids,status,reason)
 for side in ('l','r','x'):
  eyes=groups.get(('eye',side),[])
  if eyes:
   main=min(eyes,key=lambda i:len(names[i]))
   if ('eye_main',side) not in native:add('eye_main',side,[main],reason="the shortest name among the side's eye bones")
   reflected=[i for i in eyes if 'ref' in names[i].lower()]
   if ('eye_card',side) not in native:add('eye_card',side,[min(reflected,key=lambda i:len(names[i]))] if reflected else [main],reason="the reflection bone's name word" if reflected else 'no reflection bone: the eye itself')
  lids=[i for (role,sd),ids in groups.items() if sd==side and role.startswith('eyelid') for i in ids]
  if lids and ('eyelid_tweak',side) not in native:
   tweaks=[i for i in lids if 'twk' in names[i].lower()]
   add('eyelid_tweak',side,tweaks,'legacy' if tweaks else 'absent',"the tweak-handle word in a lid bone's name" if tweaks else "no tweak handle among the side's lid bones")
 chains,finger_family=rt.detect_fingers_by_name(names,p,d)
 for index,chain in enumerate(chains):add('finger','?',chain,'legacy','detect_fingers_by_name: '+finger_family,index=index)
 if not chains:add('finger','?',[],'absent','no finger vocabulary in the bone names')
 if surface is not None and mode!=3 and groups:
  from .rig_map_surface import surface_records
  result['geometry']=surface_records(names,p,d,s,h,groups,surface)
  result['geometry'].sort(key=lambda e:(ALL.index(e[1]),{'l':0,'r':1,'x':2,'?':3}[e[3]],e[4]))
 elements.sort(key=lambda e:(ALL.index(e[1]),{'l':0,'r':1,'x':2,'?':3}[e[3]],e[4]))
 return result
