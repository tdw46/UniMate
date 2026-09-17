"""Check both supported VRM thumb schemas and fail on incomplete digit metadata."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_fingers import source_finger_aliases,DIGITS
from avatar_source_import import material_role
import bpy
for version in (0,1):
 roles={}
 for side in ('left','right'):
  for digit in DIGITS:
   parts=('Metacarpal','Proximal','Distal') if digit=='Thumb' and version==1 else ('Proximal','Intermediate','Distal')
   for part in parts:roles[side+digit+part]=len(roles)
 mapped=source_finger_aliases(roles)
 assert len(mapped)==30 and set(mapped.values())=={f'{d}{i}.{s}' for d in DIGITS for i in (1,2,3) for s in ('L','R')}
 del roles['leftIndexDistal']
 try:source_finger_aliases(roles)
 except ValueError:pass
 else:raise AssertionError('Incomplete finger metadata accepted')
for suffix in ('','.001','.019'):
 m=bpy.data.materials.new('N00_Body_00_SKIN'+suffix);assert material_role(m)=='body'
print('FINGER_SCHEMA_AND_MATERIAL_TESTS_PASSED')
