"""Rest-space smoothing and short/long cloth ownership regressions in Blender."""
import bpy,json,os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights,assign
from avatar_shoulder_binding import smooth_repaired_shoulders
from avatar_pelvis_cloth import bind_short_pelvis_cloth,short_support

bpy.ops.wm.read_factory_settings(use_empty=True)
rig=bpy.data.objects.new('Rig',bpy.data.armatures.new('Rig'));bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active=rig;rig.select_set(True);bpy.ops.object.mode_set(mode='EDIT')
for name,a,tail in [('Hips',(0,0,.8),(0,0,.9)),('Chest',(0,0,.95),(0,0,1.1)),
                 ('UpperArm.L',(.2,0,.8),(.45,0,.75)),('UpperArm.R',(-.2,0,.8),(-.45,0,.75)),
                 ('Thigh.L',(.08,0,.8),(.08,0,.4)),('Thigh.R',(-.08,0,.8),(-.08,0,.4)),
                 ('Shin.L',(.08,0,.4),(.08,0,.05)),('Shin.R',(-.08,0,.4),(-.08,0,.05)),
                 ('Index1.L',(.7,0,.7),(.75,0,.7)),('Secondary_Panel_01',(0,-.14,.8),(0,-.14,.6))]:
 b=rig.data.edit_bones.new(name);b.head=a;b.tail=tail
bpy.ops.object.mode_set(mode='OBJECT')

def mesh(name,points,faces):
 data=bpy.data.meshes.new(name);data.from_pydata(points,[],faces);data.update()
 obj=bpy.data.objects.new(name,data);bpy.context.scene.collection.objects.link(obj);return obj

def coords(obj):return np.array([v.co[:] for v in obj.data.vertices])

# Alternating sharp shoulder rows and a coincident seam must average together.
n=25;points=[(.10+x*.016,0,.60+z*.016) for z in range(n) for x in range(n)]
faces=[(z*n+x,z*n+x+1,(z+1)*n+x+1,(z+1)*n+x) for z in range(n-1) for x in range(n-1)]
points.extend([points[12*n+12],(.30,-.005,.8)])
faces.append((len(points)-1,12*n+12,12*n+13))
obj=mesh('Repair',points,faces);obj.data.attributes.new(name='hallway_arm_bank',type='INT',domain='POINT')
for i in range(len(points)):assign(obj,[i],{'Chest':1.} if i%2 else {'UpperArm.L':1.})
assign(obj,[len(points)-1],{'Index1.L':1.});original=coords(obj);finger=weights(obj,len(points)-1)
unrepaired=mesh('Unrepaired',points,faces);assign(unrepaired,list(range(len(points))),{'Chest':1.})
report=smooth_repaired_shoulders(rig,[obj,unrepaired]);assert len(report)==1
assert report[0]['edge_energy_after']<report[0]['edge_energy_before']*.5,report
assert weights(obj,len(points)-1)==finger, 'Shoulder smoothing changed a nearby finger; exclude hand/digit-owned vertices from its domain'
assert weights(obj,12*n+12)==weights(obj,len(points)-2),'Coincident seam diverged'
assert all(weights(unrepaired,i)=={'Chest':1.} for i in range(len(points)))
np.testing.assert_array_equal(original,coords(obj));assert report[0]['max_influences']<=4

# Coincident cut surfaces below the socket must never become virtual welds.
cut_points=points[:n*n]+points[:n*n]
cut_faces=faces[:-1]+[tuple(i+n*n for i in face) for face in faces[:-1]]
cut=mesh('Separate cut banks',cut_points,cut_faces)
bank=cut.data.attributes.new(name='hallway_arm_bank',type='INT',domain='POINT')
strength=cut.data.attributes.new(name='hallway_arm_bank_strength',type='FLOAT',domain='POINT')
join=cut.data.attributes.new(name='hallway_arm_join',type='FLOAT',domain='POINT')
for i in range(len(cut_points)):
 arm=i>=n*n;bank.data[i].value=int(arm);strength.data[i].value=1.;join.data[i].value=0.
 assign(cut,[i],{'UpperArm.L' if arm else 'Chest':1.})
smooth_repaired_shoulders(rig,[cut])
for i in range(len(cut_points)):
 assert weights(cut,i)=={'UpperArm.L' if i>=n*n else 'Chest':1.}, 'Shoulder smoothing re-welded the separated underarm banks'

# An explicitly shared socket node must permit both sides, even on an
# unusually long arm whose ownership-falloff and socket bands overlap.
for i in range(len(cut_points)):join.data[i].value=1.
smooth_repaired_shoulders(rig,[cut])
for i in (12*n+12,12*n+12+n*n):
 assert abs(weights(cut,i).get('Chest',0)-.5)<1e-6
 assert abs(weights(cut,i).get('UpperArm.L',0)-.5)<1e-6

# Partial socket attachment must fade, not collapse to a binary bank mask.
# Positive joint sharing is valid here; the fully separated case above stays pure.
for i in range(len(cut_points)):
 arm=i>=n*n;join.data[i].value=.5
 assign(cut,[i],{'UpperArm.L':.8 if arm else .2,'Chest':.2 if arm else .8})
cut_before=coords(cut);smooth_repaired_shoulders(rig,[cut])
for i in (12*n+12,12*n+12+n*n):
 foreign='Chest' if i>=n*n else 'UpperArm.L'
 assert .05<weights(cut,i).get(foreign,0)<=.250001, 'Partial shoulder join became a hard hinge; preserve its bounded body/arm blend'
np.testing.assert_array_equal(cut_before,coords(cut))

# One connected garment: short front hem, long back coat. No labels/materials.
cols=64;rows=25
angles=np.arange(cols)*2*np.pi/cols
points=[(.15*np.cos(a),.15*np.sin(a),.85+(hem-.85)*r/(rows-1))
        for r in range(rows) for a in angles for hem in [.60 if np.sin(a)<-.1 else .10]]
faces=[(r*cols+c,r*cols+(c+1)%cols,(r+1)*cols+(c+1)%cols,(r+1)*cols+c)
       for r in range(rows-1) for c in range(cols)]
cloth=mesh('Anonymous',points,faces);assign(cloth,list(range(len(points))),{'Secondary_Panel_01':1.})
bodypoints=[(.12*np.cos(a),.12*np.sin(a),.85-.75*r/(rows-1)) for r in range(rows) for a in angles]
body=mesh('Underlying',bodypoints,faces)
for i,p in enumerate(bodypoints):
 q=float(np.clip((.8-p[2])/.16,0,1));assign(body,[i],{'Hips':1-q,'Thigh.'+('L' if p[0]>=0 else 'R'):q})
original=coords(cloth);bodyrows=[weights(body,i) for i in range(len(bodypoints))]
plan=dict(height=1.2,points={cloth.name:original},regions=[dict(obj=cloth,ids=list(range(len(points))),role='garment')])
support,_=short_support(cloth.data,original,set(range(len(points))),.8,.4,1.2)
front=(rows-1)*cols+48;back=(rows-1)*cols+16
assert support[front]>.999 and support.get(back,0)==0,(support.get(front),support.get(back))
result=bind_short_pelvis_cloth(rig,[cloth,body],plan)
assert result['vertices']>0 and result['body_samples']>0,result
assert weights(cloth,front).get('Secondary_Panel_01',0)<=.121,weights(cloth,front)
assert sum(w for n,w in weights(cloth,front).items() if not n.startswith('Secondary_'))>=.879
assert weights(cloth,back)=={'Secondary_Panel_01':1.},'Long coat changed'
assert all(len(weights(cloth,i))<=4 and abs(sum(weights(cloth,i).values())-1)<1e-6 for i in range(len(points)))
assert bodyrows==[weights(body,i) for i in range(len(bodypoints))]
np.testing.assert_array_equal(original,coords(cloth))
# The same garment made uniformly long must be a no-op.
long=mesh('Long',[(p[0],p[1],bodypoints[i][2]) for i,p in enumerate(points)],faces)
assert not short_support(long.data,coords(long),set(range(len(points))),.8,.4,1.2)[0]
result=dict(passed=True,shoulder=report,cloth=result,fingers_unchanged=True,geometry_unchanged=True,long_hem_unchanged=True)
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(result));print('SHOULDER_PELVIS_PASSED',result)
