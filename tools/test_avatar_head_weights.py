"""Exercise jaw ownership independently of scale, density and UV splits."""
import json,math,os,sys
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_head_weights import cranial_ownership, head_ownership, projecting_chin_ownership
report=[]
for segments,scale,offset in [(16,.25,(0,0,0)),(32,1.,(2,-3,4)),(64,3.,(-2,5,-1))]:
    shaft=np.array([(math.cos(t)*.4,math.sin(t)*.5,z) for z in np.linspace(0,1,81) for t in np.linspace(0,math.tau,segments,endpoint=False)])
    # Jaw hangs below the atlas and projects forward of the neck; skull above
    # the atlas must be entirely Head even at the back, close to the neck axis.
    probes=np.array([(0,-1.,.65),(0,.45,1.05),(0,0,1.5),(0,-.5,.5),(1.,0,.2)])
    cloud=np.concatenate((shaft,probes))*scale+offset
    base=np.array((0,0,0))*scale+offset;atlas=np.array((0,.1,1))*scale+offset
    values,audit=head_ownership(cloud,base,atlas)
    assert audit['applied']
    assert np.all(values[-5:-2]==1),values[-5:]
    assert np.all(values[-2:]==0),values[-5:]
    repeated=np.repeat(cloud,3,axis=0)
    again,_=head_ownership(repeated,base,atlas)
    assert np.allclose(values,np.array(again)[::3],atol=1e-12)
    # No step through the upper shaft; smooth transition ends at rigid head.
    heights=np.linspace(.55,1,1001)
    samples=np.column_stack((np.zeros(len(heights)),np.full(len(heights),.5),heights))*scale+offset
    transition,_=head_ownership(np.concatenate((cloud,samples)),base,atlas)
    tail=transition[-len(samples):]
    assert np.all(np.diff(tail)>=-1e-10) and np.max(np.diff(tail))<.005
    report.append({'segments':segments,'scale':scale,'jaw_and_skull_rigid':True,'lower_neck_and_shoulders_unchanged':True,'uv_duplication_invariant':True,'continuous_shaft_transition':True})
values,audit=head_ownership(np.empty((0,3)),(0,0,0),(0,0,1));assert not audit['applied']

# The ears sit beside the atlas, past the forward projection of the face.
# The coat collar extends above the atlas too, farther from the head axis.
Point = namedtuple('Point', 'x y z')
probes = np.array([
    (0., -.8, -.29), (.69, -.025, .14), (-.69, -.025, .14),
    (0., .58, .80), (0., 0., 1.11), (.76, .28, .14),
    (.22, .75, .44), (0., .25, -.35),
])
for scale, offset in [(1., (0.,0.,0.)), (3., (2.,-4.,1.))]:
    head = SimpleNamespace(head_local=Point(*offset), tail_local=Point(*(np.array(offset)+(0.,0.,scale))), length=scale)
    neck = SimpleNamespace(head_local=Point(*(np.array(offset)-(0.,0.,.6*scale))))
    points = probes*scale+offset

    def front_depth(x,z):
        x,z=(x-offset[0])/scale,(z-offset[2])/scale
        return offset[1] + scale*(-.81 if z >= -.30+.45*abs(x) else -.39)

    def radial_depth(z,angle):
        return np.where((z-offset[2])/scale>1.08, np.nan, .60*scale)

    chin = projecting_chin_ownership(points,head,neck,front_depth)
    cranial = cranial_ownership(points,head,radial_depth)
    fused = np.maximum(chin,cranial)
    assert chin[0]==1 and np.all(chin[1:3]<1), chin
    assert np.allclose(fused[:5],1), fused
    assert np.allclose(fused[5:],0), fused
    assert not projecting_chin_ownership(points,head,neck,lambda x,z: -.4*scale).any()
    report.append({'scale':scale,'fused_chin_and_ears_rigid':True,'outer_collar_unchanged':True})
out=Path(__file__).resolve().parents[1]/'outputs/stitched_autorig/head_generalization.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2));print('HEAD_OWNERSHIP_TESTS_PASSED',json.dumps(report))
if result_path := os.environ.get('BEYOND_UI_TEST_RESULT'):
    Path(result_path).write_text(json.dumps({'passed':True,'cases':report}))
