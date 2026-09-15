"""Neck-based atlas correction is insensitive to jaw depth, scale and UV splits."""
import json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_atlas_landmark import center_atlas
report=[]
for segments,scale,offset,jaw in [(16,.5,(0,0,0),.5),(32,1.,(3,-2,4),1.),(64,2.3,(-1,4,2),2.)]:
    points=[]
    for z in np.linspace(0,1,81):
        for j in range(segments):
            t=math.tau*j/segments;y=.2+.4*math.sin(t)
            if z>.6 and y<.2:y-=jaw
            points.append((.3*math.cos(t),y,z))
    points=np.array(points)*scale+offset
    before=np.array((0,-jaw,1))*scale+offset;base=np.array((0,.2,0))*scale+offset
    result,audit=center_atlas(points,base,before)
    assert audit['applied'] and audit['neck_center']<result.y<audit['neck_back']
    assert abs((result.y-offset[1])/scale-.28)<.025
    assert result.x==before[0] and abs(result.z-before[2])<1e-6
    repeated=np.repeat(points,3,axis=0);again,_=center_atlas(repeated,base,before)
    assert np.linalg.norm(np.array(again)-np.array(result))<1e-6
    report.append({'segments':segments,'scale':scale,'jaw_projection':jaw,'jaw_overlap_from':.6,'posterior_fraction':.6,'passed':True})
_,audit=center_atlas(np.empty((0,3)),(0,0,0),(0,0,1));assert not audit['applied']
out=Path(__file__).resolve().parents[1]/'outputs/stitched_autorig/atlas_generalization.json';out.write_text(json.dumps(report,indent=2));print('ATLAS_TESTS_PASSED',json.dumps(report),flush=True)
