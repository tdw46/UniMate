"""Rest-frame transport retains ankle motion without a posed knee-plane flip."""
import json,math,os,sys
from pathlib import Path
from mathutils import Matrix,Quaternion,Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_animation_frames import transport_rest_frame
rest=Quaternion((0,0,1),.7).to_matrix().to_4x4()
frame=Quaternion((1,0,0),.4)
last=None;maximum=0
for i in range(241):
    authored=Quaternion((1,0,0),math.radians(i*.5-60))
    pose=authored.to_matrix().to_4x4()@rest
    out=transport_rest_frame(frame,rest,pose)
    expected=authored@frame
    assert abs(out.dot(expected))>1-1e-6
    if last is not None:maximum=max(maximum,2*math.acos(min(1,abs(out.dot(last)))))
    last=out
assert maximum<math.radians(.51)
assert transport_rest_frame(None,rest,rest) is None
assert abs(transport_rest_frame(frame,rest,rest).dot(frame))>1-1e-6
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,maximum_step_degrees=math.degrees(maximum),preserves_authored_rotation=True)))
