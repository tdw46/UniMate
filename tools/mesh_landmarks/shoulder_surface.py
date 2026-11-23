"""Recover a buried shoulder pivot from the outer cap of a confirmed fused arm.

Loose coats can pull the predictor's shoulder down into their welded underarm.
Use only mesh geometry and the predicted upper-arm scale, never a source rig.
Unfused or raised arms retain the predictor result. Refinement runs before cuts;
synthetic closure surfaces must not become new anatomical observations.
"""
import numpy as np

REVISION = 1


def fit_fused_shoulders(vertices, faces, prediction):
    previous=prediction.get('shoulder_socket_refinement',{})
    if previous.get('revision')==REVISION:return previous
    from .arm_separation import plan
    v=np.asarray(vertices,dtype=float);joints=prediction['joints'];adjusted=[]
    cuts=plan(v,faces,prediction)
    for cut in cuts:
        side=cut['side'];sign=1 if side=='Left' else -1
        head=np.asarray(joints[side+'Arm']['head'],dtype=float)
        elbow=np.asarray(joints[side+'ForeArm']['head'],dtype=float)
        length=float(np.linalg.norm(elbow-head))
        if length<=1e-8 or 'Neck' not in joints or side+'Shoulder' not in joints:continue
        # Measure outboard of the pivot, excluding the collar/neck. Keep the
        # observation local in depth and below the neck's immediate vicinity.
        lateral=(v[:,0]-head[0])*sign
        candidates=v[(lateral>=0)&(lateral<length*.30)&
                     (np.abs(v[:,1]-head[1])<length*.50)&
                     (v[:,2]>head[2])&(v[:,2]<joints['Neck']['head'][2]+length*.08)]
        if len(candidates)<32:continue
        cap=float(np.quantile(candidates[:,2],.98))
        # Leave a humeral-head depth below the sleeve cap. A small dead band
        # preserves already plausible pivots; the bound rejects extreme fits.
        lift=cap-length*.22-head[2]
        if lift<length*.06 or lift>length*.40:continue
        after=head.copy();after[2]+=lift
        joints[side+'Arm']['head']=after.tolist()
        joints[side+'Shoulder']['tail']=after.tolist()
        adjusted.append(dict(side=side,before=head.tolist(),after=after.tolist(),
                             cap_height=cap,upper_arm_length=length,lift=float(lift)))
    return dict(revision=REVISION,adjusted=adjusted,source='uncut mesh cap and confirmed arm fusion')
