"""Center the head/neck pivot from neck-shaft sections, biased posteriorly.

Input coordinates are rig-local: +Y posterior, Z up. This corrects the shared
Neck tail / Head head, not the skull tip or the joint's vertical position.
"""
import numpy as np
from mathutils import Vector


def center_atlas(points,neck_base,atlas,posterior_fraction=.60):
    if not .5<=posterior_fraction<=.7:raise ValueError('Posterior fraction must remain near the neck center')
    points=np.asarray(points,dtype=float)
    base=np.asarray(neck_base,dtype=float);old=np.asarray(atlas,dtype=float)
    height=old[2]-base[2]
    audit={'applied':False,'before':old.tolist(),'posterior_fraction':posterior_fraction}
    if height<=1e-8 or not len(points):return Vector(old),audit
    points=np.unique(points[(points[:,2]>=base[2]) & (points[:,2]<=old[2])],axis=0)
    sections=[]
    # Sample the shaft below the jaw. A slice at the head pivot includes the
    # chin and can pull its depth estimate far forward.
    for fraction in np.linspace(.35,.75,9):
        z=base[2]+height*fraction
        p=points[abs(points[:,2]-z)<height*.045]
        if len(p)<8:continue
        # Restrict to the central neck strip, rejecting lateral shoulders/hair.
        widths=abs(p[:,0]-old[0]);radius=float(np.quantile(widths,.6))
        p=p[widths<=max(radius*.5,height*.03)]
        if len(p)<4:continue
        front,back=np.quantile(p[:,1],[.05,.95])
        if back-front<=height*.03:continue
        sections.append((z,float(front),float(back)))
    if len(sections)<4:
        audit['reason']='Insufficient neck shaft sections';return Vector(old),audit
    front=float(np.median([s[1] for s in sections]));back=float(np.median([s[2] for s in sections]))
    target=old.copy();target[1]=front+(back-front)*posterior_fraction
    audit.update(applied=True,after=target.tolist(),neck_front=front,neck_back=back,
                 neck_center=(front+back)*.5,posterior_shift=float(target[1]-old[1]),sections=sections)
    return Vector(target),audit
