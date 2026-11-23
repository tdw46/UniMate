"""Remove arm spill on predominantly pelvis-attached surface patches."""
from avatar_apparel_weights import weights,assign


def confine_pelvis_accessories(rig, meshes):
    pelvis={n for n in rig.data.bones.keys() if n in {'Hips','Spine'} or n.startswith('Thigh.')}
    arm={n for n in rig.data.bones.keys() if n.startswith(('UpperArm.','Forearm.','Hand.','Thumb','Index','Middle','Ring','Little'))}
    elbows=[b.head_local.z for b in rig.data.bones if b.name.startswith('Forearm.')]
    hips=rig.data.bones.get('Hips')
    if not elbows or hips is None:return dict(vertices=0)
    ceiling=min(min(elbows),hips.head_local.z+hips.length*.5)
    changed=0
    for obj in meshes:
        transform=rig.matrix_world.inverted_safe()@obj.matrix_world
        for v in obj.data.vertices:
            if (transform@v.co).z>=ceiling:continue
            row=weights(obj,v.index)
            # A hand/sleeve belongs to the articulated arm, not predominantly
            # to hips/thighs. Only mixed patches with a positive pelvic owner
            # are admitted; pure arm/finger and secondary cloth rows stay intact.
            body=sum(w for n,w in row.items() if n in pelvis)
            spill=sum(w for n,w in row.items() if n in arm)
            if body<.5 or spill<=1e-7 or any(n.startswith('Secondary_') and w>1e-5 for n,w in row.items()):continue
            kept={n:w for n,w in row.items() if n not in arm};total=sum(kept.values())
            assign(obj,[v.index],{n:w/total for n,w in kept.items()});changed+=1
    return dict(vertices=changed,geometry_changed=False)
