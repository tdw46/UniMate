"""Rest-clearance adaptation for long-dress follow; no mesh or solver edits.

A closed ring of generated garment guides approximates the rest silhouette.
Compare its inward clearance from both leg axes with untrimmed body-fit radii,
never the inflated directional guard radii. Missing skin uses the existing
anatomical radius estimate. Measurements are cached, not recomputed on sliders.
"""
import json
import math
from statistics import median

REVISION = 1


def clearance_tightness(clearance, leg_length):
    if not math.isfinite(clearance) or not math.isfinite(leg_length) or leg_length <= 0:
        raise ValueError('Invalid dress clearance measurement')
    # Dimensionless ease: close tubes get full adaptation; generous skirts none.
    t = min(1., max(0., (clearance / leg_length - .04) / .18))
    return 1. - t*t*(3.-2.*t)


def adapted_follow(base, tightness, strength=1.):
    """Boost follow smoothly while preserving exact user zero/one endpoints."""
    base, tightness, strength = (min(1., max(0., float(v))) for v in (base, tightness, strength))
    return base / (1. - .85 * tightness * strength * (1. - base))


def ring_clearance(ring, center, radius):
    """Minimum distance to the closed garment cross-section, less body radius."""
    if len(ring) < 6 or radius <= 0:
        raise ValueError('Incomplete dress ring or invalid leg radius')
    def distance(a, b):
        d = (b[0]-a[0], b[1]-a[1])
        length = d[0]*d[0]+d[1]*d[1]
        t = min(1., max(0., ((center[0]-a[0])*d[0]+(center[1]-a[1])*d[1])/length)) if length else 0.
        return math.hypot(a[0]+t*d[0]-center[0], a[1]+t*d[1]-center[1])
    edges=list(zip(ring,ring[1:]+ring[:1]))
    inside=False
    for a,b in edges:
        if (a[1]>center[1]) != (b[1]>center[1]):
            if center[0] < a[0]+(center[1]-a[1])*(b[0]-a[0])/(b[1]-a[1]):
                inside=not inside
    nearest=min(distance(a,b) for a,b in edges)
    return (nearest if inside else -nearest)-radius


def _point_at_height(points, z):
    for a,b in zip(points,points[1:]):
        if a[2] >= z >= b[2] and a[2]-b[2] > 1e-9:
            t=(a[2]-z)/(a[2]-b[2])
            return tuple(a[i]+t*(b[i]-a[i]) for i in range(3))
    return None


def measure_dress_fit(rig, meshes, body_details=None):
    from avatar_dress import full_chain_names
    hum=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    hips=rig.data.bones[hum.hips.node.bone_name].head_local
    families={}
    for spring in rig.data.vrm_addon_extension.spring_bone1.springs:
        if not spring.vrm_name.startswith('Secondary_Skirt_') or not spring.joints:
            continue
        root=rig.data.bones[spring.joints[0].node.bone_name]
        if not root.get('hallway_dress_full_chain') or root.get('hallway_dress_lower'):
            continue  # Short and mid-calf skirts keep their authored follows.
        points=[tuple(rig.data.bones[n].head_local) for n in full_chain_names(rig,spring)]
        families.setdefault(spring.vrm_name.rsplit('_',1)[0],[]).append((spring.vrm_name,points))
    if not families:
        return dict(revision=REVISION,method='rest garment ring clearance / leg length',families=[])
    if body_details is None:
        from avatar_colliders import plan_colliders
        body_details=plan_colliders(rig,meshes)['collider_details']
    radii={row['bone']:row['body_radius'] for row in body_details if row['role']=='skirt'}
    output=[]
    for family,chains in families.items():
        if len(chains)<6:
            continue
        chains.sort(key=lambda row:math.atan2(row[1][0][1]-hips.y,row[1][0][0]-hips.x))
        sections={}
        for section,role in (('upper','upper_leg'),('lower','lower_leg')):
            samples=[]
            for side in ('left','right'):
                bone=rig.data.bones[getattr(hum,side+'_'+role).node.bone_name]
                radius=radii.get(bone.name,bone.length*.12)
                for fraction in (.25,.5,.75):
                    center=bone.head_local.lerp(bone.tail_local,fraction)
                    ring=[_point_at_height(points,center.z) for _,points in chains]
                    if any(p is None for p in ring):continue
                    clearance=ring_clearance(ring,center,radius)
                    samples.append(dict(bone=bone.name,fraction=fraction,clearance=clearance,
                        ease=clearance/bone.length,body_radius=radius,
                        tightness=clearance_tightness(clearance,bone.length)))
            # Median over both sides and several heights rejects one pinched band.
            sections[section]=dict(tightness=median(s['tightness'] for s in samples) if len(samples)>=4 else 0.,samples=samples)
        output.append(dict(family=family,chains=[name for name,_ in chains],sections=sections))
    return dict(revision=REVISION,method='rest garment ring clearance / leg length',families=output)


def configure_dress_fit(rig, meshes, body_details=None):
    report=measure_dress_fit(rig,meshes,body_details)
    # Clear only our cached measures so regeneration/reclassification is idempotent.
    for bone in rig.data.bones:
        if 'hallway_dress_tightness' in bone:del bone['hallway_dress_tightness']
    for family in report['families']:
        for chain in family['chains']:
            helper=chain.replace('Secondary_Skirt_','Secondary_SkirtFollow_',1)
            for name,section in ((helper,'upper'),(helper+'_Lower','lower')):
                bone=rig.data.bones.get(name)
                if bone is not None:bone['hallway_dress_tightness']=family['sections'][section]['tightness']
    if report['families']:
        rig['hallway_dress_fit']=json.dumps(report)
    elif 'hallway_dress_fit' in rig:
        del rig['hallway_dress_fit']
    from properties_hallway_rig import apply_follow
    for group in rig.hallway_rig.follow_groups:apply_follow(rig,group)
    return report
