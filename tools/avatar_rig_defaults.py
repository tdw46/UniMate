"""Shared generation and panel-reset defaults; no Blender registration needed."""
import json
import math

FOLLOW = {
    'Skirt': .55,
    'Skirt Knee': .60,
    'Skirt Knee Side': .55,
    'Skirt Knee Back': .50,
    'Skirt Support': .75,
}
# Scale hair gravity with its restoring force to retain the downward rest bias.
# Hanging accessories retain gravity; contact margins are fitted independently.
SPRINGS = {
    'Head Accessories': dict(drag=.5, stiffness=1., non_root_stiffness=1., gravity=0.),
    'Accessory': dict(drag=.4, stiffness=1.2, non_root_stiffness=1., gravity=.035),
    'Hanging Straps': dict(drag=.15, stiffness=.12, non_root_stiffness=1., gravity=.035),
    'Skirt': dict(drag=.4, stiffness=1.6, non_root_stiffness=1., gravity=.025),
    'Hair': dict(drag=.3, stiffness=.3, non_root_stiffness=1., gravity=.0105),
}
SKIRT_THICKNESS = 1.
DRESS_FIT_STRENGTH = 1.
PHYSICS_ENABLED = True
COLLIDERS_VISIBLE = False
BONES_VISIBLE = True


def profile_defaults(rig):
    """Choose from measured full-length families; unclassified rigs stay legacy.

    The panel is rig-wide, so mixed tight/loose outfits conservatively retain
    legacy defaults instead of imposing the narrow-dress preset on loose layers.
    """
    try:
        families = json.loads(rig.get('hallway_dress_fit', '{}')).get('families', [])
        tight = bool(families) and all(
            family['sections']['lower']['tightness'] >= .5 for family in families)
    except (ValueError, TypeError, KeyError, AttributeError):
        tight = False
    follow = dict(FOLLOW)
    springs = {name: dict(values) for name, values in SPRINGS.items()}
    if tight:
        follow['Skirt'] = .89
        springs['Skirt']['drag'] = .55
        springs['Skirt']['stiffness'] = 4.
    return dict(name='tight_long_dress' if tight else 'legacy', follow=follow, springs=springs)


def spring_matches(name, prefix):
    """Split existing generated accessories without renaming saved bones."""
    if prefix.endswith('Strap_'):
        return name.startswith(prefix[:-6]+'Accessory_') and '_Strand' in name
    if prefix.endswith('Accessory_'):
        return name.startswith(prefix) and '_Strand' not in name
    return name.startswith(prefix) or (prefix.endswith('Skirt_') and name.startswith(prefix[:-6]+'HipSkirt_'))


def hair_length_defaults(relative_length):
    """Scale-independent short-to-long strand tuning, evaluated at generation.

    Short strands use the authored short-hair VRM restoring/damping values.
    Long strands retain the reviewed soft profile. Keep the requested downward
    bias proportional to stiffness, independently of the source gravity value.
    Smoothstep avoids a discontinuity when a strand crosses an anchor.
    """
    if not math.isfinite(relative_length) or relative_length < 0:
        raise ValueError('Hair length must be finite and nonnegative')
    t = min(1., max(0., (relative_length - .075) / (.23 - .075)))
    t = t * t * (3. - 2. * t)
    stiffness = 1. + (.3 - 1.) * t
    return dict(stiffness=stiffness, drag=.5 + (.3 - .5) * t,
                gravity=.035 * stiffness)


def short_head_lock(rig, bones):
    head = rig.data.bones.get('Head')
    if head is None or len(bones) < 2 or any(b is None for b in bones):
        return False
    length = sum((b.head_local-a.head_local).length for a,b in zip(bones,bones[1:]))
    return (head.length > 1e-6 and length <= head.length*1.55 and
            min(b.head_local.z for b in bones) >= head.head_local.z-head.length*.42)


def scalp_support_count(rig, bones):
    """Keep the back of a short cap glued through the occiput using ordinary FK."""
    if not short_head_lock(rig, bones):
        return 0
    head = rig.data.bones['Head']
    feet = [rig.data.bones.get('Foot.'+s) for s in ('L','R')]
    forward = sum(((b.tail_local-b.head_local) for b in feet if b), head.head_local*0.)
    forward.z = 0.
    if forward.length < 1e-6:
        return 0
    forward.normalize()
    radial = bones[0].head_local - head.head_local
    lateral = abs(radial.x*forward.y - radial.y*forward.x)
    ahead = radial.dot(forward)
    # Side locks can begin just forward of the ear. Keep their scalp section
    # anchored to temple height while leaving the central fringe free.
    temple = lateral > .19*head.length and lateral > .65*ahead
    if ahead > .18*head.length and not temple:
        return 0
    target = head.head_local.z + .40*(head.tail_local.z-head.head_local.z)
    if bones[0].head_local.z <= target or bones[-1].head_local.z > target:
        return 0
    return min(range(len(bones)-1), key=lambda i: abs(bones[i].head_local.z-target))


def configure_hair_spring(spring, rig, height):
    """Install generated-only ratios so panel edits preserve strand differences.

    Height and rest heads must share rig-local units. This is deliberately an
    explicit generation/update operation, never an implicit saved-rig migration.
    Neither source-rig data, names nor materials participate in tuning.
    """
    if not math.isfinite(height) or height <= 0:
        raise ValueError('Character height must be finite and positive')
    bones = [rig.data.bones.get(j.node.bone_name) for j in spring.joints]
    if len(bones) < 2 or any(b is None for b in bones):
        return None
    count = scalp_support_count(rig, bones)
    for _ in range(count):
        spring.joints.remove(0)
    bones = bones[count:]
    relative_length = sum((b.head_local-a.head_local).length
                          for a, b in zip(bones, bones[1:])) / height
    values = hair_length_defaults(relative_length)
    base = SPRINGS['Hair']
    spring['hallway_hair_profile_version'] = 1
    spring['hallway_hair_relative_length'] = relative_length
    for i, joint in enumerate(spring.joints):
        joint['hallway_stiffness_ratio'] = values['stiffness'] / base['stiffness'] * max(0., 1.-.1*i)
        joint['hallway_hair_drag_ratio'] = values['drag'] / base['drag']
        joint['hallway_hair_gravity_ratio'] = values['gravity'] / base['gravity']
        joint.stiffness = base['stiffness'] * joint['hallway_stiffness_ratio']
        joint.drag_force = values['drag']
        joint.gravity_power = values['gravity']
    return dict(relative_length=relative_length, **values)
