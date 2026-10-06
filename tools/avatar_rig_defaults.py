"""Shared generation and panel-reset defaults; no Blender registration needed."""
import json

FOLLOW = {
    'Skirt': .55,
    'Skirt Knee': .60,
    'Skirt Knee Side': .55,
    'Skirt Knee Back': .50,
    'Skirt Support': .75,
}
SPRINGS = {
    'Skirt': dict(drag=.4, stiffness=1.6, non_root_stiffness=1., gravity=.025),
    'Hair': dict(drag=.4, stiffness=1., non_root_stiffness=1., gravity=.035),
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
