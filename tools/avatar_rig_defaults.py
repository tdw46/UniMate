"""Shared generation and panel-reset defaults; no Blender registration needed."""
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
PHYSICS_ENABLED = True
COLLIDERS_VISIBLE = False
BONES_VISIBLE = True
