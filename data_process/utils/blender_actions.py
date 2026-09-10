"""Action access for legacy Blender and Blender's layered action API."""


def iter_fcurves(action):
    if action is None:
        return
    if getattr(action, 'is_action_layered', False):
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', ()):
                    yield from bag.fcurves
    else:
        yield from action.fcurves


def bind_action_slot(owner, action):
    """Bind the single compatible slot, preferring the owner's name.

    Refuse ambiguous multi-object actions rather than silently animating
    with an unrelated slot. Legacy actions need no slot binding.
    """
    ad = owner.animation_data
    if not hasattr(ad, 'action_slot') or not getattr(action, 'is_action_layered', False):
        return
    slots = [s for s in action.slots if s.target_id_type == owner.id_type]
    matching = [s for s in slots if s.name_display == owner.name]
    candidates = matching or slots
    if len(candidates) > 1:
        raise ValueError(f'Ambiguous action slots for {owner.name}: {action.name}')
    if candidates and ad.action_slot != candidates[0]:
        ad.action_slot = candidates[0]


def new_fcurves(action, owner):
    """Create a channel bag for a new action, or use legacy F-curves."""
    if hasattr(action, 'slots') and hasattr(action, 'layers'):
        slot = action.slots.new(id_type=owner.id_type, name=owner.name)
        layer = action.layers.new('Animation')
        strip = layer.strips.new(type='KEYFRAME')
        owner.animation_data.action_slot = slot
        return strip.channelbag(slot, ensure=True).fcurves
    return action.fcurves
