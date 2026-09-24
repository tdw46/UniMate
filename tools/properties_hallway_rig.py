"""Persistent per-rig configuration; updates are applied by an undoable operator."""
import bpy
from mathutils import Vector


def active_rig(context):
    obj = context.object
    if obj and obj.type != 'ARMATURE':
        obj = obj.find_armature()
    return obj if obj and (obj.get('hallway_generated_rig') or obj.get('unimate_secondary_generator')) else None


def springs(rig, prefix):
    ext = getattr(rig.data, 'vrm_addon_extension', None)
    return [s for s in ext.spring_bone1.springs if s.vrm_name.startswith(prefix)] if ext else []


def follow_constraints(rig, group):
    for pb in rig.pose.bones:
        if pb.bone.get('hallway_constraint_group') == group:
            for c in pb.constraints:
                if c.type == 'COPY_ROTATION' and c.name == pb.bone.get('hallway_follow_constraint'):
                    yield c


def skirt_colliders(rig):
    ext = getattr(rig.data, 'vrm_addon_extension', None)
    if not ext:
        return []
    sb = ext.spring_bone1
    owned = {r.collider_uuid for g in sb.collider_groups if g.vrm_name == 'Secondary_SkirtBody' for r in g.colliders}
    foreign = {r.collider_uuid for g in sb.collider_groups if g.vrm_name != 'Secondary_SkirtBody' for r in g.colliders}
    return [c for c in sb.colliders if c.uuid in owned - foreign and c.bpy_object and c.shape_type == 'Capsule']


def capture_collider_baselines(rig, refresh_limits=False):
    """Store fitted radii and safe expansion bounds in bone-local rest space."""
    colliders = [c for c in skirt_colliders(rig) if refresh_limits or 'hallway_base_radius' not in c.bpy_object]
    if not colliders:
        return
    from avatar_colliders import spring_samples, segment_distance
    samples = spring_samples(rig, 'skirt')
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not any(m.type == 'ARMATURE' and m.object == rig for m in obj.modifiers):
            continue
        ids = {g.index for g in obj.vertex_groups if g.name.startswith('Secondary_Skirt_')}
        transform = rig.matrix_world.inverted() @ obj.matrix_world
        samples.extend((transform @ v.co, 0., 'surface') for v in obj.data.vertices
                       if any(g.group in ids and g.weight > .05 for g in v.groups))
    height = max(b.head_local.z for b in rig.data.bones) - min(b.head_local.z for b in rig.data.bones)
    margin = max(height * .0015, 1e-6)
    previous = rig.data.pose_position
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    try:
        for c in colliders:
            shape = c.shape.capsule
            matrix = rig.data.bones[c.node.bone_name].matrix_local
            a, b = matrix @ Vector(shape.offset), matrix @ Vector(shape.tail)
            limit = min((segment_distance(p, a, b) - hit - margin for p, hit, _ in samples), default=shape.radius)
            if 'hallway_base_radius' not in c.bpy_object:
                c.bpy_object['hallway_base_radius'] = float(shape.radius)
            c.bpy_object['hallway_radius_limit'] = max(0., limit)
    finally:
        rig.data.pose_position = previous
        bpy.context.view_layer.update()


class HALLWAY_PG_Follow(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    influence: bpy.props.FloatProperty(name='Follow', default=1., min=0., max=1., subtype='FACTOR')


class HALLWAY_PG_Spring(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    prefix: bpy.props.StringProperty()
    drag: bpy.props.FloatProperty(name='Drag', min=0., max=1., default=.6)
    stiffness: bpy.props.FloatProperty(name='Root Stiffness', min=0., soft_max=4., default=1.6,
        description='Stiffness of the first simulated joint; preserves the generated taper down each chain')
    gravity: bpy.props.FloatProperty(name='Gravity', min=0., soft_max=.2, default=.025)


class HALLWAY_PG_Rig(bpy.types.PropertyGroup):
    initialized: bpy.props.BoolProperty(default=False)
    follow_groups: bpy.props.CollectionProperty(type=HALLWAY_PG_Follow)
    spring_groups: bpy.props.CollectionProperty(type=HALLWAY_PG_Spring)
    skirt_thickness: bpy.props.FloatProperty(name='Skirt Collider Thickness', default=1., min=.05, max=3.,
        description='Multiplier of fitted skirt collider radii, capped at rest clearance; hair colliders are unchanged')


def initialize(rig):
    register()
    rig['hallway_generated_rig'] = True
    settings = rig.hallway_rig
    for pb in rig.pose.bones:
        for c in pb.constraints:
            if pb.name.startswith('Secondary_SkirtFollow_') and c.name == 'UniMate skirt leg follow':
                pb.bone['hallway_constraint_group'] = 'Skirt'
                pb.bone['hallway_follow_constraint'] = c.name
    # New groups can be registered through the same bone/constraint metadata.
    names = sorted({b.get('hallway_constraint_group') for b in rig.data.bones if b.get('hallway_constraint_group')})
    for name in names:
        if name not in settings.follow_groups:
            constraints = list(follow_constraints(rig, name))
            if constraints:
                group = settings.follow_groups.add()
                group.name = name
                group.influence = constraints[0].influence
    for name in ('Skirt', 'Hair'):
        prefix = 'Secondary_' + name + '_'
        chains = springs(rig, prefix)
        if chains and name not in settings.spring_groups:
            group = settings.spring_groups.add()
            group.name, group.prefix = name, prefix
            first = chains[0].joints[0]
            group.drag, group.stiffness, group.gravity = first.drag_force, first.stiffness, first.gravity_power
    capture_collider_baselines(rig)
    settings.initialized = True
    return settings


def apply_settings(rig):
    from avatar_physics_preview import suspended
    with suspended():
        return _apply_settings(rig)


def _apply_settings(rig):
    settings = rig.hallway_rig
    capture_collider_baselines(rig, refresh_limits=True)
    count = 0
    for group in settings.follow_groups:
        for constraint in follow_constraints(rig, group.name):
            constraint.influence = group.influence  # Preserve all VRM export flags.
    for group in settings.spring_groups:
        for spring in springs(rig, group.prefix):
            root = spring.joints[0].stiffness
            for index, joint in enumerate(spring.joints):
                ratio = joint.stiffness / root if root > 1e-8 else max(0., 1. - .1 * index)
                joint['hallway_stiffness_ratio'] = joint.get('hallway_stiffness_ratio', ratio)
                joint.stiffness = group.stiffness * joint['hallway_stiffness_ratio']
                joint.drag_force, joint.gravity_power = group.drag, group.gravity
    for collider in skirt_colliders(rig):
        obj = collider.bpy_object
        requested = obj['hallway_base_radius'] * settings.skirt_thickness
        radius = min(requested, obj['hallway_radius_limit'])
        collider.shape.capsule.radius = radius
        count += radius < requested - 1e-7
    bpy.context.view_layer.update()
    return count


CLASSES = (HALLWAY_PG_Follow, HALLWAY_PG_Spring, HALLWAY_PG_Rig)


def register():
    for cls in CLASSES:
        if not getattr(cls, 'is_registered', False):
            bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Object, 'hallway_rig'):
        bpy.types.Object.hallway_rig = bpy.props.PointerProperty(type=HALLWAY_PG_Rig)


def unregister():
    if hasattr(bpy.types.Object, 'hallway_rig'):
        del bpy.types.Object.hallway_rig
    for cls in reversed(CLASSES):
        if getattr(cls, 'is_registered', False):
            bpy.utils.unregister_class(cls)
