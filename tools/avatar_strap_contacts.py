"""Fit VRM leg contacts to suspended straps using the skinned limb in rest space."""
import numpy as np
from mathutils import Vector

from avatar_vrm_colliders import add_capsule, add_group


def install_strap_contacts(rig, meshes, prefix, height):
    from avatar_colliders import segment_distance

    spring_data = rig.data.vrm_addon_extension.spring_bone1
    straps = [spring for spring in spring_data.springs
              if spring.vrm_name.startswith(prefix) and spring.get('hallway_suspended_strap')]
    if not straps:
        return
    human = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    roles = ('left_upper_leg', 'right_upper_leg', 'left_lower_leg', 'right_lower_leg')
    bones = [rig.data.bones.get(getattr(human, role).node.bone_name) for role in roles]
    samples = {bone.name: [] for bone in bones if bone is not None}
    for obj in meshes:
        groups = {g.index: samples[g.name] for g in obj.vertex_groups if g.name in samples}
        if not groups:
            continue
        transform = rig.matrix_world.inverted() @ obj.matrix_world
        for vertex in obj.data.vertices:
            for group in vertex.groups:
                if group.group in groups and group.weight > .6:
                    groups[group.group].append(tuple(transform @ vertex.co))
                    break
    limbs = []
    for bone in bones:
        if bone is None or not samples[bone.name] or bone.length < 1e-7:
            continue
        axis = bone.tail_local - bone.head_local
        points = np.asarray(samples[bone.name]) - bone.head_local
        fractions = points @ np.asarray(axis) / axis.length_squared
        radial = points - fractions[:, None] * np.asarray(axis)
        middle = radial[(fractions > .1) & (fractions < .9)]
        if len(middle):
            limbs.append((bone, axis, float(np.quantile(np.linalg.norm(middle, axis=1), .9))))

    for spring in straps:
        endpoints = [(rig.data.bones.get(j.node.bone_name), j.hit_radius) for j in spring.joints]
        if len(endpoints) < 2 or any(bone is None for bone, _ in endpoints):
            continue
        line = [bone.head_local for bone, _ in endpoints]
        group = None
        for bone, axis, radius in limbs:
            head = bone.head_local
            beside = [p for p in line if -.15 < (p-head).dot(axis)/axis.length_squared < 1.15]
            if not beside or radius <= 0:
                continue
            direction = sum(beside, Vector((0, 0, 0))) / len(beside) - head
            direction -= axis * (direction.dot(axis) / axis.length_squared)
            # Round the near leg's contact face toward the suspended web. The
            # far leg uses an axial capsule for steps crossing the body midline.
            offset = .5 * radius if direction.length <= 2.8 * radius else 0.
            shift = -direction.normalized() * offset
            a, b = head+shift, bone.tail_local+shift
            clearance = min(segment_distance(p, a, b)-hit for p, (_, hit) in zip(line[1:], endpoints))
            fitted = min(1.35 * radius + offset, clearance - .001 * height)
            if fitted <= 0:
                continue
            if group is None:
                group = add_group(rig, prefix+'StrapLeg_'+spring.vrm_name[len(prefix):])
                spring.collider_groups.add().collider_group_uuid = group.uuid
            local = bone.matrix_local.inverted()
            collider = add_capsule(rig, dict(bone=bone.name, offset=local@a, tail=local@b,
                                             radius=fitted), organize=False)
            group.colliders.add().collider_uuid = collider.uuid
