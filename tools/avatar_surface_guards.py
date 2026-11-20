"""Fit portable spring contacts to the underlying skin or garment in bind space."""

from collections import defaultdict

import numpy as np
from avatar_apparel_weights import components
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from avatar_skirt_fit_io import rest_edit
from avatar_vrm_colliders import add_capsule, organize_colliders

TAG = "hallway_"
PREFIX = "Secondary_"
FAMILIES = {"hair": ("Hair_", "HairContact_"), "jacket": ("Accessory_Jacket_", "JacketContact_")}


def _support_surface(rig, meshes, bone_names, joint_chain):
    """Partition moving shell vertices and non-spring support triangles in one mesh pass."""
    vertices, triangles, owners = [], [], []
    shell = defaultdict(list)
    for obj in meshes:
        mesh = obj.data
        transform = rig.matrix_world.inverted_safe() @ obj.matrix_world
        names = {g.index: g.name for g in obj.vertex_groups}
        start = len(vertices)
        support = []
        moving = set()
        for v in mesh.vertices:
            point = transform @ v.co
            vertices.append(point)
            influences = [(names.get(g.group, ""), g.weight) for g in v.groups if g.weight > 0.001]
            chains = {joint_chain[n] for n, w in influences if n in joint_chain and w > 0.05}
            for chain in chains:
                shell[chain].append(tuple(point))
            if any(n.startswith(PREFIX) for n, w in influences):
                moving.add(v.index)
            support.append([(n, w) for n, w in influences if n in bone_names])
        # Exclude the fixed roots of mostly dynamic islands. Other body parts
        # on the same object or texture atlas can still supply support.
        excluded = set(moving)
        if moving:
            for island in components(mesh):
                if sum(i in moving for i in island) > len(island) * 0.25:
                    excluded.update(island)
        for p in mesh.polygons:
            ids = tuple(p.vertices)
            if any(i in excluded for i in ids):
                continue
            weights = defaultdict(float)
            for i in ids:
                for bone, w in support[i]:
                    weights[bone] += w
            if not weights or sum(weights.values()) < 0.4 * len(ids):
                continue
            owners.append(max(weights, key=weights.get))
            triangles.append(tuple(start + i for i in ids))
    tree = BVHTree.FromPolygons(vertices, triangles) if triangles else None
    return tree, owners, shell


def _clearance(points, hits, a, b):
    axis = b - a
    t = np.clip((points - a) @ axis / max(float(axis @ axis), 1e-20), 0, 1)
    return float((np.linalg.norm(points - a - t[:, None] * axis, axis=1) - hits).min())


def _support_normal(tree, owners, index, location, radial, normal, height):
    # Smooth adjacent facets at bind time within the same support patch.
    reach = height * 0.018
    mean = normal.copy()
    for _, neighbor, face, separation in tree.find_nearest_range(location, reach):
        if face == index or owners[face] != owners[index]:
            continue
        if neighbor.dot(radial) < 0:
            neighbor = -neighbor
        agreement = neighbor.dot(normal)
        if agreement > 0.65:
            mean += neighbor * ((1.0 - separation / reach) ** 2 * agreement)
    normal = mean.normalized()
    outward = radial.normalized()
    agreement = normal.dot(outward)
    if agreement > 0.0:
        normal = normal.lerp(outward, 0.5 * agreement).normalized()
    return normal


def _growth_limit(relative, distances, hit_radii, direction, base, gap):
    # Once the nearest rest point is reached, broaden the guard inward.
    # Its tangent plane stays put, so increasing coverage need not kick
    # a close lock or a fitted jacket away from the body.
    # |p - (center - d*n)| >= base + hit + gap + d.
    # The quadratic terms cancel for a unit direction. Intersect
    # the remaining linear bounds across this chain's endpoints.
    n = np.array(direction)
    n /= np.linalg.norm(n)
    separation = base + hit_radii + gap
    slope = 2 * (separation - relative @ n)
    limiting = slope > 1e-7
    growth = 2 * base
    if limiting.any():
        bound = ((distances - separation) * (distances + separation))[limiting] / slope[limiting]
        growth = min(growth, max(0.0, float(bound.min())))
    return base + growth


def _fit_family(rig, meshes, family, height, groups, shapes, springs):
    if not springs:
        return 0
    hum = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    roles = ("head",) if family == "hair" else ("hips", "spine", "chest", "upper_chest")
    bones = {getattr(hum, role).node.bone_name for role in roles} - {""}
    joints = {j.node.bone_name: s.vrm_name for s, g in springs for j in s.joints}
    tree, owners, shell = _support_surface(rig, meshes, bones, joints)
    sb = rig.data.vrm_addon_extension.spring_bone1
    margin = 0.0015 * height
    caps = {}
    added = 0
    for spring, group_id in springs:
        group = sb.collider_groups[groups[group_id]]
        joint_tips = [
            (rig.data.bones[t.node.bone_name].head_local.copy(), j.hit_radius)
            for j, t in zip(spring.joints, spring.joints[1:])
            if t.node.bone_name in rig.data.bones
        ]
        if not joint_tips:
            continue
        vertices = shell[spring.vrm_name]
        points = np.array([tuple(p) for p, r in joint_tips] + vertices)
        hits = np.array([r for p, r in joint_tips] + [0.0] * len(vertices))
        for ref in group.colliders:
            if ref.collider_uuid not in shapes:
                continue
            c = sb.colliders[shapes[ref.collider_uuid]]
            obj = c.bpy_object
            bone = rig.data.bones.get(c.node.bone_name)
            if not obj or not obj.get("unimate_generated_collider") or not bone or c.shape_type != "Capsule":
                continue
            shape = c.shape.capsule
            a, b = (np.array(bone.matrix_local @ Vector(p)) for p in (shape.offset, shape.tail))
            limit = _clearance(points, hits, a, b) - margin
            caps[c.uuid] = min(limit, caps.get(c.uuid, float("inf")))
        if not tree:
            continue
        tips = points[: len(joint_tips)]
        hit_radii = hits[: len(joint_tips)]
        for tip_index, (tip, hit) in enumerate(joint_tips):
            location, normal, index, distance = tree.find_nearest(
                tip, height * (0.065 if family == "hair" else 0.095)
            )
            if location is None:
                continue
            bone = rig.data.bones[owners[index]]
            axis = bone.tail_local - bone.head_local
            t = min(1.0, max(0.0, (location - bone.head_local).dot(axis) / max(axis.length_squared, 1e-12)))
            radial = location - (bone.head_local + axis * t)
            if radial.dot(normal) < 0:
                normal = -normal
            # Use the exterior side of the supporting sheet. Rear-facing hits
            # across an opening must not attract a guard to the opposite bank.
            if (tip - location).dot(normal) < -height * 0.008:
                continue
            normal = _support_normal(tree, owners, index, location, radial, normal, height)
            # The existing fitted round contacts own the neutral clearance.
            # For nearby surfaces the directional guards can engage earlier.
            gap = 0.0005 * height
            nominal = height * (0.050 if family == "hair" else 0.090)
            radius = min(2.0 * nominal, max(nominal, radial.length * 1.25))
            if family == "jacket":
                # Leave the free hem a little travel before it contacts the trunk.
                # Taper continuously from the fitted upper section along each chain.
                fraction = tip_index / max(1, len(joint_tips) - 1)
                radius *= 1.0 - 0.28 * fraction
                gap += height * 0.001 * fraction
            directions = [normal]
            # A broad lock can sweep across a curved cheek while its guide
            # slides along a single tangent. Bracket that local arc with two
            # oblique tangent contacts where the cloth or hair rides close.
            lateral = axis.cross(normal)
            # A jacket hem needs room to slide around the trunk. A single
            # round guide there avoids competing projections during reversal.
            bracket = family == "hair" or tip_index < max(1, len(joint_tips) // 2)
            if bracket and lateral.length_squared > 1e-10 and distance < 0.07 * height:
                lateral.normalize()
                # Match the guide spread to the width of the driven shell.
                local = points[len(tips) :] - np.array(tip)
                local = local[np.linalg.norm(local, axis=1) <= 1.5 * radius]
                width = float(np.quantile(np.abs(local @ np.array(lateral)), 0.85)) if len(local) else 0.0
                slope = min(1.0, width / radius)
                if slope > 0.20:
                    directions = [(normal + lateral * slope * sign).normalized() for sign in (-1.0, 1.0)]
            for direction in directions:
                # The skin can be well behind the spring guide on a thick lock or a
                # folded shell. Preserve that shell-to-guide distance as the limb
                # turns. Only a VRM joint endpoint can displace a chain at rest.
                center = np.array(tip - direction * (radius + hit + gap))
                relative = tips - center
                distances = np.linalg.norm(relative, axis=1)
                limit = float((distances - hit_radii).min()) - gap
                base = min(radius, limit)
                if base < radius * 0.5:
                    continue
                offset = bone.matrix_local.inverted_safe() @ Vector(center)
                c = add_capsule(
                    rig, dict(bone=bone.name, offset=offset, tail=offset, radius=base), organize=False
                )
                obj = c.bpy_object
                obj[TAG + "base_radius"] = base
                obj[TAG + "radius_limit"] = _growth_limit(
                    relative, distances, hit_radii, direction, base, gap
                )
                obj[TAG + "guard_origin"] = tuple(obj.location)
                obj[TAG + "guard_inward"] = tuple(-(bone.matrix_local.to_3x3().inverted_safe() @ direction))
                group.colliders.add().collider_uuid = c.uuid
                added += 1
    owned_groups = {uid for _, uid in springs}
    shared = {
        ref.collider_uuid
        for group in sb.collider_groups
        if group.uuid not in owned_groups
        for ref in group.colliders
    }
    for uid, limit in caps.items():
        c = sb.colliders[shapes[uid]]
        obj = c.bpy_object
        base = float(c.get(TAG + "thickness_rest", obj.get(TAG + "base_radius", c.shape.capsule.radius)))
        key = TAG + family + "_"
        obj[key + "base_radius"] = base
        obj[key + "radius_limit"] = max(0.0, limit)
        # Shared contacts keep their original size for the other spring.
        # Owned geometry should already be rest-clear straight from Generate.
        if uid not in shared and c.shape.capsule.radius > obj[key + "radius_limit"]:
            c.shape.capsule.radius = obj[key + "radius_limit"]
    return added


def fit_surface_guards(rig, meshes, height, families=tuple(FAMILIES)):
    if not np.isfinite(height) or height <= 0:
        return 0
    sb = rig.data.vrm_addon_extension.spring_bone1
    groups = {g.uuid: i for i, g in enumerate(sb.collider_groups)}
    shapes = {c.uuid: i for i, c in enumerate(sb.colliders)}
    uses = defaultdict(int)
    for s in sb.springs:
        for ref in s.collider_groups:
            uses[ref.collider_group_uuid] += 1
    added = 0
    with rest_edit(rig):
        for family in families:
            strand, contact = FAMILIES[family]
            springs = []
            for s in sb.springs:
                if not s.vrm_name.startswith(PREFIX + strand):
                    continue
                for ref in s.collider_groups:
                    uid = ref.collider_group_uuid
                    if uid not in groups or uses[uid] != 1:
                        continue
                    group = sb.collider_groups[groups[uid]]
                    if group.vrm_name.startswith(PREFIX + contact) and not group.get(TAG + "surface_fit"):
                        springs.append((s, uid))
            added += _fit_family(rig, meshes, family, height, groups, shapes, springs)
            for s, uid in springs:
                sb.collider_groups[groups[uid]][TAG + "surface_fit"] = True
        if added:
            organize_colliders(rig)
    return added


def place_guard(obj, radius, base):
    if TAG + "guard_inward" in obj:
        obj.location = Vector(obj[TAG + "guard_origin"]) + Vector(obj[TAG + "guard_inward"]) * max(
            0.0, radius - base
        )


def ensure_surface_guards(rig, family):
    if not any(
        g.vrm_name.startswith(PREFIX + FAMILIES[family][1]) and not g.get(TAG + "surface_fit")
        for g in rig.data.vrm_addon_extension.spring_bone1.collider_groups
    ):
        return
    meshes = {
        o for scene in rig.users_scene for o in scene.objects if o.type == "MESH" and o.find_armature() == rig
    }
    if not meshes:
        return
    z = [p.z for b in rig.data.bones for p in (b.head_local, b.tail_local)]
    fit_surface_guards(rig, sorted(meshes, key=lambda o: o.name), max(z) - min(z), (family,))


def scale_contacts(rig, family, factor):
    import bpy

    ensure_surface_guards(rig, family)
    sb = rig.data.vrm_addon_extension.spring_bone1
    prefix = PREFIX + FAMILIES[family][1]
    shared_groups = {
        ref.collider_group_uuid
        for spring in sb.springs
        if not spring.vrm_name.startswith(PREFIX + FAMILIES[family][0])
        for ref in spring.collider_groups
    }
    ours = {
        ref.collider_uuid for g in sb.collider_groups if g.vrm_name.startswith(prefix) for ref in g.colliders
    }
    foreign = {
        ref.collider_uuid
        for g in sb.collider_groups
        if g.uuid in shared_groups or not g.vrm_name.startswith(prefix)
        for ref in g.colliders
    }
    selected = set(bpy.context.selected_objects)
    try:
        for c in sb.colliders:
            obj = c.bpy_object
            if c.uuid not in ours or c.uuid in foreign or not obj:
                continue
            key = TAG + family + "_" if TAG + family + "_base_radius" in obj else TAG
            if key + "base_radius" not in obj or key + "radius_limit" not in obj:
                continue
            base = obj[key + "base_radius"]
            radius = min(factor * base, obj[key + "radius_limit"])
            if abs(c.shape.capsule.radius - radius) > 1e-8:
                c.shape.capsule.radius = radius
                place_guard(obj, radius, base)
    finally:
        for obj in selected.symmetric_difference(bpy.context.selected_objects):
            obj.select_set(obj in selected)
