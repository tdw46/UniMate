"""Constrain opposing shaft influence around generated knees and elbows in rest space."""


def refine_hinge_weights(rig, meshes):
    hinges = []
    for side in ('.L', '.R'):
        for upper, lower in (('Thigh', 'Shin'), ('UpperArm', 'Forearm')):
            parent = rig.data.bones.get(upper + side)
            child = rig.data.bones.get(lower + side)
            if not parent or not child or child.parent != parent:
                continue
            length = min(parent.length, child.length)
            upper_axis = (parent.tail_local - parent.head_local).normalized()
            lower_axis = (child.tail_local - child.head_local).normalized()
            axis = upper_axis + lower_axis
            bend = lower_axis - upper_axis
            # Inside a flexed knee the top of the calf extends behind the pivot.
            # A nearly straight rest chain has no reliable bend-side signal.
            inner = bend / max(.12, bend.length) if upper == 'Thigh' else bend * 0.
            if length > 1e-6 and axis.length_squared > 1e-6:
                hinges.append((parent.name, child.name, child.head_local.copy(),
                               axis.normalized(), length, inner))

    counts = {lower: 0 for _, lower, *_ in hinges}
    for obj in meshes:
        to_rig = rig.matrix_world.inverted() @ obj.matrix_world
        for upper, lower, pivot, axis, length, inner in hinges:
            proximal = obj.vertex_groups.get(upper)
            distal = obj.vertex_groups.get(lower)
            if proximal is None or distal is None:
                continue
            side = upper[-2:]
            neighbors = ('Hips', 'Foot' + side) if upper.startswith('Thigh') else ('Clavicle' + side, 'Hand' + side)
            permitted = {g.index for name in (upper, lower, *neighbors)
                         if (g := obj.vertex_groups.get(name)) is not None}
            for vertex in obj.data.vertices:
                # Articulated garments and rigid fittings own their existing blend.
                row = {g.group: g.weight for g in vertex.groups if g.weight > 0}
                if proximal.index not in row or distal.index not in row or row.keys() - permitted:
                    continue
                delta = to_rig @ vertex.co - pivot
                station = delta.dot(axis)
                radius = (delta - axis * station).length
                if radius > .55 * length or abs(station) > length:
                    continue
                # A thicker limb needs a wider crease. Keep the central blend and
                # taper the maximum opposing influence smoothly to zero on either shaft.
                width = min(.22 * length, max(.15 * length, 1.15 * radius))
                contact_station = station + min(.08 * length, .50 * max(0., delta.dot(inner)))
                t = min(1., max(0., .5 + .5 * contact_station / width))
                mass = row[proximal.index] + row[distal.index]
                # For mixed hip/hand rows only enforce the envelope when the
                # hinge pair owns most of the deformation. Keep that mass fixed.
                share = min(1., max(0., (mass / sum(row.values()) - .55) / .35))
                strength = share * share * (3. - 2. * share)
                target = t * t * (3. - 2. * t)
                # A tolerance around the crease retains small variations from
                # the bind. Both edges of the envelope meet the rigid shafts.
                slack = .12 * 4. * t * (1. - t)
                low = mass * strength * max(0., target - slack)
                high = mass * (1. - strength * max(0., 1. - target - slack))
                # Outside the crease only remove opposing spill. A well-bound
                # shaft should never acquire extra influence from the other bone.
                crease = max(0., 1. - abs(station) / (.12 * length))
                crease = crease * crease * (3. - 2. * crease)
                if station < 0.:
                    low *= crease
                    # The inside-knee shift must not carry calf weight up the
                    # hamstring. Decay the remaining spill on the upper shaft.
                    upper_fade = max(0., 1. + station / width) ** 2.5
                    high = min(high, mass * (1. - strength * (1. - upper_fade)))
                else:
                    high = mass - crease * (mass - high)
                distal_mass = min(high, max(low, row[distal.index]))
                if abs(row[distal.index] - distal_mass) <= 1e-7:
                    continue
                for group, weight in ((distal, distal_mass), (proximal, mass - distal_mass)):
                    if weight <= 1e-8:
                        group.remove([vertex.index])
                    else:
                        group.add([vertex.index], weight, 'REPLACE')
                counts[lower] += 1
    return counts
