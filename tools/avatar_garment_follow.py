"""Attach hanging coat sections to the nearby leg using VRM rotation constraints."""
import bpy
from mathutils import Vector

from avatar_dress import dress_profile, knee_profile_weights
from avatar_rig_defaults import FOLLOW, SPRINGS
from avatar_skirt_fit_io import rest_edit
from avatar_skirt_follow import set_constraint, set_rest_frame

TAG = 'hallway_'
JOINT_FIELDS = ('hit_radius', 'stiffness', 'drag_force', 'gravity_power', 'gravity_dir')


def _attach(edit_bones, joint, target, name, group, profile):
    driver = edit_bones[target]
    if group is None and joint.parent and driver.parent:
        # The fixed cloth root inherits the driver's upstream torso motion.
        # The helper below it adds only the selected spine's local bend.
        anchor = joint.parent
        anchor.use_connect = False
        anchor.parent = driver.parent
    helper = edit_bones.new(name)
    set_rest_frame(helper, driver)
    # Use the garment attachment as the pivot while matching the leg's axes.
    direction = helper.tail - helper.head
    helper.head = joint.head.copy()
    helper.tail = helper.head + direction
    helper.parent = joint.parent
    if group:
        helper[TAG + 'constraint_group'] = group
    if profile is not None:
        helper[TAG + 'knee_profile'] = profile
    joint.use_connect = False
    joint.parent = helper
    return helper.name


def install_garment_follow(rig, chains):
    """Return contact plans for each simulated section and a constraint report."""
    ext = rig.data.vrm_addon_extension
    sb = ext.spring_bone1
    humanoid = ext.vrm1.humanoid.human_bones
    legs = []
    for side in ('left', 'right'):
        thigh = rig.data.bones.get(getattr(humanoid, side + '_upper_leg').node.bone_name)
        calf = rig.data.bones.get(getattr(humanoid, side + '_lower_leg').node.bone_name)
        if thigh and calf:
            legs.append((thigh, calf))
    if not legs:
        return chains, []

    lateral = legs[0][0].head_local - legs[-1][0].head_local
    lateral.z = 0.
    lateral = lateral.normalized() if lateral.length > 1e-8 else Vector((1., 0., 0.))
    forward = lateral.cross(Vector((0., 0., 1.)))
    midpoint = sum((thigh.head_local for thigh, _ in legs), Vector()) / len(legs)
    springs = {spring.vrm_name: spring for spring in sb.springs}
    plans = []
    for chain in chains:
        if not chain.get('garment'):
            continue
        spring_name = chain.get('spring_name', chain['names'][0].rsplit('_', 1)[0])
        spring = springs.get(spring_name)
        if spring is None or len(spring.joints) < 2:
            continue
        names = [joint.node.bone_name for joint in spring.joints]
        bones = [rig.data.bones[name] for name in names]
        if any(bone.parent and bone.parent.get(TAG + 'coat_follow') for bone in bones):
            continue
        if chain.get('coat_torso'):
            spine = chain.get('coat_spine')
            if spine in rig.data.bones:
                plans.append((chain, spring_name, names, [(0, spine, None, None)]))
            continue
        center = sum((bone.head_local for bone in bones), Vector()) / len(bones)
        thigh, calf = min(legs, key=lambda pair: abs((center - pair[0].head_local).dot(lateral)))
        hip, knee, ankle = thigh.head_local.z, calf.head_local.z, calf.tail_local.z
        if hip <= knee or knee <= ankle or min(bones[-1].head_local.z, bones[-1].tail_local.z) >= hip - .32 * (hip - knee):
            continue
        # A high coat attachment stays on the torso. Start thigh follow at
        # the first available joint near the socket that still has a free link.
        lower = chain.get('coat_lower_root')
        first_lower = names.index(lower) if lower in names else 0
        start = next((i for i, bone in enumerate(bones[:-1])
                      if i >= first_lower and bone.head_local.z <= hip + .18 * (hip - knee)), None)
        if start is None:
            continue
        starts = [(start, thigh.name, 'Skirt', None)]
        spine = chain.get('coat_spine')
        if start and spine in rig.data.bones:
            starts.insert(0, (0, spine, None, None))
        top, bottom = bones[start].head_local.z, bones[-1].head_local.z
        if top > knee > bottom and dress_profile(top, bottom, knee, ankle, len(bones)-start)['lower_follow']:
            candidates = range(start + 1, len(bones) - 1)
            index = min(candidates, key=lambda i: abs(bones[i].head_local.z - knee), default=None)
            if index is not None:
                radial = center - midpoint
                radial.z = 0.
                profile = knee_profile_weights(radial.normalized().dot(forward))
                starts.append((index, calf.name, 'Skirt Knee', profile))
        plans.append((chain, spring_name, names, starts))
    if not plans:
        return chains, []

    attachments = []
    tips = {}
    with rest_edit(rig):
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            edit = rig.data.edit_bones
            for _, spring_name, names, starts in plans:
                for index, target, group, profile in starts:
                    joint = edit[names[index]]
                    label = 'Spine' if group is None else 'Knee' if profile is not None else 'Thigh'
                    if index:
                        # The preceding simulation ends at a distinct terminal
                        # node. The next one inherits that motion via its helper.
                        previous = edit[names[index-1]]
                        tip = edit.new(joint.name + '_FollowTip')
                        tip.head = joint.head.copy()
                        tip.tail = tip.head + (previous.tail - previous.head) * .1
                        tip.parent = previous
                        tip.use_deform = True
                        tips[spring_name, index] = tip.name
                        joint.use_connect = False
                        joint.parent = tip
                    # Omit the torso suffix from the helper stem to fit the
                    # armature's name limit in both host packages.
                    stem = joint.name.replace('_Accessory_SurfacePanel_', '_Coat_') if group is None else joint.name
                    name = _attach(edit, joint, target, stem + '_' + label + 'Follow', group, profile)
                    edit[name][TAG + 'coat_follow'] = True
                    attachments.append((name, target, group, profile))
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')

    report = []
    for name, target, group, profile in attachments:
        value = FOLLOW[group] if group else .65
        if profile is not None:
            value = sum(weight * FOLLOW[key] for weight, key in zip(
                profile, ('Skirt Knee', 'Skirt Knee Side', 'Skirt Knee Back')))
        pose_bone = rig.pose.bones[name]
        constraint = pose_bone.constraints.new('COPY_ROTATION')
        set_constraint(constraint, rig, target, value)
        if group:
            pose_bone.bone[TAG + 'follow_constraint'] = constraint.name
        else:
            constraint.name = 'Coat spine follow'
        report.append(dict(bone=name, target=target, influence=value))

    sections = {}
    for chain, spring_name, names, starts in plans:
        upper = springs[spring_name]
        torso_boundary = next((index for index, _, group, _ in starts if group == 'Skirt'), 0)
        bounds = [0] + [index for index, *_ in starts if index] + [len(names)]
        records = []
        for joint in upper.joints:
            fields = {key: getattr(joint, key) for key in JOINT_FIELDS}
            fields['gravity_dir'] = tuple(fields['gravity_dir'])
            custom = {key: value for key, value in joint.items() if key.startswith(TAG)}
            records.append((fields, custom))
        refs = [ref.collider_group_uuid for ref in upper.collider_groups]
        center = upper.center.bone_name
        parts = []
        for a, b in zip(bounds, bounds[1:]):
            spring = upper
            section_names = names[a:b]
            if b < len(names):
                section_names += [tips[spring_name, b]]
            if a:
                spring = sb.springs.add()
                spring.vrm_name = f'{spring_name}_Follow_{a:02d}'
                spring.center.bone_name = center
                for uuid in refs:
                    spring.collider_groups.add().collider_group_uuid = uuid
            elif len(bounds) > 2:
                upper.joints.clear()
            if len(bounds) > 2:
                for name, (fields, custom) in zip(section_names, records[a:b+1]):
                    joint = spring.joints.add()
                    joint.node.bone_name = name
                    for key, value in fields.items():
                        setattr(joint, key, value)
                    for key, value in custom.items():
                        joint[key] = value
            # Include original root skin weights if the first joint was fixed.
            skin_names = list(chain['names'][:chain['names'].index(names[0])]) if a == 0 else []
            torso = bool(chain.get('coat_torso') or (chain.get('coat_spine') and 0 < torso_boundary == b))
            if chain.get('coat_lower_root') and (torso or 0 < torso_boundary <= a):
                defaults = SPRINGS['Accessory']
                # Front panels inherit motion over the knees from these
                # torso roots. Keep that side firmer so fast leg lifts don't
                # catch the hem as the side and rear fabric move more freely.
                if torso:
                    hips = rig.data.bones.get('Hips')
                    radial = rig.data.bones[names[a]].head_local - (hips.head_local if hips else midpoint)
                    radial.z = 0.
                    front = radial.normalized().dot(forward)
                    firmness = max(0., min(1., (front+.15)/.72))
                    firmness = firmness*firmness*(3.-2.*firmness)
                for j, joint in enumerate(spring.joints):
                    if torso:
                        joint.stiffness *= 1.15+1.05*firmness
                        joint.drag_force = max(joint.drag_force, .42+.14*firmness)
                        joint.gravity_power *= .28
                    else:
                        # Continue the taper across an optional knee split.
                        t = min(1., (a + j - torso_boundary) / max(1, len(names)-torso_boundary-2))
                        drape = t*t*(3.-2.*t)
                        joint.stiffness *= .90-.28*drape
                        joint.drag_force = min(1., joint.drag_force+.04*drape)
                        joint.gravity_power *= 1.+.75*drape
                    joint[TAG + 'stiffness_ratio'] = joint.stiffness / defaults['stiffness']
                    joint[TAG + 'generated_drag_ratio'] = joint.drag_force / defaults['drag']
                    joint[TAG + 'generated_gravity_ratio'] = joint.gravity_power / defaults['gravity']
            parts.append(dict(chain, names=skin_names + names[a:b], spring_name=spring.vrm_name,
                              coat_torso=torso, coat_hem=rig.data.bones[names[-1]].head_local.z))
        sections[id(chain)] = parts
    return [part for chain in chains for part in sections.get(id(chain), [chain])], report
