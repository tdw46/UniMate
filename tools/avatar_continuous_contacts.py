"""Restore ordered VRM skirt chains without altering bones or skinning."""

def merge_contact_segments(rig):
    """Merge uniquely linked segment heads; retain the last segment's tip.

    Earlier segment helper tips become ordinary unanimated leaf bones. They
    carry no skin weights and can remain for reversible comparison. Collider
    groups must be regenerated against all endpoints after this operation.
    """
    sb = rig.data.vrm_addon_extension.spring_bone1
    originals = [s for s in sb.springs if s.vrm_name.startswith('Secondary_Skirt_')]
    segmented = [s for s in originals if '_Contact_' in s.vrm_name]
    if not segmented:
        return 0
    if len(segmented) != len(originals) or any(len(s.joints) != 2 for s in segmented):
        raise ValueError('Expected a complete, uniquely segmented skirt')
    heads = {s.joints[0].node.bone_name:s for s in segmented}
    if len(heads) != len(segmented):
        raise ValueError('Shared spring heads cannot be merged')
    following = {}
    roots = []
    for name in heads:
        parent = rig.data.bones[name].parent
        if parent and parent.name in heads:
            if parent.name in following:
                raise ValueError('Branched segmented skirt requires separate chains')
            following[parent.name] = name
        else:
            roots.append(name)
    fields = ('hit_radius','stiffness','drag_force','gravity_power','gravity_dir')
    def joint_record(j):
        return dict(bone=j.node.bone_name, values={f:list(getattr(j,f)) if f=='gravity_dir' else getattr(j,f) for f in fields},
                    ratio=j.get('hallway_stiffness_ratio'))
    records = []
    for root in roots:
        names = [];name = root
        while name:
            names.append(name);name = following.get(name)
        springs = [heads[n] for n in names]
        centers = {s.center.bone_name for s in springs}
        if len(centers) != 1:
            raise ValueError('Cannot combine springs with different centers')
        records.append(dict(name=springs[0].vrm_name.rsplit('_Contact_',1)[0],center=centers.pop(),
                            joints=[joint_record(s.joints[0]) for s in springs]+[joint_record(springs[-1].joints[-1])]))
    assert sum(len(r['joints'])-1 for r in records)==len(segmented)
    for i in reversed(range(len(sb.springs))):
        if sb.springs[i].vrm_name.startswith('Secondary_Skirt_'):
            sb.springs.remove(i)
    for record in records:
        s=sb.springs.add();s.vrm_name=record['name'];s.center.bone_name=record['center']
        for item in record['joints']:
            j=s.joints.add();j.node.bone_name=item['bone']
            for f,value in item['values'].items():setattr(j,f,value)
            if item['ratio'] is not None:j['hallway_stiffness_ratio']=item['ratio']
    return len(records)
