"""Fresh symmetric skeleton specifications from an imported VRM humanoid.

Source bones supply rest landmarks only; the shared rebuilder discards their
weights, controls and physics before binding. Final mesh coordinates stay put.
"""
from mathutils import Vector
from avatar_springs import humanoid_roles


def specs_from_vrm(source):
    ext = source.data.vrm_addon_extension
    if ext.spec_version == '1.0':
        mapping = {name.value: bone.node.bone_name for name, bone in
                   ext.vrm1.humanoid.human_bones.human_bone_name_to_human_bone().items()}
    else:
        mapping = {b.bone: b.node.bone_name for b in ext.vrm0.humanoid.human_bones}
        for side in ('left', 'right'):
            mapping[side+'ThumbMetacarpal'] = mapping.get(side+'ThumbProximal', '')
            mapping[side+'ThumbProximal'] = mapping.get(side+'ThumbIntermediate', '')
    aliases = humanoid_roles()
    missing = [role for role in aliases if mapping.get(role) not in source.data.bones]
    if missing:
        raise ValueError('Missing VRM humanoid landmarks: '+', '.join(missing))
    bones = {alias: source.data.bones[mapping[role]] for role, alias in aliases.items()}
    point = lambda name: bones[name].head_local.copy()
    height = bones['Head'].tail_local.z-min(point('Foot.L').z, point('Foot.R').z)
    if height <= 0 or point('UpperArm.L').x <= point('UpperArm.R').x:
        raise ValueError('VRM rest landmarks must use the canonical upright frame')
    specs = [('Root', Vector((0,0,0)), Vector((0,0,height*.07)), None)]
    for name, end, parent in (('Hips','Spine','Root'), ('Spine','Chest','Hips'),
                              ('Chest','Neck','Spine'), ('Neck','Head','Chest')):
        specs.append((name, point(name), point(end), parent))
    specs.append(('Head', point('Head'), bones['Head'].tail_local.copy(), 'Neck'))
    for side in ('L', 'R'):
        for name, end, parent in (('Clavicle','UpperArm','Chest'), ('UpperArm','Forearm','Clavicle'),
                                 ('Forearm','Hand','UpperArm'), ('Hand','Middle1','Forearm'),
                                 ('Thigh','Shin','Hips'), ('Shin','Foot','Thigh'), ('Foot','Toe','Shin')):
            specs.append((name+'.'+side, point(name+'.'+side), point(end+'.'+side),
                          parent if parent in ('Chest','Hips') else parent+'.'+side))
        for name, parent in [('Toe','Foot')] + [(d+str(i), 'Hand' if i==1 else d+str(i-1))
                for d in ('Thumb','Index','Middle','Ring','Little') for i in range(1,4)]:
            key = name+'.'+side
            if name[-1:] in ('1','2'):
                end = point(name[:-1]+str(int(name[-1])+1)+'.'+side)
            else:
                bone = bones[key]
                end = bone.children[0].head_local.copy() if len(bone.children)==1 else bone.tail_local.copy()
            specs.append((key, point(key), end, parent+'.'+side))
    lookup = {name:(a,b) for name,a,b,_ in specs}
    for name,a,b,_ in specs:
        if name.endswith('.L'):
            ra,rb = lookup[name[:-1]+'R']
            for left,right in ((a,ra),(b,rb)):
                mean = (left+Vector((-right.x,right.y,right.z)))*.5
                left[:] = mean
                right[:] = (-mean.x,mean.y,mean.z)
        elif not name.endswith('.R'):
            a.x=b.x=0
        if (b-a).length < height*1e-6:
            raise ValueError('Degenerate VRM landmark segment: '+name)
    return specs
