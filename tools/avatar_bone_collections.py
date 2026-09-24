"""Organize generated rigs by purpose without changing deformation flags."""
NAMES = ('Physics', 'Constraints', 'Deform', 'Controls')


def organize_bones(rig):
    if rig.mode == 'EDIT':
        raise ValueError('Exit Edit Mode before organizing the rig')
    groups = {name: [] for name in NAMES}
    ext = getattr(rig.data, 'vrm_addon_extension', None)
    physics = {j.node.bone_name for s in ext.spring_bone1.springs
               if s.vrm_name.startswith('Secondary_') for j in s.joints} if ext else set()
    for bone in rig.data.bones:
        if bone.get('hallway_constraint_group') or bone.name.startswith('Secondary_SkirtFollow_'):
            role = 'Constraints'
        elif bone.name in physics or bone.name.startswith(('Secondary_Hair_', 'Secondary_Skirt_')):
            role = 'Physics'  # Includes attachment roots and terminal joints.
        else:
            role = 'Deform' if bone.use_deform else 'Controls'
        groups[role].append(bone)
    if hasattr(rig.data, 'collections'):
        collections = {}
        for name in NAMES:
            collection = next((c for c in rig.data.collections if c.get('hallway_role') == name), None)
            if collection is None:
                collection = rig.data.collections.new(name)
                collection['hallway_role'] = name
            collections[name] = collection
        for role, bones in groups.items():
            for bone in bones:
                for collection in list(bone.collections):
                    if collection.get('hallway_role') and collection != collections[role]:
                        collection.unassign(bone)
                collections[role].assign(bone)
    elif rig.data.bones and hasattr(rig.data.bones[0], 'layers'):
        # Legacy Blender armature layers: one purpose per first four layers.
        for index, bones in enumerate(groups.values()):
            for bone in bones:
                bone.layers = tuple(i == index for i in range(32))
        rig.data.layers = tuple(i < 4 for i in range(32))
    return {name: len(bones) for name, bones in groups.items()}
