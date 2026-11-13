# SPDX-License-Identifier: GPL-3.0-or-later
# Auto-Rig Pro 3.78.34, Artell. Extracted from src/auto_rig.py.
# Upstream source permits GPL-2.0-or-later; distributed here under GPL-3.0-or-later.
# See NOTICE.md and COPYING. No proprietary artwork or rig assets are included.
"""Unmodified split/heat/join/transfer body; host utilities supplied explicitly."""
import bpy
import sys
from contextlib import redirect_stdout
from .host import (get_object, set_active_object, duplicate_object, is_object_id,
                   get_object_id, delete_object, disable_mask_modifiers, restore_mask_modifiers)

def bind_parts(self, scn, obj):
    obj_name = obj.name
    has_duplicated = False
    bones_deform_dict = {b.name: b.use_deform for b in self.rig.data.bones}
    # split loose parts in option for better auto-skinning
    if scn.arp_bind_split and scn.arp_bind_engine == 'HEAT_MAP':
        # duplicate to preserve vertex ID when splitting
        if not is_object_id(obj, '_arp_temp_skin', suffix_only=True):
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')                
            dupli_obj = duplicate_object(method='data', obj=get_object(obj_name))
            set_active_object(dupli_obj.name)
            dupli_obj['arp_id'] = obj_name+'_arp_copy'
            obj = get_object(bpy.context.active_object.name)
            has_duplicated = True

        print('Split...')
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        set_active_object(obj.name)
        bpy.ops.mesh.separate(type='LOOSE')
        split_objects = [split_obj for split_obj in bpy.context.selected_objects if split_obj.type == "MESH"]
    else:
        split_objects = [obj]

    # compute skinning
    for i, split_obj in enumerate(split_objects):
        #print('skinning object:', str(i + 1) + '/' + str(len(split_objects)))
        sys.stdout.write('\r  '+'Skinning object part: '+str(i+1)+'/'+ str(len(split_objects))+'                                         ')
        try: sys.stdout.flush()# # unknown compatibility breakage with flush() in rare cases reported by a user
        except: pass

        def get_armature_mod(_name):
            obj = bpy.context.active_object
            for mod in obj.modifiers:
                if mod.type == "ARMATURE":
                    if mod.object:
                        if mod.object.name == _name:
                            return mod

        if self.bind_rig_add:
            # bind to rig add
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            set_active_object(split_obj.name)
            set_active_object(self.rig_add.name)

            bpy.ops.object.parent_set(type='ARMATURE_AUTO')

            set_active_object(split_obj.name)
            get_armature_mod(self.rig_add.name).name = "rig_add"

        if self.bind_rig:
            # bind to rig
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            set_active_object(split_obj.name)
            set_active_object(self.rig.name)

            with redirect_stdout(self.skin_prints):
                bpy.ops.object.parent_set(type='ARMATURE_AUTO')

            set_active_object(split_obj.name)
            rig_mod = get_armature_mod(self.rig.name)
            rig_mod.name = "rig"
            rig_mod.show_in_editmode = True
            rig_mod.show_on_cage = True
            rig_mod.use_deform_preserve_volume = scn.arp_bind_preserve


    # restore temporarily deform disabled bones
    for bname in bones_deform_dict:
        b = self.rig.data.bones.get(bname)
        b.use_deform = bones_deform_dict[bname]

    # merge the split objects
    print('\n')

    bpy.ops.object.select_all(action='DESELECT')
    if len(split_objects) > 1:
        for split_obj in split_objects:
            set_active_object(split_obj.name)

    set_active_object(obj.name)

    if len(split_objects) > 1:
        bpy.ops.object.join()

    # transfer weights, vertices location based

    if has_duplicated:
        bpy.ops.object.select_all(action='DESELECT')
        set_active_object(obj_name) 
        arp_copy_obj = get_object_id(obj_name+'_arp_copy')
        arp_copy_name = arp_copy_obj.name
        set_active_object(arp_copy_name)

        # Mask modifiers prevent correct data transfer
        disable_dict = disable_mask_modifiers(bpy.context.selected_objects)

        # Source object Shape Keys are not evaluated when transferring vgroups, then they must be disabled on the target object
        # Note: shape keys are not properly supported when binding anyway,
        # heat maps are performed in the Basis shape coordinates -> important shape deformations lead to wrong skinning
        if arp_copy_obj.data.shape_keys:
            if len(arp_copy_obj.data.shape_keys.key_blocks):
                arp_copy_obj.active_shape_key_index = 0# basis
                arp_copy_obj.show_only_shape_key = True

        #print("Transfer weights from", arp_copy_name, 'to', obj_name)
        try:
            bpy.ops.object.data_transfer(data_type='VGROUP_WEIGHTS', vert_mapping='NEAREST', layers_select_src='ALL', layers_select_dst='NAME')
        except:
            print("Cound not transfer vertex groups")

        restore_mask_modifiers(disable_dict)

        # remove duplicate
        delete_object(arp_copy_obj)
        set_active_object(obj_name)


