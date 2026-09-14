"""Pose, bake, unrig, Boolean-cut, clear weights, then freshly rig nine avatars.

Run in a disposable Blender process only. Source files are never overwritten.
"""
import json
import math
import os
from pathlib import Path
import sys

import bpy
import bmesh
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from avatar_apparel_weights import correct_apparel
from avatar_source_import import adapt_humanoid, split_material_regions
OUT = Path(os.environ.get('AVATAR_EVAL_ROOT', ROOT / 'outputs/avatar_grid')).resolve()
SOURCES = OUT / 'sources'
FRAMES = 360


def select_only(objects, active=None):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = active or objects[0]


def world_turn(rig, bone, axis, degrees):
    basis = (rig.matrix_world @ rig.data.bones[bone].matrix_local).to_quaternion()
    return basis.inverted() @ Quaternion(axis, math.radians(degrees)) @ basis


def prepare(entry):
    name = entry['id']
    folder = OUT / 'avatars' / name
    folder.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(SOURCES / entry['model']))
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_set(0)
    original_rig = next(o for o in scene.objects if o.type == 'ARMATURE')
    source_adapter = adapt_humanoid(SOURCES / entry['model'], original_rig)
    original_rig.data.pose_position = 'POSE'
    for obj in list(scene.objects):
        obj.animation_data_clear()
    for pb in original_rig.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
        pb.rotation_mode = 'QUATERNION'
    for suffix, sign in [('L',1),('R',-1)]:
        bone = 'UpperArm.' + suffix
        original_rig.pose.bones[bone].rotation_quaternion = world_turn(original_rig, bone, (0,1,0), sign*20)
    bpy.context.view_layer.update()
    original_bone_count = len(original_rig.data.bones)
    landmarks = {pb.name:{'head':original_rig.matrix_world @ pb.head, 'tail':original_rig.matrix_world @ pb.tail} for pb in original_rig.pose.bones}
    cutoff = landmarks['Abdomen']['head'].lerp(landmarks['Torso']['head'], .35).z
    meshes = []
    rigid_parts = {}
    removed_props = []
    deps = bpy.context.evaluated_depsgraph_get()
    original_weight_count = sum(len(v.groups) for o in scene.objects if o.type=='MESH' for v in o.data.vertices)
    for obj in list(scene.objects):
        if obj.type != 'MESH':
            continue
        lower = obj.name.lower()
        if any(word in lower for word in ('staff','bow','dagger','sword','pistol','icosphere')):
            removed_props.append(obj.name)
            continue
        # Semantic attachment labels locate fresh rigid weights, never old values.
        if obj.parent_type == 'BONE':
            old_bone = obj.parent_bone
            rigid_map = {'Head':'Head','Torso':'Chest','Chest':'Chest','Abdomen':'Spine','Hips':'Root'}
            for suffix in ('L','R'):
                rigid_map.update({'Shoulder.'+suffix:'Clavicle.'+suffix,
                                  'UpperArm.'+suffix:'UpperArm.'+suffix,
                                  'LowerArm.'+suffix:'Forearm.'+suffix,
                                  'Wrist.'+suffix:'Hand.'+suffix,'Fist.'+suffix:'Hand.'+suffix})
            if old_bone in rigid_map:
                rigid_parts[obj.name] = rigid_map[old_bone]
        if lower.endswith('_head') or lower in ('head','face'):
            rigid_parts[obj.name] = 'Head'
        world = obj.matrix_world.copy()
        evaluated = obj.evaluated_get(deps)
        mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=deps)
        obj.modifiers.clear()
        obj.data = mesh
        obj.parent = None
        obj.matrix_world = world
        obj.vertex_groups.clear()
        mesh.transform(obj.matrix_world)
        obj.matrix_world = Matrix.Identity(4)
        meshes.append(obj)
    semantic_regions = []
    if source_adapter:
        meshes, semantic_regions = split_material_regions(meshes, rigid_parts)
    for obj in list(bpy.data.objects):
        if obj not in meshes:
            bpy.data.objects.remove(obj, do_unlink=True)
    for armature in list(bpy.data.armatures):
        bpy.data.armatures.remove(armature)
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    audit = {'id':name, 'source':entry['source_page'], 'original_bones_removed':original_bone_count,
             'original_weight_assignments_removed':original_weight_count, 'preparation_pose':'20 degree A-pose, baked to mesh',
             'cutoff_z_before_normalization':cutoff, 'removed_handheld_props':removed_props, 'booleans':[]}
    if source_adapter:
        audit['source_adapter'] = source_adapter
        audit['semantic_material_regions'] = semantic_regions
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,cutoff-10))
    cutter = bpy.context.object
    cutter.name = 'Bust Boolean cutter'
    cutter.scale = (10,10,10)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for obj in list(meshes):
        before = len(obj.data.vertices)
        zs = [v.co.z for v in obj.data.vertices]
        if min(zs) >= cutoff - 1e-5:
            continue
        select_only([obj])
        modifier = obj.modifiers.new('Destructive bust cut', 'BOOLEAN')
        modifier.operation = 'DIFFERENCE'
        modifier.solver = 'EXACT'
        modifier.object = cutter
        if hasattr(modifier, 'use_hole_tolerant'):
            modifier.use_hole_tolerant = True
        result = bpy.ops.object.modifier_apply(modifier=modifier.name)
        assert result == {'FINISHED'}
        remaining = len(obj.data.vertices)
        audit['booleans'].append({'mesh':obj.name, 'operation':'DIFFERENCE', 'solver':'EXACT', 'applied':True, 'vertices_before':before, 'vertices_after':remaining})
        if not obj.data.polygons:
            meshes.remove(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
        else:
            assert min(v.co.z for v in obj.data.vertices) >= cutoff - 1e-4, obj.name
    bpy.data.objects.remove(cutter, do_unlink=True)
    assert audit['booleans'], 'Expected an actual destructive Boolean operation'
    assert meshes
    # Normalize each cut bust independently without retaining any old rig data.
    top = max(v.co.z for obj in meshes for v in obj.data.vertices)
    factor = 2.65 / (top - cutoff)
    transform = Matrix.Scale(factor,4) @ Matrix.Translation((0,0,-cutoff))
    for obj in meshes:
        obj.data.transform(transform)
        obj.vertex_groups.clear()
        # glTF splits vertices at UV/normal seams. Weld coincident geometry
        # while retaining per-corner UVs so the new heat bind sees connected skin.
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
    for point in landmarks.values():
        point['head'] = transform @ point['head']
        point['tail'] = transform @ point['tail']
    audit['unrigged_checkpoint'] = {
        'armature_objects':sum(o.type=='ARMATURE' for o in scene.objects),
        'armature_datablocks':len(bpy.data.armatures),
        'vertex_groups':sum(len(o.vertex_groups) for o in meshes),
        'weight_assignments':sum(len(v.groups) for o in meshes for v in o.data.vertices),
        'remaining_boolean_modifiers':sum(m.type=='BOOLEAN' for o in meshes for m in o.modifiers),
        'remaining_armature_modifiers':sum(m.type=='ARMATURE' for o in meshes for m in o.modifiers),
    }
    assert not any(audit['unrigged_checkpoint'].values()), audit
    scene['preparation_audit'] = json.dumps(audit)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'01_cut_unrigged.blend'))

    rig = bpy.data.objects.new(name+'_FreshRig', bpy.data.armatures.new(name+'_FreshSkeleton'))
    scene.collection.objects.link(rig)
    rig['rig_source'] = 'New 13-bone rig; source rigs and all old weights deleted before binding'
    select_only([rig])
    bpy.ops.object.mode_set(mode='EDIT')
    torso = landmarks['Torso']['head']
    neck = landmarks['Neck']['head']
    head = landmarks['Head']['head']
    root = Vector((torso.x,torso.y,.02))
    spine = root.lerp(torso,.25)
    specs = [('Root',root,spine,None), ('Spine',spine,torso,'Root'), ('Chest',torso,neck,'Spine'),
             ('Neck',neck,head,'Chest'), ('Head',head,landmarks['Head']['tail'],'Neck')]
    for suffix in ('L','R'):
        upper = landmarks['UpperArm.'+suffix]['head']
        elbow = landmarks['LowerArm.'+suffix]['head']
        hand_key = ('Wrist.' if 'Wrist.'+suffix in landmarks else 'Fist.')+suffix
        wrist = landmarks[hand_key]['head']
        hand_tail = wrist + (wrist-elbow).normalized() * (wrist-elbow).length*.45
        specs += [('Clavicle.'+suffix,landmarks['Shoulder.'+suffix]['head'],upper,'Chest'),
                  ('UpperArm.'+suffix,upper,elbow,'Clavicle.'+suffix),
                  ('Forearm.'+suffix,elbow,wrist,'UpperArm.'+suffix),
                  ('Hand.'+suffix,wrist,hand_tail,'Forearm.'+suffix)]
    for bone_name, start, end, parent in specs:
        bone = rig.data.edit_bones.new(bone_name)
        bone.head, bone.tail = start, end
        if parent:
            bone.parent = rig.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    # Run the same automatic-weight binding used by the initial bust flow.
    select_only(meshes+[rig], rig)
    binding_result = bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    assert binding_result == {'FINISHED'}
    repaired = 0
    sleeve_vertices = 0
    for obj in meshes:
        rigid = rigid_parts.get(obj.name)
        if rigid:
            obj.vertex_groups.clear()
            obj.vertex_groups.new(name=rigid).add(list(range(len(obj.data.vertices))),1,'REPLACE')
        # Disconnected clothing can have no heat solution. Assign such vertices
        # using distances to the NEW skeleton only; old weights no longer exist.
        groups = {b.name:obj.vertex_groups.get(b.name) or obj.vertex_groups.new(name=b.name) for b in rig.data.bones}
        for vertex in obj.data.vertices:
            total = sum(g.weight for g in vertex.groups)
            if total > 1e-6:
                continue
            distances = []
            for bone in rig.data.bones:
                a,b = bone.head_local,bone.tail_local
                length = (b-a).length_squared
                t = max(0,min(1,(vertex.co-a).dot(b-a)/length))
                distances.append(((vertex.co-(a+(b-a)*t)).length,bone.name))
            distances.sort()
            values = [(1/max(d,.015)**4,n) for d,n in distances[:3]]
            total = sum(w for w,n in values)
            for w,n in values: groups[n].add([vertex.index],w/total,'REPLACE')
            repaired += 1
        # Bulky sleeves can sit farther from their arm bone than from the chest.
        # For lateral sleeve/arm vertices, solve weights against the new arm
        # chain so a jacket cannot remain fixed while the arm exits its sleeve.
        if not rigid:
            for vertex in obj.data.vertices:
                suffix = 'L' if vertex.co.x >= 0 else 'R'
                upper = rig.data.bones['UpperArm.'+suffix]
                if abs(vertex.co.x) < abs(upper.head_local.x) + upper.length*.18:
                    continue
                distances = []
                for prefix in ('UpperArm.','Forearm.','Hand.'):
                    bone = rig.data.bones[prefix+suffix]
                    a,b = bone.head_local,bone.tail_local
                    t = max(0,min(1,(vertex.co-a).dot(b-a)/(b-a).length_squared))
                    distances.append(((vertex.co-(a+(b-a)*t)).length,bone.name))
                if min(d for d,n in distances) > upper.length*.65:
                    continue
                for group in obj.vertex_groups:
                    group.remove([vertex.index])
                values = [(1/max(d,.015)**4,n) for d,n in distances]
                total = sum(w for w,n in values)
                for w,n in values: groups[n].add([vertex.index],w/total,'REPLACE')
                sleeve_vertices += 1
        select_only([obj])
        bpy.ops.object.vertex_group_limit_total(limit=4)
        bpy.ops.object.vertex_group_normalize_all(lock_active=False)
        assert all(abs(sum(g.weight for g in v.groups)-1)<1e-4 for v in obj.data.vertices)
    audit['apparel_correction'] = correct_apparel(meshes, rig, rigid_parts)
    audit['fresh_rig'] = {'bones':len(rig.data.bones), 'unweighted_vertices_repaired_from_new_bones':repaired,
                          'lateral_sleeve_vertices_reweighted_from_new_bones':sleeve_vertices,
                          'rigid_parts':rigid_parts, 'bone_names':[b.name for b in rig.data.bones]}
    rig.animation_data_create()
    for frame in range(FRAMES):
        phase = frame // 120
        t = (frame % 120) / 119
        pulse = math.sin(2*math.pi*t)*math.sin(math.pi*t)
        for pb in rig.pose.bones:
            pb.rotation_mode = 'QUATERNION'
            pb.matrix_basis = Matrix.Identity(4)
        for suffix,sign in [('L',1),('R',-1)]:
            angle = 25 if phase != 2 else 25 - 75*math.sin(math.pi*t)**2
            rig.pose.bones['UpperArm.'+suffix].rotation_quaternion = world_turn(rig,'UpperArm.'+suffix,(0,1,0),sign*angle)
            if phase == 2:
                rig.pose.bones['Forearm.'+suffix].rotation_quaternion = world_turn(rig,'Forearm.'+suffix,(0,0,1),-sign*65*math.sin(math.pi*t)**2)
        if phase == 0:
            rig.pose.bones['Head'].rotation_quaternion = world_turn(rig,'Head',(0,0,1),40*pulse) @ world_turn(rig,'Head',(1,0,0),16*math.sin(4*math.pi*t)*math.sin(math.pi*t))
        elif phase == 1:
            rig.pose.bones['Neck'].rotation_quaternion = world_turn(rig,'Neck',(0,1,0),20*pulse) @ world_turn(rig,'Neck',(1,0,0),15*math.sin(4*math.pi*t)*math.sin(math.pi*t))
        for pb in rig.pose.bones:
            pb.keyframe_insert('rotation_quaternion',frame=frame)
    rig.animation_data.action.name = 'evaluation'
    scene.frame_start, scene.frame_end = 0, FRAMES-1
    scene.frame_set(0)
    scene['preparation_audit'] = json.dumps(audit)
    select_only(meshes+[rig],rig)
    bpy.ops.export_scene.gltf(filepath=str(folder/'02_fresh_rig.glb'),export_format='GLB',use_selection=True,export_animations=True,export_animation_mode='ACTIONS',export_force_sampling=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'02_fresh_rig.blend'))
    (folder/'preparation.json').write_text(json.dumps(audit,indent=2))
    print('AVATAR_PREPARED',name,len(meshes),len(rig.data.bones),flush=True)
    return audit


entries = json.loads((SOURCES/'manifest.json').read_text())
if '--first-only' in sys.argv:
    entries = entries[:1]
audits = [prepare(entry) for entry in entries]
(OUT/'preparation_validation.json').write_text(json.dumps(audits,indent=2))
