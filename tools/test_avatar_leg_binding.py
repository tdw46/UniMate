"""Compare rebuilt lower limbs with a baseline; verify original texture data.

Blender --background --factory-startup --python tools/test_avatar_leg_binding.py
-- corrected/generated.blend source.blend baseline/generated.blend report.json
"""
import bpy,sys,json,hashlib
from pathlib import Path
from mathutils import Matrix,Quaternion,Vector
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_apparel_weights import weights
from rebuild_mmd_character import fingerprint
if not bpy.app.background:raise RuntimeError('Use isolated Blender')
corrected,source,baseline,output=sys.argv[sys.argv.index('--')+1:]
for repo in getattr(getattr(bpy.context.preferences,'extensions',None),'repos',[]):
    if repo.module=='user_default':repo.use_custom_directory=True;repo.custom_directory=str(Path.home()/'Documents/Blender/extensions/user_default')
bpy.ops.preferences.addon_enable(module='bl_ext.user_default.vrm')


def textures(meshes):
    rows={}
    for o in meshes:
        values=[]
        for m in o.data.materials:
            base=m.vrm_addon_extension.mtoon1.pbr_metallic_roughness.base_color_texture
            im=base.index.source
            values.append(dict(material=m.name,image=im.name if im else None,
                packed_sha256=hashlib.sha256(im.packed_file.data).hexdigest() if im and im.packed_file else None,
                offset=list(base.extensions.khr_texture_transform.offset),scale=list(base.extensions.khr_texture_transform.scale)))
        rows[o.name]=values
    return rows


def inspect(path,pose=False):
    bpy.ops.wm.open_mainfile(filepath=str(Path(path).resolve()))
    rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig]
    geometry={o.name:fingerprint(o) for o in meshes};presentation=textures(meshes)
    if 'Thigh.L' not in rig.data.bones:return dict(geometry=geometry,textures=presentation)
    mid=rig.data.bones['Hips'].head_local.x
    cutoff=min(rig.data.bones['Thigh.'+side].tail_local.z+.8*rig.data.bones['Thigh.'+side].length for side in ('L','R'))
    vertices=[];maximum=0.;contaminated=0
    for obj in meshes:
        slots={i for i,m in enumerate(obj.data.materials) if any(t in m.name.lower() for t in ('skin','shoe','boot','sock'))}
        ids={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}
        for i in ids:
            p=obj.data.vertices[i].co
            if p.z>=cutoff or abs(p.x-mid)<1e-5:continue
            own='L' if p.x>mid else 'R';other='R' if own=='L' else 'L'
            w=weights(obj,i);bad=sum(x for n,x in w.items() if n.endswith('.'+other))
            maximum=max(maximum,bad);contaminated+=bad>1e-5;vertices.append((obj,i,own,w))
    motion=0.
    if pose:
        for side in ('L','R'):
            for n,degrees in (('Thigh',30),('Shin',70),('Foot',25)):
                p=rig.pose.bones[n+'.'+side];p.rotation_mode='QUATERNION';p.rotation_quaternion=Quaternion((1,0,0),degrees*3.141592653589793/180)
            bpy.context.view_layer.update()
            transforms={b.name:rig.pose.bones[b.name].matrix@b.matrix_local.inverted() for b in rig.data.bones}
            for obj,i,own,w in vertices:
                if own==side:continue
                p=obj.data.vertices[i].co
                deformed=sum(((transforms[n]@p)*x for n,x in w.items()),Vector())
                motion=max(motion,(deformed-p).length)
            for p in rig.pose.bones:p.matrix_basis=Matrix.Identity(4)
        assert motion<1e-6,motion
        assert maximum<1e-6,maximum
    return dict(geometry=geometry,textures=presentation,lower_vertices=len(vertices),
                opposite_leg_contaminated_vertices=contaminated,maximum_opposite_leg_weight=maximum,
                maximum_opposite_leg_motion=motion)


old=inspect(baseline);original=inspect(source);new=inspect(corrected,pose=True)
assert new['geometry']==original['geometry']==old['geometry']
assert new['textures']==original['textures']==old['textures']
report=dict(before={k:v for k,v in old.items() if k not in ('geometry','textures')},
            after={k:v for k,v in new.items() if k not in ('geometry','textures')},
            source_geometry_uvs_keys_and_packed_texture_bytes_unchanged=True)
Path(output).write_text(json.dumps(report,indent=2)+'\n');print('LEG_BINDING_OK',json.dumps(report),flush=True)
