"""Independent anatomical heat domains for overlapping or enclosed legs.

All solves use temporary surfaces and the new skeleton. Source weights and
final vertex positions never participate. Footwear samples its own leg's
fresh skin solution so coincident skin and shoe surfaces move together.
"""
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
from avatar_apparel_weights import assign, weights


def _bone_distance(point,bone):
    delta=bone.tail_local-bone.head_local
    t=max(0.,min(1.,(point-bone.head_local).dot(delta)/max(delta.length_squared,1e-20)))
    return (point-(bone.head_local+delta*t)).length_squared


def bind_leg_domains(rig, meshes):
    from autorig_bust import repaired_heat
    # This runs before Mixamo renaming and spring generation in the rebuilder.
    if not all(n in rig.data.bones for n in ('Hips','Thigh.L','Thigh.R','Shin.L','Shin.R')):
        return dict(applied=False, reason='No canonical leg landmarks')
    deform={b.name:b.use_deform for b in rig.data.bones}
    upper=rig.data.bones['Hips'].tail_local.z
    center=rig.data.bones['Hips'].head_local.x
    rows=[]
    try:
        for side,sign in (('L',1),('R',-1)):
            thigh=rig.data.bones['Thigh.'+side]
            blend_start=thigh.tail_local.z+(thigh.head_local.z-thigh.tail_local.z)*.8
            if upper<=blend_start:continue
            refs=[];faces=[];lookup={}
            legs=[rig.data.bones[n+'.'+side] for n in ('Thigh','Shin','Foot','Toe') if n+'.'+side in rig.data.bones]
            arms=[rig.data.bones[n+'.'+side] for n in ('UpperArm','Forearm','Hand') if n+'.'+side in rig.data.bones]
            for obj in meshes:
                # Lowered hands can lie below the pelvis in an A pose. Height
                # alone must never admit them to a leg's heat surface.
                eligible={v.index for v in obj.data.vertices if v.co.z<upper and sign*(v.co.x-center)>=-1e-7
                          and (not arms or min(_bone_distance(v.co,b) for b in legs)
                               <=min(_bone_distance(v.co,b) for b in arms))}
                for face in obj.data.polygons:
                    mat=obj.data.materials[face.material_index] if face.material_index<len(obj.data.materials) else None
                    label=mat.name.lower() if mat else ''
                    if not any(t in label for t in ('skin','body')) or any(t in label for t in ('cloth','hair','shoe')):continue
                    if not all(i in eligible for i in face.vertices):continue
                    indices=[]
                    for i in face.vertices:
                        key=(obj,i)
                        if key not in lookup:lookup[key]=len(refs);refs.append(key)
                        indices.append(lookup[key])
                    faces.append(indices)
            if len(faces)<8:continue
            # Do not guess a leg domain from an unrelated small skin island.
            if min(obj.data.vertices[i].co.z for obj,i in refs)>rig.data.bones['Shin.'+side].tail_local.z:
                continue
            allowed={'Hips'}|{n+'.'+side for n in ('Thigh','Shin','Foot','Toe')}
            for b in rig.data.bones:b.use_deform=b.name in allowed
            data=bpy.data.meshes.new('Temporary anatomical leg heat')
            data.from_pydata([obj.data.vertices[i].co for obj,i in refs],[],faces);data.update()
            proxy=bpy.data.objects.new('Temporary anatomical leg heat',data)
            bpy.context.scene.collection.objects.link(proxy)
            try:
                audit=repaired_heat([proxy],rig,allow_unweighted=False)
                solved=[weights(proxy,v.index) for v in data.vertices]
                for (obj,i),v,value in zip(refs,data.vertices,solved):
                    t=max(0.,min(1.,(upper-v.co.z)/(upper-blend_start)));t=t*t*(3-2*t)
                    old=weights(obj,i)
                    result={n:old.get(n,0.)*(1-t)+value.get(n,0.)*t for n in old.keys()|value.keys()}
                    total=sum(result.values());assign(obj,[i],{n:w/total for n,w in result.items()})
                data.calc_loop_triangles()
                triangles=[tuple(t.vertices) for t in data.loop_triangles]
                tree=BVHTree.FromPolygons([v.co for v in data.vertices],triangles,all_triangles=True)
                transferred=0;max_distance=0.
                for obj in meshes:
                    slots={i for i,m in enumerate(obj.data.materials) if m and any(t in m.name.lower() for t in ('shoe','boot','sock'))}
                    ids={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}
                    for i in ids:
                        p=obj.data.vertices[i].co
                        if sign*(p.x-center)<=0 or p.z>=blend_start:continue
                        hit,_,index,distance=tree.find_nearest(p)
                        if hit is None or distance>thigh.length*.25:continue
                        tri=triangles[index];corners=[data.vertices[j].co for j in tri]
                        if (corners[1]-corners[0]).cross(corners[2]-corners[0]).length_squared<1e-18:continue
                        bary=barycentric_transform(hit,*corners,Vector((1,0,0)),Vector((0,1,0)),Vector((0,0,1)))
                        value={}
                        for j,w in zip(tri,bary):
                            for n,x in solved[j].items():value[n]=value.get(n,0.)+max(0.,w)*x
                        total=sum(value.values());assign(obj,[i],{n:w/total for n,w in value.items()})
                        transferred+=1;max_distance=max(max_distance,distance)
                rows.append(dict(side=side,skin_vertices=len(refs),footwear_vertices=transferred,
                                 maximum_footwear_distance=max_distance,heat=audit))
            finally:
                bpy.data.objects.remove(proxy,do_unlink=True)
                if not data.users:bpy.data.meshes.remove(data)
    finally:
        for name,value in deform.items():rig.data.bones[name].use_deform=value
    return dict(applied=bool(rows),domains=rows)
