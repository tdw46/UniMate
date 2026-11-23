"""Round generated arm-contact interiors; never reposition source surface vertices."""
import heapq
import numpy as np
from mathutils import Vector
from avatar_apparel_weights import weights,assign
from avatar_surface_attachments import normalized

FILL_ATTRIBUTE='hallway_generated_fill'


def _circle(points):
    origin=points.mean(axis=0);scale=max(float(np.ptp(points,axis=0).max()),1e-12)
    p=(points-origin)/scale
    for _ in range(3):
        x,y=p.T
        c,d,e=np.linalg.lstsq(np.column_stack((x,y,np.ones(len(x)))),-x*x-y*y,rcond=None)[0]
        center=np.array([-c*.5,-d*.5]);r2=float(center@center-e)
        if r2<=0:return None
        error=np.abs(np.linalg.norm(p-center,axis=1)-np.sqrt(r2))
        p=p[error<=max(float(np.quantile(error,.85)),1e-8)]
        if len(p)<12:return None
    return origin+center*scale,float(np.sqrt(r2)*scale)


def _refine_fill_faces(obj,rig):
    import bmesh
    bones=[rig.data.bones.get('UpperArm.'+s) for s in ('L','R')]
    if not all(bones):return 0
    pitch=min(b.length for b in bones)*.035
    transform=rig.matrix_world.inverted_safe()@obj.matrix_world
    top=max(b.head_local.z for b in bones)
    wrists=[rig.data.bones.get('Hand.'+s) for s in ('L','R')]
    if not all(wrists):return 0
    bottom=min(b.head_local.z for b in wrists if b)+pitch*4
    bm=bmesh.new();before=len(obj.data.vertices)
    source_normals=[tuple(n.vector) for n in obj.data.corner_normals] if hasattr(obj.data,'corner_normals') else []
    source_positions=[tuple(obj.data.vertices[l.vertex_index].co) for l in obj.data.loops]
    corner_tag='_hallway_source_normal_corner'
    while obj.data.attributes.get(corner_tag):corner_tag+='_'
    try:
        bm.from_mesh(obj.data);flag=bm.verts.layers.int.get(FILL_ATTRIBUTE)
        if not flag:return 0
        normal_ids=bm.loops.layers.int.new(corner_tag)
        bm.faces.ensure_lookup_table()
        for face,polygon in zip(bm.faces,obj.data.polygons):
            for loop,index in zip(face.loops,polygon.loop_indices):
                assert tuple(loop.vert.co)==source_positions[index], 'Corner correspondence changed before fill refinement'
                loop[normal_ids]=index+1
        baseline_defects=sum(not e.is_manifold for e in bm.edges)
        for _ in range(5):
            # Subdivision of long, nearly degenerate cap fans can reconnect
            # an existing diagonal during triangulation. Commit a refinement
            # pass only if it preserves the original manifold topology.
            trial=bm.copy();trial_flag=trial.verts.layers.int[FILL_ATTRIBUTE]
            cap_faces={f for f in trial.faces if any(v[trial_flag] for v in f.verts)}
            edges=[e for e in trial.edges if len(e.link_faces)==2 and all(f in cap_faces for f in e.link_faces)
                   and (transform.to_3x3()@(e.verts[1].co-e.verts[0].co)).length>pitch
                   and bottom<(transform@((e.verts[0].co+e.verts[1].co)*.5)).z<top]
            if not edges:
                trial.free();break
            previous={tuple(v.co) for v in trial.verts}
            bmesh.ops.subdivide_edges(trial,edges=edges,cuts=1,use_grid_fill=True)
            for v in trial.verts:
                if tuple(v.co) not in previous:v[trial_flag]=1
            bmesh.ops.triangulate(trial,faces=[f for f in trial.faces if len(f.verts)>3 and any(v[trial_flag] for v in f.verts)])
            if sum(not e.is_manifold for e in trial.edges)>baseline_defects:
                trial.free();break
            bm.free();bm=trial;flag=trial_flag
        for f in bm.faces:
            if any(v[flag] for v in f.verts):f.smooth=True
        bm.normal_update();bm.to_mesh(obj.data)
    finally:bm.free()
    obj.data.update()
    ids=obj.data.attributes.get(corner_tag)
    if ids and source_normals and hasattr(obj.data,'normals_split_custom_set'):
        normals=[]
        for loop in obj.data.loops:
            index=ids.data[loop.index].value-1
            normals.append(source_normals[index] if 0<=index<len(source_normals) and tuple(obj.data.vertices[loop.vertex_index].co)==source_positions[index] else (0.,0.,0.))
        obj.data.normals_split_custom_set(normals)
    if ids:obj.data.attributes.remove(ids)
    flag=obj.data.attributes[FILL_ATTRIBUTE]
    for v in obj.data.vertices:
        if flag.data[v.index].value:assign(obj,[v.index],normalized(weights(obj,v.index)))
    return len(obj.data.vertices)-before


def round_generated_arm_fills(rig,meshes):
    reports=[]
    for obj in meshes:
        fill=obj.data.attributes.get(FILL_ATTRIBUTE);bank=obj.data.attributes.get('hallway_arm_bank')
        if not fill or not bank or obj.data.shape_keys or obj.get('hallway_arm_fill_revision')==1:continue
        fixed_source={tuple(v.co) for v in obj.data.vertices if not fill.data[v.index].value}
        added=_refine_fill_faces(obj,rig)
        fill=obj.data.attributes[FILL_ATTRIBUTE];bank=obj.data.attributes['hallway_arm_bank']
        transform=rig.matrix_world.inverted_safe()@obj.matrix_world;inverse=transform.inverted_safe()
        points=np.array([transform@v.co for v in obj.data.vertices]);original=points.copy()
        synthetic=np.array([d.value>0 for d in fill.data]);banks=np.array([d.value for d in bank.data])
        if not synthetic.any():continue
        # Intrinsic distance from the fixed rim makes the displacement and its
        # first derivative vanish at the surviving original surface.
        graph=[[] for _ in points];distance=np.full(len(points),np.inf);queue=[]
        for edge in obj.data.edges:
            a,b=edge.vertices
            if not (synthetic[a] or synthetic[b]):continue
            length=float(np.linalg.norm(points[a]-points[b]))
            if synthetic[a] and synthetic[b]:graph[a].append((b,length));graph[b].append((a,length))
            else:
                n=a if synthetic[a] else b
                if length<distance[n]:distance[n]=length;heapq.heappush(queue,(length,n))
        while queue:
            d,a=heapq.heappop(queue)
            if d>distance[a]:continue
            for b,length in graph[a]:
                candidate=d+length
                if candidate<distance[b]:distance[b]=candidate;heapq.heappush(queue,(candidate,b))
        profiles=[];changed=set();reweighted=0
        for side,sign in [('L',1),('R',-1)]:
            upper=rig.data.bones.get('UpperArm.'+side);lower=rig.data.bones.get('Forearm.'+side)
            if not upper or not lower:continue
            shoulder=np.array(upper.head_local);elbow=np.array(lower.head_local);wrist=np.array(lower.tail_local)
            if shoulder[2]-elbow[2]<upper.length*.65:continue
            top=shoulder[2]-.20*upper.length;bottom=wrist[2]+.28*lower.length
            levels=np.linspace(bottom,top,25);fitted=[];section_rows=[]
            for z in levels:
                a,b=(shoulder,elbow) if z>=elbow[2] else (elbow,wrist)
                center=a+(b-a)*np.clip((z-a[2])/(b[2]-a[2]),0,1)
                sample=np.flatnonzero((~synthetic)&(banks==sign)&(np.abs(points[:,2]-z)<upper.length*.04)&
                            ((points[:,0]-center[0])*sign>upper.length*.04)&
                            ((points[:,0]-center[0])*sign<upper.length*.8)&
                            (np.abs(points[:,1]-center[1])<upper.length*.7))
                band=points[sample,:2]
                if len(band)<32:continue
                circle=_circle(band)
                if circle is None:continue
                c,r=circle
                if not upper.length*.12<r<upper.length*.65 or np.linalg.norm(c-center[:2])>upper.length*.4:continue
                fitted.append((z,float(c[0]),float(c[1]),r))
                row={}
                for index in sample:
                    for name,value in weights(obj,int(index)).items():row[name]=row.get(name,0.)+value/len(sample)
                section_rows.append(row)
            if len(fitted)<8:continue
            profile=np.asarray(fitted)
            for _ in range(3):profile[1:-1,1:]=(profile[:-2,1:]+2*profile[1:-1,1:]+profile[2:,1:])*.25
            ids=np.flatnonzero(synthetic&((banks==sign)|(banks==0))&(points[:,2]>profile[0,0])&(points[:,2]<profile[-1,0]))
            z=points[ids,2];cx=np.interp(z,profile[:,0],profile[:,1]);cy=np.interp(z,profile[:,0],profile[:,2]);r=np.interp(z,profile[:,0],profile[:,3])
            q=(points[ids,1]-cy)/r
            medial=cx*sign-np.sqrt(np.maximum(0.,r*r-(points[ids,1]-cy)**2))
            # The companion body fill retreats with the rounded arm, leaving a
            # small gap. Its original boundary and the bag stay fixed as well.
            medial-=np.where(banks[ids]==0,upper.length*.008,0.)
            displacement=medial-points[ids,0]*sign
            rim=np.clip(distance[ids]/(r*.45),0,1);rim=rim*rim*(3-2*rim)
            end=np.minimum((z-profile[0,0])/(upper.length*.15),(profile[-1,0]-z)/(upper.length*.15));end=np.clip(end,0,1);end=end*end*(3-2*end)
            valid=(abs(q)<.95)&((points[ids,0]*sign-cx*sign)<r*.5)&(abs(points[ids,0]*sign-cx*sign)<r*1.3)&(displacement<0)
            lateral=np.clip((.98-np.abs(q))/.30,0,1);lateral=lateral*lateral*(3-2*lateral)
            # A missing contact surface is underdetermined. Reconstruct a
            # bounded part of its curvature, rather than inflating the cap
            # to an entire inferred cross-section in one pass.
            movement=np.clip(displacement,-r*.25,0)*rim*end*lateral
            for j in np.flatnonzero(valid&(abs(movement)>upper.length*1e-6)):
                index=int(ids[j]);points[index,0]+=sign*movement[j];changed.add(index)
            names=sorted({name for row in section_rows for name in row})
            field=np.array([np.interp(z,profile[:,0],[row.get(name,0.) for row in section_rows]) for name in names]).T
            for j in np.flatnonzero(valid&(banks[ids]==sign)):
                index=int(ids[j]);blend=float(.35*rim[j]*end[j]*lateral[j]);old=weights(obj,index)
                row={name:(1-blend)*old.get(name,0.)+blend*field[j,k] for k,name in enumerate(names)}
                for name,value in old.items():
                    if name not in row:row[name]=(1-blend)*value
                assign(obj,[index],normalized(row));reweighted+=1
            profiles.append(dict(side=side,sections=len(profile),minimum_radius=float(profile[:,3].min()),maximum_radius=float(profile[:,3].max())))
        for index in changed:
            obj.data.vertices[index].co=inverse@Vector(points[index])
        obj.data.update()
        # Retire inherited flat cap normals while retaining original surface
        # corner normals. Zero vectors request Blender's automatic normals.
        if hasattr(obj.data,'corner_normals') and hasattr(obj.data,'normals_split_custom_set'):
            normals=[tuple(n.vector) for n in obj.data.corner_normals]
            for face in obj.data.polygons:
                if any(synthetic[i] for i in face.vertices):
                    for loop in face.loop_indices:normals[loop]=(0.,0.,0.)
            obj.data.normals_split_custom_set(normals)
        assert fixed_source <= {tuple(v.co) for v in obj.data.vertices}, 'Generated-fill repair moved an original surface vertex'
        obj['hallway_arm_fill_revision']=1
        assert np.array_equal(points[~synthetic],original[~synthetic]),'Original arm surface moved'
        reports.append(dict(object=obj.name,generated_vertices_changed=len(changed),generated_vertices_added=added,generated_vertices_reweighted=reweighted,source_vertices_unchanged=True,maximum_fill_displacement=float(np.linalg.norm(points-original,axis=1).max()),profiles=profiles))
    return reports
