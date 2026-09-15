"""Separate torso and arm ownership along concave surface creases.

Fresh heat supplies intra-region bone ratios. A curvature-weighted surface cut
finds each arm attachment; only a narrow geodesic band shares arm influence.
Analysis welds a temporary surface. Render geometry, UVs and topology stay intact.
Requires SciPy, also used by the UniMate evaluation tools.
"""
import numpy as np
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order, dijkstra
from avatar_apparel_weights import weights, assign, smooth, classify_regions

TORSO=('Root','Spine','Chest')


def crease_field(meshes,rig):
    from avatar_voxel_seams import skin_faces
    parts=[(o,skin_faces(o)) for o in meshes if o.get('binding_role')=='body']
    parts=[(o,f) for o,f in parts if f]
    if not parts:
        carrier=classify_regions(meshes,rig,{})[0];o=carrier['object'];ids=set(carrier['indices'])
        parts=[(o,[f for f in o.data.polygons if all(i in ids for i in f.vertices)])]
    bm=bmesh.new()
    try:
        for obj,faces in parts:
            ids=sorted({i for f in faces for i in f.vertices});lookup={i:bm.verts.new(obj.data.vertices[i].co) for i in ids}
            for f in faces:bm.faces.new([lookup[i] for i in f.vertices])
        height=max(v.co.z for v in bm.verts)-min(v.co.z for v in bm.verts)
        bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=height*1e-6)
        bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bmesh.ops.triangulate(bm,faces=list(bm.faces))
        bm.normal_update();bm.verts.ensure_lookup_table();bm.verts.index_update()
        points=np.array([v.co[:] for v in bm.verts]);triangles=[tuple(v.index for v in f.verts) for f in bm.faces]
        edge_data=[(e.verts[0].index,e.verts[1].index,e.calc_face_angle_signed(0.) if e.is_manifold else 0.) for e in bm.edges]
    finally:bm.free()
    edges=np.array(edge_data);ij=edges[:,:2].astype(int);n=len(points)
    length=np.linalg.norm(points[ij[:,0]]-points[ij[:,1]],axis=1)
    rows=np.r_[ij[:,0],ij[:,1]];cols=np.r_[ij[:,1],ij[:,0]]
    geometry=coo_matrix((np.tile(length,2),(rows,cols)),shape=(n,n)).tocsr()
    # Concave edges are cheap places to separate; flat/convex surface crossings
    # are expensive. Physical edge length avoids using raw tessellation counts.
    cost=np.maximum(1,1000*length/np.median(length)*(.01+np.exp(-6*np.maximum(0,-edges[:,2])))).astype(np.int64)
    signed={};audits=[];cx=rig.data.bones['Neck'].head_local.x
    for side,sign in [('L',1),('R',-1)]:
        bone=rig.data.bones['UpperArm.'+side];s=np.array(bone.head_local);delta=np.array(bone.tail_local)-s;radius=bone.length*.22
        along=(points-s)@delta/np.dot(delta,delta)
        arm_distance=np.linalg.norm(points-s-np.clip(along,0,1)[:,None]*delta,axis=1)
        arm=(along>.45)&(arm_distance<radius*1.35)
        for prefix in ('Forearm','Hand'):
            b=rig.data.bones.get(prefix+'.'+side)
            if b is None:continue
            a=np.array(b.head_local);d=np.array(b.tail_local)-a;t=(points-a)@d/np.dot(d,d)
            distance=np.linalg.norm(points-a-np.clip(t,0,1)[:,None]*d,axis=1)
            arm |= (t>.1)&(distance<radius*1.35)
        lateral=sign*(points[:,0]-cx);span=abs(s[0]-cx)
        arm &= lateral>span*.98
        body=lateral<span*.7
        arm &= ~body
        ar=np.flatnonzero(arm);bo=np.flatnonzero(body)
        if len(ar)<3 or len(bo)<3:raise ValueError('Body surface lacks reliable arm/torso seeds')
        row=np.r_[rows,np.full(len(ar),n),bo];col=np.r_[cols,ar,np.full(len(bo),n+1)]
        capacity=coo_matrix((np.r_[cost,cost,np.full(len(ar)+len(bo),10000000)],(row,col)),shape=(n+2,n+2)).tocsr()
        flow=maximum_flow(capacity,n,n+1);residual=capacity-flow.flow
        residual.data=(residual.data>0).astype(np.int64);residual.eliminate_zeros()
        reachable=breadth_first_order(residual,n,directed=True,return_predecessors=False)
        label=np.zeros(n,bool);label[reachable[reachable<n]]=True
        crossing=label[ij[:,0]]!=label[ij[:,1]];boundary=np.unique(ij[crossing])
        if not len(boundary):raise ValueError('No connected armpit boundary found')
        distance=dijkstra(geometry,indices=boundary,directed=False,min_only=True)
        distance=np.minimum(distance,bone.length*10)
        signed[side]=np.where(label,distance,-distance)
        audits.append({'side':side,'arm_vertices':int(label.sum()),'crease_boundary_vertices':len(boundary),
                       'concave_cut_edges':int(np.sum(edges[crossing,2]<-.01)),
                       'boundary_min':points[boundary].min(axis=0).tolist(),'boundary_max':points[boundary].max(axis=0).tolist()})
    return points,triangles,signed,audits


def capped_weights(old,caps,rig,point):
    values=dict(old);removed=0.
    for side in ('L','R'):
        names=[n for n in values if n.endswith('.'+side) and n.split('.')[0] in ('Clavicle','UpperArm','Forearm','Hand')]
        mass=sum(values[n] for n in names);allowed=min(mass,caps[side])
        if allowed==mass:continue
        for n in names:values[n]*=allowed/mass
        removed+=mass-allowed
    if removed<1e-8:return old
    torso={n:values.get(n,0.) for n in TORSO if n in rig.data.bones};total=sum(torso.values())
    if total<1e-8:
        def distance(n):
            b=rig.data.bones[n];a=b.head_local;delta=b.tail_local-a
            t=max(0.,min(1.,(point-a).dot(delta)/delta.length_squared))
            return (point-a-delta*t).length
        torso={min(torso,key=distance):1.};total=1.
    for n,w in torso.items():values[n]=values.get(n,0.)+removed*w/total
    # Preserve the caps when pruning; never normalize discarded mass back to arm.
    body={n:w for n,w in values.items() if n in TORSO and w>1e-8}
    other={n:w for n,w in values.items() if n not in TORSO and w>1e-8}
    other=dict(sorted(other.items(),key=lambda x:-x[1])[:3])
    body=dict(sorted(body.items(),key=lambda x:-x[1])[:4-len(other)])
    total=sum(body.values());remainder=1-sum(other.values())
    return {**other,**{n:w/total*remainder for n,w in body.items()}}


def separate_arm_weights(meshes,rig):
    points,triangles,signed,audits=crease_field(meshes,rig)
    tree=cKDTree(points);bvh=BVHTree.FromPolygons([Vector(p) for p in points],triangles,all_triangles=True)
    changed=0;torso_count=0;max_torso_arm=0.;max_crease_arm=0.;crease_count=0
    for obj in meshes:
        # Explicit head/hair/jewelry semantics are outside this correction.
        if obj.get('binding_role') in ('head','neckwear','rigid'):continue
        coords=np.array([v.co[:] for v in obj.data.vertices]);distances,nearest=tree.query(coords)
        for v,distance,index in zip(obj.data.vertices,distances,nearest):
            old=weights(obj,v.index)
            if not any(n.endswith(('.L','.R')) for n in old):continue
            if distance<rig.data.bones['Neck'].length*1e-5:
                sample={side:float(signed[side][index]) for side in ('L','R')}
            else:
                hit,_,face,_=bvh.find_nearest(v.co);ids=triangles[face]
                bary=barycentric_transform(hit,*[Vector(points[i]) for i in ids],Vector((1,0,0)),Vector((0,1,0)),Vector((0,0,1)))
                sample={side:sum(float(signed[side][i])*factor for i,factor in zip(ids,bary)) for side in ('L','R')}
            caps={}
            for side,d in sample.items():
                radius=rig.data.bones['UpperArm.'+side].length*.22
                caps[side]=(.1+.9*smooth(0,radius*.65,d)) if d>=0 else .1*(1-smooth(0,radius*.35,-d))
            values=capped_weights(old,caps,rig,v.co)
            if values!=old:assign(obj,[v.index],values);changed+=1
            mass=sum(w for n,w in values.items() if n.endswith(('.L','.R')))
            if max(caps.values())<1e-8:
                torso_count+=1;max_torso_arm=max(max_torso_arm,mass)
            elif all(d<=0 for d in sample.values()):
                crease_count+=1;max_crease_arm=max(max_crease_arm,mass)
    return {'method':'concavity-weighted surface cut and geodesic crease band','vertices_changed':changed,
            'torso_interior_vertices_checked':torso_count,'maximum_torso_interior_arm_weight':max_torso_arm,
            'crease_torso_vertices_checked':crease_count,'maximum_crease_torso_arm_weight':max_crease_arm,
            'crease_arm_cap':.1,'render_geometry_changed':False,'sides':audits}
