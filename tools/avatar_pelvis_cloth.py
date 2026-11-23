"""Body-supported short pelvis cloth, with a small retained spring response.

Classify hems on connected strips below the waistband, so a short shirt front
can coexist with a long coat in one mesh/material. Sample the generated body,
not imported weights. No mesh coordinates or spring/collider definitions change.
"""
import heapq
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
from avatar_apparel_weights import weights, assign
from avatar_virtual_topology import connected_graph
from avatar_surface_attachments import normalized


def short_support(mesh, points, ids, hip_z, knee_z, height, center=(0.,0.)):
    """Return a geodesically bounded blend for short connected hem pieces."""
    waist=hip_z-height*.025
    graph,_,_=connected_graph(mesh,points,ids,height*1e-5)
    adjacency={i:[] for i in ids}
    for e in graph.edges:
        a,b=e.vertices;adjacency[a].append(b);adjacency[b].append(a)
    remaining={i for i in ids if points[i,2]<waist};short=set();components=[]
    angle=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])
    sectors=np.floor((angle+np.pi/16)%(2*np.pi)/(np.pi/8)).astype(int)
    cutoff=knee_z+(hip_z-knee_z)*.20
    while remaining:
        seed=remaining.pop();group=[seed]
        for i in group:
            for j in adjacency[i]:
                if j in remaining:remaining.remove(j);group.append(j)
        # A connected coat can wrap around a short shirt panel. Inspect
        # connected angular strips as well as the complete hanging piece.
        pieces=[group]
        if np.quantile(points[group,2],.01)<=cutoff:
            pieces=[];remaining_local=set(group)
            while remaining_local:
                seed=remaining_local.pop();piece=[seed]
                for i in piece:
                    for j in adjacency[i]:
                        if j in remaining_local and sectors[j]==sectors[seed]:
                            remaining_local.remove(j);piece.append(j)
                pieces.append(piece)
        for piece in pieces:
            p=points[piece]
            if len(piece)<24 or np.ptp(p[:,2])<height*.015:continue
            hem=float(np.quantile(p[:,2],.01))
            if hem>cutoff:
                short.update(piece);components.append(dict(vertices=len(piece),hem=hem))
    if not short:return {},components
    distance={i:0. for i in short};queue=[(0.,i) for i in short];heapq.heapify(queue)
    reach=height*.065
    while queue:
        d,i=heapq.heappop(queue)
        if d!=distance[i]:continue
        for j in adjacency[i]:
            if j not in short and points[j,2]<waist:continue
            step=d+float(np.linalg.norm(points[i]-points[j]))
            if step<reach and step<distance.get(j,float('inf')):
                distance[j]=step;heapq.heappush(queue,(step,j))
    ids=sorted(ids);lookup={i:j for j,i in enumerate(ids)}
    field=np.array([float((1-distance[i]/reach)**2*(1+2*distance[i]/reach)) if i in distance else 0. for i in ids])
    edges=np.array([(lookup[e.vertices[0]],lookup[e.vertices[1]]) for e in graph.edges],dtype=int).reshape(-1,2)
    target=np.r_[np.arange(len(ids)),edges[:,0],edges[:,1]];source=np.r_[np.arange(len(ids)),edges[:,1],edges[:,0]]
    degrees=np.bincount(target,minlength=len(ids))
    for _ in range(20):field=.5*field+.5*np.bincount(target,weights=field[source],minlength=len(ids))/degrees
    from avatar_weight_seams import coincident_groups
    for group in coincident_groups(points[ids],1e-6):field[group]=field[group].mean()
    return {i:float(value) for i,value in zip(ids,field) if value>1e-4},components


class BodySupport:
    def __init__(self,rig,meshes,excluded,height):
        self.rig=rig;self.height=height
        self.allowed={n for n in ('Hips','Spine','UpperSpine','Chest','Thigh.L','Thigh.R') if n in rig.data.bones}
        self.points=[];self.rows=[];self.faces=[]
        hip=rig.data.bones['Hips'].head_local.z
        for obj in meshes:
            matrix=rig.matrix_world.inverted_safe()@obj.matrix_world
            points=[matrix@v.co for v in obj.data.vertices];offset=len(self.points)
            rows=[weights(obj,v.index) for v in obj.data.vertices]
            valid=[i not in excluded.get(obj,set()) and hip-height*.45<p.z<hip+height*.12 and
                   sum(w for n,w in row.items() if n in self.allowed)>.95 and
                   not any(n.startswith('Secondary_') and w>1e-6 for n,w in row.items())
                   for i,(p,row) in enumerate(zip(points,rows))]
            self.points.extend(points);self.rows.extend(normalized({n:w for n,w in row.items() if n in self.allowed}) for row in rows)
            obj.data.calc_loop_triangles()
            self.faces.extend(tuple(offset+i for i in t.vertices) for t in obj.data.loop_triangles if all(valid[i] for i in t.vertices))
        self.trees=[]
        center=rig.data.bones['Hips'].head_local.x
        for sign in (1,-1):
            faces=[f for f in self.faces if sum(self.points[i].x-center for i in f)*sign>=0]
            self.trees.append((BVHTree.FromPolygons(self.points,faces,all_triangles=True) if faces else None,faces))
        self.sampled=0;self.fallback=0

    def sample(self,point):
        bones=self.rig.data.bones;center=bones['Hips'].head_local
        halfspan=abs(bones['Thigh.L'].head_local.x-bones['Thigh.R'].head_local.x)*.5
        u=np.clip((point.x-center.x)/max(halfspan*.65,1e-8),-1,1)
        left=.5+.5*u*(1.5-.5*u*u)
        result={}
        for side,factor,(tree,faces) in zip(('L','R'),(left,1-left),self.trees):
            if factor<1e-6:continue
            bone=bones['Thigh.'+side];d=bone.tail_local-bone.head_local
            t=np.clip((point.z-bone.head_local.z)/min(d.z,-1e-8),0,1)
            toward=bone.head_local+d*t-point
            hit=tree.ray_cast(point,toward.normalized(),self.height*.12) if tree and toward.length>1e-8 else (None,)*4
            if hit[0] is None and tree:hit=tree.find_nearest(point,self.height*.10)
            row=None
            if hit[0] is not None:
                p,_,index,_=hit;tri=faces[index];a,b,c=(self.points[i] for i in tri)
                if (b-a).cross(c-a).length_squared>1e-20:
                    bary=barycentric_transform(p,a,b,c,Vector((1,0,0)),Vector((0,1,0)),Vector((0,0,1)))
                    row={}
                    for i,f in zip(tri,bary):
                        for n,w in self.rows[i].items():row[n]=row.get(n,0.)+max(0.,f)*w
                    row=normalized(row)
            if not row:
                # Anatomical fallback is bounded below the hip, never propagated
                # into the spine. It is used only when no underlying surface exists.
                q=float(np.clip((bone.head_local.z-point.z)/(bone.length*.4),0,1));q=q*q*(3-2*q)
                row={'Hips':1-q,'Thigh.'+side:q};self.fallback+=1
            else:self.sampled+=1
            opposite='Thigh.'+('R' if side=='L' else 'L')
            row['Hips']=row.get('Hips',0.)+row.pop(opposite,0.)
            for n,w in row.items():result[n]=result.get(n,0.)+factor*w
        return normalized(result)


def bind_short_pelvis_cloth(rig,meshes,plan):
    regions=[r for r in plan['regions'] if r['role'] in {'garment','skirt'} and not r.get('garment_strap') and not r.get('panel_mount')]
    if not regions:return dict(vertices=0,components=[])
    if not all(n in rig.data.bones for n in ('Hips','Thigh.L','Thigh.R','Shin.L','Shin.R')):return dict(vertices=0,components=[])
    height=plan['height'];hip=rig.data.bones['Hips'].head_local.z;knee=max(rig.data.bones['Shin.'+s].head_local.z for s in ('L','R'))
    excluded={obj:{i for r in plan['regions'] if r['obj']==obj for i in r['ids']} for obj in meshes}
    supports=[];components=[]
    for obj in meshes:
        ids={i for r in regions if r['obj']==obj for i in r['ids']}
        if not ids:continue
        points=plan['points'][obj.name]
        support,parts=short_support(obj.data,points,ids,hip,knee,height,rig.data.bones['Hips'].head_local[:2])
        if support:supports.append((obj,points,support));components.extend(dict(object=obj.name,**p) for p in parts)
    if not supports:return dict(vertices=0,components=[])
    body=BodySupport(rig,meshes,excluded,height);changed=0;maximum=0.
    for obj,points,support in supports:
        for i,factor in support.items():
            old=weights(obj,i);spring={n:w for n,w in old.items() if n.startswith('Secondary_')}
            if not spring:continue
            mass=sum(spring.values());p=Vector(points[i]);base=body.sample(p)
            # Keep the hem lively, but never let physics carry most of a short
            # flap. A compact body distribution reserves room for its springs.
            target=min(mass,.12)
            base=normalized(base,limit=3)
            free=normalized(spring,limit=4-len(base))
            target_row={n:w*(1-target) for n,w in base.items()}
            for n,w in free.items():target_row[n]=target*w
            row={n:old.get(n,0.)*(1-factor)+target_row.get(n,0.)*factor for n in old.keys()|target_row.keys()}
            row=normalized(row)
            assign(obj,[i],row);changed+=1
            if factor>.999:maximum=max(maximum,sum(w for n,w in row.items() if n.startswith('Secondary_')))
    return dict(vertices=changed,components=components,body_samples=body.sampled,fallback_samples=body.fallback,maximum_short_panel_spring_weight=maximum,geometry_changed=False)
