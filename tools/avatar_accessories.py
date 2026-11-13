"""Mesh-only torso attachments: straps and branched hanging accessories.

Rest geometry and the freshly inferred humanoid are the only inputs. No material,
source bone, weight, character name or reflection pairing is used for admission.
"""
import heapq
import math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
from avatar_virtual_topology import connected_graph, protected_extremities


def segment_distance(points,a,b):
    d=b-a;t=np.clip((points-a)@d/max(float(d@d),1e-16),0,1)
    return np.linalg.norm(points-a-t[:,None]*d,axis=1)


def plan_accessories(rig,obj,points,excluded,height,prefix,start_index=0):
    protected=protected_extremities(rig,points,height)
    allowed=set(range(len(points)))-set(excluded)-set(np.flatnonzero(protected))
    graph,groups,links=connected_graph(obj.data,points,allowed,height*.0001)
    hips=rig.data.bones['Hips'];neck=rig.data.bones['Neck']
    torso=[rig.data.bones[n] for n in ('Hips','Spine','UpperSpine','Chest') if n in rig.data.bones]
    admitted=[]
    for ids in groups:
        p=points[ids];span=np.ptp(p,axis=0)
        if len(ids)<12 or max(span)<height*.045 or max(span)>height*.38:continue
        if span[0]>height*.22 or span[1]>height*.20:continue
        if p[:,2].max()<hips.head_local.z-height*.015 or p[:,2].max()>neck.head_local.z:continue
        # Reject sleeves/limb surfaces following a mapped anatomical axis.
        limb=False
        for side in ('L','R'):
            for n in ('UpperArm.','Forearm.','Thigh.','Shin.'):
                b=rig.data.bones.get(n+side)
                if b is None:continue
                a=np.array(b.head_local);d=np.array(b.tail_local)-a
                radius=max(b.length*.35,height*.045) if n in ('UpperArm.','Forearm.') else b.length*.20
                if np.mean(segment_distance(p,a,a+d)<radius)>.65:limb=True;break
            if limb:break
        if limb:continue
        distances=np.stack([segment_distance(p,np.array(b.head_local),np.array(b.tail_local)) for b in torso])
        bi,vi=np.unravel_index(np.argmin(distances),distances.shape)
        if distances[bi,vi]>height*.14:continue
        if np.ptp(p[:,2])<height*.045:continue
        admitted.append(dict(ids=ids,anchor=ids[int(vi)],parent=torso[int(bi)].name))
    # Detached seams and small decorations belong to one nearest admitted
    # surface. They cannot recruit a humanoid-sized component or a hand.
    owner={i:j for j,r in enumerate(admitted) for i in r['ids']}
    if not owner:return [],[],dict(components=0,virtual_links=len(links))
    obj.data.calc_loop_triangles()
    faces=[tuple(t.vertices) for t in obj.data.loop_triangles
           if all(i in owner for i in t.vertices) and len({owner[i] for i in t.vertices})==1]
    surface=BVHTree.FromPolygons(points.tolist(),faces,all_triangles=True) if faces else None
    attachments=[]
    for ids in groups:
        if any(i in owner for i in ids) or surface is None:continue
        p=points[ids]
        if max(np.ptp(p,axis=0))>height*.06:continue
        matches=[surface.find_nearest(Vector(points[i])) for i in ids]
        votes={}
        for location,_,face,d in matches:
            if location is not None and d<height*.008:
                j=owner[faces[face][0]];votes[j]=votes.get(j,0)+1
        if not votes:continue
        j=max(votes,key=votes.get)
        close_coverage=votes[j]>=len(ids)*.8
        same_owner=[float(d) for location,_,face,d in matches
                    if location is not None and owner[faces[face][0]]==j]
        # A rigid pendant can hang just beyond the last strap surface. Require
        # a touching attachment and one consistent nearby owner for its whole
        # component, rather than dropping its far end or seeking a thigh.
        attached_end=(len(same_owner)>=len(ids)*.9 and min(same_owner)<height*.002
                      and max(same_owner)<height*.025)
        if not (close_coverage or attached_end) or len(ids)>max(16,len(admitted[j]['ids'])*.6):continue
        pairs=admitted[j].setdefault('satellites',{})
        for i,(location,_,face,_) in zip(ids,matches):
            if location is None:continue
            tri=faces[face]
            if owner[tri[0]]!=j:continue
            bary=barycentric_transform(location,*(Vector(points[v]) for v in tri),
                                       Vector((1,0,0)),Vector((0,1,0)),Vector((0,0,1)))
            weights=np.maximum(np.asarray(bary),0);weights/=max(float(weights.sum()),1e-12)
            pairs[i]=list(zip(tri,weights))
        attachments.append(dict(vertices=len(ids),owner=j))
    chains=[];regions=[]
    neighbors={i:[] for i in allowed}
    for e in graph.edges:
        a,b=e.vertices;w=float(np.linalg.norm(points[a]-points[b]));neighbors[a].append((b,w));neighbors[b].append((a,w))
    for index,region in enumerate(admitted):
        ids=region['ids'];valid=set(ids)
        complete=sorted(valid|set(region.get('satellites',{})))
        complete_set=set(complete)
        geometry=points[complete];span=np.ptp(geometry,axis=0)
        if span[2]>max(span[:2])*3 and span[2]>height*.08:
            # Narrow hanging strips have a single gravity-oriented centerline.
            # Geodesic rings on a two-sided/disconnected ribbon can turn back
            # upward. Height sections cannot reverse or create phantom forks.
            from avatar_hair_guides import branch_line
            from avatar_springs import _vertical_weights
            top=float(geometry[:,2].max());bottom=float(geometry[:,2].min())
            root=np.median(geometry[geometry[:,2]>top-height*.008],axis=0);root[2]=top
            free_top=top-min(height*.012,span[2]*.08)
            edges=np.array([[points[a],points[b]] for e in graph.edges
                            for a,b in [e.vertices] if a in complete_set and b in complete_set])
            line=branch_line(dict(edges=edges,points=geometry,anchor=Vector((root[0],root[1],free_top))),4)
            line=[Vector(root)]+line
            names=[f'{prefix}Accessory_{index+start_index:02d}_Strand_{i:02d}' for i in range(len(line))]
            chains.append(dict(role='accessory',names=names,line=line,parent=region['parent'],physics_start=1))
            values={i:_vertical_weights(names[1:],region['parent'],(free_top-points[i,2])/(free_top-bottom),smooth_transition=True) for i in complete}
            regions.append(dict(obj=obj,role='accessory',ids=complete,values=values))
            region['layout']='downward_strand'
            continue
        dist={region['anchor']:0.};queue=[(0.,region['anchor'])]
        while queue:
            d,i=heapq.heappop(queue)
            if d!=dist[i]:continue
            for j,w in neighbors[i]:
                nd=d+w
                if j in valid and nd<dist.get(j,math.inf):dist[j]=nd;heapq.heappush(queue,(nd,j))
        width=max(height*.025,max(dist.values())/8)
        bins={i:int(dist[i]/width) for i in ids};node_for={};nodes=[]
        for level in sorted(set(bins.values())):
            band={i for i in ids if bins[i]==level}
            while band:
                root=min(band);found={root};pending=[root];band.remove(root)
                while pending:
                    for j,_ in neighbors[pending.pop()]:
                        if j in band:band.remove(j);found.add(j);pending.append(j)
                node=len(nodes);node_for.update({i:node for i in found})
                nodes.append(dict(ids=sorted(found),level=level,center=np.mean(points[sorted(found)],axis=0)))
        # Parent selection follows the measured geodesic graph. Every edge is a
        # separate two-joint VRM spring, so forks never share simulated joints.
        centers={};bone_for={};values={};root_nodes=[]
        for ni,node in enumerate(nodes):
            candidates={node_for[j] for i in node['ids'] for j,_ in neighbors[i]
                        if j in node_for and nodes[node_for[j]]['level']<node['level']}
            if not candidates:
                if node['level']==0:
                    root_nodes.append(ni);centers[ni]=node['center'];bone_for[ni]=region['parent']
                    values.update({i:{region['parent']:1.} for i in node['ids']});continue
                candidates=set(range(ni))
            parent=min(candidates,key=lambda j:np.linalg.norm(nodes[j]['center']-node['center']))
            a=centers[parent];b=node['center'];parent_name=bone_for[parent]
            if np.linalg.norm(a-b)<height*.001:
                centers[ni]=a;bone_for[ni]=parent_name
                values.update({i:{parent_name:1.} for i in node['ids']});continue
            label=f'{prefix}Accessory_{index+start_index:02d}_{ni:02d}'
            names=[label+'_00',label+'_01']
            chains.append(dict(role='accessory',names=names,line=[Vector(a),Vector(b)],parent=parent_name,physics_start=0))
            centers[ni]=b;bone_for[ni]=names[0]
            for i in node['ids']:
                blend=float(np.clip((dist[i]-node['level']*width)/width,0,1))*.5+.5
                values[i]={names[0]:blend,parent_name:1-blend}
        for i,donors in region.get('satellites',{}).items():
            blended={}
            for v,factor in donors:
                for name,weight in values[v].items():blended[name]=blended.get(name,0.)+float(factor)*weight
            values[i]=blended
        regions.append(dict(obj=obj,role='accessory',ids=sorted(values),values=values))
    return chains,regions,dict(components=len(admitted),virtual_links=len(links),satellites=attachments,
                              parents=[r['parent'] for r in admitted],layouts=[r.get('layout','branched') for r in admitted])
