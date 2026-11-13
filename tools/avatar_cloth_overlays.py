"""Bind long surface trim to broader generated cloth; no source-rig inputs."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from avatar_virtual_topology import connected_graph
from avatar_apparel_weights import weights,assign
from avatar_garment_attachments import _barycentric
from avatar_cloth_smoothing import chain_profile, preserve_attachment


def bind_cloth_overlays(rig, meshes, *, refine=True):
    """A covered ribbon inherits its larger supporting panel's skin field.

    Require an elongated component, broader/larger donor, close coverage along
    its whole length and interior support. Independent hanging straps fail the
    coverage gate. Snapshot all donors before writing; no cyclic transfers.
    """
    points={o.name:np.asarray([rig.matrix_world.inverted()@o.matrix_world@v.co for v in o.data.vertices],dtype=float) for o in meshes}
    if not points:return dict(components=[],vertices=0)
    height=max(p[:,2].max() for p in points.values())-min(p[:,2].min() for p in points.values())
    if height<=1e-8:return dict(components=[],vertices=0)
    def mass(row):return sum(w for n,w in row.items() if 'Secondary_Skirt_' in n or 'Accessory_Jacket_' in n)
    records=[]
    for obj in meshes:
        p=points[obj.name];obj.data.calc_loop_triangles()
        _,groups,_=connected_graph(obj.data,p,range(len(p)),height*.0001)
        owner={i:j for j,g in enumerate(groups) for i in g};faces=[[] for _ in groups]
        for t in obj.data.loop_triangles:
            ids=tuple(t.vertices);j=owner[ids[0]]
            if all(owner[i]==j for i in ids):faces[j].append(ids)
        for ids,tri in zip(groups,faces):
            if len(ids)<8 or not tri:continue
            q=p[ids];span=np.sort(np.ptp(q,axis=0));values={i:weights(obj,i) for i in ids}
            if sum(mass(row)>.25 for row in values.values())<len(ids)*.5:continue
            if span[-1]<height*.20:continue
            area=float(sum(np.linalg.norm(np.cross(p[b]-p[a],p[c]-p[a]))*.5 for a,b,c in tri))
            records.append(dict(obj=obj,ids=ids,p=p,q=q,span=span,tri=tri,area=area,values=values))
    report=[]
    for r in records:
        if r['span'][-1]<3*r['span'][-2]:continue
        possible=[d for d in records if d is not r and d['area']>r['area']*2
                  and d['span'][-2]>r['span'][-2]*1.5 and d['span'][-1]>=r['span'][-1]*.8]
        best=None
        for d in possible:
            if 'tree' not in d:d['tree']=BVHTree.FromPolygons(d['p'],d['tri'],all_triangles=True)
            samples=[]
            for point in r['q']:
                hit,normal,face,distance=d['tree'].find_nearest(Vector(point))
                if face is None:break
                ids=d['tri'][face];factors=_barycentric(np.asarray(hit),*d['p'][list(ids)])
                samples.append((ids,factors,float(distance)))
            if len(samples)!=len(r['ids']):continue
            distances=[s[2] for s in samples]
            if max(distances)>height*.04 or np.median(distances)>height*.01:continue
            if sum(min(s[1])>1e-4 for s in samples)<len(samples)*.5:continue
            key=float(np.mean(distances))
            if best is None or key<best[0]:best=(key,d,samples)
        if best is None:continue
        _,d,samples=best
        if r['obj'] is not d['obj']:continue  # Refinement owns one mesh transaction.
        # Trim stitched over a panel must share its local bend carrier too.
        # Independent neighboring sector chains can shear the two surfaces even
        # with identical nearest-vertex weights. Choose the existing dominant
        # cloth chain from the support footprint, never a model-specific name.
        support={j for ids,_,_ in samples for j in ids}
        chains={n.rsplit('_',1)[0] for j in support for n in d['values'][j]
                if 'Secondary_Skirt_' in n or 'Accessory_Jacket_' in n}
        if not chains:continue
        chain=max(sorted(chains),key=lambda c:sum(w for j in support for n,w in d['values'][j].items() if n.rsplit('_',1)[0]==c))
        if 'Secondary_Skirt_' not in chain:continue
        # Only a well-supported carrier may replace competing cloth sectors.
        if any(mass(d['values'][j])>.01 and not any(n.rsplit('_',1)[0]==chain for n in d['values'][j]) for j in support):continue
        adjacency={i:set() for i in d['ids']}
        for tri in d['tri']:
            for a in tri:
                adjacency[a].update(j for j in tri if j!=a)
        # UV/normal seam duplicates are one cloth station.
        from avatar_weight_seams import coincident_groups
        seam_groups=[[d['ids'][k] for k in g] for g in coincident_groups(d['q'])]
        for g in seam_groups:
            if len(g)>1:
                for i in g:adjacency[i].update(j for j in g if j!=i)
        distance={i:0 for i in support}
        for step in (1,2):
            for i in tuple(distance):
                if distance[i]!=step-1:continue
                for j in adjacency[i]:distance.setdefault(j,step)
        for g in seam_groups:
            level=min((distance.get(i,3) for i in g),default=3)
            if level<3:
                for i in g:distance[i]=level
        carrier_bones=sorted((b for b in rig.data.bones if b.use_deform and b.name.rsplit('_',1)[0]==chain and b.name.rsplit('_',1)[1].isdigit()),key=lambda b:int(b.name.rsplit('_',1)[1]))
        if len(carrier_bones)<2:continue
        direction=carrier_bones[-1].head_local-carrier_bones[0].head_local
        axis=Vector((0,0,0));component=int(np.argmax(np.abs(direction)));axis[component]=direction[component]
        if axis.length<=height*1e-6:continue
        axis.normalize()
        stations=np.array([b.head_local.dot(axis) for b in carrier_bones])
        if np.any(np.diff(stations)<=height*1e-6):continue
        changed_support={}
        for j,level in distance.items():
            old=d['values'][j];cloth_mass=mass(old)
            selected={n:w for n,w in old.items() if n.rsplit('_',1)[0]==chain}
            total=sum(selected.values())
            if total<=1e-10:continue
            target={n:w for n,w in old.items() if not ('Secondary_Skirt_' in n or 'Accessory_Jacket_' in n)}
            coordinate=float(Vector(d['p'][j]).dot(axis))
            target=preserve_attachment(old,chain_profile([b.name for b in carrier_bones],stations,coordinate,4-len(target)))
            amount=(1.,.5,0.)[level]
            if not amount:continue
            value={n:(1-amount)*old.get(n,0)+amount*target.get(n,0) for n in set(old)|set(target)}
            changed_support[j]=preserve_attachment(old,value)
        if not all(j in changed_support or mass(d['values'][j])<=.01 for j in support):continue
        changes=[];correspondence=[]
        for i,(ids,factors,distance) in zip(r['ids'],samples):
            # Use the same smooth carrier as the support, retaining this
            # vertex's anatomical weights and remaining influence budget.
            coordinate=float(Vector(r['p'][i]).dot(axis))
            original=weights(r['obj'],i)
            budget=4-sum('Secondary_Skirt_' not in n for n in original)
            row=preserve_attachment(original,chain_profile([b.name for b in carrier_bones],stations,coordinate,budget))
            if len(row)>4:break
            total=sum(row.values())
            if total<=1e-10:break
            # Preserve original anatomical weights exactly; no full-row normalization.
            changes.append((i,row));correspondence.append(dict(vertex=i,triangle=list(ids),factors=list(map(float,factors))))
        if len(changes)!=len(r['ids']):continue
        for j,row in changed_support.items():assign(d['obj'],[j],row)
        for i,row in changes:assign(r['obj'],[i],row)
        report.append(dict(object=r['obj'].name,ids=r['ids'],donor_object=d['obj'].name,donor_ids=d['ids'],
                           mode='cloth_overlay',maximum_distance=max(s[2] for s in samples),
                           correspondence=correspondence,carrier=chain,support_vertices=sorted(changed_support),maximum_discarded_weight=0.,
                           max_influences=max(len(row) for _,row in changes)))
    if refine:refine_bound_overlays(rig,meshes,dict(components=report))
    return dict(components=report,vertices=sum(len(r['ids']) for r in report),height=height)


def refine_bound_overlays(rig,meshes,report):
    for row in report.get('components',[]):
        row['refinement']=refine_attachment(rig,next(o for o in meshes if o.name==row['object']),row)
    return report


def refine_attachment(rig, obj, record):
    """Refine only the overlapping ribbons; interpolate existing shape layers.

    Linear skinning of differently tessellated surfaces can intersect even with
    the same continuous weights. This adds local samples without moving an old
    vertex, smoothing the rest mesh, or projecting the authored loose tip.
    """
    import bmesh
    from avatar_apparel_weights import assign
    source=obj.data
    refined_seeds=set(source.get('hallway_cloth_overlay_refined_seeds',[]))
    seed=min(record['ids']);already_refined=seed in refined_seeds
    support=set(record['support_vertices']);trim=set(record['ids'])
    chain=record['carrier']
    bones=sorted((b for b in rig.data.bones if b.use_deform and b.name.rsplit('_',1)[0]==chain
                  and b.name.rsplit('_',1)[1].isdigit()),key=lambda b:int(b.name.rsplit('_',1)[1]))
    direction=bones[-1].head_local-bones[0].head_local
    axis=Vector();component=int(np.argmax(np.abs(direction)));axis[component]=direction[component];axis.normalize()
    stations=np.asarray([b.head_local.dot(axis) for b in bones])
    original=np.asarray([v.co for v in source.vertices])
    keys={k.name:np.asarray([v.co for v in k.data]) for k in source.shape_keys.key_blocks} if source.shape_keys else {}
    source.calc_loop_triangles();triangles=[tuple(t.vertices) for t in source.loop_triangles]
    source_tree=BVHTree.FromPolygons(original,triangles,all_triangles=True)
    tolerance=max(float(np.ptp(original,axis=0).max())*1e-5,1e-8)
    mesh=source.copy();bm=bmesh.new()
    try:
        bm.from_mesh(mesh)
        tag=bm.faces.layers.int.new('hallway_overlay_refinement');bm.verts.ensure_lookup_table()
        for f in bm.faces:
            ids={v.index for v in f.verts};f[tag]=2 if ids<=trim else 1 if ids<=support else 0
        faces=[f for f in bm.faces if f[tag]]
        if not already_refined:
            bmesh.ops.triangulate(bm,faces=faces,quad_method='FIXED',ngon_method='EAR_CLIP')
            edges=list({e for f in bm.faces if f[tag] for e in f.edges})
            bmesh.ops.subdivide_edges(bm,edges=edges,cuts=3,use_grid_fill=True,smooth=0.)
        bm.verts.index_update()
        new_trim={v.index for f in bm.faces if f[tag]==2 for v in f.verts}
        new_cloth={v.index for f in bm.faces if f[tag]==1 for v in f.verts}
        bm.faces.layers.int.remove(tag);bm.to_mesh(mesh);mesh.update()
        assert np.array_equal(original,np.asarray([v.co for v in mesh.vertices[:len(original)]]))
        for name,co in keys.items():
            assert np.array_equal(co,np.asarray([v.co for v in mesh.shape_keys.key_blocks[name].data[:len(co)]])),name
        for i in range(len(original),len(mesh.vertices)):
            hit,_,face,distance=source_tree.find_nearest(mesh.vertices[i].co)
            assert face is not None and distance<tolerance,'Refinement moved the rest surface'
            ids=triangles[face];factors=_barycentric(np.asarray(hit),*original[list(ids)])
            for name,co in keys.items():
                expected=factors@co[list(ids)];actual=np.asarray(mesh.shape_keys.key_blocks[name].data[i].co)
                assert np.linalg.norm(expected-actual)<tolerance,'Refinement changed an expression surface'
        assert len(mesh.materials)==len(source.materials) and len(mesh.uv_layers)==len(source.uv_layers)
        obj.data=mesh
        transform=rig.matrix_world.inverted()@obj.matrix_world
        for i in new_cloth|new_trim:
            old=weights(obj,i);mass=sum(w for n,w in old.items() if 'Secondary_Skirt_' in n)
            if mass<=1e-8:continue
            coordinate=(transform@mesh.vertices[i].co).dot(axis)
            budget=4-sum('Secondary_Skirt_' not in n for n in old)
            row=preserve_attachment(old,chain_profile([b.name for b in bones],stations,coordinate,budget))
            assign(obj,[i],row)
        mesh['hallway_cloth_overlay_refined_seeds']=sorted(refined_seeds|{seed})
        record['refined_ids']=sorted(new_trim);record['refined_support']=sorted(new_cloth)
        return dict(added_vertices=len(mesh.vertices)-len(original),original_vertices_unchanged=True,
                    shape_keys_preserved=len(keys),max_influences=4)
    except BaseException:
        obj.data=source
        import bpy
        if mesh.users==0:bpy.data.meshes.remove(mesh)
        raise
    finally:bm.free()
