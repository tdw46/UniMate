"""Attach detached garment details to the generated cloth's rest-space field.

No imported armature or weights are read. Cloth donors are snapshotted after
final generated skirt binding; secondary shells never feed back into donors.
"""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
from avatar_virtual_topology import connected_graph
from avatar_apparel_weights import weights, assign
from avatar_material_roles import semantic_label
from avatar_surface_attachments import normalized


def _barycentric(p, a, b, c):
    ab, ac, ap = b-a, c-a, p-a
    aa, bb, cc = ab@ab, ab@ac, ac@ac
    det = aa*cc-bb*bb
    if det <= max(aa*cc, 1e-30)*1e-12:
        # Collapsed cloth triangle: interpolate its longest surviving edge.
        pairs = [(0, 1, a, b), (1, 2, b, c), (2, 0, c, a)]
        i, j, u, v = max(pairs, key=lambda x: float((x[3]-x[2])@(x[3]-x[2])))
        d = v-u; t = float(np.clip((p-u)@d/max(float(d@d), 1e-30), 0, 1))
        out = np.zeros(3); out[i] = 1-t; out[j] = t
        return out
    v = (cc*(ap@ab)-bb*(ap@ac))/det
    w = (aa*(ap@ac)-bb*(ap@ab))/det
    out = np.maximum([1-v-w, v, w], 0)
    return out/out.sum()


def bind_garment_attachments(rig, meshes, *, preserve_anatomical=False):
    """Transfer soft patches per vertex and solid assemblies at one anchor.

    Generic semantic hints disambiguate cloth from hardware. Spatial admission
    and actual generated cloth ownership are both required; hands/head, packs,
    hair and remote accessories cannot become skirt donors or attachments.
    """
    points = {o.name: np.asarray([rig.matrix_world.inverted()@o.matrix_world@v.co
                                for v in o.data.vertices], dtype=float) for o in meshes}
    if not points:return dict(components=[],vertices=0)
    height = max(p[:,2].max() for p in points.values())-min(p[:,2].min() for p in points.values())
    records=[];vertices=[];rows=[];triangles=[];soft_triangles=[]
    from avatar_cloth_smoothing import preserve_attachment
    def write(obj,ids,row):
        if preserve_anatomical:
            for i in ids:assign(obj,[i],preserve_attachment(weights(obj,i),row))
        else:assign(obj,ids,row)
    cloth_words=('hem','skirt','jacket','coat','vest','holster','pocket','裙','スカート','치마')
    solid_words=('tool','grenade','canister','bottle','hardware','buckle','装備','工具','도구')
    def cloth_weight(row):
        return sum(w for n,w in row.items() if 'Secondary_Skirt_' in n or 'Accessory_Jacket_' in n)
    for obj in meshes:
        p=points[obj.name];labels={i:set() for i in range(len(p))}
        for f in obj.data.polygons:
            mat=obj.data.materials[f.material_index] if f.material_index<len(obj.data.materials) else None
            for i in f.vertices:
                if mat:labels[i].add(semantic_label(mat.name))
        _,groups,_=connected_graph(obj.data,p,range(len(p)),height*.0001)
        obj.data.calc_loop_triangles();donors=set();long_donors=set()
        for ids in groups:
            q=p[ids];label=' '.join({n for i in ids for n in labels[i]})
            solid=any(t in label for t in solid_words)
            cloth=any(t in label for t in cloth_words) and not solid
            if not (solid or cloth):continue
            span=np.ptp(q,axis=0)
            values={i:weights(obj,i) for i in ids}
            if cloth and max(span)>height*.20:
                supported=[i for i in ids if cloth_weight(values[i])>.5]
                donors.update(supported)
                # Preserve short skirt and jacket panel layouts. Soft detached
                # overlays are repaired on long articulated cloth surfaces;
                # rigid hardware can attach to either short or long cloth.
                if supported and np.ptp(p[supported,2])>height*.24:long_donors.update(supported)
            elif max(span)<height*.19 and q[:,2].max()<rig.data.bones['Chest'].head_local.z:
                records.append(dict(obj=obj,ids=ids,q=q,span=span,solid=solid))
        offset=len(vertices);vertices.extend(p);rows.extend(weights(obj,i) for i in range(len(p)))
        for t in obj.data.loop_triangles:
            if all(i in donors for i in t.vertices):
                triangles.append(tuple(offset+i for i in t.vertices))
                soft_triangles.append(all(i in long_donors for i in t.vertices))
    if not triangles:return dict(components=[],vertices=0)
    xyz=np.asarray(vertices);tree=BVHTree.FromPolygons(vertices,triangles,all_triangles=True)
    def sample(point):
        hit,_,face,distance=tree.find_nearest(Vector(point))
        if face is None:return None,float('inf'),False
        ids=triangles[face];factors=_barycentric(np.asarray(hit),*xyz[list(ids)])
        value={}
        for i,f in zip(ids,factors):
            for n,w in rows[i].items():value[n]=value.get(n,0)+float(f)*w
        return normalized(value),distance,bool(min(factors)>1e-4 and soft_triangles[face])
    # Major solid shells seed assemblies; tiny caps/rivets join the nearest
    # shell. Never join two main shells merely because their bounding boxes touch.
    solid=[r for r in records if r['solid']]
    anchors=[r for r in solid if len(r['ids'])>=70 and min(r['span'])>height*.012]
    def assembly(r):
        tree=KDTree(len(r['q']))
        for i,p in enumerate(r['q']):tree.insert(Vector(p),i)
        tree.balance()
        return dict(seed=r,parts=[r],tree=tree)
    assemblies=[assembly(r) for r in anchors]
    for r in solid:
        if any(r is a for a in anchors):continue
        best=None;distance=float('inf');best_key=(2,float('inf'))
        for a in assemblies:
            q=a['seed']['q']
            contained=bool(np.all(r['q'].min(0)>=q.min(0)-height*.003)
                           and np.all(r['q'].max(0)<=q.max(0)+height*.003))
            dist=0. if contained else min(a['tree'].find(Vector(p))[2] for p in r['q'])
            key=(0,float(np.linalg.norm(r['q'].mean(0)-q.mean(0)))) if contained else (1,dist)
            if key<best_key:best,distance,best_key=a,0. if contained else dist,key
        if best is not None and distance<height*.012:best['parts'].append(r)
        else:assemblies.append(assembly(r))
    report=[]
    for assembly in assemblies:
        q=assembly['seed']['q'];samples=[sample(p) for p in q]
        # Closest cloth-facing patch is the attachment, not the object's tip.
        order=np.argsort([s[1] for s in samples])[:max(3,len(q)//8)]
        if np.median([samples[i][1] for i in order])>height*.035:continue
        row={}
        for i in order:
            value,_,_=samples[i]
            for n,w in value.items():row[n]=row.get(n,0)+w/len(order)
        row=normalized(row)
        if cloth_weight(row)<.5:continue
        for part in assembly['parts']:
            write(part['obj'],part['ids'],row)
            report.append(dict(object=part['obj'].name,ids=part['ids'],mode='solid',anchor=assembly['seed']['ids'][0],weights=row))
    for r in records:
        if r['solid']:continue
        samples=[sample(p) for p in r['q']]
        if max(s[1] for s in samples)>height*.035:continue
        if min(cloth_weight(s[0]) for s in samples)<.5:continue
        # A neighboring cloth panel projects onto the donor's boundary, whereas
        # an overlaid pocket has support across the interior of that surface.
        if sum(s[2] for s in samples)<len(samples)*.5:continue
        for i,(row,_,_) in zip(r['ids'],samples):write(r['obj'],[i],row)
        report.append(dict(object=r['obj'].name,ids=r['ids'],mode='soft'))
    return dict(components=report,vertices=sum(len(r['ids']) for r in report))
