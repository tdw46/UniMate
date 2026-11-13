"""Gentle front-pelvis ownership in generated rest space, shared by both hosts."""
from mathutils import Vector


def smooth(value):
    t=max(0.,min(1.,value))
    return t*t*(3.-2.*t)


def front_envelope(point, hips, spine, thighs):
    """Smooth compact support from generated sockets, not character names."""
    left,right=thighs
    sockets=(left.head_local+right.head_local)*.5
    lateral=left.head_local-right.head_local
    width=lateral.length
    up=spine.tail_local-hips.head_local
    if width<=1e-8 or up.length<=1e-8:return 0.
    lateral.normalize();up.normalize()
    front=lateral.cross(up)
    if front.length<=1e-8:return 0.
    front.normalize()
    delta=Vector(point)-sockets
    leg=(left.length+right.length)*.5
    low=-leg*.25
    high=max((spine.head_local-sockets).dot(up),width*.3)
    z=delta.dot(up)
    vertical=smooth((z-low)/max(-low,1e-8))*(1.-smooth((z-high)/max(spine.length*.6,1e-8)))
    side=1.-smooth((abs(delta.dot(lateral))-.35*width)/(.65*width))
    # Zero behind the socket plane, full strength in front; no side seam.
    anterior=smooth(delta.dot(front)/(.45*width))
    return vertical*side*anterior


def add_hips(row, hips_name, envelope):
    """Move at most a quarter of remaining mass to Hips, maximum four bones.

    This final-generation blend is applied once, after waistband handoff.
    Existing hips ownership is never reduced and free cloth below is untouched.
    """
    if envelope<=1e-8:return dict(row)
    total=sum(row.values())
    if total<=0:return dict(row)
    old=row.get(hips_name,0.)/total
    target=old+(1.-old)*.25*envelope
    other=sorted(((n,w/total) for n,w in row.items() if n!=hips_name and w>1e-10),key=lambda x:(-x[1],x[0]))[:3]
    mass=sum(w for _,w in other)
    if not mass:return {hips_name:1.}
    result={n:w*(1.-target)/mass for n,w in other}
    result[hips_name]=target
    return result


def strengthen_front_pelvis(rig, meshes, plan):
    from avatar_pelvis_binding import landmarks
    from avatar_material_roles import semantic_label
    from avatar_apparel_weights import weights,assign
    hips,spine,chest,thighs,arms,*_=landmarks(rig)
    anatomical={b.name for b in (hips,spine,chest,*thighs)}
    parent=chest.parent
    while parent is not None and parent!=hips:
        anatomical.add(parent.name);parent=parent.parent
    report=[]
    for obj in meshes:
        cloth={i for r in plan['regions'] if r['obj']==obj and r['role']=='skirt' for i in r['ids']}
        slots=set()
        for slot,material in enumerate(obj.data.materials):
            label=semantic_label(material.name) if material else ''
            if any(t in label for t in ('skin','body','pants','trouser','shorts','ズボン','裤','바지')) and not any(t in label for t in ('hair','shoe','cloth')):
                slots.add(slot)
        body={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}-cloth
        matrix=rig.matrix_world.inverted()@obj.matrix_world
        changed=[]
        for i in sorted(body|cloth):
            row=weights(obj,i)
            # A hand at hip height is not pelvis skin, even on a shared mesh.
            if i in body and sum(w for n,w in row.items() if n in anatomical)<.95:continue
            amount=front_envelope(matrix@obj.data.vertices[i].co,hips,spine,thighs)
            if amount<=1e-8:continue
            new=add_hips(row,hips.name,amount)
            gain=new.get(hips.name,0)-row.get(hips.name,0)
            if gain<=1e-8:continue
            assign(obj,[i],new)
            changed.append((i,gain))
        if changed:report.append(dict(object=obj.name,vertices=len(changed),maximum_hips_gain=max(x[1] for x in changed),ids=[x[0] for x in changed]))
    return dict(objects=report,maximum_remaining_mass_transfer=.25,max_influences=4,domain='front_pelvis_from_generated_landmarks')
