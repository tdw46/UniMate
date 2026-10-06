"""Fresh pelvis heat domain and matching garment attachment; no mesh edits."""
import bpy
from mathutils import Vector
from avatar_apparel_weights import weights,assign
from avatar_leg_binding import _bone_distance


def thigh_envelope(point,hips,spine,thighs):
    """Compact socket envelope in the pelvis frame, independent of mesh scale."""
    up=spine.tail_local-hips.head_local
    if up.length_squared<1e-16:up=Vector((0,0,1))
    else:up.normalize()
    thigh=min(thighs,key=lambda b:_bone_distance(point,b))
    start=-thigh.length*.25
    # End by the pelvis center, with a short anatomical fallback for rigs
    # whose Hips head is below the sockets. ARP's helper occupies ~1/7 thigh.
    end=min(thigh.length/7,max(thigh.length*.04,(hips.head_local-thigh.head_local).dot(up)))
    t=max(0.,min(1.,((point-thigh.head_local).dot(up)-start)/(end-start)))
    return 1.-t*t*(3.-2.*t)


def pelvis_owned_weights(value,point,hips,spine,thighs):
    """Cap total thigh ownership; give excess to the pelvis, not another leg.

    This is a projection, not repeated attenuation: applying it twice is inert.
    It implements the pelvis-ownership principle of ARP's improved-hips pass
    without adding export bones or depending on its operators/source.
    """
    result=dict(value);mass=sum(value.get(b.name,0.) for b in thighs)
    excess=max(0.,mass-thigh_envelope(point,hips,spine,thighs))
    if excess:
        for b in thighs:result[b.name]=value.get(b.name,0.)*(mass-excess)/mass
        result[hips.name]=result.get(hips.name,0.)+excess
    return {n:w for n,w in result.items() if w>1e-10}


def confine_pelvis_weights(rig,meshes):
    """Remove thigh bleed from torso skin before sampling garment attachment."""
    hips,spine,chest,thighs,arms,low,core_low,high=landmarks(rig)
    changed=0;max_removed=0.
    for obj in meshes:
        m=rig.matrix_world.inverted()@obj.matrix_world
        slots={i for i,mat in enumerate(obj.data.materials) if mat and
               any(t in mat.name.lower() for t in ('skin','body')) and
               not any(t in mat.name.lower() for t in ('cloth','hair','shoe'))}
        ids={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}
        for i in ids:
            p=m@obj.data.vertices[i].co
            if min(_bone_distance(p,b) for b in (hips,spine,chest,*thighs))>min(_bone_distance(p,b) for b in arms):continue
            old=weights(obj,i)
            if any(n.startswith('Secondary_') for n in old):continue
            new=pelvis_owned_weights(old,p,hips,spine,thighs)
            removed=sum(old.get(b.name,0.)-new.get(b.name,0.) for b in thighs)
            if removed<=1e-7:continue
            assign(obj,[i],new);changed+=1;max_removed=max(max_removed,removed)
    if changed:rig['hallway_waist_transfer_revision']=1
    return dict(vertices=changed,maximum_thigh_mass_removed=max_removed)


def landmarks(rig):
    h=rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
    def bone(role):
        name=getattr(h,role).node.bone_name
        fallback={'hips':('Hips',),'spine':('Spine',),'chest':('Chest','Spine1'),
                  'left_upper_leg':('Thigh.L','LeftUpLeg'),'right_upper_leg':('Thigh.R','RightUpLeg'),
                  'left_upper_arm':('UpperArm.L','LeftArm'),'right_upper_arm':('UpperArm.R','RightArm')}
        name=next((n for n in (name,*fallback[role]) if n in rig.data.bones),None)
        if name is None:raise ValueError('Missing pelvis landmark: '+role)
        return rig.data.bones[name]
    hips=bone('hips');spine=bone('spine');chest=bone('chest')
    thighs=[bone(s+'_upper_leg') for s in ('left','right')]
    arms=[bone(s+'_upper_arm') for s in ('left','right')]
    low=min(b.tail_local.z+b.length*.35 for b in thighs)
    core_low=min(b.head_local.z-b.length*.12 for b in thighs)
    high=chest.head_local.z+chest.length*.5
    return hips,spine,chest,thighs,arms,low,core_low,high


def repair_pelvis_weights(rig,meshes,force=False):
    """Heat only torso/upper-leg skin together, fading at anatomical bounds."""
    if rig.get('hallway_pelvis_binding_revision')==1 and not force:return dict(applied=False,reason='Pelvis domain already repaired')
    from autorig_bust import repaired_heat
    from avatar_skirt_fit_io import rest_edit
    hips,spine,chest,thighs,arms,low,core_low,high=landmarks(rig)
    allowed={b.name for b in (hips,spine,chest,*thighs)}
    refs=[];points=[];faces=[];lookup={}
    for obj in meshes:
        m=rig.matrix_world.inverted()@obj.matrix_world
        coords=[m@v.co for v in obj.data.vertices]
        eligible={i for i,p in enumerate(coords) if low<=p.z<=high and
            min(_bone_distance(p,b) for b in (hips,spine,chest,*thighs))<=min(_bone_distance(p,b) for b in arms)}
        for f in obj.data.polygons:
            mat=obj.data.materials[f.material_index] if f.material_index<len(obj.data.materials) else None
            label=mat.name.lower() if mat else ''
            if not any(t in label for t in ('skin','body')) or any(t in label for t in ('cloth','hair','shoe')):continue
            if not all(i in eligible for i in f.vertices):continue
            ids=[]
            for i in f.vertices:
                key=(obj.name,i)
                if key not in lookup:lookup[key]=len(refs);refs.append((obj,i));points.append(coords[i])
                ids.append(lookup[key])
            faces.append(ids)
    if len(faces)<8:return dict(applied=False,reason='No reliable pelvis skin surface')
    contaminated=sum(core_low<=p.z<=chest.head_local.z and
        sum(w for n,w in weights(obj,i).items() if n not in allowed)>.02
        for (obj,i),p in zip(refs,points))
    if not contaminated and not force:return dict(applied=False,reason='Pelvis weights already anatomically scoped')
    deform={b.name:b.use_deform for b in rig.data.bones}
    data=None;proxy=None
    def smooth(x):
        x=max(0.,min(1.,x));return x*x*(3-2*x)
    with rest_edit(rig):
        try:
            for b in rig.data.bones:b.use_deform=b.name in allowed
            data=bpy.data.meshes.new('Temporary pelvis heat domain')
            data.from_pydata(points,[],faces);data.update()
            proxy=bpy.data.objects.new('Temporary pelvis heat domain',data);bpy.context.scene.collection.objects.link(proxy)
            audit=repaired_heat([proxy],rig)
            solved=[weights(proxy,v.index) for v in data.vertices]
            changed=0
            for (obj,i),p,new in zip(refs,points,solved):
                # A shared pelvis solve must not reconnect the independent
                # leg domains below the hip. Move cross-leg heat to the
                # nearest thigh, fading this restriction across its socket.
                own=min(thighs,key=lambda b:_bone_distance(p,b))
                other=next(b for b in thighs if b!=own)
                local=smooth((own.head_local.z-p.z)/max(own.length*.12,1e-8))
                cross=new.get(other.name,0.)*local
                new[other.name]=new.get(other.name,0.)-cross
                new[own.name]=new.get(own.name,0.)+cross
                t=smooth((p.z-low)/(core_low-low))*smooth((high-p.z)/(high-chest.head_local.z))
                old=weights(obj,i);value={n:old.get(n,0)*(1-t)+new.get(n,0)*t for n in old.keys()|new.keys()}
                total=sum(value.values());assign(obj,[i],{n:w/total for n,w in value.items()});changed+=1
        finally:
            for name,value in deform.items():rig.data.bones[name].use_deform=value
            if proxy:bpy.data.objects.remove(proxy,do_unlink=True)
            if data and not data.users:bpy.data.meshes.remove(data)
    rig['hallway_pelvis_binding_revision']=1
    return dict(applied=True,vertices=changed,contaminated_vertices=contaminated,low=low,core_low=core_low,core_high=chest.head_local.z,high=high,heat=audit)


def transfer_garment_waist(rig,meshes,force=False):
    """Extend underlying-body attachment upward through the owned dress bodice.

    Only materials already containing generated skirt vertices participate.
    Spring-weighted rows, including the overlap above roots, are left to the
    strip binder so the two stages cannot erase one another's work.
    """
    if rig.get('hallway_waist_transfer_revision')==4 and not force:return dict(vertices=0,reason='Upper garment already attached')
    from avatar_waist_attachment import WaistAttachment
    hips,spine,chest,thighs,arms,low,core_low,high=landmarks(rig)
    roots=[b for b in rig.data.bones if 'hallway_garment_top' in b]
    if not roots:return dict(vertices=0)
    lower=min(b.head_local.z for b in roots)
    attachment=WaistAttachment(rig,meshes)
    changed=0;fallback_start=attachment.fallback
    for obj in meshes:
        m=rig.matrix_world.inverted()@obj.matrix_world
        seed={v.index for v in obj.data.vertices if any(n.startswith(('Secondary_Skirt_', 'Secondary_SkirtSupport_')) for n in weights(obj,v.index))}
        slots={f.material_index for f in obj.data.polygons if any(i in seed for i in f.vertices)}
        ids={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}
        for i in ids:
            p=m@obj.data.vertices[i].co
            if not lower<=p.z<=high:continue
            if min(_bone_distance(p,b) for b in (hips,spine,chest))>min(_bone_distance(p,b) for b in arms):continue
            value=attachment.sample(p)
            # Do not replace existing bodice bindings where no body is found.
            if attachment.fallback!=fallback_start:
                fallback_start=attachment.fallback;continue
            old=weights(obj,i)
            # The strip binder already samples the body and combines it with
            # the local spring pair here. Do not overwrite its upward overlap.
            if any(n.startswith(('Secondary_Skirt_', 'Secondary_SkirtSupport_')) and w>1e-8 for n,w in old.items()):continue
            t=max(0.,min(1.,(high-p.z)/(high-chest.head_local.z)));t=t*t*(3-2*t)
            new={n:old.get(n,0)*(1-t)+value.get(n,0)*t for n in old.keys()|value.keys()}
            new=pelvis_owned_weights(new,p,hips,spine,thighs)
            new=attachment.side_owned(p,new)
            keep=sorted(new,key=lambda n:(-new[n],n))[:4];total=sum(new[n] for n in keep)
            assign(obj,[i],{n:new[n]/total for n in keep});changed+=1
    if changed:rig['hallway_waist_transfer_revision']=4
    return dict(vertices=changed,surface_samples=attachment.sampled,fallback=attachment.fallback,lower=lower,upper=high)
