"""Group mounted assemblies and garment surfaces before hanging-strand inference.

Only rest geometry, material domains and the newly generated humanoid/weights
are inputs. Source skeletons and imported weights never participate.
"""
import math
import numpy as np
from mathutils import Vector
from avatar_virtual_topology import connected_graph, protected_extremities
from avatar_apparel_weights import weights
from avatar_material_roles import semantic_label
from avatar_accessories import segment_distance


def normalized(row, limit=4):
    row=dict(sorted(((n,w) for n,w in row.items() if w>1e-8),key=lambda p:-p[1])[:limit])
    total=sum(row.values())
    return {n:w/total for n,w in row.items()} if total else {}


def plan_mounted(rig, meshes, all_points, height, prefix):
    """A mounted pack is one assembly, even across mesh/material seams."""
    chest=rig.data.bones['Chest'];hips=rig.data.bones['Hips'];head=rig.data.bones['Head']
    records=[];fixed=[]
    for obj in meshes:
        p=all_points[obj.name]
        _,groups,_=connected_graph(obj.data,p,range(len(p)),height*.0001)
        labels={i:set() for i in range(len(p))}
        for face in obj.data.polygons:
            mat=obj.data.materials[face.material_index] if face.material_index<len(obj.data.materials) else None
            for i in face.vertices:
                if mat:labels[i].add(mat.name)
        for ids in groups:
            q=p[ids];span=np.ptp(q,axis=0);names={n for i in ids for n in labels[i]}
            label=' '.join(semantic_label(n) for n in names)
            # Rigid equipment hugging the front/side head surface, including
            # its disconnected lenses and bands. Hair/bows keep their route.
            if (any(t in label for t in ('goggle','glass','tool','helmet','visor'))
                    and q[:,2].min()>head.head_local.z-height*.02
                    and q[:,2].max()<head.tail_local.z+height*.06):
                fixed.append(dict(obj=obj,role='mounted',ids=ids,values={i:{'Head':1.} for i in ids}))
                continue
            if len(ids)<4 or q[:,2].min()<hips.head_local.z-height*.02 or q[:,2].max()>chest.tail_local.z+height*.10:continue
            if max(span)>height*.30 or span[0]>height*.22:continue
            if any(t in label for t in ('hair','skin','face','glove','shoe','skirt','hem','topwear','jacket','coat','vest','clothingaccessory')):continue
            rear=(np.quantile(q[:,1],.1)>chest.head_local.y+height*.035
                  and np.mean(q[:,1])>chest.head_local.y+height*.065)
            named=any(t in label for t in ('backpack','rucksack','bag','リュック','背包','배낭'))
            records.append(dict(obj=obj,ids=ids,q=q,names=names,seed=(rear and max(span)>height*.045) or named))
    seeds=[r for r in records if r['seed']]
    if not seeds:return [],fixed
    # Material-domain ownership recruits narrow pack straps; spatial adjacency
    # recruits differently shaded lids/panels. Never recruit a full torso shell.
    owned=list(seeds);remaining=[r for r in records if not r['seed']]
    for _ in range(3):
        added=[]
        for r in remaining:
            q=r['q']
            for owner in owned:
                same=bool(r['names']&owner['names'])
                gap=np.maximum(np.maximum(q.min(0)-owner['q'].max(0),owner['q'].min(0)-q.max(0)),0)
                if ((same and np.linalg.norm(gap)<height*.12) or np.linalg.norm(gap)<height*.012):
                    # Do not use a shared anonymous material to swallow the
                    # front torso. Mounted components stay compact/lateral.
                    if np.ptp(q[:,0])<height*.18 and np.ptp(q[:,1])<height*.22:
                        added.append(r);break
        if not added:break
        owned+=added;remaining=[r for r in remaining if not any(r is a for a in added)]
    q=np.concatenate([r['q'] for r in owned]);top=float(q[:,2].max());bottom=float(q[:,2].min())
    rear=q[q[:,1]>chest.head_local.y+height*.02]
    if len(rear)<8:rear=q
    root=np.median(q[q[:,2]>top-max((top-bottom)*.15,height*.005)],axis=0)
    support=np.median(rear[rear[:,2]>np.quantile(rear[:,2],.65)],axis=0)
    support[2]=min(top-height*.015,float(np.quantile(rear[:,2],.80)))
    end=np.median(rear,axis=0);end[2]=bottom
    if np.linalg.norm(end-support)<height*.01:return [],fixed
    # A lateral sling attaches at its shoulder. A centered pack attaches to
    # the chest. The upper strap is fixed; the pouch has one shared moving bone.
    lateral=float(np.median(q[:,0])-chest.head_local.x)
    parent=('Clavicle.L' if lateral>0 else 'Clavicle.R') if abs(lateral)>height*.045 else 'Chest'
    if parent not in rig.data.bones:parent='Chest'
    centered = abs(lateral) <= height*.045 and np.ptp(rear[:,0]) >= height*.10 and np.ptp(rear[:,2]) >= height*.10
    if centered:
        # One shared pivot for the assembled pouch, independent of mesh seams.
        # Front shoulder straps are outside this rear-volume center estimate.
        root=(rear.min(0)+rear.max(0))*.5
        support=root.copy()
    names=[prefix+f'Accessory_Pack_{i:02d}' for i in range(3)]
    chains=[dict(role='accessory',names=names,line=[Vector(root),Vector(support),Vector(end)],parent=parent,physics_start=1,mounted_pack=True)]
    if centered:
        names=names[:2]
        chains=[dict(role='accessory',names=names,line=[Vector(root),Vector(end)],parent='Chest',physics_start=0,mounted_pack=True,centered_pack=True)]
    front=q[q[:,1]<chest.head_local.y-height*.015]
    front_names=None;front_line=[]
    if not centered and len(front)>24 and np.ptp(front[:,2])>height*.08:
        for z in np.linspace(top,bottom,5):
            near=front[np.argsort(abs(front[:,2]-z))[:max(4,len(front)//12)]]
            point=np.median(near,axis=0);point[2]=z;front_line.append(Vector(point))
        front_line[-1]=Vector(end)
        front_names=[prefix+f'Accessory_PackFront_{i:02d}' for i in range(5)]
        chains.append(dict(role='accessory',names=front_names,line=front_line,parent=parent,physics_start=1,mounted_pack=True))
    from avatar_springs import _vertical_weights
    for r in owned:
        values={}
        for i,p in zip(r['ids'],r['q']):
            # Hardware and panels use the same spatial field, including across
            # disconnected seams. Main pouch ownership is rigid, not independently
            # simulated panels. Centered packs add the shared body blend below.
            t=float(np.clip((top-p[2])/max(top-support[2],height*.005),0,1))
            t=t*t*(3-2*t);back={names[0]:1-t,names[1]:t}
            if front_names:
                front_row=_vertical_weights(front_names[1:],front_names[0],(top-p[2])/max(top-bottom,height*.01),True)
                f=float(np.clip((chest.head_local.y+height*.02-p[1])/(height*.04),0,1));f=f*f*(3-2*f)
                row={n:w*(1-f) for n,w in back.items()}
                for n,w in front_row.items():row[n]=row.get(n,0)+w*f
            else:row=back
            if centered:
                # Keep the whole rear assembly on one body frame and one
                # spring, so disconnected hardware cannot split at seams.
                row={'Chest':.5,names[0]:.5}
            values[i]=normalized(row)
        fixed.append(dict(obj=r['obj'],role='mounted',ids=r['ids'],values=values))
    return chains,fixed


def plan_jackets(rig,obj,points,excluded,height,prefix,start_index=0):
    """Lower torso panels, with sleeves fixed and inset hardware surface-owned."""
    slots={i for i,m in enumerate(obj.data.materials) if m and any(t in semantic_label(m.name)
           for t in ('jacket','coat','vest','topwear','ジャケット','上着','外套','재킷'))}
    candidates={i for f in obj.data.polygons if f.material_index in slots for i in f.vertices}-set(excluded)
    if not candidates:return [],[]
    hips=rig.data.bones['Hips'];chest=rig.data.bones['Chest']
    center=np.array(hips.head_local)[:2]
    # Sleeve admission uses the anatomical segment, not component size. A
    # bulky sleeve must remain with the arm just as a close-fitting sleeve does.
    limb=set()
    for side in ('L','R'):
        for name in ('UpperArm.','Forearm.'):
            bone=rig.data.bones.get(name+side)
            if bone is None:continue
            a=np.array(bone.head_local);b=np.array(bone.tail_local)
            axis=b-a;t=(points-a)@axis/max(float(axis@axis),1e-12)
            mask=(t>.12)&(t<1.3)&(segment_distance(points,a,b)<height*.065)
            limb.update(np.flatnonzero(mask))
    torso=sorted(candidates-limb)
    if not torso:return [],[]
    q=points[torso];angles=np.arctan2(q[:,1]-center[1],q[:,0]-center[0])%math.tau
    # A zipper alone cannot establish a garment. Require a shell around the
    # torso and a substantial lower edge, independently of its material name.
    top=min(float(chest.head_local.z),float(np.quantile(q[:,2],.65)))
    low=float(q[:,2].min())
    bins=set((angles[q[:,2]<top]/math.tau*8).astype(int))
    if len(bins)<6 or top-low<height*.025 or low>hips.tail_local.z+height*.025:return [],[]
    count=12;chains=[];sector_names=[];bottoms=[]
    from avatar_springs import _vertical_weights
    for s in range(count):
        angle=s*math.tau/count
        dist=np.abs((angles-angle+math.pi)%math.tau-math.pi)
        band=q[dist<math.tau/count]
        bottom=float(np.quantile(band[:,2],.03)) if len(band) else low
        bottom=min(bottom,top-height*.01);bottoms.append(bottom)
        line=[]
        from avatar_cloth_handoff import cloth_segments
        joints=cloth_segments(top-bottom,height,3,short_jacket=True)+1
        for z in np.linspace(top,bottom,joints):
            score=dist/(math.tau/count)+np.abs(q[:,2]-z)/max(top-bottom,height*.01)
            near=q[np.argsort(score)[:max(4,len(q)//80)]]
            xy=np.median(near[:,:2],axis=0)
            line.append(Vector((*xy,z)))
        names=[f'{prefix}Accessory_Jacket_{start_index:02d}_{s:02d}_{j:02d}' for j in range(joints)]
        chains.append(dict(role='accessory',names=names,line=line,parent='Chest',physics_start=0,garment=True))
        sector_names.append(names)
    values={}
    for i in sorted(candidates):
        base=normalized({n:w for n,w in weights(obj,i).items() if 'Secondary_' not in n}) or {'Chest':1.}
        if i in limb or points[i,2]>=top:values[i]=base;continue
        a=(math.atan2(points[i,1]-center[1],points[i,0]-center[0])%math.tau)/math.tau*count
        sector=int(a);row={}
        for s,f in ((sector,1-(a-sector)),((sector+1)%count,a-sector)):
            t=(top-points[i,2])/(top-bottoms[s]);t=float(np.clip(t,0,1))
            for n,w in _vertical_weights(sector_names[s],'Chest',t,True).items():row[n]=row.get(n,0)+f*w
        values[i]=normalized(row)
    return chains,[dict(obj=obj,role='garment',ids=sorted(candidates),values=values)]


def install_garment_contacts(rig,meshes,plan,prefix):
    """Per-chain torso/leg capsules fitted inside the garment's rest envelope."""
    from avatar_colliders import plan_colliders
    from avatar_vrm_colliders import add_capsule,add_group,organize_colliders
    from avatar_colliders import segment_distance as distance
    fit=plan_colliders(rig,meshes)
    body=[s for s in fit['collider_details'] if s['role']=='accessory']
    # Covered torsos may expose no skin for the generic collider fitter.
    # Use the inner envelope of the admitted garment, excluding sleeves, to
    # recover that volume; per-chain rest limits below prevent deflection.
    cloth=np.concatenate([plan['points'][r['obj'].name][r['ids']]
                          for r in plan['regions'] if r['role']=='garment'])
    for name in ('Hips','Spine','UpperSpine','Chest'):
        bone=rig.data.bones.get(name)
        if bone is None:continue
        a=np.array(bone.head_local);b=np.array(bone.tail_local)
        band=cloth[(cloth[:,2]>=min(a[2],b[2])-fit['height']*.015)
                   &(cloth[:,2]<=max(a[2],b[2])+fit['height']*.015)
                   &(abs(cloth[:,0]-a[0])<fit['height']*.15)]
        if len(band)<8:continue
        radius=float(np.quantile(segment_distance(band,a,b),.25))*.90
        if radius<fit['height']*.01:continue
        existing=next((spec for spec in body if spec['bone']==name),None)
        if existing:existing['body_radius']=max(existing['body_radius'],radius)
        else:body.append(dict(bone=name,body_radius=radius))
    sb=rig.data.vrm_addon_extension.spring_bone1;count=0;clearances=[];shared={}
    from avatar_vrm_colliders import call_operator,cleanup_orphan_displays
    owned={g.uuid for g in sb.collider_groups if g.vrm_name.startswith(prefix+'JacketContact_')}
    used={r.collider_uuid for g in sb.collider_groups if g.uuid in owned for r in g.colliders}
    other={r.collider_uuid for g in sb.collider_groups if g.uuid not in owned for r in g.colliders}
    for i in reversed(range(len(sb.collider_groups))):
        if sb.collider_groups[i].uuid in owned:call_operator('remove_spring_bone1_collider_group',rig,collider_group_index=i)
    for i in reversed(range(len(sb.colliders))):
        if sb.colliders[i].uuid in used-other:call_operator('remove_spring_bone1_collider',rig,collider_index=i)
    cleanup_orphan_displays(rig)
    for chain in plan['chains']:
        if not chain.get('garment'):continue
        name=chain['names'][0].rsplit('_',1)[0]
        spring=next(s for s in sb.springs if s.vrm_name==name)
        # Generic accessory capsules were fitted to all chains together. Own
        # this generated jacket chain's tighter, individually clear envelope.
        spring.collider_groups.clear()
        group=add_group(rig,prefix+'JacketContact_'+name)
        samples=[(rig.data.bones[t.node.bone_name].head_local,j.hit_radius)
                 for j,t in zip(spring.joints,spring.joints[1:])]
        for spec in body:
            b=rig.data.bones[spec['bone']]
            if max(b.head_local.z,b.tail_local.z)<min(p.z for p,_ in samples)-fit['height']*.03:continue
            if min(b.head_local.z,b.tail_local.z)>max(p.z for p,_ in samples)+fit['height']*.05:continue
            for section in range(2):
                offset=Vector((0,b.length*(section+.5)/2,0))
                limit=min(distance(p,b.matrix_local@offset,b.matrix_local@offset)-r-fit['margin'] for p,r in samples)
                radius=min(spec['body_radius'],limit)
                if radius<=fit['height']*.001:continue
                key=(b.name,*offset,float(radius))
                if key not in shared:
                    shared[key]=add_capsule(rig,dict(bone=b.name,offset=list(offset),tail=list(offset),radius=radius)).uuid
                    count+=1
                group.colliders.add().collider_uuid=shared[key]
                clearances.append(min(distance(p,b.matrix_local@offset,b.matrix_local@offset)-r-radius for p,r in samples))
        spring.collider_groups.add().collider_group_uuid=group.uuid
    organize_colliders(rig)
    return dict(colliders=count,minimum_rest_clearance=min(clearances) if clearances else None)
