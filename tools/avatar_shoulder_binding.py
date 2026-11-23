"""Smooth repaired shoulder skinning without changing the rest mesh or hands."""
import numpy as np
from avatar_apparel_weights import weights,assign
from avatar_weight_seams import coincident_groups
from avatar_weight_smoothing import smooth_limited_weights


def smooth_repaired_shoulders(rig,meshes):
    reports=[]
    if not all('UpperArm.'+s in rig.data.bones for s in ('L','R')):return reports
    for obj in meshes:
        if not obj.data.attributes.get('hallway_arm_bank'):continue
        matrix=rig.matrix_world.inverted_safe()@obj.matrix_world
        points=np.array([matrix@v.co for v in obj.data.vertices]);selected=np.zeros(len(points),dtype=bool)
        lengths=[]
        for side in ('L','R'):
            bone=rig.data.bones['UpperArm.'+side];a=np.array(bone.head_local);d=np.array(bone.tail_local)-a
            t=(points-a)@d/max(float(d@d),1e-16)
            distance=np.linalg.norm(points-a-np.clip(t,0,1)[:,None]*d,axis=1)
            selected|=(t>-.35)&(t<.95)&(distance<bone.length*.7)
            lengths.append(bone.length)
        # Nearby curled fingers or loose physics surfaces are not shoulder skin.
        # Keep ownership explicit even when an unusual rest pose places them
        # inside the anatomical smoothing radius.
        protected=0
        for i in np.flatnonzero(selected):
            row=weights(obj,int(i))
            if any(n.startswith(('Secondary_', 'Hand.', 'Thumb', 'Index', 'Middle', 'Ring', 'Little', 'Thigh.', 'Shin.', 'Foot.')) and value>1e-5 for n,value in row.items()):
                selected[i]=False;protected+=1
        ids=np.flatnonzero(selected)
        if not len(ids):continue
        bank=obj.data.attributes['hallway_arm_bank']
        strength=obj.data.attributes.get('hallway_arm_bank_strength')
        join=obj.data.attributes.get('hallway_arm_join')
        groups=[]
        for group in coincident_groups(points[ids],1e-6):
            vertices=[int(ids[j]) for j in group]
            if join and all(join.data[i].value>.999 for i in vertices):
                groups.append(vertices)
            else:
                # Coincident cut banks are separate surfaces below the socket.
                # Welding these virtual nodes recreates the original webbing.
                for side in sorted({bank.data[i].value for i in vertices}):
                    groups.append([i for i in vertices if bank.data[i].value==side])
        node={i:j for j,g in enumerate(groups) for i in g}
        rows={int(i):weights(obj,int(i)) for i in ids}
        names=sorted({n for row in rows.values() for n in row})
        w=np.array([[sum(rows[i].get(n,0.) for i in group)/len(group) for n in names] for group in groups])
        edges=set();pinned=set();steps=[]
        for edge in obj.data.edges:
            a,b=edge.vertices
            if a in node and b in node:
                if node[a]!=node[b]:
                    edges.add(tuple(sorted((node[a],node[b]))));steps.append(np.linalg.norm(points[a]-points[b]))
            elif a in node:pinned.add(node[a])
            elif b in node:pinned.add(node[b])
        if not edges:continue
        edges=np.array(sorted(edges));movable=np.ones(len(w),dtype=bool);movable[list(pinned)]=False
        # Bound diffusion in anatomical units instead of growing indefinitely
        # with source density. The outer rim pins the existing body solution.
        radius=min(lengths)*.16
        # Retain gentle diffusion before the local uncut-surface heat solve.
        # Applying these ownership masks after heat would recreate the hinge.
        limit=64
        iterations=int(np.clip((radius/max(float(np.median(steps)),1e-8))**2,10,limit))
        allowed=np.ones_like(w,dtype=bool)
        transition=[]
        arm_prefixes=('UpperArm.','Forearm.','Hand.','Thumb','Index','Middle','Ring','Little')
        for row,group in enumerate(groups):
            if strength and len({bank.data[i].value for i in group})==1 and all(strength.data[i].value>.999 for i in group):
                side=bank.data[group[0]].value
                support=np.array([(n.startswith(arm_prefixes) and n.endswith('.L' if side>0 else '.R')) if side else not n.startswith(arm_prefixes) for n in names])
                share=min(join.data[i].value for i in group) if join else 0.
                if share<=0:allowed[row]=support
                else:transition.append((row,support,share))
        filtered=smooth_limited_weights(w,edges,np.full(len(w),4),movable,iterations=iterations,factor=.6,mask_iterations=8,allowed=allowed)
        # A partially shared shoulder rim is not a fully separated cut bank.
        # Let its body/arm mixture fade with the existing smooth join field;
        # a binary ownership mask here creates a new hard hinge inside it.
        for index,support,share in transition:
            foreign=filtered[index,~support].sum()
            ceiling=.5*share
            if movable[index] and foreign>ceiling and filtered[index,support].sum()>1e-10:
                filtered[index,~support]*=ceiling/foreign
                filtered[index,support]*=(1-ceiling)/filtered[index,support].sum()
        for group,row in zip(groups,filtered):assign(obj,group,{n:float(x) for n,x in zip(names,row) if x>0})
        def energy(values):return float(np.mean(np.sum((values[edges[:,0]]-values[edges[:,1]])**2,axis=1)))
        reports.append(dict(object=obj.name,vertices=len(ids),protected_vertices=protected,iterations=iterations,blended_join_nodes=len(transition),edge_energy_before=energy(w),edge_energy_after=energy(filtered),max_influences=int((filtered>0).sum(1).max()),geometry_changed=False))
    return reports
