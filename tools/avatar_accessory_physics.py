"""Keep unbranched accessory runs within one VRM spring integration chain.

The Blender VRM simulation publishes separate springs together. Avoid making
every serial segment depend on a separate spring publication; keep serial runs
in one chain. Cross-chain branching order is undefined by VRM 1.0, so retain
the runtime algorithm and do not promise identical fork timing across engines.
Bone names, geometry, skin weights and collider roles are unchanged.
"""

def physics_chains(chains):
    eligible={c['names'][0]:c for c in chains if c['role']=='accessory' and len(c['names'])==2}
    children={n:[] for n in eligible}
    for name,c in eligible.items():
        if c['parent'] in children:children[c['parent']].append(name)
    consumed=set();result=[]
    for c in chains:
        name=c['names'][0]
        if name not in eligible:result.append(c);continue
        if name in consumed:continue
        # The planner emits parents before children; roots/fork children begin runs.
        run=[c];consumed.add(name)
        while len(children[run[-1]['names'][0]])==1:
            next_name=children[run[-1]['names'][0]][0]
            if next_name in consumed:break
            run.append(eligible[next_name]);consumed.add(next_name)
        merged=dict(c)
        merged['names']=[item['names'][0] for item in run]+[run[-1]['names'][-1]]
        merged['line']=[item['line'][0] for item in run]+[run[-1]['line'][-1]]
        merged['serial_accessory']=True
        result.append(merged)
    return result


def merge_generated_springs(rig, prefix):
    """Upgrade only matching generated serial springs in an existing review rig.

    Runs with different center/collider ownership remain separate. Preserve all
    native joint parameters, custom properties and the first spring's identity.
    """
    sb=rig.data.vrm_addon_extension.spring_bone1
    candidates=[];by_head={}
    for index,s in enumerate(sb.springs):
        if not s.vrm_name.startswith(prefix+'Accessory_') or '_Strand' in s.vrm_name or len(s.joints)<2:continue
        names=[j.node.bone_name for j in s.joints]
        if not all(n.startswith(prefix+'Accessory_') and n in rig.data.bones for n in names):continue
        first=rig.data.bones[names[0]]
        row=dict(index=index,spring=s,names=names,parent=first.parent.name if first.parent else '',
                 owner=(s.center.bone_name,tuple(ref.collider_group_uuid for ref in s.collider_groups)))
        candidates.append(row);by_head[names[0]]=row
    children={n:[] for row in candidates for n in row['names'][:-1]}
    for row in candidates:
        if row['parent'] in children:children[row['parent']].append(row)
    consumed=set();removed=[];runs=0
    def snapshot(j):
        return dict(name=j.node.bone_name,stiffness=j.stiffness,drag_force=j.drag_force,gravity_power=j.gravity_power,
                    gravity_dir=tuple(j.gravity_dir),hit_radius=j.hit_radius,custom={k:v for k,v in j.items() if k.startswith(('bvt_', 'hallway_', 'generated_')) and isinstance(v,(str,int,float,bool))})
    for row in candidates:
        if row['index'] in consumed:continue
        run=[row];consumed.add(row['index'])
        while len(children[run[-1]['names'][-2]])==1:
            child=children[run[-1]['names'][-2]][0]
            if child['index'] in consumed or child['owner']!=row['owner']:break
            run.append(child);consumed.add(child['index'])
        if len(run)==1:continue
        records=[snapshot(j) for c in run for j in list(c['spring'].joints)[:-1]]+[snapshot(run[-1]['spring'].joints[-1])]
        s=row['spring'];s.joints.clear()
        for record in records:
            j=s.joints.add();j.node.bone_name=record['name']
            for key in ('stiffness','drag_force','gravity_power','gravity_dir','hit_radius'):setattr(j,key,record[key])
            for key,value in record['custom'].items():
                if key not in ('node','animation_state'):j[key]=value
        removed.extend(c['index'] for c in run[1:]);runs+=1
    for index in sorted(removed,reverse=True):sb.springs.remove(index)
    return dict(merged_runs=runs,removed_fragment_springs=len(removed))
