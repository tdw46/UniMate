"""Compare shoulder weights against the preserved baseline on all nine avatars."""
import json, os, sys
from pathlib import Path
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from avatar_apparel_weights import weights
from avatar_shoulder_weights import shoulder_mask
OUT=Path(os.environ.get('AVATAR_EVAL_ROOT',ROOT/'outputs/complex_avatar_grid'))

def snapshot():
    return {o.name:{'points':[tuple(v.co) for v in o.data.vertices],
                   'uvs':sorted((p.material_index, tuple(sorted((o.data.loops[i].vertex_index, tuple(tuple(layer.data[i].uv) for layer in o.data.uv_layers)) for i in p.loop_indices))) for p in o.data.polygons),
                   'weights':[weights(o,v.index) for v in o.data.vertices]}
            for o in bpy.context.scene.objects if o.type=='MESH'}

report=[]
for entry in json.loads((OUT/'sources/manifest.json').read_text()):
    name=entry['id']
    bpy.ops.wm.open_mainfile(filepath=str(OUT/'before_shoulder_smoothing'/name/'02_fresh_rig.blend'))
    before=snapshot()
    bpy.ops.wm.open_mainfile(filepath=str(OUT/'avatars'/name/'02_fresh_rig.blend'))
    after=snapshot();rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
    changed=protected=0;energy=[0.,0.];faces={'front':0,'back':0}
    for obj_name,a in after.items():
        b=before[obj_name];obj=bpy.data.objects[obj_name]
        assert a['points']==b['points'] and a['uvs']==b['uvs'],obj_name
        for i,(w,old) in enumerate(zip(a['weights'],b['weights'])):
            assert len(w)<=4 and abs(sum(w.values())-1)<1e-5
            delta=max(abs(w.get(k,0)-old.get(k,0)) for k in w.keys()|old.keys())
            mask,_=shoulder_mask(rig,Vector(a['points'][i]))
            if mask<=1e-6 or obj.get('binding_role') not in ('body','torso') or old.get('Head',0)+old.get('Neck',0)>=1e-8:
                protected+=1
                assert delta<1e-6,(name,obj_name,i,delta)
            if delta>1e-6:
                changed+=1
                side='L' if a['points'][i][0]>rig.data.bones['Neck'].head_local.x else 'R'
                front=a['points'][i][1]<rig.data.bones['UpperArm.'+side].head_local.y
                faces['front' if front else 'back']+=1
        for edge in obj.data.edges:
            i,j=edge.vertices
            mask=max(shoulder_mask(rig,Vector(a['points'][k]))[0] for k in (i,j))
            if mask<=1e-6:continue
            for n,state in enumerate((b,a)):
                x,y=state['weights'][i],state['weights'][j]
                energy[n]+=sum((x.get(k,0)-y.get(k,0))**2 for k in x.keys()|y.keys())
    assert changed and all(faces.values())
    assert energy[1]<energy[0],(name,energy)
    report.append({'id':name,'changed_vertices':changed,'protected_vertices_unchanged':protected,
                   'changed_front_back':faces,'shoulder_edge_weight_energy_before':energy[0],
                   'shoulder_edge_weight_energy_after':energy[1],
                   'energy_reduction_percent':100*(1-energy[1]/energy[0]),
                   'geometry_and_uvs_unchanged':True,'passed':True})
(OUT/'shoulder_validation.json').write_text(json.dumps(report,indent=2))
print('SHOULDERS_VALIDATED',json.dumps(report),flush=True)
