"""Evaluated fabric stretch, with a fixture-only check of 07's printed bow.

UV/color selection is used only to measure the reported visual defect. No
binding algorithm receives avatar IDs, texture pixels, or these UV bounds.
"""
import json, os, sys
from pathlib import Path
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from avatar_apparel_weights import weights, classify_regions
OUT=Path(os.environ.get('AVATAR_EVAL_ROOT',ROOT/'outputs/complex_avatar_grid'))

def snapshot():
    result={}
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH':continue
        m=obj.data
        faces=[]
        for p in m.polygons:
            corners=[(m.loops[i].vertex_index,tuple(tuple(uv.data[i].uv) for uv in m.uv_layers)) for i in p.loop_indices]
            start=min(range(len(corners)),key=lambda i:corners[i])
            faces.append((p.material_index,tuple(corners[start:]+corners[:start])))
        result[obj.name]={'geometry':([tuple(v.co) for v in m.vertices],sorted(faces)),
                          'weights':[weights(obj,i) for i in range(len(m.vertices))]}
    return result


def strain(bow=False):
    subjects=[];pixel_cache={}
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH' or obj.get('binding_role')!='torso':continue
        m=obj.data;m.calc_loop_triangles();ids=[]
        for t in m.loop_triangles:
            if bow:
                mat=m.materials[t.material_index]
                if not mat or 'Tops' not in mat.name:continue
                uv=np.mean([m.uv_layers.active.data[i].uv[:] for i in t.loops],axis=0)
                if not (.32<uv[1]<.48 and (.35<uv[0]<.435 or .565<uv[0]<.65)):continue
                image=next((n.image for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image),None)
                if image is None:continue
                if image.name not in pixel_cache:pixel_cache[image.name]=np.array(image.pixels[:]).reshape(image.size[1],image.size[0],4)
                rgb=pixel_cache[image.name][int(uv[1]*image.size[1]),int(uv[0]*image.size[0]),:3]
                if not (rgb[2]>rgb[0]*1.25 and rgb[1]>rgb[0]*1.1 and rgb[0]<.3):continue
            ids.append(tuple(t.vertices))
        if not ids:continue
        tr=np.array(ids);points=np.array([v.co[:] for v in m.vertices]);edges=points[tr[:,1:]]-points[tr[:,0,None]]
        areas=np.linalg.norm(np.cross(edges[:,0],edges[:,1]),axis=1)*.5
        valid=areas>1e-10;tr,edges,areas=tr[valid],edges[valid],areas[valid]
        subjects.append((obj,tr,np.linalg.pinv(edges.transpose(0,2,1)),areas))
    assert subjects
    arm_weights=[]
    if bow:
        for obj,tr,_,_ in subjects:
            for i in np.unique(tr):
                arm_weights.append(sum(w for n,w in weights(obj,int(i)).items()
                                       if n.startswith(('UpperArm.','Forearm.','Hand.'))))
    maximum=peak_rms=peak_p99=0.;minimum=1.;worst_triangle={};pose_extremes={}
    frames=list(range(240,360)) if bow else sorted(set(range(240,360,6))|{275,299,359})
    for frame in frames:
        bpy.context.scene.frame_set(frame);deps=bpy.context.evaluated_depsgraph_get();all_sv=[];all_area=[]
        for obj,tr,inverse,areas in subjects:
            ev=obj.evaluated_get(deps);mesh=ev.to_mesh();p=np.array([v.co[:] for v in mesh.vertices]);ev.to_mesh_clear()
            edges=p[tr[:,1:]]-p[tr[:,0,None]]
            gradient=edges.transpose(0,2,1)@inverse
            sv=np.linalg.svd(gradient,compute_uv=False)[:,:2];all_sv.append(sv);all_area.append(areas)
            if float(sv.max())>maximum:
                k=int(np.argmax(sv[:,0]));maximum=float(sv[k,0])
                rest=np.array([obj.data.vertices[int(i)].co[:] for i in tr[k]])
                worst_triangle={'mesh':obj.name,'vertices':tr[k].tolist(),'frame':frame,'rest_area':float(areas[k]),'rest_center':rest.mean(axis=0).tolist(),'posed_longest_edge':float(max(np.linalg.norm(p[tr[k,i]]-p[tr[k,j]]) for i,j in [(0,1),(1,2),(2,0)]))}
        sv=np.concatenate(all_sv);area=np.concatenate(all_area)
        maximum=max(maximum,float(sv.max()))
        minimum=min(minimum,float(sv[:,1].min()))
        if frame in (240,275,299,359):
            pose_extremes[str(frame)]={'max_stretch':float(sv[:,0].max()),
                                      'min_scale':float(sv[:,1].min())}
        peak_p99=max(peak_p99,float(np.quantile(sv[:,0],.99)))
        peak_rms=max(peak_rms,float(np.sqrt(np.average(np.sum((sv-1)**2,axis=1),weights=area))))
    return {'triangles':sum(len(s[1]) for s in subjects),'frames':len(frames),'max_principal_stretch':maximum,
            'printed_bow_max_arm_weight':max(arm_weights,default=0.) if bow else None,
            'printed_bow_arm_rigid_vertices':sum(w>.99 for w in arm_weights) if bow else None,
            'min_principal_scale':minimum,'pose_extremes':pose_extremes,
            'peak_p99_stretch':peak_p99,'peak_area_weighted_rms_strain':peak_rms,'worst_triangle':worst_triangle}

report=[]
for entry in json.loads((OUT/'sources/manifest.json').read_text()):
    name=entry['id'];case={'id':name}
    bpy.ops.wm.open_mainfile(filepath=str(OUT/'before_front_torso_fix'/name/'02_fresh_rig.blend'))
    baseline=snapshot()
    for state,folder in [('rejected','before_front_torso_fix'),('corrected','avatars')]:
        bpy.ops.wm.open_mainfile(filepath=str(OUT/folder/name/'02_fresh_rig.blend'))
        if state=='corrected':
            current=snapshot()
            assert all(current[n]['geometry']==baseline[n]['geometry'] for n in baseline)
            for obj in bpy.context.scene.objects:
                if obj.type=='MESH' and obj.get('binding_role') not in ('torso','body'):
                    assert current[obj.name]['weights']==baseline[obj.name]['weights'],obj.name
            rig=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
            meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
            protected_count=0;max_protected_drift=0.
            for region in classify_regions(meshes,rig,{})[1]:
                if region['role'] not in ('body','torso'):continue
                name_obj=region['object'].name
                neck=rig.data.bones['Neck']
                for i in region['indices']:
                    p=region['object'].data.vertices[i].co
                    side='L' if p.x>=neck.head_local.x else 'R'
                    arm=rig.data.bones['UpperArm.'+side]
                    axis=(arm.tail_local-arm.head_local).normalized()
                    delta=p-arm.head_local;radial=delta-axis*delta.dot(axis)
                    # Independent exclusion audit: actual radial top/underside,
                    # cut plane, and neck seam, not whole incident triangles.
                    protected=(radial.y>=0 or abs(radial.y)<=radial.length*.15 or
                               p.z<=region['low'].z or
                               p.z>=neck.head_local.z+neck.length*.2)
                    if protected:
                        old=baseline[name_obj]['weights'][i];new=current[name_obj]['weights'][i]
                        drift=max((abs(old.get(n,0)-new.get(n,0)) for n in old.keys()|new.keys()),default=0)
                        # Rebuilding Blender heat can vary at float precision.
                        assert drift<1e-5,(name,name_obj,i,drift)
                        max_protected_drift=max(max_protected_drift,drift)
                        protected_count+=1
            case['protected_top_underarm_cut_neck_vertices_unchanged']=protected_count
            case['max_protected_weight_rebuild_drift']=max_protected_drift
            case['geometry_uvs_unchanged']=True;case['head_accessory_weights_unchanged']=True
        case[state]=strain()
        if name=='avatarsample_a':case[state+'_printed_bow']=strain(bow=True)
    case['rms_strain_change_percent']=100*(case['corrected']['peak_area_weighted_rms_strain']/case['rejected']['peak_area_weighted_rms_strain']-1)
    # Include poses beyond the binder's three probes. Allow one percentage
    # point beyond its 5% bound for the independently sampled animation peak.
    assert case['rms_strain_change_percent']<=6., (name,case['rms_strain_change_percent'])
    if name=='avatarsample_a':
        before=case['rejected_printed_bow'];after=case['corrected_printed_bow']
        print('PRINTED_BOW_METRICS',json.dumps(after),flush=True)
        assert after['max_principal_stretch']<1.35
        assert after['min_principal_scale']>.75
        assert after['printed_bow_arm_rigid_vertices']==0
        # Halve this local defect while honoring the whole-garment bound;
        # the extension/compression limits above remain independent guards.
        assert after['peak_area_weighted_rms_strain']<before['peak_area_weighted_rms_strain']*.5
    case['passed']=True;report.append(case);print('GARMENT_STRAIN',json.dumps(case),flush=True)
(OUT/'garment_strain_validation.json').write_text(json.dumps(report,indent=2))
