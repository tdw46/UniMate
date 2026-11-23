"""Synthetic closed-solid checks for optional fused-arm preparation.

Run in Blender's background test runner. No downloaded character is required.
"""
import copy,json,os,sys
from pathlib import Path
import bpy,bmesh,numpy as np
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mesh_landmarks.arm_separation import PreparedMeshes,plan,_section,_thickness
from mesh_landmarks.blender_bridge import geometry,geometry_digest


def fixture(fused):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objects=[]
    for x,sc in [(0,(.15,.09,.33)),((.19 if fused else .29),(.075,.07,.29)),((-.19 if fused else -.29),(.075,.07,.29))]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=24,location=(x,0,.25))
        obj=bpy.context.object;obj.scale=sc;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);objects.append(obj)
    if fused:
        # Curled fingers lie inside the torso slit in X, but extend in front
        # of it in Y. A proximity-only/straight-plane classifier loses them.
        for sign in (-1,1):
            for location,scale in [((sign*.18,-.045,.035),(.035,.05,.045)),((sign*.145,-.085,.015),(.032,.035,.042))]:
                bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=24,location=location)
                obj=bpy.context.object;obj.scale=scale;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True);objects.append(obj)
    body=objects[0];bpy.context.view_layer.objects.active=body
    for obj in objects[1:]:
        mod=body.modifiers.new('Fixture solid','BOOLEAN');mod.operation='UNION';mod.object=obj;mod.solver='EXACT';bpy.ops.object.modifier_apply(modifier=mod.name);bpy.data.objects.remove(obj,do_unlink=True)
    joints={'Hips':dict(head=[0,0,.18],tail=[0,0,.3])}
    for side,sign in [('Left',1),('Right',-1)]:
        x=sign*(.19 if fused else .29)
        for name,a,b in [('Arm',.5,.25),('ForeArm',.25,.02),('Hand',.02,-.04)]:joints[side+name]=dict(head=[x,0,a],tail=[x,0,b])
    return body,dict(joints=joints)

reports=[]
for fused in (False,True):
    obj,pred=fixture(fused);v,f=geometry([obj]);digest=geometry_digest(v,f);base=plan(v,f,pred)
    assert bool(base)==fused,(fused,base)
    if fused:
        sleeve=copy.deepcopy(pred)
        for side in ('Left','Right'):
            sleeve['joints'][side+'Hand']['head'][2]=float(v[:,2].min())-.005
            sleeve['joints'][side+'Hand']['tail'][2]=float(v[:,2].min())-.065
        sleeve_cuts=plan(v,f,sleeve)
        assert sleeve_cuts and not any(c['release_hand'] for c in sleeve_cuts), 'Sleeve-only bridge incorrectly extends hand ownership below the wrist'

    shifted=copy.deepcopy(pred);offset=np.array([2.,-3.,4.]);scale=2.3
    for joint in shifted['joints'].values():
        for key in ('head','tail'):joint[key]=(np.array(joint[key])*scale+offset).tolist()
    moved=plan(v*scale+offset,f,shifted)
    assert len(moved)==len(base)
    for a,b in zip(base,moved):np.testing.assert_allclose(np.array(b['curve']),np.array(a['curve'])*scale+offset[[0,2]],atol=2e-5)
    before=set(bpy.data.objects)
    if fused:
        try:
            PreparedMeshes([obj],pred,True,runtime_root=Path('/missing-hallway-test-runtime'))
        except RuntimeError as error:
            assert 'Prepare the mesh rigging engine' in str(error)
        else:
            raise AssertionError('Missing segmentation runtime was silently accepted')
        assert set(bpy.data.objects)==before,'Failed preparation leaked copies'
        assert geometry_digest(*geometry([obj]))==digest,'Failed preparation changed the source'
    if fused:
        stale=obj.data.attributes.new(name='hallway_generated_fill',type='INT',domain='POINT')
        stale.data.foreach_set('value',np.ones(len(obj.data.vertices),dtype=np.int32))
        obj['hallway_arm_fill_revision']=1
    prepared=PreparedMeshes([obj],pred,True,runtime_root=os.environ.get('HALLWAY_TEST_RUNTIME'))
    try:
        assert geometry_digest(*geometry([obj]))==digest,'Source changed'
        assert prepared.report['applied']==fused
        json.dumps(prepared.report)
        if fused:
            assert all(d.value==1 for d in obj.data.attributes['hallway_generated_fill'].data), 'Preparation changed source provenance'
            assert 'hallway_arm_fill_revision' not in prepared.meshes[0], 'Stale repair revision suppressed a fresh fill pass'
            bm=bmesh.new();bm.from_mesh(prepared.meshes[0].data)
            assert all(e.is_manifold for e in bm.edges),'Cut holes were not filled'
            assert max(len(face.verts) for face in bm.faces)==3,'Cap is not tessellated'
            bm.free()
            repaired,faces=geometry(prepared.meshes)
            if prepared.report['meshes'][0].get('method')=='closed_surface_partition':
                data=prepared.meshes[0].data;attr=data.attributes['hallway_arm_bank']
                labels=np.array([d.value for d in attr.data])
                assert all(len(set(labels[t]))==1 for t in faces),'Cap reconnects the banks'
                assert set(labels)=={-1,0,1}
                join=np.array([d.value for d in data.attributes['hallway_arm_join'].data])
                for cut in prepared.report['cuts']:
                    sign=1 if cut['side']=='Left' else -1
                    upper=pred['joints'][cut['side']+'Arm'];length=np.linalg.norm(np.array(upper['tail'])-upper['head'])
                    below=(repaired[:,0]*sign>0)&(repaired[:,2]<cut['curve'][0][1]-.201*length)
                    assert not np.any(join[below]>0),'Shoulder attachment extends down the released seam'
                for sign in (-1,1):
                    # Visible fingertip surface must leave with the hand;
                    # the nearby central garment must remain on the body.
                    index=np.linalg.norm(repaired-np.array([sign*.145,-.12,.015]),axis=1).argmin()
                    assert labels[index]==sign,('Curled fingertip assigned to garment',sign,repaired[index])
                index=np.linalg.norm(repaired-np.array([0,-.09,.25]),axis=1).argmin()
                assert labels[index]==0,'Central garment assigned to a hand'
                original_positions={tuple(np.round(p,6)) for p in v}
                fill=np.array([d.value for d in data.attributes['hallway_generated_fill'].data])
                assert fill.any() and not fill.all(), 'Generated-fill mask must distinguish repairs from the surviving surface'
                assert all(tuple(np.round(p,6)) in original_positions for p in repaired[fill==0]), 'Unmarked generated interior can bypass curvature repair'
                assert original_positions <= {tuple(np.round(p,6)) for p in repaired},'Original surface moved'
            else:
                for cut in prepared.report['cuts']:
                    curve=np.array(cut['curve'])
                    for a,b in zip(curve[:-1],curve[1:]):
                        x,z=(a+b)/2
                        assert _thickness(_section(repaired[faces],z),x)<1e-5,'Cutter did not create an air gap'
    finally:prepared.close()
    assert set(bpy.data.objects)==before,'Temporary objects leaked'
    reports.append(dict(fused=fused,passed=True))
Path(os.environ['BEYOND_UI_TEST_RESULT']).write_text(json.dumps(dict(passed=True,cases=reports)))
print('ARM_SEPARATION_TEST_PASSED',reports)
