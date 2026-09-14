"""Anonymous body/cloth attachment, exclusions and coordinate invariance."""
import json, sys
from pathlib import Path
from types import SimpleNamespace as NS
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from avatar_garment_transition import front_torso_transition, loose_front_strength


class Bones(dict):
    def __iter__(self): return iter(self.values())


def fixture(points, scale=1., offset=(0,0,0), role='torso', raw=False):
    def transform(p): return Vector(p)*scale+Vector(offset)
    specs={'Root':((0,0,0),(0,0,.2)), 'Spine':((0,0,.2),(0,0,.5)),
           'Chest':((0,0,.5),(0,0,1.2)), 'Neck':((0,0,1.2),(0,0,1.5)),
           'Head':((0,0,1.5),(0,0,1.8))}
    for side,sign in [('L',1),('R',-1)]:
        specs.update({'Clavicle.'+side:((sign*.1,0,1.15),(sign*.4,0,1.)),
                      'UpperArm.'+side:((sign*.4,0,1.),(sign*1.4,0,1.)),
                      'Forearm.'+side:((sign*1.4,0,1.),(sign*2.2,0,1.)),
                      'Hand.'+side:((sign*2.2,0,1.),(sign*2.5,0,1.))})
    bones=Bones({n:NS(name=n,head_local=transform(a),tail_local=transform(b),length=(Vector(b)-Vector(a)).length*scale)
                 for n,(a,b) in specs.items()})
    rig=NS(data=NS(bones=bones))
    obj=NS(data=NS(vertices=[NS(co=transform(p)) for p in points]))
    region={'object':obj,'indices':list(range(len(points))),'role':role,
            'low':transform((-2.5,-.4,0)), 'high':transform((2.5,.4,1.3))}
    source={i:{'UpperArm.'+('L' if p[0]>=0 else 'R'):1.} for i,p in enumerate(points)}
    if raw: return region,rig,source
    return front_torso_transition(region,rig,source),source


# Front/back points cross the previous abrupt arm threshold; top/underside,
# cut and head/neck samples must stay exactly fixed. No faces or UVs needed.
points=[(.65,-.3,.85),(.65,.3,.85),(-.65,-.3,.85),(-.65,.3,.85),
        (.65,0,1.3),(.65,0,.7),(.65,-.3,0),(.4,-.3,1.5),(2.,-.3,1.)]
(base,mask),source=fixture(points)
for i in (0,2):
    assert sum(w for n,w in base[i].items() if n in ('Root','Spine','Chest'))>.9
    assert abs(sum(base[i].values())-1)<1e-7 and len(base[i])<=4
assert abs(base[0].get('UpperArm.L',0)-base[2].get('UpperArm.R',0))<1e-7
assert base[1]==source[1] and base[3]==source[3]
assert all(base[i]==source[i] for i in range(4,len(points)))
body=fixture(points,role='body')[0][0]
assert body==base, 'coincident body and cloth must move together'
for scale,offset in [(.03,(3.,-4.,7.)),(2.7,(-8.,2.,-3.)),(100.,(0,0,0))]:
    actual=fixture(points,scale,offset)[0][0]
    assert max(abs(actual[i].get(n,0)-base[i].get(n,0)) for i in base for n in actual[i].keys()|base[i].keys())<2e-5
# Subdividing/reordering the samples does not alter the field at existing points.
dense=[(.5+i*.001,-.3,.85) for i in range(701)]
values=fixture(dense)[0][0]
arm=[v.get('UpperArm.L',0) for v in values.values()]
assert max(abs(b-a) for a,b in zip(arm,arm[1:]))<.003
assert all(b>=a-1e-7 for a,b in zip(arm,arm[1:]))
reordered=fixture(list(reversed(dense)))[0][0]
assert all(values[i]==reordered[len(dense)-1-i] for i in values)
assert fixture(points,role='head')[0][0]==source
for scale in (.1,1.,5.):
    region,rig,_=fixture([(.6,-.3,.8),(.7,-.3,.8),(.65,-.3,.9)],scale=scale,raw=True)
    region['object'].data.calc_loop_triangles=lambda:None
    region['object'].data.loop_triangles=[NS(vertices=(0,1,2))]
    for gap,expected in ((.1,0.),(.8,1.)):
        surface=NS(tree=NS(find_nearest=lambda p:(p,None,0,gap*rig.data.bones['Neck'].length)))
        strength,clearance=loose_front_strength(region,rig,surface)
        assert abs(strength-expected)<1e-6
        assert abs(clearance-gap)<1e-6
    # Zero clearance strength is an exact no-op, even on an eligible front.
    region,rig,original=fixture(points,scale=scale,raw=True)
    assert front_torso_transition(region,rig,original,0.)[0]==original
report={'passed':True,'body_cloth_coherence':True,'top_underside_cut_neck_exact':True,
        'front_only_and_bilateral_symmetry':True,'scale_translation_invariance':True,
        'continuous_arm_transition':True,'vertex_order_invariance':True,
        'fitted_vs_loose_clearance_and_scale':True}
(ROOT/'outputs/complex_avatar_grid/garment_transition_generalization.json').write_text(json.dumps(report,indent=2))
print('GARMENT_TRANSITION_PASSED',json.dumps(report),flush=True)
