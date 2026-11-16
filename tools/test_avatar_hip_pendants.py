"""Run in Blender on anonymous, disconnected geometry; no source rig or materials."""
from pathlib import Path
from types import SimpleNamespace
from mathutils import Matrix, Vector
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar_hip_pendants import arm_contacts, plan_pendants, waist_contacts


def ring(radius_y, radius_z, center_z, filled=False):
    points = []
    for t in np.linspace(0., 2.*np.pi, 48, endpoint=False):
        for x in (-.0006, .0006):
            for r in ((.2, .6, 1.) if filled else (.9, 1.)):
                points.append((.1+x, r*radius_y*np.cos(t), center_z+r*radius_z*np.sin(t)))
    return np.asarray(points)


def cord(top, bottom, radius):
    return np.array([(.1+radius*np.cos(t), radius*np.sin(t), z)
                     for z in np.linspace(bottom, top, 13)
                     for t in np.linspace(0., 2.*np.pi, 12, endpoint=False)])


def check(scale, side):
    pieces = [cord(1.065, 1.024, .0025), ring(.023, .024, 1., True),
              ring(.009, .027, .949), cord(.922, .88, .0035)]
    surfaces = [(i, p*np.array([side, 1., 1.])*scale, [list(range(len(p)))], set())
                for i, p in enumerate(pieces)]
    def plan(rows):
        return plan_pendants(rows, np.asarray([0., 0., scale]), 1.7*scale, 'Test_')
    chains, regions = plan(surfaces[::-1])
    assert len(chains)==1 and len(regions)==4
    chain = chains[0]
    assert chain['parent']=='Hips' and len(chain['line'])==len(chain['names'])==5
    assert all(a.z>b.z for a,b in zip(chain['line'], chain['line'][1:]))
    for region in regions:
        for row in region['values'].values():
            assert len(row)<=4 and min(row.values())>0 and abs(sum(row.values())-1.)<1e-8
            if region['obj']==1:
                assert row=={chain['names'][1]:1.}
    hips = np.array([0., 0., scale])
    spine = hips + np.array([0., 0., .07*scale])
    upper, bindings = plan_pendants(surfaces, hips, 1.7*scale, 'Test_', spine)
    assert upper[0]['parent']=='Spine'
    assert any('Spine' in row for region in bindings for row in region['values'].values())
    owner = SimpleNamespace(name='Spine', matrix_local=Matrix.Identity(4))
    root = SimpleNamespace(head_local=upper[0]['line'][0], length=.041*scale, parent=owner)
    spring = SimpleNamespace(vrm_name='Test_Pendant_00', joints=[SimpleNamespace(
        node=SimpleNamespace(bone_name=name)) for name in upper[0]['names'][:4]])
    hips_bone = SimpleNamespace(head_local=Vector(hips))
    rig = SimpleNamespace(data=SimpleNamespace(bones={'Hips':hips_bone, 'Test_Pendant_00_00':root},
        vrm_addon_extension=SimpleNamespace(spring_bone1=SimpleNamespace(springs=[spring]))))
    contact, = waist_contacts(rig, 'Test_', 1.7*scale)
    assert contact['role']=='pendant_00' and contact['bone']=='Spine'
    assert contact['radius']>0 and np.isclose(contact['offset'][0], 0.)
    assert contact['tail'][0]*side>0 and contact['tail'][2]<root.head_local.z
    assert waist_contacts(rig, 'SomeoneElse_', 1.7*scale)==[]
    assert plan([r for r in surfaces if r[0]!=2])==([],[])
    occupied = [(i,p,g,{0} if i==1 else taken) for i,p,g,taken in surfaces]
    assert plan(occupied)==([],[])


for size, side in ((.16, 1.), (4., -1.)):
    check(size, side)

def check_arm_clearance():
    rig = SimpleNamespace(data=SimpleNamespace(bones={'forearm': SimpleNamespace(matrix_local=Matrix.Identity(4))}))
    tube = [Vector((.03*np.cos(t), y, .03*np.sin(t))) for y in np.linspace(0., .3, 9)
            for t in np.linspace(0., 2*np.pi, 12, endpoint=False)]
    def fitted(x):
        samples = [(Vector((x, .15, 0.)), 0., 'joint'), (Vector((x, .19, 0.)), 0., 'surface')]
        return arm_contacts(rig, {'forearm': tube}, samples, 1.8)[0]
    far = fitted(.2)
    near = fitted(.04)
    assert far['radius'] > far['body_radius']
    assert near['radius'] < far['radius'] and near['minimum_rest_clearance'] >= .0015*1.8-1e-8
    assert arm_contacts(rig, {'forearm': tube}, [], 1.8) == []


check_arm_clearance()
