"""Run headlessly in Blender. Check front-panel binding on anonymous geometry."""
from pathlib import Path
import sys

from types import SimpleNamespace

import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avatar_trailing_surfaces import _blend_panel_handoff, _fit, _smooth_panel_row


def check_panel(scale, width, depth, role, close, leg_center=0., pitch=.1):
    stations = np.linspace(0., 1., 17)
    columns = np.linspace(-.5, .5, 5)
    points = scale * np.array([(width*x, depth*t, 1.-t) for t in stations for x in columns])
    faces = []
    for i in range(len(stations)-1):
        for j in range(len(columns)-1):
            a = i*len(columns)+j
            b = a+len(columns)
            faces.extend(((a, b, a+1), (a+1, b, b+1)))
    chains, regions = _fit(None, points, np.array(faces), list(range(len(points))),
                           np.array([0., 0., 1.]), scale, pitch*scale,
                           (.12, .5, .88), 'Hips', role, 'Test_', 0, leg_center=leg_center)
    assert len(chains) == 3 and all(c['close_panel'] == close for c in chains)
    segments = len(chains[1]['names']) - 1
    assert segments == int(np.ceil(1. / (pitch * (.72 if close else 1.))))
    for chain in chains:
        assert ('panel_mount' in chain) == close
        if close:
            first_free = chain['panel_free_start']
            assert 0 < first_free < segments - 1
            stop = max(2, first_free - 1)
            active = chain['panel_transition']
            assert active == chain['names'][max(1, stop - 2):stop]
            assert 1 <= len(active) <= 2
            assert chain['names'][0] not in active and chain['names'][first_free] not in active
            assert abs(first_free / segments - .25) <= .5 / segments
            assert chain['panel_mount_head'].z > chain['line'][0].z
            assert chain['panel_mount'] not in chain['names']
    middle = set(chains[1]['names'])
    weights = regions[0]['values']
    assert all(type(vertex_id) is int for vertex_id in weights)
    for i, t in enumerate(stations):
        for j in range(len(columns)):
            row = weights[i*len(columns)+j]
            assert len(row) <= 4 and min(row.values()) >= 0.
            assert abs(sum(row.values())-1.) < 1e-7
            if not close or leg_center is None:
                assert abs(row.get('Hips', 0.) - max(0., 1.-segments*t)) < 1e-7
                assert not {'Thigh.L', 'Thigh.R'}.intersection(row)
            elif t == 0.:
                assert row == {'Hips': 1.}
            elif t < .10:
                support = sum(row.get(n, 0.) for n in ('Hips', 'Thigh.L', 'Thigh.R'))
                assert support > .95
                if j in (0, len(columns)-1):
                    assert row.get('Thigh.R' if j == 0 else 'Thigh.L', 0.) > .2
            if close and .12 <= t <= .20 and leg_center is not None:
                near = 'Thigh.L' if j == len(columns)-1 else 'Thigh.R' if j == 0 else None
                if near:
                    far = 'Thigh.R' if near == 'Thigh.L' else 'Thigh.L'
                    assert row.get(near, 0.) > .50 and row.get(far, 0.) == 0.
            if close and t >= .25:
                allowed = middle | ({'Thigh.L', 'Thigh.R'} if leg_center is not None and t < .31 else set())
                assert set(row) <= allowed
            elif j == 0 and t > (.23 if close and leg_center is not None else 0.):
                assert any(name in row for name in chains[0]['names'])


def check_handoff_support():
    rails = [f'cloth_{i}' for i in range(20)]
    fabric = set(rails)
    for cut in (2, 5, 8):
        supplied = {'Pelvis': .13, 'NearThigh': .27, rails[cut-1]: .39, rails[cut]: .21}
        for u in np.linspace(0., 19., 201):
            row = _blend_panel_handoff(supplied, rails, fabric, u, cut)
            assert row['Pelvis'] == .13 and row['NearThigh'] == .27
            assert len(row) <= 4 and abs(sum(row.values()) - 1.) < 1e-7
            if u <= max(1, cut-2) or u >= cut + min(5, cut):
                assert row == supplied


def check_free_panel_transitions():
    controls = ['root', 'a', 'b', 'c', 'hem']
    assert _smooth_panel_row(controls, 0.) == {'root': 1.}
    # Off-axis points expose a jump in deformation rate when a bone goes
    # abruptly rigid. Use arbitrary unequal pose offsets across each knot.
    offsets = dict(zip(controls, (0., -.12, .31, -.18, .09)))
    def displaced(t):
        row = _smooth_panel_row(controls, t)
        assert 1 <= len(row) <= 4 and min(row.values()) >= 0.
        assert abs(sum(row.values()) - 1.) < 1e-10
        return sum(offsets[n]*w for n,w in row.items())
    step = 1e-4
    for t in (1., 2., 3.):
        left = (displaced(t) - displaced(t-step)) / step
        right = (displaced(t+step) - displaced(t)) / step
        assert abs(left-right) < 2e-4
    assert len(_smooth_panel_row(controls, 2.5)) == 4
    assert len(_smooth_panel_row(controls, 4.)) > 1


def check_hem_clearance():
    from avatar_springs import _fit_free_panel_radii
    def joint(name):
        return SimpleNamespace(node=SimpleNamespace(bone_name=name), hit_radius=0.)
    joints = [joint(n) for n in ('root', 'near', 'far')]
    spring = SimpleNamespace(vrm_name='free', joints=joints)
    untouched = SimpleNamespace(vrm_name='other', joints=[joint('root'), joint('far')])
    bones = dict(leg=SimpleNamespace(matrix_local=Matrix.Identity(4)),
                 near=SimpleNamespace(head_local=Vector((.102, 0., .5))),
                 far=SimpleNamespace(head_local=Vector((.2, 0., .6))))
    data = SimpleNamespace(bones=bones, vrm_addon_extension=SimpleNamespace(
        spring_bone1=SimpleNamespace(springs=[spring, untouched])))
    capsule = dict(role='frontpanel', bone='leg', offset=(0., 0., 0.),
                   tail=(0., 0., 1.), radius=.1)
    _fit_free_panel_radii(SimpleNamespace(data=data), [capsule], 1., {'free'})
    assert abs(joints[0].hit_radius - .0005) < 1e-7
    assert joints[1].hit_radius == .0025
    assert joints[-1].hit_radius == untouched.joints[0].hit_radius == 0.


check_handoff_support()
check_free_panel_transitions()
check_hem_clearance()
check_panel(1., .2, .03, 'frontpanel', True, pitch=.18)
for scale in (.25, 1., 3.):
    check_panel(scale, .2, .03, 'frontpanel', True)
    check_panel(scale, .2, .03, 'frontpanel', True, leg_center=None)
    check_panel(scale, .2, .03, 'backpanel', False)
    check_panel(scale, .5, .03, 'frontpanel', False)
    check_panel(scale, .2, .09, 'frontpanel', False)
print('Front-panel drape checks passed')
