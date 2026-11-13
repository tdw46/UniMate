"""Repair a bad hand prediction only when the opposite mesh proves the fit.

Uses generated joints and base surface positions, never imported bones/weights.
Asymmetric hands or ambiguous fits retain their independent predictions.
"""
import copy
import numpy as np
from mathutils.kdtree import KDTree

DIGITS = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')


def _tree(points):
    tree = KDTree(len(points))
    for i, point in enumerate(points):
        tree.insert(point, i)
    tree.balance()
    return tree


def _nearest(tree, points):
    rows = [tree.find(p) for p in points]
    return np.array([r[2] for r in rows]), np.array([r[1] for r in rows])


def refine(vertices, prediction):
    result = copy.deepcopy(prediction)
    joints = result['joints']
    vertices = np.asarray(vertices, dtype=np.float64)
    audit = dict(revision=2, repaired=[], source='base mesh and predicted joints only')
    result['hand_refinement'] = audit
    required = ['LeftUpLeg', 'RightUpLeg'] + [s+n for s in ('Left', 'Right')
        for n in ('Hand', 'ForeArm', *[f'Hand{d}{i}' for d in DIGITS for i in (1, 2, 3)])]
    if any(n not in joints for n in required):
        audit['skipped'] = 'incomplete predicted hands'
        return result
    def point(name, key='head'):
        return np.array(joints[name][key], dtype=np.float64)
    center = (point('LeftUpLeg') + point('RightUpLeg'))*.5
    normal = point('LeftUpLeg') - point('RightUpLeg')
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return result
    normal /= norm
    # Inputs are already aligned to the pipeline's anatomical X/Z frame.
    # Small unequal hip predictions must not tilt the sagittal plane enough
    # to crop different parts of otherwise matching hands. ICP still has to
    # prove the correspondence; this does not force asymmetric meshes to match.
    if abs(normal[0])>.95:
        normal=np.array([np.sign(normal[0]),0.,0.])
    mirror = np.eye(3)-2*np.outer(normal, normal)
    def reflect(points):
        return np.einsum('...i,ji->...j', points-center, mirror)+center
    length = max(np.linalg.norm(point(s+'Hand', 'tail')-point(s+'Hand'))
                 for s in ('Left', 'Right'))
    if length < 1e-8:
        return result
    # Start with the surface-supported hand, not the midpoint of two wrist
    # predictions: an elbow mistaken for a wrist can move that midpoint off
    # both hands. Each candidate must still prove bilateral surface agreement.
    surface = _tree(vertices)
    def hand_score(side):
        samples = np.array([point(side+'Hand'+d+str(i),k)
                            for d in DIGITS for i in (1,2,3) for k in ('head','tail')])
        axis=point(side+'Hand','tail')-point(side+'Hand')
        fore=point(side+'Hand')-point(side+'ForeArm')
        cosine=float(axis@fore/max(np.linalg.norm(axis)*np.linalg.norm(fore),1e-12))
        return float(_nearest(surface,samples)[0].mean())+length*max(0.,.25-cosine)
    donor = min(('Left','Right'),key=hand_score)
    wrist = point(donor+'Hand')
    forward = wrist-point(donor+'ForeArm')
    if donor=='Right':
        wrist=reflect(wrist)
        forward=mirror@forward
    forward /= max(np.linalg.norm(forward), 1e-8)
    clouds = []
    for points in (vertices, reflect(vertices)):
        delta = points-wrist
        ids = (np.linalg.norm(delta, axis=1)<length*2.8)
        ids &= np.einsum('ni,i->n', delta, forward)>-length*.65
        cloud = np.unique(points[ids], axis=0)
        if len(cloud)<100:
            audit['skipped'] = 'insufficient local surface'
            return result
        clouds.append(cloud)
    # Fit reflected right geometry onto left; only a rigid correspondence is
    # permitted. Uniform scale/shear could conceal genuinely different hands.
    target, source = clouds
    tree = _tree(target)
    samples = source[::max(1, len(source)//1600)]
    rotation = np.eye(3); translation = np.zeros(3)
    for _ in range(32):
        moved = np.einsum('ni,ji->nj', samples, rotation)+translation
        distance, indices = _nearest(tree, moved)
        keep = distance <= np.quantile(distance, .9)
        x = moved[keep]; y = target[indices[keep]]
        xc = x.mean(0); yc = y.mean(0)
        u, _, vt = np.linalg.svd(np.einsum('ni,nj->ij', x-xc, y-yc))
        step = vt.T@u.T
        if np.linalg.det(step)<0:
            vt[-1] *= -1; step = vt.T@u.T
        translation = step@translation+yc-step@xc
        rotation = step@rotation
    moved = np.einsum('ni,ji->nj', source, rotation)+translation
    forward_error = _nearest(tree, moved)[0]
    reverse_error = _nearest(_tree(moved), target)[0]
    error = max(float(np.quantile(forward_error, .95)),
                float(np.quantile(reverse_error, .95)))
    audit['surface_error_relative_to_palm'] = error/length
    audit['alignment_shift_relative_to_palm'] = float(np.linalg.norm(rotation@wrist+translation-wrist)/length)
    if error > length*.035 or audit['alignment_shift_relative_to_palm']>2:
        audit['skipped'] = 'opposite surfaces do not establish a reliable correspondence'
        return result
    surface = _tree(vertices)
    def hand_points(side):
        return np.array([[joints[side+'Hand'+d+str(i)][k] for k in ('head','tail')]
                         for d in DIGITS for i in (1,2,3)])
    left = hand_points('Left'); right = hand_points('Right')
    candidates = {
        'Left': np.einsum('...i,ji->...j', reflect(right), rotation)+translation,
        'Right': reflect(np.einsum('...i,ij->...j', left-translation, rotation)),
    }
    def score(points):
        samples = np.concatenate([a[None,:]+np.linspace(0,1,5)[:,None]*(b-a)
                                  for a,b in points])
        return float(_nearest(surface, samples)[0].mean())
    scores = {s:max(score(p),hand_score(s)) for s,p in [('Left',left),('Right',right)]}
    audit['original_surface_scores'] = scores
    # Choose at most one donor, and require a substantial independently measured
    # improvement. Ordinary small bilateral prediction differences are untouched.
    side = max(scores, key=scores.get)
    candidate_score = score(candidates[side])
    audit['candidate_surface_score'] = candidate_score
    if scores[side] < length*.075 or candidate_score > scores[side]*.7:
        audit['skipped'] = 'no confidently inferior prediction'
        return result
    for d, chain in zip(DIGITS, candidates[side].reshape(5,3,2,3)):
        for i, (head, tail) in enumerate(chain, 1):
            joints[f'{side}Hand{d}{i}'] = dict(head=head.tolist(), tail=tail.tolist())
    # Wrist and palm must use the same validated rigid correspondence.
    other = 'Right' if side=='Left' else 'Left'
    pair = np.array([point(other+'Hand', k) for k in ('head','tail')])
    pair = (np.einsum('...i,ji->...j', reflect(pair), rotation)+translation
            if side=='Left' else reflect(np.einsum('...i,ij->...j', pair-translation, rotation)))
    joints[side+'Hand'] = dict(head=pair[0].tolist(), tail=pair[1].tolist())
    joints[side+'ForeArm']['tail'] = pair[0].tolist()
    audit['repaired'] = [side]
    return result
