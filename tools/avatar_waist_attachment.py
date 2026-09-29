"""Body-surface attachment for the fixed waistband; never edits geometry."""
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform
from avatar_apparel_weights import weights


WAIST_FRACTION = .10
TRANSITION_FRACTION = .05


class WaistAttachment:
    def __init__(self, rig, meshes):
        self.rig = rig
        human = rig.data.vrm_addon_extension.vrm1.humanoid.human_bones
        self.torso = [getattr(human, name).node.bone_name for name in
                      ('hips', 'spine', 'chest', 'upper_chest')]
        self.torso = [name for name in self.torso if name in rig.data.bones]
        self.allowed = set(self.torso + [getattr(human, side+'_upper_leg').node.bone_name
                                       for side in ('left', 'right')])
        self.allowed.discard('')
        self.center = rig.data.bones[self.torso[0]].head_local.copy()
        self.distance = max(rig.data.bones[n].length for n in self.torso)*.35
        self.points, self.values, self.triangles = [], [], []
        for obj in meshes:
            matrix = rig.matrix_world.inverted() @ obj.matrix_world
            obj.data.calc_loop_triangles()
            offset = len(self.points)
            self.points.extend(matrix @ v.co for v in obj.data.vertices)
            self.values.extend({n: w for n, w in weights(obj, v.index).items() if n in self.allowed}
                               for v in obj.data.vertices)
            for tri in obj.data.loop_triangles:
                mat = obj.data.materials[tri.material_index] if tri.material_index < len(obj.data.materials) else None
                label = mat.name.lower() if mat else ''
                if not any(token in label for token in ('skin', 'body')) or any(
                        token in label for token in ('cloth', 'skirt', 'bottom', 'hair')):
                    continue
                indices = tuple(i+offset for i in tri.vertices)
                if all(sum(self.values[i].values()) >= .95 for i in indices):
                    self.triangles.append(indices)
        self.tree = BVHTree.FromPolygons(self.points, self.triangles, all_triangles=True) if self.triangles else None
        self.sampled = self.fallback = 0

    def sample(self, point):
        if self.tree:
            # Prefer the surface immediately underneath, not a nearby sleeve or
            # the opposite leg. A bounded nearest hit covers open body surfaces.
            radial = Vector((self.center.x-point.x, self.center.y-point.y, 0.))
            hit = self.tree.ray_cast(point, radial.normalized(), self.distance) if radial.length > 1e-8 else (None,)*4
            if hit[0] is None:
                hit = self.tree.find_nearest(point, self.distance)
            co, _, face, _ = hit
            if co is not None and face is not None:
                triangle = self.triangles[face]
                a, b, c = (self.points[i] for i in triangle)
                if (b-a).cross(c-a).length_squared > 1e-16:
                    bary = barycentric_transform(co, a, b, c, Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1)))
                else:
                    # Collapsed triangles are local failures, not a reason to
                    # discard the entire body reference.
                    corners = (a, b, c)
                    i, j = max(((0,1), (1,2), (2,0)),
                               key=lambda pair: (corners[pair[1]]-corners[pair[0]]).length_squared)
                    edge = corners[j]-corners[i]
                    t = max(0., min(1., (co-corners[i]).dot(edge)/edge.length_squared)) if edge.length_squared > 1e-16 else 0.
                    bary = [0., 0., 0.]
                    bary[i], bary[j] = 1.-t, t
                values = {}
                for i, factor in zip(triangle, bary):
                    for name, w in self.values[i].items():
                        values[name] = values.get(name, 0.) + max(0., factor)*w
                total = sum(values.values())
                if total > 1e-8:
                    self.sampled += 1
                    return {n: w/total for n, w in values.items() if w > 1e-8}
        self.fallback += 1
        # With no underlying body, a waistband follows the mapped pelvis.
        # This stays independent of leg follow and does not invent leg weights.
        return {self.torso[0]: 1.}
