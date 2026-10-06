"""Blender-side regression for per-chain obstacle scope and containment pruning."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from avatar_dress_contacts import prune_competing_contacts,is_tight,corner_normals


class DressContactPolicyTest(unittest.TestCase):
    def test_scope_and_redundant_capsules_at_multiple_scales(self):
        for scale in (.001,1.,1000.):
            def shape(bone,r,offset=0):
                return dict(bone=bone,radius=r*scale,offset=[offset*scale,0,0],tail=[offset*scale,scale,0])
            payload=dict(colliders=[shape('LeftLeg',.05),shape('LeftLeg',.06),shape('RightLeg',.06),shape('Hips',.5),shape('LeftLeg',.06,.3)])
            group=dict(colliders=[0,1,2,3,4])
            prune_competing_contacts(payload,group,{'LeftLeg'})
            self.assertEqual(group['colliders'],[1,4])
            prune_competing_contacts(payload,group,{'LeftLeg'})
            self.assertEqual(group['colliders'],[1,4])

    def test_keep_single_equal_shape_and_distinct_bones(self):
        a=dict(bone='LeftLeg',radius=.05,offset=[0,0,0],tail=[0,1,0])
        b=dict(a,bone='LeftUpLeg')
        payload=dict(colliders=[a,dict(a),b]);group=dict(colliders=[0,1,2])
        prune_competing_contacts(payload,group,{'LeftLeg','LeftUpLeg'})
        self.assertEqual(group['colliders'],[1,2])

    def test_corner_faces_are_orthogonal_and_outward(self):
        from mathutils import Vector
        for center in (Vector((.1,0,.2)),Vector((-.1,.3,-.2))):
            a,b=corner_normals(center)
            self.assertEqual(a.dot(b),0.)
            self.assertGreater(a.dot(center),0.)
            self.assertGreater(b.dot(center),0.)

    def test_loose_and_absent_profiles_do_not_activate(self):
        self.assertFalse(is_tight({}))
        self.assertFalse(is_tight(dict(upper=.2,lower=.1)))
        self.assertFalse(is_tight(dict(upper=1.,lower=.1)))
        self.assertTrue(is_tight(dict(upper=.2,lower=.8)))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(DressContactPolicyTest)
    result=unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():raise RuntimeError('Contact policy regression')
