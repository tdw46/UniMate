"""Rest-clearance classifier and slider endpoint regressions (no Blender needed)."""
import math
import unittest
from avatar_dress_fit import clearance_tightness, adapted_follow, ring_clearance


class DressFitTest(unittest.TestCase):
    def test_scale_independent_and_monotone(self):
        for scale in (.001,1,1000):
            values=[clearance_tightness(gap*scale,.4*scale) for gap in (0,.01,.04,.09,.2)]
            self.assertEqual(values[0],1)
            self.assertEqual(values[-1],0)
            self.assertEqual(values,sorted(values,reverse=True))
            self.assertAlmostEqual(values[2],clearance_tightness(.04,.4))

    def test_follow_controls_preserve_endpoints(self):
        for tightness in (0,.25,1):
            self.assertEqual(adapted_follow(0,tightness),0)
            self.assertEqual(adapted_follow(1,tightness),1)
            for base in (.15,.5,.55,.6,.95):
                self.assertEqual(adapted_follow(base,tightness,0),base)
                self.assertGreaterEqual(adapted_follow(base,tightness),base)
        self.assertEqual(adapted_follow(.55,0),.55)
        self.assertAlmostEqual(adapted_follow(.55,1),.8906882591093117)

    def test_narrow_tube_vs_flared_skirt(self):
        def ring(rx,ry):return [(rx*math.cos(i*math.tau/24),ry*math.sin(i*math.tau/24)) for i in range(24)]
        narrow=ring_clearance(ring(.14,.12),(.085,0),.05)
        wide=ring_clearance(ring(.35,.30),(.085,0),.05)
        self.assertGreater(clearance_tightness(narrow,.4),.95)
        self.assertEqual(clearance_tightness(wide,.4),0)
        self.assertLess(ring_clearance(ring(.14,.12),(.3,0),.05),0)
        self.assertAlmostEqual(narrow,ring_clearance(ring(.14,.12),(-.085,0),.05))

    def test_invalid_measurement(self):
        for args in ((0,0),(float('nan'),1),(1,float('inf'))):
            with self.assertRaises(ValueError):clearance_tightness(*args)


if __name__=='__main__':unittest.main()
