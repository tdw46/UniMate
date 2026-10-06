"""Scale-independent hem classification and knee sampling regression."""
import unittest
import math
from avatar_dress import dress_profile,knee_profile_weights,skirt_sector_angle


class DressProfileTest(unittest.TestCase):
    def test_mid_calf_boundary(self):
        for scale in (.01,1.,100.):
            for hem,lower in ((.7,False),(.4,False),(.3,False),(.299,True),(.05,True)):
                profile=dress_profile(.95*scale,hem*scale,.5*scale,.1*scale,4)
                self.assertEqual(profile['lower_follow'],lower)
                self.assertTrue(all(a>b for a,b in zip(profile['heights'],profile['heights'][1:])))
                if hem<.5:self.assertAlmostEqual(profile['heights'][profile['knee_index']],.5*scale)

    def test_short_skirt_is_unchanged(self):
        self.assertEqual(dress_profile(1.,.6,.5,.1,4)['heights'],[1.,.9,.8,.7,.6])

    def test_both_long_sections_have_simulated_segments(self):
        profile=dress_profile(.95,.05,.5,.1,2)
        self.assertGreaterEqual(profile['knee_index'],2)
        self.assertGreaterEqual(len(profile['heights'])-1-profile['knee_index'],2)

    def test_angular_follow_anchors_and_taper(self):
        for angle,expected in ((0,1.),(30,.75),(60,.25),(90,0.),(120,.0125),(150,.0375),(180,.05)):
            weights=knee_profile_weights(math.cos(math.radians(angle)))
            self.assertAlmostEqual(sum(weights),1.)
            self.assertTrue(all(0<=w<=1 for w in weights))
            self.assertAlmostEqual(sum(w*v for w,v in zip(weights,(1.,0.,.05))),expected)
            self.assertEqual(weights,knee_profile_weights(math.cos(math.radians(-angle))))

    def test_bilateral_layout(self):
        for count in (6,8,10,12,14,16,24):
            angles=[skirt_sector_angle(i,count) for i in range(count)]
            points=[(math.cos(a),math.sin(a)) for a in angles]
            self.assertTrue(all(abs(x)>1e-6 for x,y in points))
            for x,y in points:
                self.assertLess(min(abs(x+u)+abs(y-v) for u,v in points),1e-10)
        for count in (5,7,11):
            with self.assertRaises(ValueError):skirt_sector_angle(0,count)

    def test_invalid_intervals(self):
        for values in ((.5,.6,.4,.1,4),(1.,.1,.4,.5,4),(1.,.1,float('nan'),0.,4)):
            with self.assertRaises(ValueError):dress_profile(*values)


if __name__=='__main__':unittest.main()
