"""Garment-specific defaults without Blender or changes to legacy presets."""
import json
import unittest
from avatar_rig_defaults import FOLLOW, SPRINGS, profile_defaults


def rig(*lower):
    return {'hallway_dress_fit': json.dumps({'families': [
        {'sections': {'upper': {'tightness': 1.}, 'lower': {'tightness': value}}}
        for value in lower]})}


class ProfileTests(unittest.TestCase):
    def test_short_loose_flared_and_mixed_keep_legacy(self):
        for value in ({}, rig(), rig(.066), rig(.49), rig(1., .1)):
            p = profile_defaults(value)
            self.assertEqual(p['name'], 'legacy')
            self.assertEqual(p['follow'], FOLLOW)
            self.assertEqual(p['springs'], SPRINGS)
            self.assertEqual(p['follow']['Skirt'], .55)
            self.assertEqual(p['springs']['Skirt']['drag'], .4)
            self.assertEqual(p['springs']['Skirt']['stiffness'], 1.6)

    def test_tight_screenshot_profile(self):
        p = profile_defaults(rig(1.))
        self.assertEqual(p['name'], 'tight_long_dress')
        self.assertEqual(p['follow']['Skirt'], .89)
        self.assertEqual(p['springs']['Skirt']['drag'], .55)
        self.assertEqual(p['springs']['Skirt']['stiffness'], 4.)
        self.assertEqual([p['follow'][n] for n in ('Skirt Knee', 'Skirt Knee Side', 'Skirt Knee Back')], [.6, .55, .5])
        self.assertEqual(p['springs']['Hair'], SPRINGS['Hair'])
        p['springs']['Hair']['drag'] = 0
        self.assertEqual(SPRINGS['Hair']['drag'], .4)

    def test_invalid_measurement_keeps_legacy(self):
        for value in ('bad', 'null', '{"families": [{"sections": {}}]}'):
            self.assertEqual(profile_defaults({'hallway_dress_fit': value})['name'], 'legacy')
