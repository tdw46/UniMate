# Generated cloth contact response

The generation path now installs the reviewed contact response after fitting and
profile initialization. It applies only to generated skirt chains and their owned
contact groups. Existing saved rigs are not migrated on load.

- Merge parallel support/anatomical envelopes only when the enclosing capsule
  remains clear of every affected rest joint. Keep the anatomical capsules.
- Anchor additional lower-leg support to mapped humanoid hips, preserving its
  rest-space endpoints and radius. Actual anatomical leg colliders still move
  with the legs.
- Add 0.25 drag (capped at 0.95) and smoothly floor stiffness over the final 35%
  of full cloth-chain arc length, reaching 35% of the root stiffness at the tip.
- Store generated joint ratios so settings updates/reset retain the profile.

No character names, source skeletons, source weights or material labels drive
this adjustment. It changes neither mesh weights nor the VRM Physics integrator,
collision arithmetic, gravity, chain topology or simulation cadence.

Qualification: the integrated 144-frame walk and run reproduce the reviewed
candidate exactly for all captured vertices and bones. Paired BVT/Hallway tests
on four generated characters check unchanged geometry/weights, unchanged other
springs, retained anatomical capsules (within native serialization tolerance),
idempotence and settings round trips. Blender 4.2 has a native synthetic test;
5.2 character files are not claimed compatible with 4.2. See
`contact-integration-247.json` and the earlier contact audit records.

Residual motion near backswing is still under investigation. This is not a claim
of zero jerk or exhaustive collision-free movement on every character.
