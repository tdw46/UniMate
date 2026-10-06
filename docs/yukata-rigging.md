# Yukata fresh rig validation — 2026-10-06

Ran the current full-character pipeline on the local `yukata.vrm`. The original
247-bone rig is preserved in a separate source-reference scene; the working
character uses new symmetric humanoid bones and fresh weights, with native VRM
spring/collider metadata and BVT simulation.

This asset exposed a complete Blender bone-heat failure when skin and layered
clothing were solved together, with and without temporary hole caps. The new
fallback retries independently by object/material surface, using only geometry
and the generated skeleton. Every one of the 7,478 body/outfit vertices then
received a fresh heat solution; no distant-surface transfer was needed. Shared
vertices average incident solutions before normalization. Ordinary successful
combined solves retain their existing path.

Execution and data checks passed in Blender 5.2.2:

- Original coordinates, topology, UVs and all shape keys unchanged.
- All 29,350 vertices have normalized weights on generated bones.
- 700 dress vertices have at most four influences.
- Symmetric humanoid landmarks; 24 single local XYZ ADD rotation constraints.
- 119 spring entries (83 hair, 24 upper/lower dress and 12 hip support), no centers.
- 179 native colliders; no rest spring/collider contacts in the rest audit.
- BVT ran 180 mixed head/arm/leg simulation steps with finite results and movement
  on 179 frame transitions.

Detailed results are in `yukata-validation.json`. These are execution and
integrity checks, not a guarantee against mesh clipping in all poses. The final
`Yukata_Hallway.blend` in Downloads includes `Yukata - Hallway` and
`Yukata - Source Reference`; visual evaluation is left to the user.

## Tight long-dress follow adaptation

`avatar_dress_fit.py` measures closed rest cross-sections from the generated dress
guides at three heights along each thigh and calf. It compares inward distance
from the leg axes with the anatomical body-fit radius (or the existing 12%-of-
leg-length estimate where covered skin is absent). Inflated contact-guard radii
and posed/simulated mesh positions do not enter this measurement.

Clearance divided by leg length maps smoothly from full tightness at 0.04 to
zero at 0.22. The median over both legs and sampled heights rejects isolated
pinches. Upper and lower sections are classified independently. Only garments
with generated long-dress knee sections participate; short skirts retain their
current follow. This is a rest-fit heuristic, not a collision oracle.

The saved **Tight Dress Follow** slider controls adaptation strength (default 1,
reset restores 1). Effective follow is `base / (1 - .85*tightness*strength*(1-base))`.
Zero follow remains zero, full follow remains full, and adaptation strength zero
restores the authored sliders. The existing directional knee interpolation and
same-side ownership remain intact. Panel labels display the effective ranges.
No additional constraint, direct leg skin weight, collider, or solver is added.

Yukata's estimated clearance is 1.3–1.7 cm. With its existing controls, the result
is 89.1% upper follow and 87.0–90.9% lower follow. Live readback verified unchanged
geometry, weights, humanoid pose, selection and spring parameters. The file was
saved with BVT enabled. A fresh full generation independently produced the same
upper follow, confirming this is integrated into generation rather than a
model-specific edit. Unit tests cover size invariance, loose versus tight shapes,
mirroring, outside-ring containment and slider endpoints; Blender tests cover
idempotence, immediate slider updates, reset, and single portable constraints.
A further 180-step BVT test including 90-degree leg rotation inputs remained
finite; visual clipping review remains separate.

## Contact-conflict investigation

The next live snapshot had upper follow raised by the user to 0.89156 (effective
0.98208 with adaptation). The tests preserve those controls. Smooth rise/hold/
return trajectories exposed discontinuous contact reactions despite finite
simulation output. This supersedes the earlier finite-only motion acceptance.

Ablation tests isolated competing fixed pelvis/opposite-leg contacts. A solver
trace recorded left-front cloth receiving up to 14 mm of correction from a
right-calf collider. Some groups also included geometrically contained duplicate
capsules, causing repeated length/collision projections. Removing all collisions
was smoother but allowed the dress to pass through the calf; increasing drag or
lowering stiffness did not provide a satisfactory replacement for correct contact
geometry. These experimental changes were confined to copies.

The narrow-dress policy in `avatar_dress_contacts.py` scopes each chain to its
own thigh and calf, omits fixed opposite-pelvis supports and prunes contained
capsules. Knee-level guards use broader, offset capsule faces with orthogonal
transverse normals, fitted against all endpoints of their own spring at rest.
The larger blocked side flattens the face without moving resting vertices or
inflating the active contact face into them. The overhead fold-over barriers
remain. Every shape remains a native VRM capsule, not a mesh collider or extended
plane type. Hair groups, weights, constraints, user controls and BVT remain intact.

This policy requires a tight lower section; a close waist alone does not activate
it for a flared gown. The earlier LongDress fixture scored 0.066 on lower tightness
and retains its existing contact policy. Refit tests check native exporter output,
idempotence, pose preservation, rest clearance and disjoint hair groups.

`tools/measure_dress_contact_stability.py` reproduces the three motion/hold/return
cases in isolated Blender. Run it on the same snapshot with and without
`--rebuild-contacts`. Joint endpoint discrete acceleration measures snapping;
evaluated triangle intersections measure sampled surface crossings. Neither is a
claim of collision-free behavior at every possible pose. Detailed before/after
results are in `yukata-contact-stability.json`.

Across the three smooth trajectories, the maximum endpoint acceleration spike
fell from 97.05 to 14.57 mm/frame² (85%), and sampled intersecting triangle pairs
fell from 522 to 254 (51%). With 12-frame ramps, the corresponding reductions
were 65% and 27%. Smooth held poses settled without measured endpoint jitter.
Combined-angle and transient surface crossings remain; these results do not
establish complete collision prevention. The final native rebuild passed repeat
generation, exporter-data and rest-clearance checks, and was applied to the open
Yukata file without changing geometry, skin weights or user controls.

## Garment-specific defaults

Generation and the panel Reset action now share a measured-fit preset. Tight
full-length dresses use Skirt Follow 0.89, skirt Drag 0.55 and root stiffness 4.00;
shorter, unclassified or loose/flared skirts retain the last committed 0.55 follow,
0.40 drag and 1.60 root stiffness.
The existing lower-section tightness threshold (0.5) distinguishes a narrow dress
from one with only a tight waist. Mixed tight/loose families retain the legacy
rig-wide preset so loose layers are not forced into the narrow preset.

Both presets keep knee follow at front/side/back 0.60/0.55/0.50, thickness 1,
skirt non-root stiffness 1, and the existing hair defaults. Gravity
retains its underlying 0.025 skirt / 0.035 hair values, which the screenshot
displays rounded to 0.03 / 0.04. Ordinary initialization preserves saved edits;
explicit Reset restores the detected preset and all other panel defaults.
This preset change does not alter collider construction. Narrow-dress contacts
remain gated per full-length family; other skirts retain the prior contact path.
The contact benchmark above predates these user-selected spring defaults; its
measurements isolate the collider correction using the earlier spring settings.
