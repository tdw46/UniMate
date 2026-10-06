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
