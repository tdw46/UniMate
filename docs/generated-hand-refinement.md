# Generated hand refinement

The mesh auto-rigger checks generated hand joints against the base mesh before
skinning. It transfers a better opposite-hand prediction only after a rigid fit
establishes close correspondence between the two hand surfaces and the candidate
substantially improves surface proximity. Asymmetric or ambiguous fits are rejected.
The wrist and forearm endpoint remain continuous. Skinning and rig construction
consume the same corrected prediction.

After the full-body solver, `tools/avatar_hand_binding.py` applies the existing
split-surface heat binder to small generated-hand regions. It reduces neighboring
finger and detached-nail leakage while preserving outside-region and hair weights.
No original skeleton or original weights are inputs; Preserve Volume stays off.
This code is mirrored in BVT's `third_party/unimate` integration.

The 2026-11-11 Nicole regression passes helper checks in Blender 4.2 and 5.2,
including transformed geometry and rejection of asymmetric hands. Hallway's hand
cleanup passes in Blender 5.2. The fresh full operator and three library animations
were exercised through BVT. This is not Windows or all-character qualification.
Evidence: `/Users/tylerwalker/Documents/Codex/2026-10-29/autoskin-rea-research/delivery/nicole-221/`.
