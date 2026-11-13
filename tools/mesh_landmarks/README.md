# Mesh-only humanoid rigging

The Blender add-on now has an **Automatic Rigging** panel that works without an
existing armature. Prepare the local engine, download the model, select the entire
upright character in a neutral pose facing front (-Y), then generate a new scene.
The original scene is retained. Review joint placement and bending before export.

This feature uses Make-It-Animatable's two-stage pretrained joint predictor.
UniMate's motion model is not used to invent skeletal landmarks. The landmark-prediction input file
contains only triangle vertices and indices. Original bones, weights, naming,
humanoid mappings and landmark measurements are excluded. Tests erase the original
rig before invoking `hallway.generate_mesh_rig`. No reference fallback is allowed.

`manifest.json` pins upstream source and weights. The inference port preserves
checkpoint parameter layout, replaces native farthest-point sampling with a
portable greedy equivalent, and uses evaluation-only dropout. Standard Mixamo
joint names/parent topology define the output schema; they contain no coordinates
from a target rig. Training data/template FBX downloads are not required.

Setup and model download are separate cancellable tasks. Dependencies are installed
in a private venv; weights are verified before use. CPU is the default. Metal is
optional when detected. Windows/Linux, Intel Mac and older-Python wheel routes are
explicitly guarded but have not been executed in this Mac qualification.

Use `python -m mesh_landmarks.test_contract` in the isolated engine environment
for the portable operation/input-contract tests. The BVT vendor contains identical
predictor/job sources; its first-party adapter owns BVT names, UI, localization and
physics controls. Legacy `rebuild_mmd_character` reference utilities are not called
with an original rig by the new generator.

The current validation avatar differs from its original at major joints by 2.53 cm
on average, with a 7.66 cm worst shoulder. These are estimates; do not present the
result as ground-truth accuracy or all-pose deformation proof. Unusual proportions,
loose clothing and non-neutral poses need artist review.

The sibling-character suite now covers Anby, Belle, Grace and Nicole in explicit
rest pose, with original rigs/weights removed before generation. Shared backend
repairs cover uneven waistbands, partial heat failures, high heels, detached
surface details and multilingual material roles. The public operator was checked
independently on Nicole (221 bones, 28 hair chains). These operational checks do
not establish exact landmark accuracy or all-pose quality.

## Bundled skinning solver

The Generate operator now uses **Hallway Solver** after mesh-only joint prediction.
CPU is the default; GPU and the previous Blender Weighting route are explicit
choices. See [Hallway Solver](../SKINNING_SOLVER.md) for usage, preserved cleanup,
Blender 4.2/5.2 checks and native Mac/Windows qualification boundaries. The earlier
measurements above concern the predictor and previous binding route.
