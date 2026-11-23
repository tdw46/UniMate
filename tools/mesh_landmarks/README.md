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

## Fused arms and anonymous generated meshes

**Separate Fused Arms** is an optional topology-changing preparation on disposable
copies. It uses predicted joints and cross-sectional saddles to open narrow
arm/torso bridges and tessellate closed cut surfaces before skinning. Curled
hand contacts use a concavity-weighted face graph cut and separately capped banks,
so the cuff and curled fingertips stay with the hand instead of a geometric
proximity boundary cutting through them. Coincident shoulder rim weights share a
short blend so the repaired arm remains continuous at its attachment. Confirmed
fused arms also receive a bounded shoulder-cap pivot fit before preparation;
the slit ends just below that socket. Unfused arms retain their existing pivots. A bounded
rest-space shoulder filter then averages neighboring weights, treating coincident
seam vertices as one node and limiting the result to four influences. Opposite
cut banks share nodes only at the socket attachment; below it, each bank retains
its arm or torso ownership, preventing the filter from re-welding the underarm. The filter
ends above the elbow and leaves the hand/finger domain unchanged. Existing
arm gaps are left alone. Open/nonmanifold surfaces and shape-key meshes are not
cut. Cancellation removes prepared copies, and the original selection is retained.
Generated bank attributes keep the volumetric hand field from crossing back into
the body through a narrow cut. This does not reconstruct missing anatomy; fused
hands and severely ambiguous contacts still need deformation review.

Anonymous materials can use a conservative thin-panel fallback for hanging cloth.
It checks opposed surfaces, connected vertical extent and anatomical exclusion
regions. Named/semantic material workflows keep their existing route. Accepted
panels use native VRM springs and fitted colliders, simulated by BVT. This fallback
does not claim to identify hair, skin or every garment in a monolithic mesh.

Short pelvis panels receive mostly underlying body support, with up to 12%
retained spring weight in the short-panel core. The classifier compares local
connected hems against the inferred hip/knee span; angular strips distinguish
a short shirt front from an attached long coat. It samples freshly generated
hips/thigh weights on nearby non-cloth surfaces, with a bounded anatomical
fallback when no donor surface exists. A smooth local handoff retains the
existing long-cloth solution. This changes weights only, preserves native VRM
spring/collider definitions, and uses no imported rig or material-name exception.

Background regression fixtures: `mesh_landmarks/test_arm_separation.py`,
`test_anonymous_cloth.py`, `test_hand_domain.py`, and
`test_shoulder_pelvis_binding.py` under `tools/`. The first
checks closed caps, source preservation, translation/scale invariance, and the
no-op case for already separated arms, and inward-curled fingertip ownership; the others check admission and ownership.

For the isolated-profile arm fixture, set `HALLWAY_TEST_RUNTIME` to the prepared
rigging-engine data directory. The public operator passes its configured runtime
directly. Missing runtimes fail before changing the source and clean up copies.

Repaired down-arm shoulders use a local heat solve on the uncut reference
surface. The public operator passes the original meshes separately from the
prepared copies; the builder supplies the explicit world-to-rig transform.
The cap blend ends by 45% of upper-arm length. A separate half-strength
underside blend extends to 60%, followed by a bounded upper-rim handoff ending
at 62%. The two heat blends share one disposable uncut-surface solve. Distal
cut banks, fingers, secondary surfaces and lower-body ownership stay protected. Modified rows are normalized with at most four influences.
This stage changes neither bone rest frames nor mesh coordinates, topology,
UVs, materials, or normals. Unrepaired meshes are an exact no-op.

The upper split rims share smoothly matched weights with a short intrinsic
smoothing band on each side. Protected hand/leg rows cannot seed that diffusion.
The operation is idempotent on an already updated generated copy. The accepted
combination needs no vertex displacement, added bones, or new modifiers.
`test_armpit_attachment.py` checks rim continuity, no finger leakage, unchanged
cap/distal weights, repeat-call behavior, four weights and exact mesh geometry.

The additional clavicle-weight ceiling and edge-error pivot optimization were
removed from generation after pose review showed worse shoulder pinching.
The generated-fill rounding experiment is also disabled in the generation
path: moving only added fill vertices left a ridge at the fixed cut rim.
`hallway_generated_fill` provenance remains available for reconstruction work.
Do not silently expand a fill-only repair onto surviving source vertices.

`test_shoulder_heat.py` checks continuity across coincident attachment banks,
protected-domain preservation, reference-space conversion, four-weight behavior,
no bone/geometry changes, invalid correspondence rejection, and temporary-data
cleanup on both success and solver failure. Run it with the repository's
existing background Blender test runner. Model pose comparisons are still
required; these invariants alone do not prove an anatomically correct contour.
