# Portable skirt contact: pose fitting and validation

## Ordered skirt chains (2026-09-25, current generation)

The production generator now keeps each skirt strip in one continuous VRM spring
chain. It upgrades older segment-per-simulation setups by joining their original
deform joints in hierarchy order and retaining only the last tip as a spring
terminal. Rest frames, parent relationships, geometry, weights, and each simulated
joint's stiffness/drag/gravity/hit radius are unchanged. Old helper bones remain
unanimated leaves; no generated mesh uses them. Hair springs are untouched.

This removes inter-simulation parent dependencies. The VRM specification defines
ordered updates within a chain but leaves execution order between branching
chains undefined. The previous split arrangement also gave each segment a
separate collision scope and could settle into a different collision basin after
reversals. The new setup rebuilds one rest-clear collider group per complete skirt
chain. Support radii remain capped by body thickness; owning-side attachment,
stationary pelvis support, and physical cross-leg fallback remain in place.
On Belle that reduces support capsules from 96 to 24, total colliders from 133 to
61, and total springs from 126 to 90, without reducing simulated skirt segments.

The fast diagnostic moves either leg through 60–65 degrees in eight 60 Hz steps,
holds for three seconds, returns in eight steps, then settles. It includes local
X, local Z, and combined X/Z/Y rotations. A second diagnostic adds five rapid
loops before the hold and return, exercising path-dependent trapped states.

Compared with the committed segmented setup, with identical spring settings:

- Mean third-difference motion proxy decreases 56.9% on the fast outward ramp
  and 64.5% on return; in the repeated loops it decreases 76.5%.
- A reproduced trapped state after repeated motion remains 408.2 mm away from
  its settled initial position with split simulations. All continuous-chain
  return errors are below 0.5 mm in this diagnostic.
- Static-hold jitter was not reproduced in either version. This does not prove
  that every interactive hold pose is stable.
- Sampled skirt/body triangle-overlap pairs increase 2.4% in the fast-ramp set
  and decrease 2.8% in the loop set. Mesh clipping remains unresolved. Some held
  endpoints still have small collider penetrations after length projection.
- Repeated installation produces no count growth; geometry, UVs, existing shape
  keys, weights and live settings remain unchanged. Native VRM export has no
  extended colliders or shared spring joints. The official three-vrm runtime
  loads and steps the export without warnings.

Results: `docs/skirt-fast-motion-results.json`; detailed comparisons, source
copies and export: `outputs/skirt_fast_motion_20260925/`. The standalone test
uses the installed BVT solver; Hallway adds no solver, frame handler, or pose
correction callback. Collider-only fitting cannot guarantee collision-free
arbitrary teleports or full cloth self-collision.

Spec: https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_springBone-1.0/README.md

## Body-sized support capsules (2026-09-25)

The thigh-length guard radius was too broad: Belle had 96 support capsules with
0.3791 m radii, while physical body colliders were roughly 0.031–0.069 m. Clearing
one assigned endpoint at rest does not bound the rest of a collider's volume.

Generation now caps support radius at the body's measured cross-sectional radius
(the existing skin-based fit, with its bone-proportion fallback). It retains the
continuous thigh axis, chain-specific groups, own-leg attachment, stationary
pelvis support, and physical cross-leg fallback. Older guards are removed through
native VRM APIs before replacement. The original stiffness, drag, follow, spring
radii, rest meshes, shape keys, weights and hair colliders remain unchanged.

Six 30-degree BVT motion/hold/return comparisons use the same live source and
settings. The bounded continuous variant reduces the mean motion jerk proxy by
4.5%, but worsens the return proxy by 12.9% and sampled triangle-overlap pairs by
9.5%. Both versions settle in the static hold, so these tests do not reproduce
ongoing hold jitter or establish that all instability is resolved. This is a
correction to oversized collision volumes, not a zero-clipping claim. Shortening
the support axes did not improve the tradeoff. Measurements and source copies:
`outputs/compact_colliders_20260925/`.

## Side ownership and stability update (2026-09-24)

The current default is **0.55 skirt follow** and **0.4 skirt drag**, in the native
joint generator, RNA property defaults and regeneration fallback. Existing user
settings remain intact until explicitly changed. Belle already used these values;
her manually chosen stiffness of 1.42 is preserved.

Each skirt chain derives its side from the ancestor follow constraint's humanoid
thigh target, with rest-position fallback for rigs without a follow helper. All
segments keep the same ownership. Its broad moving guard follows only that thigh.
The former opposite-thigh guard is transformed into Hips-local coordinates with
identical rest placement, retaining support without the opposite leg sweeping it
through the skirt. This historical revision used one thigh length for guard radius; the
2026-09-25 correction above replaces that sizing. Rest-clearance checks still apply.

Detailed body capsules remain available on the owning side. Only full-length,
central capsules remain as opposite-leg fallback, allowing actual cross-body leg
contact without broad opposite-side guards or local spherical sections acting
across the skirt. All collider changes use native VRM capsule properties; no new
solver, temporary shape keys, mesh edits or skin-weight changes are involved.

Six 30-degree motion/hold/return tests compare identical 0.55 follow, 0.4 drag and
1.42 stiffness. Opposite-side spring motion falls from 23–138 mm to zero in the
five non-crossing tests. In the deep right-leg combined pose, physical cross-leg
contact remains: maximum excursion falls from 177 mm to 57 mm. Mean third-finite-
difference motion magnitude falls 17.7%, and return motion falls 12.4%. This is an
abrupt-motion proxy measured at 60 Hz, not a visual rating. Both versions settle
without oscillation during the static hold; the reported improvement concerns
motion/contact transitions. Lowering stiffness worsened this proxy and was rejected.

Tradeoff: total triangle-pair overlaps in this diagnostic rise from 8,281 to 9,165
(10.7%). Removing the opposite-side pulling sacrifices some of its clipping
suppression. Residual combined-pose clipping remains; this change is not presented
as an improvement in every contact metric. Export and the external VRM spring
runtime pass; geometry, existing keys and skin weights remain unchanged. Repeated
installation retains 640 bones, 126 springs, 133 colliders and 49 used groups.

Evidence: `docs/skirt-side-stability-results.json`. Reproduce with
`tools/measure_skirt_stability.py` using `--variant baseline` or
`--variant pelvishybrid` on the same isolated input scene. The sections below
record the earlier development stages and their results.

## Current implementation: spring-bound skirt contact (2026-09-24)

The direct-leg weight fit was rejected: it reduced secondary motion by baking
leg attachment into skirt weights. Belle's 555 fitted vertices were restored to
their pre-fit spring/waist weights, with **zero direct leg influence**. The weight
fitter is retired, its panel entry is removed, and bundle installation refuses
fits that add direct leg weights to skirt vertices. Earlier geometry-editing
experiments below are also superseded and disabled.

`avatar_directional_contacts.install_skirt_contact_rig()` now implements the
physics-only path. It preserves every existing skin weight and all mesh data:

1. Give each original skirt segment a unique terminal helper, allowing that
   segment to have its own VRM collider group. Deform bones and their hierarchy
   are unchanged. The helper is a sibling of the next deform bone, so spring
   joints are not shared and no reciprocal collider dependency is introduced.
2. Retain existing full-leg and local fallback capsules. Add one broad offset
   capsule per thigh for each segment. Each guard presents an outward-facing
   contact surface to its segment, with its radius capped by body thickness.
3. Fit each guard to the rest endpoint including its joint hit radius and a
   height-scaled clearance margin. Lower sections can therefore have broader
   coverage without being limited by a tight waist endpoint. Every collider is
   an ordinary VRM capsule; this does not require extended-plane support.

`generate_secondary()` installs this automatically after spring/waist strip
binding. Existing rigs use **Upgrade Skirt Contacts**; **Refit Skirt Colliders**
preserves this layout. Generated helpers join the Physics bone collection.
Repeated installation removes old guards and unused groups, reuses the helpers,
and does not accumulate duplicate bones or colliders. The stiffness callback
preserves the original per-segment taper after splitting spring definitions.
BVT is still the sole solver; Hallway has no integration loop or custom runtime.

Checks preserve the original coordinate/topology/UV/shape-key hashes. Temporary
setup shape keys are permitted only if removed afterward and the original mesh
and existing keys are restored exactly; none were needed. Belle keeps all 52
existing keys on each affected mesh. No working blend file is saved implicitly.

### Measured outcome and limits

The independent 56-motion, 224-sample sweep uses both legs, follow 0 and 0.2,
local X/Z rotations up to 30 degrees, yaw and combinations. Total intersecting
triangle pairs decreased **59,233 → 50,316 (15.1%)**. Pairs absent from the
unchanged neutral contact set decreased **31,139 → 27,755 (10.9%)**. These count
triangle pairs, not penetration depth or visible area. Original neutral mesh
contacts remain, and combined rotations still clip. This is not a zero-contact
solution. An additional 28-motion sweep checks the live 0.4 follow setting.

The implementation adds 48 unweighted contact tips; the final rig has 126
springs, 133 native capsules (96 directional guards), and 49 referenced groups.
All centers remain empty. Native VRM export preserves the mesh exactly and has
no duplicate spring joints or extended collider shapes. The official three-vrm
spring loader/solver steps the exported hierarchy for 180 frames without warnings
or nonfinite transforms; this does not establish rendered mesh-contact parity.

Alternative tests included denser chains, wide capsule fans, nearest-leg guards,
surface-centered spring patches and reversed collision order. Denser chains
alone barely helped; very broad guards produced conflicting contact constraints.
Reducing stiffness to one-half or one-quarter did not materially improve the
six-motion pilot, so the existing stiffness was retained. These alternatives
are research variants, not the production defaults. Details and remaining
limitations are recorded in `docs/skirt-physics-only-results.json`.

## Result of the connection-spring experiment

Ordinary outside colliders, including L-shaped pairs of planes, impose contact
half-spaces. They do not encode a rest length, compliance, shear resistance or
equal-and-opposite tension between two moving points. An inside sphere can
approximate a maximum separation, but the two points then depend on each other.

Two implementations were tested using only VRM data and the installed BVT solver:

* Whole-chain inside capsules, each following a neighboring chain's root.
* Per-segment inside spheres, with unique terminal helper bones so spring
  definitions never share a joint. Separate body capsules fit each segment.

Neither materially reduced mesh intersections. The second exported correctly,
but the official three-vrm spring loader/manager emitted **Circular dependency
detected**. BVT builds collider transforms before its joint pass, whereas
three-vrm sorts dependencies including collider attachments. A reciprocal
network is therefore not a reliable portable cloth system. A one-way dependency
can avoid the cycle, but cannot transmit tension back around a closed skirt.

References: [VRM spring chains](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_springBone-1.0/README.md),
[extended colliders](https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_springBone_extended_collider-1.0/README.md),
[three-vrm dependency sorting](https://github.com/pixiv/three-vrm/blob/dev/packages/three-vrm-springbone/src/VRMSpringBoneManager.ts).

## Strongest tested candidate

The candidate combines three generation-time changes:

1. Fit the garment's rest surface outside actual body geometry. Check vertices,
   edge midpoints and triangle centers; moving vertices alone made the pilot
   worse because faces still crossed the body. Apply the same displacement to
   every shape key, preserving its relative deformation.
2. Bind each strip primarily to the bone segment that spans it. Blend near
   joints and a short waist attachment, rather than mixing the whole upper
   segment back toward fixed hips.
3. Use localized, chain-specific body volumes so a tight waist contact does not
   force every lower-thigh collider to shrink. Keep the shared continuous leg
   capsules as a fallback.

On Belle, 157 rest vertices move by at most 17.73 mm. The source's 234 detected
rest triangle intersections become zero. Across 56 smooth motions involving
both legs, follow 0/0.2 and rotations up to 30 degrees including yaw, sampled
intersection pairs fall from 62,561 to 19,608 (68.66%). There are four samples
per motion; only 56 of 224 candidate samples are intersection-free. This is not
68.66% less penetration depth or visible clipping. The raised-leg render still
shows substantial clipping, and spring contact residual rises from 0.785 mm
to 2.618 mm. Much of the improvement comes from correcting the starting mesh.

The exported candidate uses 177 ordinary capsules and 90 springs, with no
extended colliders. UVs, face/material assignments and all shape-key deltas
are preserved. Its exported node hierarchy and springs run for 180 frames in
three-vrm-springbone 3.5.5 / three 0.186.1 without warnings or nonfinite poses.
This test does not load/render the mesh in three-vrm, apply VRM node constraints,
or establish collision parity with BVT. The high collider count also needs
deduplication and performance testing before production adoption.

## Implemented offline pose fitter

`fit_avatar_skirt.py` prepares strip weights, a rest-clear surface and native
per-chain colliders, then runs BVT once to cache body surfaces and affine skin
matrices. The cache is checked against Blender's evaluated mesh before use.
Optimization therefore does not rerun physics for each candidate. It uses only
NumPy bundled with Blender, with no additional runtime package or solver.

The fitting objective samples skirt vertices, edge midpoints and face centers.
Radial rays from the leg axes find outward exits through the actual leg surface.
Those contacts form local half-space constraints on rest-vertex displacements.
Eight iterations of bounded projection keep recent contact planes, weld seam
vertices and limit each additional displacement to 40 mm, including a 10 mm
vertical limit. The rest preparation runs before this budget; the combined
maximum change on Belle is 49.80 mm. This is a local bounded fit, not a global
minimum-distortion solution. It does not optimize bone density or add cloth
constraints, and it has no texture-distortion objective.

The training set contains 73 sampled poses with follow 0/0.3, both legs tested
separately, local X/Z rotations up to 30 degrees and combinations with yaw.
Twenty-four held-out samples use simultaneous legs, independently generated
angles, faster motion and whole-body translation. Held-out poses are measured,
not used to choose optimizer iterations. All samples use BVT at 60 Hz.

Identical colliders are shared between chains: 32 skirt capsules supply 192
group references, rather than the earlier 177 collider records. The four
continuous leg capsules remain, plus localized thigh sections scoped to each
chain. Five existing hair/body capsules remain unchanged, giving 37 exported
colliders and 90 springs. There are no extended collider shapes, reciprocal
spring dependencies, custom playback handlers or changes to BVT's solver.

### Belle results, 2026-09-24

| Check | Before | Fitted |
| --- | ---: | ---: |
| Training triangle pairs (prepared baseline) | 7,758 | 232 |
| Held-out triangle pairs (prepared baseline) | 7,801 | 593 |
| Independent 56-motion sweep (original baseline) | 62,561 | 2,582 |
| Intersection-free samples in that sweep | 0 / 224 | 164 / 224 |
| Rest triangle intersections | 234 | 0 |

The independent sweep includes follow 0/0.2 and local X, Z, yaw and combined
rotations on both legs. The 95.87% reduction measures sampled intersecting
triangle pairs, not penetration depth or visible area. Sixty sampled poses
still contain intersections. Spring contact residual remains as high as
2.618 mm. Neither unsampled instants nor arbitrary motions are guaranteed.
Body/skirt faces are selected by semantic bone weights, so the measurements
cover those selected surfaces, not every possible garment/body material.

UVs, topology, material assignments and all relative shape-key displacements
are preserved. The fitter changes 312 rest vertices and rebinds 574 skirt
vertices. The exported spring hierarchy ran for 180 frames using the official
three-vrm-springbone 3.5.5 runtime without warnings or non-finite poses. That
second-runtime check does not render meshes or validate node-constraint/contact
parity with BVT.

### Reusing the workflow

Run with an isolated Blender process that has the VRM and BVT add-ons available:

```sh
"/path/to/blender" --background --factory-startup \
  --python tools/fit_avatar_skirt.py -- /path/to/source.blend /path/to/fit-output \
  --budget 0.04
```

The source must contain a generated Hallway skirt rig and leg-weighted body
geometry. Distances use Blender scene coordinates; choose the displacement
budget for the model's scale. `--reuse` repeats optimization from the prepared
scene and cached poses, skipping simulation. The output contains `fit.json`,
`fitted.blend`, `report.json`, `prepared.blend` and `poses.npz`.

In the Hallway rig panel choose **Apply Offline Pose Fit…** and select `fit.json`.
The installer rejects changed source geometry, weights, transforms, bones,
spring parameters or skirt colliders. Current expressions, pose and skirt
follow remain intact. It applies only affected skirt vertices and weights,
adds the displacement equally to every shape key, and recreates the exact
validated collider parameters through native VRM APIs. Selection/mode and BVT
state are restored. Reapplying the same bundle is a no-op. Complete source
snapshots provide comparison/rollback; the installer also rolls back its mesh
and collider writes if an operation fails.

Skirt thickness edits include all chain-specific capsules and retain their
scoped radius limits. Ordinary refits recognize and replace these groups,
preventing duplicate legacy capsules. Settings changes/refits mark validation
as stale in the panel. Shrinking collider thickness or changing motion/physics
can reintroduce clipping; rerun the offline fit for a new validation report.

### Evidence and live state

`outputs/skirt_pose_fit_20260924/` contains source snapshots, cached poses,
optimizer reports, the independent sweep, native VRM export and installation
checks. `before_apply.blend` preserves the live scene immediately before
installation. Belle received the fit through `hallway.apply_skirt_fit`; its
25.588% follow setting, BVT enabled state and Pose mode were preserved. The
working .blend was not saved by automation.

The earlier tension/plane experiments remain research tools and are not used
by the production fitter. `avatar_skirt_fit_io.py`, `avatar_contact_colliders.py`
and `avatar_pose_contact.py` contain the reusable installation, native collider
and offline fitting code. Research evidence remains under
`outputs/skirt_tension_research_20260924/`.


## Rest-layer regression fixed (2026-09-24)

The earlier rest claim covered leg-weighted body contact after settling BVT. It
missed the skirt's own faces and surrounding clothing. The unconstrained vertex
fit introduced 437 nonadjacent skirt self-intersections and increased vest
contacts. A closed internal waist fan also intersected the shorts. That earlier
metric was insufficient to accept the fitted rest surface.

`avatar_rest_layers.py` now reconstructs a coherent radial displacement profile
from the original pre-optimization mesh. It grows the skirt selection over
welded garment pieces, including unweighted waist vertices, and identifies
inner/outer clothing from source geometry without material-name rules. Adjacent
outer layers share the smooth displacement field. Vertex, edge and face samples
constrain clearance; a complete triangle audit rejects remaining skirt-to-body,
skirt-to-clothing and nonadjacent skirt self-intersections.

A conservative waist-fan detector identifies a pole inside its surrounding
ring, with upward closure faces intersecting the inner body. On Belle it removes
19 hidden cap faces. BMesh removes faces only: all vertex indices, weights,
shape-key deltas and surviving face UV/material assignments are preserved.
The repair updates 3,785 garment vertices; its largest displacement is 59.39 mm.
No bones, constraints, colliders or spring parameters change during this repair.

The automatic pose-fitting workflow runs this rest gate before producing an
installable bundle, then re-evaluates training and held-out contact with cached
body and garment skin matrices. `--rest-budget` explicitly sets its separate
rest-shell displacement limit (default 0.10 scene units); `--budget` retains the
raw pose optimizer's limit. An older pose cache without body skin matrices must
be rebuilt. A fresh end-to-end run passed with 183 training pairs and zero
held-out pairs after the rest repair.

Existing fits can be repaired with `repair_avatar_skirt_rest.py`, passing the
current .blend and its original pre-optimization reference .blend. Its bundle
requires independent motion validation before the Hallway installer accepts it.
The installer keeps native collider identities when their data is unchanged,
leaves weights untouched for a rest-only repair, and retains its rollback and
source-mismatch safeguards.

Verified on Belle's repaired copy and applied to live Belle:

- Zero skirt intersections in true armature REST, neutral POSE with physics off,
  and BVT at frames 1, 20 and 120.
- The same 56-motion, follow 0/0.2 sweep improves from 2,582 to 701 triangle pairs;
  clear samples increase from 164/224 to 214/224.
- At the user's current 0.4 follow, all 112 samples across 28 motions are clear.
- VRM export retains 37 standard colliders and 90 springs, with no extended
  shapes or duplicate joints; the official three-vrm spring runtime passes.
- Live follow 0.4, BVT on, Pose mode and collider UUIDs remain unchanged. The
  working .blend remains unsaved; a separate applied scene copy is available.

These are bounded checks of skirt contact, not a claim that every original
body/face/clothing component has no pre-existing intersection or that arbitrary
future motion is collision-free. Evidence is in
`outputs/skirt_rest_fix_20260924/`.
