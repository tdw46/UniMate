# Generated hair and skirt spring rigs

Final source mesh coordinates, topology, UVs and existing shape keys must remain
unchanged. Skirt vertices remain bound to springs and the waist, with no direct
leg-weight fitting. Use the physics contact regeneration workflow documented
in `docs/skirt-contact-design.md`; earlier rest-geometry repair experiments below
are superseded and their executable entry points are disabled.

`tools/avatar_springs.py` owns secondary bone generation, skin weights, body
colliders, collider groups and VRM 1 spring definitions. It does not import or
call external spring generators or physics operators. The official VRM add-on
supplies the standard RNA schema. BVT owns all spring simulation. Hallway
controls BVT through `avatar_physics_preview.py`; the demo records BVT
playback poses into an ordinary baked action.

The preparation flow accepts `--full-body --springs`. Generation runs after rest
alignment and body animation authoring, leaving secondary bones free of body
animation channels. Source skeletons and skin weights are removed before fresh
binding. VRM 1 files without fingertip/toe terminal nodes now obtain those
landmarks from the posed source surface before its rig is removed.

## Generalized attachment rules

- `avatar_apparel_weights.head_cap_vertices()` identifies cranial skin using
  semantic materials, a head/face reference and central anatomical bounds. It
  includes separate scalp islands in mixed body meshes; it does not depend on
  avatar names, vertex indices, or source skin weights. Missing material slots
  are handled without error.
- The cap, the short attachment section of each long strand, and cranial skin
  above the head attachment are **100% Head**. This is
  enforced both by ordinary apparel binding and after secondary weight creation.
  No arm, neck, torso or spring-bone influences remain in the protected region.
- Long connected hair strands are detected from hair materials and dimensions.
  Each chain now begins at the strand's actual upper attachment to the cap. A
  short fixed first bone spans the smaller of 2% of character height or 8% of
  strand height. Physics starts on **bone two**, followed by four simulated
  skinning segments and a terminal node along geometric cross-section centers.
  Fixed roots are excluded from VRM spring joints and spring baking. Head-to-hair and
  segment-to-segment weights use smoothstep transitions with zero slope at each
  endpoint. The cap is rigid; smoothing occurs in the hanging hair below the
  fixed root. Hanging strands may begin above the anatomical head joint without
  being mistaken for the rigid cap.
- Skirts must surround the pelvis at several heights and bridge the sagittal
  plane near the hem. Separate trouser legs are rejected. Long dresses labeled
  as Tops are detected from cloth below the waist, followed by the same geometric
  checks. Twelve circumferential chains blend continuously in angle and height;
  the attachment row stays fixed to Hips.
- Each skirt chain has an unweighted leg-follow parent. Matching leg rest axes
  make local rotation copying well-defined. Every panel defaults to 55% follow of its corresponding upper leg. Helpers
  copy the leg head, tail and roll explicitly; assigning a matrix to a newly
  allocated zero-length bone can discard its direction and reverse local Z.
  Skirt drag defaults to 0.40. Spring bones are separate children, so the
  constraint does not overwrite simulation rotations.
- Thighs and calves receive up to three fitted capsules per bone. Body fit uses
  freshly weighted skin and an 85th-percentile radius, bounded by the section
  length. Rounded ends stay within that section. Radius is then reduced against
  every associated spring tail (including its parent joint collision radius)
  and the weighted garment surface. A height-scaled clearance is mandatory;
  infeasible capsules are omitted instead of inflated by a minimum-radius rule.
  Skirt groups contain leg capsules, not a broad pelvis capsule. Hair groups
  have their own clearance-constrained head, neck, chest and arm capsules.
- Generated springs have no center by default, so both object and hips translation
  produce inertia. A center is assigned only through an explicit `spring_center`
  request; normal center compensation remains intact. The recording uses a 60-frame solver
  warm-up before capturing the take. Walking warms up on preceding animation
  cycles, with continuous phase and root translation at the recording boundary.

Geometry, topology, UVs and materials are unchanged. Only detected hair, skirt
and cranial regions receive new weights; weights outside them are preserved.
Repeated generation rejects before adding duplicate chains. Artist-authored
spring data is not deleted. The detector expects upright prepared humanoids;
`material_roles` can identify hair, skirt and skin or exclude ambiguous materials.
This is a bone-based spring setup, without cloth or strand self-collision.

## VRM rotation constraint export

Follow controls use enabled Copy Rotation constraints, XYZ enabled, no inversion,
no vertex group, Add mixing and Local Space for both owner and target. Each has
one constraint and no cyclic dependency. They are flagged as deform bones solely
so the official glTF/VRM export filter retains them; they receive no vertex weights
and are not spring joints.

The official import of the supplied `AvatarSample_O.vrm` contained 140 skirt/coat
bones and **no rotation constraints**. Its front/back skirt bones inherit upper-leg
motion through parenting. The generated follow controls reproduce that kind of
leg inheritance explicitly with exportable constraints.

## Earlier walking validation (generator revision 2)

Tested in Blender **5.2.2** using the user-supplied **AvatarSample O** from Downloads
(pixiv / VRoid Project). The untouched source, manifest, imported source scene and
constraint inventory are under `outputs/spring_walk_demo/`.

The fresh rig contains 24 hair chains, 12 skirt chains, 144 skinning bones,
36 terminal nodes, 12 leg-follow controls, 19 body capsules and two collider groups.
Generation and its first audit run before BVT is enabled. The test explicitly calls:

```python
bpy.ops.bvt.apply_bundled_animation(item_id="bundled_8f26b2b3938f58d32519")
bpy.ops.bvt.set_spring_simulation(enabled=True)
```

This uses the bundled Walk01 take, repeated with its actual root travel. The camera
follows and orbits the character. Actual rendering advances BVT's 60 Hz solver;
its measured rotations are recorded and baked to a separate portable scene.

Verified results:

- 246 recorded frames, 1080×1080, 30 fps, 8.2 seconds; complete H.264 decode passes.
- No skirt/leg triangle intersections in any recorded frame, including stockings
  and shoes in the leg surface check. This checks this walk, not every possible
  pose or garment-layer collision.
- All 10,158 protected cap/cranial vertices have exactly one influence: Head 1.0.
- All 12 follow constraints are accepted by the official exporter, present in the
  actual VRM JSON and reconstructed by official re-import with the required settings.
- The saved bake exactly reproduces recorded quaternion components; maximum
  secondary bone-head position error is below 1e-6 normalized scene units.
- All nine existing prepared fixtures pass preserved geometry/unrelated weights,
  normalized weights, at most four influences, neutral-surface checks and cap
  deformation checks. Independently moving arms, neck, head and hair bones keeps
  cap vertices on the rigid Head transform within 5.3e-7 units. These are numeric
  generation tests; the current walking render is Sample O.
- Procedural apparel generalization and local neck-seam regression checks pass.

The earlier Sample B demonstration remains in `outputs/spring_avatar_demo/` and
its original report is retained in `docs/avatar-spring-validation.json`. Current
revision-2 walking results are summarized in `docs/avatar-walk-validation.json`.

## Extended hair roots and three demos (generator revision 3)

The revised generator adds 24 fixed hair roots on Sample O, retaining 144 simulated
skinning bones, 36 terminal nodes, 12 leg-follow controls and 19 capsules. The cap
and fixed attachment sections remain Head-only; the free portion starts at joint
`_01` and blends smoothly down the hair. Root `_00` is parented to Head, receives
no vertex weights, and never appears in a VRM spring's joint list. All nine fixture
generation and independent cap-deformation checks pass.

`outputs/spring_revision3/` contains three 1080×1080, 30 fps tests, each with a packed
editable scene, separately baked scene, VRM export and validation reports:

- `walk/sample_o_walk.mp4`: 246 frames of the same Walk01 animation.
- `library/sample_o_library.mp4`: all 181 frames of **Idle07_SpinningJump**, applied
  by `bpy.ops.bvt.apply_user_library_animation(item_id="user_library_552db934798bb8907ece")`.
- `drag/sample_o_drag.mp4`: 240 frames of armature-object translation and settling.
  Only object location is animated in the input action; all body pose matrices
  remain fixed. The same generated spring setup is used for every demonstration.

The spin/jump and dragging motions expose remaining garment intersections and
twisting. They are stress-test recordings, not collision-free cloth simulations.
`surface_contact.json` records actual skirt/leg triangle contacts for every frame;
it does not measure garment self-collision. See `docs/avatar-spring-revision3-validation.json`
for current contact counts, export round trips, saved-bake checks and video checks.

## Run and inspect

```sh
BLENDER='/Applications/Blender 5.2.2.app/Contents/MacOS/Blender'
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  -P tools/demo_avatar_springs.py -- \
  --source "$PWD/outputs/spring_walk_demo/prepared/avatars/avatarsample_o/02_fresh_rig.blend" \
  --output "$PWD/outputs/spring_revision3/walk" --name sample_o_walk \
  --bundled-animation bundled_8f26b2b3938f58d32519 --frames 246 --size 1080
"$BLENDER" -b --factory-startup --python-exit-code 1 -P tools/validate_avatar_walk.py -- \
  --output "$PWD/outputs/spring_revision3/walk"
"$BLENDER" -b --factory-startup --python-exit-code 1 -P tools/validate_avatar_contacts.py -- \
  "$PWD/outputs/spring_revision3/walk" --report-only
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  -P tools/validate_avatar_springs.py -- --generation-only \
  --output "$PWD/outputs/spring_revision3"
```

For the library take, replace `--bundled-animation ... --frames 246` with
`--user-library-animation user_library_552db934798bb8907ece --frames 0`; zero preserves
the entire source range. For the object-drag demonstration use
`--drag-armature --frames 240`. Supply separate `--output` and `--name` values.
The default generation-only validator checks nine prepared fixtures. Run export
validation separately for each demo with its matching output folder and name.

The final folder contains `sample_o_walk.mp4`, an editable `sample_o_walk.blend`
with simulation enabled, a portable `sample_o_walk_baked.blend` with simulation
disabled, and `sample_o_walk.vrm` with exported constraints/colliders/springs.
Generation, runtime, export and per-frame contact audits accompany those files.
The scripts use the canonical shared extensions folder, do not save preferences,
and do not change a live Blender session. Ordinary GLB export does not preserve
VRM spring behavior; use VRM or the baked action for the intended playback path.

### Rebinding a neutral MMD character without losing facial shapes

`tools/rebuild_mmd_character.py` adapts English MMD semantic joint names into
new, symmetric humanoid bones, including fingers. The source supplies rest
landmarks only: its armature, vertex groups, control bones and spring settings
are not reused. The adapter requires an upright neutral pose and matching mesh
and rig transforms; it rejects unsupported input before deleting the source.
It supports A-pose geometry without changing the mesh rest positions.

Fresh heat is solved on a temporary welded/capped surface. Detached vertices
without a heat solution transfer barycentric weights from nearby solved
triangles, with a distance bound and rejection when there is no valid source.
The original topology, UVs and all shape-key coordinates are hash checked.
The existing head-cap and maximum-two-ring neck-seam rules then apply.

Short hanging hair uses the shared `hair_free_top()` geometry rule in
`avatar_apparel_weights.py`, also consumed by `avatar_springs.py`. Short strands
must be vertically elongated and extend sufficiently below the crown; crown
vertices remain Head-only. The first short root bone remains outside physics.
Existing long-hair behavior is preserved. Secondary bones, weights, capsules
and VRM-compatible leg-follow constraints are generated by our code.
Simulation is provided by the installed BVT addon.

`tools/install_rebuilt_character.py` installs a validated result on the existing
live mesh objects. It verifies source hashes, archives the original rig and
independent mesh copies in a hidden comparison collection, replaces weights
and armature modifiers, remaps imported dependencies, and enables BVT physics preview.
It preserves unrelated objects and never saves over the active blend file.

The September 24 Belle fixture passed preservation checks for 18 meshes,
23,167 vertices and 52 shape-key blocks per mesh. Its generated setup has
12 skirt chains, 78 short-hair chains and 19 fitted capsules. A 180-substep
runtime test produced finite hair/skirt motion. All nine prior full-body
fixtures passed geometry, normalized weight, fixed-root and posed head-cap
regression checks. These are execution and deformation-invariant checks;
they do not establish collision-free motion. Evidence is under
`outputs/belle_live_20260924/`.


## Revision 4: rest-clear colliders

`avatar_colliders.plan_colliders()` derives a complete fitting plan before any
scene mutation. `rebuild_colliders()` replaces only our collider groups; unrelated
artist groups, bones, weights, geometry and body poses remain untouched.
`rest_contacts()` audits the stored VRM capsule properties against the actual
associated joint endpoints. The joint's `hit_radius` is included in every test.
The generator and all nine regression fixtures now require zero rest overlaps.

All spring integration, center compensation and collision response now come
from the installed BVT addon. The project contains no independent solver.
The **Hallway** sidebar uses `bpy.ops.bvt.set_spring_simulation`; rig generation
continues to use our own code and the official VRM RNA schema.

Skirt physics now uses continuous, ordered VRM chains with body-sized contact
capsules. Existing segmented rigs are upgraded by **Refit Skirt Colliders**;
rest geometry, weights, bone frames and spring tuning are preserved. See
`docs/skirt-contact-design.md` for fast-motion evidence and remaining limitations.

Each spring group has a saved **Non-root Stiffness** slider below **Root
Stiffness** (`hallway_rig.spring_groups[...].non_root_stiffness`). It multiplies
the existing taper below each original simulated chain root: `1.0` preserves
the current result, `0.25` uses one quarter of the distal stiffness, and `0.0`
removes distal stiffness. Root stiffness is unchanged by this slider. Root
classification follows the bone hierarchy across separate VRM contact segments;
the first joint of every segment is not treated as a new chain root. Skirt and
hair controls update their own native VRM joint values immediately, without
resetting BVT simulation or changing geometry, weights, or colliders.

`demo_avatar_springs.py` asks BVT to prepare its playback/render pose cache,
records those poses, and renders a baked take. Headless numerical checks call
BVT's duration-step API directly, with a capability check and a clear error
when the installed BVT version does not expose the required entry point.
No integration, collision or timing math is duplicated here. The normal
installed Hallway addon has no simulation timer or frame handler.

Animation demo options accept local `.vrma` files imported by
`bpy.ops.import_scene.vrma`. BVT must be enabled for simulation; geometry,
weights, constraints, colliders and spring generation remain independent of it.
The measurements below describe the historical revision-4 test run, not a new
BVT result. Current BVT checks are recorded separately under
`outputs/bvt_solver_only_20260924/`.

On the Belle fixture, the old capsules had 51 rest overlaps, with a deepest
penetration of 0.09066 model units. Revision 4 has 17 capsules (12 leg, 5 hair-body),
zero rest overlaps and a minimum clearance of 0.002376. In a 300-step stationary
comparison, collider-enabled and collider-disabled endpoints match exactly,
including gravity. A separate alternating leg sweep activates skirt collisions;
its maximum collider-driven endpoint difference is 0.08663. All nine full-body
fixtures pass rest-clearance, unchanged-geometry, normalized-weight and rigid-cap
checks. This verifies rest clearance and moving-leg response, not cloth
self-collision or clipping elimination in every possible pose. Evidence:
`outputs/collider_clearance_20260924/`.


## Skirt follow revision 2: uniform 90% and matching local axes

`avatar_skirt_follow.py` owns the shared helper setup and live migration. Each
helper has the leg's head, tail, roll and parent, so local XYZ rotations have the
same anatomical direction on both bones. Existing spring bones and mesh weights
are preserved. All twelve constraints now use 1.0 influence with no angular falloff,
XYZ enabled, no inversion, Add mixing, Local owner/target spaces, no vertex group
and no dependency cycle. The owned constraint is first in its stack.

Generated rigs use their authored rest pose for VRM export instead of automatic
posing. Automatic export posing can straighten a humanoid thigh independently
of its non-humanoid helper, invalidating their matching rest frames. The rest
pose setting preserves the current authored skeleton; it does not convert an
A-pose model into a T-pose model. Explicit artist-selected custom export poses
are not overwritten by the migration.

`validate_skirt_follow.py` checks both signs of local X/Y/Z with neutral and
rotated hips, arbitrary rolled fresh helper bones, unchanged skinning, official
export eligibility, serialized 0.9 weights, and actual VRM export/reimport.
The Belle fixture retained all 12 constraints; maximum matrix error was
6.56e-7 before export and 1.20e-6 after import. All nine full-body fixtures also
pass the axis checks. Evidence: `outputs/skirt_follow_90_20260924/`.

Follow strength was subsequently raised to 100% in the shared `INFLUENCE` default and all twelve live constraints; the axis setup and VRM constraint flags are unchanged.


## Explicit spring centers only

`generate_secondary(..., spring_center=None)` leaves the VRM spring center unset.
An explicit bone name opts into translation compensation and is validated before
any generation mutation. `set_generated_spring_center(rig, center=None)` clears
centers on existing generated springs; passing a bone name assigns it explicitly.
Collider attachment bones are independent and remain unchanged.

The runtime continues to honor assigned ancestor centers. The old diagnostic
`unimate_spring_world_inertia` bypass was removed rather than weakening center
semantics. Mixamo renaming accepts the intentionally empty optional center nodes.
`validate_spring_centers.py` checks both whole-object and hips translation against
matching stationary BVT runs: springs react without a center, while an explicit
Hips center compensates hips-joint translation. Whole-object behavior remains
BVT-owned and is recorded separately. The live Belle setup has all 90 spring
centers cleared.


## VRM collider API and coverage revision 5

`avatar_vrm_colliders.py` owns the integration boundary. It creates colliders
and groups through the official VRM `add_collider()` / `add_collider_group()`
methods, falling back to the corresponding VRM operators on older schemas.
Removal uses official operators so UUID references and active indices stay
consistent. Capsule type, bone attachment, offset, tail and radius are written
through VRM RNA. No collider mesh is created or deformed. Capsule generation
requires VRM 1; the visibility helper also understands VRM 0 sphere references.

Leg capsules now overlap adjacent fitted sections instead of being shortened
inside every third of a limb. Radius limits still include spring hit radii and
rest garment clearance. A skirt-only refit preserves hair collider objects,
artist groups and body pose, and rejects groups shared with artist springs.
The Hallway panel exposes refit and display controls. The MMD installer disables
legacy rigid-body mesh viewport display so ordinary Unhide does not make the
inactive comparison volumes look like the generated VRM colliders.

On the posed Belle snapshot, all 17 active colliders were already VRM empties
with correct bone parents. The 395 stationary colored mesh volumes were legacy
MMD rigid bodies. The new fit keeps 12 skirt capsules and 5 hair-body capsules.
`validate_vrm_colliders.py` tests local X/Z leg rotations, armature translation,
zero spring/collider overlap at rest, and real VRM export/re-import including
capsule shape parameters, bone references and collider-group references.

The overlapping fit modestly reduces triangle intersections in the raised-leg
fixture; it does not eliminate skirt penetration. The rest mesh itself contains
skirt/pants and skirt/skin intersections, and VRM collision acts on spring joint
endpoints rather than every skinned cloth triangle. Enlarging capsules globally
would reintroduce the previously fixed rest-pose displacement. Further cloth
clearance work must address the rest garment surface and panel skinning/chain
coverage while retaining standard exported VRM shapes; editable collider meshes
are not a solution. Evidence: `outputs/vrm_collider_fit_20260924/`.

## Continuous leg capsules, revision 6

Skirt colliders now use one standard VRM capsule per upper/lower leg, with its
center segment exactly on the bone head/tail. The radius remains fitted to body
samples and limited by rest spring/garment clearance. Hair fitting is unchanged.
Belle has four skirt capsules and five hair/body capsules; refitting removes
the previous section records and their display objects through the VRM API.

`diagnose_skirt_collision.py` compares the current setup, continuous capsules,
combined sections, coaxial sections, per-chain capsules, and sector-specific
extended planes. Run with `--full --variant generated` for both upper legs,
0% and 20% follow, and eight combinations of +/-30 degrees about local X/Z.
Each case smoothly moves out and back over 91 frames at 60 Hz using BVT only.
The optional `--bvt-runtime-only` flag loads the installed, unmodified BVT solver
without its unrelated UI modules for isolated testing; it is never used live.

Across 32 cases, exact spring-segment crossings dropped from 4.14 mm to zero.
The collision spheres retain up to 0.79 mm of penetration after BVT's length
projection, compared with 6.21 mm before. These are different metrics: zero
segment crossings does not mean perfectly separated collision spheres or cloth.
No spring/collider overlap was introduced at rest. Nine-avatar generation,
repeated-refit object counts, attachment movement, and VRM export/re-import pass.

Rendered front/back diagnostics still show skirt/thigh clipping. The source has
234 skirt/body triangle intersections at rest; sampled intersections over the
matrix were 35,595 before and 35,707 after. This collider change improves spring
containment, **not** overall cloth-surface clearance. Per-chain volumes reduced
that sample count to 34,800 but required substantially more collider references
and did not eliminate clipping. Sector planes increased contact error and can
fight fixed roots when legs rotate, so they are not generated. A collision-free
skinned surface will require separate garment/weight/chain work or solver-level
cloth contact beyond these exported endpoint colliders. BVT was not modified.

Evidence: `docs/skirt-collision-sweep.json` and ignored artifacts under
`outputs/skirt_collision_sweep_20260924/`.

## Capsule display and chain-plane yaw experiment

Hallway resolves its sidebar target through the VRM add-on's `current_armature`,
the same resolver used by BVT. The panel stays available when a mesh or collider
is active; configuration operators still require a Hallway-generated target.
The selected target is shown explicitly, including a notice for other rigs.

VRM represents a capsule using two sphere-shaped empty displays. The collider
already spans between them, but those empties do not draw its cylindrical part.
`avatar_collider_overlay.py` adds read-only capsule wires from the official VRM
radius and display endpoints. It respects collection visibility, viewport
overlays and the collider's in-front flag. It adds no scene objects, export
geometry, solver or timer. Register/unregister owns a single draw handler.

The `planes_centroid_body` diagnostic variant fits one outward plane for each
skirt chain, using the leg-axis-to-chain-centroid direction as its normal. Each
plane has a group referenced by exactly one chain. All chains retain the four
shared leg capsules, including during yaw. Planes use the VRM extended-collider
RNA, setting offset before normal because the offset setter resets rotation.
They require a runtime that supports the VRM extended-collider extension.
An isolated VRM export preserves all 21 colliders, including 12 extended planes;
setup assertions verify one plane per skirt chain plus all four shared leg
capsules. This export check does not establish support in other applications.

`--full --yaw` tests 56 smooth motions: both upper legs, follow 0 and 0.2, X/Z
rotations, local-Y yaw and combinations, each up to 30 degrees. BVT alone runs
the simulation. At four samples per motion, capsules alone produce 62,561
skirt/leg triangle pairs; adding these planes produces 59,365, about 5.1% fewer.
This is an intersection-count diagnostic, not a penetration depth or area.
Both have 234 intersecting triangle pairs at rest and zero rest spring overlap.

The planes leave up to 37.15 mm of spring collision-sphere penetration versus
0.785 mm for capsules alone. An anchored segment can lack up to 35.27 mm of
reach to get onto its plane's allowed side. Front/back renders still show cloth
penetration. The candidate therefore remains experimental and is not generated
by default or applied to live Belle. Capsules remain the live collider setup.
The thigh fit is constrained to roughly 31 mm radius by rest clearance despite
body samples suggesting roughly 69 mm; visible capsule spans make that coverage
limit apparent. Enlarging them without changing rest geometry/chain placement
would restore the unwanted displacement at rest.

Evidence: `docs/skirt-plane-yaw-sweep.json`; candidate scenes, pose snapshots,
full measurements and renders are under
`outputs/collider_display_audit_20260924/`.


### Offline skirt pose fitting

For portable contact optimization, use `tools/fit_avatar_skirt.py`, then the
Hallway **Apply Offline Pose Fit…** operator. It caches BVT poses, bounds rest
mesh edits, shares equivalent native capsules between chains, and preserves
relative shape keys. No extra runtime solver is installed. See
[design, measured coverage and limitations](skirt-contact-design.md). The
ordinary generator still produces its initial rest fit; pose optimization is
an explicit offline refinement because it alters the skirt's rest silhouette.
