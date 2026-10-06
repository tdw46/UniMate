# Long dress rigging

The secondary-rig generator classifies each garment using its hem and the
humanoid knee and ankle heights. This uses the same normalized, Z-up rest
coordinates as the existing skirt planner.

- Chains crossing the knee receive a joint at knee height.
- Hems at or above mid-calf keep upper-leg follow. Below-knee sections also
  receive calf contact guards.
- Hems below mid-calf receive separate upper and lower spring sections, with
  at least two segments in each. An unweighted helper at the knee adds calf
  rotation to the inherited upper-section motion. Each joint belongs to only
  one spring; constraints never drive a simulated joint directly.

The helper shares the calf's rest axes and uses the existing portable local
XYZ Copy Rotation setup. Hallway exposes Skirt Follow (0.55) and directional knee controls: Front Knee
Follow (1.0), Side Knee Follow (0.0), and Back Knee Follow (0.15). Skirt drag
defaults to 0.4.
The lower section retains the existing physics taper, follows its own calf
contact guards, and shares its upper section's ceiling. Hair collider groups
remain separate. Cloth weights still follow the original full spring chain;
the helper has no vertex weights. Source vertices and shape keys are immutable.

`avatar_dress.py` owns height classification and spring splitting.
`avatar_springs.py` integrates it with generation. Binding, follow upgrades,
directional contacts and ceiling generation understand the split sections.

For long dresses, the owning upper-thigh guard uses a flatter capsule support
(four times body radius, bounded by bone length). Its front is refitted to all
rest endpoints; its enlarged back side belongs only to that upper dress spring.
Calf guards, fixed pelvis supports, shorter skirts and hair keep their existing
sizes and scopes. This remains an ordinary VRM capsule, without extended planes
or a custom solver. The user follow value remains unchanged.

`avatar_leg_binding.py` isolates fresh left/right anatomical leg heat domains
and samples footwear weights from the matching skin surface. This prevents the
opposite leg from seeding enclosed skin underneath a dress. A torso transition
preserves the upper body, and distance to arm landmarks excludes lowered hands.
Fallback dress detection validates each cloth material separately so a tail
accessory cannot inherit another material's skirt classification.

## Verification

The 2026-10-01 `longdress.vrm` rebuild produced 12 knee transitions and 24 dress
spring sections. Anby and LongerSkirt correctly did not activate calf follow.
The report in `longdress-validation.json` records geometry preservation,
normalized weights, live control updates, knee continuity, collider lifecycle,
and VRM constraint export/reimport checks on Blender 5.2.2.

The leg regression found opposite-leg influence on 1,430 of 1,860 tested lower
skin/shoe vertices in the initial build, reaching 88.17%. The corrected build
has zero such influence and less than 0.000061 mm opposite-leg movement in the
unilateral motion check. Geometry, UVs, shape keys and packed base texture bytes
match the source. Hand texture display was deferred at the user's request.

The captured roughly 90-degree sideways leg raise initially produced 84 held
triangle intersection pairs. The targeted upper support reduces that to 16
after either a smooth or quick raise, with zero measured settled oscillation.
The quick-ramp sample decreases from 197 to 70 pairs. Return error stays below
0.64 mm. Wider guards everywhere and extra spring joints were less effective
and are not production defaults. Reinstallation leaves object/collider counts
stable, and the rest audit remains clear.

There is still clipping: the separate knee and combined hip/knee holds have 72
and 36 sampled pairs respectively, unchanged by the targeted collider tuning.
A rapid-reversal sample changes from 42 to 45 pairs, then clears. These counts
are diagnostic triangle pairs, not visual severity estimates or a guarantee
against clipping. Numerical checks do not replace visual review.

Run pure classification tests with:

```sh
python3 tools/test_avatar_dress_profile.py
```

Blender diagnostic entry points are `test_avatar_dress_rig.py`, `test_avatar_leg_binding.py`,
`validate_skirt_follow.py`, `test_avatar_skirt_ceiling.py`, and
`measure_dress_motion.py` / `measure_skirt_pose.py`. Their command-line arguments are documented by their
entry points; run them on isolated copies, not a live working scene.

## Combined-pose contact update

Revision 7 keeps the upper and lower sections of the narrow center strip in
contact with both moving legs. A strip qualifies when its original attachment
is within one quarter of the upper-leg head separation from the pelvis center.
The lower section resolves its upper attachment before classification, so a
flared hem cannot accidentally change side ownership. Only long dresses with
split knee sections use this exception; the other panels retain their own leg
and fixed pelvis support. The test fixture gains two native calf capsules,
with 107 total colliders and no hair-group overlap.

The user-captured combined hip/knee pose previously had 183–185 held triangle
intersection pairs. Revision 7 with this dress's drag tuned to 0.2 has zero at
all 21 sampled times across smooth, quick and repeated-cycle tests. Repeated
motion RMS third difference decreases from 1.358 to 0.646 mm, and its
peak from 46.08 to 18.68 mm. These are discrete third frame differences sampled at 60 Hz,
not continuous jerk in mm/s³. Smooth-motion peak decreases from 27.32 to 6.37 mm.
The general saved-setting default remains 0.4; no hidden drag multiplier or
solver change is introduced.

Very fast motion is still imperfect: the eight-frame raise has RMS third
difference 3.36 mm versus 2.87 mm before, despite its slightly smaller peak and
zero sampled clipping. The earlier sideways 90-degree hold still has 16 pairs;
its quick-raise sample improves to 57 from 70. Separate knee and combined
regressions remain at 72 and 36 held pairs. Rapid reversal has 40 pairs at one
sample then clears. Do not interpret this improvement as all-angle clearance.

Rest endpoint clearance remains at least 2.587 mm. Repeated installation keeps
107 colliders / 37 groups / 318 objects in the isolated generated scene.
`test_avatar_dress_contacts.py` verifies group ownership, hair isolation, native
capsule export, unchanged vertices/weights/pose, and cleanup on regeneration.

The applied live scene was re-read with BVT enabled: revision 7, 107 colliders,
121 springs, preserved user leg pose, and zero measured dress/leg triangle
intersections. Saved and left open as `LongDress_Hallway_Stable.blend`.

## Center-chain trapping correction

Revision 8 replaces the two upper-leg directional guards of each shared center
section with transverse capsules. Their axes cross the panel's width at the
middle of the thigh, perpendicular to the leg axis and the fitted outward
normal. Half-span is the greater of hip separation or twice measured body
radius. Radius retains the existing broad upper support: four body radii,
bounded by bone length. The complete capsule is refitted against rest endpoints.
Longitudinal body fallback capsules, calf supports, side-panel contacts,
ceilings, skinning, follow and spring parameters remain unchanged.

The highlighted front chain previously settled across the adjacent strip: its
knee endpoint was x=-0.283 m while its neighbors were x=-0.074 and x=0.241 m.
With transverse support it settles at x=0.076 m, between those neighbors, and
the live user pose still measures zero dress/leg triangle intersections. The
sideways collision escape no longer provides the same low-energy trap.

A smaller transverse radius passed smooth approaches but failed when physics
started directly in the posed rig. For this reason `measure_skirt_pose.py` now
also tests `direct` startup, and production retains the deeper support. This
case complements smooth, quick and cyclic approaches; checking only a smooth
ramp missed an alternate settling state.

For the captured pose, revision 8 keeps all sampled intersections at zero.
Smooth-motion RMS third difference is 0.518 mm (previously 0.518), cyclic is
0.611 mm (previously 0.646), and the quick eight-frame raise is 3.663 mm
(previously 3.363). The quick case remains a limitation; the correction is not
a claim of universal jitter elimination. No new colliders or solver behavior
were added. The live file retains drag 0.2, follow 0.55, the user's selected
chain and leg pose, and BVT physics enabled.

## Independent knee follow

Lower dress sections now use a saved `Skirt Knee Follow` group, default 0.15,
independent of the upper `Skirt Follow` group. Both appear in the existing
Hallway configuration panel and update their own constraints immediately.

The shared center strips use two serial, unweighted helpers at the existing
knee junction. Each copies a different calf using one local XYZ ADD rotation
constraint, at half the group's influence (0.075 at the default). This provides
a response to either knee without doubling the total when both knees bend.
Side strips retain one source at 0.15. The original upper support, spring
joints, collider shapes/groups, rest geometry and skin weights are unchanged.
Regeneration and follow upgrades preserve the separate setting and normalized
shares. `avatar_dress.upgrade_knee_follow` also upgrades existing long dresses
idempotently and preserves existing bone rest frames and selection.

The live highlighted strip previously copied the stationary right calf while
the left calf was bent. After the update its lower section bends about 6.66
degrees relative to the upper segment in the captured pose. The rig has two
additional helpers, 732 bones, 26 portable follow constraints, and the same 107
colliders / 121 springs. Current-pose smooth, quick, cyclic and direct-start
samples still have zero measured intersections; live read-back is also zero.

Independent slider updates and responses to both calf sources pass. VRM
export/reimport retains all 26 constraints, with maximum reimport transform
error 1.79e-6. The separate knee-hold regression improves from 72 to 46 held
intersection pairs, though its early sample changes from 0 to 19. Combined
hold remains at 36; the rapid-reversal sample drops from 40 to 36. Other-angle
clipping and very fast motion remain limitations. The live file keeps upper
follow 0.55, drag 0.2, and the user's pose and selection.

## Directional knee-follow profile

Knee-follow revision 2 replaces the uniform 15% lower-leg setting with three
saved, live Hallway controls: Front Knee Follow 100%, Side Knee Follow 0%, and
Back Knee Follow 15%. The old `Skirt Knee` storage key is retained for the front
control; `Skirt Knee Side` and `Skirt Knee Back` store the other two. Migration
runs once, then regeneration preserves edited settings.

Each original waist attachment is projected onto the horizontal humanoid
frame. The anatomical left-to-right landmarks and the generator's Z-up axis
resolve front without avatar or chain-name rules. If c is the signed front
cosine, front weight is max(c,0)^2, back weight is max(-c,0)^2, and side weight
is the remaining fraction. Their weighted control values set the knee-follow
strength. This continuous rule is independent of chain count and object scale.
For the twelve-chain fixture it produces 100%, 75%, 25%, 0% toward either side,
then 3.75%, 11.25%, 15% toward the back. Center chains retain their two normalized
calf sources, each taking half the combined strength.

The selected lower section bends 35.34 degrees in the current live pose, up
from 6.66 degrees with uniform 15% follow. No gravity increase was needed for
this result. Geometry, skin weights, colliders, upper follow, stiffness and
drag are unchanged. All 28 sampled current-pose times (smooth, quick, cyclic
and direct startup) remain free of measured dress/leg intersections; live
read-back also remains zero. The existing separate knee-hold regression has
46 held pairs and 29 at its early sample (previously 19); combined hold remains
36 and rapid reversal 36. These other-angle limits are not claimed fixed.

All three controls update native constraint weights immediately, preserve the
upper follow value, and survive regeneration. Five pure profile tests and the
Blender rig/binding checks pass. All 26 rotation constraints survive native VRM
export/reimport, including zero-influence side constraints. The open file keeps
the full selected front chain and the user's pose for review.


## Bilateral chains replace center blending

Knee-follow revision 3 supersedes the center-source averaging above. Generated
skirts use an even number of mirrored sectors whose front and rear axes fall
between chains. There is no sagittal center chain. Each upper helper follows
one thigh and each lower helper follows the same-side calf, with exactly one
native XYZ local/local ADD Copy Rotation constraint per helper. The obsolete
serial blending helpers are removed by `upgrade_bilateral_skirt`.

The nearest front pair reaches Front Knee Follow (100% by default). Angular
weights taper toward the sides; the nearest rear pair reaches Back Knee Follow
(5% by default). Source influences are never divided between knees, and bending
both knees cannot double either chain's constraint rotation. The existing
live Hallway controls still apply immediately.

Fresh generation uses the paired layout. Existing generated rigs can explicitly
migrate with `upgrade_bilateral_skirt`, followed by `install_skirt_contact_rig`.
Migration refits skirt guides, rebinds only skirt vertices to those spring
chains, and rebuilds their native contacts. It never changes vertex coordinates,
shape keys, hair setup, humanoid bones, non-skirt weights, materials or textures.
Hair planning is skipped during this migration. Repeated migration is a no-op.

LongDress checks in `outputs/longdress_bilateral_20261001`: six pure profile
checks pass; isolated Blender verifies independent 90-degree left/right knee
response, one constraint per helper, live controls, continuous junctions,
idempotent collider replacement, hair isolation and unchanged geometry.
All 24 constraints survive official VRM export/reimport. The 28 captured-pose
samples and live read-back have zero measured dress/leg triangle intersections.
The selected lower section now bends 68.00 degrees (previously 35.34) in the
unchanged user pose. There are no spring/collider rest contacts (minimum
clearance 2.587 mm). Separate stress poses still have clipping: knee hold
48 pairs when held, combined hold 70. These limits remain open; the current-pose
success is not an all-angle collision guarantee.

Applied and saved to `Downloads/LongDress_Hallway_Stable.blend`, left open in
Pose mode with the same selected chain, physics enabled, upper follow 55%,
and existing skirt drag 0.2. No gravity increase was needed.


### Updated follow settings

Front remains 100%; back now defaults to 50% and sides to 15%. The angular
profile is anchored to the actual front, side and rear pairs, so off-axis
lateral chains receive exactly the side setting. For the twelve-chain layout,
each half has the sequence front 100%, 57.5%, side 15%, side 15%, 32.5%, rear
50%. Intermediate chains interpolate between the controls. One same-side
source and one native VRM rotation constraint per helper are retained.
Applied through the live callbacks and saved in the open LongDress file;
colliders, meshes and humanoid pose were not changed by this settings update.


### Knee-junction weight smoothing

Dress binding now uses a wider smoothstep transition at the upper/lower spring
junction: a half-width of 40% of the shorter adjacent segment, versus 20% at
ordinary joints. It is confined to the anatomical knee marker, scales with the
rig, and leaves the waist and other joints unchanged. Fresh generation marks
the knee before initial skin binding; rebinding an existing rig uses that same
marker. No direct leg weights or new constraints are introduced.

On LongDress, 119 vertices change inside that band, and the count sharing both
upper and lower spring influences increases from 57 to 119. Mesh coordinates,
UVs, shape keys and non-skirt weights are unchanged. Rebinding is idempotent.
The checks are reproducible with `tools/test_avatar_dress_smoothing.py`.


### Twenty-pass smoothing with four influences

Binding now finishes with 20 topology-neighbor diffusion passes (factor 0.2),
followed by a spatially softened removal mask and strict four-weight limiting.
This follows the approach inspected in Robust Weight Transfer's
`weighttransfer.py::smooth_weigths` and `limit_mask`: average through mesh
edges, seed removal outside the strongest influences, propagate that removal
mask five times, attenuate, and normalize. Hallway implements this with NumPy
edge gathers; it has no dependency on that add-on or its operators.

Smoothing is scoped to the generated skirt family and mesh topology. The
waist transition is pinned during diffusion; total body attachment and total
spring weight are preserved per vertex. Where the pre-existing waist blend
already exceeds four weights, its strongest body/spring influences receive
slots while retaining both contributions. Fixed waistband vertices with no
spring influence and all non-skirt vertices are untouched. Rebinding starts
from the deterministic guide binding each time, so repeated rig generation
does not accumulate additional smoothing. `smooth_iterations=0` permits
isolated testing of the preceding guide-binding stage.

`test_avatar_skirt_smooth_limit.py` verifies normalization, the four-weight
limit, preservation of body attachment mass and mesh geometry, unchanged
non-skirt vertices, and repeatable rebinding. LongDress's 659 skirt vertices
now have at most four influences; mean squared weight difference across
edges drops from 0.20257 to 0.09479 (about 53%). Fifty-two transitional waist
rows require influence limiting; none gains a new humanoid bone source.

The tested factor 0.5 variant caused 37 held-pose triangle intersections and
was not applied. Factor 0.2 matches the inspected add-on default and clears
the captured held-pose check; all weights remain normalized and limited to
four. Artifacts are in `outputs/longdress_smooth20_20261001/factor02`.

All 28 samples in the captured-pose smooth, quick, cyclic and direct-start
checks have zero measured dress/leg triangle intersections at factor 0.2.
Live read-back confirms 659 skirt vertices, maximum four influences, and
maximum normalization error 6.52e-8. Saved and left open for user review.


### Coherent chain overlap replaces global diffusion

The earlier free-skirt smoothing could leave four weights spanning three
chains. Around the inspected knee vertex, neighbors acquired the opposite
leg's front chain; independent removal masks also gave different longitudinal
blend fractions to each circumferential chain. The weight-count limit alone
did not prevent a sharp fold or competing chain motion.

The production pass now diffuses the longitudinal joint coordinate for twenty
iterations, preserving each vertex's original circumferential chain shares.
It reconstructs a continuous patch using two neighboring chains and two
adjacent joints, giving at most four spring weights without a global ranking
competition. Mixed waistband rows retain the existing influence-budget
handling and their total body attachment. Free skirt rows preserve their
circumferential shares to floating-point precision and gain no third chain.

The selected LongDress vertex's maximum incident-face dihedral in the held
pose drops from about 80.5 to 25.8 degrees. All 28 captured-pose motion samples
remain free of measured leg intersections. The mesh probes did not reproduce
sustained held-pose oscillation before or after (both held step RMS zero).
Moving-pose mesh jerk is slightly higher with the restored local ownership,
so these tests establish improved fold continuity, not a general temporal
jitter reduction. Spring dynamics and colliders were not retuned.

`outputs/longdress_overlap_20261001` contains the selected-vertex/neighbor
inspection, before snapshot, before/after motion probes and final binding
validation. The maximum is two chains and four influences per vertex; free
chain-share error is below 3e-16. Vertex coordinates, pose and non-skirt
vertices are unchanged. The open file retains the selected mesh vertex.


### Sixty-percent front follow and underlying-body waist attachment

Front lower-leg follow now defaults to 0.60. The existing circumferential
profile blends to 0.15 at the sides and 0.50 at the back: the halfway
front-side chains use 0.375, and halfway rear-side chains use 0.325.
Existing saved controls remain user-configurable; the current LongDress
front control was explicitly updated to 0.60.

The inspected upper-dress clipping was partly caused by the underlying
abdomen skin carrying substantial arm and calf weights. Copying those weights
would reproduce the error. `avatar_pelvis_binding` now checks a bounded
anatomical skin region for contamination and, when necessary, solves a
five-bone pelvis/spine/chest/thigh heat domain on a temporary surface. The
correction fades at its upper and lower bounds. Cross-leg heat fades out
across each hip socket and is assigned to the nearest thigh below it, keeping
the independently bound legs separate. Arm-adjacent geometry is excluded.
Healthy body weights skip this repair.

The existing waist-strip body sampling then uses the corrected surface.
An additional body-surface transfer attaches the owned dress material above
the spring roots through the bodice, fading near the upper bound. Unsupported
surface samples retain their original bindings. Garment rows are normalized
and limited to four influences. No free-skirt vertices below the attachment
region are moved onto leg weights. Revision markers prevent repeated repairs
from accumulating when the rig is regenerated. The generation pipeline calls
both stages, so the fix is not a model-specific weight patch.

`outputs/longdress_waist_20261001/scoped` holds the final isolated regression
and motion checks. The upper-region direct-start pose had 77 intersecting
body/dress triangle pairs before the correction and zero afterward. All 28
upper-region samples across smooth, quick, cyclic and direct-start cases are
zero. The separate lower-skirt check still measures 53 pairs in the held
pose, unchanged from before; this is not a claim that all dress clipping is
resolved. Geometry, topology, UVs, shape keys, pose, hair and colliders are
preserved. The lower-thigh audit found no opposite-leg influence above 0.005.
The corrected file is saved at `~/Downloads/LongDress_Hallway_Stable.blend`
and remains open with BVT physics enabled.


### Compact thigh-to-pelvis ownership

The preceding upper-dress correction exposed excessive thigh heat in the
abdomen (up to roughly 77% on inspected vertices). Zero surface intersection
alone was insufficient: the torso and dress were being dragged by the leg.

The installed Auto-Rig Pro source was inspected for its improved-hips pass:
`auto_rig.py::bind_improve_weights` enables short upper-thigh helper bones
for binding, transfers their influence to the root/pelvis group, and removes
the temporary groups. Its ordinary helper extends approximately one seventh
of the thigh length above the socket. Hallway adopts that pelvis-ownership
principle in an independently implemented bounded weight projection; it does
not copy ARP's operator implementation or add export bones.

`avatar_pelvis_binding.thigh_envelope` uses the rest pelvis/spine direction,
the nearest thigh socket and thigh length. The smooth transition starts in
the proximal quarter of the thigh and ends at the pelvis center, capped at
one seventh of thigh length above the socket (with a small fallback for
unusual Hips-head placement). Total thigh weight is capped by this envelope;
excess weight transfers to Hips. This preserves existing spine weights and
is idempotent, unlike repeatedly multiplying the current weights by a mask.
Source skin and the sampled upper dress use this ownership rule. The free
spring region retains its spring-weight mass and local two-chain blending.

The 10%-below-socket trial removed torso bleed but produced 91 triangle
intersections in the deep held pose. Extending the transition on the thigh
side to 25% reduced that count to 67 without extending leg influence into
the torso. These remaining contacts are a known regression relative to the
previous, incorrectly leg-driven torso. They are not presented as resolved.
Geometry, pose, spring parameters, native colliders and 60/15/50 knee follow
remain unchanged. Rest-pose intersection samples remain zero.

`test_avatar_hip_falloff.py` checks migration, normalization, repeatability,
geometry/pose preservation, fixed distal-leg weights, unchanged spring mass,
and zero thigh weights beyond the socket envelope on the relevant torso
skin and garment surfaces. The 672 checked torso vertices have zero thigh
weight beyond that boundary. The live weights match the isolated result;
602 skin vertices were corrected and 345 upper-garment samples refreshed.
Artifacts: `outputs/longdress_hips_20261001/applied`, with pre-change backups
in the parent directory. The corrected Stable file is saved and left open.


### Side-scoped transfer and smoothed attachment handoff (2026-10-02)

The live dress contained 19 spring-weighted vertices with opposite upper-leg
weights inherited from the body surface. Angular chain interpolation also
spread opposite-side front/back chains onto 54 vertices outside a narrow
center strip, increasingly far from the center as the hem flared.

`WaistAttachment.side_owned` now hands opposite-leg sampled mass back to Hips.
Each leg fades to zero at the sagittal plane over a band derived from 8% of
hip-socket separation. Existing upper-dress blends receive the same filter.
Chain blending across the front/back center is bounded to that physical band;
it no longer expands with the hem radius. A narrow continuous center blend
remains to avoid a weight discontinuity; the code adds no center chain.

The body-to-spring scalar transition now spans several mesh rows, computed
from local vertical edge spacing and bounded to 55% of the first spring
segment. Twenty neighbor-averaging passes smooth this scalar with the fully
attached/free boundaries pinned. LongDress's span increases from 3.89 cm to
10.41 cm. This intentionally changes the attachment mass in that upper band;
it does not diffuse complete weight vectors into unrelated chains. Mixed
rows reserve two body slots and two spring slots to avoid abrupt chain loss
when a third body influence wins a weight ranking. The independent compact
hip envelope and free-skirt local-coordinate smoothing remain in force.

`test_avatar_dress_handoff.py` and the updated smoothing regression pass.
The worst handoff spring-mass jump across an edge falls from 0.88471 to
0.45230; mean squared handoff difference falls from 0.19772 to 0.03253.
Live readback confirms zero opposite-leg assignments in the spring handoff,
zero opposite-chain assignments outside the 1.24 cm center half-band,
at most two chains and four weights, and repeatable rebinding. Rest geometry,
body/hair weights, pose, knee-follow settings, springs and colliders remain
unchanged. The updated Stable file is saved and left open with BVT enabled.

The currently captured pose differs from earlier fixtures. Its direct-start
upper-region intersection count decreases from 14 to 7; all four motion
cases remain at seven or fewer measured pairs and return to zero in rest.
Remaining contacts are not claimed fixed. Results and backups are under
`outputs/longdress_handoff_20261002`.


### Upward spring overlap into the transferred outfit (2026-10-02)

The previous handoff only admitted vertices with existing spring weights, and
`transfer_garment_waist` replaced the weights above the chain roots. This kept
the scalar smoothing below the body-transfer boundary.

The binder now grows the owned domain upward through connected vertices of
the same garment material, bounded by its stored top and restricted to torso
attachment sources. It lifts the transition start by the smaller of 65% of
the old fixed-band height and 25% of the first spring-segment length. The
lower end of the blend remains in place, retaining a fully body-attached top
band. The body transfer explicitly skips spring-weighted overlap vertices.
These rules are used by fresh generation and live rebinding alike.

On LongDress the overlap moves 4.73 cm upward and gains 47 previously
body-only vertices. There are 706 spring-weighted vertices and 33 fully
attached vertices in the rebound domain. The largest scalar handoff jump
falls from 0.45230 to 0.25328; edge energy falls from 0.03253 to 0.01676.
Twenty smoothing passes, four influences maximum, two local chains, and the
narrow sagittal blend remain enforced. Independent live readback and the
isolated regression agree, and body transfer no longer erases the overlap.
Geometry, body/hair bindings, rig pose, springs and colliders are preserved.

Artifacts and pre-change backups: `outputs/longdress_upblend_20261002`.
The Stable file is saved and remains open. Motion checks of the newly
captured pose still measure some upper-region intersections (14 pairs in
the held pose), so smooth weights do not establish collision-free motion.


### Smoothing the transferred body distribution at the hitch

In the latest captured pose, the overlap carried direct thigh weights at 82
vertices alongside skirt roots that follow the same thigh with a different
influence and physics rotation. Body weight limiting also switched its chosen
bones between adjacent rows. Increasing only the skirt-mass transition did
not remove this competing deformation.

`smooth_body_attachment` now smooths the normalized body distribution over
20 mesh-neighbor passes, fading the correction toward the pure-body boundary.
In spring-weighted overlap rows, direct thigh mass transfers to Hips; the
skirt chains carry the leg response. A Hips-anchored, two-influence body
representation softens competition for the remaining body slot. Total body
and spring masses, upward overlap, local chain pairs and the four-weight
limit remain intact. Underlying body meshes and rig/collider settings are
unchanged. No helper bones were added or enabled for deformation.

On eight explicitly identified vertices around the screenshot's hitch, the
worst incident-face dihedral in the frozen captured pose drops from 45.36 to
30.74 degrees. Several neighboring folds improve from roughly 35 to 15–17
degrees. The viewport still shows a notch; this is a softening, not a claim of
perfectly rounded geometry. A helper-weight bridge and a different scalar
profile increased folding in isolated tests and were rejected.

`test_avatar_dress_handoff.py` verifies zero remaining direct thigh weights
in the overlap, no opposite-side leakage outside the narrow center strip,
normalization, four influences, repeatability and immutable mesh geometry.
Full-weight handoff edge energy falls from 0.08814 to 0.07454. Frozen-pose
crease probes are reproducible with `test_avatar_dress_hitch.py`; supply the
source file, output directory, mesh name and comma-separated vertex IDs.
These IDs are test probes only, not generation rules.

Current-pose upper-region intersections decrease from 24 to 19 in the held
pose, with zero in the rest samples. Contacts remain unresolved. The saved
live file matches the isolated weights and remains open with BVT enabled.
Artifacts: `outputs/longdress_hitch_20261002/final`, `crease.json`, live
readback and the before/after viewport captures. The explicitly named
rejected helper fixture is not the applied result.

## Independent upper-skirt support (2026-10-02)

`avatar_skirt_support.py` creates two bilateral socket helpers with the source
thigh's rest axes and a short display length. Their independently saved
`Skirt Support` follow group defaults to 0.15. Each has exactly one native
LOCAL/LOCAL, ADD, XYZ Copy Rotation constraint, no inversions, and sits in the
Constraints bone collection. They do not parent spring chains or colliders.
Generation/rebinding reuses the owned names and refuses conflicting bones or
constraints instead of adding duplicates.

Support replaces part of the Hips attachment only inside the existing smooth
body/spring overlap. A compact bell is zero at both overlap endpoints; support
also fades to zero at the sagittal seam before changing sides. Hips remains
an anchor, and competition between the one spine/support slot fades toward
Hips. This retains two body slots plus the existing local spring pair, without
increasing the four-influence budget, altering spring mass, or moving any
rest vertices. Free skirt, fixed bodice and skin/hair weights are untouched.

A preliminary variant multiplied support by the direct-thigh pelvis envelope;
that suppressed support too strongly and did not improve intersection counts.
The final support is bounded by the garment handoff itself. This differs from
the rejected earlier bridge through existing 55% spring follow controllers.

In the freshly captured raised-leg pose, the worst of eight diagnostic crease
probes was 47.07 degrees before rebinding, 31.73 with support disabled, and
25.00 at 15% support. At 25% support the worst probe increased to 28.83, so the
lower 15% default was retained. The probes are fixture data, never generator
rules. Upper-region intersecting triangle pairs in the held pose decreased
25 -> 22 (support disabled) -> 19 (support 15%). Rest samples have zero pairs.
The smooth/quick/direct held phases settled with zero measured spring motion.
The cycle still has 8 intersecting pairs at one intermediate sample: this is
not a complete collision solution. The viewport still shows a small upper
notch and visible leg contact in the deep pose.

`test_avatar_skirt_support.py` validates duplicate-free repeated generation,
unchanged original rest bones and mesh geometry, pose preservation, bilateral
ownership, normalized weights, the four-weight limit, live follow callbacks,
and actual VRM export/reimport of both support constraints. The live file
matches the isolated weights: 110 support-weighted vertices, 732 total bones,
121 springs and 105 colliders. Existing follow settings remain unchanged.
Saved and left open: `~/Downloads/LongDress_Hallway_Stable.blend`.
Evidence and pre-change backups: `outputs/longdress_support_20261002/`.

## Surface support for every skirt chain (2026-10-02, revision 2)

Replaces the two socket helpers with one support per skirt chain. Each head is
ray-fitted to the garment's rest surface along that chain's radial line, near
the top of the garment and above its own thigh socket. The original leg axes
are retained for portable local XYZ Copy Rotation; the short bone display
therefore follows those axes, while its pivot lies on the skirt. Surface
fitting uses existing coordinates only. On LongDress there are 12 pivots,
7.34 cm above the sockets. The obsolete L/R helpers and their groups are
removed; unrelated dependencies cause migration to stop before deletion.

Support now occupies a wider fraction of the body/spring overlap and follows
the same two neighboring chains as the spring weights. Twenty smoothing
passes operate only inside each vertex's original set of influences. Twenty
removal-mask passes then limit the result to four skin weights. Pinned free
skirt rows stay unchanged; no third-chain columns can appear. Garment waist
transfer recognizes support weights as an owned attachment and preserves them.

An initial two-stage body -> support -> spring blend compressed the lower
handoff too sharply and was rejected. The applied blend keeps concurrent
support/spring response and spreads the weight limiting over mesh neighbors.
The old socket setup covered 110 vertices; the final chain supports cover 171.
Neither mesh positions nor topology, UVs or shape keys are changed.

The captured live support slider was 0.9465 and the side-knee control was
0.5570. Tests compared several support settings. At 0.55 the probed bend was
smooth but still had 9 upper-region intersecting triangle pairs in the held
pose. At 0.75, all sampled upper-region intersections were zero in smooth,
quick, cyclic and held-pose cases, compared with up to 14 before this change.
The worst of the eight existing frozen-pose crease probes fell from 124.15 to
27.51 degrees. These fixture measurements do not establish zero clipping for
every pose or every part of the dress. LongDress was left at 0.75 support;
other user controls, including the changed side-knee value, were preserved.
The generator's default remains 0.15 and existing saved values are retained.

`test_avatar_skirt_support.py` verifies all 12 surface pivots, their height
above the socket, native one-constraint eligibility, actual VRM export and
reimport, old-helper cleanup, repeatable generation, bilateral ownership,
live control updates, and geometry/rest-bone invariants. The live rig has
742 bones, the original 121 springs and 105 colliders, and matches the offline
weights. Saved and left open: `~/Downloads/LongDress_Hallway_Stable.blend`.
Evidence: `outputs/longdress_chain_support_20261002/`, including backups,
parameter probes, motion reports, export fixture, and live readback.

## Canonical defaults and complete panel reset (2026-10-02)

`avatar_rig_defaults.py` is shared by generation and panel reset. Follow values
are Skirt .55, Front Knee .60, Side Knee .55, Back Knee .50, and legacy Skirt
Support .75; thickness is 1.0. Existing saved values are preserved on normal
initialization. The Reset button now invokes `hallway.reset_rig_settings`,
not the old simulation-cache-only reset. It restores all extant follow groups,
thickness, spring sliders, physics on (when BVT is available), visible owned
bone collections, and hidden collider parent collections. It works with BVT
absent and resolves the rig when a bound mesh is selected. It never regenerates
bones/weights or overwrites unrelated rig data.

Spring defaults remain the generator's values: Skirt drag .4, root stiffness
1.6, non-root multiplier 1, gravity .025; Hair .4, 1.0, 1, .035. Native spring
joints, constraint influences and VRM capsule radii are read back in tests.
`test_hallway_rig_defaults.py` deliberately changes every panel setting and
executes the actual operator, then checks repeatability, new group defaults,
retention of saved settings, and immutable geometry/humanoid pose. Live
execution additionally verifies that Reset restores physics from Off to On.
Evidence: `outputs/longdress_defaults_20261002/`.

## Upper-skirt physics replaces surface constraints (2026-10-02)

`avatar_skirt_hip_physics.py` converts the 12 surface support bones into native
VRM spring segments. Each retains its garment-surface pivot, is parented to
Hips, and gains an unweighted tip extending 20% into its original first skirt
segment. No Copy Rotation remains on these upper bones. The weighted surface
pivots stay body-attached while their orientation reacts to leg contacts and
BVT spring simulation. Existing lower-skirt and knee hierarchies, constraints,
weights and rest frames are unchanged. These are separate short upper spring
chains, not one continuous extension through the lower leg-follow drivers.

A continuous extension through those drivers was tested and rejected because
it increased jerking. The final independent upper chains use ordinary exported
VRM springs/capsules, no center, and no custom solver. Each reuses only its own
chain's collider candidates that clear the new tip in rest pose. Per-chain
collider groups are reused on repeated installation and refreshed after a
contact refit; no new collider objects are created. Artist spring sharing is
still rejected, while generated upper-skirt springs are recognized as owned.

The new upper springs participate in the existing Skirt Spring controls. The
obsolete Skirt Support Follow group is removed, so Reset resets the controls
that still exist. Generation now installs upper physics after lower contacts.
Weight rebind preserves these physics bones and does not recreate constraints.
The legacy support default remains defined for older, unconverted rigs.

In the current captured pose with default drag .4, held upper-region triangle
intersections decreased from 35 to 21; quick-motion initial intersections
were 40 versus 24. Smooth-motion spring-tip RMS jerk decreased from .425 to
.381 mm, quick motion 3.178 to 2.869 mm. Held springs settled (zero measured
step RMS), and rest samples had zero intersections. Some deep-pose mesh
clipping and angular folds remain; these results are not collision-free.
The direct continuous-extension prototype is not the installed result.

`test_avatar_skirt_hip_physics.py` validates native export/reimport of all 12
new springs, center-free setup, zero upper constraints, unchanged geometry and
weights, preservation of lower rest frames and collider identities, repeatable
installation and rebind, and positive rest clearance. Contact refitting was
also executed on an isolated copy. Live readback: 754 bones, 133 springs,
105 unchanged collider objects, BVT enabled, original pose/selection preserved.
Saved and left open: `~/Downloads/LongDress_Hallway_Stable.blend`.
Evidence: `outputs/longdress_hip_physics_20261002/`.

### Upper hip contact patches

Upper skirt springs now have their own `Secondary_HipContact_` groups. Each
uses six ordinary VRM capsules to form two rounded, finite contact faces in the
lateral and front/back directions. Faces follow only the chain's owning thigh;
their outward surfaces are fitted against its short spring endpoint with a
positive rest margin. Hair and lower dress chains cannot reference these patches.
The existing rest-clear hips roof and own-thigh centerline capsule remain as
fallbacks. Native VRM properties own attachment, offsets, endpoints and radius.
Repeated fitting updates owned shapes in place and removes unused old references;
the thickness control includes the new groups and respects their rest-safe cap.

The upper origins are fitted on the garment surface just below its attachment
height. The first skirt segment shares a bounded, smooth weight transition with
its own upper spring, ending at that segment's tail. Twenty local smoothing and
removal-mask passes keep at most four influences within the original adjacent
chain pair. No mesh coordinates, UVs or shape keys change. Full regeneration
reconstructs this transition from the source strip weights instead of repeatedly
accumulating it.

The 2026-10-02 isolated LongDress tests are in
`outputs/longdress_hip_guards_20261002/`. They distinguish upper-region triangle
intersections from collider contact depths and test captured, forward, combined
and mirrored leg poses with BVT. These are sampled diagnostics, not a guarantee
of collision-free arbitrary poses or other applications' spring solvers.

### Averaged upper/lower weights and virtual seams

Upper and lower free-skirt segments now use the same 20 gentle neighborhood
averages (factor 0.08) instead of a nonlinear chain-ownership curve. Eligible
support is limited to the dominant chain and its immediate same-leg neighbors,
with two adjacent longitudinal joints. Mixed attachment rows stay outside this
free-cloth averaging pass. The final four-weight projection preserves each
averaged chain share and the longitudinal joint coordinate; independently
pruning the smallest weights would reintroduce abrupt changes between rows.
No current pose or selected vertex index enters production weighting.

Before any weight filter, coincident rest-space vertices within each garment
family/object are represented by a single virtual vertex with the union of
its mesh neighbors. Self edges and duplicate edges are removed. All copies
receive identical final weights. The mesh, topology, UV seams and shape keys
are never merged or altered. The tolerance is one millionth of garment extent,
with bounded representative clusters to avoid transitive welding across a band.
At an existing duplicated center seam, this joins the already present left/right
cloth boundary; distant or unrelated garment surfaces do not join the graph.

`test_avatar_dress_projection.py` verifies noisy-field attenuation, identical
upper/lower behavior, pinned rows, normalized four-weight projection, coordinate
preservation and virtual seam grouping. `test_avatar_dress_boundaries.py` checks
geometry and collider invariants, unchanged unrelated weights, allowed ownership,
repeatable regeneration, selected-pose fold improvement and identical seam weights.
The 2026-10-02 fixture has 21 duplicated seam groups; maximum pre-fix weight
mismatch was 0.90465 and the post-fix mismatch is zero. Its selected folds changed
from 50–86 degrees to 33–47 degrees, and the sampled lower free-cloth maximum
changed from 92 to 51 degrees. Evidence: `outputs/longdress_average_20261002/`.
These skinning diagnostics do not establish collision-free cloth behavior.

### Helper-to-chain interpolation

The upper handoff receives a conservative 20-pass neighborhood average at
factor 0.08. Helpers and the first main-chain segments participate together;
one neighboring ring provides the transition to the rest of the cloth.
Real body weights remain exact. Four-influence reduction preserves both each
chain's averaged share and the separate helper/main-joint totals, rather than
only the average longitudinal coordinate. This prevents exchanging helper
weight for a lower segment and creating a peak between them. Rows needing more
than four independent preserved quantities retain their prior binding.

The filter uses generated helper metadata and the already virtually welded
garment graph. No selected indices, posed coordinates, bone edits or additional
constraints enter generation. `test_avatar_hip_weight_interpolation.py` validates
body/chain/joint conservation, noise attenuation and four-weight output.
The captured 2026-10-02 upper-field maximum dropped from 65.6 to 56.4 degrees;
the original highest edge dropped to 40.9 degrees. Not every crease improves:
some body-boundary edges remain around 54–56 degrees. All 21 virtual seam groups
still match exactly, and geometry, contacts and regeneration checks pass.
Evidence: `outputs/longdress_upper_average_20261002/`.
