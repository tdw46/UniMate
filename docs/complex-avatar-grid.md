# Nine anime avatars: destructive bust cuts and fresh rig evaluation

Nine additional VRoid/pixiv characters were prepared and evaluated in Blender
5.2 using the same shared flow as the [first nine avatars](nine-avatar-grid.md).
The new 3×3 grid includes layered jackets, loose sleeves, uniforms, ribbons,
cyber clothing, transparent details, and dense anime hair.

## Sources and scope

| Card | Character | Source license |
| --- | --- | --- |
| 01 | Darkness Shibu | [CC0 beta sample 1](https://vroid.pixiv.help/hc/en-us/articles/360012381793) |
| 02 | Sakurada Fumiriya | [CC0](https://vroid.pixiv.help/hc/en-us/articles/360014788554) |
| 03 | Sendagaya Shino | [CC0](https://vroid.pixiv.help/hc/en-us/articles/360013482714) |
| 04 | Victoria Rubin | [CC0 beta sample 4](https://vroid.pixiv.help/hc/en-us/articles/360014900233) |
| 05 | Vita | [CC0 beta sample 3](https://vroid.pixiv.help/hc/en-us/articles/360014900113) |
| 06 | Vivi | [CC0 beta sample 2](https://vroid.pixiv.help/hc/en-us/articles/360014900273) |
| 07–09 | AvatarSample A, B, C | [VRoidPreset conditions, not CC0](https://vroid.pixiv.help/hc/en-us/articles/4402394424089) |

The creator permits images, videos, and modification of these samples. A–C
retain their specific terms, including restrictions on redistribution and
character creation services. They are evaluation fixtures, not a redistributable
CC0 training dataset. The creator's primary license pages were checked on
September 14, 2026. Their help center blocked command-line HTML snapshots;
license URLs, embedded VRM metadata, exact model URLs, and SHA-256 hashes are
preserved in [the manifest](complex-avatar-sources.json).

The VRMs came from the public `madjin/vrm-samples` mirror pinned at commit
`e16eb187100149a315ad92c3c9968f1d5baa6c7d`. Sketchfab access was unavailable
in the configured Blender connection, so this batch uses the requested
alternative source option. These are nine distinct characters from the same
VRoid authoring family; several share underlying body/clothing topology.
This broadens garment and material coverage, but is not a benchmark across
nine unrelated modeling systems.

## Shared preparation

The shared preparation script now accepts `AVATAR_EVAL_ROOT` so the original
nine outputs remain separate. `avatar_source_import.py` reads VRM humanoid
metadata and maps semantic joints to the same preparation landmarks. A frame
built from anatomical left/right and hips-to-neck landmarks rotates the entire
import consistently before posing. This avoids assuming a source's forward
axis or matching a character's original bone names.

The original rig is used only to bake the 20-degree A-pose and record joint
locations. All original armatures, animation, vertex groups, and weight values
are then deleted. An Exact Boolean Difference removes geometry below the
upper abdomen; the modifiers are applied destructively and the cutter deleted.
Each saved `01_cut_unrigged.blend` was reopened to verify zero rigs, groups,
weights, Boolean modifiers, and Armature modifiers.

VRoid semantic material tokens split mixed meshes into body, head/hair,
rigid neck accessory, and clothing regions while retaining UVs. Neck accessories
marked as cloth join the garment regions so ties and bows can follow the chest.
This is explicit
source-format information, not a per-character allowlist. Unknown materials
retain the existing geometric classification path. Arbitrary assets without
those conventions may require explicit `binding_role` metadata.

The same new 13-bone skeleton, heat binding, anatomical influence constraints,
new-body surface transfer, and unweighted-vertex fallback used in the original
batch then create entirely new weights. Head/hair is rigid to Head; explicit
non-cloth neck accessories are rigid to Neck; torso clothing excludes Head and receives
local body/shoulder/arm/collar influences. No avatar identities, source vertex
counts, or old weight values are inputs to the apparel algorithm.

After apparel binding, `avatar_voxel_seams.py` makes a temporary joined skin
proxy. Skin surfaces use explicit `binding_surface=skin` metadata; the VRoid
adapter derives that tag from the source `SKIN` material convention. This
excludes hair, clothing, eyes, and separate accessories from the voxel union.
Other source formats can supply the same metadata directly on objects or
materials, together with `binding_role=body` or `head` for the head boundary.

Across the batch, preparation removed **1,202 original bones** and **329,370
original weight assignments**, and destructively applied **19 Boolean
modifiers**. Individual cut busts contain 11,374–17,758 welded vertices and
19,625–34,462 triangles. Counts describe actual prepared geometry rather than
summing repeated glTF accessors across material primitives.

## Validation

- `validate_avatar_regions.py` checks every vertex for normalized weights and
  at most four influences. Isolated Head motion causes zero torso-garment
  movement; Neck causes zero movement below the collar. Explicit neckwear
  follows the evaluated Neck transform and stays fixed under Head-only motion.
  Garments respond to Spine/Chest and both arm chains wherever sleeves exist.
- `test_avatar_source_import.py` uses synthetic, arbitrarily named skeletons
  with both VRM 0 and VRM 1 metadata and three different source rotations. It
  verifies canonical landmark recovery and UV-preserving semantic splits.
  Cloth and metal neck-accessory materials are tested separately: cloth uses
  body-surface transfer while rigid neckwear retains its Neck attachment.
  The nine downloaded files themselves are VRM 0.
- The original nine-avatar apparel regression checks were rerun and passed.
- All nine new avatars passed the actual UniMate GLB → NPZ → GLB path, checking
  all 13 bones across 360 frames and evaluated surfaces at 13 frames.

| Reconstruction measurement | Maximum observed | Tolerance |
| --- | ---: | ---: |
| World rotation difference | 0.0000505 degrees | 0.1 degrees |
| Joint position difference | 0.00000159 scene units | 0.0001 |
| Surface difference | 0.00000285 scene units | 0.0001 |

The motion is authored diagnostic animation: frames 0–119 head turns/nods,
120–239 neck bends/tilts, and 240–359 arm raises/elbow flexion. **This does not
run pretrained model inference.** Round-trip accuracy measures preservation,
while separate region tests check the specific binding failure modes.

Eevee renders the imported glTF unlit shader graphs, preserving texture tint,
alpha masks, and blending. Workbench texture previews omit some of those
material effects and are unsuitable for this batch. This is the source glTF
material appearance, not a reconstruction of every proprietary MToon effect.
Hair is rigid to Head, and the 13-bone test has no facial animation, finger
articulation, cloth/hair simulation, or collision handling. Shoulder creasing
and hair/clothing intersections can still occur during these stress motions;
the checks do not certify production-ready skinning.

Evidence: [preparation](complex-avatar-preparation_validation.json),
[region motion checks](complex-avatar-region_validation.json),
[reconstruction](complex-avatar-pipeline_validation.json),
[decoded videos](complex-avatar-render_validation.json), and
[saved scenes](complex-avatar-scene_validation.json).

### Neckline cloth correction

The first render attached both 02's tie and 03's bow entirely to Neck. The
source material convention identified their location, but its `CLOTH` suffix
had been ignored. The importer now routes neck cloth through the existing
garment transfer and anatomical constraints. There are no character-specific
binding overrides or new fixed blend percentages.

The tie's mean vertex influence is now approximately 20% Neck and 80% upper
torso, with Neck influence falling to zero down its length. The bow lies entirely
below the neck base and follows its shirt: approximately 98.5% upper torso,
0.025% Neck, and the remainder other nearby body influences. Those percentages
are measured outcomes, not hard-coded targets. Neck weighting is not artificially
introduced where the freshly bound body surface supplies almost none.

`validate_neck_cloth.py` checks the material regions independently of their
containing objects. Both have zero Head-only movement and zero Neck-only
movement below the collar. It also verifies Chest response, mixed influences,
and retained Neck response wherever cloth extends above the neck base.
The [old-weight audit](complex-avatar-neck_cloth_validation_before.json)
reproduces both failures; the [new audit](complex-avatar-neck_cloth_validation.json)
passes. All nine general region checks and full reconstruction checks were
rerun, along with the original nine-avatar apparel regression checks.

### Voxel skin seams and the top-of-neck anchor

The original independently bound face and body opened along their shared skin
boundary during Head rotation. The face was rigid to Head while coincident
body vertices used different Head/Neck weights. Welding UV vertices within
each mesh did not address that cross-object mismatch.

The new stage joins only skin into a temporary proxy, virtually welds nearby
boundary vertices, closes proxy-only eye/mouth openings, and applies Blender's
Voxel Remesh at one sixteenth of the new Neck bone length. The original render
meshes are never remeshed. Fresh heat weights are generated on the solid proxy
and transferred locally by nearest-triangle barycentric interpolation, fading
into existing weights away from seams. Eight proxies obtained a heat solution;
Sample C used the recorded fallback based on distance to the new skeleton.
All nine proxies were actually voxelized, with 22,908–41,376 vertices.

**The head/neck boundary itself is exactly 100% Head on both sides.** Head
influence blends downward into Neck below that boundary. Nearby UV duplicates
also receive identical weights, while existing rest-position offsets are
preserved. Facial geometry and other head features retain rigid Head weights.
The neck-cloth correction remains in effect; garments and accessories do not
participate in this skin pass.

`validate_avatar_seams.py` independently finds 360 matching skin-vertex pairs
across the nine characters, checks every animated frame, and adds twelve
isolated Head/Neck stress poses about all three axes at ±0.8 radians. It asserts
100% Head at every matching head/neck boundary pair. Maximum additional gap
growth fell from **0.107109** to **0.000000597** scene units during the animation;
the stress-pose maximum fell from **0.177057** to below **0.0000004**.
Tiny gaps already present in the source remain (up to 0.000543 scene units);
the pass changes weights, not source vertex positions.

Geometry, material assignments, and per-corner UV fingerprints match the
previously prepared meshes. Fingerprints canonicalize polygon order because
rerunning Exact Boolean can reorder its cap faces. Three procedural tests use
different tessellations, duplicated UV vertices, scales, and translations;
their seams stay closed and nearby clothing/hair weights remain unchanged.
All nine region checks, neck-cloth checks, original-avatar apparel regressions,
and complete UniMate reconstruction checks pass after the change.

Evidence: [old seam gaps](complex-avatar-seam_validation_before.json),
[corrected seam gaps](complex-avatar-seam_validation.json), and
[procedural seam tests](complex-avatar-seam_generalization.json).

### Front and back shoulder smoothing

`avatar_shoulder_weights.py` adds a modest weight-smoothing stage after apparel
and seam binding. Each fresh upper-arm axis defines a shoulder cross-section;
the bilateral shoulder landmarks and chest/neck axis distinguish front/back
from top/underside. A smooth angular mask excludes the top and armpit sectors,
with additional distance falloffs around each shoulder and away from the collar.
The radii scale with bone length, with no character or topology exceptions.

Three local averaging steps at 0.45 strength blend only eligible body and torso
weights. Surface-area weighting reduces sensitivity to vertex density. Samples
stay within a connected component and compatible surface-facing directions,
so separate garment layers do not exchange weights. Vertices with any existing
Head or Neck influence are excluded from both editing and sampling, preserving
the head boundary and neckline cloth behavior. Geometry and UVs stay unchanged.

All nine avatars have reduced shoulder edge-weight variation: **7.5–19.3%**,
including **12.1% on 07**. This measures the summed squared weight differences
across shoulder-region edges; it is a smoothness diagnostic, not a percentage
improvement in visual quality. Protected weights and geometry/UVs match the
preserved baseline. Synthetic mirrored shoulder cylinders at three densities,
scales, and origins also pass, including explicit top/underside and Head/Neck
protection. The nine seam, region, neckline-cloth, and reconstruction checks
pass after smoothing. Existing modeled folds and extreme-pose creases remain.

Evidence: [shoulder comparison checks](complex-avatar-shoulder_validation.json)
and [procedural shoulder tests](complex-avatar-shoulder_generalization.json).
The numerical comparison requires the preserved `before_shoulder_smoothing/`
baseline; save each avatar's pre-change `02_fresh_rig.blend` there before
rebuilding. The close-up renderer uses that same baseline for the before views.

## Deliverables and reproduction

Local outputs are in `outputs/complex_avatar_grid/`:

- `nine_avatar_grid.mp4`: 1920×1920, 30 fps, 24 seconds; front then oblique.
- Nine individual `{id}_demo.mp4` clips and two 12-second view clips.
- `neck_cloth_before_after.mp4`: close-ups of 02 and 03, before on the left
  and corrected on the right, including both camera views and all motion phases.
- `voxel_seams_before_after.mp4`: the two largest original seam failures
  (02 and 07), selected from the measured baseline, before and after correction.
- `shoulders_before_after.mp4`: 07 front and back, before/after side by side,
  through the arm phase at half speed (8 seconds), plus a comparison still.
- `avatar_grid_front.blend` and `avatar_grid_oblique.blend`, with packed
  textures and nine reconstructed rigs, totaling 117 bones.
- Poster, six-panel contact sheet, image sequences, source manifest, audits,
  and each avatar's unrigged checkpoint, fresh rig, NPZ, and reconstructed GLB.
- Per-avatar `03_voxel_skin_proxy.blend` libraries contain the actual generated
  proxy and its fresh weights; append their objects to inspect the voxel surface.

The scripts run in disposable background Blender processes. The open, dirty
user scene was not changed or saved. Large source/model/media files remain
local and are excluded from Git; source provenance and numerical audits are
included in the fork.

```sh
python3 tools/download_complex_avatar_sources.py
export AVATAR_EVAL_ROOT="$PWD/outputs/complex_avatar_grid"
blender -b --factory-startup --python-exit-code 1 -P tools/prepare_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/test_avatar_source_import.py
blender -b --factory-startup --python-exit-code 1 -P tools/validate_avatar_regions.py
blender -b --factory-startup --python-exit-code 1 -P tools/validate_neck_cloth.py
blender -b --factory-startup --python-exit-code 1 -P tools/test_avatar_voxel_seams.py
blender -b --factory-startup --python-exit-code 1 -P tools/validate_avatar_seams.py
blender -b --factory-startup --python-exit-code 1 -P tools/test_avatar_shoulder_weights.py
# These two comparisons require the preserved pre-smoothing baseline.
blender -b --factory-startup --python-exit-code 1 -P tools/validate_avatar_shoulders.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_shoulder_comparison.py
# Use the isolated dependencies documented in blender-5.2-evaluation.md.
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/evaluate_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_avatar_grid.py -- --oblique
python3 tools/encode_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/validate_avatar_grid_scene.py
```
