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
neck accessory, and clothing regions while retaining UVs. This is explicit
source-format information, not a per-character allowlist. Unknown materials
retain the existing geometric classification path. Arbitrary assets without
those conventions may require explicit `binding_role` metadata.

The same new 13-bone skeleton, heat binding, anatomical influence constraints,
new-body surface transfer, and unweighted-vertex fallback used in the original
batch then create entirely new weights. Head/hair is rigid to Head; explicit
neck accessories are rigid to Neck; torso clothing excludes Head and receives
local body/shoulder/arm/collar influences. No avatar identities, source vertex
counts, or old weight values are inputs to the apparel algorithm.

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

## Deliverables and reproduction

Local outputs are in `outputs/complex_avatar_grid/`:

- `nine_avatar_grid.mp4`: 1920×1920, 30 fps, 24 seconds; front then oblique.
- Nine individual `{id}_demo.mp4` clips and two 12-second view clips.
- `avatar_grid_front.blend` and `avatar_grid_oblique.blend`, with packed
  textures and nine reconstructed rigs, totaling 117 bones.
- Poster, six-panel contact sheet, image sequences, source manifest, audits,
  and each avatar's unrigged checkpoint, fresh rig, NPZ, and reconstructed GLB.

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
# Use the isolated dependencies documented in blender-5.2-evaluation.md.
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/evaluate_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_avatar_grid.py -- --oblique
python3 tools/encode_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/validate_avatar_grid_scene.py
```
