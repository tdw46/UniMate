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

The fresh 13-bone skeleton uses Blender heat binding. Broad body and garment
surfaces retain that solution, with Head influence on clothing reassigned to
the torso while preserving the other heat weights. Explicit head/neck jewelry
attachments and the
previous detached neckline-cloth attachment rule remain. The rejected broad
voxel transfer, sleeve replacement, and front/back shoulder fields are removed.

## Local seam correction

`avatar_voxel_seams.py` finds coincident boundary vertices on skin surfaces.
The temporary voxel mesh contains only faces within **two mesh-edge steps** of
an eligible seam. Transfer uses the same graph boundary: weights outside it
remain exactly unchanged. Zero- and one-loop modes are also supported. Original
geometry and UVs are never remeshed.

Explicit head/body boundaries receive 100% Head weights at the shared seam.
UV splits alone do not establish a head/body boundary. Already matching UV
weights are left intact; ambiguous atlas seams use a tighter coincidence
tolerance so neighboring dense vertices are not merged into one seam group.
Skin metadata excludes clothing, hair and accessories from this operation.

## Heat restoration and atlas placement

The experimental crease-based arm/torso cut is disabled in the preparation
flow. It created a hard shoulder-top transition on the fused, lowered-arm bust.
Fresh heat weights now control the shoulders and arms, with the existing
clothing Head exclusion and detached attachment semantics retained.
`avatar_arm_boundary.py` remains an experimental tool, not a default stage.

`avatar_atlas_landmark.py` corrects the shared Neck tail / Head head using neck
shaft cross-sections below the jaw. Its depth is 60% from the shaft's front to
back: the center plus a 10%-of-depth posterior bias. The joint's height and
lateral position are preserved. UV duplicate positions are deduplicated before
sampling, and insufficient section coverage leaves the input joint unchanged.
The helper postprocesses inferred joints in the geometric bust fitter.
The grid preserves its authoritative source-skeleton landmarks.

Future generated inputs should use an A or T pose with visible space between
the upper arms and torso. Preserve the anatomical axillary recess at the
shoulder; posing a fused lowered-arm mesh afterward cannot recover geometry
that the generator merged away.

## Validation and artifacts

- Saved pre/post seam files are compared by `validate_seam_locality.py`:
  geometry/UV equality, normalized weights, four influences maximum, and zero
  changed vertices outside an independently reconstructed two-loop boundary.
- Procedural seam tests cover zero/one/two loops, different tessellations,
  scales and origins, nearby disconnected skin, and already matching UV splits.
- Atlas tests vary jaw projection, neck tessellation, scale, origin and UV
  duplication; the result stays inside the posterior half of the neck.
- `validate_avatar_seams.py` evaluates the head/neck boundaries through all
  360 authored diagnostic frames.
- `evaluate_avatar_grid.py` checks actual GLB → UniMate NPZ → GLB motion and
  evaluated surface correspondence. This is not pretrained model inference.

Outputs are local under `outputs/complex_avatar_grid/`: the packed Blender
stages, per-avatar rig files, 3×3 front/oblique animation, individual card clips,
and machine-readable validation reports. The current acceptance reports are
`complex-avatar-locality_validation.json`, `complex-avatar-seam_validation.json`,
`stitched-atlas_generalization.json`, and
`complex-avatar-pipeline_validation.json`. Older region/strain reports describe
their historical revisions. Historical shoulder comparison
renders represent rejected revisions and are no longer generated or advertised
as current results. Source characters and media are not committed.

```sh
export AVATAR_EVAL_ROOT="$PWD/outputs/complex_avatar_grid"
export PYTHONPATH="$PWD/.local_deps/site-packages:$PWD/.local_deps/Motion:$PWD"
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/prepare_avatar_grid.py
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/validate_seam_locality.py -- --root "$AVATAR_EVAL_ROOT"
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/validate_avatar_seams.py
# The reconstruction runner requires the locally installed SciPy/Motion dependencies.
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/evaluate_avatar_grid.py
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/render_avatar_grid.py
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/render_avatar_grid.py -- --oblique
python3 tools/encode_avatar_grid.py
```
