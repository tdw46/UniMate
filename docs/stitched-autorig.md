# Stitched bust auto-rig evaluation

Input: `~/Downloads/stitched.glb`, modified September 15, 2026.
SHA-256: `fa47b66de44108d88680379258cf438f0970e527c8f48b7d9e7eb96fe25391be`.
The supplied file has one textured mesh, 262,768 imported vertices, 500,000
triangles, no skeleton, and no animation. It is already a cropped upper-body
bust; no additional Boolean cut was applied.

`tools/autorig_bust.py` fits an 11-bone skeleton from surface cross-sections.
This fitter assumes a roughly symmetric upright bust, lowered arms, cropped
forearms, Blender -Y forward and Z up. It is not a general full-body landmark
model and does not run pretrained UniMate inference.

The first Blender heat solve failed. A temporary copy with duplicate vertices
welded, normals recalculated, and a numerical scale adjustment solved all
250,000 unique vertices. No caps were needed and no body voxelization occurred.
Every original vertex matched its heat-copy position exactly. Original geometry
and UVs remain unchanged, including the imported atlas splits.

The same crease-based torso/arm boundary used by the nine-avatar flow is applied
to the fresh heat weights. A concavity-weighted surface cut locates each arm
attachment; the torso side of the narrow crease band retains at least 90%
Root/Spine/Chest influence, and the torso interior has zero arm influence.
Distal arm heat ratios remain intact. The head/neck seam pass made no changes:
the welded heat solve had already assigned matching weights to its atlas splits.
It must not treat facial UV seams as rigid anatomical head boundaries.

## Saved-artifact checks

- 11 bones; 240 frames at 30 fps: head turn/nod, neck tilt/bend, arm raise/elbow bend.
- Original geometry and every UV loop unchanged; all vertices weighted,
  normalized and limited to four influences.
- Maximum torso-interior arm weight: **0**.
- Maximum torso-side crease arm weight: **0.100000006** (float storage).
- Minimum torso-side crease Root/Spine/Chest weight: **0.899999969**.
- Torso displacement during the isolated arm raise: **0**.
- Coincident atlas-vertex gap across sampled head, neck and arm poses: **0**.
- Actual GLB → UniMate NPZ → GLB reconstruction passes all 11 bones over
  240 frames and evaluated surface comparisons on nine sampled frames.

The extreme raised-arm render still exposes stretched folds and texture in the
source's tightly compressed underarm geometry. The ownership checks above do
not establish production-quality deformation. This is a reviewable rig test.

Local outputs: `outputs/stitched_autorig/rigged.blend`, `rigged.glb`,
`evaluation.mp4`, `evaluation_stage.blend`, `saved_rig_validation.json`,
`pipeline_validation.json`, and `locality_validation.json`. Source/model/media
files stay local. The 8-second H.264 preview is decode-checked.

```sh
export PYTHONPATH="$PWD/.local_deps/site-packages:$PWD/.local_deps/Motion:$PWD"
blender -b --factory-startup --python-use-system-env --python-exit-code 1 \
  -P tools/autorig_bust.py -- --input "$HOME/Downloads/stitched.glb" \
  --output "$PWD/outputs/stitched_autorig"
blender -b --factory-startup --python-use-system-env --python-exit-code 1 \
  -P tools/validate_autorig_bust.py
blender -b --factory-startup --python-use-system-env --python-exit-code 1 \
  -P tools/evaluate_autorig_bust.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_autorig_bust.py -- \
  --input outputs/stitched_autorig/rigged.blend --output outputs/stitched_autorig
python3 tools/encode_autorig_bust.py
```
