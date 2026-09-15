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

The experimental crease-based arm/torso constraint is disabled. It introduced
an unwanted notch at the shoulder top. Ordinary normalized heat now supplies
the final weights, including the smooth shoulder transition. The local seam
pass remains limited to two edge loops; this fixture already has continuous
weights across its atlas UV splits, so it requires no additional seam changes.

The anatomical atlas pivot (shared Neck tail / Head head) is estimated from
neck-shaft sections below the jaw rather than the jaw-contaminated head-base
slice. Its depth is 60% from front to back, slightly behind the neck center.
Height and lateral placement stay unchanged. This is a geometric postprocess
on the fitted skeleton, not a change to any pretrained model.

Future source generations should use an A or T pose with space between arms
and torso, preserving a high axillary recess. The current mesh has merged
surfaces in its lowered-arm source pose; changing weights cannot reconstruct
that missing geometry. The current cross-section arm fitter remains scoped
to lowered, cropped arms; the grid flow uses its source skeleton landmarks.

## Saved-artifact checks

- 11 bones; 240 frames at 30 fps: head turn/nod, neck tilt/bend, arm raise/elbow bend.
- Original geometry and every UV loop unchanged; all vertices weighted,
  normalized and limited to four influences.
- Final weights exactly match the normalized ordinary heat checkpoint.
- The atlas joint is inside the posterior half of the measured neck shaft.
- Neck tail and Head head coincide exactly.
- Coincident atlas-UV vertices stay closed through sampled diagnostic poses.
- Actual GLB → UniMate NPZ → GLB reconstruction passes all 11 bones over
  240 frames and evaluated surface comparisons on nine sampled frames.

The raised-arm render is a reviewable deformation test. The fused source
underarm still limits the result; the rejected sharp shoulder weight boundary
is removed. The previous constrained rig/video is preserved locally under
`before_atlas_heat_restore/`.

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
