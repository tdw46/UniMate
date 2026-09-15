# Clean bust auto-rig evaluation

Input: `~/Downloads/clean-bust.glb`, SHA-256
`4685a58e9708f56acffba90c7d93b16a485bda375fc9b9e0a32b77c4be4ad955`.
One unrigged textured mesh: 8,969 imported vertices and 13,674 triangles.
It is already cropped, so no additional Boolean cut was needed.

The same geometric bust fitter and ordinary heat workflow produces 11 bones.
Sparse original vertex slices could not identify the arms. Inputs below 100,000
vertices now use 250,000 deterministic, area-uniform surface samples for landmark
estimation and neck-envelope measurement. Sampling never changes render geometry,
UVs, topology, or the heat solver. Dense inputs retain the established path.
The fitter still assumes an upright, forward-facing bust with lowered, cropped arms.

Direct heat failed on the UV-split input; the existing welded-copy retry weighted
all 6,837 unique vertices, with exact coincident transfer to the original mesh.
No cap faces or whole-body voxelization were needed. The posterior atlas placement,
head/jaw ownership and smooth upper-neck transition are enabled. Ordinary heat
remains on the shoulders and torso. The seam pass is still limited to two loops;
this mesh already has continuous UV-seam weights, so the pass makes no changes.

Saved-rig validation confirms unchanged geometry and UVs, normalized weights with
at most four influences, zero coincident-UV gaps in sampled motion, 2,076 tested
head/jaw vertices at full Head weight, and exact heat preservation on 6,581
lower-neck, torso and arm vertices. The GLB → UniMate NPZ → GLB round trip passes
240 frames and nine evaluated surface samples, with maximum surface error below
3.9e-7 scene units. These are authored diagnostic motions, not pretrained inference.

The rendered raised-arm test still exhibits torso/underarm stretching. The result
is provided for comparison without adding another torso/arm weight constraint.

Outputs are separate from the original stitched-bust evaluation:
`outputs/clean_bust_autorig/rigged.blend`, `rigged.glb`, `evaluation_stage.blend`,
`evaluation.mp4`, and individual `head.png`, `neck.png`, `arms.png` previews.
Source/model/media files remain local.

```sh
blender -b --factory-startup --python-exit-code 1 -P tools/autorig_bust.py -- \
  --input "$HOME/Downloads/clean-bust.glb" --output outputs/clean_bust_autorig
blender -b --factory-startup --python-exit-code 1 -P tools/validate_autorig_bust.py -- \
  --root outputs/clean_bust_autorig
PYTHONPATH="$PWD/.local_deps/site-packages:$PWD/.local_deps/Motion:$PWD" \
blender -b --factory-startup --python-use-system-env --python-exit-code 1 \
  -P tools/evaluate_autorig_bust.py -- --root outputs/clean_bust_autorig
blender -b --factory-startup --python-exit-code 1 -P tools/render_autorig_bust.py -- \
  --input outputs/clean_bust_autorig/rigged.blend --output outputs/clean_bust_autorig
python3 tools/encode_autorig_bust.py --root outputs/clean_bust_autorig
```
