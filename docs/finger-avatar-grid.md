# Nine VRoid hand-rig evaluations

This adds optional `--hands` support to the existing destructive VRoid bust flow.
The nine source models are the locally retained complex-avatar-grid VRMs. They
remain unchanged. The output is isolated in `outputs/finger_avatar_grid/`.

Each source is posed with separated fingers while its original skinning still
exists. A measured palm frame supplies mirrored fan angles for Thumb, Index,
Middle, Ring and Little. The posed surface is baked, the original rig and every
old weight are deleted, and the existing destructive Boolean bust cut is applied.
The saved `01_cut_unrigged.blend` is verified to contain no armatures, weights,
or remaining Boolean/Armature modifiers.

The fresh skeleton has 43 bones: the existing 13 torso/arm/hand bones plus 30
finger bones. VRM humanoid metadata supplies joint locations; terminal child
nodes supply fingertip positions. Imported glTF bone display tails are not valid
finger directions, so they are never used for these chains. VRM 0 and VRM 1 thumb
schemas are supported; incomplete/ambiguous finger metadata fails explicitly.
Fresh ordinary heat binding uses the spread geometry. Existing apparel handling
and the maximum two-loop head/neck seam correction are retained.

After binding, the upper arms, forearms, hands and four non-thumb finger chains
are posed parallel to canonical ±X. The evaluated vertex positions are captured,
the posed skeleton becomes the new rest skeleton, and those captured positions
become the new mesh rest coordinates. Thumb transforms relative to each hand
are preserved. No pose delta remains in the aligned-rest checkpoint. This handles
the mesh reset associated with Blender's
[Pose as Rest Pose operation](https://docs.blender.org/manual/nl/5.2/animation/armatures/posing/editing/apply.html).

The installed Auto-Rig Pro smart-finger implementation was reviewed for its
hand isolation/canonical-frame and voxel-centroid approach. The current VRoid path
uses its authoritative joint metadata, so geometry-only finger detection and
hand voxelization are unnecessary here. The new implementation has no Auto-Rig
Pro dependency or copied source. It does not claim finger landmark inference on
arbitrary unrigged meshes or pretrained UniMate inference.

## Review artifacts

- `hand_grid_dorsal.mp4` and `hand_grid_palm.mp4`: 3×3 grids, both hands per avatar,
  1920×1920, 30 fps, 12 seconds.
- First four seconds: wrist flexion and deviation.
- Next four seconds: separate Thumb, Index, Middle, Ring and Little curls.
- Final four seconds: combined fist/thumb motion followed by spreading.
- `hand_grid_dorsal.blend` and `hand_grid_palm.blend`: packed hand-closeup scenes.
- `avatars/<id>/02_fresh_rig.blend` and `.glb`: complete rigged busts with outfits.
- `avatars/<id>/00_source_spread.blend`, `03_spread_bound.blend` and
  `04_aligned_rest.blend`: reviewable preparation/rest checkpoints.

Sleeves are hidden on the hand-display copies to expose the deformations. Skin
textures, including painted gloves, rings and nails, remain visible. Full avatar
assets retain their outfits. Every hand slot is fitted over all 360 evaluated
frames to prevent neighboring hands from overlapping. The authored curl tests
are deformation probes, not a collision-aware grasp/contact solver.

## Validation

The saved spread/aligned pair is checked with an independent linear-blend-skinning
calculation. This verifies the updated mesh rest coordinates against the new bone
rest matrices, unchanged UVs/topology/weights, bone lengths, thumb-relative rest
transforms, straight non-thumb chains and identity pose bases. All 90 sampled
fingertips receive full weight from their own finger chains and move during their
individual curl tests. Local seam limits are validated before the deliberate rest
geometry change.

All 43 bones on all nine avatars pass 360-frame GLB → UniMate NPZ → GLB transform
comparisons, with evaluated surfaces checked on 15 frames including each isolated
finger curl. This also reads back the unrigged checkpoints rather than relying on
in-memory counters. Final videos are frame-counted and fully decoded.

```sh
export AVATAR_EVAL_ROOT="$PWD/outputs/finger_avatar_grid"
export AVATAR_SOURCE_ROOT="$PWD/outputs/complex_avatar_grid/sources"
blender -b --factory-startup --python-exit-code 1 -P tools/prepare_avatar_grid.py -- --hands
blender -b --factory-startup --python-exit-code 1 -P tools/validate_finger_grid.py
PYTHONPATH="$PWD/.local_deps/site-packages:$PWD/.local_deps/Motion:$PWD" \
blender -b --factory-startup --python-use-system-env --python-exit-code 1 \
  -P tools/evaluate_avatar_grid.py -- --hands
blender -b --factory-startup --python-exit-code 1 -P tools/render_finger_grid.py --
blender -b --factory-startup --python-exit-code 1 -P tools/render_finger_grid.py -- --palm
python3 tools/encode_finger_grid.py
```

Model/texture/media files remain local. Source identities and original licensing
records are in `outputs/complex_avatar_grid/sources/manifest.json`.
