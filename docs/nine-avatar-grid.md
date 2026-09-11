# Nine-avatar Boolean bust and fresh-rig evaluation

Nine freely available Quaternius avatars were posed, converted to upper-body
busts with applied Boolean modifiers, stripped of their original rigs and
weights, freshly rigged, and passed through the same UniMate reconstruction
flow as the earlier mannequin test. The final presentation is a 3×3 grid.

## Sources

All nine models are CC0-1.0 according to the creator's pack pages:

| Avatar | Creator's pack |
| --- | --- |
| Cleric, Monk, Ranger, Rogue, Warrior, Wizard | [RPG Character Pack](https://quaternius.com/packs/rpgcharacters.html) |
| Adventurer, SciFi, Witch | [Ultimate Modular Women](https://quaternius.com/packs/ultimatemodularwomen.html) |

The creator's Google Drive downloads were temporarily quota-limited, so the
files were retrieved from pinned public GitHub mirrors. The women's mirror
documents minor scene organization adjustments. The original pack licenses
and URLs, exact download URLs, and SHA-256 hashes are retained in
[the source manifest](avatar-grid-sources.json). No source avatar files were
overwritten. Handheld weapons were excluded from the bust evaluation.

## Preparation

1. Clear source animation and place the arms in a 20-degree A-pose.
2. Bake evaluated geometry, preserving UVs, materials, and rigid attachments.
3. Delete the original armatures, actions, and all original vertex groups and
   weight assignments.
4. Apply an Exact Boolean Difference using a box below the upper abdomen.
   Apply the modifier destructively and remove the cutter. This also removes
   lower-body objects that fall entirely inside the cutter.
5. Weld coincident seam vertices while retaining per-corner UVs; clear weights
   again and save `01_cut_unrigged.blend` for each avatar.
6. Create a new 13-bone armature, then bind with new automatic weights.
   Anatomical positions recorded during posing place the fresh bones; no
   source armature datablock or old weight values are reused. Rigid head and
   armor attachments receive new rigid weights on the corresponding new bone.
7. Fill unweighted vertices using distances to the new bones. Adventurer
   required this fallback for 2,055 vertices. Lateral arm/sleeve vertices are
   weighted against the new arm chain to keep bulky clothing moving with arms.
   Limit to four influences and normalize every vertex.
8. Author head, neck, and arm diagnostics and execute actual UniMate
   `GLB → export NPZ → reconstructed GLB` processing.

The destructive preparation applied **21 Boolean modifiers** and removed
**378 original bones** and **102,475 original weight assignments**. Saved
unrigged checkpoints were reopened to verify zero armatures, vertex groups,
weight assignments, and remaining Boolean/Armature modifiers.

## Evaluation and limits

All nine avatars passed all-bone comparisons over 360 frames each at 30 fps.
Frames 0–119 test head turns/nods, 120–239 test neck bends/tilts, and 240–359
test bilateral arm raises and elbow flexion. Each intended phase was checked
for actual evaluated joint motion, not only for the presence of keyframes.

The comparison baseline is each freshly rigged GLB before UniMate processing.
Every bone's neutral-relative world rotation and world joint position is
checked at every frame. Evaluated surface vertices are compared in both
directions at 13 frames spanning all phases.

| Measurement | Maximum observed | Tolerance |
| --- | ---: | ---: |
| World rotation difference | 0.0000561 degrees | 0.1 degrees |
| Joint position difference | 0.00000142 scene units | 0.0001 |
| Surface difference | 0.00000214 scene units | 0.0001 |

Full evidence: [preparation audit](avatar-grid-preparation_validation.json)
and [pipeline results](avatar-grid-pipeline_validation.json).
The [video audit](avatar-grid-render_validation.json) confirms all 720 frames
of the combined video and each individual demo. The
[saved-scene audit](avatar-grid-scene_validation.json) confirms nine new rigs,
117 total bones, packed textures, and no unapplied Boolean modifiers in both views.

These measurements verify preservation through the reconstruction pipeline;
they do not certify production-quality anatomical skinning. The stylized
meshes retain their original faceted design. Some small detached garment
fittings, particularly on Adventurer, remain candidates for manual cleanup.
Facial expressions and finger animation are outside this 13-bone test.
Motion is authored diagnostic animation, not model-generated animation:
upstream still lists pretrained weights as unreleased on September 11, 2026.

## Deliverables

Outputs are local under `outputs/avatar_grid/`:

- `nine_avatar_grid.mp4`: 24 seconds, 1920×1920, 30 fps; front then oblique.
- `avatar_grid_front.mp4` and `avatar_grid_oblique.mp4`: 12 seconds each.
- Nine `{avatar}_demo.mp4` files: individual cropped cards from the grid.
- `avatar_grid_front.blend` and `avatar_grid_oblique.blend`: self-contained
  scenes with packed textures and the actual reconstructed animations.
- Per-avatar `01_cut_unrigged.blend`, `02_fresh_rig.blend`, fresh-rig GLB,
  exported NPZ, reconstructed GLB, and preparation JSON.
- Render-validation JSON, poster, contact sheet, and image sequences.

## Reproduce

Use Blender 5.2 and the isolated pipeline dependencies documented in
[the earlier evaluation](blender-5.2-evaluation.md). Run only in disposable
Blender processes; preparation and evaluation reset their own scenes.

```sh
python3 tools/download_avatar_sources.py
blender -b --factory-startup --python-exit-code 1 -P tools/prepare_avatar_grid.py
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/evaluate_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_avatar_grid.py -- --oblique
python3 tools/encode_avatar_grid.py
```

The encoder decodes every delivered video and validates frame count, frame
rate and duration. Export caches are keyed by the freshly rigged source GLB's
hash, so changed preparation does not silently reuse old NPZ data.
