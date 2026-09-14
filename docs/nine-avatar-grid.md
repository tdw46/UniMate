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
8. Correct clothing by region using `avatar_apparel_weights.py`: necklace beads
   and scarf receive full Neck weights; jacket and backpack/clothing components
   receive barycentric weights from the freshly bound body surface. Exclude Head
   from torso garments and cap Neck influence to the collar region. Blend hood
   bases toward Neck/Chest, and anchor remaining cut-base fittings to Root.
9. Author head, neck, and arm diagnostics and execute actual UniMate
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
meshes retain their original faceted design. Garment collision and cloth
simulation are outside this test.
Facial expressions and finger animation are outside this 13-bone test.
Motion is authored diagnostic animation, not model-generated animation:
upstream still lists pretrained weights as unreleased on September 11, 2026.

### Clothing regression checks — September 14, 2026

The initial round-trip checks preserved incorrect weights as faithfully as
correct ones. Monk's necklace beads shared a rigid Head assignment with facial
hair, while SciFi's disconnected jacket torso had a failed heat-bind result
dominated by Neck. Adventurer's detached clothing also contained stray Head
and Neck weights. Object names and original bone-parent labels alone were
insufficient to classify these mixed meshes.

The correction retains fresh body weights as the transfer source; it never
uses the avatars' original weights. `avatar_apparel_weights.py` uses no avatar
identifiers, source object-name matches, component vertex counts, or fixed
component indices. It identifies a connected torso/shoulder carrier by spatial
coverage, then classifies connected regions relative to skeleton proportions.
Repeated compact regions arranged around the neck form a jewelry assembly;
area-weighted surface moments distinguish those regions from elongated hair.
Circumferential collars and continuous head-to-neck shells receive neckwear
and hood treatment respectively. Remaining body-adjacent clothing receives
nearest-triangle barycentric weights from the new body carrier. Head influence
is excluded from torso garments and Neck influence is limited anatomically.

Hair and beards remain attached to Head; necklace beads and the scarf receive
Neck weight 1.0. Torso garments receive local body, shoulder, arm, and collar
influences. Old bone-parent labels may suggest a rigid attachment, but garment
geometry takes precedence. Explicit `binding_role` metadata is supported for
ambiguous geometry (`body`, `head`, `neckwear`, `torso`, `hood`, or `rigid`).
The algorithm expects meshes baked into the new rig's local space, Z up and X
lateral, and the semantic bone names used by this 13-bone flow. It is a general
geometric heuristic, not guaranteed semantic recognition of arbitrary clothing;
mixed or ambiguous assets may require explicit region information.

`validate_avatar_apparel.py` clears animation and applies isolated bone motions
to evaluated meshes. It verifies normalized weights with at most four
influences across all nine avatars, no lower-body movement from Head/Neck,
rigid Neck following for the 1,460 necklace vertices and 29 scarf vertices,
and independent Spine/Chest/Neck/shoulder/arm responses for the 378 jacket and
647 Adventurer garment vertices. Cut-base fittings must remain stationary
under isolated arm motion.

The [original-weight audit](avatar-grid-apparel_validation_before.json)
reproduces the failures. The [corrected-weight audit](avatar-grid-apparel_validation.json)
passes: Head-only displacement is zero for the tested neckwear and garments,
and Neck-only displacement is zero below the collar on both torso garments.
For SciFi, the latter displacement fell from 0.55825 scene units to zero.
The full GLB → NPZ → GLB checks are rerun after correction before rendering.

`test_avatar_apparel_generalization.py` tests the same algorithm with every
mesh renamed, object order reversed, triangulation/subdivision changed, and
geometry/skeleton scaled and translated together. Additional procedural bodies, necklaces,
collars, jackets, and elongated hair decoys vary bead count, tessellation, and
scale without using any downloaded avatar geometry. The fixture-specific
selections in `validate_avatar_apparel.py` are test ground truth only and are
not inputs to the binding algorithm.
The [generalization audit](avatar-grid-apparel_generalization.json) records
all nine invariance cases and three procedural cases, including an ambiguous
pendant with an explicit role.

## Deliverables

Outputs are local under `outputs/avatar_grid/`:

- `nine_avatar_grid.mp4`: 24 seconds, 1920×1920, 30 fps; front then oblique.
- `avatar_grid_front.mp4` and `avatar_grid_oblique.mp4`: 12 seconds each.
- `apparel_before_after.mp4`: 24-second Monk/SciFi close-up comparison, with
  the original evaluation on the left and corrected weights on the right.
  Generated when the preserved earlier videos exist in `before_apparel_fix/`.
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
blender -b --factory-startup --python-exit-code 1 -P tools/validate_avatar_apparel.py
blender -b --factory-startup --python-exit-code 1 -P tools/test_avatar_apparel_generalization.py
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/evaluate_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_avatar_grid.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_avatar_grid.py -- --oblique
python3 tools/encode_avatar_grid.py
```

The encoder decodes every delivered video and validates frame count, frame
rate and duration. Export caches are keyed by the freshly rigged source GLB's
hash, so changed preparation does not silently reuse old NPZ data.
