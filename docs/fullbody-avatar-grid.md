# Nine full-character VRoid rigs

`--full-body` extends the existing fresh hand-rig flow to the complete nine VRoid
source characters. It adds hips and bilateral thighs, shins, feet and toes, plus a
nondeforming ground root: 52 bones total, including all 30 finger bones. Full-body
mode retains the lower geometry instead of applying the bust Boolean. The original
bust and hand-only modes still perform their destructive cuts.

VRM humanoid metadata supplies joint landmarks. The original skinning is used only
to prepare the A-pose and separated fingers, then its evaluated geometry is baked
and every source armature and weight is removed. The new skeleton receives fresh
heat weights, with the existing new-bone-distance fallback for vertices where
heat cannot solve. No character-specific weight masks are added. Floor-level parts
on full characters are excluded from the bust-base classification.

The established head/neck correction remains confined to two topology loops.
Arms, hands and non-thumb fingers are aligned into the final T-pose after binding,
with both bone rest transforms and mesh rest coordinates updated together. Thumb
rest transforms relative to the hands are retained. Blender's single-precision
rest-matrix reconstruction is allowed a 5e-5 basis tolerance; evaluated surfaces,
independent linear blend skinning, lengths and thumb transforms retain 1e-5 limits.

## Movement and camera views

Both videos contain 720 frames at 30 fps, 1920 × 1920, for 24 seconds:

| Time | Movement |
| --- | --- |
| 0–4 s | Greeting, head turns/nods, neck tilt, arm and wrist wave |
| 4–8 s | Spine/chest turns, bilateral arm reach |
| 8–12 s | Alternating march, hip turn, knee bend, ankle/toe flex |
| 12–16 s | Squat and rise with both ankle targets planted |
| 16–20 s | Wrist movement, finger curl and spread, thumb flex |
| 20–24 s | Wave, torso/hip weight shift and return to the rest pose |

A two-segment analytic leg solver generates the hip/knee rotations for ankle
targets. Those rotations are baked to the exported FK skeleton. The resulting
GLBs have no runtime dependency on an IK constraint or this demo script.

`fullbody_overview.mp4` keeps all nine characters visible. Each grid cell is fitted
over all 720 evaluated frames. `fullbody_tour.mp4` moves from the grid to head/neck,
leg, oblique full-body and hand views, holding each closeup before returning.
Imported glTF bone-display helper meshes are excluded from character bounds.
Outfits remain visible in these full-body scenes.

These are authored deformation demonstrations reconstructed through UniMate,
not pretrained motion or joint inference. They include skeletal outfit binding,
not cloth/hair simulation, collision-aware grasping or garment collision solving.
Loose sleeves and skirts can intersect during the larger poses.

The largest round-trip differences across all nine were 0.000280° rotation,
9.22e-6 joint position and 1.09e-5 evaluated surface distance. Independent rest
skinning stayed below 4.69e-6; planted-ankle drift stayed below 4.38e-6.
Positions use normalized character units (full height 2.65).

## Saved assets

All outputs live in `outputs/fullbody_avatar_grid/`:

- `fullbody_overview.mp4`, `fullbody_tour.mp4`: rendered videos.
- `fullbody_overview.blend`, `fullbody_tour.blend`: packed animated 3×3 scenes.
- `nine_fullbody_rigged_avatars.zip`: nine reconstructed full-character GLBs,
  source/license manifest and a README.
- `avatars/<id>/02_fresh_rig.blend`: editable fresh full rig and complete action.
- `avatars/<id>/04_aligned_rest.blend`: aligned, baked mesh and armature rest pose.
- `avatars/<id>/03_spread_bound.blend`: fresh binding before rest alignment.
- `avatars/<id>/01_cut_unrigged.blend`: legacy checkpoint filename; in full-body
  mode it contains the **uncut** full geometry with no old rig or weights.
- `avatars/<id>/00_source_spread.blend`: source preparation snapshot.
- `avatars/<id>/reconstructed/<id>-evaluation.glb`: actual UniMate round-trip result.

The nine source hashes match the retained acquisition manifest. The source VRMs,
previous bust/hand demonstrations and unrelated live Blender scene are unchanged.
Licenses and source identities are retained in the package manifest.

## Verification

- Saved unrigged checkpoints contain no armatures, vertex groups, weights or
  remaining Boolean/Armature modifiers.
- Independent rest-pose skinning validates all nine baked meshes, unchanged UVs,
  topology and weights, unchanged bone lengths, straight fingers and zero pose.
- All 90 fingertip sample sets are bound to their own digit chains. Full-body
  finger motion is measured after removing wrist/arm movement.
- All nine nondeforming roots have zero skin influence; weights are normalized.
  Foot samples favor their own leg chains. Evaluated lower-body surfaces move.
- Both ankle targets remain planted throughout each 120-frame squat while the
  hips lower; both feet lift during the marching phase.
- Every one of the 52 bones is compared on all 720 frames through GLB → UniMate
  NPZ → GLB. Surfaces are compared on 25 frames. Requested joint motion is also
  measured relative to its parent, excluding inherited ancestor movement.
- Both final videos are checked for dimensions/frame rate/frame count and fully
  decoded with ffmpeg. Reports are retained alongside this document.

```sh
export AVATAR_EVAL_ROOT="$PWD/outputs/fullbody_avatar_grid"
export AVATAR_SOURCE_ROOT="$PWD/outputs/complex_avatar_grid/sources"
blender -b --factory-startup --python-exit-code 1 -P tools/prepare_avatar_grid.py -- --full-body
blender -b --factory-startup --python-exit-code 1 -P tools/validate_finger_grid.py -- --full-body
blender -b --factory-startup --python-exit-code 1 -P tools/validate_fullbody_grid.py
PYTHONPATH="$PWD/.local_deps/site-packages:$PWD/.local_deps/Motion:$PWD" \
blender -b --factory-startup --python-use-system-env --python-exit-code 1 \
  -P tools/evaluate_avatar_grid.py -- --full-body
blender -b --factory-startup --python-exit-code 1 -P tools/render_fullbody_grid.py --
blender -b --factory-startup --python-exit-code 1 -P tools/render_fullbody_grid.py -- --tour
python3 tools/encode_fullbody_grid.py
```

Validated with Blender 5.2.0 LTS (`fbe6228777e7`). Existing capability gates for
EEVEE and layered/legacy actions remain in place; this task does not change the
project's declared Blender support range.
