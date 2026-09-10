# Blender 5.2 bust evaluation

The fork supports a tested **GLB → UniMate export NPZ → reconstructed GLB**
path in Blender 5.2.0 LTS. This is a data-pipeline test, not a model-inference
result. On September 10, 2026, upstream's README still marked pretrained
checkpoints and processed data as coming soon. No model checkpoint was loaded.

## Test assets and results

Two original procedural mannequins include torso, head, neck and arms:
`slender` has 13 bones; `broad` has 14, including a second neck joint.
Each has three authored diagnostic animations at 30 fps, 120 frames each:

- Head yaw and nodding.
- Neck lateral tilt and forward/backward bending.
- Bilateral arm raises and elbow flexion.

All six cases passed. Every bone's neutral-relative world rotation and joint
position was compared over every frame after export, reconstruction and GLB
reimport. The evaluated skin was also compared in both directions at frames
0, 30, 60, 90 and 119 using nearest-vertex distances.

Maximum observed differences:

| Measurement | Maximum | Acceptance tolerance |
| --- | ---: | ---: |
| World rotation | 0.000040 degrees | 0.1 degrees |
| Joint position | 0.000000879 scene units | 0.0001 |
| Evaluated surface | 0.000001470 scene units | 0.0001 |

See [the complete results](blender-5.2-bust-results.json) for per-bone values.
The original GLB is the comparison baseline; automatic weight normalization
during its initial creation is outside this round-trip measurement. These
stylized, automatically weighted fixtures are useful diagnostics, not a
production anatomical deformation benchmark. Pruning is disabled so every
authored bone remains available for comparison.

## Compatibility changes

- Discover pose curves in layered actions, with legacy `Action.fcurves` access
  retained for older Blender versions.
- Create reconstruction curves in a slot/channel bag when that API exists.
- Bind the matching action slot explicitly and reject ambiguous matches.
  Avoid reassigning an already selected slot: the tested Blender build crashed
  during redundant assignment in the focused regression test.
- Set reconstructed pose channels to quaternion mode and update F-curves.
- Fix single-file export's uninitialized failure counter and return failure
  when a directory export contains failed assets.

Only the tested export/reconstruction path is covered. The upstream ML
environment remains pinned to Python 3.10, CUDA and `bpy==4.0.0`; it was not
installed into Blender. Other upstream renderers and full model inference
have not been certified for Blender 5.2. The legacy curve adapter has a focused
test double; a Blender 4.0 runtime test has not been performed.

API references consulted: [5.2 action slots](https://docs.blender.org/api/5.2/bpy.types.ActionSlots.html),
[5.2 channel bags](https://docs.blender.org/api/5.2/bpy.types.ActionKeyframeStrip.html),
[5.2 armature deformation](https://docs.blender.org/manual/en/5.2/modeling/modifiers/deform/armature.html),
[4.0 actions](https://docs.blender.org/api/4.0/bpy.types.Action.html), and
[4.0 action manual](https://docs.blender.org/manual/en/4.0/animation/actions.html).

## Reproduce

Run in a disposable Blender process. The scripts reset their process to factory
state; do not execute them inside a session containing unsaved work.

Create an isolated dependency directory, using Blender's bundled Python with
`uv pip install --python <bundled-python> --target .local_deps/site-packages
loguru tqdm matplotlib imageio imageio-ffmpeg scipy`. Clone
`https://github.com/inbar-2344/Motion.git` into `.local_deps/Motion` (tested commit
`ac236251f90e5ca37c444c53ad383fc85de6d833`). Add both directories and the repository
root to `PYTHONPATH`; pass `--python-use-system-env` to Blender for the pipeline.

```sh
blender -b --factory-startup --python-exit-code 1 -P tools/create_bust_fixtures.py
blender -b --factory-startup --python-exit-code 1 -P tools/test_blender_actions.py
blender -b --factory-startup --python-use-system-env --python-exit-code 1 -P tools/evaluate_bust_pipeline.py
blender -b --factory-startup --python-exit-code 1 -P tools/render_bust_evaluation.py
python3 tools/encode_bust_renders.py
```

`ffmpeg` and `ffprobe` must be on PATH for encoding. The encoder validates
decoded frame counts, fps and duration. Output stays in the ignored
`outputs/bust_evaluation/` directory:

- `unimate_bust_evaluation.mp4`: 24-second, 1600×900, 30 fps compilation.
- `{head,neck,arms}_{front,oblique}.mp4`: six individual four-second clips.
- `evaluation_*.blend`: six self-contained scenes that render the reconstructed
  GLBs, with diagnostic provenance burned into their camera views.
- `slender.glb`, `broad.glb`, source `.blend` files, NPZ exports and reconstructed
  GLBs, plus numerical and render-validation JSON files.

The rendered views use Workbench studio shading for clear surface inspection.
No user scene or Blender preferences need to be modified or saved.
