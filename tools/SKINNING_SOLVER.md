# Hallway Solver

Automatic Rigging uses **Hallway Solver** to create skin weights after predicting the
humanoid joints. The solver runs locally in a separate process. It requires no
separate solver installation, compiler or original commercial add-on.

## Use it

1. Open **Automatic Rigging** and complete its existing engine/model setup.
2. Select the character meshes in their neutral rest shape, facing front (-Y).
3. Leave **Skin Weights → Hallway Solver** and **Skinning Device → CPU** selected.
   Choose **GPU** to use Metal on Apple Silicon or the native Windows GPU backend.
   The landmark prediction device is a separate setting.
4. Choose the desired secondary physics options, then **Generate Rig in New Scene**.
5. Review joint placement, bending and physics on the generated copy before export.
   Undo removes the generated scene. Escape cancels either worker stage.

**Blender Weighting** retains the previous heat-weighting route and its cleanup.
Choose it explicitly on an unsupported solver platform. A failed solver job reports
an error and preserves the source scene; it does not silently switch methods.

## What is preserved

- Landmark prediction receives geometry only. The solver receives that geometry
  and the newly predicted skeleton, never original skin weights or rig landmarks.
- The original scene, mesh coordinates, shape keys, materials and weights are retained.
- Generated weights use ordinary linear armature deformation. **Preserve Volume is off.**
- Cranial ownership cleanup runs after binding. Existing secondary-chain, apparel,
  hip/skirt and collider setup remains in its existing stages. Heat-specific retry
  and repair logic stays with Blender Weighting; it is not applied over solver output.
- Defaults use production density 92, adaptive depth 2, 64 sampling neighbors,
  up to 8 influences and finger isolation. They are recorded in the rig report.
  This integration does not lower quality to obtain faster timing.

## Supported bundle and evidence

The same independent engine is branded **BVT Solver** in the other add-on.
The packages contain CPython 3.11+ native libraries for Apple Silicon macOS and
Windows x64, CPU and GPU variants. Native loading checks the platform and SHA-256
manifest in the isolated worker. Other platforms retain Blender Weighting.
The solver is separate from Beyond NPR; its GPU backend does not select or alter
Blender's viewport backend.

The 2026-11-11 integration checks are recorded in
[skinning-integration-218.json](skinning-integration-218.json):

| Check | Result |
| --- | --- |
| Final packaged Mac CPU / Metal bind | Exact influence arrays against corresponding original controls |
| Final packaged Windows CPU / RTX 4090 Vulkan bind | Exact influence arrays against corresponding original controls |
| Five rest-character bindings in BVT, Blender 5.2.2 | Passed source preservation, normalized weights, secondary setup, numeric pose and rollback checks |
| Hallway binding in Mac and Windows Blender 5.2.2 | Passed; Windows secondary generation remains substantially slower |
| Real BVT and Hallway Generate operators on Mac 5.2.2 | Mesh-only prediction, solver, Undo and reload passed; test windows did not take focus |
| BVT binding in stock Mac Blender 4.2.0 | Passed with geometry-only input, Blender's Python 3.11, pose and rollback checks |
| Worker lifecycle | Cancellation, failure, changed geometry and changed scene checks passed |
| Release inclusion | Windows libraries, native source and license notices included; unrelated archives excluded |

These checks establish execution and measured parity of the solver routes. They
are not a new all-pose visual approval, VRM export round trip, native Windows BVT UI
qualification, or every API/error-boundary proof. The reconstruction ledger remains
58/59 full API contracts. Tyler accepted the remaining speed gap for this integration;
no source-speed-parity claim is made. Existing historical predictor tests in the
parent README do not qualify this new weighting stage.

## Maintenance

Host integration: `tools/ui_mesh_rig.py`, `tools/rebuild_mmd_character.py` and
`tools/mesh_landmarks/`. Independent package: `tools/beyond_skinning/`. Its `manifest.json` pins native
artifacts; `native-source.zip` contains the locked Rust/WGSL build source and Windows
adapter patch. `dependency-licenses.zip` and `NOTICE.md` retain provenance.
Do not substitute the original binary or refresh hashes to admit untested binaries.

The installed Hallway loader points to this project's `tools` directory; it does not
copy implementation files. A portable Hallway distribution must include the entire
`tools/beyond_skinning/` package, including native libraries and source/license
archives. BVT's `tools/check_skinning_package.py` also accepts this package path.
The two product copies must remain identical; UI branding belongs to the host adapter.

## Hair-chain contact work (222)

The shared mesh-only hair planner now separates substantial forks joined by scalp
geometry. Per-chain head/torso/arm capsule contacts and outward guards preserve
rest geometry, weights and artist-owned spring groups. The ordinary Hair gravity
default remains 0.035; zero gravity is diagnostic only. Hallway native VRM repeat
installation and ownership checks pass on Mac Blender 5.2.2; BVT checks pass on
4.2.0. The Nicole fresh-generation regression has 38 chains instead of 28.

Dynamic contact qualification remains open: about 6 mm of residual penetration
in a head-turn stress test. Do not describe these candidate contacts as clipping
free. The owning BVT evidence is `documentation/auto-rig/nicole-hair-222.json`;
research logs and diagnostic clips are in `delivery/nicole-222/`.

The subsequent 223 correction identifies the actual thick tied ponytails by a
shared hair base-color image plus head attachment and hanging geometry. Their
unnamed material had been skipped entirely; earlier chain-count/loose-hair checks
did not establish their coverage. Explicit nonhair roles, semantic face/skin and
lower garment surfaces sharing an atlas remain excluded. Detection is shared by
secondary planning, head-cap binding and hair exclusion from body colliders.
Nested loose-hair fork splitting was also corrected. See BVT
`documentation/auto-rig/nicole-ponytails-223.json` for actual per-tail weight,
native spring and deformation checks. Collision acceptance remains open.
