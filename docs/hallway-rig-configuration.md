# Hallway rig configuration

Select a generated armature or one of its skinned meshes, then open **N > Hallway**.
Use **Apply Rig Settings** after editing values; this operation supports Undo.
Settings belong to each rig and are saved with the `.blend`. Opening a file does
not regenerate weights, assign spring centers, or overwrite its configuration.

- **Skirt Follow** controls the existing export-compatible leg constraints, with
  a default of 100%. Additional follow groups use the same group metadata on
  their owner bones; the UI discovers those groups during configuration.
- **Skirt Thickness** scales each skirt capsule relative to its fitted radius.
  It does not compound on repeated applies or affect hair colliders. Each radius
  is capped by spring endpoint and skirt surface clearance in the rest pose.
  The operator reports how many radii were capped. This prevents initial
  collider overlap; it does not guarantee clipping-free motion.
- **Skirt/Hair Springs** expose drag, root stiffness and gravity separately.
  Stiffness preserves its taper along the chain. Existing spring centers remain
  unchanged; generated springs have no center unless explicitly requested.
- **BVT Physics On/Off** and **Reset** control BVT through its simulation operator.
  Hallway registers no live solver, timer, or simulation handler; without BVT,
  rig configuration remains available and the physics buttons are disabled. The preview
  toggle is scene-wide, while configuration edits affect only the selected rig.
- Collection visibility controls show or hide **Physics**, **Constraints**,
  **Deform**, and **Controls**. Physics includes fixed attachment roots and tips;
  constraint helpers take precedence over their export-required deform flag.
  Other weighted bones are Deform; remaining bones are Controls. Custom artist
  collections are preserved. Older Blender versions use four bone layers.

`avatar_springs.generate_secondary()` initializes settings and collections;
body-only and bust generation also organize their bones. **Refresh Bone
Organization** upgrades existing generated rigs without changing their weights.

The development loader in `blender_addon/` registers the panel and BVT controls.
Run `tools/install_hallway_addon.py` inside Blender to install it in the canonical
shared addon folder. It references this checkout through its adjacent
`source.json`; implementation stays here, and moving the checkout requires
rerunning the installer. No scene is saved by installation.

Validated in Blender 5.2.2 using `tools/validate_hallway_rig.py`: target isolation,
VRM constraint flags, safe thickness limits, idempotence, registration cycles,
and save/reopen persistence. The legacy layer fallback is capability-gated;
that runtime was not available for validation.

All previews, numerical physics checks, and demo recordings now use BVT's
solver. The previous project solver has been deleted. Spring, collider,
weight and constraint generation remain our own code.
