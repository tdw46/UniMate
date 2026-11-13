# Auto-Rig Pro binding excerpts

Source: locally installed Auto-Rig Pro 3.78.34, by Artell.
Official project: https://www.lucky3d.fr/auto-rig-pro/doc/
Source file: `src/auto_rig.py`.
The source-folder license grants GPL-2.0-or-later; the package root states GPL-3.0. These excerpts are distributed under GPL-3.0-or-later, compatible with both notices.

`core.py` retains the split/automatic-heat/join/nearest-transfer body verbatim (indentation only). Its small function wrapper and the host utilities are new. The adapter supplies disposable geometry and a generated skeleton; it does not load Auto-Rig Pro assets or register its panels.

This is the Heat Maps binding core, not the complete Auto-Rig Pro rig generator, facial rig, voxel solver, twist controls or weighting guarantee. Differences and validation belong in the host audit. Original skeletons are never binding inputs.
