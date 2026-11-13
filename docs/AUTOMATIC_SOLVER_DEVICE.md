# Automatic skinning device selection

New auto-rig jobs default to Automatic: the isolated solver worker probes its
hash-verified GPU library and initializes a Metal/Vulkan compute device. If no
supported device is available, that library executes its CPU path. Platforms
without a bundled GPU library select their bundled CPU library. Explicit CPU/GPU
choices and the existing saved enum numbers are preserved. Device detection does
not run in panel draw or Blender's main thread.

Only availability probing can fall back. Package verification failures, bind
errors, cancellation and invalid inputs remain errors; there is no silent retry
of a failed solve. Binding reports record requested and actual backends, adapter
and any fallback reason. Landmark prediction settings are unchanged.

Verified on Mac with Blender 4.2/Python 3.11 and Blender 5.2/Python 3.13:
Automatic output equals forced Metal output exactly; injected adapter absence
produces CPU output identical to forced CPU. Both BVT and Hallway register the
Automatic default, preserving CPU=0 and GPU=1. Eight focused selection tests and
all BVT locale checks pass. Live BVT/Hallway owner reload restored existing scene,
frame, explicit backend choices, preview and physics without saving the file.

Windows implementation shares the same loader and worker; native libraries and
renderer paths were not changed. The paired laptop was unreachable on 2026-11-13,
so a fresh native Vulkan run remains open. Reuse `gpu_default_test.py` from the
turn-246 evidence directory in a project-owned Windows runner subfolder; require
Automatic to report Vulkan and compare its arrays to forced GPU, plus the
no-adapter fallback comparison to forced CPU. No claim of new Windows acceptance.

Evidence: `/Users/tylerwalker/Documents/Codex/2026-10-29/autoskin-rea-research/delivery/zzz-246`.
