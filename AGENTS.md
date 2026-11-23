# Generated media housekeeping

- At the start of every turn, inspect this repository's prior generated media and remove superseded renders, previews, screenshots, and demo videos from earlier turns. Do not accumulate obsolete media versions.
- Keep the current final video deliverables. When a new video supersedes an older one, verify the replacement before deleting the old video. If an older clip is still needed as a current comparison/control, explicitly retain it with the current deliverables.
- Delete video frame sequences on the same turn that creates them, after checking encoding, expected frame count, and complete decoding. Remove temporary stills and media intermediates before finishing; only final video media needs to survive the turn.
- This is generated-media cleanup, not source-asset deletion. Preserve original/reference images and textures, downloaded assets, models and editable scenes, code, test fixtures, logs/manifests, backups, and uncommitted work. Check ownership and active jobs before removing files; preserve anything still in use or whose role is uncertain.
- Record removed paths and reasons, removed allocation, and the separately observed free-space change. Keep the retained video paths discoverable in the current output README.
