# Secondary geometry increment 225

Shared mesh-only fitting now covers virtual secondary seams, small surface
satellites, asymmetric torso straps and branched hanging accessories. No original
rig or weights are generation inputs. Hands/feet are excluded before secondary
admission. Existing per-product naming and preview ownership remain unchanged.

The automatic rigging panel also offers opt-in Generate Breast Bones. It uses
paired frontal volume and center-separation tests, redistributes only freshly
computed torso weights and installs standard VRM spring metadata/colliders.

BVT and Hallway region plans match on the disposable Nicole fixture. Hallway
breast creation, VRM springs, settings initialization and UI registration pass
Blender 5.2. Shared geometry regression passes Blender 4.2/5.2. BVT's fresh public
operator passes from zero source bones/groups; original coordinates and source
file are preserved. Full dynamic surface contact, Windows/export and new pose
visual review remain separate open checks.

Detailed evidence:
`/Users/tylerwalker/Documents/Codex/2026-10-29/autoskin-rea-research/delivery/nicole-225/`

## Follow-up 226

Preserve all local seam correspondences through guide cuts; tiny floating patches
receive coherent hair attachments even when partly assigned already. Narrow
hanging torso strips now use a single downward height-section chain, while broad
ornaments retain branches. Breast tips use the actual forwardmost surface level;
bases use a surrounding attachment-loop fit. These are shared geometry-only
changes, with BVT/Hallway plan parity and 4.2/5.2 regression evidence retained under
`delivery/nicole-226` in the same research project. Full dynamic surface contact
and native Windows/export acceptance remain open.

## Turn 227 — breast volume coverage

Shared geometry-only breast weighting now uses a strong interior with independent perimeter, torso-attachment and center-seam fades. Publish inherited transforms before fitting the breast bones. The pivot is inside the lobe and is not its rear weight boundary. Hallway's generated breast weights match BVT within 2e-7 and native breast physics installation passes. Nicole's source rig/weights remain validation-only; fresh BVT generation begins without either. See BVT `documentation/auto-rig/nicole-breast-volume-227.json` and research `delivery/nicole-227/` for coverage, placement, protected-weight and runtime evidence. Full platform/export/contact qualification remains open.

## Turn 228 — shared center and head accessories

Mirrored paired breast-center weights (25% torso, 37.5% each breast) and material-independent compact head-sheet detection. Four bow branches are generated from geodesic rest geometry, with two deform bones per branch. Independent Head Accessories controls preserve stiffness 1, drag .5, gravity 0 while ordinary hair retains .035 gravity. Fitted head/neck/torso capsules remain clear at rest. BVT/Hallway paired-weight, geometry, registration and settings parity passes. Original rig/weights remain validation-only; the original-reference spring constants are official VRM conversion defaults, not exact MMD-system parity. Evidence lives in research `delivery/nicole-228/` and BVT `documentation/auto-rig/nicole-center-bows-228.json`.

## Turn 230 — softer physics with rest-clear contact margins

Shared Hair defaults are stiffness .3, drag .3, gravity .0105; hanging Accessory defaults are .12, .15, .035. Hair keeps a downward gravity/restoring-force balance. Head Accessories stay at zero gravity; breast/skirt settings remain unchanged. Generated hair hit radii grow by at most 2% of character height, capped against all assigned rest colliders with a 0.15% gap and an idempotent saved base. Body colliders and assignments are unchanged.

Fresh Hallway generation and BVT/Hallway defaults parity pass; 4.2/5.2 radius/contact regression passes. Under identical drag, hair/strap RMS movement is 1.89x/1.94x the prior fixture. Twelve-pose sampled surface penetrations over 1 mm fall from 774 to 43. Original rig and weights remain comparison-only. This is not complete collision/export/native Windows acceptance. Reproducible 36-second comparison and evidence: research `delivery/nicole-230/`, BVT `documentation/auto-rig/nicole-physics-tuning-230.json`.

## Seam-aware four-influence smoothing and compact accessories — turn 231

BVT and Hallway share rest-space seam welding for weight diffusion, a smoothed four-influence limit, stronger accessory smoothing and lighter family-scoped hair smoothing. Mesh topology is untouched. Original weights/bones remain validation-only. Hanging Straps keep stiffness .12 / drag .15 / gravity .035; compact Accessory restores 1.2 / .4 / .035. Unbranched accessory segments form continuous VRM springs, retaining forks, native parameters and collider ownership. Existing rigs are upgraded only by explicit operation; loading a file does not silently rewrite it.

Blender 4.2/5.2 numeric checks pass. Hair and accessory weight seam mismatches are zero; 4,411 protected extremity weights and unrelated weights remain exact. Toy acceleration proxy decreases about 59%, but residual branch jitter remains. The strict sampled contact no-regression gate fails: 43 to 48 pairs over 1 mm, max depth .01428 to .01507 m. This is not collision acceptance.

No simulation algorithm change is retained. UniVRM processes joints within each chain and schedules separate chains in parallel; VRM 1 explicitly leaves branching-chain order undefined. Native UniVRM reproduction is required before an upstream bug claim or PR. BVT was safely reloaded and only the current generated review rig updated without saving. See [evidence](/Users/tylerwalker/Documents/Blender/extensions/user_default/beyond_vrm_extension_suite/documentation/auto-rig/nicole-secondary-smoothing-231.json) and its source-review/video records. Full platform/export/contact gates remain open.

## Turn 232 — sibling model execution matrix and host naming

Anby, Belle, Grace and Wise pass fresh public generation in both Hallway and BVT (eight Mac Blender 5.2.2 runs). Original rigs and weights are removed only after rest-pose preparation in disposable processes; all source hashes remain unchanged. Host-normalized bone geometry and skin weights match exactly, with zero hair/limb mixing, secondary four-influence smoothing, valid spring/collider references and bounded bend/drag tests. Nine shared core algorithms match. New public outputs use Hallway rig/armature/breast names; legacy helper defaults and saved identifiers are retained. New and legacy breast physics references pass Blender 4.2 and 5.2 tests. Live owner reload preserved the Nicole review without saving.

See research delivery/zzz-232/README.md and BVT documentation/auto-rig/zzz-siblings-232.json. No model-specific rules were introduced. Visual, complete collision, export/UniVRM and native Windows acceptance remain separate open gates.

### Hair length tuning and comparison capture (233)

Generated hair now blends short-strand VRM stiffness/drag (1.0/0.5) into the reviewed long-strand profile (0.3/0.3), using rest arc length divided by character height. The smooth transition spans 7.5–23% of height. Downward gravity remains proportional to stiffness. Runtime tuning uses no character identity, source rig or source weights. Existing rigs are not silently migrated; stored generated ratios preserve strand differences through rig-control changes.

The short reference is Wise after official MMD-to-VRM spring conversion. Its converted VRM values are not the MMD solver's mass/damping parameters. BVT/Hallway tests preserve geometry, weights, colliders and non-hair springs; the new generation path and legacy settings behavior are checked separately. Anby's back-hair drag is additionally halved only in the user-requested review scene; that is an artist override, not a character-specific rule in the auto-rigger. Evidence: `/Users/tylerwalker/Documents/Codex/2026-10-29/autoskin-rea-research/delivery/zzz-233/hair`.

All four 48-second videos passed full decode, source hash and no-focus checks. Each includes six physics cuts plus Walk01 and Run01 through the animation panel operator, labeled VRM Physics. Artifacts and exact capture caveats: `/Users/tylerwalker/Documents/Codex/2026-10-29/autoskin-rea-research/delivery/zzz-233/README.md`. Original/generated differences remain visible for review.

### Mounted gear, jacket ownership and finger repair — turn 234

Wise's source backpack is the structural reference: supported upper attachment, coherent pouch and a separate front strap where geometry supports one. The generated pack springs use VRM stiffness 1, drag .5, zero gravity and zero joint radius. This replaces the earlier proposed fixed two-bone / 50% body blend. Runtime classification uses rest geometry and generic semantic domains, never character identities or source rig/weight data. Generic material labels remain useful hints; this is not proof of arbitrary material-free apparel detection.

Both hosts now generate lower-torso jacket panels, keep sleeves anatomical, attach zippers to the garment field, keep close-fitting head equipment rigid, and handle long open hems with local sector bottoms. Jacket capsules fit the covered torso and remain clear of their own spring endpoints at rest. The four-model fixture checks pass; Anby/Belle/Wise jacket minimum rest clearances are 2.34/2.38/2.68 mm. This is not all-pose or mesh-mesh collision acceptance. Existing closed-skirt and hair tuning paths are retained.

Anby's hand landmark repair now selects the better-supported hand before strict bilateral surface agreement. A bounded distal-digit pass removes majority foreign-finger influence only on confident tips; ambiguous webbing and minor blends stay unchanged. The helper passes Blender 4.2 execution. Fresh generation and final fixture checks ran on Mac Blender 5.2.2. Nicole's joint layout stays unchanged; a disposable regression copy changes 94 confident distal finger vertices and was compared in a curl pose. No original weights enter generation and preserve volume stays off.

Final saved Anby fixtures match exactly between BVT/Hallway for bones, all mesh weights and the new pack/jacket physics after host-name normalization. Whole-rig physics does not match: Hallway carries the requested half-drag back-hair review override, and its breast Chest collider radius is .065597 m versus .040759 m in BVT. Those are recorded outside the attachment parity gate, not silently normalized away. Native Windows and full export/UniVRM acceptance remain untested here.

The previous Belle/Grace/Wise original drag captures were invalid: transform location locks prevented the operator from moving them. New disposable captures apply and verify the same world translation on both sides without editing source locks. Four 18-second, 1080p/24 fps clips show front/back/hand cuts with one second of bone overlay per cut and fixed 60 Hz VRM Physics. These focused clips do not recapture the previous Walk/Run clips. All four videos fully decode; source hashes remain unchanged. BVT and Hallway were reloaded without saving or altering the live dirty Nicole scene.

Evidence and reproduction: `delivery/zzz-234` in the autoskin research workspace; see `zzz-attachments-234.json`. The full solver API and delivery gates do not close from these character-specific validation fixtures.

### Garment-mounted tools, canisters and pocket overlays — turn 235

The shared `avatar_garment_attachments` pass runs after final generated skirt binding. It snapshots the generated cloth surface in rig-local rest space, admits nearby hardware through generic semantic hints plus spatial support, and gives each solid assembly one coherent four-influence field. Small fittings join the nearest or enclosing major shell; separate main shells do not merge just because their boxes touch. KD trees bound the proximity work without allocating pairwise vertex-distance matrices. No source rig, original skin weights, character identifiers or preserve-volume deformation are used.

Soft overlays use barycentric sampling from the supporting cloth. Collapsed donor triangles fall back to their longest edge. The soft route requires interior support on long articulated cloth (vertical support over 24% of character height); it preserves ordinary short skirt/jacket layouts. Generic material hints remain required for this attachment classifier; it is not universal unlabeled accessory detection. Existing rigs are only updated by an explicit rebuild or application; file load does not rewrite them.

Grace's corrected fixture has 3,863 admitted vertices across 59 disconnected components, including three separate strap canister assemblies, rear tools and soft pocket overlays. Both host helpers give identical repeatable results; untouched vertices remain exact and transferred rows have at most four normalized influences. Applying the final pass to the accepted Anby, Belle, Wise and Nicole fixtures changes zero weights. Blender 4.2 synthetic execution covers coherent separate caps, soft bending, short-garment preservation, scale invariance, degenerate triangles, idempotence and untouched skin/donor surfaces. Real fixture and public auto-generation execution use Mac Blender 5.2.2. This does not qualify native Windows, arbitrary models, all-pose collision or export parity.

Final comparison videos add six seconds each of Walk01 and Run01 via `bvt.apply_bundled_animation` on both original and generated rigs, with fixed 60 Hz VRM Physics sampled at 24 fps. The three previously accepted models retain their approved drag sections and now run 30 seconds. Grace is recaptured with extra front/back gear close-ups and runs 36 seconds. The animation presentation is in-place FK: interfering source controller/IK constraints are muted only in disposable copies and horizontal root travel is removed. This is visual movement review, not a new full retarget-delta acceptance claim. Original source files remain hash-identical.

Both add-ons' runtime helpers are loaded. The live dirty Nicole scene was not saved or reweighted. Evidence: `garment-attachments-235.json` and research `delivery/zzz-235/`; videos include bones briefly, then surfaces, and label the simulator VRM Physics.

### Cloth handoff and centered backpacks — turn 236

Hallway and BVT share `avatar_cloth_handoff`. The top attachment band has a smooth body-to-cloth profile: 100% body at its top, 90/10 at one fifth of its depth, then smoothly 0/100. Its depth is bounded to the smaller of 20% of garment length and 5% of character height. Underlying generated skin is sampled in rest space. Transverse cloth strips receive seam-coherent diffusion after that attachment; only the cloth distribution changes, preserving body mass/distribution and a four-influence budget. This avoids the old narrow left/right angular clamp and prevents later passes undoing the handoff. Mounted tools inherit the final field. Pants keep their generated anatomical field below the overlap.

Long cloth receives finer normalized segment spacing; short skirts retain their density and short jackets use slightly longer segments. Large centered packs use a common central rear-volume pivot and a 50/50 Chest/pack field across their disconnected parts. The off-center sling route is unchanged. Runtime admission uses rest geometry and generated ownership, never character names or source weights.

Fresh final Grace generation matches exactly between BVT and Hallway after canonical bone-name alignment. Final strap edge-energy falls from 0.26809 to 0.04734 (82.3%) with four influences; every tested waistband vertex has a body-surface hit. Anby/Belle/Wise hair weights are unchanged, finger differences are at most 6e-8, and Wise pack weights/placement are exact. Synthetic mass/seam/scale tests pass in 4.2 and 5.2; public generation, pose/drag and jacket collider-clearance checks pass in 5.2. Hallway and BVT are loaded live without reweighting or saving the dirty Nicole file. Review blends, source hashes and limitations: `cloth-handoff-236.json`, research `delivery/zzz-236/`. No new video or native Windows qualification is claimed. Existing rest intersections and all-pose penetration require separate visual review.

### Updated four-character movement review — turn 237

`delivery/zzz-237` contains fresh 36-second 1080p/24 fps original-versus-generated comparisons for Anby, Belle, Grace and Wise using the turn236 qualified rigs. Each video has four drag/detail cuts (including waistbands, straps, jacket tails, backpacks where present, and Grace hardware), followed by Walk01 at 24 seconds and Run01 at 30 seconds from the BVT animation library on both rigs. Bones appear for the first second of each cut. Labels identify the simulator as VRM Physics. Original and generated drag captures both verify identical world translation and nonzero relative spring motion.

Both original and generated physics/animation are freshly sampled. Original MMD physics uses the official VRM conversion in disposable copies; animations use the same public BVT operation and in-place FK presentation as turn235. Full decode, dimensions/frame counts, camera framing, source/qualified-input hashes and no-focus launch audits passed for all four. No rigging source or live scene was changed. Intermediate render frames and completed test profiles were removed after verification; final videos, compact visual checks, material/geometry motion caches, scripts and provenance remain. See `movement-review-237.json`.
