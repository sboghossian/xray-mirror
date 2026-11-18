# X-ray Mirror

Point a webcam at your face and see the skull, brain and eyes inside your head, tracked live. Open your mouth and the jaw opens. Hold up a hand and you see its bones. Step back and the whole skeleton and the main muscle groups follow your body; muscles glow by how hard they are estimated to be working, and brain areas light up from a textbook map.

**The activity display is an estimate, not a measurement.** A webcam cannot read muscles or a brain. The page says so on screen.

One static web page. Tracking runs in the browser (MediaPipe face, hand and pose landmarkers, WASM) and rendering is three.js. All libraries, tracking models and anatomy meshes are served from `public/`; the page makes no network requests beyond its own folder, and no video frame leaves the computer.

## Run

```bash
python3 tools/dev-server.py 8797
```

Then open http://localhost:8797/public/ and press Start camera. Any static file server works; it has to be `localhost` or HTTPS for the browser to allow camera access.

## How it works

- **Head**: `FaceLandmarker` returns a 4x4 face pose matrix in MediaPipe's canonical-face space (cm, 63 degree vertical FOV camera). The skull, brain, eyes and neck are pre-registered to that space at build time (eye centres, nasion and chin, about 6 mm residual), so the pose matrix is applied directly.
- **Jaw**: the mandible is a separate mesh hinged on the line through its condyles; the `jawOpen` blendshape drives the rotation.
- **Hands**: `HandLandmarker` gives 21 points per hand. Carpals follow a palm affine (wrist, index knuckle, little-finger knuckle); each metacarpal and phalanx is stretched between its two landmarks. Left hands are the right-hand bones mirrored.
- **Body**: `PoseLandmarker` gives 33 points. The rib cage and pelvis are affine blocks placed from the shoulders and hips; each long bone (humerus, forearm, femur, shin, foot) is stretched between its two landmarks and rolled to the limb's bending plane. Each of 17 muscle groups per side rides rigidly on one segment. If the hips are out of frame the trunk hangs straight down from the shoulders. When the face tracker loses a small, distant face, the head is placed from the pose's eyes and mouth instead.
- **Muscle estimate** (`public/activity.js`): from the pose's 3D joints it works out, per joint, the torque needed to hold the limb against gravity plus the limb's acceleration, and shares it between the muscles that produce that torque. Lowering an arm slowly therefore still loads the deltoid. Weight on each leg comes from which foot is on the floor.
- **Brain map**: motor and sensory strips (precentral and postcentral gyrus) light on the side opposite the moving limb, at the height on the strip that textbooks assign to leg, trunk, arm, hand and face. The cerebellum follows overall movement, the speech area follows jaw movement, the visual cortex is always on. It is an illustration of the textbook map driven by what moved, nothing more.
- **Fit to my face**: size, height, depth and tilt sliders, saved in `localStorage`.

## Limits

- BodyParts3D 4.0 has no teeth and no facial or chewing muscles (no masseter, no temporalis). The muscle layer is the eye and soft-palate muscles plus ten neck muscles (platysma, sternocleidomastoid, digastric, mylohyoid, geniohyoid, stylohyoid and the four strap muscles), cut off at the bottom of the seventh cervical vertebra.
- The neck muscles are rigid: they turn with the head and do not follow the jaw when it opens.
- One generic skull is scaled to every face; it does not adapt to individual bone structure.
- The neck vertebrae rotate rigidly with the head.
- Muscle activity is inferred from posture and movement only. It knows nothing about loads you hold, things you lean on, grip, or co-contraction: resting or folding your arms still shows biceps and forearms, sitting on a chair reads as holding a squat, and lying down is not recognised.
- BodyParts3D 4.0 has no latissimus dorsi and no rectus abdominis: the "lats" group is teres major only and the "obliques" group is the external oblique only.
- Muscles are rigid on one segment and do not stretch across joints: the deltoid swings with the upper arm, the chest muscles stay on the rib cage. Forearm and shin muscles are cut to the part between their two joints. No forearm rotation.
- When a face is tracked and the pose model puts the eyes somewhere else (off by more than half the eye spacing), the pose is treated as a wrong guess and the body is hidden. This was added after the pose model drew a small sideways torso on the hands test photo. Under 900 px of window width only the "estimate" label is shown, not the brain view and bars.
- Pose depth is the model's guess. Limb length is clamped to body proportions, so a bad depth guess shortens a limb rather than stretching it.
- Body tracking adds about 7 ms per frame here (about 25 ms per frame in total against 18 ms without, measured on the 1280x720 video fixture in the Claude app's browser pane on this Mac).
- The body layer has only been checked on synthetic poses (`__xray.forcePose`) and on the waist-up portrait. No full-body photo or video, and no moving person, has been through it; left and right were checked against the face tracker on the portrait (the pose's left eye lands on the face tracker's left eye), not against a moving person.
- Verified on test fixtures only (a portrait and a two-hand photo, as stills and as 1280x720 video) through `?src=<file>&test`. Not yet verified on a live webcam: camera permission, mirroring, frame rate and left/right hand labelling are untested.

## Rebuild the models

`public/models/head.glb` and `hand.glb` are decimated from the two BodyParts3D OBJ archives (`partof_BP3D_4.0_obj_99.zip` unpacked to `raw/obj/`, and `isa_BP3D_4.0_obj_99.zip` unpacked to `raw/obj-isa/` for the neck muscles) plus MediaPipe's `canonical_face_model.obj` in `raw/`:

```bash
cd tools && npm install && cd ..
node tools/index.mjs && node tools/build-models.mjs && node tools/build-body.mjs
```

`build-body.mjs` writes `body.glb` and `body-meta.json` (full skeleton and muscle groups) from the same archives and `raw/isa_element_parts.txt`.

## Tests

```bash
node --test test-unit/activity.test.mjs
```

13 synthetic poses for the muscle and brain estimate (standing, arm held out, elbow holds for biceps and triceps, slow lowering, squat, knee raise, heel raise, hip hinge, plank, waving, talking, hidden limb, jitter). `?src=<image or video>&test` plus `__xray.shot(name)` saves a composite to `test/out/`; `&stick` draws the pose skeleton (green = your left, magenta = your right).

## Credits and licences

- Anatomy models: BodyParts3D, © The Database Center for Life Science, licensed under CC Attribution-Share Alike 2.1 Japan. The meshes in `public/models/` are decimated and re-oriented derivatives and carry the same licence.
- MediaPipe Tasks Vision and its models: Apache 2.0. three.js: MIT.
