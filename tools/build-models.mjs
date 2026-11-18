// Builds public/models/head.glb, hand.glb and meta.json from the BodyParts3D OBJ archive.
// Run from the project root:  node tools/build-models.mjs
// Source meshes: BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 Japan.
import fs from "fs";
import { Document, NodeIO } from "@gltf-transform/core";
import { MeshoptSimplifier } from "meshoptimizer";

const DIR = "raw/obj/partof_BP3D_4.0_obj_99/";
// Neck muscles only exist in the is-a archive (isa_BP3D_4.0_obj_99.zip, unpacked to raw/obj-isa/).
const ISA_DIR = "raw/obj-isa/isa_BP3D_4.0_obj_99/";
const index = JSON.parse(fs.readFileSync("raw/index.json", "utf8"));
const idsByName = (re) =>
  index.filter((p) => re.test(p.name || "")).map((p) => p.id);

// ---------- OBJ ----------
function loadObj(id) {
  const pos = [],
    idx = [];
  const file = [DIR, ISA_DIR]
    .map((d) => d + id + ".obj")
    .find((f) => fs.existsSync(f));
  if (!file) throw new Error(`no OBJ for ${id}`);
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    if (line.startsWith("v ")) {
      const p = line.split(/\s+/);
      pos.push(+p[1], +p[2], +p[3]);
    } else if (line.startsWith("f ")) {
      const p = line
        .trim()
        .split(/\s+/)
        .slice(1)
        .map((t) => parseInt(t, 10) - 1);
      for (let i = 1; i + 1 < p.length; i++) idx.push(p[0], p[i], p[i + 1]);
    }
  }
  return { pos, idx };
}
function merge(ids) {
  const pos = [],
    idx = [];
  for (const id of [...new Set(ids)]) {
    const m = loadObj(id),
      base = pos.length / 3;
    for (const v of m.pos) pos.push(v);
    for (const i of m.idx) idx.push(i + base);
  }
  return { pos: new Float32Array(pos), idx: new Uint32Array(idx) };
}
function weld({ pos, idx }) {
  const map = new Map(),
    remap = new Uint32Array(pos.length / 3),
    out = [];
  for (let i = 0; i < pos.length / 3; i++) {
    const k = `${pos[3 * i].toFixed(3)},${pos[3 * i + 1].toFixed(3)},${pos[3 * i + 2].toFixed(3)}`;
    let j = map.get(k);
    if (j === undefined) {
      j = out.length / 3;
      map.set(k, j);
      out.push(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    }
    remap[i] = j;
  }
  return { pos: new Float32Array(out), idx: idx.map((i) => remap[i]) };
}
function compact({ pos, idx }) {
  const remap = new Int32Array(pos.length / 3).fill(-1),
    out = [],
    oi = new Uint32Array(idx.length);
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    if (remap[i] < 0) {
      remap[i] = out.length / 3;
      out.push(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
    }
    oi[k] = remap[i];
  }
  return { pos: new Float32Array(out), idx: oi };
}
function simplify(mesh, targetTris) {
  let m = weld(mesh);
  if (m.idx.length / 3 > targetTris) {
    const [idx] = MeshoptSimplifier.simplify(
      m.idx,
      m.pos,
      3,
      targetTris * 3,
      0.02,
      [],
    );
    m = { pos: m.pos, idx };
  }
  return compact(m);
}
const mapPos = (mesh, fn) => {
  const out = new Float32Array(mesh.pos.length);
  for (let i = 0; i < mesh.pos.length; i += 3) {
    const [x, y, z] = fn(mesh.pos[i], mesh.pos[i + 1], mesh.pos[i + 2]);
    out[i] = x;
    out[i + 1] = y;
    out[i + 2] = z;
  }
  return { pos: out, idx: mesh.idx };
};
const verts = (mesh) => {
  const a = [];
  for (let i = 0; i < mesh.pos.length; i += 3)
    a.push([mesh.pos[i], mesh.pos[i + 1], mesh.pos[i + 2]]);
  return a;
};
const mean = (pts) =>
  pts
    .reduce((s, p) => [s[0] + p[0], s[1] + p[1], s[2] + p[2]], [0, 0, 0])
    .map((v) => v / pts.length);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(...a);
const norm = (a) => mul(a, 1 / len(a));

// BodyParts3D: mm, +x = subject's left, -y = anterior, +z = up.
// Head frame: cm, +x = subject's left, +y = up, +z = anterior (same axes as MediaPipe's canonical face).
const toHead = (x, y, z) => [x * 0.1, z * 0.1, -y * 0.1];

// ---------- part selection ----------
const exact = (names) =>
  idsByName(new RegExp("^(" + names.join("|") + ")$", "i"));
const sides = (n) => [`Left ${n}`, `Right ${n}`];
const PARTS = {
  cranium: exact([
    "Ethmoid",
    "Frontal bone",
    "Occipital bone",
    "Sphenoid bone",
    "Vomer",
    "Gingiva of upper jaw",
    "Septal nasal cartilage",
    ...[
      "inferior nasal concha",
      "lacrimal bone",
      "maxilla",
      "nasal bone",
      "palatine bone",
      "parietal bone",
      "temporal bone",
      "zygomatic bone",
      "major alar cartilage",
      "lateral nasal cartilage",
    ].flatMap(sides),
  ]),
  neck: idsByName(
    /^(Atlas|Axis|Hyoid bone|(Third|Fourth|Fifth|Sixth|Seventh) cervical vertebra|Intervertebral disk of (third|fourth|fifth|sixth|seventh) cervical vertebra)$/i,
  ),
  jaw: exact(["Mandible", "Gingiva of lower jaw", "Tongue"]),
  eyes: exact(["sclera", "cornea", "iris", "lens"].flatMap(sides)),
  muscles: exact([
    ...[
      "inferior oblique",
      "inferior rectus",
      "lateral rectus",
      "medial rectus",
      "superior oblique",
      "superior rectus",
      "levator palpebrae superioris",
      "levator veli palatini",
      "tensor veli palatini",
    ].flatMap(sides),
    "Uvular muscle",
  ]),
};
// brain = every element of the FMA "brain" concept
{
  const rows = fs
    .readFileSync("raw/partof_element_parts.txt", "utf8")
    .trim()
    .split("\n")
    .slice(1)
    .map((l) => l.split("\t"));
  PARTS.brain = rows.filter((r) => r[1] === "brain").map((r) => r[2].trim());
}
// neck muscles = every element of these FMA concepts in the is-a archive
{
  const NECK = new Set([
    "platysma",
    "sternocleidomastoid",
    "digastric",
    "mylohyoid",
    "geniohyoid",
    "stylohyoid",
    "sternohyoid",
    "omohyoid",
    "sternothyroid",
    "thyrohyoid",
  ]);
  const ids = fs
    .readFileSync("raw/isa_element_parts.txt", "utf8")
    .trim()
    .split("\n")
    .slice(1)
    .map((l) => l.split("\t"))
    .filter((r) => NECK.has(r[1]))
    .map((r) => r[2].trim());
  PARTS.muscles.push(...ids);
}
// brain regions that light up on their own: pulled out of the merged brain, not drawn twice
const REGIONS = {
  brain_m1_L: ["left precentral gyrus"],
  brain_m1_R: ["right precentral gyrus"],
  brain_s1_L: ["left postcentral gyrus"],
  brain_s1_R: ["right postcentral gyrus"],
  brain_visual: ["left occipital lobe", "right occipital lobe"],
  brain_broca: ["left inferior frontal gyrus"],
  brain_cerebellum: ["cerebellum"],
};
{
  const rows = fs
    .readFileSync("raw/isa_element_parts.txt", "utf8")
    .trim()
    .split("\n")
    .slice(1)
    .map((l) => l.split("\t"));
  const whole = new Set(PARTS.brain);
  for (const [node, names] of Object.entries(REGIONS)) {
    const ids = rows
      .filter((r) => names.includes(r[1].toLowerCase()))
      .map((r) => r[2].trim());
    const inBrain = ids.filter((id) => whole.has(id));
    if (!inBrain.length) throw new Error(`${node}: not part of the brain mesh`);
    if (inBrain.length !== ids.length)
      console.log(
        "  note:",
        node,
        ids.length - inBrain.length,
        "elements outside the brain concept, skipped",
      );
    for (const id of inBrain) whole.delete(id);
    PARTS[node] = inBrain;
  }
  PARTS.brain = [...whole];
}
const TARGET = {
  cranium: 70000,
  neck: 10000,
  jaw: 8000,
  brain: 40000,
  eyes: 12000,
  muscles: 40000,
  brain_m1_L: 5000,
  brain_m1_R: 5000,
  brain_s1_L: 5000,
  brain_s1_R: 5000,
  brain_visual: 8000,
  brain_broca: 4000,
  brain_cerebellum: 10000,
};

// ---------- registration: head frame -> MediaPipe canonical face ----------
const canon = fs
  .readFileSync("raw/canonical_face_model.obj", "utf8")
  .split("\n")
  .filter((l) => l.startsWith("v "))
  .map((l) => l.split(/\s+/).slice(1, 4).map(Number));
const mid = (a, b) => mul(add(a, b), 0.5);
const headPts = (name) => verts(mapPos(merge(exact([name])), toHead));
const eyeL = mean(headPts("Left sclera")),
  eyeR = mean(headPts("Right sclera"));
const nasal = [
  ...headPts("Left nasal bone"),
  ...headPts("Right nasal bone"),
].sort((a, b) => b[1] - a[1]);
const nasion = mean(nasal.slice(0, Math.ceil(nasal.length * 0.05)));
const mand = headPts("Mandible");
const midline = mand
  .filter((p) => Math.abs(p[0]) < 0.8)
  .sort((a, b) => a[1] - b[1]);
const menton = mean(midline.slice(0, Math.ceil(midline.length * 0.03)));
// skin landmark -> underlying bone/organ, soft-tissue offsets in cm
const cEyeL = add(mid(canon[362], canon[263]), [0, 0, -0.9]);
const cEyeR = add(mid(canon[33], canon[133]), [0, 0, -0.9]);
const cNasion = add(canon[168], [0, 0, -0.6]);
const cMenton = add(canon[152], [0, 0.65, -0.35]);
const P = [eyeL, eyeR, nasion, menton],
  Q = [cEyeL, cEyeR, cNasion, cMenton];
const pm = mean(P),
  qm = mean(Q);
let num = 0,
  den = 0;
P.forEach((p, i) => {
  num += dot(sub(p, pm), sub(Q[i], qm));
  den += dot(sub(p, pm), sub(p, pm));
});
const S = num / den,
  T = sub(qm, mul(pm, S));
const toCanon = (x, y, z) => add(mul(toHead(x, y, z), S), T);
const h2c = (p) => add(mul(p, S), T);
console.log(
  "registration scale",
  S.toFixed(4),
  "translation",
  T.map((v) => v.toFixed(2)),
);
["eyeL", "eyeR", "nasion", "menton"].forEach((n, i) =>
  console.log(
    "  residual",
    n,
    sub(h2c(P[i]), Q[i])
      .map((v) => v.toFixed(2))
      .join(", "),
    "cm",
  ),
);

// jaw hinge = line through the two condyle tops
const condyle = (sign) => {
  const s = mand.filter((p) => p[0] * sign > 3).sort((a, b) => b[1] - a[1]);
  return mean(s.slice(0, Math.ceil(s.length * 0.02)));
};
const hinge = h2c(mid(condyle(1), condyle(-1)));
console.log(
  "jaw hinge (canonical cm)",
  hinge.map((v) => v.toFixed(2)),
  "chin",
  h2c(menton).map((v) => v.toFixed(2)),
);

// ---------- glTF writing ----------
async function writeGlb(file, meshes) {
  const doc = new Document(),
    buf = doc.createBuffer(),
    scene = doc.createScene();
  let tris = 0;
  for (const [name, m] of Object.entries(meshes)) {
    const prim = doc
      .createPrimitive()
      .setAttribute(
        "POSITION",
        doc.createAccessor().setType("VEC3").setArray(m.pos).setBuffer(buf),
      )
      .setIndices(
        doc
          .createAccessor()
          .setType("SCALAR")
          .setArray(m.pos.length / 3 > 65535 ? m.idx : new Uint16Array(m.idx))
          .setBuffer(buf),
      );
    scene.addChild(
      doc.createNode(name).setMesh(doc.createMesh(name).addPrimitive(prim)),
    );
    tris += m.idx.length / 3;
  }
  doc.getRoot().getAsset().copyright =
    "BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 Japan (decimated)";
  await new NodeIO().write(file, doc);
  console.log(
    file,
    (fs.statSync(file).size / 1e6).toFixed(2),
    "MB",
    tris,
    "tris",
  );
}

await MeshoptSimplifier.ready;

const head = {};
// Platysma and the strap muscles run on to the chest, which the face tracker cannot place:
// cut the muscle layer off at the bottom of the seventh cervical vertebra.
const neckFloor = merge(PARTS.neck).pos.reduce(
  (m, v, i) => (i % 3 === 2 ? Math.min(m, v) : m),
  Infinity,
);
function clipBelow({ pos, idx }, z) {
  const keep = [];
  for (let i = 0; i < idx.length; i += 3)
    if (
      pos[3 * idx[i] + 2] >= z &&
      pos[3 * idx[i + 1] + 2] >= z &&
      pos[3 * idx[i + 2] + 2] >= z
    )
      keep.push(idx[i], idx[i + 1], idx[i + 2]);
  return { pos, idx: new Uint32Array(keep) };
}
for (const [name, ids] of Object.entries(PARTS)) {
  let raw = merge(ids);
  if (name === "muscles") raw = clipBelow(raw, neckFloor);
  head[name] = mapPos(simplify(raw, TARGET[name]), toCanon);
  console.log(
    " ",
    name,
    ids.length,
    "parts",
    raw.idx.length / 3,
    "->",
    head[name].idx.length / 3,
  );
}
await writeGlb("public/models/head.glb", head);
// height range of each motor / sensory strip, for the body map drawn along it
const strips = {};
for (const k of ["m1_L", "m1_R", "s1_L", "s1_R"]) {
  const ys = verts(head["brain_" + k])
    .map((p) => p[1])
    .sort((a, b) => a - b);
  strips[k] = [
    ys[Math.floor(ys.length * 0.02)],
    ys[Math.floor(ys.length * 0.98)],
  ];
}
const brainCentre = mean(verts(head.brain));

// ---------- right hand ----------
// Hand frame: mm, BodyParts3D axes remapped like the head, origin at the carpal centroid.
const toHandRaw = (x, y, z) => [x, z, -y];
const FINGERS = [
  "thumb",
  "index finger",
  "middle finger",
  "ring finger",
  "little finger",
];
const ORD = ["first", "second", "third", "fourth", "fifth"];
const CARPALS = [
  "capitate",
  "hamate",
  "lunate",
  "pisiform",
  "scaphoid",
  "trapezium",
  "trapezoid",
  "triquetral",
];
const one = (name) => {
  const ids = exact([name]);
  if (ids.length !== 1) throw new Error(`${name}: ${ids.length} matches`);
  return mapPos(weld(merge(ids)), toHandRaw);
};
const carpalMeshes = Object.fromEntries(
  CARPALS.map((c) => [c, one(`Right ${c}`)]),
);
const origin = mean(Object.values(carpalMeshes).flatMap(verts));
const recenter = (m) =>
  mapPos(m, (x, y, z) => [x - origin[0], y - origin[1], z - origin[2]]);
const hand = {},
  bones = [];
// endpoints of a long bone: extremes along its principal axis, proximal = nearer the carpus
function endpoints(m) {
  const v = verts(m),
    c = mean(v);
  let a = [0, 1, 0];
  for (let it = 0; it < 30; it++) {
    // power iteration on the covariance
    let n = [0, 0, 0];
    for (const p of v) {
      const d = sub(p, c);
      n = add(n, mul(d, dot(d, a)));
    }
    a = norm(n);
  }
  const t = v.map((p) => dot(sub(p, c), a));
  let e0 = add(c, mul(a, Math.min(...t))),
    e1 = add(c, mul(a, Math.max(...t)));
  if (len(e0) > len(e1)) [e0, e1] = [e1, e0];
  return [e0, e1];
}
for (const c of CARPALS) {
  hand[`carpal_${c}`] = recenter(carpalMeshes[c]);
  bones.push({ name: `carpal_${c}`, kind: "carpal" });
}
FINGERS.forEach((f, i) => {
  const segs = [
    ["mc", `Right ${ORD[i]} metacarpal bone`],
    ["prox", `Proximal phalanx of right ${f}`],
  ];
  if (i > 0) segs.push(["mid", `Middle phalanx of right ${f}`]);
  segs.push(["dist", `Distal phalanx of right ${f}`]);
  for (const [seg, name] of segs) {
    const m = recenter(one(name)),
      [p, d] = endpoints(m),
      key = `${seg}_${i}`;
    hand[key] = m;
    bones.push({ name: key, kind: seg, finger: i, prox: p, dist: d });
  }
});
// rest-pose equivalents of MediaPipe hand landmarks
const proximalRow = mean(
  ["scaphoid", "lunate", "triquetral"].flatMap((c) =>
    verts(hand[`carpal_${c}`]),
  ),
);
const mc3 = bones.find((b) => b.name === "mc_2");
const wrist = add(proximalRow, mul(norm(sub(proximalRow, mc3.dist)), 8));
const rest = {
  wrist,
  mcp: [0, 1, 2, 3, 4].map((i) => bones.find((b) => b.name === `mc_${i}`).dist),
  thumbCmc: bones.find((b) => b.name === "mc_0").prox,
};
console.log(
  "hand rest: wrist",
  wrist.map((v) => v.toFixed(0)),
  "palm width",
  len(sub(rest.mcp[1], rest.mcp[4])).toFixed(0),
  "mm",
);
await writeGlb("public/models/hand.glb", hand);

const r = (a) => a.map((v) => +v.toFixed(3));
fs.writeFileSync(
  "public/models/meta.json",
  JSON.stringify(
    {
      attribution:
        "BodyParts3D, © The Database Center for Life Science, licensed under CC Attribution-Share Alike 2.1 Japan",
      head: {
        units: "cm, MediaPipe canonical face space",
        scale: +S.toFixed(5),
        jawHinge: r(hinge),
        strips: Object.fromEntries(
          Object.entries(strips).map(([k, v]) => [k, r(v)]),
        ),
        brainCentre: r(brainCentre),
      },
      hand: {
        units: "mm",
        rest: {
          wrist: r(rest.wrist),
          mcp: rest.mcp.map(r),
          thumbCmc: r(rest.thumbCmc),
        },
        bones: bones.map((b) => ({
          ...b,
          prox: b.prox && r(b.prox),
          dist: b.dist && r(b.dist),
        })),
      },
    },
    null,
    1,
  ),
);
console.log("wrote public/models/meta.json");
