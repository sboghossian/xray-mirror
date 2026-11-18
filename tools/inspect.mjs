import fs from "fs";
const rows = fs
  .readFileSync("raw/partof_element_parts.txt", "utf8")
  .trim()
  .split("\n")
  .slice(1)
  .map((l) => l.split("\t"));
const byConcept = new Map(),
  names = new Map();
for (const [c, n, e] of rows) {
  if (!byConcept.has(c)) byConcept.set(c, new Set());
  byConcept.get(c).add(e);
  names.set(c, n);
}
// leaf name per element = concept with smallest element set containing it
const leaf = new Map();
for (const [c, set] of byConcept)
  for (const e of set) {
    const cur = leaf.get(e);
    if (!cur || set.size < cur.size)
      leaf.set(e, { c, name: names.get(c), size: set.size });
  }
const dir = "raw/obj/partof_BP3D_4.0_obj_99/";
const stat = (e) => {
  try {
    const t = fs.readFileSync(dir + e + ".obj", "utf8");
    return {
      v: (t.match(/^v /gm) || []).length,
      f: (t.match(/^f /gm) || []).length,
    };
  } catch {
    return null;
  }
};
const want = process.argv[2];
const id = [...names].find(([c, n]) => n === want)?.[0];
const els = [...byConcept.get(id)];
let tv = 0,
  tf = 0;
const out = [];
for (const e of els) {
  const s = stat(e);
  if (!s) {
    out.push(e + " MISSING");
    continue;
  }
  tv += s.v;
  tf += s.f;
  out.push(`${e}\t${s.f}\t${leaf.get(e).name}`);
}
console.log(want, id, "elements", els.length, "verts", tv, "faces", tf);
if (process.argv[3])
  console.log(
    out
      .sort((a, b) => a.split("\t")[2]?.localeCompare(b.split("\t")[2]))
      .join("\n"),
  );
