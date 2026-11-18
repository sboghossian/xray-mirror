import fs from "fs";
const dir = "raw/obj/partof_BP3D_4.0_obj_99/";
const out = [];
for (const f of fs.readdirSync(dir)) {
  if (!f.endsWith(".obj")) continue;
  const t = fs.readFileSync(dir + f, "utf8");
  const name = (t.match(/English name : (.*)/) || [])[1]?.trim();
  const b = t.match(/Bounds\(mm\): \(([^)]*)\)-\(([^)]*)\)/) || [];
  const lo = b[1]?.split(",").map(Number),
    hi = b[2]?.split(",").map(Number);
  out.push({
    id: f.replace(".obj", ""),
    name,
    lo,
    hi,
    faces: (t.match(/^f /gm) || []).length,
  });
}
fs.writeFileSync("raw/index.json", JSON.stringify(out));
console.log(out.length);
