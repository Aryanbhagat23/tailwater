// Verify the JavaScript simulator against the Python reference results.
//   python experiments/export_site.py && node site/check_sim.js
const fs = require("fs");
const path = require("path");
const TW = require("./sim.js");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const M = TW.makeModel(D);

function policyFor(name, fold) {
  if (name === "rain-fed") return TW.makePolicy("rain-fed", {}, M);
  if (D.baselines[name]) return TW.makePolicy(name, D.baselines[name][fold].params, M);
  return TW.makePolicy("ppo", null, M, D.models[name][fold]);
}

let worst = 0, mismatches = 0;
for (const [name, ref] of Object.entries(D.reference)) {
  let maxW = 0, maxY = 0;
  D.years.forEach((_, i) => {
    const res = TW.runSeason(M, D.rain[i], D.et0[i], D.forecast[i], policyFor(name, D.fold_of[i]));
    maxW = Math.max(maxW, Math.abs(res.water - ref.water[i]));
    maxY = Math.max(maxY, Math.abs(res.yield - ref.yield[i]));
  });
  // ET0 and forecasts are exported to 1e-5 mm and weights to 1e-7, so every decision must match
  const ok = maxY < 1e-3 && maxW < 1e-6;
  if (!ok) mismatches++;
  worst = Math.max(worst, maxY);
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(20)} max |dWater| ${maxW.toFixed(2)} mm  max |dYield| ${maxY.toFixed(4)}`);
}
process.exit(mismatches ? 1 : 0);
