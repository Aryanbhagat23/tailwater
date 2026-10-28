// JavaScript port of the tailwater simulator (src/tailwater/env), the demo
// baselines and the PPO forward pass. Verified against the Python
// implementation by site/check_sim.js.
const TW = (() => {
  function interp(x, xs, ys) {
    if (x <= xs[0]) return ys[0];
    for (let i = 1; i < xs.length; i++) {
      if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
    }
    return ys[ys.length - 1];
  }

  function makeModel(D) {
    const T = D.season.length;
    const kcx = D.crop.kc_points.map(p => p[0]), kcy = D.crop.kc_points.map(p => p[1]);
    const kc = Array.from({ length: T }, (_, d) => interp(d, kcx, kcy));
    const starts = D.crop.stage_starts;
    const stage = Array.from({ length: T }, (_, d) => {
      let s = 0;
      for (let i = 0; i < starts.length; i++) if (d >= starts[i]) s = i;
      return s;
    });
    return { T, kc, stage, ky: D.crop.ky, taw: D.soil.taw, p: D.soil.p,
             s0: D.soil.initial_fraction * D.soil.taw, actions: D.actions };
  }

  // ---- policies: each is { reset(), act(obs) -> action index } ----
  const largestAtMost = (A, mm) => { let k = 0; for (let i = 0; i < A.length; i++) if (A[i] <= mm) k = i; return k; };
  const idxOf = (A, mm) => A.indexOf(mm);

  function makePolicy(kind, params, M, net) {
    const A = M.actions;
    switch (kind) {
      case "rain-fed": return { reset() {}, act() { return 0; } };
      case "threshold": { const a = idxOf(A, params.amount);
        return { reset() {}, act(o) { return o.moisture < params.trigger ? a : 0; } }; }
      case "threshold+forecast": { const a = idxOf(A, params.amount);
        return { reset() {}, act(o) { return (o.moisture < params.trigger && o.forecast < params.skip_mm) ? a : 0; } }; }
      case "stage threshold": { const a = idxOf(A, params.amount);
        return { reset() {}, act(o) {
          const trig = (o.stage === 2 || o.stage === 3) ? params.trigger_sensitive : params.trigger_tolerant;
          return o.moisture < trig ? a : 0; } }; }
      case "ET controller": { let deficit = 0;
        return { reset() { deficit = 0; }, act(o) {
          if (o.day > 0) deficit = Math.max(0, deficit + o.etc_y - o.rain_y);
          let a = 0;
          if (deficit >= params.trigger) a = Math.max(largestAtMost(A, params.refill * deficit), 1);
          deficit = Math.max(0, deficit - A[a]);
          return a; } }; }
      case "ppo": return { reset() {}, act(o) { return argmax(forward(net, features(o, M))); } };
    }
    throw new Error("unknown policy " + kind);
  }

  function features(o, M) {
    const f = new Array(12).fill(0);
    f[0] = o.moisture / M.taw;
    f[1] = o.day / M.T;
    f[2 + o.stage] = 1;
    f[7] = Math.log1p(o.forecast) / Math.log1p(50);
    f[8] = Math.min(o.dsr, 15) / 15;
    f[9] = o.kc;
    f[10] = o.et0_y / 8;
    f[11] = o.etc_y / 8;
    return f;
  }

  function forward(net, x) {
    let h = x;
    for (let l = 0; l < net.W.length; l++) {
      const W = net.W[l], b = net.b[l], out = b.slice();
      for (let i = 0; i < h.length; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = W[i];
        for (let j = 0; j < out.length; j++) out[j] += hi * row[j];
      }
      h = l < net.W.length - 1 ? out.map(Math.tanh) : out;
    }
    return h;
  }
  const argmax = v => { let k = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i; return k; };

  // ---- one season, recording the daily trajectory ----
  function runSeason(M, rain, et0, fc, policy) {
    const T = M.T, ns = M.ky.length, stressPt = (1 - M.p) * M.taw;
    let s = M.s0, water = 0, drain = 0, dsr = 0, rainY = 0, et0Y = et0[0], etcY = et0[0] * M.kc[0];
    const eta = new Array(ns).fill(0), etm = new Array(ns).fill(0);
    const tr = { moisture: [], irr: [], ks: [], yieldSoFar: [], water: [] };
    policy.reset();
    for (let d = 0; d < T; d++) {
      const st = M.stage[d];
      const o = { day: d, stage: st, kc: M.kc[d], moisture: s, forecast: fc[d], dsr,
                  rain_y: rainY, et0_y: et0Y, etc_y: etcY };
      const a = policy.act(o), irr = M.actions[a];
      const etc = M.kc[d] * et0[d];
      let s1 = s + rain[d] + irr;
      const dr = Math.max(0, s1 - M.taw);
      const s2 = s1 - dr;
      const ks = Math.min(1, Math.max(0, s2 / stressPt));
      const ea = Math.min(ks * etc, s2);
      s = s2 - ea;
      water += irr; drain += dr; eta[st] += ea; etm[st] += etc;
      dsr = rain[d] >= 2 ? 0 : dsr + 1;
      rainY = rain[d]; et0Y = et0[d]; etcY = etc;
      tr.moisture.push(s); tr.irr.push(irr); tr.ks.push(ks); tr.water.push(water);
      tr.yieldSoFar.push(fao33(eta, etm, M.ky));
    }
    return { water, yield: fao33(eta, etm, M.ky), drain, traj: tr };
  }

  function fao33(eta, etm, ky) {
    let y = 1;
    for (let s = 0; s < ky.length; s++) {
      const ratio = etm[s] > 0 ? eta[s] / etm[s] : 1;
      y *= Math.max(0, 1 - ky[s] * (1 - ratio));
    }
    return y;
  }

  function summarize(water, yields) {
    const n = yields.length, sorted = yields.slice().sort((a, b) => a - b);
    const k = Math.max(1, Math.ceil(0.05 * n));
    const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
    return { water: mean(water), mean_yield: mean(yields),
             cvar5_yield: mean(sorted.slice(0, k)), worst_yield: sorted[0] };
  }

  return { makeModel, makePolicy, runSeason, summarize, forward, features, argmax };
})();
if (typeof module !== "undefined") module.exports = TW;
