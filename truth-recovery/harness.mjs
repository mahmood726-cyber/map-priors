// ============================================================
// harness.mjs -- Truth-recovery yardstick for map-priors.
//
// Derives the MAP prior with the app's OWN deriveMAPContinuous (engine.mjs,
// verbatim from map-priors.html), then forms the EXACT 2-component normal-mixture
// posterior for a new-trial parameter and measures, against a known truth:
//   - exchangeable case: does borrowing cut MSE / sharpen the posterior without
//     losing coverage of theta_new?
//   - conflict case: does the ROBUST mixture limit the bias that the pure MAP
//     prior (w=1) suffers, and restore coverage?
//
// Truth-first: every number printed comes from seeded simulation here.
// Run:  node truth-recovery/harness.mjs --reps 2000
// ============================================================

import { deriveMAPContinuous, setStudies, normalPDF, normalCDF, normalQuantile } from './engine.mjs';
import { generate, makeRng } from './dgp-map.mjs';

const BASE_SEED = 20260613;

// posterior of a normal-mixture prior {pi,m,v} updated by N(y; theta, s^2).
function mixturePosterior(comps, y, s2) {
  const post = comps.map(c => {
    const v = 1 / (1 / c.v + 1 / s2);
    const m = v * (c.m / c.v + y / s2);
    const ml = normalPDF(y, c.m, Math.sqrt(c.v + s2));   // marginal likelihood
    return { pi: c.pi * ml, m, v };
  });
  const Z = post.reduce((s, c) => s + c.pi, 0);
  post.forEach(c => c.pi /= Z);
  const mean = post.reduce((s, c) => s + c.pi * c.m, 0);
  // exact equal-tailed credible interval via mixture-CDF bisection
  const cdf = (x) => post.reduce((s, c) => s + c.pi * normalCDF((x - c.m) / Math.sqrt(c.v)), 0);
  const q = (p) => { let lo = mean - 30, hi = mean + 30;
    for (let i = 0; i < 80; i++) { const mid = (lo + hi) / 2; if (cdf(mid) < p) lo = mid; else hi = mid; } return (lo + hi) / 2; };
  return { mean, lo: q(0.025), hi: q(0.975) };
}

function methods(studies, y_new, se_new) {
  setStudies(studies);
  const r = deriveMAPContinuous(0.5, 0.95);           // w=0.5 robust derivation
  const s2 = se_new * se_new;
  const mapComp = [{ pi: 1, m: r.map_mu, v: r.map_var }];
  // robust mixture as the app builds it: w * MAP + (1-w) * vague, both centred at mu_post
  const w = 0.5;
  const robustComp = [{ pi: w, m: r.map_mu, v: r.map_var },
                      { pi: 1 - w, m: r.vague_mu, v: r.vague_se * r.vague_se }];
  const zc = normalQuantile(0.975);
  return {
    vague:  { mean: y_new, lo: y_new - zc * se_new, hi: y_new + zc * se_new },
    MAP:    mixturePosterior(mapComp, y_new, s2),
    robust: mixturePosterior(robustComp, y_new, s2),
  };
}

const METHODS = ['vague', 'MAP', 'robust'];
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

export function runCell(muHist, tau2, kHist, scenario, shift, reps, rng) {
  const acc = {};
  for (const m of METHODS) acc[m] = { cov: 0, sq: 0, wSum: 0, n: 0 };
  for (let r = 0; r < reps; r++) {
    const { studies, theta_new, y_new, se_new } = generate(muHist, tau2, kHist, scenario, shift, rng);
    let res; try { res = methods(studies, y_new, se_new); } catch { continue; }
    for (const m of METHODS) {
      const o = res[m];
      if (!o || !isFinite(o.mean) || !isFinite(o.lo)) continue;
      const a = acc[m];
      a.n++;
      a.sq += (o.mean - theta_new) ** 2;
      a.wSum += o.hi - o.lo;
      if (o.lo <= theta_new && theta_new <= o.hi) a.cov++;
    }
  }
  const res = {};
  for (const m of METHODS) {
    const a = acc[m];
    res[m] = { coverage: a.n ? +(a.cov / a.n).toFixed(4) : null,
               rmse: a.n ? +Math.sqrt(a.sq / a.n).toFixed(4) : null,
               ciWidth: a.n ? +(a.wSum / a.n).toFixed(4) : null };
  }
  return res;
}

const isMain = process.argv[1]?.endsWith('harness.mjs');
if (isMain) {
  const i = process.argv.indexOf('--reps');
  const reps = i >= 0 ? Number(process.argv[i + 1]) : 2000;
  const t0 = Date.now();
  const rng = makeRng(BASE_SEED);
  console.log(`\n# Truth-recovery yardstick -- map-priors`);
  console.log(`reps=${reps}/cell  muHist=0.0  kHist=5  sigma=1  seed=${BASE_SEED}\n`);
  console.log('## Exchangeable new trial (theta_new ~ N(muHist, tau2)) -- borrowing should help\n');
  console.log('tau2   method    coverage  RMSE     CIwidth');
  for (const tau2 of [0.02, 0.10]) {
    for (const m of METHODS) {
      const r = runCell(0.0, tau2, 5, 'exchangeable', 0, reps, rng);
      if (m === 'vague') console.log('');
      console.log(String(tau2).padEnd(6), m.padEnd(9),
        String(r[m].coverage).padStart(8), String(r[m].rmse).padStart(7), String(r[m].ciWidth).padStart(9));
    }
  }
  console.log('\n## Prior-data CONFLICT (theta_new = muHist + shift) -- robust should protect\n');
  console.log('shift  method    coverage  RMSE     CIwidth');
  for (const shift of [0.5, 1.0, 2.0]) {
    const r = runCell(0.0, 0.05, 5, 'conflict', shift, reps, rng);
    for (const m of METHODS)
      console.log(String(shift).padEnd(6), m.padEnd(9),
        String(r[m].coverage).padStart(8), String(r[m].rmse).padStart(7), String(r[m].ciWidth).padStart(9));
    console.log('');
  }
  console.log(`(${(Date.now() - t0) / 1000}s)`);
}
