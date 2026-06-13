// ============================================================
// dgp-map.mjs -- Known-truth DGP for MAP-PRIOR / dynamic-borrowing evaluation.
//
// map-priors derives a Meta-Analytic-Predictive (MAP) prior from historical
// studies and a robust (vague-mixture) version of it. The claims it exists to
// support are operating characteristics (Schmidli et al. 2014): borrowing should
// improve precision WITHOUT bias when the new trial is exchangeable with history,
// and the robust component should LIMIT the bias under prior-data CONFLICT. Only
// simulation against a known truth can confirm that. This DGP supplies it.
//
// Historical studies: y_h ~ N(mu_hist, tau2 + sigma^2/n_h)  (random-effects).
// New-trial true parameter theta_new:
//   exchangeable: theta_new ~ N(mu_hist, tau2)
//   conflict:     theta_new = mu_hist + shift
// New-trial data: y_new ~ N(theta_new, sigma^2/n_new).
//
// Seeded -> reproducible. Standalone.
// ============================================================

export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng) {
  let u1 = rng(), u2 = rng();
  if (u1 < 1e-12) u1 = 1e-12;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function generate(muHist, tau2, kHist, scenario, shift, rng,
                         { sigma = 1.0, nHist = 60, nNew = 60 } = {}) {
  const tau = Math.sqrt(tau2);
  const studies = [];
  for (let i = 0; i < kHist; i++) {
    const theta_h = muHist + tau * randn(rng);
    const se_h = sigma / Math.sqrt(nHist);
    const y_h = theta_h + se_h * randn(rng);
    studies.push({ mean: y_h, sd: sigma, n: nHist });
  }
  const theta_new = scenario === 'conflict' ? muHist + shift : muHist + tau * randn(rng);
  const se_new = sigma / Math.sqrt(nNew);
  const y_new = theta_new + se_new * randn(rng);
  return { studies, theta_new, y_new, se_new, info: { muHist, tau2, scenario } };
}
