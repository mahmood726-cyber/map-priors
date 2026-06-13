// engine.mjs -- pure MAP-prior derivation core EXTRACTED VERBATIM from
// map-priors.html (normalPDF/CDF/Quantile, moritaESSMixture, deriveMAPContinuous).
// The function reads a module-level 'studies' array (as the app does); set it
// via setStudies() before calling. The SAME math the app ships is measured.

let studies = [];
export function setStudies(s){ studies = s; }

function normalPDF(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}
function normalCDF(x) {
  if (x === 0) return 0.5;
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.2316419 * x);
  const d = 0.3989422804014327 * Math.exp(-0.5 * x * x);
  const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return s < 0 ? p : 1 - p;
}
function normalQuantile(p) {
  if (p <= 0) return -Infinity; if (p >= 1) return Infinity; if (p === 0.5) return 0;
  const a=[-3.969683028665376e1,2.209460984245205e2,-2.759285104469687e2,1.383577518672690e2,-3.066479806614716e1,2.506628277459239e0];
  const b=[-5.447609879822406e1,1.615858368580409e2,-1.556989798598866e2,6.680131188771972e1,-1.328068155288572e1];
  const c=[-7.784894002430293e-3,-3.223964580411365e-1,-2.400758277161838e0,-2.549732539343734e0,4.374664141464968e0,2.938163982698783e0];
  const d=[7.784695709041462e-3,3.224671290700398e-1,2.445134137142996e0,3.754408661907416e0];
  const pL=0.02425; let q,r;
  if(p<pL){q=Math.sqrt(-2*Math.log(p));return(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)}
  if(p<=1-pL){q=p-0.5;r=q*q;return(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)}
  q=Math.sqrt(-2*Math.log(1-p));return-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
}


function moritaESSMixture(mu_map, var_map, w, mu_vague, var_vague) {
  // Find the mode numerically (near mu_map for w > 0.5)
  const se_map = Math.sqrt(var_map);
  const se_vague = Math.sqrt(var_vague);
  let mode = mu_map;
  // Newton-Raphson to find mode of mixture
  for (let iter = 0; iter < 30; iter++) {
    const f1 = w * normalPDF(mode, mu_map, se_map);
    const f2 = (1-w) * normalPDF(mode, mu_vague, se_vague);
    const f = f1 + f2;
    if (f < 1e-300) break;
    // f'(x) = -w*N*(x-mu1)/var1 - (1-w)*N*(x-mu2)/var2
    const fp = -f1*(mode-mu_map)/var_map - f2*(mode-mu_vague)/var_vague;
    // f''(x) for Newton on f'=0
    const fpp = f1*((mode-mu_map)**2/var_map - 1)/var_map
              + f2*((mode-mu_vague)**2/var_vague - 1)/var_vague;
    // We want f'(mode) = 0, so Newton step on f':
    const wfpp = -f1/var_map + f1*(mode-mu_map)**2/(var_map*var_map)
               - f2/var_vague + f2*(mode-mu_vague)**2/(var_vague*var_vague);
    if (Math.abs(wfpp) < 1e-300) break;
    const step = fp / wfpp;
    mode -= step;
    if (Math.abs(step) < 1e-12) break;
  }

  // Compute -d²log f / dθ² at the mode
  const f1 = w * normalPDF(mode, mu_map, se_map);
  const f2 = (1-w) * normalPDF(mode, mu_vague, se_vague);
  const f = f1 + f2;
  if (f < 1e-300) return 1 / var_map; // fallback

  // f''(x) = f1*[(x-mu1)^2/var1^2 - 1/var1] + f2*[(x-mu2)^2/var2^2 - 1/var2]
  const fpp = f1*((mode-mu_map)**2/(var_map*var_map) - 1/var_map)
            + f2*((mode-mu_vague)**2/(var_vague*var_vague) - 1/var_vague);

  // f'(mode) should be ~0 at the mode
  const fp = -f1*(mode-mu_map)/var_map - f2*(mode-mu_vague)/var_vague;

  // -d²log f/dθ² = -(f''*f - f'^2) / f^2 = -f''/f + (f'/f)^2
  const negCurv = -fpp/f + (fp/f)**2;

  // This is the effective precision of the mixture at the mode
  return Math.max(0, negCurv);
}

// ---- MAP Prior Derivation ----

function deriveMAPContinuous(w, confLevel) {
  const k = studies.length;
  const yi = studies.map(s => s.mean);
  const vi = studies.map(s => (s.sd * s.sd) / s.n);

  // DL
  const wi = vi.map(v => 1/v);
  const sumW = wi.reduce((a,b)=>a+b,0);
  const mu_fe = wi.reduce((s,w,i)=>s+w*yi[i],0)/sumW;
  const Q = wi.reduce((s,w,i)=>s+w*(yi[i]-mu_fe)**2,0);
  const C = sumW - wi.reduce((s,w)=>s+w*w,0)/sumW;
  let tau2 = Math.max(0, (Q-(k-1))/C);

  // REML
  for (let iter = 0; iter < 50; iter++) {
    const wi2 = vi.map(v=>1/(v+tau2));
    const sw2 = wi2.reduce((a,b)=>a+b,0);
    const mu = wi2.reduce((s,w,i)=>s+w*yi[i],0)/sw2;
    const wi3 = wi2.map(w=>w*w);
    const r2 = yi.map((y,i)=>(y-mu)**2);
    const dL = -0.5*wi2.reduce((s,w)=>s+w,0) + 0.5*wi3.reduce((s,w)=>s+w,0)/sw2 + 0.5*wi3.reduce((s,w,i)=>s+w*r2[i],0);
    // Expected Fisher information: 0.5 * tr(P^2)
    const sumW2 = wi3.reduce((s,w)=>s+w,0);
    const sumW3 = wi2.reduce((s,w)=>s+w*w*w,0);
    const ddL = 0.5*sumW2 - sumW3/sw2 + 0.5*(sumW2*sumW2)/(sw2*sw2);
    if (Math.abs(ddL)<1e-15) break;
    tau2 = Math.max(0, tau2 + dL/ddL);
    if (Math.abs(dL/ddL)<1e-10) break;
  }

  const wi_f = vi.map(v=>1/(v+tau2));
  const sw_f = wi_f.reduce((a,b)=>a+b,0);
  const mu_post = wi_f.reduce((s,w,i)=>s+w*yi[i],0)/sw_f;
  const se_mu = Math.sqrt(1/sw_f);

  const map_var = tau2 + se_mu*se_mu;
  const map_se = Math.sqrt(map_var);

  // Vague: centered on grand mean with very large variance
  const grand_sd = Math.sqrt(vi.reduce((s,v)=>s+v,0)/k) * 10;
  const vague_mu = mu_post;
  const vague_var = grand_sd * grand_sd;

  const robust_mu = w*mu_post + (1-w)*vague_mu;
  const robust_var = w*(map_var + mu_post*mu_post) + (1-w)*(vague_var+vague_mu*vague_mu) - robust_mu*robust_mu;
  const robust_se = Math.sqrt(Math.max(0.001, robust_var));

  // ESS: Morita method — use per-patient Fisher information (1/sigma^2)
  const avg_sigma2 = studies.reduce((s,st) => s + st.sd * st.sd, 0) / k;
  const single_info_cont = 1 / avg_sigma2; // per-patient info
  const ess_map = (1/map_var) / single_info_cont;
  const ess_robust = moritaESSMixture(mu_post, map_var, w, vague_mu, vague_var) / single_info_cont;

  const z = normalQuantile(1-(1-confLevel)/2);

  return {
    type: 'continuous',
    mu: mu_post, se_mu, tau2, tau: Math.sqrt(tau2),
    map_mu: mu_post, map_se, map_var,
    robust_mu, robust_se, robust_var,
    vague_mu, vague_se: Math.sqrt(vague_var),
    map_lower: mu_post - z*map_se, map_upper: mu_post + z*map_se,
    robust_lower: robust_mu - z*robust_se, robust_upper: robust_mu + z*robust_se,
    ess_map, ess_robust,
    k, Q, I2: Q>(k-1) ? Math.max(0,(Q-(k-1))/Q*100) : 0,
    w, confLevel, yi, vi
  };
}

export { deriveMAPContinuous, normalPDF, normalCDF, normalQuantile, moritaESSMixture };
