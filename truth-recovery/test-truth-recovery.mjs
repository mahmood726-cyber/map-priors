// node --test truth-recovery/test-truth-recovery.mjs
// Measured invariants for the map-priors truth-recovery yardstick. Seeded; no
// hand-entered numbers.
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { generate, makeRng } from './dgp-map.mjs';
import { runCell } from './harness.mjs';

describe('MAP DGP', () => {
  it('is reproducible for a fixed seed', () => {
    const a = generate(0, 0.05, 5, 'exchangeable', 0, makeRng(7));
    const b = generate(0, 0.05, 5, 'exchangeable', 0, makeRng(7));
    assert.deepEqual(a.studies, b.studies);
    assert.equal(a.theta_new, b.theta_new);
  });
});

describe('Truth-recovery (measured)', () => {
  it('exchangeable: robust borrowing cuts RMSE vs a vague prior while staying ~nominal', () => {
    const r = runCell(0.0, 0.02, 5, 'exchangeable', 0, 3000, makeRng(20260613));
    assert.ok(r.robust.rmse < r.vague.rmse, `robust RMSE ${r.robust.rmse} !< vague ${r.vague.rmse}`);
    assert.ok(r.robust.ciWidth < r.vague.ciWidth, `robust CI ${r.robust.ciWidth} !< vague ${r.vague.ciWidth}`);
    assert.ok(r.robust.coverage > 0.92, `robust coverage ${r.robust.coverage} below nominal`);
  });

  it('the PURE MAP prior (w=1) under-covers even under exchangeability (over-confident)', () => {
    const r = runCell(0.0, 0.02, 5, 'exchangeable', 0, 3000, makeRng(20260613));
    assert.ok(r.MAP.coverage < r.vague.coverage - 0.05,
      `pure-MAP coverage ${r.MAP.coverage} not under-covering vs vague ${r.vague.coverage}`);
  });

  it('conflict: pure MAP (w=1) suffers catastrophic bias/under-coverage; robust mixture protects', () => {
    const r = runCell(0.0, 0.05, 5, 'conflict', 1.0, 3000, makeRng(20260615));
    // pure MAP collapses...
    assert.ok(r.MAP.coverage < 0.55, `pure-MAP conflict coverage ${r.MAP.coverage} not collapsed`);
    assert.ok(r.MAP.rmse > r.robust.rmse + 0.1, `pure-MAP RMSE ${r.MAP.rmse} not much worse than robust ${r.robust.rmse}`);
    // ...robust stays calibrated.
    assert.ok(r.robust.coverage > 0.92, `robust conflict coverage ${r.robust.coverage} below nominal`);
  });
});
