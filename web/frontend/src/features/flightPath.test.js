import assert from 'node:assert/strict';
import { test } from 'node:test';
import { easeFlight, sampleFlight } from './flightPath.js';

test('flight starts and finishes at the authored endpoints with unit tangents', () => {
  const start = sampleFlight(0);
  const end = sampleFlight(1);
  assert.deepEqual([start.x, start.y], [24, 218]);
  assert.deepEqual([end.x, end.y], [704, 37]);
  for (const progress of [0, .15, .4, .7, 1]) {
    const { tangentX, tangentY } = sampleFlight(progress);
    assert.ok(Math.abs(Math.hypot(tangentX, tangentY) - 1) < 1e-9);
  }
});

test('eased distance advances monotonically through the flight', () => {
  let previous = -1;
  for (let step = 0; step <= 100; step += 1) {
    const distance = easeFlight(step / 100);
    assert.ok(distance >= previous);
    previous = distance;
  }
  assert.equal(easeFlight(0), 0);
  assert.equal(easeFlight(1), 1);
});

test('the flight path ends inside the window, so the exit curve must carry her off-screen', async () => {
  const { cubicPoint } = await import('./flightPath.js');
  const end = sampleFlight(1);
  // exit curve as built in Intro.jsx for a 1440x900 window with the 680-wide mark centred at scale 1.35
  const scale = 920 / 680;
  const left = 260;
  const top = 300;
  const target = [(1440 + 220 - left) / scale, (-220 - top) / scale];
  const start = [end.x, end.y];
  const finish = [Math.max(target[0], start[0] + 200), Math.min(target[1], start[1] - 60)];
  const curve = [start, [start[0] + end.tangentX * 70, start[1] + end.tangentY * 70],
    [finish[0] - (finish[0] - start[0]) * .3, finish[1] - (finish[1] - start[1]) * .3], finish];
  const [x, y] = cubicPoint(curve, 1);
  assert.ok(left + x * scale > 1440 && top + y * scale < 0);
});

test('tangent direction is continuous at every joint of the route', async () => {
  const { FLIGHT_SEGMENTS, cubicDerivative } = await import('./flightPath.js');
  for (let i = 0; i < FLIGHT_SEGMENTS.length - 1; i += 1) {
    const [ax, ay] = cubicDerivative(FLIGHT_SEGMENTS[i], 1);
    const [bx, by] = cubicDerivative(FLIGHT_SEGMENTS[i + 1], 0);
    const dot = (ax * bx + ay * by) / (Math.hypot(ax, ay) * Math.hypot(bx, by));
    assert.ok(dot > 1 - 1e-9, `joint ${i}: dot ${dot}`);
  }
});

test('sampled heading and speed change gradually, with no jumps', async () => {
  const { easeRoute, createRoute, FLIGHT_SEGMENTS, steer } = await import('./flightPath.js');
  const route = createRoute(FLIGHT_SEGMENTS);
  let prev = route.at(0); let worstTurn = 0; let worstSpeed = 0; let lastStep = null;
  for (let i = 1; i <= 2000; i += 1) {
    const u = i / 2000;
    const p = route.at(easeRoute(u));
    const turn = Math.abs(steer(prev.tangentX, prev.tangentY, p.tangentX, p.tangentY).angle);
    worstTurn = Math.max(worstTurn, turn);
    const step = Math.hypot(p.x - prev.x, p.y - prev.y);
    if (lastStep !== null) worstSpeed = Math.max(worstSpeed, Math.abs(step - lastStep));
    lastStep = step; prev = p;
  }
  assert.ok(worstTurn < .08, `largest heading change per step ${worstTurn}`);
  assert.ok(worstSpeed < .6, `largest speed change per step ${worstSpeed}`);
});

test('steer clamps the dot and gives the shortest signed angle', async () => {
  const { steer, easeRoute } = await import('./flightPath.js');
  assert.ok(Math.abs(steer(3, 0, 3, 1e-17).dot) <= 1);
  assert.ok(Math.abs(steer(1, 0, -1, 1e-9).angle) > 3.14);
  assert.ok(Math.abs(steer(1, 0, 0, 1).angle - Math.PI / 2) < 1e-12);
  assert.ok(easeRoute(1) === 1 && easeRoute(0) === 0);
});

async function demoPlan() {
  const F = await import('./flightPath.js');
  const knots = [[306, 125]];
  for (let i = 0; i <= 12; i += 1) { const p = F.sampleFlight(.5 + .5 * i / 12); knots.push([p.x, p.y]); }
  const e = F.sampleFlight(1);
  knots.push([e.x + e.tangentX * 160, e.y + e.tangentY * 160], [1250, -220]);
  return { F, knots, plan: F.buildFlightPlan(knots) };
}

test('the spline is C2: position, velocity and acceleration agree on both sides of every knot', async () => {
  const { F, knots } = await demoPlan();
  const spline = F.createSpline(knots);
  let tau = 0;
  for (let i = 1; i < knots.length - 1; i += 1) {
    tau += Math.hypot(knots[i][0] - knots[i - 1][0], knots[i][1] - knots[i - 1][1]);
    const a = spline.evaluate(tau - 1e-7); const b = spline.evaluate(tau + 1e-7);
    for (const key of ['pos', 'vel', 'acc']) for (const axis of [0, 1]) assert.ok(Math.abs(a[key][axis] - b[key][axis]) < 1e-3, `${key} jumps at knot ${i}`);
  }
});

test('flight plan starts at rest, ends at the exit, and never jumps in speed, heading, lean or facing', async () => {
  const { F, plan } = await demoPlan();
  assert.equal(plan.at(0).speed, 0);
  assert.deepEqual([plan.at(0).x, plan.at(0).y], [306, 125]);
  assert.ok(plan.at(1).speed > 300);
  let prev = plan.at(0);
  let worstSpeed = 0; let worstLean = 0; let worstFacing = 0;
  for (let i = 1; i <= 2400; i += 1) {
    const p = plan.at(i / 2400);
    worstSpeed = Math.max(worstSpeed, Math.abs(p.speed - prev.speed));
    worstLean = Math.max(worstLean, Math.abs(p.lean - prev.lean));
    worstFacing = Math.max(worstFacing, Math.abs(p.facing - prev.facing));
    assert.ok(p.flap >= prev.flap);
    prev = p;
  }
  assert.ok(worstSpeed < 5, `speed step ${worstSpeed}`);
  assert.ok(worstLean < .6, `lean step ${worstLean}`);
  assert.ok(worstFacing < .005, `facing step ${worstFacing}`);
});

test('she slows for tight turns and runs on straights', async () => {
  const { plan } = await demoPlan();
  const speeds = plan.frames.map((f) => f.speed).slice(600);
  assert.ok(Math.max(...speeds) > 1.8 * Math.min(...speeds));
});
