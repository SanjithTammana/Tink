// Authored in the same 680 × 270 space as the wordmark. Each segment crosses a letter stroke.
const AUTHORED_SEGMENTS = [
  [[24, 218], [75, 161], [119, 82], [204, 79]],
  [[204, 79], [183, 143], [206, 229], [274, 189]],
  [[274, 189], [318, 155], [335, 113], [386, 157]],
  [[386, 157], [425, 212], [452, 91], [503, 116]],
  [[503, 116], [537, 132], [548, 221], [581, 178]],
  [[581, 178], [627, 130], [638, 52], [704, 37]],
];

const unit = (x, y) => { const m = Math.hypot(x, y) || 1; return [x / m, y / m]; };

// Joined cubic segments are only position-continuous (C0): the incoming and outgoing handles at a joint point in
// different directions, so the heading snaps there. Rotating both handles onto their shared bisector, keeping each
// handle's length, makes the tangent direction continuous (G1) and, with arc-length timing, the velocity too (C1).
export function smoothJoints(segments) {
  const out = segments.map((segment) => segment.map((point) => [...point]));
  for (let i = 0; i < out.length - 1; i += 1) {
    const joint = out[i][3];
    const incoming = out[i][2];
    const outgoing = out[i + 1][1];
    const a = Math.hypot(joint[0] - incoming[0], joint[1] - incoming[1]);
    const b = Math.hypot(outgoing[0] - joint[0], outgoing[1] - joint[1]);
    const [ux, uy] = unit(joint[0] - incoming[0], joint[1] - incoming[1]);
    const [vx, vy] = unit(outgoing[0] - joint[0], outgoing[1] - joint[1]);
    const [dx, dy] = unit(ux + vx, uy + vy);
    out[i][2] = [joint[0] - dx * a, joint[1] - dy * a];
    out[i + 1][1] = [joint[0] + dx * b, joint[1] + dy * b];
  }
  return out;
}

export const FLIGHT_SEGMENTS = smoothJoints(AUTHORED_SEGMENTS);

export function cubicPoint(points, t) {
  const u = 1 - t;
  return [0, 1].map((axis) => u ** 3 * points[0][axis]
    + 3 * u ** 2 * t * points[1][axis]
    + 3 * u * t ** 2 * points[2][axis]
    + t ** 3 * points[3][axis]);
}

export function cubicDerivative(points, t) {
  const u = 1 - t;
  return [0, 1].map((axis) => 3 * u ** 2 * (points[1][axis] - points[0][axis])
    + 6 * u * t * (points[2][axis] - points[1][axis])
    + 3 * t ** 2 * (points[3][axis] - points[2][axis]));
}

// De Casteljau: the part of a cubic from t to 1, as a cubic of its own (same curve, same tangent).
export function splitCubicFrom(points, t) {
  const mix = (a, b) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const [p0, p1, p2, p3] = points;
  const p01 = mix(p0, p1); const p12 = mix(p1, p2); const p23 = mix(p2, p3);
  const p012 = mix(p01, p12); const p123 = mix(p12, p23);
  return [mix(p012, p123), p123, p23, p3];
}

// A route is a chain of cubics sampled by arc length, so equal steps of `progress` are equal distances.
export function createRoute(segments, steps = 64) {
  const samples = [{ segment: 0, t: 0, distance: 0 }];
  let distance = 0;
  let previous = segments[0][0];
  segments.forEach((segment, index) => {
    if (index) samples.push({ segment: index, t: 0, distance });
    for (let step = 1; step <= steps; step += 1) {
      const t = step / steps;
      const point = cubicPoint(segment, t);
      distance += Math.hypot(point[0] - previous[0], point[1] - previous[1]);
      samples.push({ segment: index, t, distance });
      previous = point;
    }
  });
  const total = distance;

  function at(progress) {
    const target = Math.max(0, Math.min(1, progress)) * total;
    let low = 0;
    let high = samples.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (samples[middle].distance < target) low = middle + 1;
      else high = middle;
    }
    const right = samples[low];
    const left = samples[Math.max(0, low - 1)];
    const portion = right.distance === left.distance ? 0 : (target - left.distance) / (right.distance - left.distance);
    const t = left.segment === right.segment ? left.t + (right.t - left.t) * portion : right.t * portion;
    const segment = segments[right.segment];
    const point = cubicPoint(segment, t);
    const derivative = cubicDerivative(segment, t);
    const [tangentX, tangentY] = unit(derivative[0], derivative[1]);
    return { x: point[0], y: point[1], tangentX, tangentY, segment: right.segment, t };
  }
  return { at, total };
}

const defaultRoute = createRoute(FLIGHT_SEGMENTS, 48);
export const sampleFlight = defaultRoute.at;

export function easeFlight(progress) {
  const p = Math.max(0, Math.min(1, progress));
  return (1 - Math.cos(Math.PI * p)) / 2;
}

// Distance covered over the whole route as a function of time: starts at rest, accelerates smoothly (C1) and
// never stops again, so she leaves the screen at speed instead of halting at the end of each phase.
export function easeRoute(progress) {
  const u = Math.max(0, Math.min(1, progress));
  return u * u * (1.6 - .6 * u);
}

// Steering from two vectors. Both are normalised first and the dot is clamped, so rounding can never push it
// outside [-1, 1]. The signed angle uses atan2(cross, dot), which is continuous across the whole circle and
// always the shortest rotation, unlike acos.
export function steer(ax, ay, bx, by) {
  const [nax, nay] = unit(ax, ay);
  const [nbx, nby] = unit(bx, by);
  const dot = Math.max(-1, Math.min(1, nax * nbx + nay * nby));
  const cross = nax * nby - nay * nbx;
  return { dot, cross, angle: Math.atan2(cross, dot) };
}

// ---- Pre-determined flight plan ---------------------------------------------------------------------------------
// The whole flight is computed once, before the first frame, and the animation only reads it back by time.
//
// Path: a natural cubic spline through the knots, parameterised by chord length. Position, tangent AND curvature
// are continuous at every knot (C2), so there is no corner for the heading or the dot products to snap at.
//
// Speed: v follows curvature. Tight turns slow her down, straights let her run, and a first-order lag on the
// speed means acceleration is continuous as well. Time then runs through a fixed-step simulation, so the result
// is the same on any display.
function solveSecondDerivatives(values, h) {
  const n = values.length;
  const m = new Array(n).fill(0);
  if (n < 3) return m;
  // Thomas algorithm for the tridiagonal system
  const lower = []; const mid = []; const upper = []; const rhs = [];
  for (let i = 1; i < n - 1; i += 1) {
    lower.push(h[i - 1]); mid.push(2 * (h[i - 1] + h[i])); upper.push(h[i]);
    rhs.push(6 * ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1]));
  }
  for (let i = 1; i < mid.length; i += 1) {
    const w = lower[i] / mid[i - 1];
    mid[i] -= w * upper[i - 1];
    rhs[i] -= w * rhs[i - 1];
  }
  const solution = new Array(mid.length);
  solution[mid.length - 1] = rhs[mid.length - 1] / mid[mid.length - 1];
  for (let i = mid.length - 2; i >= 0; i -= 1) solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / mid[i];
  solution.forEach((value, i) => { m[i + 1] = value; });
  return m;
}

export function createSpline(knots) {
  const h = [];
  const tau = [0];
  for (let i = 1; i < knots.length; i += 1) {
    h.push(Math.hypot(knots[i][0] - knots[i - 1][0], knots[i][1] - knots[i - 1][1]));
    tau.push(tau[i - 1] + h[i - 1]);
  }
  const axes = [0, 1].map((axis) => {
    const values = knots.map((k) => k[axis]);
    return { values, m: solveSecondDerivatives(values, h) };
  });
  const length = tau[tau.length - 1];

  function evaluate(parameter) {
    const p = Math.max(0, Math.min(length, parameter));
    let i = 0;
    while (i < h.length - 1 && p > tau[i + 1]) i += 1;
    const d = p - tau[i];
    const a = h[i] - d;
    const out = { pos: [0, 0], vel: [0, 0], acc: [0, 0] };
    axes.forEach(({ values, m }, axis) => {
      const lo = values[i] / h[i] - m[i] * h[i] / 6;
      const hi = values[i + 1] / h[i] - m[i + 1] * h[i] / 6;
      out.pos[axis] = m[i] * a ** 3 / (6 * h[i]) + m[i + 1] * d ** 3 / (6 * h[i]) + lo * a + hi * d;
      out.vel[axis] = -m[i] * a ** 2 / (2 * h[i]) + m[i + 1] * d ** 2 / (2 * h[i]) - lo + hi;
      out.acc[axis] = m[i] * a / h[i] + m[i + 1] * d / h[i];
    });
    return out;
  }
  return { evaluate, length };
}

const smootherstep = (x) => { const t = Math.max(0, Math.min(1, x)); return t * t * t * (t * (6 * t - 15) + 10); };

export function buildFlightPlan(knots, options = {}) {
  const {
    durationMs = 2400, rampMs = 450, curvatureScale = 60, lookAhead = 70, speedLag = .14, facingLag = .13, leanLag = .26,
    wingStartMs = 200, wingSlowMs = 760, wingFastMs = 520,
  } = options;
  const spline = createSpline(knots);

  // Dense samples of the path with unit tangent and signed curvature (kappa = (x'y'' - y'x'') / |v|^3).
  const count = Math.max(200, Math.ceil(spline.length));
  const dense = [];
  let arc = 0;
  for (let i = 0; i <= count; i += 1) {
    const { pos, vel, acc } = spline.evaluate(spline.length * i / count);
    const speed = Math.hypot(vel[0], vel[1]) || 1;
    if (i) arc += Math.hypot(pos[0] - dense[i - 1].x, pos[1] - dense[i - 1].y);
    dense.push({ x: pos[0], y: pos[1], tx: vel[0] / speed, ty: vel[1] / speed, kappa: (vel[0] * acc[1] - vel[1] * acc[0]) / speed ** 3, arc });
  }
  // The curvature of a C2 spline is already continuous; a light moving average removes sampling ripple.
  for (let pass = 0; pass < 3; pass += 1) {
    const copy = dense.map((p) => p.kappa);
    for (let i = 0; i < dense.length; i += 1) {
      let sum = 0; let n = 0;
      for (let j = Math.max(0, i - 12); j <= Math.min(dense.length - 1, i + 12); j += 1) { sum += copy[j]; n += 1; }
      dense[i].kappa = sum / n;
    }
  }
  const total = dense[dense.length - 1].arc;
  // Anticipation: she slows for a turn before she reaches it, using the sharpest curvature in the next stretch
  // of path (a running maximum of a continuous function is continuous, then smoothed again).
  const stepLength = total / (dense.length - 1);
  const window = Math.ceil(lookAhead / stepLength);
  const ahead = dense.map((_, i) => { let m = 0; for (let j = i; j <= Math.min(dense.length - 1, i + window); j += 1) m = Math.max(m, Math.abs(dense[j].kappa)); return m; });
  for (let pass = 0; pass < 3; pass += 1) {
    const copy = ahead.slice();
    for (let i = 0; i < ahead.length; i += 1) {
      let sum = 0; let n = 0;
      for (let j = Math.max(0, i - 15); j <= Math.min(ahead.length - 1, i + 15); j += 1) { sum += copy[j]; n += 1; }
      ahead[i] = sum / n;
    }
  }
  dense.forEach((p, i) => { p.ahead = ahead[i]; });
  const sampleAt = (distance) => {
    const target = Math.max(0, Math.min(total, distance));
    let low = 0; let high = dense.length - 1;
    while (low < high) { const mid = (low + high) >> 1; if (dense[mid].arc < target) low = mid + 1; else high = mid; }
    const b = dense[low]; const a = dense[Math.max(0, low - 1)];
    const f = b.arc === a.arc ? 0 : (target - a.arc) / (b.arc - a.arc);
    const lerp = (x, y) => x + (y - x) * f;
    const tx = lerp(a.tx, b.tx); const ty = lerp(a.ty, b.ty);
    const m = Math.hypot(tx, ty) || 1;
    return { x: lerp(a.x, b.x), y: lerp(a.y, b.y), tx: tx / m, ty: ty / m, kappa: lerp(a.kappa, b.kappa), ahead: lerp(a.ahead, b.ahead) };
  };

  const DT = .001;
  function simulate(timeScale) {
    const frames = [];
    const cruise = total / (durationMs / 1000) * 1.25;
    let s = 0; let v = 0; let facing = 1; let lean = 0; let flap = 0; let t = 0;
    for (let step = 0; step < 20000; step += 1) {
      const p = sampleAt(s);
      const target = cruise / (1 + p.ahead * curvatureScale);
      v += (target - v) * (1 - Math.exp(-DT / speedLag));
      const ramp = smootherstep(t * 1000 / rampMs);
      const heading = steer(1, 0, p.tx, p.ty);
      const facingTarget = Math.tanh(3 * heading.dot);
      const angularRate = p.kappa * v * ramp;   // rad/s the heading is turning at
      const leanTarget = 35 * Math.tanh((Math.atan2(heading.cross, Math.sqrt(heading.dot ** 2 + .04)) * 180 / Math.PI * .48 + 2.2 * angularRate * 180 / Math.PI * .1) / 35);
      facing += (facingTarget - facing) * (1 - Math.exp(-DT / facingLag));
      lean += (leanTarget - lean) * (1 - Math.exp(-DT / leanLag));
      const realMs = t * timeScale * 1000;
      if (realMs >= wingStartMs) flap += DT * timeScale * 1000 / (wingSlowMs + (wingFastMs - wingSlowMs) * smootherstep(s / total));
      frames.push({ x: p.x, y: p.y, tx: p.tx, ty: p.ty, speed: v * ramp, facing, lean, flap });
      if (s >= total - 1e-6) break;
      s = Math.min(total, s + v * ramp * DT);
      t += DT;
    }
    return frames;
  }
  const first = simulate(1);
  const scale = (durationMs / 1000) / (first.length * DT);
  const frames = simulate(scale);
  const last = frames.length - 1;

  function at(u) {
    const position = Math.max(0, Math.min(1, u)) * last;
    const i = Math.floor(position);
    const f = position - i;
    const a = frames[i]; const b = frames[Math.min(i + 1, last)];
    const lerp = (x, y) => x + (y - x) * f;
    const tx = lerp(a.tx, b.tx); const ty = lerp(a.ty, b.ty);
    const m = Math.hypot(tx, ty) || 1;
    return { x: lerp(a.x, b.x), y: lerp(a.y, b.y), tangentX: tx / m, tangentY: ty / m, speed: lerp(a.speed, b.speed) / scale,facing: lerp(a.facing, b.facing), lean: lerp(a.lean, b.lean), flap: lerp(a.flap, b.flap) };
  }
  return { at, frames, spline, total };
}

// Frame-rate independent exponential smoothing: the same curve at 30, 60 or 144 Hz.
export function approach(current, target, dt, timeConstant) {
  return current + (target - current) * (1 - Math.exp(-dt / timeConstant));
}

// Moves an angle toward a target along the shortest way round (no 359° to 1° swings).
export function approachAngle(current, target, dt, timeConstant) {
  const difference = Math.atan2(Math.sin(target - current), Math.cos(target - current));
  return current + difference * (1 - Math.exp(-dt / timeConstant));
}
