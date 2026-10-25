import { useEffect, useRef } from 'react';
import { buildFlightPlan, sampleFlight } from './flightPath';
import './intro.css';

const STROKES = [
  { name: 'T flourish', d: 'M70 111 C100 78 156 64 219 73 C244 76 262 88 271 99 C228 80 174 86 141 105', width: 4 },
  { name: 'T stem', d: 'M187 77 C181 124 170 176 157 216 C151 234 162 228 177 212 C195 193 208 169 215 151', width: 9 },
  { name: 'i stem', d: 'M300 137 C295 165 283 201 288 213 C295 227 320 190 329 175', width: 8 },
  { name: 'i dot', d: 'M310 106 C306 103 302 108 306 111 C310 114 314 108 310 106', width: 4 },
  { name: 'n', d: 'M351 139 C344 166 331 206 337 217 C343 226 353 204 364 179 C378 149 391 134 402 141 C414 149 393 193 398 211 C402 227 424 206 439 182', width: 8 },
  { name: 'k', d: 'M492 81 C482 127 458 207 466 216 C473 225 489 183 500 163 C516 136 543 135 542 151 C540 169 502 180 483 180 C511 181 522 222 541 216', width: 8 },
];

const DUST_COUNT = 18;
const LIFT_START_MS = 700;
const SPIN_START_MS = 1000;
const SPIN_END_MS = 1450;
const EXIT_END_MS = 3850; // after the authored flight she keeps going, off the edge of the screen
const REVEAL_START_MS = 3850;
const DURATION_MS = 4550;
const FLIGHT_PATH_START = .5;
const STANDING = { x: 306, y: 185 };
const FRAME_X_OFFSET = [0, -5, -5, 0, 8, -5, -4, 5, 4, 7];

function Fairy({ fairyRef, firstFrameRef, secondFrameRef }) {
  return (
    <g ref={fairyRef} className="intro-fairy" transform="translate(306 185)" aria-hidden="true">
      <image ref={firstFrameRef} className="intro-fairy-frame" href="/fairy/frame-0.webp" x="-43" y="-112" width="86" height="160" />
      <image ref={secondFrameRef} className="intro-fairy-frame" href="/fairy/frame-1.webp" x="-43" y="-112" width="86" height="160" opacity="0" />
    </g>
  );
}

export default function Intro({ onComplete, reducedMotion }) {
  const overlayRef = useRef(null);
  const svgRef = useRef(null);
  const skipRef = useRef(null);
  const fairyRef = useRef(null);
  const firstFrameRef = useRef(null);
  const secondFrameRef = useRef(null);
  const strokeRefs = useRef([]);
  const dustRefs = useRef([]);

  useEffect(() => {
    skipRef.current?.focus();
    const lengths = strokeRefs.current.map((path) => path.getTotalLength());
    strokeRefs.current.forEach((path, index) => {
      path.style.strokeDasharray = `${lengths[index]}`;
      path.style.strokeDashoffset = STROKES[index].name.startsWith('i ') ? `${lengths[index]}` : '0';
    });

    const onKeyDown = (event) => { if (event.key === 'Escape') onComplete(); };
    window.addEventListener('keydown', onKeyDown);

    if (reducedMotion) {
      const timeout = window.setTimeout(onComplete, 1200);
      return () => { window.clearTimeout(timeout); window.removeEventListener('keydown', onKeyDown); };
    }

    const frames = Array.from({ length: 10 }, (_, index) => {
      const image = new Image();
      image.src = `/fairy/frame-${index}.webp`;
      return image;
    });
    const frameElements = [firstFrameRef.current, secondFrameRef.current];
    const setFrame = (element, index, opacity) => {
      const source = frames[index].src;
      if (element.href.baseVal !== source && !element.href.baseVal.endsWith(`/fairy/frame-${index}.webp`)) element.setAttribute('href', source);
      element.setAttribute('x', -43 + FRAME_X_OFFSET[index]);
      element.setAttribute('opacity', opacity);
    };
    const pose = (sequenceFrames, phase) => {
      const position = Math.max(0, Math.min(1, phase)) * (sequenceFrames.length - 1);
      const first = Math.floor(position);
      const blend = Math.max(0, Math.min(1, (position - first - .78) / .22));
      setFrame(frameElements[0], sequenceFrames[first], 1 - blend);
      setFrame(frameElements[1], sequenceFrames[Math.min(first + 1, sequenceFrames.length - 1)], blend);
    };
    const smooth = (value) => {
      const t = Math.max(0, Math.min(1, value));
      return t * t * (3 - 2 * t);
    };
    // The whole flight is planned once, here, before the first frame: a C2 spline (continuous position, heading
    // and curvature) through knots taken from the authored path, with speed that follows curvature. The animation
    // below only reads the plan back by time. The exit knots reach beyond the real viewport (screen pixels mapped
    // into SVG space), so it works at any window size.
    const hover = [STANDING.x, STANDING.y - 60];
    const knots = [hover];
    for (let i = 0; i <= 12; i += 1) {
      const p = sampleFlight(FLIGHT_PATH_START + (1 - FLIGHT_PATH_START) * i / 12);
      knots.push([p.x, p.y]);
    }
    const pathEnd = sampleFlight(1);
    const matrix = svgRef.current?.getScreenCTM?.();
    if (matrix) {
      const target = new DOMPoint(window.innerWidth + 220, -220).matrixTransform(matrix.inverse());
      const end = [Math.max(target.x, pathEnd.x + 200), Math.min(target.y, pathEnd.y - 60)];
      knots.push([pathEnd.x + pathEnd.tangentX * 160, pathEnd.y + pathEnd.tangentY * 160], end);
    }
    const plan = buildFlightPlan(knots, { durationMs: EXIT_END_MS - SPIN_END_MS });
    const JOIN_BLEND_MS = 200;
    let frameId;
    let startedAt;
    let lastEmission = -100;
    let emissionCount = 0;
    const dust = Array.from({ length: DUST_COUNT }, () => ({ born: -1000, x: 0, y: 0, radius: 0 }));

    function animate(now) {
      if (startedAt === undefined) startedAt = now;
      const elapsed = now - startedAt;
      let point = { ...STANDING, tangentX: 1, tangentY: 0 };
      let lean = 0;
      let facing = 1;
      let turn = 1;
      let bobAmount = 0;

      if (elapsed < 450) pose([0], 0);
      else if (elapsed < LIFT_START_MS) pose([0, 1, 2, 1, 0], (elapsed - 450) / (LIFT_START_MS - 450));
      else if (elapsed < SPIN_START_MS) {
        const lift = smooth((elapsed - LIFT_START_MS) / (SPIN_START_MS - LIFT_START_MS));
        point.y = STANDING.y - 60 * lift;
        pose([0, 1, 2, 3, 4], lift);
      } else if (elapsed < SPIN_END_MS) {
        const spin = (elapsed - SPIN_START_MS) / (SPIN_END_MS - SPIN_START_MS);
        point.y = STANDING.y - 60;
        const width = Math.cos(spin * Math.PI * 2);
        turn = width || 1e-3;   // the squash through zero is continuous; no sign jump
        pose([4, 3, 2, 1, 0, 1, 2, 3, 4], spin);
      } else {
        // Read the pre-computed flight at this moment. Position, heading, facing, lean and wingbeat are all
        // continuous functions of time, so nothing here is smoothed or decided frame by frame.
        const u = Math.min((elapsed - SPIN_END_MS) / (EXIT_END_MS - SPIN_END_MS), 1);
        const state = plan.at(u);
        point = state;
        facing = state.facing;
        lean = state.lean;
        const sinceJoin = elapsed - SPIN_END_MS;
        if (sinceJoin < JOIN_BLEND_MS) pose([4, 5], smooth(sinceJoin / JOIN_BLEND_MS));
        else pose([5, 6, 7, 8, 9, 8, 7, 6, 5], state.flap % 1);
        bobAmount = smooth(sinceJoin / 300) * (1 - smooth((elapsed - (EXIT_END_MS - 500)) / 500));
      }
      const bobY = Math.sin(elapsed * .011) * 1.1 * bobAmount;
      const bobX = Math.sin(elapsed * .0073 + 1.3) * .7 * bobAmount;
      fairyRef.current?.setAttribute('transform', `translate(${point.x + bobX} ${point.y + bobY}) rotate(${lean}) scale(${facing * turn} 1)`);

      strokeRefs.current.forEach((path, index) => {
        const stroke = STROKES[index];
        if (!stroke.name.startsWith('i ')) return;
        const start = stroke.name === 'i dot' ? 1260 : 1050;
        const drawn = smooth((elapsed - start) / 440);
        path.style.opacity = drawn > 0 ? '1' : '0';
        path.style.strokeDashoffset = `${lengths[index] * (1 - drawn)}`;
      });

      if (elapsed >= LIFT_START_MS && elapsed < EXIT_END_MS && elapsed - lastEmission >= 100) {
        lastEmission = elapsed;
        const index = emissionCount % DUST_COUNT;
        const side = emissionCount % 2 ? 1 : -1;
        const normalX = -point.tangentY;
        const normalY = point.tangentX;
        dust[index] = {
          born: elapsed,
          x: point.x - point.tangentX * 15 + normalX * side * (3 + emissionCount % 4),
          y: point.y - point.tangentY * 15 + normalY * side * (3 + emissionCount % 4),
          radius: emissionCount % 5 === 0 ? 2 : 1.2,
        };
        emissionCount += 1;
      }
      dust.forEach((particle, index) => {
        const age = elapsed - particle.born;
        const dot = dustRefs.current[index];
        if (!dot) return;
        const opacity = age < 0 || age > 850 ? 0 : Math.sin(Math.PI * age / 850) * .72;
        dot.setAttribute('cx', particle.x);
        dot.setAttribute('cy', particle.y - Math.max(0, age) * .008);
        dot.setAttribute('r', particle.radius);
        dot.setAttribute('opacity', opacity);
      });

      if (elapsed > REVEAL_START_MS) {
        const reveal = Math.min(1, (elapsed - REVEAL_START_MS) / (DURATION_MS - REVEAL_START_MS));
        const easedReveal = smooth(reveal);
        overlayRef.current.style.clipPath = `inset(0 0 0 ${easedReveal * 100}%)`;
      }
      if (elapsed < DURATION_MS) frameId = window.requestAnimationFrame(animate);
      else onComplete();
    }

    frameId = window.requestAnimationFrame(animate);
    return () => { window.cancelAnimationFrame(frameId); window.removeEventListener('keydown', onKeyDown); };
  }, [onComplete, reducedMotion]);

  return (
    <div className="intro" ref={overlayRef} role="dialog" aria-modal="true" aria-label="Tink introduction">
      <button className="intro-skip" ref={skipRef} type="button" onClick={onComplete}>Skip intro <span aria-hidden="true">↗</span></button>
      <div className="intro-center">
        <svg ref={svgRef} className="intro-mark" viewBox="0 0 680 270" role="img" aria-label="Tink" xmlns="http://www.w3.org/2000/svg">
          {STROKES.filter((stroke) => !stroke.name.startsWith('i ')).map((stroke) => <path key={`guide-${stroke.name}`} className="intro-ink-guide" d={stroke.d} strokeWidth={stroke.width} />)}
          {STROKES.map((stroke, index) => <path key={stroke.name} ref={(node) => { strokeRefs.current[index] = node; }} className={`intro-ink ${stroke.name.startsWith('i ') ? 'intro-i-ink' : ''}`} d={stroke.d} strokeWidth={stroke.width} />)}
          <g className="intro-dust" aria-hidden="true">
            {Array.from({ length: DUST_COUNT }, (_, index) => <circle key={index} ref={(node) => { dustRefs.current[index] = node; }} />)}
          </g>
          <Fairy fairyRef={fairyRef} firstFrameRef={firstFrameRef} secondFrameRef={secondFrameRef} />
        </svg>
        <p className="intro-caption">A little clarity changes everything.</p>
      </div>
    </div>
  );
}
