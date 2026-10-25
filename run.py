"""Tink live pipeline: video -> initial distribution, audio -> Bayesian update -> posterior over intents,
semantic retrieval -> top interpretations, user confirmation -> text-to-speech.

Keys: 1-3 confirm that interpretation and speak it, 0 reject all (logged), N switch user, Q quit.
Run: python run.py --camera 0 --audio
     python run.py --video sample.mp4 --no-audio [--threads on]   (on: play the file in real time like a camera)
"""
import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from audio_text import AudioPipeline, BayesianFusion
from retrieval import PhraseDatabase, PhraseRanker
from tts import Speaker
from video import ThreadedVideoAnalyzer, VideoAnalyzer

ROOT = Path(__file__).resolve().parent
TEAL, WHITE, BLUE, GREEN = (90, 230, 190), (235, 235, 235), (130, 180, 250), (90, 220, 90)


def draw_panel(frame, visual, audio, result, top, notice, display_fps=0.0):
    h, w = frame.shape[:2]
    panel_w = 520
    canvas = np.zeros((max(h, 760), w + panel_w, 3), dtype=np.uint8)
    canvas[:h, :w] = frame
    x = w + 14
    y = [26]
    def line(msg, color=WHITE, scale=0.5, gap=22):
        cv2.putText(canvas, msg[:64], (x, y[0]), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y[0] += gap
    def header(msg):
        y[0] += 6
        line(msg, TEAL, 0.52)
    target_id = visual.get('target_id')
    for person in visual.get('people', []):
        a, b, c, d = person['bbox']
        is_user = person['id'] == target_id
        color = TEAL if is_user else (150, 150, 150)
        cv2.rectangle(canvas, (a, b), (c, d), color, 3 if is_user else 1)
        hands = person.get('hands', {})
        # per hand: L/R, 'up' if the wrist is above its shoulder, wrist speed in shoulder-widths per second
        tags = [f"{side[0].upper()}{' up' if h['raised'] else ''}"
                + (f" {h['speed']:.1f}" if h.get('speed') is not None else '')
                for side, h in sorted(hands.items())]
        label = f"#{person['id']}{' USER' if is_user else ''}  " + '  '.join(tags)
        cv2.putText(canvas, label, (a + 4, max(16, b - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    line('TINK - LIVE COMMUNICATION EVIDENCE', TEAL, 0.6, 26)
    people = visual.get('people', [])
    user = f"user #{target_id} (N = switch)" if target_id is not None else 'no user selected'
    line(f"People: {len(people)}   {user}")
    header('VIDEO EVIDENCE')
    for g in visual['gestures'][:2]:
        line(f"Hand: {g['label']} {g['score']*100:.0f}%")
    if not visual['gestures']:
        line('Hand: (none)')
    env = visual.get('envision') or {}
    if env.get('status') == 'ok':
        line(f"Envision P(Gesture): {env['probs']['Gesture']*100:.0f}%  ({env['label']})")
    else:
        line(f"Envision: {env.get('status', 'missing')}")
    rates = visual.get('rates')
    if rates:
        # Envision's velocity features were trained at 25-30 fps; below ~20 fps its score is less reliable.
        line(f"fps  display {display_fps:.0f} | track {rates['tracker']:.0f} | mediapipe {rates['mediapipe']:.0f}"
             f" | envision {rates['envision']:.0f}",
             WHITE if rates['envision'] >= 20 or not env.get('probs') else BLUE)
    head = visual.get('head') or {}
    if head.get('q'):
        q = head['q']
        line(f"Head: nod {q['nod']:.2f}  shake {q['shake']:.2f}   (pitch {head['pitch']:.0f}, yaw {head['yaw']:.0f})")
    else:
        line(f"Head: {head.get('status', 'missing')}")
    user = next((p for p in visual.get('people', []) if p['id'] == target_id), None)
    if user and user.get('signals'):
        line(f"Arm: hold {user['signals']['hold']:.2f}  wave {user['signals']['wave']:.2f}")
    face = sorted(visual['face_movements'].items(), key=lambda z: -z[1])[:2]
    line('Face (not emotion): ' + ', '.join(f'{k} {v*100:.0f}%' for k, v in face))
    header('AUDIO EVIDENCE')
    line(f"Speech: {audio.get('transcript', '')[:44] or '(no recent transcript)'}")
    voc = audio.get('vocalization') or {}
    if voc.get('status') == 'ok':
        line(f"Vocalization: {voc['label']} {voc['probs'][voc['label']]*100:.0f}%")
    else:
        line(f"Vocalization: {voc.get('status', 'missing')}   [{audio.get('status', '')}]")
    header('DISTRIBUTIONS (this moment)')
    (i0, p0), (q0, r0) = next(iter(result['initial_distribution'].items())), next(iter(result['posterior'].items()))
    line(f'Video {p0*100:4.1f}% {i0[:20]}  |  +Audio {r0*100:4.1f}% {q0[:20]}')
    header('BELIEF (builds while a signal is held)')
    for intent, p in list(result['belief'].items())[:3]:
        line(f'{p*100:5.1f}%  {intent}')
    header('TOP INTERPRETATIONS - press 1-3, 0 = none')
    for i, t in enumerate(top, 1):
        line(f"[{i}] {t['text']}  ({t['score']:.2f})", WHITE, 0.52, 24)
    y[0] += 4
    if notice and time.monotonic() - notice[1] < 4:
        line(notice[0], GREEN, 0.52)
    line('Distributions are hand-set placeholders, not calibrated', BLUE, 0.45)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--audio', action='store_true', help='enable microphone + Whisper + voc2vec')
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('--no-tracker', action='store_true',
                        help='no person tracking: single-person models see the whole frame')
    parser.add_argument('--yolo', '--no-yolo', action='store_true', help=argparse.SUPPRESS)  # old flags, no-ops
    parser.add_argument('--no-envision', action='store_true')
    parser.add_argument('--threads', choices=['auto', 'on', 'off'], default='auto',
                        help='analysis in background threads (auto: on for camera, off for --video files)')
    parser.add_argument('--no-tts', action='store_true')
    parser.add_argument('--whisper', default='base.en')
    parser.add_argument('--top', type=int, default=3)
    parser.add_argument('--save-every', type=float, default=1.0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--headless', action='store_true', help='no window; save panel_snapshot.png')
    args = parser.parse_args()

    fusion = BayesianFusion.from_json()
    belief_filter = fusion.make_filter()
    belief_user = None
    ranker = PhraseRanker(PhraseDatabase(), fusion.phrases)
    speaker = Speaker(enabled=not args.no_tts)
    audio = AudioPipeline(enabled=args.audio and not args.no_audio, whisper_size=args.whisper)
    visual = VideoAnalyzer(use_tracker=not args.no_tracker, use_envision=not args.no_envision)
    threaded = args.threads == 'on' or (args.threads == 'auto' and not args.video)
    # Threaded: the display runs at camera rate and analysis uses the newest frame when it is free.
    # Unthreaded (video files by default): every frame is analyzed in order, as fast as the CPU allows.
    worker = ThreadedVideoAnalyzer(visual) if threaded else None
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        (worker or visual).close()
        raise RuntimeError('Cannot open video/camera')
    audio.start()
    start = time.monotonic()
    last_save = 0.0
    frame_index = 0
    latest = None
    notice = None
    shown = deque(maxlen=30)
    print('Keys: 1-3 confirm + speak, 0 reject, N switch user, Q quit.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if args.video:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = int(pos_ms) if pos_ms > 0 else frame_index * 33
                if threaded:  # play the file in real time, like a camera
                    time.sleep(max(0.0, start + ts / 1000 - time.monotonic()))
            else:
                ts = int((time.monotonic() - start) * 1000)
            if worker:
                worker.submit(frame, ts)
                latest = worker.latest()
            else:
                # Every frame: Envision's velocity features assume consecutive frames at 25-30 fps.
                latest = visual.analyze(frame, ts)
            ap = audio.get()
            result = fusion.infer(latest, ap)
            if latest.get('target_id') != belief_user:   # a different person: start their belief fresh
                belief_filter.reset()
                belief_user = latest.get('target_id')
            result['belief'] = belief_filter.step(result['posterior'], ts / 1000)
            top = ranker.rank(result['belief'], k=args.top)
            shown.append(time.monotonic())
            display_fps = (len(shown) - 1) / max(shown[-1] - shown[0], 1e-6) if len(shown) > 1 else 0.0
            panel = draw_panel(frame, latest, ap, result, top, notice, display_fps)
            if args.headless:
                if frame_index % 25 == 0:
                    cv2.imwrite(str(ROOT / 'panel_snapshot.png'), panel)
                    rates = latest.get('rates') or {}
                    ppl = {p['id']: p.get('hands') for p in latest.get('people', [])}
                    print(f"frame {frame_index}: people={len(ppl)} ids={list(ppl)} user={latest.get('target_id')} "
                          f"hands={ppl.get(latest.get('target_id'))} envision={latest['envision']['status']} "
                          f"belief={list(result['belief'].items())[0]} say={top[0]['text']!r} "
                          f"fps display={display_fps:.1f} mediapipe={rates.get('mediapipe', 0):.1f} "
                          f"envision={rates.get('envision', 0):.1f}")
                if args.max_frames and frame_index >= args.max_frames:
                    break
                continue
            cv2.imshow('Tink', panel)
            if args.max_frames and frame_index >= args.max_frames:
                break
            if time.monotonic() - last_save >= args.save_every:
                packet = {'timestamp': datetime.now(timezone.utc).isoformat(),
                          'visual': latest, 'audio': ap,
                          'initial_distribution': result['initial_distribution'],
                          'posterior': result['posterior'], 'belief': result['belief'], 'evidence': result['evidence'],
                          'top_interpretations': top,
                          'score_note': 'hand-set placeholder likelihoods; not calibrated probabilities'}
                tmp = ROOT / 'multimodal_packet.tmp'
                tmp.write_text(json.dumps(packet, indent=2, default=str), encoding='utf-8')
                tmp.replace(ROOT / 'multimodal_packet.json')
                last_save = time.monotonic()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if ord('1') <= key < ord('1') + len(top):
                chosen = top[key - ord('1')]['text']
                speaker.say(chosen)
                ranker.log(result['belief'], top, chosen)
                notice = ('Spoke: ' + chosen, time.monotonic())
            elif key == ord('0'):
                ranker.log(result['belief'], top, None)
                notice = ('Rejected - use the board', time.monotonic())
            elif key in (ord('n'), ord('N')) and visual.tracker:
                picked = worker.select_next() if worker else visual.tracker.select_next()
                notice = (f'User is now #{picked}' if picked is not None else 'Nobody to select',
                          time.monotonic())
    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio.close()
        (worker or visual).close()
        speaker.close()


if __name__ == '__main__':
    main()
