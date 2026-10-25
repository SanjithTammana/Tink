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
from meaning import NONE, MeaningComposer
from retrieval import PhraseDatabase, PhraseRanker
from signs import NO_SIGN, SignLibrary, SignSpotter, load_meanings
from tts import Speaker
from video import ThreadedVideoAnalyzer, TrackerProcess, VideoAnalyzer

ROOT = Path(__file__).resolve().parent
TEAL, WHITE, BLUE, GREEN, ORANGE = (90, 230, 190), (235, 235, 235), (130, 180, 250), (90, 220, 90), (60, 160, 255)


def draw_panel(frame, visual, audio, result, top, notice, display_fps=0.0):
    h, w = frame.shape[:2]
    panel_w = 520
    canvas = np.zeros((max(h, 820), w + panel_w, 3), dtype=np.uint8)
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
    for o in visual.get('objects', []):
        a, b, c, d = o['bbox']
        cv2.rectangle(canvas, (a, b), (c, d), BLUE, 1)
        cv2.putText(canvas, o['label'], (a + 3, b + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE, 1, cv2.LINE_AA)
    for person in visual.get('people', []):
        pointing = person.get('pointing') or {}
        if person['id'] == target_id and pointing.get('score', 0) >= 0.3:
            o = np.array(pointing['origin'])
            cv2.arrowedLine(canvas, tuple(o.astype(int)), tuple((o + 160 * np.array(pointing['dir'])).astype(int)),
                            GREEN, 2, tipLength=0.15)
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
    # channel health: which signals are actually arriving right now
    usr = next((p for p in people if p['id'] == target_id), None)
    ok = lambda b: 'ok' if b else '--'
    voc_status = (audio.get('vocalization') or {}).get('status', '')
    line(f"Channels  face {ok((visual.get('head') or {}).get('status') in ('ok', 'warming up'))}"
         f" | hands {ok(bool(visual.get('gestures')))} | body {ok(bool(usr and usr.get('hands')))}"
         f" | envision {ok((visual.get('envision') or {}).get('status') == 'ok')}"
         f" | mic {ok(audio.get('status') not in (None, 'disabled') and 'error' not in str(audio.get('status')))}",
         WHITE, 0.45, 20)
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
    sg = visual.get('sign') or {}
    if sg.get('probs'):
        lead = sorted(sg['probs'].items(), key=lambda z: -z[1])[:2]
        line('Sign: ' + ', '.join(f'{k} {v*100:.0f}%' for k, v in lead))
    else:
        line(f"Sign: {sg.get('status', 'off')}")
    face = sorted(visual['face_movements'].items(), key=lambda z: -z[1])[:2]
    line('Face (not emotion): ' + ', '.join(f'{k} {v*100:.0f}%' for k, v in face))
    header('AUDIO EVIDENCE')
    history = result.get('history') or []
    for who, text in history[-2:]:
        line(f"{'Partner' if who == 'partner' else 'You'}: {text[:50]}", GREEN if who == 'you' else WHITE)
    if not history:
        line("Partner: (nothing heard yet)")
    voc = audio.get('vocalization') or {}
    if voc.get('status') == 'ok':
        line(f"Vocalization: {voc['label']} {voc['probs'][voc['label']]*100:.0f}%")
    else:
        line(f"Vocalization: {voc.get('status', 'missing')}   [{audio.get('status', '')}]")
    header('DISTRIBUTIONS (this moment)')
    (i0, p0), (q0, r0) = next(iter(result['initial_distribution'].items())), next(iter(result['posterior'].items()))
    line(f'Video {p0*100:4.1f}% {i0[:20]}  |  +Audio {r0*100:4.1f}% {q0[:20]}')
    header('BELIEF (builds while a signal is held)')
    for intent, p in list(result['belief'].items())[:2]:
        line(f'{p*100:5.1f}%  {intent}')
    meaning = result.get('meaning') or {}
    refs = [(r, p) for r, p in (meaning.get('referents') or {}).items() if r != NONE][:2]
    ctx = (meaning.get('context') or {}).get('type')
    line('About: ' + (', '.join(f'{r} {p*100:.0f}%' for r, p in refs) or '-')
         + (f'   | partner asked: {ctx}' if ctx else ''))
    clarify = meaning.get('clarify')
    if clarify:
        line(clarify['prompt'][:64], ORANGE, 0.48)
        if len(clarify['prompt']) > 64:
            line(clarify['prompt'][64:128], ORANGE, 0.48)
    header('TOP INTERPRETATIONS - press 1-3, 0 = none')
    if meaning.get('idle'):
        line('(no signal yet - suggestions are low confidence)', BLUE, 0.45)
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
    parser.add_argument('--no-objects', action='store_true', help='no object detection (pointing targets)')
    parser.add_argument('--no-signs', action='store_true', help='ignore enrolled keyword signs')
    parser.add_argument('--user', default='default', help='whose enrolled signs to load (data/signs/<user>)')
    parser.add_argument('--no-conversation', action='store_true',
                        help='no dialogue-corpus context model (use the built-in question rules)')
    parser.add_argument('--mapping', choices=['config', 'provisional'], default='config',
                        help='provisional: overlay config/mapping_provisional.json (conventions, for testing)')
    parser.add_argument('--threads', choices=['auto', 'on', 'off'], default='auto',
                        help='analysis in background threads (auto: on for camera, off for --video files)')
    parser.add_argument('--no-tts', action='store_true')
    parser.add_argument('--whisper', default='base.en')
    parser.add_argument('--top', type=int, default=3)
    parser.add_argument('--save-every', type=float, default=1.0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--headless', action='store_true', help='no window; save panel_snapshot.png')
    args = parser.parse_args()

    threaded = args.threads == 'on' or (args.threads == 'auto' and not args.video)
    # start the tracking process first: its model load overlaps everything loaded below
    early_tracker = TrackerProcess(use_objects=not args.no_objects) if threaded and not args.no_tracker else None
    fusion = BayesianFusion.from_json()
    if args.mapping == 'provisional':
        prov = json.loads((ROOT / 'config' / 'mapping_provisional.json').read_text(encoding='utf-8'))
        for src, spec in prov['sources'].items():
            fusion.config['sources'][src].update({'labels': spec['labels'], 'table': spec['table']})
    meanings, spotter = load_meanings(), None
    if not args.no_signs:
        library = SignLibrary(args.user)
        if library.signs:
            fusion.add_sign_source(meanings, library.signs)
            spotter = SignSpotter(library)
            print(f'Enrolled signs: {", ".join(library.signs)}')
        else:
            print('No enrolled signs: run `python signs.py enroll` to teach Tink your signs.')
    belief_filter = fusion.make_filter()
    belief_user = None
    ranker = PhraseRanker(PhraseDatabase(), fusion.phrases)
    conversation = None
    if not args.no_conversation:
        from conversation import INDEX, ConversationModel
        if INDEX.exists():
            conversation = ConversationModel([i for i in fusion.phrases if i != '(no message)'])
        else:
            print('No conversation index: run `python conversation.py fetch` then `build` to enable it.')
    composer = MeaningComposer(ranker, conversation=conversation)
    speaker = Speaker(enabled=not args.no_tts)
    audio = AudioPipeline(enabled=args.audio and not args.no_audio, whisper_size=args.whisper)
    # Threaded: the display runs at camera rate; tracking and objects run in a child process, hands, face and
    # Envision in threads, each on the newest frame. Unthreaded (video files by default): every frame is
    # analyzed in order in this process, as fast as the CPU allows.
    visual = VideoAnalyzer(use_tracker=not args.no_tracker and not threaded, use_envision=not args.no_envision,
                           use_objects=not args.no_objects and not threaded)
    worker = (ThreadedVideoAnalyzer(visual, track=not args.no_tracker, objects=not args.no_objects,
                                    tracker=early_tracker) if threaded else None)
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
    history, last_heard = [], None     # the conversation so far: ('partner' | 'you', text)
    last_hands_ts, sign = None, {'status': 'not enrolled' if spotter is None else 'starting', 'probs': None}
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
            if spotter and latest.get('hand_landmarks') is not None and latest['timestamp_ms'] != last_hands_ts:
                last_hands_ts = latest['timestamp_ms']
                sign = spotter.update(ts / 1000, latest['hand_landmarks'])
            latest['sign'] = sign
            ap = audio.get()
            result = fusion.infer(latest, ap)
            if latest.get('target_id') != belief_user:   # a different person: start their belief fresh
                belief_filter.reset()
                composer.referents.reset()
                if spotter:
                    spotter.reset()
                belief_user = latest.get('target_id')
            result['belief'] = belief_filter.step(result['posterior'], ts / 1000)
            user = next((p for p in latest.get('people', []) if p['id'] == latest.get('target_id')), {})
            fresh = ap.get('timestamp') is not None and time.time() - ap['timestamp'] < 10
            result['meaning'] = composer.compose(
                result['belief'], {'objects': latest.get('objects', []), 'pointing': user.get('pointing'),
                                   'referent_hints': {meanings['signs'][s]['referent']: p
                                                      for s, p in (sign.get('probs') or {}).items()
                                                      if s != NO_SIGN and meanings['signs'].get(s, {}).get('referent')}},
                ap.get('transcript', '') if fresh else '', ts / 1000, k=args.top, conflicts=result['conflicts'])
            if fresh and ap.get('transcript') and ap['timestamp'] != last_heard:   # a new partner turn
                history.append(('partner', ap['transcript']))
                last_heard = ap['timestamp']
            result['history'] = history
            top = result['meaning']['candidates']
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
                          f"envision={rates.get('envision', 0):.1f} track={rates.get('tracker', 0):.1f} "
                          f"objects={rates.get('objects', 0):.1f} face={rates.get('face', 0):.1f}")
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
                history.append(('you', chosen))
                ranker.log(result['belief'], top, chosen)
                notice = ('Spoke: ' + chosen, time.monotonic())
            elif key == ord('0'):
                ranker.log(result['belief'], top, None)
                notice = ('Rejected - use the board', time.monotonic())
            elif key in (ord('n'), ord('N')) and (worker or visual.tracker):
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
