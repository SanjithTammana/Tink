"""Live integration lab: automatic tracking, motifs, fingerspelling, context and confirmed speech.

Legacy cloud modes require --gemini / --gemini-asr; --assessment explicitly uploads completed clips.
No speech without a confirmation key.
Saved replay and --headless never open the camera or microphone.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import time

import cv2
import numpy as np

from audio_layer import SR, EnergySegmenter, Voc2VecEncoder, rms_db, safe_name
from gemini_client import GeminiClient, GeminiError
from session_capture import SessionClock, SharedAudioCapture
from session_engine import SessionEngine
from tts import Speaker
from visual_session import VisualSession
from tools.audio_layer_lab import input_stream, read_audio
from tools.video_layer_lab import LiveCapture
from tools.video_overlay import render_tracking

ROOT = Path(__file__).resolve().parent
WINDOW = 'Tink / synchronized layer lab'


class Lane:
    """One active job and no waiting backlog; capture never waits for inference."""
    def __init__(self, name):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self.future, self.tag, self.dropped = None, None, 0

    def submit(self, fn, tag, *args):
        if self.future is not None:
            self.dropped += 1
            return False
        self.tag, self.future = tag, self.pool.submit(fn, *args)
        return True

    def take(self):
        if self.future is None or not self.future.done():
            return None
        future, tag = self.future, self.tag
        self.future, self.tag = None, None
        try:
            return tag, future.result(), None
        except Exception as error:
            return tag, None, error

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def interval_owner(history, start, end):
    """Require an owned role before the event and continuous observations through it."""
    before = [h for h in history if h[0] <= start]
    during = [h for h in history if start < h[0] <= end]
    after = [h for h in history if h[0] >= end]
    if not before or not after:
        return None
    points = [before[-1], *during, after[0]]
    if any(not owned for _, _, owned in points) or len({pid for _, pid, _ in points}) != 1:
        return None
    if any(b[0] - a[0] > 1. for a, b in zip(points, points[1:])):
        return None
    return points[0][1]


def local_transcriber():
    from huggingface_hub import try_to_load_from_cache
    from transcribe import Transcriber
    cached = try_to_load_from_cache('Systran/faster-whisper-base.en', 'config.json')
    if not isinstance(cached, str):
        raise RuntimeError('Local Whisper model is not cached; transcription disabled, no download')
    return Transcriber(str(Path(cached).parent)).load()


def panel(frame, obs, engine, visual, segmenter, lanes, caption, recording):
    image = render_tracking(frame, obs, event_count=len(engine.events),
                            neutral_ready=visual.neutral is not None, recording=recording)
    width = max(1280, image.shape[1])
    out = np.full((image.shape[0] + 360, width, 3), (22, 27, 34), np.uint8)
    out[:image.shape[0], :image.shape[1]] = image
    y = image.shape[0] + 24
    def line(text, color=(220, 230, 237)):
        nonlocal y
        cv2.putText(out, str(text)[:155], (16, y), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
        y += 25
    line(caption)
    line(f'Audio: {segmenter.state} | {segmenter.level_db:.1f} dBFS | speaker UNKNOWN unless --user-mic asserted')
    discovery = engine.last_discovery
    line(f"Motif: {discovery.get('motif_id', '--')} | {discovery.get('status', '--')} | RAW / UNCALIBRATED")
    belief = engine.packet(obs['t_capture'])['belief']
    line('Estimated belief: ' + ' | '.join(f"{i['label']}: {i['probability']:.2f}" for i in belief['items'][:4]))
    asl = obs.get('asl', {})
    line(f"ASL alphabet: {asl.get('best', '-')} ready={asl.get('ready', False)} | buffer: {engine.spelled}")
    turns = engine.context.snapshot(obs['t_capture'])['finalized_turns']
    line('Context: ' + (f"[{turns[-1]['speaker_id']}] {turns[-1]['text']}" if turns else 'waiting for speech'))
    for i, option in enumerate(engine.options):
        line(f"{i+1}. [{option['kind']}] {option['text']}", (130, 225, 180))
    line(engine.status, (120, 190, 255))
    line('1-4 confirm+speak | R options | 0 reject | U undo | F type | A letter | SPACE gap | D delete | S speak spelling')
    line('Y confirm role | N change person | C clear spelling | Q quit | drops ' + '/'.join(str(l.dropped) for l in lanes))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assessment', action='store_true', help='Record a bounded audio/video clip, infer partner context and return three Gemini options')
    p.add_argument('--auto-record-seconds', type=float, default=0., help='Assessment test: record once after a five-second preview countdown, then assess (1-30 seconds)')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--video', type=Path)
    p.add_argument('--wav', type=Path)
    p.add_argument('--audio-offset', type=float, default=0., help='Replay WAV start in video seconds; operator-supplied alignment')
    p.add_argument('--device', type=int)
    p.add_argument('--profile', default='default')
    p.add_argument('--record', action='store_true', help='Save private raw video/audio and evidence logs')
    p.add_argument('--no-audio', action='store_true')
    p.add_argument('--no-asr', action='store_true')
    p.add_argument('--no-asl', action='store_true')
    p.add_argument('--no-tts', action='store_true')
    p.add_argument('--user-mic', action='store_true', help='Operator asserts this dedicated input carries only communicator audio; unsafe for a shared room mic')
    p.add_argument('--gemini', action='store_true', help='Send compact text evidence to Google for options')
    p.add_argument('--gemini-asr', action='store_true', help='Send raw audio chunks to Google for uncertain chunk-local speaker captions')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--max-frames', type=int, default=0)
    args = p.parse_args(argv)
    if args.assessment:
        if args.video or args.wav or args.headless or args.no_audio or (args.auto_record_seconds and not 1 <= args.auto_record_seconds <= 30):
            p.error('--assessment uses live camera/mic; auto recording must be 1-30 seconds')
        from assessment_session import run
        return run(args)
    if args.headless and not args.video:
        p.error('--headless requires saved --video; it never opens live sensors')
    if args.wav and not args.video:
        p.error('--wav replay requires --video')
    if args.no_asr and args.gemini_asr:
        p.error('--no-asr and --gemini-asr conflict')
    if args.audio_offset < 0:
        p.error('--audio-offset must be nonnegative')
    safe_name(args.profile)
    clock = SessionClock()
    home = ROOT / '.scratch_session' / args.profile
    home.mkdir(parents=True, exist_ok=True)
    (home / '.gitignore').write_text('*\n', encoding='utf-8')
    folder = home / clock.session_id
    folder.mkdir()
    client = GeminiClient(ROOT) if args.gemini or args.gemini_asr else None
    if client and not client.configured:
        p.error('Add GEMINI_API_KEY in tink-alt/.env before enabling cloud modes')
    from motif_learning import MotifLearner
    with ExitStack() as stack:
        speaker = Speaker(enabled=not (args.no_tts or args.headless))
        stack.callback(speaker.close)
        engine = SessionEngine(clock.session_id, MotifLearner(path=home / 'motifs.json'), speaker, ROOT)
        stack.callback(engine.learner.save)
        visual = VisualSession(clock.session_id, home / 'person.json', asl=not args.no_asl)
        stack.callback(visual.close)
        lanes = [Lane('audio-features'), Lane('transcription'), Lane('gemini-options')]
        for lane in lanes:
            stack.callback(lane.close)
        embedding_lane, asr_lane, suggestion_lane = lanes
        audio_enabled = bool(args.wav or (not args.video and not args.no_audio))
        encoder = Voc2VecEncoder() if audio_enabled else None
        asr, caption = None, 'Local only: raw scores; confirmation required'
        if audio_enabled and not args.no_asr:
            if args.gemini_asr:
                asr = client.transcribe_audio
            else:
                try:
                    asr = local_transcriber().transcribe_turns
                except (RuntimeError, OSError):
                    caption = 'Local ASR unavailable: cached base.en model required'
        if args.gemini or args.gemini_asr:
            caption = 'CLOUD ENABLED: ' + ('text options ' if args.gemini else '') + ('raw audio ASR' if args.gemini_asr else '')
        capture = SharedAudioCapture(clock)
        segmenter, ownership = EnergySegmenter(), deque(maxlen=600)
        video = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        stack.callback(video.release)
        if not video.isOpened():
            raise RuntimeError('Cannot open requested video source')
        fps = video.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.
        live = None
        if not args.video:
            live = LiveCapture(video, folder if args.record else None, time_source=clock.now).start()
            stack.callback(live.close)
            if audio_enabled:
                stack.enter_context(input_stream(capture, args.device))
        replay = read_audio(args.wav) if args.wav else None
        replay_n, frame_index, processed = 0, -1, 0
        asr_parts, asr_n, asr_start, asr_expected = [], 0, None, None
        log = stack.enter_context((folder / 'evidence.jsonl').open('w', encoding='utf-8')) if args.record else None
        suggestion_log = stack.enter_context((folder / 'suggestions.jsonl').open('w', encoding='utf-8')) if args.record else None
        raw_audio = None
        if args.record and audio_enabled:
            import soundfile as sf
            raw_audio = stack.enter_context(sf.SoundFile(str(folder / 'capture.wav'), mode='w', samplerate=SR, channels=1, subtype='PCM_16'))
        audio_times = stack.enter_context((folder / 'capture_audio_times.jsonl').open('w', encoding='utf-8')) if raw_audio else None
        manifest = {'schema': 1, 'session_id': clock.session_id, 'video_source': 'replay' if args.video else 'camera',
                    'video_time_basis': 'source_fps' if args.video else 'host_read_shared_monotonic',
                    'replay_audio_offset': args.audio_offset if args.wav else None,
                    'cloud_text': args.gemini, 'cloud_audio': args.gemini_asr, 'user_mic_asserted': args.user_mic}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        typed, typed_ids, typed_person = None, [], None
        role_since, previous_capture = 0., None
        last_suggest = float('-inf')
        suggestion_failed, requested_version = False, None
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1280, 900)
        print('Tink session ready. Private output:', folder)
        while True:
            if live:
                item = live.read_after(frame_index)
                if item is None:
                    break
                frame_index, now, frame, _label = item
            else:
                ok, frame = video.read()
                if not ok:
                    break
                frame_index += 1
                now = frame_index / fps
            obs, events = visual.observe(frame, now)
            obs['session_mode'] = 'integrated'
            obs['tracking_fps'] = 1 / (now - previous_capture) if previous_capture is not None and now > previous_capture else None
            obs.get('channels', {}).get('pose', {})['observed_hz'] = obs['tracking_fps']
            previous_capture = now
            old_person = engine.person_id
            engine.set_role(visual.target.logical_id, obs['ownership_valid'])
            if old_person is not None and old_person != engine.person_id:
                asr_parts, asr_n, asr_start, asr_expected = [], 0, None, None
                typed = None
                role_since = now
            ownership.append((now, engine.person_id, engine.owned))
            for event in events:
                engine.observe(event)
                if log:
                    log.write(json.dumps(event) + '\n')
            if replay is not None:
                end = min(len(replay), max(0, int((now - args.audio_offset) * SR)))
                blocks = [(i, replay[i:min(i+320, end)], False) for i in range(replay_n, end, 320)]
                replay_n = end
                capture.anchor_s = args.audio_offset
            else:
                blocks = capture.drain() if audio_enabled else []
            engine.clock_valid = capture.clock_valid
            for start, wave, bad in blocks:
                if raw_audio is not None:
                    raw_audio.write(wave)
                t0, t1 = capture.at(start), capture.at(start + len(wave))
                if audio_times:
                    audio_times.write(json.dumps({'sample_start': start, 'samples': len(wave), 't_start': t0,
                        't_end': t1, 'clock_valid': capture.clock_valid, 'device_status': bad}) + '\n')
                contaminated = t0 is not None and speaker.overlaps(t0, t1, clock.origin)
                if t0 is None or contaminated or bad:
                    segmenter.reset_gap()
                    asr_parts, asr_n, asr_start, asr_expected = [], 0, None, None
                    continue
                for event in segmenter.feed(start, wave):
                    if not event.complete:
                        continue
                    begin, end = capture.at(event.start), capture.at(event.end)
                    owner = interval_owner(ownership, begin, end) if args.user_mic else None
                    meta = {'event_id': f'{clock.session_id}-audio-{event.start}', 'session_id': clock.session_id,
                        'modality': 'audio', 't_start': begin, 't_end': end, 'person_id': owner,
                        'ownership_valid': owner is not None, 'clock_valid': capture.clock_valid,
                        'quality': {'source': 'voc2vec_layer5', 'speaker_source': 'dedicated_input_operator_assertion' if owner else 'unknown',
                                    'timestamp_basis': capture.timestamp_basis, 'overlap': False, 'tts_echo': False}, 'context_id': engine.context.revision}
                    embedding_lane.submit(encoder, meta, event.wave)
                if asr and t0 >= role_since:
                    if asr_expected is not None and start != asr_expected:
                        asr_parts, asr_n, asr_start = [], 0, None
                    asr_start = t0 if asr_start is None else asr_start
                    asr_parts.append(wave)
                    asr_n += len(wave)
                    asr_expected = start + len(wave)
                    if asr_n >= 6 * SR:
                        speech = np.concatenate(asr_parts)
                        if rms_db(speech) > -55 and not speaker.overlaps(asr_start, t1, clock.origin):
                            asr_lane.submit(asr, (asr_start, t1, engine.person_id), speech, asr_start)
                        asr_parts, asr_n, asr_start = [], 0, None
            result = embedding_lane.take()
            if result:
                event, vector, error = result
                if error:
                    engine.status = 'Local audio embedding failed; event discarded'
                elif capture.clock_valid and not speaker.overlaps(event['t_start'], event['t_end'], clock.origin):
                    event['features'] = np.asarray(vector).tolist()
                    engine.observe(event)
                    if log:
                        log.write(json.dumps(event) + '\n')
            result = asr_lane.take()
            if result:
                window, turns, error = result
                if error:
                    engine.status = 'Transcription unavailable; no context added'
                elif capture.clock_valid and window[2] == engine.person_id and not speaker.overlaps(window[0], window[1], clock.origin):
                    for turn in turns:
                        engine.context.add_turn(turn)
            result = suggestion_lane.take()
            if result:
                (packet, version), options, error = result
                if error:
                    suggestion_failed = True
                    engine.status = 'Gemini request failed or output invalid; R retries, no automatic retry'
                else:
                    accepted = engine.present(options, packet, version, now_s=now)
                    if suggestion_log:
                        suggestion_log.write(json.dumps({'at_s': now, 'source': 'gemini', 'accepted': accepted,
                            'packet': packet, 'result': options}) + '\n')
                    if not engine.options:
                        suggestion_failed = True
            # Freeze a displayed choice set. New context/evidence never changes a selection under the user.
            newly_eligible = set(engine.packet(now)['event_ids']) - set((engine.option_packet or {}).get('event_ids', []))
            if engine.options and all(o['kind'] == 'clarify' for o in engine.options) and newly_eligible:
                engine.options, engine.option_packet = [], None
            if not engine.options and not suggestion_failed and suggestion_lane.future is None and now - last_suggest >= 2:
                packet, version = engine.packet(now), engine.version()
                if args.gemini and version != requested_version:
                    suggestion_lane.submit(client.generate, (packet, version), packet)
                    requested_version = version
                elif not args.gemini:
                    engine.suggest_local(now)
                    if suggestion_log:
                        suggestion_log.write(json.dumps({'at_s': now, 'source': 'local_candidates', 'packet': packet,
                            'options': engine.options}) + '\n')
                last_suggest = now
            if args.headless:
                key = -1
            else:
                image = panel(frame, obs, engine, visual, segmenter, lanes, caption, args.record)
                if typed is not None:
                    cv2.putText(image, 'TYPE + ENTER confirms/speaks: ' + typed, (16, image.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 220, 100), 1)
                cv2.imshow(WINDOW, image)
                key = cv2.waitKey(1) & 255
            if typed is not None:
                if key in (10, 13):
                    eligible = typed_ids if engine.owned and typed_person == engine.person_id else []
                    engine.confirm_text(typed, eligible, now)
                    typed = None
                elif key == 27:
                    typed = None
                elif key in (8, 127):
                    typed = typed[:-1]
                elif 32 <= key <= 126:
                    typed = (typed + chr(key))[:160]
            elif key in (ord('q'), 27):
                break
            elif ord('1') <= key <= ord('4'):
                engine.confirm(key - ord('1'), now)
            elif key == ord('0'):
                engine.reject(now)
            elif key == ord('r'):
                engine.options, engine.option_packet = [], None
                last_suggest = float('-inf')
                suggestion_failed, requested_version = False, None
            elif key == ord('u'):
                engine.undo()
            elif key == ord('f'):
                typed = ''
                typed_ids, typed_person = engine.packet(now)['event_ids'], engine.person_id
            elif key == ord('y'):
                visual.target.confirm_role()
            elif key == ord('n'):
                visual.target.select_next(obs['people'])
            elif key == ord('a') and engine.owned and obs['asl'].get('ready'):
                engine.spelled += obs['asl']['best']
                engine.options, engine.option_packet = [], None
            elif key == 32:
                engine.spelled += ' '
            elif key in (ord('d'), 8):
                engine.spelled = engine.spelled[:-1]
            elif key == ord('c'):
                engine.spelled = ''
            elif key == ord('s') and engine.spelled.strip():
                engine.confirm_text(engine.spelled, [], now)
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
        manifest.update(processed_frames=processed, captured_frames=live.captured if live else processed,
                        audio_clock_valid=capture.clock_valid, audio_timestamp_basis=capture.timestamp_basis,
                        dropped_jobs=[l.dropped for l in lanes], events=len(engine.events), asl=visual.asl_status)
        manifest.update(audio_queue_dropped_blocks=capture.dropped_blocks, audio_device_status_flags=capture.overflows)
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        engine.learner.save()
        cv2.destroyAllWindows()
        print(json.dumps(manifest))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped; local state preserved.')
    except (GeminiError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
