"""Independently enroll and test nonverbal sounds; no video/fusion/TTS.

  python tools/audio_layer_lab.py devices
  python tools/audio_layer_lab.py enroll --labels hum ah --takes 3
  python tools/audio_layer_lab.py live --record
  python tools/audio_layer_lab.py wav path/to/clip.wav --headless
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_layer import (SR, NEGATIVES, EnergySegmenter, SampleCapture, Voc2VecEncoder,
                         EventProcessor, safe_name, save_profile, load_profile, rms_db,
                         annotate_capture_timing)

HOME = ROOT / '.scratch_audio_layer'
WINDOW = 'Tink audio layer / speaker unknown'


def private_folder(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path / '.gitignore').write_text('*\n', encoding='utf-8')
    return path


def profile_path(name):
    return HOME / 'profiles' / (safe_name(name) + '.npz')


def read_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, sr = sf.read(str(path), dtype='float32', always_2d=True)
    wave = wave.mean(axis=1)
    if sr != SR:
        divisor = math.gcd(sr, SR)
        wave = resample_poly(wave, SR // divisor, sr // divisor).astype(np.float32)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('WAV has no finite audio samples')
    return wave


def draw_panel(wave, caption, state, level, noise=None, result=None, stats=None, progress=None):
    """The visible values are raw measurements/template scores, not intents."""
    image = np.full((700, 1000, 3), (24, 28, 35), np.uint8)
    def text(value, y, color=(224, 231, 239), size=.55):
        cv2.putText(image, str(value)[:115], (24, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
    text('TINK / AUDIO LAYER', 34, size=.75)
    text(caption, 68)
    text('Speaker UNKNOWN: one microphone cannot identify who made a sound.', 99, (110, 180, 255), .52)
    cv2.rectangle(image, (24, 122), (976, 286), (49, 57, 68), 1)
    if len(wave):
        indices = np.linspace(0, len(wave) - 1, min(952, len(wave))).astype(int)
        values = np.clip(np.asarray(wave)[indices], -1, 1)
        points = np.c_[np.linspace(25, 975, len(values)), 204 - values * 78].astype(np.int32)
        cv2.polylines(image, [points], False, (115, 224, 170), 1, cv2.LINE_AA)
    text(f'Level {level:.1f} dBFS | noise floor {noise:.1f} dBFS' if noise is not None
         else f'Level {level:.1f} dBFS | calibrating quiet room', 320)
    fraction = np.clip((level + 80) / 80, 0, 1)
    cv2.rectangle(image, (24, 336), (976, 350), (53, 60, 72), -1)
    cv2.rectangle(image, (24, 336), (24 + int(952 * fraction), 350), (115, 224, 170), -1)
    text(state, 386, (110, 200, 255))
    if progress is not None:
        text(progress, 417)
    if result:
        age = result.get('display_age_s', result.get('age_s_at_publish'))
        if age is not None:
            text(f"Last event age {age:.2f}s | embedding {result.get('embedding_ms', 0):.0f} ms", 434, size=.46)
        text(f"Last event: {result.get('label', 'unknown')} | {result.get('reason', '')}", 455)
        scores = sorted(result.get('scores', {}).items(), key=lambda item: item[1], reverse=True)
        for i, (label, value) in enumerate(scores[:5]):
            y = 484 + 24 * i
            text(f'{label:20s} cosine {value:.3f}', y, size=.48)
            cv2.rectangle(image, (430, y - 10), (930, y - 2), (53, 60, 72), -1)
            cv2.rectangle(image, (430, y - 10), (430 + int(500 * np.clip((value + 1) / 2, 0, 1)), y - 2), (115, 224, 170), -1)
    if stats:
        text(stats, 638, size=.46)
    text('RAW / UNCALIBRATED | Q quit | enrollment: SPACE starts a take', 676, (150, 165, 185), .5)
    return image


def show(image, headless=False):
    if headless:
        return -1
    cv2.imshow(WINDOW, image)
    return cv2.waitKey(15) & 255


def trimmed_take(wave, noise, allow_quiet=False):
    count = len(wave) // 320
    levels = np.array([rms_db(wave[i * 320:(i + 1) * 320]) for i in range(count)])
    active = np.flatnonzero(levels > max(-80., noise) + 5)
    if not len(active):
        if allow_quiet:
            return wave
        raise ValueError('no clear sound above room noise; retry this take')
    start, end = max(0, int(active[0]) * 320 - 1280), min(len(wave), (int(active[-1]) + 1) * 320 + 1280)
    if end - start < int(.12 * SR):
        raise ValueError('sound is too brief; retry this take')
    return wave[start:end]


def input_stream(capture, device):
    import sounddevice as sd
    return sd.InputStream(samplerate=SR, channels=1, blocksize=320, dtype='float32',
                          device=device, callback=capture.callback)


def enroll(args, encoder, folder):
    labels = [safe_name(label) for label in args.labels]
    if len(set(labels)) != len(labels) or any(label in (*NEGATIVES, 'unknown') for label in labels):
        raise ValueError('use distinct personal labels; speech/other/unknown are reserved')
    if args.takes < 3 or not .5 <= args.take_seconds <= 4:
        raise ValueError('use at least 3 takes per label and 0.5-4 seconds per take')
    if profile_path(args.profile).exists() and not args.replace:
        raise ValueError('profile already exists; choose another --profile or explicitly use --replace')
    tasks = [(label, i + 1) for label in labels + list(NEGATIVES) for i in range(args.takes)]
    capture, history, quiet = SampleCapture(), deque(maxlen=100), []
    vectors, finished_labels = [], []
    current, future, task_index, message, noise = None, None, 0, 'Keep quiet for one second', None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='audio-enrollment')
    try:
        with input_stream(capture, args.device):
            while task_index < len(tasks):
                for start, wave, bad in capture.drain():
                    history.append(wave)
                    if noise is None:
                        quiet.append(rms_db(wave))
                        if len(quiet) >= 50:
                            noise = float(np.median(quiet))
                            message = 'Ready: press SPACE, then make the requested sound'
                    if current is not None:
                        expected = current['start'] + current['n']
                        if start + len(wave) <= expected:
                            continue  # a callback already in flight when SPACE set the start
                        if start < expected:
                            wave, start = wave[expected - start:], expected
                        if bad or start != expected:
                            current, message = None, 'Audio gap: retry this take with SPACE'
                            continue
                        remaining = int(args.take_seconds * SR) - current['n']
                        current['wave'].append(wave[:remaining])
                        current['n'] += min(remaining, len(wave))
                        if current['n'] >= int(args.take_seconds * SR):
                            label, take = tasks[task_index]
                            try:
                                sample = trimmed_take(np.concatenate(current['wave']), noise, label == 'other')
                            except ValueError as error:
                                current, message = None, str(error)
                                continue
                            if args.record:
                                import soundfile as sf
                                sf.write(str(folder / f'{label}-{take}.wav'), sample, SR)
                            future = executor.submit(encoder, sample)
                            current, message = None, 'Embedding take; microphone capture continues'
                if future is not None and future.done():
                    vectors.append(future.result())
                    finished_labels.append(tasks[task_index][0])
                    task_index += 1
                    future, message = None, 'Take saved. Rest, then press SPACE for the next take'
                    if task_index >= len(tasks):
                        break
                label, take = tasks[task_index]
                prompt = ('Say ordinary sentences: NEGATIVE speech example' if label == 'speech' else
                          'Room/background/cough/keyboard: NEGATIVE other example' if label == 'other' else
                          f'Make your {label} sound')
                wave = np.concatenate(history) if history else np.empty(0)
                state = f'RECORDING {current["n"] / SR:.1f}/{args.take_seconds:.1f}s' if current else message
                panel = draw_panel(wave, f'Enroll {args.profile}: {label}, take {take}/{args.takes} | {prompt}',
                                   state, rms_db(wave[-320:]), noise,
                                   progress=f'{task_index}/{len(tasks)} examples saved; includes required speech/other negatives')
                key = show(panel)
                if key in (ord('q'), ord('Q'), 27):
                    raise RuntimeError('enrollment cancelled; incomplete profile was not saved')
                if key in (32, 13) and noise is not None and current is None and future is None:
                    # Start after all blocks drained this iteration. Sample counter includes
                    # callback audio even if inference/enrollment was busy.
                    capture.drain()
                    with capture.lock:
                        start = capture.sample_count
                    current = {'start': start, 'n': 0, 'wave': []}
        save_profile(profile_path(args.profile), vectors, finished_labels, noise, args.min_score, args.min_margin,
                     revision=encoder.revision)
        print(f'Profile saved: {profile_path(args.profile)}\nTest: python tools/audio_layer_lab.py live --profile {args.profile}', flush=True)
    finally:
        executor.shutdown(wait=True)
        cv2.destroyAllWindows()


class InferenceWorker:
    def __init__(self, processor):
        self.processor, self.jobs, self.results = processor, queue.Queue(maxsize=4), queue.Queue()
        self.stop = threading.Event()
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, name='audio-event-inference', daemon=True)
        self.thread.start()

    def submit(self, segment):
        try:
            self.jobs.put_nowait(segment)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not self.stop.is_set():
            try:
                segment = self.jobs.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                self.results.put(self.processor.process(segment))
            except Exception as error:
                self.error = error
                self.stop.set()

    def drain(self):
        if self.error:
            raise RuntimeError('audio event inference failed') from self.error
        result = []
        while True:
            try:
                result.append(self.results.get_nowait())
            except queue.Empty:
                return result

    def close(self):
        self.stop.set()
        self.thread.join(30)
        if self.thread.is_alive():
            raise RuntimeError('inference did not stop within 30 seconds')
        self.dropped += self.jobs.qsize()
        while not self.jobs.empty():
            self.jobs.get_nowait()


def live(args, processor, folder):
    capture, detector = SampleCapture(), EnergySegmenter(noise_db=args.noise_db)
    worker, history = InferenceWorker(processor), deque(maxlen=100)
    result, count, record, written = None, 0, None, 0
    begin = time.perf_counter()
    events = (folder / 'events.jsonl').open('w', encoding='utf-8')
    if args.record:
        import soundfile as sf
        record = sf.SoundFile(str(folder / 'capture.wav'), mode='w', samplerate=SR, channels=1, subtype='FLOAT')
    def record_block(start, wave):
        nonlocal written
        if record is not None:
            if start > written:
                record.write(np.zeros(start - written, np.float32))
            record.write(wave)
            written = start + len(wave)
    def collect_results():
        nonlocal count, result
        for result in worker.drain():
            result = annotate_capture_timing(result, capture.sample_count, capture.clock_valid)
            events.write(json.dumps(result, allow_nan=False) + '\n')
            events.flush()
            count += 1
            when = f"{result['t_start']:.2f}-{result['t_end']:.2f}s" if capture.clock_valid else 'capture time unknown'
            print(f"{when} {result['label']} ({result['reason']}); speaker unknown", flush=True)
    try:
        with input_stream(capture, args.device):
            while True:
                for start, wave, bad in capture.drain():
                    history.append(wave)
                    record_block(start, wave)
                    for segment in detector.feed(start, wave, bad):
                        worker.submit(segment)
                collect_results()
                wave = np.concatenate(history) if history else np.empty(0)
                stats = (f'Clock {capture.sample_count / SR:.1f}s | events {count} | dropped capture blocks '
                         f'{capture.dropped_blocks} / samples {capture.dropped_samples} | device flags {capture.overflows} | inference drops {worker.dropped}')
                display_result = dict(result) if result is not None else None
                if display_result is not None:
                    display_result['display_age_s'] = (max(0., capture.sample_count / SR - result['t_end'])
                                                       if capture.clock_valid and result['t_end'] is not None else None)
                state = detector.state if capture.clock_valid else 'DEVICE LOSS: timing unknown; Q and restart the test'
                image = draw_panel(wave, f'Live profile {args.profile} | names are your sounds, not intentions',
                                   state, detector.level_db, detector.noise_db, display_result, stats)
                if args.save_preview:
                    cv2.imwrite(str(folder / 'audio_preview.png'), image)
                key = show(image, args.headless)
                if key in (ord('q'), ord('Q'), 27) or (args.seconds and time.perf_counter() - begin >= args.seconds):
                    break
                if args.headless:
                    time.sleep(.01)
        # The stream is stopped before draining and finalizing local recording.
        for start, wave, bad in capture.drain():
            record_block(start, wave)
        partial = detector.finish()
        if partial is not None:
            detector.discarded_segments += 1  # Q ends capture; never promote a truncated sound
        worker.close()
        collect_results()
        summary = {'events': count, 'captured_samples': capture.sample_count,
                   'captured_s': capture.sample_count / SR, 'dropped_blocks': capture.dropped_blocks,
                   'dropped_samples': capture.dropped_samples, 'device_status_flags': capture.overflows,
                   'gaps': detector.gaps, 'discarded_segments': detector.discarded_segments,
                   'capture_clock_valid': capture.clock_valid,
                   'inference_drops_or_cancelled': worker.dropped, 'person_id': None, 'calibrated': False}
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        try:
            worker.close()
        finally:
            events.close()
            if record:
                record.close()
            if not args.headless:
                cv2.destroyAllWindows()


def replay(args, processor, folder):
    wave = read_audio(args.path)
    detector = EnergySegmenter(noise_db=args.noise_db)
    count, last = 0, None
    with (folder / 'events.jsonl').open('w', encoding='utf-8') as events:
        for start in range(0, len(wave), 320):
            segments = detector.feed(start, wave[start:start + 320])
            if start + 320 >= len(wave):
                final = detector.finish()
                if final is not None:
                    segments.append(final)
            for segment in segments:
                last = processor.process(segment)
                events.write(json.dumps(last, allow_nan=False) + '\n')
                print(json.dumps(last), flush=True)
                count += 1
            image = draw_panel(wave[max(0, start - 2 * SR):start + 320], f'WAV replay: {args.path.name}',
                               detector.state, detector.level_db, detector.noise_db, last,
                               f'Source clock {(start + 320) / SR:.2f}s | events {count} | speaker unknown')
            if args.save_preview and (last is not None or start == 0):
                cv2.imwrite(str(folder / 'audio_preview.png'), image)
            if show(image, args.headless) in (ord('q'), ord('Q'), 27):
                break
    summary = {'events': count, 'samples': len(wave), 'source_s': len(wave) / SR,
               'gaps': detector.gaps, 'person_id': None, 'calibrated': False}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if not args.headless:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('devices')
    for name in ['enroll', 'live', 'wav']:
        p = sub.add_parser(name)
        p.add_argument('--profile', default='default')
        p.add_argument('--out', type=Path)
        p.add_argument('--record', action='store_true')
        p.add_argument('--device', type=int)
        p.add_argument('--min-score', type=float, default=.75 if name == 'enroll' else None)
        p.add_argument('--min-margin', type=float, default=.05 if name == 'enroll' else None)
        if name == 'enroll':
            p.add_argument('--labels', nargs='+', required=True)
            p.add_argument('--takes', type=int, default=3)
            p.add_argument('--take-seconds', type=float, default=2.)
            p.add_argument('--replace', action='store_true')
        else:
            p.add_argument('--noise-db', type=float, default=None, help='fixed initial room floor; otherwise quiet calibration for one second')
            p.add_argument('--headless', action='store_true')
            p.add_argument('--save-preview', action='store_true')
            p.add_argument('--without-profile', action='store_true', help='measure events/embeddings with unknown class; no identity attribution')
        if name == 'live':
            p.add_argument('--seconds', type=float, default=0.)
        if name == 'wav':
            p.add_argument('path', type=Path)
    args = ap.parse_args()
    if args.command == 'devices':
        import sounddevice as sd
        print(sd.query_devices())
        return
    safe_name(args.profile)
    if args.min_score is not None and not -1 <= args.min_score <= 1:
        ap.error('--min-score must be between -1 and 1')
    if args.min_margin is not None and not 0 <= args.min_margin <= 2:
        ap.error('--min-margin must be between 0 and 2')
    if args.command == 'live' and args.headless and args.seconds <= 0:
        ap.error('headless live capture needs --seconds')
    if args.command == 'live' and args.seconds < 0:
        ap.error('--seconds must be nonnegative')
    if args.command == 'wav' and (args.record or args.device is not None):
        ap.error('--record and --device are for microphone modes; WAV replay reads the supplied file')
    if args.command == 'wav' and not args.path.is_file():
        ap.error('WAV path does not exist')
    classifier = profile_meta = None
    if args.command != 'enroll' and not args.without_profile:
        if not profile_path(args.profile).is_file():
            ap.error('profile missing: enroll first with enroll --labels hum ah --takes 3, or use --without-profile')
        classifier, profile_meta = load_profile(profile_path(args.profile))
        if args.min_score is not None:
            classifier.min_score = args.min_score
        if args.min_margin is not None:
            classifier.min_margin = args.min_margin
    folder = private_folder(args.out or HOME / 'trials' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]))
    print(f'Output: {folder}\nLoading cached voc2vec layer 5; no network download. Speaker attribution stays UNKNOWN.', flush=True)
    encoder = Voc2VecEncoder()
    if profile_meta is not None and profile_meta['model_revision'] != encoder.revision:
        raise ValueError('profile was enrolled with a different model revision; reenroll')
    if args.command == 'enroll':
        enroll(args, encoder, folder)
    else:
        processor = EventProcessor(encoder, classifier)
        if args.command == 'live':
            live(args, processor, folder)
        else:
            replay(args, processor, folder)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print(f'Audio lab: {error}', file=sys.stderr)
        sys.exit(1)
