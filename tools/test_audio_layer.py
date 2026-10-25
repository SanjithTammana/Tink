"""Audio contracts with synthetic samples/fake encoder; never opens a microphone."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_layer import (SR, AudioSegment, EnergySegmenter, SampleCapture, TemplateClassifier,
                         EventProcessor, safe_name, save_profile, load_profile, annotate_capture_timing)


def tone(duration, amplitude=.03):
    return (amplitude * np.sin(2 * np.pi * 190 * np.arange(round(duration * SR)) / SR)).astype(np.float32)


def segmented(wave, **kwargs):
    detector, output = EnergySegmenter(**kwargs), []
    # Deliberately unaligned input block sizes exercise the sample clock/remainder.
    for start in range(0, len(wave), 517):
        output += detector.feed(start, wave[start:start + 517])
    return detector, output


class AudioContracts(unittest.TestCase):
    def test_brief_sound_survives_a_four_second_window(self):
        wave = np.r_[tone(1., .0001), tone(.16), tone(2.84, .0001)]
        self.assertLess(float(np.sqrt(np.mean(wave * wave))), .01)
        detector, events = segmented(wave)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertTrue(event.complete)
        self.assertLessEqual(event.start / SR, 1.)
        self.assertGreaterEqual(event.end / SR, 1.16)
        self.assertLess(len(event.wave) / SR, .5)
        self.assertIsNone(detector.finish())

    def test_silence_does_not_create_an_event(self):
        _, events = segmented(tone(4., .0001))
        self.assertEqual(events, [])

    def test_two_separated_sounds_emit_twice_not_each_frame(self):
        wave = np.r_[tone(1., .0001), tone(.3), tone(.6, .0001), tone(.3), tone(.6, .0001)]
        _, events = segmented(wave)
        self.assertEqual(len(events), 2)
        self.assertLess(events[0].end, events[1].start)

    def test_short_pauses_merge_into_one_event(self):
        wave = np.r_[tone(1., .0001), tone(.2), tone(.1, .0001), tone(.2), tone(.6, .0001)]
        _, events = segmented(wave)
        self.assertEqual(len(events), 1)

    def test_long_sound_is_not_recounted_at_every_maximum_window(self):
        wave = np.r_[tone(1., .0001), tone(3.), tone(.6, .0001), tone(.3), tone(.6, .0001)]
        _, events = segmented(wave, max_s=1.)
        self.assertEqual(len(events), 2)
        self.assertFalse(events[0].complete)
        self.assertEqual(events[0].reason, 'max_duration')
        self.assertTrue(events[1].complete)

    def test_capture_gap_discards_partial_event(self):
        detector = EnergySegmenter(noise_db=-70.)
        detector.feed(0, tone(.1))
        detector.feed(SR, tone(.5, .0001))
        self.assertEqual(detector.gaps, 1)
        self.assertEqual(detector.discarded_segments, 1)
        self.assertIsNone(detector.finish())

    def test_callback_clock_advances_even_when_queue_is_full(self):
        capture = SampleCapture(max_blocks=2)
        for _ in range(4):
            capture.callback(np.zeros((320, 1), np.float32), 320, None, False)
        self.assertEqual(capture.sample_count, 1280)
        self.assertEqual(capture.dropped_blocks, 2)
        self.assertEqual(capture.dropped_samples, 640)
        self.assertEqual([start for start, _, _ in capture.drain()], [640, 960])

    def test_device_status_is_explicit_discontinuity(self):
        capture = SampleCapture()
        capture.callback(np.zeros((320, 1), np.float32), 320, None, True)
        self.assertEqual(capture.overflows, 1)
        self.assertFalse(capture.clock_valid)
        start, wave, bad = capture.drain()[0]
        detector = EnergySegmenter(noise_db=-70.)
        detector.feed(start, wave, bad)
        self.assertEqual(detector.gaps, 1)

    def test_device_loss_cannot_publish_a_valid_time_or_accepted_match(self):
        event = {'t_start': 1., 't_end': 2., 'accepted': True, 'label': 'hum'}
        valid = annotate_capture_timing(event, 3 * SR, True)
        self.assertEqual(valid['age_s_at_publish'], 1.)
        invalid = annotate_capture_timing(event, 3 * SR, False)
        self.assertIsNone(invalid['t_start'])
        self.assertIsNone(invalid['t_end'])
        self.assertFalse(invalid['accepted'])
        self.assertEqual(invalid['sample_t_start'], 1.)
        self.assertEqual(invalid['reason'], 'device_overflow_clock_unknown')
        self.assertTrue(event['accepted'])

    @staticmethod
    def examples():
        vectors = np.repeat(np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], np.float32), 3, axis=0)
        labels = np.repeat(['hum', 'ah', 'speech', 'other'], 3)
        return vectors, labels

    def test_personal_match_and_negative_examples(self):
        model = TemplateClassifier(*self.examples())
        self.assertEqual(model.predict([1, 0, 0])['label'], 'hum')
        self.assertTrue(model.predict([1, 0, 0])['accepted'])
        negative = model.predict([-1, 0, 0])
        self.assertEqual(negative['label'], 'speech')
        self.assertFalse(negative['accepted'])

    def test_unknown_and_ambiguous_are_rejected(self):
        model = TemplateClassifier(*self.examples(), min_score=.65)
        self.assertEqual(model.predict([0, 0, 1])['label'], 'unknown')
        self.assertEqual(model.predict([1, 1, 0])['reason'], 'ambiguous')
        self.assertEqual(model.predict([0, 0, 0])['reason'], 'invalid_embedding')

    def test_profile_requires_negatives_and_validates_encoder_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'profile.npz'
            save_profile(path, *self.examples(), noise_db=-60., revision='synthetic-revision')
            model, meta = load_profile(path, expected_dim=3, revision='synthetic-revision')
            self.assertEqual(model.predict([1, 0, 0])['label'], 'hum')
            self.assertEqual(meta['layer'], 5)
            with self.assertRaises(ValueError):
                load_profile(path, expected_dim=3, revision='different-revision')
            with self.assertRaises(ValueError):
                load_profile(path)  # actual encoder expects 768 dimensions
            with np.load(path, allow_pickle=False) as data:
                vectors, labels = data['vectors'].copy(), data['labels'].copy()
                meta = json.loads(str(data['metadata'].item()))
            meta['layer'] = 12
            np.savez(path, vectors=vectors, labels=labels, metadata=json.dumps(meta))
            with self.assertRaises(ValueError):
                load_profile(path, expected_dim=3)
            with self.assertRaises(ValueError):
                save_profile(path, [[1., 0., 0.]] * 3, ['hum'] * 3, -60., revision='synthetic-revision')

    def test_events_are_unique_sample_clocked_and_never_person_attributed(self):
        model = TemplateClassifier(*self.examples())
        processor = EventProcessor(lambda wave: np.array([1., 0., 0.]), model)
        segment = AudioSegment(SR, 2 * SR, tone(1.))
        first, second = processor.process(segment), processor.process(segment)
        self.assertEqual(first['t_start'], 1.)
        self.assertEqual(first['t_end'], 2.)
        self.assertNotEqual(first['event_id'], second['event_id'])
        self.assertIsNone(first['person_id'])
        self.assertFalse(first['calibrated'])
        partial = processor.process(AudioSegment(0, SR, tone(1.), False, 'max_duration'))
        self.assertFalse(partial['accepted'])

    def test_unenrolled_events_are_unknown(self):
        processor = EventProcessor(lambda wave: np.ones(3))
        result = processor.process(AudioSegment(0, SR, tone(1.)))
        self.assertEqual(result['reason'], 'unenrolled')
        self.assertFalse(result['accepted'])

    def test_names_and_thresholds_are_bounded(self):
        for name in ['../someone', 'a/b', '', 'x' * 40]:
            with self.assertRaises(ValueError):
                safe_name(name)
        with self.assertRaises(ValueError):
            TemplateClassifier(*self.examples(), min_margin=-1)


if __name__ == '__main__':
    unittest.main()
