"""Synthetic contract tests. No cameras, microphones, models or API calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motif_learning import MotifLearner, UNKNOWN


def event(identity, at=0., features=(1., 0., 0.), modality='audio', **changes):
    value = dict(event_id=identity, session_id='s1', modality=modality,
                 t_start=at, t_end=at + .5, person_id='communicator-1',
                 ownership_valid=True, clock_valid=True, features=list(features), quality={},
                 feature_schema='synthetic-v1')
    value.update(changes)
    return value


def learn_messages(learner):
    for prefix, vector, text, base in [('a', (1., 0., 0.), 'More water please', 0.),
                                       ('b', (0., 1., 0.), 'Please stop', 8.)]:
        for i in range(3):
            item = event(f'{prefix}{i}', base + i * 2., vector)
            learner.observe(item)
        for i in range(2):
            assert learner.confirm([f'{prefix}{i}'], text, base + i * 2. + .6)['status'] == 'confirmed'


class MotifTests(unittest.TestCase):
    def test_discovery_matures_without_inventing_meaning(self):
        learner = MotifLearner()
        results = [learner.observe(event(str(i), i)) for i in range(3)]
        self.assertEqual([r['status'] for r in results], ['provisional', 'provisional', 'matched'])
        self.assertEqual(len({r['motif_id'] for r in results}), 1)
        novel = learner.observe(event('novel', 4, (0., 1., 0.)))
        self.assertNotEqual(novel['motif_id'], results[0]['motif_id'])
        self.assertEqual(novel['status'], 'provisional')
        posterior = learner.posterior(['2'], 2.6)
        self.assertEqual(posterior['kind'], 'ungrounded_motifs')
        self.assertEqual(posterior['items'][0]['label'], UNKNOWN)
        self.assertFalse(results[-1]['calibrated'])
        self.assertEqual(results[-1]['score_kind'], 'cosine_similarity')

    def test_anonymous_patterns_cannot_gain_semantics(self):
        learner = MotifLearner()
        for i in range(3):
            learner.observe(event(str(i), i, ownership_valid=False, person_id=None))
        self.assertEqual(len(learner.prototypes), 1)
        self.assertEqual(learner.confirm(['2'], 'Hello', 2.6)['status'], 'no_eligible_evidence')
        self.assertEqual(learner.posterior(['2'], 2.6)['items'][0]['probability'], 1.)

    def test_duplicates_are_idempotent_and_conflicts_rejected(self):
        learner = MotifLearner()
        first = learner.observe(event('a'))
        self.assertTrue(learner.observe(event('a'))['duplicate'])
        self.assertEqual(learner.prototypes[first['motif_id']]['count'], 1)
        with self.assertRaises(ValueError):
            learner.observe(event('a', features=(0., 1., 0.)))

    def test_validation(self):
        learner = MotifLearner()
        for vector in [(float('nan'), 1.), (0., 0.), [[1., 2.]], (1.,)]:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                learner.observe(event('bad', features=vector))
        learner.observe(event('ok'))
        for changes in [dict(features=[1., 2.]), dict(feature_schema='other-model'), dict(t_end=-1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                learner.observe(event('changed', **changes))
        with self.assertRaises(ValueError):
            learner.posterior('ok', 1.)

    def test_pairs_require_common_clock_owner_and_time(self):
        a, v = event('a'), event('v', modality='video')
        for changes in [dict(person_id='someone-else'), dict(ownership_valid=False),
                        dict(session_id='s2'), dict(clock_valid=False),
                        dict(quality={'tts': True}), dict(quality={'overlap': True}),
                        dict(quality={'tracking_lost': True}), dict(t_start=3., t_end=3.5)]:
            learner = MotifLearner()
            self.assertEqual(learner.pair(a, {**v, **changes})['pair_status'], 'rejected')
            self.assertEqual(len(learner.prototypes), 0)
        learner = MotifLearner()
        pair = learner.pair(a, v)
        self.assertEqual(pair['pair_status'], 'paired')
        self.assertEqual(set(pair['event']['source_event_ids']), {'a', 'v'})
        self.assertTrue(learner.pair(a, v)['duplicate'])
        self.assertEqual(learner.pair(a, event('v2', modality='video'))['reason'], 'source_already_paired')

    def test_explicit_bounded_lag(self):
        a, v = event('a'), event('v', .6, modality='video')
        self.assertEqual(MotifLearner().pair(a, v)['pair_status'], 'rejected')
        self.assertEqual(MotifLearner().pair(a, v, max_lag_s=.11)['pair_status'], 'paired')
        with self.assertRaises(ValueError):
            MotifLearner().pair(a, v, max_lag_s=.3)

    def test_confirmations_ground_heldout_motif(self):
        learner = MotifLearner()
        learn_messages(learner)
        learner.observe(event('heldout', 10., (1., .03, 0.)))
        posterior = learner.posterior(['heldout'], 10.6)
        self.assertEqual(posterior['kind'], 'intent_posterior')
        self.assertEqual(posterior['items'][0]['label'], 'More water please')
        self.assertFalse(posterior['calibrated'])
        self.assertAlmostEqual(sum(x['probability'] for x in posterior['items']), 1.)
        self.assertIn(UNKNOWN, [x['label'] for x in posterior['items']])
        duplicate = learner.posterior(['heldout', 'heldout'], 10.6)
        self.assertEqual(duplicate, posterior)
        self.assertEqual(learner.confirm(['a0'], 'A different wording', .6)['status'], 'already_confirmed')
        self.assertEqual(learner.posterior(['heldout'], 20.)['kind'], 'ungrounded_motifs')

    def test_one_confirmation_and_novel_motif_abstain(self):
        learner = MotifLearner()
        for i in range(3):
            learner.observe(event(str(i), i))
        learner.confirm(['0'], 'A chosen message', .6)
        self.assertEqual(learner.posterior(['2'], 2.6)['kind'], 'ungrounded_motifs')
        learner.confirm(['1'], 'A chosen message', 1.6)
        learner.observe(event('new', 4., (0., 0., 1.)))
        self.assertEqual(learner.posterior(['new'], 4.6)['kind'], 'ungrounded_motifs')

    def test_pair_components_are_one_vote(self):
        learner = MotifLearner()
        pairs = []
        for i in range(3):
            pairs.append(learner.pair(event(f'a{i}', 2. * i), event(f'v{i}', 2. * i, modality='video')))
        for i in range(2):
            pair_id = pairs[i]['event_id']
            confirmed = learner.confirm([pair_id, f'a{i}', f'v{i}'], 'Please help', 2. * i + .6)
            self.assertEqual(confirmed['evidence_ids'], [pair_id])
        pair_id = pairs[-1]['event_id']
        self.assertEqual(learner.posterior([pair_id], 4.6), learner.posterior([pair_id, 'a2', 'v2'], 4.6))
        self.assertEqual(learner.confirm(['a0'], 'Different', .6)['status'], 'already_confirmed')

    def test_correlated_confirmations_do_not_count_twice(self):
        learner = MotifLearner()
        learner.observe(event('a'))
        learner.observe(event('overlap', .1))
        learner.confirm(['a'], 'Hello', .6)
        self.assertEqual(learner.confirm(['overlap'], 'Hello', .7)['status'], 'correlated_confirmation_rejected')
        self.assertEqual(len(learner.confirmations), 1)

    def test_rejection_is_not_negative_training_and_undo_works(self):
        learner = MotifLearner()
        for i in range(3):
            learner.observe(event(str(i), i))
        learner.confirm(['0'], 'Hello', .6)
        learner.confirm(['1'], 'Hello', 1.6)
        before = learner.posterior(['2'], 2.6)
        rejection = learner.reject(['wording-1'], ['2'], 2.6)
        self.assertFalse(rejection['trained_negative'])
        self.assertEqual(before, learner.posterior(['2'], 2.6))
        self.assertTrue(learner.undo())
        self.assertEqual(learner.posterior(['2'], 2.6)['kind'], 'ungrounded_motifs')

    def test_invalid_clock_and_only_explicit_negative_are_excluded(self):
        learner = MotifLearner()
        self.assertEqual(learner.observe(event('bad-clock', clock_valid=False))['status'], 'invalid_clock')
        self.assertEqual(learner.observe(event('unknown', is_communicative=False))['status'], 'provisional')
        self.assertEqual(learner.observe(event('negative', negative=True,
                                              quality={'confirmed_by_user': True}))['status'], 'confirmed_negative')

    def test_bounded_history_does_not_relearn_evicted_events(self):
        learner = MotifLearner(max_events=2)
        for i in range(3):
            learner.observe(event(str(i), i))
        count = next(iter(learner.prototypes.values()))['count']
        self.assertEqual(learner.observe(event('0', 0))['status'], 'expired_event')
        self.assertEqual(next(iter(learner.prototypes.values()))['count'], count)

    def test_persistence_roundtrip_and_metadata_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'private' / 'motifs.json'
            learner = MotifLearner(path)
            learn_messages(learner)
            learner.observe(event('probe', 10., (1., .03, .0001)))
            learner.observe(event('negative', 11., negative=True, quality={'confirmed_by_user': True}))
            learner.save()
            restored = MotifLearner(path)
            self.assertEqual(learner.posterior(['probe'], 10.6), restored.posterior(['probe'], 10.6))
            self.assertTrue(restored.observe(event('probe', 10., (1., .03, .0001)))['duplicate'])
            self.assertTrue(restored.events['negative']['confirmed_negative'])
            self.assertEqual((path.parent / '.gitignore').read_text().strip(), '*')
            data = json.loads(path.read_text())
            data['feature_spaces']['audio'][0] = 4
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                MotifLearner(path)

    def test_profile_refuses_tracked_source_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.git').mkdir()
            (root / '.gitignore').write_text('user-rule\n')
            with self.assertRaises(ValueError):
                MotifLearner().save(root / 'motifs.json')
            self.assertEqual((root / '.gitignore').read_text(), 'user-rule\n')
            MotifLearner().save(root / '.scratch-learning' / 'motifs.json')


if __name__ == '__main__':
    unittest.main()
