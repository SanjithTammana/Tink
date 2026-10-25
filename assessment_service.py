"""Django-callable assessment service; UI-independent recorded media -> JSON packet."""
from collections import deque
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from audio_layer import SR, EnergySegmenter, Voc2VecEncoder
from gemini_assessment import AssessmentClient
from session_capture import SessionClock
from session_engine import SessionEngine
from visual_session import VisualSession

ROOT = Path(__file__).resolve().parent


class ClipAssessmentService:
    def __init__(self, profile='default', speaker=None, root=ROOT):
        from audio_layer import safe_name
        from motif_learning import MotifLearner
        safe_name(profile)
        self.root, self.clock = Path(root), SessionClock()
        self.home = self.root / '.scratch_session' / profile
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / '.gitignore').write_text('*\n', encoding='utf-8')
        self.engine = SessionEngine(self.clock.session_id, MotifLearner(path=self.home / 'assessment-motifs.json'), speaker, self.root)
        self.visual, self.encoder, self.transcriber = None, None, None
        self.warnings, self.recent_assessments = [], deque(maxlen=5)

    def prepare(self):
        self.visual = VisualSession(self.clock.session_id, self.home / 'person.json')
        try:
            self.encoder = Voc2VecEncoder()
        except (RuntimeError, OSError):
            self.warnings.append('Local sound embeddings unavailable; raw audio is still included.')
        try:
            from huggingface_hub import try_to_load_from_cache
            from transcribe import Transcriber
            cached = try_to_load_from_cache('Systran/faster-whisper-base.en', 'config.json')
            if isinstance(cached, str):
                self.transcriber = Transcriber(str(Path(cached).parent)).load()
            else:
                self.warnings.append('Local Whisper not cached; Gemini can still assess clip audio.')
        except (RuntimeError, OSError, ImportError):
            self.warnings.append('Local transcription unavailable; Gemini can still assess clip audio.')
        return self

    def observe(self, frame, at_s):
        observation, events = self.visual.observe(frame, at_s)
        observation['session_mode'] = 'integrated'
        self.engine.set_role(self.visual.target.logical_id, observation['ownership_valid'])
        for event in events:
            event['feature_schema'] = 'face-pose-hands-temporal-64-v1'
            self.engine.observe(event)
        return observation

    def packet(self, clip, capture, noise_db=None):
        """Finalize local evidence once. No cloud request or speech occurs here."""
        import soundfile as sf
        wave, sr = sf.read(capture['audio_path'], dtype='float32')
        if sr != SR or wave.ndim != 1:
            raise ValueError('Recording must be 16 kHz mono')
        prior = self.engine.context.snapshot(capture['start_s'])
        turns, warnings = [], list(self.warnings)
        if self.transcriber is not None:
            try:
                turns = self.transcriber.transcribe_turns(wave, capture['start_s'])
            except Exception:
                warnings.append('Local ASR failed; raw clip audio remains available for Gemini.')
            for turn in turns:
                # Whisper estimate must remain within finalized clip boundaries.
                turn['start_s'] = min(capture['end_s'], max(capture['start_s'], turn['start_s']))
                turn['end_s'] = min(capture['end_s'], max(turn['start_s'], turn['end_s']))
                self.engine.context.add_turn(turn)
        audio_events = []
        if self.encoder is not None and capture['clock_valid']:
            segmenter = EnergySegmenter(noise_db=noise_db)
            candidates = []
            for start in range(0, len(wave), 320):
                candidates.extend(segmenter.feed(start, wave[start:start+320]))
            for source in [e for e in candidates if e.complete][:8]:
                try:
                    vector = self.encoder(source.wave)
                except Exception:
                    warnings.append('A local audio embedding failed; raw clip audio remains attached.')
                    continue
                event = {'event_id': clip.clip_id + '-audio-' + str(source.start), 'session_id': self.clock.session_id,
                    'modality': 'audio', 'person_id': None, 'ownership_valid': False,
                    'clock_valid': capture['clock_valid'], 't_start': capture['start_s'] + source.start / SR,
                    't_end': capture['start_s'] + source.end / SR, 'features': vector.tolist(),
                    'feature_schema': 'voc2vec-layer5-768-v1', 'quality': {'speaker_unknown': True}, 'context_id': 0}
                discovered = self.engine.observe(event)
                audio_events.append({k: v for k, v in event.items() if k != 'features'} | {'discovery': discovered})
            if len(candidates) > 8:
                warnings.append('Local embedding analysis capped at eight sound events; complete raw audio attached.')
        observations = deepcopy(clip.observations)
        # Confirmed role may cover the same uninterrupted track seen provisionally earlier.
        logical = self.engine.person_id
        stable = bool(observations) and all(o.get('logical_person_id') == logical and o.get('person_id') is not None for o in observations)
        owned = bool(stable and self.visual.target.ownership_valid)
        if len(observations) > 64:
            indices = np.linspace(0, len(observations)-1, 64).astype(int)
            observations = [observations[i] for i in indices]
        events = [e for e in self.engine.events if capture['start_s'] <= e['t_start'] <= e['t_end'] <= capture['end_s']]
        ids = list(dict.fromkeys([clip.clip_id] + [e['event_id'] for e in events]))[:128]
        clean_capture = {k: v for k, v in capture.items() if k not in {'video_path', 'audio_path'}}
        clean_capture['limitations'] = warnings
        clean_capture['audio_speaker_attribution'] = 'shared microphone; unknown'
        clean_capture['feature_sampling'] = {'captured_observations': len(clip.observations),
            'transmitted_observations': len(observations), 'complete_local_features': 'features.json'}
        packet = {'schema_version': 1, 'session_id': self.clock.session_id, 'clip_id': clip.clip_id,
            'event_ids': ids, 'capture': clean_capture,
            'communicator': {'person_id': logical, 'ownership_valid': owned,
                'role_source': self.visual.target.role_source,
                'role_scope': 'same uninterrupted selected track; initial provisional frames may be operator-confirmed later',
                'limitations': 'appearance heuristic; no disability or persistent speaker identity inferred'},
            'signals': {'visual_observations': observations,
                'events': [{k: v for k, v in e.items() if k != 'features'} for e in events], 'audio_events': audio_events},
            'conversation': {'prior_context': prior, 'local_transcript': turns,
                'prior_assessments_unconfirmed': list(self.recent_assessments)},
            'belief': self.engine.learner.posterior([e['event_id'] for e in events if e['ownership_valid']], capture['end_s']),
            'spelled': self.engine.spelled}
        # Enforce the client's serialized byte limit, not an assumed per-frame size.
        while len(json.dumps(packet, ensure_ascii=False, allow_nan=False).encode('utf-8')) > 240 * 1024:
            transmitted = packet['signals']['visual_observations']
            if len(transmitted) <= 2:
                raise ValueError('Evidence packet exceeds its bounded context budget; record a shorter clip')
            indices = np.linspace(0, len(transmitted)-1, max(2, len(transmitted)//2)).astype(int)
            packet['signals']['visual_observations'] = [transmitted[i] for i in indices]
            packet['capture']['feature_sampling']['transmitted_observations'] = len(indices)
            packet['capture']['feature_sampling']['reason'] = 'bounded JSON size; complete measured features remain in features.json'
        (clip.folder / 'assessment-input.json').write_text(json.dumps(packet, indent=2, allow_nan=False), encoding='utf-8')
        return packet

    def assess(self, clip, capture, packet=None, noise_db=None):
        packet = packet if packet is not None else self.packet(clip, capture, noise_db)
        # Reload ignored .env on each explicit attempt; adding a key needs no restart.
        result = AssessmentClient(self.root).assess(packet, capture['video_path'], capture['audio_path'])
        (clip.folder / 'assessment.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        self.recent_assessments.append({'clip_id': clip.clip_id, 'unconfirmed': True,
            'assessment_summary': result['assessment']['summary'], 'partner_context': result['partner_context']})
        return result

    def confirm(self, option, packet, at_s):
        # Raw clip references and partner hypotheses never become motif labels.
        evidence = [identity for identity in option['evidence_ids'] if identity != packet['clip_id']]
        if (not packet['communicator']['ownership_valid'] or packet['communicator']['person_id'] != self.engine.person_id
                or packet.get('capture', {}).get('clock_valid') is not True
                or option['kind'] not in {'message', 'hypothesis'}):
            evidence = []
        return self.engine.confirm_text(option['text'], evidence, at_s)

    def close(self):
        self.engine.learner.save()
        if self.visual is not None:
            self.visual.close()
