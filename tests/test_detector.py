import unittest

import numpy as np

from clip_service.detector import CandidateDetector
from clip_service.frame_buffer import EncodedFrame, FrameRingBuffer


class CandidateDetectorTests(unittest.TestCase):
    def test_five_frames_span_the_entire_configured_change_window(self):
        self.detector.stable_seconds = 2.0
        divergent = np.array([0.0, 1.0])
        decision = None
        for index in range(9):
            timestamp = index / 4
            self.ring.append(EncodedFrame(timestamp, b"frame", index))
            decision = self.detector.observe(timestamp, divergent, self.ring)
            if timestamp < 2.0:
                self.assertIsNone(decision)
        self.assertIsNotNone(decision)
        self.assertEqual([f.timestamp for f in decision.frames], [0, 0.5, 1, 1.5, 2])

    def test_frames_outside_change_window_cannot_complete_candidate(self):
        divergent = np.array([0.0, 1.0])
        self.detector.observe(0, divergent, self.ring)
        for index, timestamp in enumerate([-0.01, 0.25, 0.5, 0.75, 1.01]):
            self.ring.append(EncodedFrame(timestamp, b"frame", index))
        self.assertIsNone(self.detector.observe(1.1, divergent, self.ring))

    def setUp(self) -> None:
        self.detector = CandidateDetector(
            similarity_threshold=0.8,
            stable_seconds=1.0,
            frame_count=5,
        )
        self.detector.set_baseline(np.array([1.0, 0.0]))
        self.ring = FrameRingBuffer(3.0, max_fps=20)

    def test_requires_continuous_one_second_divergence(self) -> None:
        divergent = np.array([0.0, 1.0])
        similar = np.array([1.0, 0.0])
        decision = None
        for index in range(7):
            timestamp = index / 10
            self.ring.append(EncodedFrame(timestamp, bytes([index]), index))
            decision = self.detector.observe(timestamp, divergent, self.ring)
        self.assertIsNone(decision)

        self.ring.append(EncodedFrame(0.7, b"reset", 7))
        self.assertIsNone(self.detector.observe(0.7, similar, self.ring))
        for index in range(8, 18):
            timestamp = index / 10
            self.ring.append(EncodedFrame(timestamp, bytes([index]), index))
            decision = self.detector.observe(timestamp, divergent, self.ring)
        self.assertIsNone(decision)

        self.ring.append(EncodedFrame(1.8, b"confirmed", 18))
        decision = self.detector.observe(1.8, divergent, self.ring)
        self.assertIsNotNone(decision)

    def test_freezes_five_frames_at_quarter_second_targets(self) -> None:
        divergent = np.array([0.0, 1.0])
        decision = None
        for index in range(11):
            timestamp = index / 10
            self.ring.append(EncodedFrame(timestamp, str(timestamp).encode(), index))
            decision = self.detector.observe(timestamp, divergent, self.ring)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(len(decision.frames), 5)
        targets = [0.0, 0.25, 0.5, 0.75, 1.0]
        for frame, target in zip(decision.frames, targets):
            self.assertAlmostEqual(frame.timestamp, target, delta=0.051)

    def test_pending_suppresses_repeated_candidates_until_ack(self) -> None:
        divergent = np.array([0.0, 1.0])
        for index in range(11):
            timestamp = index / 10
            self.ring.append(EncodedFrame(timestamp, b"frame", index))
            decision = self.detector.observe(timestamp, divergent, self.ring)
        self.assertIsNotNone(decision)

        for index in range(11, 31):
            timestamp = index / 10
            self.ring.append(EncodedFrame(timestamp, b"frame", index))
            self.assertIsNone(self.detector.observe(timestamp, divergent, self.ring))

        self.detector.acknowledge()
        self.assertIsNone(self.detector.observe(3.1, divergent, self.ring))

    def test_missing_historical_frames_do_not_prevent_future_candidates(self) -> None:
        divergent = np.array([0.0, 1.0])
        self.ring.append(EncodedFrame(0, b"before-disconnect", 1))
        self.assertIsNone(self.detector.observe(0, divergent, self.ring))
        # The old window was lost during a long outage. A fresh complete window must work.
        candidate = None
        for index in range(31):
            timestamp = 10 + index / 10
            self.ring.append(EncodedFrame(timestamp, b"new-frame", index + 2))
            candidate = self.detector.observe(timestamp, divergent, self.ring)
            if candidate:
                break
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertGreaterEqual(candidate.started_at, 10)
        self.assertEqual(len(candidate.frames), 5)


if __name__ == "__main__":
    unittest.main()
