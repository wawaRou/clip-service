import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from clip_service.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_minimal_frigate_configuration_uses_defaults_and_camera_stream_mapping(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.toml"
            path.write_text(
                '[frigate]\nrtsp_base_url = "rtsp://frigate.local:8554"\n'
                '[cameras.living_room]\nstream = "living_room_sub"\n'
                '[cameras.door]\nstream = "door"\n',
                encoding="utf-8",
            )
            config = load_config(path, environ={})
            self.assertEqual(config.server.port, 18080)
            self.assertEqual(config.camera_settings("living_room").inference_fps, 10)
            self.assertEqual(
                config.camera_url("living_room"),
                "rtsp://frigate.local:8554/living_room_sub",
            )
            self.assertEqual(config.camera_url("door"), "rtsp://frigate.local:8554/door")
            self.assertEqual(config.model.path, Path(directory).resolve() / "models/clip")
            self.assertEqual(config.data_dir, Path(directory).resolve() / "data")

    def test_global_detection_defaults_allow_independent_camera_overrides(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.toml"
            path.write_text(
                '[frigate]\nrtsp_base_url = "rtsp://frigate:8554"\n'
                "[detection]\ninference_fps = 20\nstable_seconds = 0.2\n"
                '[cameras.hall]\nstream = "hall"\n'
                '[cameras.door]\nstream = "door"\ninference_fps = 15\nsimilarity_threshold = 0.95\n'
                "enabled = false\n",
                encoding="utf-8",
            )
            config = load_config(path, environ={})
            hall = config.camera_settings("hall")
            door = config.camera_settings("door")
            self.assertEqual(hall.inference_fps, 20)
            self.assertEqual(door.inference_fps, 15)
            self.assertEqual(hall.similarity_threshold, 0.9)
            self.assertEqual(door.similarity_threshold, 0.95)
            self.assertEqual(door.stable_seconds, 0.2)
            self.assertEqual(door.ack_timeout_seconds, 60)
            self.assertFalse(config.cameras["door"].enabled)

    def test_deployment_environment_overrides_file_and_encodes_credentials(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.toml"
            path.write_text(
                'data_dir = "stored"\n[model]\npath = "weights"\ndevice = "cpu"\n'
                '[frigate]\nrtsp_base_url = "rtsp://frigate:8554"\n'
                'username = "old-user"\npassword = "old-password"\n'
                '[cameras.door]\nstream = "door sub"\n',
                encoding="utf-8",
            )
            config = load_config(
                path,
                environ={
                    "CLIP_MODEL_PATH": "new-weights",
                    "CLIP_DATA_DIR": "new-data",
                    "CLIP_DEVICE": "mps",
                    "CLIP_FRIGATE_URL": "rtsp://new-frigate:8554/",
                    "CLIP_FRIGATE_USERNAME": "viewer@example",
                    "CLIP_FRIGATE_PASSWORD": "secret:/@",
                },
            )
            self.assertEqual(config.model.path, Path(directory).resolve() / "new-weights")
            self.assertEqual(config.data_dir, Path(directory).resolve() / "new-data")
            self.assertEqual(config.model.device, "mps")
            self.assertEqual(
                config.camera_url("door"),
                "rtsp://viewer%40example:secret%3A%2F%40@new-frigate:8554/door%20sub",
            )
            self.assertNotIn("secret", repr(config))
            self.assertNotIn("viewer@example", repr(config))

    def test_invalid_settings_are_located_without_exposing_input_values(self):
        cases = [
            ('unexpected = "private-value"\n', "unexpected"),
            ('[server]\nport = "private-value"\n', "server.port"),
            ("[server]\nport = 70000\n", "server.port"),
            ('[model]\ndevice = "private-value"\n', "model.device"),
            ("[detection]\ninference_fps = 0\n", "detection.inference_fps"),
            ("[detection]\nsimilarity_threshold = 1.2\n", "detection.similarity_threshold"),
            ("[detection]\nstable_seconds = nan\n", "detection.stable_seconds"),
            ("[detection]\nread_timeout_seconds = inf\n", "detection.read_timeout_seconds"),
            ("[detection]\njpeg_quality = true\n", "detection.jpeg_quality"),
            ("[detection]\ncandidate_frame_count = 3\n", "detection.candidate_frame_count"),
            ('[cameras.door]\nstream = "door"\ninference_fps = -1\n', "cameras.door.inference_fps"),
            ('[cameras.door]\nstream = ""\n', "cameras.door.stream"),
            ("[cameras.door]\n", "cameras.door.stream"),
            ("[cameras.door]\nread_timout_seconds = 3\n", "cameras.door.read_timout_seconds"),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.toml"
            for settings, expected_field in cases:
                with self.subTest(field=expected_field, settings=settings):
                    path.write_text(
                        settings + '[frigate]\nrtsp_base_url = "rtsp://frigate:8554"\n'
                        'username = "viewer"\npassword = "private-value"\n',
                        encoding="utf-8",
                    )
                    with self.assertRaises(ConfigError) as raised:
                        load_config(path, environ={})
                    self.assertIn(expected_field, str(raised.exception))
                    self.assertIn("clip.toml", str(raised.exception))
                    self.assertNotIn("private-value", str(raised.exception))

    def test_camera_sampling_and_candidate_window_must_fit_the_buffer(self):
        cases = [
            ("[detection]\ncandidate_frame_interval = 0.01\n", "candidate_frame_interval"),
            ("[detection]\ncandidate_frame_interval = 2\n", "candidate_frame_interval"),
            ("[detection]\nstable_seconds = 7\n", "stable_seconds"),
            ("[cameras.door]\nring_seconds = 0.2\n", "cameras.door"),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.toml"
            for settings, field in cases:
                with self.subTest(settings=settings):
                    path.write_text(
                        settings + '[frigate]\nrtsp_base_url = "rtsp://frigate:8554"\n',
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ConfigError, field):
                        load_config(path, environ={})

    def test_frigate_endpoint_rejects_invalid_urls_and_embedded_credentials(self):
        urls = [
            "https://frigate:8554",
            "rtsp:///no-host",
            "rtsp://frigate:invalid",
            "rtsp://private-value:private-value@frigate:8554",
            "rtsp://frigate:8554?token=secret",
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.toml"
            for url in urls:
                with self.subTest(url=url):
                    path.write_text(f'[frigate]\nrtsp_base_url = "{url}"\n', encoding="utf-8")
                    with self.assertRaisesRegex(ConfigError, "frigate.rtsp_base_url") as raised:
                        load_config(path, environ={})
                    self.assertNotIn("private-value", str(raised.exception))

    def test_file_errors_and_malformed_environment_sections_have_safe_diagnostics(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.toml"
            with self.assertRaisesRegex(ConfigError, "clip.toml"):
                load_config(path, environ={})
            path.write_text('[frigate]\npassword = "private-value', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "TOML") as raised:
                load_config(path, environ={})
            self.assertNotIn("private-value", str(raised.exception))
            path.write_text('model = "private-value"\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "model") as raised:
                load_config(path, environ={"CLIP_MODEL_PATH": "weights"})
            self.assertNotIn("private-value", str(raised.exception))

    def test_empty_cameras_remain_empty_and_file_paths_resolve_relative_to_config(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.toml"
            path.write_text(
                'cameras = {}\ndata_dir = "recordings"\n'
                '[model]\npath = "weights"\n'
                '[frigate]\nrtsp_base_url = "rtsp://frigate:8554"\n',
                encoding="utf-8",
            )
            config = load_config(path, environ={})
            self.assertEqual(config.cameras, {})
            self.assertEqual(config.model.path, Path(directory).resolve() / "weights")
            self.assertEqual(config.data_dir, Path(directory).resolve() / "recordings")


if __name__ == "__main__":
    unittest.main()
