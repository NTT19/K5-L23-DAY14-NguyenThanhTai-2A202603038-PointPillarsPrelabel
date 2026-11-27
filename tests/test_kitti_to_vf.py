"""Tests for the KITTI -> VF converter and the CVAT bundle-only import."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import kitti_to_vf  # noqa: E402
from scripts import vf_common  # noqa: E402
from scripts import vf_cvat  # noqa: E402


CALIB = """P2: 700 0 600 0 0 700 180 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0
"""


def _args(input_path: Path, output_path: Path, **overrides):
    values = dict(
        input=str(input_path), output=str(output_path), chunk_size=60,
        max_sequences=None, camera_name="cam05", images=True, labels=True,
        write_calib=True, strict_labels=True, force=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class TestKittiToVf(unittest.TestCase):
    def test_pcd_intensity_mapping_keeps_low_returns_visible(self):
        self.assertEqual(vf_common.intensity_gray(0.0), 96)
        self.assertGreater(vf_common.intensity_gray(0.02), 96)
        self.assertGreater(vf_common.intensity_gray(0.44), vf_common.intensity_gray(0.02))
        self.assertEqual(vf_common.intensity_gray(10.0), 255)

    def test_object_points_labels_images_and_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "training"
            (source / "velodyne").mkdir(parents=True)
            (source / "image_2").mkdir()
            (source / "label_2").mkdir()
            (source / "calib").mkdir()
            points = np.array([[10.0, 2.0, -1.0, 0.7]], dtype=np.float32)
            points.tofile(source / "velodyne" / "000000.bin")
            (source / "calib" / "000000.txt").write_text(CALIB, encoding="ascii")
            (source / "label_2" / "000000.txt").write_text(
                "Car 0 0 0 0 0 10 10 1.5 1.8 4.0 2.0 1.5 10.0 0.0\n",
                encoding="ascii",
            )
            from PIL import Image
            Image.new("RGB", (12, 8), (20, 30, 40)).save(source / "image_2" / "000000.png")

            output = base / "Data-VF"
            self.assertEqual(kitti_to_vf.convert(_args(source, output)), 0)

            converted = np.fromfile(output / "sequence_01" / "lidar" / "000000.bin", dtype=np.float32).reshape(-1, 4)
            np.testing.assert_allclose(converted[0], [2.0, -10.0, -1.0, 0.7])
            label = (output / "sequence_01" / "000000.txt").read_text(encoding="utf-8").split()
            self.assertEqual(label[-1], "1")
            self.assertAlmostEqual(float(label[0]), -2.0, places=5)
            self.assertAlmostEqual(float(label[1]), -10.0, places=5)
            self.assertAlmostEqual(float(label[2]), -0.75, places=5)
            self.assertAlmostEqual(float(label[6]), 3 * math.pi / 2, places=5)
            self.assertTrue((output / "sequence_01" / "cameras" / "cam05" / "000000.jpg").is_file())
            self.assertTrue((output / "sequence_01" / "calib.json").is_file())
            manifest = json.loads((output / "conversion_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frames"], 1)
            self.assertEqual(manifest["source_format"], "kitti-object")

    def test_tracking_labels_are_filtered_by_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "kitti"
            (source / "velodyne" / "0000").mkdir(parents=True)
            (source / "calib").mkdir()
            (source / "label_02").mkdir()
            for index in range(2):
                np.zeros((1, 4), dtype=np.float32).tofile(source / "velodyne" / "0000" / f"{index:06d}.bin")
            (source / "calib" / "0000.txt").write_text(CALIB, encoding="ascii")
            (source / "label_02" / "0000.txt").write_text(
                "0 1 Car 0 0 0 0 0 10 10 1.5 1.8 4.0 2.0 1.5 10.0 0.0\n"
                "1 2 Pedestrian 0 0 0 0 0 10 10 1.7 0.6 0.6 1.0 1.5 8.0 0.2\n",
                encoding="ascii",
            )
            output = base / "Data-VF"
            self.assertEqual(kitti_to_vf.convert(_args(source, output, chunk_size=2)), 0)
            first = (output / "sequence_01" / "000000.txt").read_text(encoding="utf-8").split()
            second = (output / "sequence_01" / "000001.txt").read_text(encoding="utf-8").split()
            self.assertEqual(first[-1], "1")
            self.assertEqual(second[-1], "5")

    def test_import_vf_dry_run_creates_cvat_pointcloud_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "Data-VF" / "sequence_01" / "lidar"
            data.mkdir(parents=True)
            np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32).tofile(data / "000000.bin")
            bundle = base / "import.zip"
            self.assertEqual(vf_cvat.main([
                "import-vf", "--data", str(base / "Data-VF"), "--task-name", "KITTI-VF",
                "--frames", "s01f00", "--dry-run", "--bundle-out", str(bundle),
            ]), 0)
            with zipfile.ZipFile(bundle) as archive:
                self.assertIn("pointcloud/s01f00.pcd", archive.namelist())


if __name__ == "__main__":
    unittest.main()
