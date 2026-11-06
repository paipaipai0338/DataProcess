"""Run with: python -m unittest Img2Keypoint.test_check_results"""
import csv
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Img2Keypoint import check_results as check
from Img2Keypoint.utils import get_matched_pairs


class ReprojectionTests(unittest.TestCase):
    def test_inverse_extrinsics_distortion_and_invalid_depth(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
            for cam in check.CAMS:
                np.savez(folder / f"intrinsic_cam_{cam}.npz", K=K, dist=np.array([.1, 0, 0, 0, 0]))
                T = np.eye(4)
                T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
                T[:3, 3] = [1, 2, 0]
                np.save(folder / f"extrinsic_T_cam_{cam}_to_cam_A.npy", T)
            calibration = check.load_calibration(folder, "A")
            poses = np.tile([1., 3., 5.], (1, 17, 1))
            poses[0, 1] = [1, 3, -5]
            poses[0, 2] = np.nan
            pixels = check.project_poses(poses, calibration["B"])
            np.testing.assert_allclose(pixels[0, 0], [70.08, 50])
            self.assertTrue(np.isnan(pixels[0, 1:3]).all())

    def test_duplicate_competition_and_large_errors_are_preserved(self):
        observations = np.zeros((1, 17, 2))
        projections = np.zeros((2, 17, 2))
        projections[1, :, 0] = 30
        people = check.match_camera(projections, observations, np.ones((1, 17)), .5, 5)
        self.assertEqual(people[0]["person_2d_id"], 0)
        self.assertEqual(people[1]["status"], "conflict")
        self.assertTrue(np.isnan(people[1]["error_px"]))
        self.assertEqual(people[1]["nearest_error_px"], 30)
        projections = projections[:1].copy()
        projections[0, 0, 0] = 1700
        person = check.match_camera(projections, observations, np.ones((1, 17)), .5, 5)[0]
        self.assertEqual(person["error_px"], 100)
        scores = np.ones((1, 17))
        scores[0, 0] = .5
        person = check.match_camera(projections, observations, scores, .5, 5)[0]
        self.assertEqual(person["error_px"], 0)
        self.assertEqual(person["valid_joint_count"], 16)

    def test_reordered_people_insufficient_points_and_empty_frames(self):
        projections = np.zeros((3, 17, 2))
        projections[1, :, 0] = 200
        projections[2] = np.nan
        observations = projections[[1, 0]].copy()
        people = check.match_camera(projections, observations, np.ones((2, 17)), .5, 5)
        self.assertEqual([p["person_2d_id"] for p in people], [1, 0, None])
        self.assertEqual(people[2]["status"], "no_valid_projection")
        scores = np.zeros((2, 17))
        scores[:, :4] = 1
        people = check.match_camera(projections, observations, scores, .5, 5)
        self.assertEqual(people[0]["status"], "insufficient_joints")
        self.assertEqual(people[0]["nearest_valid_joint_count"], 4)
        people = check.match_camera(projections, np.empty((0, 17, 2)),
                                   np.empty((0, 17)), .5, 5, "no_detections")
        self.assertTrue(all(p["status"] == "no_detections" for p in people))
        self.assertEqual(check.match_camera(np.empty((0, 17, 2)), observations,
                                           np.ones((2, 17)), .5, 5), [])

    def test_indexed_synchronization_matches_existing_used_and_tie_rules(self):
        with TemporaryDirectory() as directory:
            camera = Path(directory)
            # Variable width names, equal millisecond timestamps, ties, missing frames.
            for index, cam in enumerate(check.CAMS):
                frames = camera / f"cam_{cam}" / "frames"
                frames.mkdir(parents=True)
                ns_values = [0, 10_000_000, 20_000_000, 40_000_000, 100_000_000]
                if index % 2:
                    ns_values = [5_000_000, 15_000_000, 15_000_001, 60_000_000]
                for ns in ns_values:
                    (frames / f"1790000000_{ns}.jpg").touch()
            original = list(get_matched_pairs(camera, max_error_ms=10).values())
            indexed = list(check.synchronized_images(camera, 10))
            self.assertEqual(indexed, original)

    def test_csv_both_sources_missing_data_and_viewer_navigation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / "data_collection" / "group_011"
            calib = root / "calib"
            calib.mkdir()
            K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
            calibration = (K, np.zeros(5), np.eye(3), np.zeros(3))
            pose = np.zeros((1, 17, 3))
            pose[0, :, 0] = np.linspace(-.3, .3, 17)
            pose[0, :, 1] = np.linspace(-.5, .5, 17)
            pose[..., 2] = 5
            ghost = pose.copy()
            ghost[..., 0] += 1
            smoothed = pose.copy()
            smoothed[..., 0] += .01  # Known one-pixel displacement.
            groups = [{}, {}]
            for cam in check.CAMS:
                np.savez(calib / f"intrinsic_cam_{cam}.npz", K=K, dist=np.zeros(5))
                np.save(calib / f"extrinsic_T_cam_{cam}_to_cam_A.npy", np.eye(4))
                frames = group / "camera" / f"cam_{cam}" / "frames"
                output = group / "camera results" / "2D" / f"cam_{cam}"
                frames.mkdir(parents=True)
                output.mkdir(parents=True)
                for index in range(2):
                    if cam == "H":
                        groups[index][cam] = None
                        continue
                    image = frames / f"1790000000_{index * 50_000_000:09d}.jpg"
                    cv2.imencode(".jpg", np.full((480, 640, 3), 225, np.uint8))[1].tofile(image)
                    groups[index][cam] = image
                    if cam == "G":
                        continue
                    kpts = check.project_poses(pose, calibration)
                    scores = np.ones((1, 17))
                    if cam == "F":
                        kpts, scores = np.empty((0, 17, 2)), np.empty((0, 17))
                    elif cam == "E":
                        scores[:, 4:] = 0
                    np.savez(output / f"{image.stem}.npz", kpts=kpts, scores=scores)
            names = [check.frame_name(g) for g in groups]
            for source, value in (("3D", np.concatenate([pose, ghost])), ("smoothed 3D", smoothed)):
                folder = group / "camera results" / source
                folder.mkdir()
                for index, name in enumerate(names):
                    with (folder / f"{name}.pkl").open("wb") as stream:
                        pickle.dump(value if index == 0 else np.empty((0, 17, 3)), stream)
                with (folder / "1790000003_000000000.pkl").open("wb") as stream:
                    pickle.dump(pose, stream)  # Deliberately no corresponding synchronized images.
            args = check.parse_args(["--root_path", str(root / "data_collection"),
                                     "--output_dir", str(root / "debug"), "--no_view"])
            samples = check.collect_samples(args)
            output = check.export_csv(samples, args)
            with output.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 5)
            raw, ghost_row, smooth = rows[:3]
            self.assertEqual([raw["source"], smooth["source"]], ["raw", "smoothed"])
            self.assertEqual(int(raw["valid_camera_count"]), 4)
            self.assertAlmostEqual(float(smooth["mean_error_px"]), 1)
            self.assertEqual(raw["cam_H_status"], "missing_image")
            self.assertEqual(raw["cam_G_status"], "missing_2d_file")
            self.assertEqual(raw["cam_F_status"], "no_detections")
            self.assertEqual(raw["cam_E_status"], "insufficient_joints")
            self.assertEqual(ghost_row["cam_A_status"], "conflict")
            self.assertEqual(float(ghost_row["cam_A_nearest_error_px"]), 100)
            self.assertTrue(np.isnan(float(ghost_row["mean_error_px"])))
            self.assertEqual(rows[-1]["cam_A_status"], "missing_sync")
            viewer = check.ReprojectionViewer(samples, args)
            self.assertEqual(viewer.source, "smoothed")
            viewer.toggle_source()
            self.assertEqual(viewer.source, "raw")
            self.assertEqual(len(viewer.axes), 8)
            self.assertTrue(viewer.save().is_file())
            viewer.move(1)  # Empty-person frame.
            viewer.move(1)  # Missing-synchronization frame.
            viewer.move(-1)
            self.assertEqual(viewer.index, 1)
            plt.close(viewer.fig)


if __name__ == "__main__":
    unittest.main()
