"""Offline tests: python -m pytest tests/test_hand_eye_evaluation.py."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import cv2
import matplotlib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")


@pytest.fixture
def module(monkeypatch):
    # The script parses argv and imports hardware SDKs at module scope.
    camera = ModuleType("airbot_camera")
    camera.RealsenseCamera = camera.USBCamera = Mock()
    arm = ModuleType("airbot_py.arm")
    arm.AIRBOTPlay = MagicMock()
    arm.RobotMode = SimpleNamespace(GRAVITY_COMP=0)
    monkeypatch.setitem(sys.modules, "airbot_camera", camera)
    monkeypatch.setitem(sys.modules, "airbot_py", ModuleType("airbot_py"))
    monkeypatch.setitem(sys.modules, "airbot_py.arm", arm)
    monkeypatch.setattr(sys, "argv", ["airbot_calibration.py"])
    spec = importlib.util.spec_from_file_location(
        "calibration_under_test", Path(__file__).resolve().parents[1] / "airbot_calibration.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.args.no_display_mode = True
    return result


def pose(rotvec, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = translation
    return result


@pytest.fixture
def scene(module, tmp_path):
    calibrator = module.AirbotCalibration.__new__(module.AirbotCalibration)
    calibrator.chessboard = module.ChessBoard()
    calibrator.cam2end = pose([0.12, -0.07, 0.09], [0.03, 0.02, 0.06])
    calibrator.hand_eye_errors = None
    calibrator._reviewed_corners = {}
    calibrator.image_indices = None
    calibrator.project_error = None
    calibrator.cam_intrinsic = np.array([[620., 0, 320], [0, 615, 240], [0, 0, 1]])
    calibrator.cam_distortion = np.array([[0.02, -0.01, 0.001, -0.002, 0.0]])
    calibrator.camera = SimpleNamespace(WIDTH=640, HEIGHT=480)
    calibrator.save_path = str(tmp_path)
    points = np.zeros((80, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:8, 0:10].T.reshape(-1, 2) * 0.02
    board_to_base = pose([0.1, 0.2, -0.1], [0.3, -0.1, 0.2])
    rng = np.random.default_rng(42)
    samples = []
    for i in range(12):
        board_to_camera = pose(rng.uniform(-0.4, 0.4, 3),
                               [-0.07 + i * 0.002, -0.09, 0.6 + i * 0.01])
        end_to_base = board_to_base @ np.linalg.inv(board_to_camera) @ np.linalg.inv(calibrator.cam2end)
        projected, _ = cv2.projectPoints(points, cv2.Rodrigues(board_to_camera[:3, :3])[0],
                                         board_to_camera[:3, 3], calibrator.cam_intrinsic,
                                         calibrator.cam_distortion)
        samples.append(dict(image_index=i, image_points=projected,
                            board_to_camera=board_to_camera, end_to_base=end_to_base))
    return calibrator, points, samples


def evaluate(scene):
    calibrator, points, samples = scene
    return calibrator.evaluate_hand_eye(samples)


def test_exact_solution(scene):
    result = evaluate(scene)
    assert result["status"] == "ok"
    assert max(result["summary"].values()) < 1e-8
    assert result is scene[0].hand_eye_errors


@pytest.mark.parametrize("component", ["rotation", "translation"])
def test_wrong_extrinsics_increase_errors(scene, component):
    calibrator = scene[0]
    if component == "translation":
        calibrator.cam2end[:3, 3] += [0.01, -0.02, 0.015]
    else:
        calibrator.cam2end[:3, :3] = (calibrator.cam2end[:3, :3]
                                    @ Rotation.from_euler("x", 5, degrees=True).as_matrix())
    result = evaluate(scene)
    assert result["status"] == "ok"
    assert result["summary"]["translation_rms_mm"] > 0.1
    if component == "rotation":
        assert result["summary"]["rotation_rms_deg"] > 0.1


def test_evaluation_uses_only_transforms(scene):
    for sample in scene[2]:
        del sample["image_points"]
    scene[0].cam_intrinsic = scene[0].cam_distortion = None
    result = evaluate(scene)
    assert result["status"] == "ok"
    assert max(result["summary"].values()) < 1e-8
    for sample, row in zip(scene[2], result["per_frame"]):
        expected = sample["end_to_base"] @ scene[0].cam2end @ sample["board_to_camera"]
        np.testing.assert_allclose(row["board_to_base"], expected, atol=1e-12)
        np.testing.assert_allclose(row["board_to_base"], result["board_to_base_reference"], atol=1e-12)


def test_millimetre_and_degree_units(scene):
    calibrator, points, _ = scene
    samples = []
    for i, offset in enumerate([-1, 0, 1]):
        samples.append(dict(image_index=i, end_to_base=np.eye(4),
                            board_to_camera=pose([0, 0, np.radians(offset)], [offset * 0.001, 0, 1]),
                            image_points=np.zeros((len(points), 1, 2))))
    calibrator.cam2end = np.eye(4)
    result = evaluate((calibrator, points, samples))
    for key in ["translation_rms_mm", "rotation_rms_deg"]:
        assert result["summary"][key] == pytest.approx(np.sqrt(2 / 3))
    for key in ["translation_max_mm", "rotation_max_deg"]:
        assert result["summary"][key] == pytest.approx(1)


def test_rotation_mean_handles_angle_wrap(scene):
    calibrator, points, _ = scene
    calibrator.cam2end = np.eye(4)
    samples = [dict(image_index=i, end_to_base=np.eye(4),
                    board_to_camera=pose([0, 0, np.radians(angle)], [0, 0, 1]))
               for i, angle in enumerate([179, 180, -179])]
    result = evaluate((calibrator, points, samples))
    assert result["status"] == "ok"
    assert result["summary"]["rotation_max_deg"] == pytest.approx(1)
    np.testing.assert_allclose(result["board_to_base_reference"][:3, :3],
                               np.diag([-1, -1, 1]), atol=1e-12)


@pytest.mark.parametrize("failure", ["few_frames", "nan", "nonrigid", "invalid_sample"])
def test_evaluation_failure_has_no_precision_numbers(scene, failure):
    calibrator, points, samples = scene
    assert evaluate(scene)["status"] == "ok"  # Failure must clear a previous success.
    if failure == "few_frames":
        samples = samples[:2]
    elif failure == "nan":
        calibrator.cam2end[0, 3] = np.nan
    elif failure == "nonrigid":
        calibrator.cam2end[0, 0] = 2
    else:
        samples[0]["board_to_camera"][3, 0] = 1
    result = evaluate((calibrator, points, samples))
    assert result["status"] == "failed"
    assert "summary" not in result and "per_frame" not in result
    calibrator.report_calibration()
    report = Path(calibrator.save_path, "Calibration_Report.txt").read_text()
    assert "Evaluation FAILED:" in report
    assert "Reprojection RMSE (pixels):" not in report


def mock_detection(module, scene, monkeypatch, missing=(), pnp_failed=(), pnp_raised=()):
    calibrator, _, samples = scene
    calibrator.images = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(len(samples))]
    calibrator.end_pose_matrixes = [s["end_to_base"] for s in samples]
    def detect(gray, *args, **kwargs):
        i = int(gray[0, 0])
        return (False, None) if i in missing else (True, samples[i]["image_points"].copy())
    monkeypatch.setattr(module.cv2, "findChessboardCornersSB", detect)
    def pnp(points, corners, *args):
        i = next(i for i, s in enumerate(samples) if np.array_equal(corners, s["image_points"]))
        if i in pnp_raised:
            raise cv2.error("synthetic PnP failure")
        if i in pnp_failed:
            return False, None, None
        board = samples[i]["board_to_camera"]
        return True, cv2.Rodrigues(board[:3, :3])[0], board[:3, 3].reshape(3, 1)
    monkeypatch.setattr(module.cv2, "solvePnP", pnp)


def test_calibration_skips_pairs_and_writes_outputs(module, scene, monkeypatch, capsys):
    calibrator, _, samples = scene
    expected = copy.deepcopy(calibrator.cam2end)
    mock_detection(module, scene, monkeypatch, missing=[1], pnp_failed=[3], pnp_raised=[5])
    # Use the real OpenCV hand-eye solver to also check robot/PnP pairing.
    actual = calibrator.calibrate_hand_eye(calibrator.cam_intrinsic, calibrator.cam_distortion)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    result = calibrator.hand_eye_errors
    assert result["status"] == "ok"
    assert result["valid_frames"] == len(samples) - 3
    assert [r["image_index"] for r in result["per_frame"]] == [0, 2, 4, 6, 7, 8, 9, 10, 11]
    assert max(result["summary"].values()) < 1e-4
    assert Path(calibrator.save_path, "hand_eye_error_analysis.jpg").stat().st_size > 0
    calibrator.report_calibration()
    report = Path(calibrator.save_path, "Calibration_Report.txt").read_text()
    assert "标定数据一致性误差" in report
    assert "Valid frames: 9" in report
    assert "Rotation RMS (degrees)" in report
    assert "--Extrinsic--" in report
    assert "reprojection" not in calibrator.format_hand_eye_report().lower()
    saved = json.loads(Path(calibrator.save_path, "hand_eye_consistency.json").read_text())
    assert saved["transform"] == "board_to_base"
    assert saved["translation_unit"] == "m"
    np.testing.assert_allclose(saved["board_to_base_reference"], result["board_to_base_reference"])
    assert [row["image_index"] for row in saved["per_frame"]] == [0, 2, 4, 6, 7, 8, 9, 10, 11]
    for actual_row, expected_row in zip(saved["per_frame"], result["per_frame"]):
        np.testing.assert_allclose(actual_row["board_to_base"], expected_row["board_to_base"])
    assert "--Hand-eye Evaluation--" in capsys.readouterr().out
    module.AIRBOTPlay.assert_not_called()


def test_calibration_requires_three_pairs(module, scene, monkeypatch):
    mock_detection(module, scene, monkeypatch, missing=range(2, 12))
    calibrator = scene[0]
    with pytest.raises(ValueError, match="at least 3"):
        calibrator.calibrate_hand_eye(calibrator.cam_intrinsic, calibrator.cam_distortion)


def test_invalid_solver_result_is_reported_without_plot(module, scene, monkeypatch):
    calibrator = scene[0]
    mock_detection(module, scene, monkeypatch)
    monkeypatch.setattr(module.cv2, "calibrateHandEye", lambda *args, **kwargs:
                        (np.full((3, 3), np.nan), np.zeros((3, 1))))
    calibrator.calibrate_hand_eye(calibrator.cam_intrinsic, calibrator.cam_distortion)
    assert calibrator.hand_eye_errors["status"] == "failed"
    assert not Path(calibrator.save_path, "hand_eye_error_analysis.jpg").exists()


def test_calibration_rejects_unpaired_data(scene):
    calibrator = scene[0]
    calibrator.images = [None]
    calibrator.end_pose_matrixes = []
    with pytest.raises(ValueError, match="matching lengths"):
        calibrator.calibrate_hand_eye(calibrator.cam_intrinsic, calibrator.cam_distortion)


def test_collection_persists_poses_before_next_capture(module, scene, monkeypatch):
    calibrator = scene[0]
    calibrator.type = "hand_eye"
    calibrator.images = []
    calibrator.end_pose_matrixes = []
    calibrator.chessboard.number_of_image_needed = 3
    robot = module.AIRBOTPlay.return_value.__enter__.return_value
    robot.get_end_pose.side_effect = [([0.1, 0.2, 0.3], [0, 0, 0, 1]),
                                     ([0.4, 0.5, 0.6], [0, 0, 1, 0])]
    pose_path = Path(calibrator.save_path, "robot_poses.json")
    count = 0
    def capture(*args):
        nonlocal count
        if count:
            saved = json.loads(pose_path.read_text())
            assert saved["transform"] == "end_to_base"
            assert saved["translation_unit"] == "m"
            assert list(saved["poses"]) == [f"image{i}.png" for i in range(count)]
            for i in range(count):
                np.testing.assert_allclose(saved["poses"][f"image{i}.png"],
                                           calibrator.end_pose_matrixes[i])
                assert Path(calibrator.save_path, f"image{i}.png").exists()
        if count == 2:
            raise RuntimeError("capture interrupted")
        count += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(calibrator, "choose_image", capture)
    monkeypatch.setattr(module.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(RuntimeError, match="capture interrupted"):
        calibrator.data_collect()
    assert len(json.loads(pose_path.read_text())["poses"]) == 2
    assert not Path(str(pose_path) + ".tmp").exists()


@pytest.mark.parametrize("observed_shape,projected_shape", [
    ((80, 2), (80, 1, 2)), ((80, 1, 2), (80, 2)), ((80, 1, 2), (80, 1, 2)),
])
def test_intrinsic_error_accepts_corner_layouts(module, scene, monkeypatch,
                                               observed_shape, projected_shape):
    calibrator = scene[0]
    calibrator.images = [np.zeros((480, 640, 3), dtype=np.uint8)]
    projected = np.arange(160, dtype=np.float64).reshape(80, 2)
    observed = (projected + [3, 4]).astype(np.float32).reshape(observed_shape)
    monkeypatch.setattr(module.cv2, "findChessboardCornersSB", lambda *args, **kwargs: (True, observed))
    for name in ["drawChessboardCorners", "imshow", "waitKey", "destroyAllWindows"]:
        monkeypatch.setattr(module.cv2, name, lambda *args: None)
    monkeypatch.setattr(module.cv2, "calibrateCamera", lambda *args:
                        (0, calibrator.cam_intrinsic, calibrator.cam_distortion,
                         [np.zeros(3)], [np.array([0., 0., 1.])]))
    monkeypatch.setattr(module.cv2, "projectPoints", lambda *args:
                        (projected.reshape(projected_shape), None))
    histogram_values = []
    real_hist = module.plt.hist
    def capture_hist(values, *args, **kwargs):
        histogram_values.append(values.copy())
        return real_hist(values, *args, **kwargs)
    monkeypatch.setattr(module.plt, "hist", capture_hist)
    try:
        calibrator.calibrate_camera()
        # Preserve the existing intrinsic report's L2/N definition.
        assert calibrator.project_error == pytest.approx(5 / np.sqrt(80))
        assert histogram_values[0].shape == (80,)
        np.testing.assert_allclose(histogram_values[0], 5)
        assert Path(calibrator.save_path, "error_analysis.jpg").stat().st_size > 0
    finally:
        module.plt.close("all")


def mock_review_window(module, monkeypatch, keys):
    module.args.no_display_mode = False
    for name in ["imshow", "destroyWindow"]:
        monkeypatch.setattr(module.cv2, name, Mock())
    wait_key = Mock(side_effect=keys)
    monkeypatch.setattr(module.cv2, "waitKey", wait_key)
    return wait_key


def test_manual_rejection_shared_by_intrinsic_and_hand_eye(module, scene, monkeypatch):
    calibrator, _, samples = scene
    mock_detection(module, scene, monkeypatch)
    keys = [13] * len(samples)
    keys[3], keys[7] = ord("D"), ord("d")
    wait_key = mock_review_window(module, monkeypatch, keys)
    accepted = [s for i, s in enumerate(samples) if i not in (3, 7)]
    def intrinsic_solver(object_points, image_points, *args):
        assert len(image_points) == len(accepted)
        for observed, sample in zip(image_points, accepted):
            np.testing.assert_array_equal(observed, sample["image_points"])
        return (0, calibrator.cam_intrinsic, calibrator.cam_distortion,
                [cv2.Rodrigues(s["board_to_camera"][:3, :3])[0] for s in accepted],
                [s["board_to_camera"][:3, 3] for s in accepted])
    monkeypatch.setattr(module.cv2, "calibrateCamera", intrinsic_solver)
    original_poses = np.array(calibrator.end_pose_matrixes).copy()
    try:
        calibrator.calibrate_camera()
        calibrator.calibrate_hand_eye(calibrator.cam_intrinsic, calibrator.cam_distortion)
        assert wait_key.call_count == len(samples)
        assert [r["image_index"] for r in calibrator.hand_eye_errors["per_frame"]] == [
            s["image_index"] for s in accepted]
        assert max(calibrator.hand_eye_errors["summary"].values()) < 1e-4
        assert len(calibrator.images) == len(samples)
        np.testing.assert_array_equal(calibrator.end_pose_matrixes, original_poses)
    finally:
        module.plt.close("all")


def test_all_frames_discarded_reports_clear_error(module, scene, monkeypatch):
    calibrator = scene[0]
    mock_detection(module, scene, monkeypatch)
    mock_review_window(module, monkeypatch, [ord("d")] * len(scene[2]))
    with pytest.raises(ValueError, match="No accepted chessboard frames"):
        calibrator.calibrate_camera()
    with pytest.raises(ValueError, match="at least 3"):
        calibrator.calibrate_hand_eye(calibrator.cam_intrinsic, calibrator.cam_distortion)


def write_offline_data(directory):
    directory.mkdir()
    poses = {}
    for index in [10, 2, 0]:
        cv2.imwrite(str(directory / f"image{index}.png"),
                    np.full((24, 32, 3), index, dtype=np.uint8))
        poses[f"image{index}.png"] = pose([0, 0, 0], [index * 0.01, 0, 0]).tolist()
    saved = {"transform": "end_to_base", "translation_unit": "m", "poses": poses}
    (directory / "robot_poses.json").write_text(json.dumps(saved))
    return saved


def test_offline_constructor_pairs_by_filename_without_hardware(module, tmp_path):
    source = tmp_path / "source"
    write_offline_data(source)
    module.args.input_dir = str(source)
    module.args.output_path = str(tmp_path / "results")
    module.RealsenseCamera.side_effect = AssertionError("Offline mode must not open a camera")
    module.AIRBOTPlay.side_effect = AssertionError("Offline mode must not connect to a robot")
    calibrator = module.AirbotCalibration()
    assert calibrator.image_indices == [0, 2, 10]
    assert [image[0, 0, 0] for image in calibrator.images] == [0, 2, 10]
    np.testing.assert_allclose([m[0, 3] for m in calibrator.end_pose_matrixes], [0, 0.02, 0.1])
    assert (calibrator.camera.WIDTH, calibrator.camera.HEIGHT) == (32, 24)
    assert Path(calibrator.save_path).is_relative_to(tmp_path / "results")
    calibrator.save_robot_poses()
    assert json.loads(Path(calibrator.save_path, "robot_poses.json").read_text()) == json.loads(
        (source / "robot_poses.json").read_text())
    module.RealsenseCamera.assert_not_called()
    module.AIRBOTPlay.assert_not_called()


@pytest.mark.parametrize("failure, message", [
    ("missing_file", "requires.*robot_poses.json"),
    ("missing_pose", "Missing robot pose for image2.png"),
    ("invalid_pose", "Invalid robot pose for image2.png"),
    ("wrong_units", "translation_unit=m"),
    ("resolution", "same resolution"),
    ("unreadable", "Cannot read calibration image"),
])
def test_offline_invalid_data(module, scene, tmp_path, failure, message):
    source = tmp_path / "source"
    saved = write_offline_data(source)
    if failure == "missing_pose":
        del saved["poses"]["image2.png"]
    elif failure == "invalid_pose":
        saved["poses"]["image2.png"][0][0] = 2
    elif failure == "wrong_units":
        saved["translation_unit"] = "mm"
    (source / "robot_poses.json").write_text(json.dumps(saved))
    if failure == "missing_file":
        (source / "robot_poses.json").unlink()
    elif failure == "resolution":
        cv2.imwrite(str(source / "image2.png"), np.zeros((8, 8, 3), np.uint8))
    elif failure == "unreadable":
        (source / "image2.png").write_bytes(b"not an image")
    calibrator = scene[0]
    calibrator.type = "hand_eye"
    with pytest.raises(ValueError, match=message):
        calibrator.load_data(source)


def test_offline_intrinsic_needs_no_pose_file(scene, tmp_path):
    source = tmp_path / "source"
    write_offline_data(source)
    (source / "robot_poses.json").unlink()
    calibrator = scene[0]
    calibrator.type = "intrinsic"
    calibrator._reviewed_corners[0] = None
    calibrator.load_data(source)
    assert len(calibrator.images) == 3
    assert calibrator.end_pose_matrixes == []
    assert calibrator._reviewed_corners == {}
