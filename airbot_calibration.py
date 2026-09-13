import argparse
import sys
import os
import datetime
import json
import re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2

import threading
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

DETECT_FLAG = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

parser = argparse.ArgumentParser(description="Airbot Calibration Tool")
parser.add_argument(
    "-t", "--type", type=str, default="hand_eye", choices=["hand_eye", "intrinsic"], help="Calibartion type, available calibration type: [hand_eye, intrinsic], default is hand_eye"
)
parser.add_argument(
    "-c", "--camera-type", type=str, default="realsense", choices=["usbcam", "realsense", "ros"], help="Camera type, available camera type: [usbcam, realsense], default is realsense"    
)
parser.add_argument(
    "-o", "--output-path", type=str, default="calib/", help="Directory to save the generated calibration data.",
)
parser.add_argument(
    "-p", "--port", type=int, default=50010, help="Robot server port number (default: 50010; must match airbot_server -p).",
)
parser.add_argument(
    "--ros-topic", type=str, default="/camera/image_raw", help="ROS image topic (valid when using ros cam)"
)
parser.add_argument(
    "--no-display-mode", action="store_true", help="Calibrate in no display mode."
)
parser.add_argument(
    "--input-dir", type=str, help="Load image<N>.png and robot_poses.json from an existing capture directory; skip hardware capture."
)

args = parser.parse_args()
space_len = 6
if args.no_display_mode:
    plt.switch_backend("Agg")

if not args.input_dir:
    try:
        from airbot_camera import RealsenseCamera, USBCamera
        from airbot_py.arm import AIRBOTPlay, RobotMode
    except ImportError as e:
        print(f"Failed to import capture hardware dependencies: {e}")
        sys.exit(1)

if args.camera_type == "ros" and not args.input_dir:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    from rclpy.executors import MultiThreadedExecutor
    class RosCamera:
        
        def __init__(self, topic: str, node_name: str = "airbot_calib_node"):
            # 如果外部还没初始化 rclpy，就先初始化
            if not rclpy.ok():
                rclpy.init()

            # 创建 ROS 2 节点
            self.node = Node(node_name)
            self.bridge = CvBridge()
            self.topic = topic

            # 缓存帧 & 锁
            self.latest_frame = None
            self.lock = threading.Lock()
            
            self.WIDTH = 640
            self.HEIGHT = 480

            # 订阅 Image 话题
            # 10 深度的 QoS 对实时性有帮助
            qos = rclpy.qos.QoSProfile(depth=10)
            self.subscription = self.node.create_subscription(
                Image,
                self.topic,
                self._callback,
                qos
            )

            # MultiThreadedExecutor + 背景线程，用于持续 spin
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = threading.Thread(target=self._spin, daemon=True)
            self.spin_thread.start()

        def _spin(self):
            """在后台线程里不断 spin"""
            try:
                self.executor.spin()
            except Exception as e:
                self.node.get_logger().error(f"Executor spin error: {e}")

        def _callback(self, msg: Image):
            """收到 ROS 图像消息后，用 CvBridge 转成 OpenCV，并缓存"""
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                self.node.get_logger().error(f"CvBridgeError: {e}")
                return

            with self.lock:
                self.latest_frame = cv_img
                self.WIDTH, self.HEIGHT = cv_img.shape[1], cv_img.shape[0]

        def get_frame(self, frame_type="bgr", align=False, timeout: float = 5.0):
            """
            拉取最新一帧（最多等待 timeout 秒）
            返回：OpenCV BGR 或 RGB 图像
            """
            start = time.time()
            while rclpy.ok() and (time.time() - start) < timeout:
                with self.lock:
                    if self.latest_frame is not None:
                        img = self.latest_frame.copy()
                        break
                time.sleep(0.01)
            else:
                raise RuntimeError(f"No image received on '{self.topic}' within {timeout}s")

            if frame_type.lower() == "rgb":
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        def destroy(self):
            """退出时清理资源"""
            # 停掉 executor
            self.executor.shutdown()
            # 销毁节点
            self.node.destroy_node()
            # 如果这就是唯一的节点，可以 shutdown rclpy
            # rclpy.shutdown()

class ChessBoard:
    def __init__(self):
        self.rows = 11
        self.cols = 8
        self.square_size = 0.02 # m
        self.number_of_image_needed = 30


class AirbotCalibration:
    def __init__(self):
        self.type = args.type
        if self.type == "hand_eye" and not hasattr(cv2, "calibrateHandEye"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} does not provide cv2.calibrateHandEye. "
                'Install a compatible version: python -m pip install "opencv-python>=4.5,<5"'
            )
        self.camera = None
        self.cam_intrinsic = None
        self.cam_distortion = None
        self.cam2end = None
        self.project_error = None
        self.hand_eye_errors = None
        
        self.images = []
        self.end_pose_matrixes = []
        self._reviewed_corners = {}
        self.image_indices = None
        
        if args.input_dir:
            self.load_data(args.input_dir)
        elif args.camera_type == "realsense":
            try:
                # Calibration uses image corners only; do not wait on depth frames.
                self.camera = RealsenseCamera(color_only=True)
            except ImportError as e:
                print(f"Error importing RealsenseCamera: {e}")
                print("Please make sure airbot_realsense module is installed.")
                sys.exit(1)
        elif args.camera_type == "usbcam":
            self.camera = USBCamera()
        elif args.camera_type == "ros":
            self.camera = RosCamera(args.ros_topic)
        else:
            raise ValueError(f"Unsupported camera type: {args.camera_type}")
        
        self.chessboard = ChessBoard()
        time_format = "%Y%m%d%H%M%S_offline_%f" if args.input_dir else "%Y%m%d%H%M%S"
        self.time_str = datetime.datetime.now().strftime(time_format)
        self.save_path = os.path.join(args.output_path, f"{self.type}_{self.camera.WIDTH}x{self.camera.HEIGHT}", self.time_str)
        os.makedirs(self.save_path, exist_ok=True)

    def load_data(self, input_dir):
        """Load saved images and end-to-base matrices, pairing by filename."""
        directory = Path(input_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Calibration input directory does not exist: {directory}")
        files = sorted(
            [p for p in directory.iterdir() if p.is_file() and re.fullmatch(r"image\d+\.png", p.name)],
            key=lambda p: int(p.stem[5:]))
        if not files:
            raise ValueError(f"No image<N>.png files found in {directory}")
        poses = {}
        if self.type == "hand_eye":
            pose_path = directory / "robot_poses.json"
            if not pose_path.is_file():
                raise ValueError(f"Hand-eye calibration requires {pose_path}; images alone only support --type intrinsic")
            with pose_path.open(encoding="utf-8") as file:
                saved = json.load(file)
            if (not isinstance(saved, dict) or saved.get("transform") != "end_to_base"
                    or saved.get("translation_unit") != "m" or not isinstance(saved.get("poses"), dict)):
                raise ValueError("robot_poses.json must contain transform=end_to_base, translation_unit=m and a poses dictionary")
            poses = saved["poses"]
        images, matrices, indices = [], [], []
        shape = None
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read calibration image: {path}")
            if shape is not None and image.shape[:2] != shape:
                raise ValueError(f"Calibration images must have the same resolution: {path.name}")
            shape = image.shape[:2]
            if self.type == "hand_eye":
                if path.name not in poses:
                    raise ValueError(f"Missing robot pose for {path.name} in robot_poses.json")
                matrix = np.asarray(poses[path.name], dtype=np.float64)
                try:
                    self._validate_hand_eye_pose(matrix)
                except ValueError as error:
                    raise ValueError(f"Invalid robot pose for {path.name}: {error}") from error
                matrices.append(matrix)
            images.append(image)
            indices.append(int(path.stem[5:]))
        if self.type == "hand_eye" and len(images) < 3:
            raise ValueError("Hand-eye calibration requires at least 3 saved image/pose pairs")
        self.images = images
        self.end_pose_matrixes = matrices
        self.image_indices = indices
        self._reviewed_corners.clear()
        # Resolution metadata only: no camera is opened in offline mode.
        self.camera = SimpleNamespace(WIDTH=shape[1], HEIGHT=shape[0])
        print(f"Loaded {len(images)} images from {directory}")

    def original_image_index(self, index):
        return self.image_indices[index] if self.image_indices is not None else index
        
    def choose_image(self, name="Image"):
        if args.no_display_mode:
            input("\nNo-display mode, Press Enter to capture image.")
            return self.camera.get_frame(frame_type="bgr", align=True)
        else:
            # In display mode, show the image and wait for ESC key
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            while True:
                image = self.camera.get_frame(frame_type="bgr", align=True)
                cv2.imshow(name, image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    return image
        
    def data_collect(self):
        self._reviewed_corners.clear()
        with AIRBOTPlay(port=args.port) as robot:
            robot.switch_mode(RobotMode.GRAVITY_COMP)
            print("Robot switched to GRAVITY_COMP mode.")
            if args.no_display_mode:
                print("Running in no-display mode.")
            else:
                print("Move the robot to capture positions. Press ESC to capture each position.")
            
        for i in range(self.chessboard.number_of_image_needed):
            image = self.choose_image(f"Collect data {i+1}/{self.chessboard.number_of_image_needed}")
            self.images.append(image)
            image_name = os.path.join(self.save_path, f"image{i}.png")
            if not cv2.imwrite(image_name, image):
                raise OSError(f"Failed to save calibration image: {image_name}")
            if self.type == "hand_eye":
                pose_matrix = None
                with AIRBOTPlay(port=args.port) as robot:
                    pose = robot.get_end_pose()
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, :3] = R.from_quat(pose[1]).as_matrix()
                    pose_matrix[:3, 3] = pose[0]
                self.end_pose_matrixes.append(pose_matrix)
                pose_file = self.save_robot_poses()
                print(f"--Data{i} Saved--\n  Image: {image_name}\n  Pose: {pose_matrix.flatten()}")
                print(f"  Pose file: {pose_file}")
            elif self.type == "intrinsic":
                print(f"--Data{i} Saved--\n  Image: {image_name}")
            else:
                raise ValueError("Unsupported calibration type")

            cv2.destroyAllWindows()

    def save_robot_poses(self):
        """Persist collected end-to-base poses, keyed by their image filenames."""
        data = {
            "transform": "end_to_base",
            "translation_unit": "m",
            "poses": {
                f"image{self.original_image_index(i)}.png": matrix.tolist()
                for i, matrix in enumerate(self.end_pose_matrixes)
            },
        }
        path = os.path.join(self.save_path, "robot_poses.json")
        # Replace only after the new file is complete, preserving earlier samples
        # if writing the next update fails or the process is interrupted.
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        return path
    
    def plot_calibration_result(self, project_errors, image_points, object_points, rvecs, tvecs, mtx, dist):
        plt.figure(figsize=(15,5))
        # Per-image error curve
        plt.subplot(131)
        plt.plot(project_errors, 'b-')
        plt.xlabel('Image Index'), plt.ylabel('Error (pixels)')
        plt.title('Per-image Reprojection Error')
        plt.grid(True)
        # Error histogram
        plt.subplot(132)
        all_errors = np.concatenate([
            np.linalg.norm(
                np.asarray(observed, dtype=np.float64).reshape(-1, 2)
                - np.asarray(cv2.projectPoints(o, r, t, mtx, dist)[0],
                             dtype=np.float64).reshape(-1, 2), axis=1)
            for observed, o, r, t in zip(image_points, object_points, rvecs, tvecs)
        ])
        plt.hist(all_errors.ravel(), bins=50, color='g')
        plt.xlabel('Error (pixels)'), plt.ylabel('Count')
        plt.title('Error Histogram')
        plt.grid(True)
        # Error distribution box plot
        plt.subplot(133)
        plt.boxplot(all_errors.ravel(), showfliers=False)
        plt.ylabel('Error (pixels)')
        plt.title('Error Distribution')

        plt.tight_layout()
        # Save the analysis plot
        save_dir = os.path.join(self.save_path, "error_analysis.jpg")
        plt.savefig(save_dir, dpi=300, bbox_inches='tight')
        print(f"Error analysis plot saved to: {save_dir}\n\n")
        
    def reviewed_corners(self, image_index):
        """Detect and review once; reuse the same selection for both calibrations."""
        if image_index in self._reviewed_corners:
            return self._reviewed_corners[image_index]
        image = self.images[image_index]
        original_index = self.original_image_index(image_index)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(
            gray, (self.chessboard.cols, self.chessboard.rows), flags=DETECT_FLAG)
        if not found:
            print(f"Image {original_index}: chessboard pattern not found; skipping")
            corners = None
        elif not args.no_display_mode:
            preview = image.copy()
            cv2.drawChessboardCorners(
                preview, (self.chessboard.cols, self.chessboard.rows),
                np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2), True)
            cv2.putText(preview, f"Image {original_index}: D = discard, other key = keep",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            window = "Review chessboard corners"
            cv2.imshow(window, preview)
            try:
                key = cv2.waitKey(0) & 0xFF
            finally:
                cv2.destroyWindow(window)
            if key in (ord("d"), ord("D")):
                print(f"Image {original_index}: discarded by user (including robot pose)")
                corners = None
        self._reviewed_corners[image_index] = corners
        return corners

    def calibrate_camera(self):
        print("\nStarting camera calibration...")
        # 3D object points of the chessboard
        object_point = np.zeros((self.chessboard.rows * self.chessboard.cols, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.chessboard.cols, 0:self.chessboard.rows].T.reshape(-1, 2)
        object_point *= self.chessboard.square_size
        
        object_points = []
        image_points = []
        for i in range(len(self.images)):
            corners = self.reviewed_corners(i)
            if corners is not None:
                object_points.append(object_point)
                image_points.append(corners)

        if not image_points:
            raise ValueError("No accepted chessboard frames remain for camera calibration")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (self.camera.WIDTH, self.camera.HEIGHT), None, None)
        
        self.cam_intrinsic = mtx
        self.cam_distortion = dist
        
        project_errors = []
        for i in range(len(object_points)):
            imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
            # OpenCV versions can return corners as (N, 2) or (N, 1, 2).
            # Normalize both layout and dtype before comparing corresponding points.
            observed = np.asarray(image_points[i], dtype=np.float64).reshape(-1, 2)
            projected = np.asarray(imgpoints2, dtype=np.float64).reshape(-1, 2)
            error = cv2.norm(observed, projected, cv2.NORM_L2) / len(projected)
            project_errors.append(error)
        self.project_error = np.mean(np.array(project_errors))
        
        self.plot_calibration_result(project_errors, image_points, object_points, rvecs, tvecs, mtx, dist)
        
        
    def calibrate_hand_eye(self, intrinsic, distortion):
        self.hand_eye_errors = None
        if len(self.images) != len(self.end_pose_matrixes):
            raise ValueError("Hand-eye images and robot poses must have matching lengths")
        # 3D object points of the chessboard
        object_point = np.zeros((self.chessboard.rows * self.chessboard.cols, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.chessboard.cols, 0:self.chessboard.rows].T.reshape(-1, 2)
        object_point *= self.chessboard.square_size
        
        R_checkerboard_to_camera_poses = []
        T_checkerboard_to_camera_poses = []
        R_end_to_base_poses = []
        T_end_to_base_poses = []
        valid_samples = []
        
        for i in range(len(self.images)):
            original_index = self.original_image_index(i)
            corners = self.reviewed_corners(i)
            if corners is None:
                continue
            else:
                try:
                    ret, rvec, tvec = cv2.solvePnP(object_point, corners, intrinsic, distortion)
                except cv2.error as error:
                    print(f"PnP failed in image {original_index}; skipping image and robot pose: {error}")
                    continue
                if not ret:
                    print(f"PnP failed in image {original_index}; skipping image and robot pose")
                    continue
                R_cam_pose, _ = cv2.Rodrigues(rvec)
                board_to_camera = np.eye(4)
                board_to_camera[:3, :3] = R_cam_pose
                board_to_camera[:3, 3] = tvec.flatten()
                self._validate_hand_eye_pose(board_to_camera)
                self._validate_hand_eye_pose(self.end_pose_matrixes[i])
                valid_samples.append({
                    "image_index": original_index,
                    "board_to_camera": board_to_camera,
                    "end_to_base": self.end_pose_matrixes[i].copy(),
                })
                R_checkerboard_to_camera_poses.append(R_cam_pose)
                T_checkerboard_to_camera_poses.append(tvec.flatten())
                
                R_end_to_base_poses.append(self.end_pose_matrixes[i][:3, :3])
                T_end_to_base_poses.append(self.end_pose_matrixes[i][:3, 3])
            
        if len(valid_samples) < 3:
            raise ValueError("Hand-eye calibration requires at least 3 valid image/pose pairs")

        R_cam2end, T_cam2end = cv2.calibrateHandEye(
            R_end_to_base_poses, T_end_to_base_poses, 
            R_checkerboard_to_camera_poses, T_checkerboard_to_camera_poses,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        self.cam2end = np.eye(4)
        self.cam2end[:3, :3] = R_cam2end
        self.cam2end[:3, 3] = T_cam2end.flatten()
        self.evaluate_hand_eye(valid_samples)
        if self.hand_eye_errors["status"] == "ok":
            self.plot_hand_eye_result()
        
        return self.cam2end

    @staticmethod
    def _validate_hand_eye_pose(pose):
        pose = np.asarray(pose)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Hand-eye transform must be a finite 4x4 matrix")
        rotation = pose[:3, :3]
        if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
            raise ValueError("Hand-eye transform must be a rigid transform")

    def evaluate_hand_eye(self, samples):
        """Measure board-to-base pose consistency for a fixed board (eye-in-hand).

        Each transform maps source coordinates into destination coordinates:
        board_to_base = end_to_base @ cam2end @ board_to_camera.
        Compare translations with their arithmetic mean and rotations with their
        SO(3) mean, using Euclidean distance (mm) and relative rotation angle (deg).
        Lower deviations mean better consistency, not absolute accuracy.
        """
        self.hand_eye_errors = {"status": "failed", "valid_frames": len(samples)}
        try:
            if len(samples) < 3:
                raise ValueError("Evaluation requires at least 3 valid image/pose pairs")
            self._validate_hand_eye_pose(self.cam2end)
            board_poses = []
            for sample in samples:
                self._validate_hand_eye_pose(sample["end_to_base"])
                self._validate_hand_eye_pose(sample["board_to_camera"])
                board_poses.append(sample["end_to_base"] @ self.cam2end
                                   @ sample["board_to_camera"])
            board_poses = np.asarray(board_poses)
            if not np.isfinite(board_poses).all():
                raise ValueError("Non-finite board-to-base transform")
            reference = np.eye(4)
            reference[:3, :3] = R.from_matrix(board_poses[:, :3, :3]).mean().as_matrix()
            reference[:3, 3] = board_poses[:, :3, 3].mean(axis=0)
            per_frame = []
            for sample, board_pose in zip(samples, board_poses):
                per_frame.append({
                    "image_index": sample["image_index"],
                    "board_to_base": board_pose,
                    "translation_error_mm": float(1000 * np.linalg.norm(board_pose[:3, 3] - reference[:3, 3])),
                    "rotation_error_deg": float(np.degrees(R.from_matrix(
                        reference[:3, :3].T @ board_pose[:3, :3]).magnitude())),
                })
            translations = np.array([row["translation_error_mm"] for row in per_frame])
            rotations = np.array([row["rotation_error_deg"] for row in per_frame])
            summary = {
                "translation_rms_mm": float(np.sqrt(np.mean(translations ** 2))),
                "translation_max_mm": float(translations.max()),
                "rotation_rms_deg": float(np.sqrt(np.mean(rotations ** 2))),
                "rotation_max_deg": float(rotations.max()),
            }
            if not np.isfinite(list(summary.values())).all():
                raise ValueError("Non-finite hand-eye error statistics")
            self.hand_eye_errors = {
                "status": "ok", "valid_frames": len(samples),
                "board_to_base_reference": reference, "summary": summary, "per_frame": per_frame,
            }
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            self.hand_eye_errors["reason"] = str(error)
            print(f"Hand-eye evaluation failed: {error}")
        return self.hand_eye_errors

    def plot_hand_eye_result(self):
        rows = self.hand_eye_errors["per_frame"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, key, label in zip(axes,
                ["translation_error_mm", "rotation_error_deg"],
                ["Translation deviation (mm)", "Rotation deviation (degrees)"]):
            ax.plot([row["image_index"] for row in rows], [row[key] for row in rows], "o-")
            ax.set_xlabel("Original image index (zero-based)")
            ax.set_ylabel(label)
            ax.grid(True)
        fig.suptitle("Board-to-base pose consistency")
        fig.tight_layout()
        path = os.path.join(self.save_path, "hand_eye_error_analysis.jpg")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Hand-eye error analysis plot saved to: {path}")

    def format_hand_eye_report(self):
        result = self.hand_eye_errors
        if result is None:
            return ""
        lines = ["\n--Hand-eye Evaluation--",
                 "Board-to-base pose consistency (标定数据一致性误差); fixed board, eye-in-hand.",
                 "board_to_base = end_to_base @ cam2end @ board_to_camera",
                 "Deviations from mean pose; smaller is better, not absolute accuracy.",
                 f"Valid frames: {result['valid_frames']}"]
        if result["status"] != "ok":
            lines.append(f"Evaluation FAILED: {result['reason']}")
        else:
            labels = {"translation_rms_mm": "Translation RMS (mm)",
                      "translation_max_mm": "Translation max (mm)",
                      "rotation_rms_deg": "Rotation RMS (degrees)",
                      "rotation_max_deg": "Rotation max (degrees)"}
            for key, label in labels.items():
                lines.append(f"{label}: {result['summary'][key]:.6f}")
            lines.append("Mean board-to-base transform (translation in m):")
            lines.append(np.array2string(result["board_to_base_reference"], precision=8))
            lines.append("Image index (zero-based), translation deviation (mm), rotation deviation (degrees)")
            for row in result["per_frame"]:
                lines.append(f"{row['image_index']}, "
                             f"{row['translation_error_mm']:.6f}, {row['rotation_error_deg']:.6f}")
        return "\n".join(lines) + "\n"
    
    def report_calibration(self):
        reporter_head = f"""---Calibration Report---
Camera Type: {args.camera_type}
Resolution: {self.camera.WIDTH}x{self.camera.HEIGHT}
Chessboard: {self.chessboard.rows}x{self.chessboard.cols}-{self.chessboard.square_size}m
"""
        if self.project_error is not None:
            reporter_head += f"Project Error: {self.project_error}\n"
        reporter_head += self.format_hand_eye_report()
            
        print(reporter_head)
        
        if self.cam_intrinsic is not None:
            print("--Intrinsic--")
            MatrixPrinter.print_matrix(self.cam_intrinsic)
        
        if self.cam_distortion is not None:
            print("\n--Distortion--")
            MatrixPrinter.print_matrix(self.cam_distortion)
        
        if self.cam2end is not None:
            print("\n--Extrinsic--")
            MatrixPrinter.print_matrix(self.cam2end)
        
        file_name = os.path.join(self.save_path, "Calibration_Report.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(reporter_head)
            
        if self.cam_intrinsic is not None:
            MatrixPrinter.save_matrix(self.cam_intrinsic, "Intrinsic", file_name)
            
        if self.cam_distortion is not None:
            MatrixPrinter.save_matrix(self.cam_distortion, "Distortion", file_name)
            
        if self.cam2end is not None:
            MatrixPrinter.save_matrix(self.cam2end, "Extrinsic", file_name)
            
        if self.hand_eye_errors is not None:
            pose_file = os.path.join(self.save_path, "hand_eye_consistency.json")
            with open(pose_file, "w", encoding="utf-8") as file:
                json.dump({"transform": "board_to_base", "translation_unit": "m",
                           **self.hand_eye_errors}, file, indent=2, allow_nan=False,
                          default=lambda value: value.tolist())
                file.write("\n")
            print(f"Hand-eye consistency data saved to: {pose_file}")

        print(f"\nCalibration report saved to: {file_name}")
        
        
class MatrixPrinter:
    """Utilities for formatted printing of matrices"""
    
    @staticmethod
    def format_number(v: float) -> str:
        """Format a number for display with consistent spacing"""
        if abs(v - 0) < 1e-12:
            return "0.        "
        elif abs(v - 1) < 1e-12:
            return "1.        "
        else:
            return f"{v:.8f}"
    
    @staticmethod
    def print_matrix(matrix: np.ndarray) -> None:
        """Print a matrix in two different formats for readability"""
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy ndarray.")

        # Format all values
        str_matrix = [[MatrixPrinter.format_number(val) for val in row] for row in matrix]
        
        # Calculate max width per column for alignment
        col_widths = [max(len(row[i]) for row in str_matrix) for i in range(matrix.shape[1])]

        # Create formatted row strings
        def format_row(row):
            return ", ".join(f"{val:>{col_widths[i]}}" for i, val in enumerate(row))

        print("\nlist format:")
        if matrix.shape[0] == 1:
            print(f"[{format_row(str_matrix[0])}]")
        else:
            print("[")
            for i, row in enumerate(str_matrix):
                comma = "," if i < len(str_matrix) - 1 else ""
                print(f" [{format_row(row)}]{comma}")
            print("]")

        print("\nyaml format:")
        for row in str_matrix:
            print(" " * space_len + f"- [{format_row(row)}]")
    
    @staticmethod
    def save_matrix(matrix: np.ndarray, segment_name, file_path: str) -> None:
        """Save matrix to file in both list and yaml formats"""
        with open(file_path, "a") as f:
            if not isinstance(matrix, np.ndarray):
                raise TypeError("Input must be a numpy ndarray.")

            # Format all values
            str_matrix = [[MatrixPrinter.format_number(val) for val in row] for row in matrix]
            
            # Calculate max width per column for alignment
            col_widths = [max(len(row[i]) for row in str_matrix) for i in range(matrix.shape[1])]

            # Create formatted row strings
            def format_row(row):
                return ", ".join(f"{val:>{col_widths[i]}}" for i, val in enumerate(row))

            f.write(f"\n--{segment_name}--\n")
            f.write("list format:\n")
            if matrix.shape[0] == 1:
                f.write(f"[{format_row(str_matrix[0])}]\n")
            else:
                f.write("[\n")
                for i, row in enumerate(str_matrix):
                    comma = "," if i < len(str_matrix) - 1 else ""
                    f.write(f" [{format_row(row)}]{comma}\n")
                f.write("]\n")

            f.write("yaml format:\n")
            for row in str_matrix:
                f.write(f"    - [{format_row(row)}]\n")

def draw_frame(T, ax=None, name='frame'):
    length=0.1
    linewidth=1.0
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    origin = T[:3, 3]
    x_axis = T[:3, 0] * length
    y_axis = T[:3, 1] * length
    z_axis = T[:3, 2] * length

    ax.quiver(*origin, *x_axis, color='r', linewidth=linewidth)
    ax.quiver(*origin, *y_axis, color='g', linewidth=linewidth)
    ax.quiver(*origin, *z_axis, color='b', linewidth=linewidth)

    if origin[0] == 0 and origin[1] == 0 and origin[2] == 0:
        coord_str = f'{name} (0,0,0)'
    else:
        coord_str = f'{name} ({origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f})'
    ax.text(*origin, coord_str, fontsize=10)


    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([-0.25, 0.25])
    ax.set_zlim([-0.25, 0.25])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([0.5,0.5,0.5])

    return ax

if __name__ == "__main__":
    calibrator = None
    try:
        calibrator = AirbotCalibration()
        print(f"\n{'='*60}")
        print(f"Airbot Calibration Tool")
        print(f"{'='*60}")
        print(f"Calibration type: {args.type}")
        print(f"Camera type: {args.camera_type}")
        print(f"Resolution: {calibrator.camera.WIDTH}x{calibrator.camera.HEIGHT}")
        print(f"Output path: {calibrator.save_path}")
        print(f"Robot port: {args.port}")
        if args.camera_type == "usbcam" and not args.input_dir:
            print(f"USB camera device ID: {calibrator.camera.device_id}")
        print(f"{'='*60}\n")
    
        print("Initialized calibrator successfully.")
        
        if not args.input_dir:
            calibrator.data_collect()
            print("Data collection complete.")
        
        calibrator.calibrate_camera()
            
        if args.type == "hand_eye":
            calibrator.calibrate_hand_eye(calibrator.cam_intrinsic, calibrator.cam_distortion)
            
        calibrator.report_calibration()
        if calibrator.hand_eye_errors is not None and calibrator.hand_eye_errors["status"] != "ok":
            print("\nCalibration parameters computed, but hand-eye evaluation FAILED; see the report.")
        else:
            print("\nCalibration process completed successfully.")

        if calibrator.cam2end is not None:
            ax = draw_frame(np.eye(4), name='eef')
            draw_frame(calibrator.cam2end, ax=ax, name='cam')
            ax.view_init(elev=20, azim=70)
            frame_plot_path = os.path.join(calibrator.save_path, "ee_camera_frames.png")
            ax.figure.savefig(frame_plot_path, dpi=300, bbox_inches="tight")
            print(f"EE/camera coordinate frame plot saved to: {frame_plot_path}")
            if args.no_display_mode:
                plt.close(ax.figure)
        if not args.no_display_mode:
            plt.show()
                
    except Exception as e:
        print(f"\nError during calibration: {e}")
        sys.exit(1)
    finally:
        if calibrator is not None and not args.input_dir:
            if hasattr(calibrator.camera, "deinit"):
                calibrator.camera.deinit()
        cv2.destroyAllWindows()
