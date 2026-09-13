## Dependencies Installation
```bash
# Install airbot configuration package (skip this step on Pro version)
sudo apt install ./airbot-configure_5.1.6-1_all.deb

# Set up Python virtual environment
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate

# Optional:
mkdir -p ~/.pip && echo -e "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.pip/pip.conf

# Arm environment installation reference:https://docs.airbots.online/airbot-play/quick-start/software-setup.html
# Install required packages 
pip install ./airbot_py-5.1.6-py3-none-any.whl
# Proxy might be required in command line or you can manually download and install the source code
pip install -r requirements.txt -i https://mirrors.huaweicloud.com/repository/pypi/simple
```
### Hand-Eye Calibration
**Note:** Default calibration resolution is 480p. Modify `configs/sam_simplegrasp.yaml` to change settings.

#### Run arm server
```bash
airbot_server -i can0 -p 50010
```
#### Run calib

The calibration board is a 9×11, 20mm black and white chessboard pattern calibration board.

```bash
python3 airbot_calibration.py  # Use -h flag for help options
```
Drag arm to change robot pose, make sure that the chessboard in the camera view, Press `ESC` to capture img and pose.

To recalibrate from saved data without connecting the camera or robot:

```bash
python3 airbot_calibration.py --input-dir calib/hand_eye_640x480/20260912183126
```

The directory must contain `image<N>.png` and, for hand-eye calibration, the
`robot_poses.json` saved during capture (`end_to_base` matrices in metres, keyed
by image filename). Images are sorted numerically and paired by filename; extra
pose entries for removed images are ignored. Use the same chessboard dimensions
and square size configured in `ChessBoard` as during capture.
Press `D` in the corner preview to discard a frame, or another key to keep it.
Add `--no-display-mode` to skip manual review and run without windows.
Use `--type intrinsic` to compute only intrinsics from images without robot poses.
Reports and plots are saved in a new timestamped directory under `--output-path`
(default: `calib/`); the input data is not overwritten.

Hand-eye evaluation assumes an eye-in-hand camera and a board fixed relative to
the robot base. For each accepted frame it computes
`board_to_base = end_to_base @ cam2end @ board_to_camera`, where `board_to_camera`
is the PnP pose and `cam2end` is the calibrated camera extrinsic. Translation is
compared to the mean position, and rotation to the mean rotation on SO(3).
The report gives RMS and maximum deviations in mm and degrees; smaller values
indicate better consistency across calibration frames, not absolute accuracy.
`hand_eye_error_analysis.jpg` plots the per-frame deviations, and
`hand_eye_consistency.json` saves every board-to-base matrix and the mean pose
(matrix translations are in metres). Camera intrinsic reprojection errors remain
in the separate intrinsic report and plot.

Calibration Suggestions:Capture images from as many different orientations and positions as possible during calibration. Be careful not to move the arm to its limit positions.

![alt text](assets/image-1.png)

After calibration completes,  The calibration results will be displayed in the command line ,update the corresponding parameters in `configs/sam_simplegrasp.yaml` with the calibration results.
![Calibration Result](assets/image-2.png)

You can use the following reference parameters under the conditions mentioned below.

Calibration Setup: The arm and the calibration board are placed on a white table at a height of 74.5 cm, and both are on the same plane.



![image-20250718163325802](/home/peng/snap/typora/96/.config/Typora/typora-user-images/image-20250718163325802.png)

```
480p:
    profile: [640, 480, 30]
    intrinsic:
      - [604.77563127,   0.        , 318.34741824]
      - [  0.        , 604.65868699, 249.83140396]
      - [  0.        ,   0.        ,   1.        ]
    distortion:
      - [0.04401480, 0.47978715, -0.00054849, -0.00361947, -1.93856636]
    extrinsic:
      - [ 0.00564713, -0.36529553,  0.93087447, -0.15035552]
      - [-0.99998351, -0.00109528,  0.00563656,  0.03493759]
      - [-0.00103944, -0.93089096, -0.36529570,  0.10947199]
      - [ 0.        ,  0.        ,  0.        ,  1.        ]
```



#### Run grasp app
```bash
source venv/bin/activate
python3 airbot_interface.py
```

#### Run arm server if not running

```bash
airbot_server -i can0 -p 50010
```

Basic Usage:

Click "Capture" to take a snapshot of the scene. Then click on the object in the image at the lower-left corner, and click "Pick and Place" to automatically recognize and perform the grasping action.

![image-20250718163630327](/home/peng/snap/typora/96/.config/Typora/typora-user-images/image-20250718163630327.png)

The basic graspable area is shown in the figure below, covering approximately 80% of the workspace.

![img](https://w79rvfxw83.feishu.cn/space/api/box/stream/download/asynccode/?code=NDhjNmE4MTA0YjJmZTFlMGU4OTc1YzFlYmU3YTBkMGVfbnh5WmUyVWpqeXZvVWpiWlNQaU9UZ0hKZFRvTUduMU5fVG9rZW46Q01mbGJzTUlJb01CZ1Z4cEhqMmNKRFVSbmVjXzE3NTI4MjgxNTQ6MTc1MjgzMTc1NF9WNA)

#### Debug

1. If the observe pose and place pose need to be changed, you can enter gravity compensaton mode, drag arm to the property pose, and copy the pose, Modify them in the `config/sam_simplegrasp.yaml`
![image-20250718164005357](/home/peng/snap/typora/96/.config/Typora/typora-user-images/image-20250718164005357.png)
