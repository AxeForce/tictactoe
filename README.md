# TicTacToe Robot 🤖♟️

A robot that plays TicTacToe using chess pieces (without pawns).

## Overview

This project uses:
- **Computer Vision** (YOLO + ArUco markers) to detect the board and pieces
- **Inverse Kinematics** (ikpy) to calculate arm movements
- **Minimax AI** to play optimal moves

## Piece Order

Each player must place pieces in this order:
1. Knight
2. Bishop
3. Rook
4. Queen
5. King

## Project Structure

```
tictactoe/
├── vision/                 # Computer vision module
│   ├── config.py          # Camera and detection settings
│   ├── camera.py          # Camera wrapper
│   ├── board_detector.py  # ArUco-based board detection
│   └── piece_detector.py  # YOLO-based piece detection
│
├── logic/                  # Game logic module
│   ├── game_state.py      # Game state management
│   ├── move_validator.py  # Move validation
│   ├── win_checker.py     # Win/draw detection
│   └── ai_player.py       # Minimax AI opponent
│
├── arm_control/           # Arm control module
│   ├── config.py          # Physical dimensions and calibration
│   ├── dofbot_ik.py       # Inverse kinematics solver
│   ├── gripper.py         # Gripper control
│   └── arm_controller.py  # High-level arm control
│
├── main.py                # Main orchestration script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Board

- Use a board with 4 ArUco markers at the corners (IDs 0-3)
- Place markers: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
- Markers should be from the `DICT_4X4_50` dictionary

### 3. Calibrate the Arm

Edit `arm_control/config.py` and update:
- `BOARD_ORIGIN_X/Y/Z` - Position of board relative to robot
- `CELL_SIZE` - Size of each cell in meters
- `WHITE_STAGING_X/Y/Z` - Where white pieces wait
- `BLACK_STAGING_X/Y/Z` - Where black pieces wait

### 4. Prepare the Pieces

Place pieces in the staging areas:
- White pieces on the left (Knight, Bishop, Rook, Queen, King)
- Black pieces on the right

## Running the Game

### Simulation Mode (no arm)

```bash
python -m tictactoe.main --simulate
```

### Full Mode (with arm control)

```bash
python -m tictactoe.main
```

### Robot Goes First

```bash
python -m tictactoe.main --robot-first
```

## Controls

- **q** - Quit the game
- **s** - Save screenshot
- **r** - Reset the game

## How It Works

1. **Human's Turn (WHITE)**:
   - Pick up your next piece (follow the order!)
   - Place it on an empty cell
   - The robot will detect the move via camera

2. **Robot's Turn (BLACK)**:
   - AI calculates the best move using Minimax
   - Arm picks piece from staging area
   - Arm places piece on the board

3. **Game End**:
   - First to get 3 in a row wins!
   - If all pieces are placed with no winner, it's a draw

## Quick Start Guide

### Step 1: Install Dependencies
```bash
cd tictactoe
pip install -r requirements.txt
```

### Step 2: Test All Modules
```bash
python test_modules.py
```
This will verify that vision, logic, and arm control modules are working.

### Step 3: Calibrate the Arm (on Raspberry Pi)
```bash
python calibrate_arm.py
```
Use this interactive tool to position the arm and record cell positions.

### Step 4: Run in Simulation Mode (Test First)
```bash
python -m tictactoe.main --simulate
```
This lets you test the vision and game logic without moving the arm.

### Step 5: Run the Full Game
```bash
python -m tictactoe.main
```

---

## Next Steps for Your Setup

Before playing, you need to configure these files for your specific hardware:

### 1. Update `vision/config.py`

| Setting | What to Change |
|---------|----------------|
| `CAMERA_INDEX` | Try `0` or `1` depending on your USB camera |
| `TFLITE_MODEL_PATH` | Path to your trained TFLite model (e.g., `models/best.tflite`) |
| `TFLITE_INPUT_SIZE` | Model input size (usually 640 for YOLO) |
| `PIECE_CLASS_NAMES` | List of class names in your model's order |
| `CELL_SIZE_MM` | Measure your actual board cell size |

### 2. Update `arm_control/config.py`

| Setting | What to Change |
|---------|----------------|
| `BOARD_ORIGIN_X/Y/Z` | Position of board corner relative to robot base (meters) |
| `CELL_SIZE` | Size of each cell in meters (e.g., `0.08` for 8cm) |
| `WHITE_STAGING_X/Y/Z` | Where white pieces are staged |
| `BLACK_STAGING_X/Y/Z` | Where black pieces are staged |
| `URDF_PATH` | Path to DOFBOT URDF file (if using ikpy) |

### 3. Print ArUco Markers

Generate and print 4 ArUco markers from `DICT_4X4_50`:
- **ID 0** → Top-left corner
- **ID 1** → Top-right corner
- **ID 2** → Bottom-right corner
- **ID 3** → Bottom-left corner

Use this Python snippet to generate them:
```python
import cv2
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for i in range(4):
    marker = cv2.aruco.generateImageMarker(aruco_dict, i, 200)
    cv2.imwrite(f"marker_{i}.png", marker)
```

### 4. Arrange the Staging Areas

Place pieces in order (Knight → Bishop → Rook → Queen → King):
- **White pieces**: Left side of the robot
- **Black pieces**: Right side of the robot

---

## Troubleshooting

### Camera not detected
- Check `vision/config.py` - try `CAMERA_INDEX = 0`
- Make sure camera is connected

### Board not detected
- Ensure all 4 ArUco markers are visible
- Check lighting conditions
- Print markers larger if needed

### Pieces not detected
- Make sure YOLO model path is correct in `vision/config.py`
- Train or fine-tune the model for your pieces

### Arm not moving
- Check if `Arm_Lib` is installed (only works on DOFBOT Pi)
- Run with `--simulate` to test without hardware

### IK solutions fail
- Calibrate board position in `arm_control/config.py`
- Make sure target positions are within arm reach

## License
cpk_dd33376397cd421b9b5a57503f775bd5.f61ee196c28c5f45b5a914e70340f5f6.rD7lg5vWbCZzfry1NQCheqLBwk8LMvC6

Part of the ChessRobot project.
