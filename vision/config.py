"""
Vision configuration for TicTacToe robot.
All the settings for camera, board detection, and piece recognition.

Raspberry Pi 4 Setup:
    pip install tflite-runtime opencv-python-headless numpy
    
    For camera: sudo apt install libcamera-apps
"""

import cv2


class VisionConfig:
    """
    Configuration class for vision settings.
    Change these values based on your setup!
    """
    
    # ==================== CAMERA SETTINGS ====================
    CAMERA_INDEX = 1  # USB camera index (try 0 if 1 doesn't work)
    # Use CAP_V4L2 on Raspberry Pi, CAP_DSHOW on Windows
    CAMERA_BACKEND = cv2.CAP_DSHOW  # CAP_DSHOW for Windows, CAP_V4L2 for RPi
    CAMERA_WIDTH = 640   # Lower resolution for faster processing on RPi
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # ==================== BOARD SETTINGS ====================
    # TicTacToe is a 3x3 grid
    BOARD_SIZE = 3
    
    # Physical size of each cell in millimeters
    # Measure your actual board and update this!
    CELL_SIZE_MM = 80.0
    
    # Total board size
    BOARD_SIZE_MM = CELL_SIZE_MM * BOARD_SIZE  # 240mm
    
    # ArUco marker settings
    # We use 4 markers at the corners of the board
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    ARUCO_MARKER_IDS = [0, 1, 2, 3]  # top-left, top-right, bottom-right, bottom-left
    
    # Output size for warped board image (pixels)
    BOARD_OUTPUT_SIZE = 600
    CELL_OUTPUT_SIZE = BOARD_OUTPUT_SIZE // BOARD_SIZE  # 200 pixels per cell
    
    # ==================== TFLITE MODEL SETTINGS ====================
    # Path to your trained TFLite model
    TFLITE_MODEL_PATH = "models/best_int8.tflite"
    TFLITE_CONFIDENCE = 0.25
    TFLITE_IOU_THRESHOLD = 0.45  # For NMS
    
    # Model input size (must match your trained model!)
    TFLITE_INPUT_SIZE = 416   # Usually 416x416 for YOLO
    
    # Piece classes from the TFLite model
    # Index corresponds to class ID in the model
    PIECE_CLASS_NAMES = [
        "black-bishop",   # 0
        "black-king",     # 1
        "black-knight",   # 2
        "black-queen",    # 3
        "black-rook",     # 4
        "white-bishop",   # 5
        "white-king",     # 6
        "white-knight",   # 7
        "white-queen",    # 8
        "white-rook",     # 9
    ]
    
    # Maps model class names to internal names (with underscores)
    PIECE_CLASSES = {
        "black-bishop": "black_bishop",
        "black-king": "black_king",
        "black-knight": "black_knight",
        "black-queen": "black_queen",
        "black-rook": "black_rook",
        "white-bishop": "white_bishop",
        "white-king": "white_king",
        "white-knight": "white_knight",
        "white-queen": "white_queen",
        "white-rook": "white_rook",
    }
    
    # ==================== ROBOT COORDINATE SYSTEM ====================
    # Physical position of board corners in robot frame (METERS)
    # These values MUST match arm_control/config.py!
    # Used for pixel-to-world coordinate transform
    
    # Board origin (top-left corner, marker 0 inner corner)
    BOARD_ORIGIN_X = 0.135   # 15cm in front of robot
    BOARD_ORIGIN_Y = -0.12  # 12cm to the right
    BOARD_ORIGIN_Z = 0.00   # 2cm above base level (board surface height)
    
    # Physical cell size in METERS (must match arm_control/config.py)
    CELL_SIZE_M = 0.03  # 3cm = 30mm per cell
    
    # ==================== DEBUG SETTINGS ====================
    DEBUG_MODE = True
    SHOW_DETECTION_WINDOW = True
    SAVE_DEBUG_IMAGES = False
    DEBUG_OUTPUT_DIR = "debug_output"
