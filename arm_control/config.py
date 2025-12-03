"""
Arm control configuration for TicTacToe robot.
Physical dimensions and calibration values.
"""

import numpy as np


class ArmConfig:
    """
    Configuration for the DOFBOT arm.
    
    IMPORTANT: Measure your actual robot and update these values!
    All distances are in METERS for ikpy compatibility.
    """
    
    # ==================== URDF PATH ====================
    # Path to the DOFBOT URDF file (on the Raspberry Pi)
    URDF_PATH = "/home/dofbot/dobot_ws/src/dofbot_moveit/urdf/dofbot.urdf"
    
    # ==================== DOFBOT LINK LENGTHS (meters) ====================
    # These are approximate - measure your actual robot!
    BASE_HEIGHT = 0.061      # Height of base to first joint
    LINK_1 = 0.0435          # Length of link 1 (shoulder)
    LINK_2 = 0.105           # Length of link 2 (upper arm)
    LINK_3 = 0.098           # Length of link 3 (forearm)
    LINK_4 = 0.0             # Link 4 (wrist pitch - no offset)
    LINK_5 = 0.065           # Length of link 5 (to gripper)
    
    # ==================== SERVO LIMITS ====================
    # Servo angle limits in degrees
    SERVO_LIMITS = {
        1: (0, 180),    # Base rotation
        2: (0, 180),    # Shoulder
        3: (0, 180),    # Elbow
        4: (0, 180),    # Wrist pitch
        5: (0, 270),    # Wrist roll
        6: (0, 180),    # Gripper
    }
    
    # ==================== BOARD POSITION ====================
    # Position of the TicTacToe board relative to robot base
    # (x, y, z) in METERS
    # x = forward from robot
    # y = left/right (positive = left)
    # z = up/down
    
    # Board origin (corner closest to robot)
    # CALIBRATE THESE FOR YOUR SETUP!
    BOARD_ORIGIN_X = 0.15   # 15cm in front of robot
    BOARD_ORIGIN_Y = -0.12  # 12cm to the right
    BOARD_ORIGIN_Z = 0.02   # 2cm above base level
    
    # Board cell size in METERS
    CELL_SIZE = 0.08  # 8cm = 80mm per cell
    
    # ==================== MOVEMENT HEIGHTS ====================
    # All heights relative to board surface (in meters)
    SAFE_HEIGHT = 0.10      # Height for moving between positions
    HOVER_HEIGHT = 0.05     # Height just above piece for picking
    GRASP_HEIGHT = 0.02     # Height when gripping piece
    PLACE_HEIGHT = 0.03     # Height when placing piece
    
    # ==================== PIECE STAGING AREA ====================
    # Where pieces wait before being placed
    # This is where the robot picks up pieces to place on the board
    
    # White pieces staging position
    WHITE_STAGING_X = 0.10
    WHITE_STAGING_Y = 0.15   # Left of robot
    WHITE_STAGING_Z = 0.02
    
    # Black pieces staging position
    BLACK_STAGING_X = 0.10
    BLACK_STAGING_Y = -0.20  # Right of robot
    BLACK_STAGING_Z = 0.02
    
    # Spacing between staged pieces
    STAGING_SPACING = 0.05  # 5cm between pieces
    
    # ==================== MOVEMENT SETTINGS ====================
    MOVE_SPEED_MS = 800     # Default movement time in milliseconds
    GRIPPER_SPEED_MS = 300  # Gripper open/close time
    
    # Delays between movements (seconds)
    MOVE_DELAY = 0.2
    GRIPPER_DELAY = 0.3
    
    # ==================== GRIPPER SETTINGS ====================
    GRIPPER_OPEN_ANGLE = 30     # Servo angle for open
    GRIPPER_CLOSED_ANGLE = 120  # Servo angle for closed (gripping)
    
    # ==================== IK SETTINGS ====================
    # Whether to use ikpy for IK (True) or pre-calibrated positions (False)
    USE_IK = True
    
    # Fallback: pre-calibrated cell positions (servo angles)
    # Format: (row, col): (s1, s2, s3, s4, s5)
    # Only used if USE_IK = False
    CALIBRATED_POSITIONS = {
        (0, 0): (45, 80, 60, 90, 90),
        (0, 1): (90, 80, 60, 90, 90),
        (0, 2): (135, 80, 60, 90, 90),
        (1, 0): (45, 90, 70, 90, 90),
        (1, 1): (90, 90, 70, 90, 90),
        (1, 2): (135, 90, 70, 90, 90),
        (2, 0): (45, 100, 80, 90, 90),
        (2, 1): (90, 100, 80, 90, 90),
        (2, 2): (135, 100, 80, 90, 90),
    }
    
    # Home position (servo angles)
    HOME_POSITION = (90, 90, 90, 90, 90, 90)  # All centered
    
    @classmethod
    def cell_to_xyz(cls, row: int, col: int) -> tuple:
        """
        Convert a cell position to XYZ coordinates.
        
        Args:
            row: Row index (0-2).
            col: Column index (0-2).
            
        Returns:
            (x, y, z) in meters.
        """
        x = cls.BOARD_ORIGIN_X + (row + 0.5) * cls.CELL_SIZE
        y = cls.BOARD_ORIGIN_Y + (col + 0.5) * cls.CELL_SIZE
        z = cls.BOARD_ORIGIN_Z
        
        return x, y, z
    
    @classmethod
    def get_staging_position(cls, color: str, piece_index: int) -> tuple:
        """
        Get the staging position for a piece.
        
        Args:
            color: "white" or "black"
            piece_index: 0-4 (which piece in the sequence)
            
        Returns:
            (x, y, z) in meters.
        """
        if color == "white":
            x = cls.WHITE_STAGING_X + piece_index * cls.STAGING_SPACING
            y = cls.WHITE_STAGING_Y
            z = cls.WHITE_STAGING_Z
        else:
            x = cls.BLACK_STAGING_X + piece_index * cls.STAGING_SPACING
            y = cls.BLACK_STAGING_Y
            z = cls.BLACK_STAGING_Z
        
        return x, y, z
