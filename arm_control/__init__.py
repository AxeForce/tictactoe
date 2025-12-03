"""
Arm control module for TicTacToe robot.
Handles robot arm movement using inverse kinematics.
"""

from .config import ArmConfig
from .dofbot_ik import DofbotIK
from .gripper import Gripper
from .arm_controller import ArmController
