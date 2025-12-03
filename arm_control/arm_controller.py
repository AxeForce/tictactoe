"""
High-level arm controller for TicTacToe robot.
Handles pick and place operations using IK.
"""

import time
from typing import Optional, List, Tuple
from .config import ArmConfig
from .dofbot_ik import DofbotIK
from .gripper import Gripper


class ArmController:
    """
    High-level controller for the DOFBOT arm.
    
    Provides methods for picking and placing pieces on the TicTacToe board.
    Uses IK to calculate joint angles for target positions.
    """
    
    def __init__(
        self, 
        config: Optional[ArmConfig] = None,
        simulate: bool = False
    ):
        """
        Initialize the arm controller.
        
        Args:
            config: Arm configuration.
            simulate: If True, don't actually control hardware.
        """
        self.config = config or ArmConfig()
        self.simulate = simulate
        
        # Initialize components
        self.ik_solver = DofbotIK(self.config)
        self.gripper = Gripper(self.config, simulate=simulate)
        
        # Hardware connection
        self.arm = None
        if not simulate:
            self._init_arm()
        
        print("ArmController initialized!")
    
    def _init_arm(self):
        """Initialize the Arm_Lib connection."""
        try:
            from Arm_Lib import Arm_Device
            self.arm = Arm_Device()
            print("Arm hardware connected!")
        except ImportError:
            print("WARNING: Arm_Lib not available. Running in simulation mode.")
            self.simulate = True
        except Exception as e:
            print(f"ERROR initializing arm: {e}")
            self.simulate = True
    
    def move_to_angles(self, angles: List[float], speed_ms: int = None) -> bool:
        """
        Move arm to specified joint angles.
        
        Args:
            angles: List of 5 joint angles [s1, s2, s3, s4, s5].
            speed_ms: Movement speed in milliseconds.
            
        Returns:
            True if successful.
        """
        if len(angles) < 5:
            print("ERROR: Need 5 joint angles!")
            return False
        
        speed = speed_ms or self.config.MOVE_SPEED_MS
        
        # Clamp angles to valid range
        clamped = []
        for i, angle in enumerate(angles):
            servo_id = i + 1
            min_a, max_a = self.config.SERVO_LIMITS[servo_id]
            clamped.append(max(min_a, min(max_a, int(angle))))
        
        if self.simulate:
            print(f"[SIM] Moving to angles: {clamped}")
            time.sleep(speed / 1000)
            return True
        
        try:
            # Add current gripper angle
            gripper_angle = self.arm.Arm_serial_servo_read(6) or 90
            
            self.arm.Arm_serial_servo_write6(
                clamped[0],  # Base
                clamped[1],  # Shoulder
                clamped[2],  # Elbow
                clamped[3],  # Wrist pitch
                clamped[4],  # Wrist roll
                int(gripper_angle),  # Gripper (keep current)
                speed
            )
            
            # Wait for movement to complete
            time.sleep(speed / 1000 + self.config.MOVE_DELAY)
            return True
            
        except Exception as e:
            print(f"ERROR moving arm: {e}")
            return False
    
    def move_to_xyz(
        self, 
        x: float, 
        y: float, 
        z: float,
        speed_ms: int = None
    ) -> bool:
        """
        Move arm to XYZ position using IK.
        
        Args:
            x, y, z: Target position in meters.
            speed_ms: Movement speed.
            
        Returns:
            True if successful.
        """
        print(f"Moving to XYZ: ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # Solve IK
        angles = self.ik_solver.solve_ik(x, y, z)
        
        if angles is None:
            print("ERROR: Could not solve IK for target position!")
            return False
        
        if not self.ik_solver.verify_angles(angles):
            print("WARNING: Angles out of range, clamping...")
        
        return self.move_to_angles(angles, speed_ms)
    
    def move_to_cell(
        self, 
        row: int, 
        col: int, 
        height: float = None
    ) -> bool:
        """
        Move arm above a TicTacToe cell.
        
        Args:
            row: Row index (0-2).
            col: Column index (0-2).
            height: Height above board (meters). Uses HOVER_HEIGHT if not specified.
            
        Returns:
            True if successful.
        """
        if height is None:
            height = self.config.HOVER_HEIGHT
        
        print(f"Moving to cell ({row}, {col}) at height {height:.3f}m")
        
        # Get cell position
        x, y, z = self.config.cell_to_xyz(row, col)
        z += height
        
        return self.move_to_xyz(x, y, z)
    
    def move_to_safe_height(self) -> bool:
        """
        Move arm to safe height above current XY position.
        
        Returns:
            True if successful.
        """
        # Just go to home position for safety
        return self.go_home()
    
    def go_home(self) -> bool:
        """
        Move arm to home position.
        
        Returns:
            True if successful.
        """
        print("Going to home position...")
        angles = list(self.config.HOME_POSITION[:5])
        return self.move_to_angles(angles)
    
    def pick_piece_from_cell(self, row: int, col: int) -> bool:
        """
        Pick up a piece from a cell on the board.
        
        Args:
            row: Row index (0-2).
            col: Column index (0-2).
            
        Returns:
            True if successful.
        """
        print(f"\n=== Picking piece from cell ({row}, {col}) ===")
        
        # Step 1: Move to safe height above cell
        if not self.move_to_cell(row, col, self.config.SAFE_HEIGHT):
            return False
        
        # Step 2: Open gripper
        self.gripper.open()
        
        # Step 3: Lower to hover height
        if not self.move_to_cell(row, col, self.config.HOVER_HEIGHT):
            return False
        
        # Step 4: Lower to grasp height
        if not self.move_to_cell(row, col, self.config.GRASP_HEIGHT):
            return False
        
        # Step 5: Close gripper
        self.gripper.close()
        
        # Step 6: Lift to safe height
        if not self.move_to_cell(row, col, self.config.SAFE_HEIGHT):
            return False
        
        print(f"=== Piece picked from ({row}, {col}) ===\n")
        return True
    
    def place_piece_on_cell(self, row: int, col: int) -> bool:
        """
        Place a piece on a cell.
        
        Args:
            row: Row index (0-2).
            col: Column index (0-2).
            
        Returns:
            True if successful.
        """
        print(f"\n=== Placing piece on cell ({row}, {col}) ===")
        
        # Step 1: Move to safe height above cell
        if not self.move_to_cell(row, col, self.config.SAFE_HEIGHT):
            return False
        
        # Step 2: Lower to place height
        if not self.move_to_cell(row, col, self.config.PLACE_HEIGHT):
            return False
        
        # Step 3: Open gripper to release piece
        self.gripper.open()
        
        # Step 4: Lift to safe height
        if not self.move_to_cell(row, col, self.config.SAFE_HEIGHT):
            return False
        
        print(f"=== Piece placed on ({row}, {col}) ===\n")
        return True
    
    def pick_from_staging(self, color: str, piece_index: int) -> bool:
        """
        Pick a piece from the staging area.
        
        Args:
            color: "white" or "black"
            piece_index: 0-4 (which piece in the sequence)
            
        Returns:
            True if successful.
        """
        print(f"\n=== Picking {color} piece #{piece_index} from staging ===")
        
        # Get staging position
        x, y, z = self.config.get_staging_position(color, piece_index)
        
        # Step 1: Move to safe height above staging
        if not self.move_to_xyz(x, y, z + self.config.SAFE_HEIGHT):
            return False
        
        # Step 2: Open gripper
        self.gripper.open()
        
        # Step 3: Lower to grasp
        if not self.move_to_xyz(x, y, z + self.config.GRASP_HEIGHT):
            return False
        
        # Step 4: Close gripper
        self.gripper.close()
        
        # Step 5: Lift to safe height
        if not self.move_to_xyz(x, y, z + self.config.SAFE_HEIGHT):
            return False
        
        print(f"=== Picked {color} piece #{piece_index} ===\n")
        return True
    
    def execute_robot_move(
        self, 
        color: str, 
        piece_index: int, 
        target_row: int, 
        target_col: int
    ) -> bool:
        """
        Execute a complete robot move: pick from staging and place on board.
        
        Args:
            color: "white" or "black"
            piece_index: 0-4 (which piece)
            target_row: Target cell row.
            target_col: Target cell column.
            
        Returns:
            True if successful.
        """
        print(f"\n{'='*50}")
        print(f"Executing move: {color} piece #{piece_index} to ({target_row}, {target_col})")
        print(f"{'='*50}")
        
        # Pick from staging
        if not self.pick_from_staging(color, piece_index):
            print("ERROR: Failed to pick from staging!")
            return False
        
        # Place on board
        if not self.place_piece_on_cell(target_row, target_col):
            print("ERROR: Failed to place on board!")
            return False
        
        # Go home
        self.go_home()
        
        print(f"{'='*50}")
        print("Move completed successfully!")
        print(f"{'='*50}\n")
        
        return True


# Quick test
if __name__ == "__main__":
    print("Testing ArmController (simulation mode)...")
    
    controller = ArmController(simulate=True)
    
    # Test going home
    controller.go_home()
    
    # Test moving to cells
    for row in range(3):
        for col in range(3):
            controller.move_to_cell(row, col)
    
    # Test pick and place
    controller.pick_piece_from_cell(1, 1)
    controller.place_piece_on_cell(0, 0)
    
    # Test full move
    controller.execute_robot_move("white", 0, 1, 1)
    
    print("\nArmController test done!")
