"""
Inverse Kinematics solver for DOFBOT using ikpy.
Converts XYZ coordinates to joint angles.
"""

import numpy as np
from typing import Optional, List, Tuple
from .config import ArmConfig


class DofbotIK:
    """
    Inverse Kinematics solver for the Yahboom DOFBOT.
    
    Uses ikpy library to solve IK from the URDF file.
    Falls back to analytical IK if ikpy is not available.
    """
    
    def __init__(self, config: Optional[ArmConfig] = None):
        """
        Initialize the IK solver.
        
        Args:
            config: Arm configuration.
        """
        self.config = config or ArmConfig()
        self.chain = None
        self.use_ikpy = False
        
        # Try to load ikpy and the URDF
        self._try_load_ikpy()
        
        if not self.use_ikpy:
            print("Using analytical IK fallback...")
    
    def _try_load_ikpy(self):
        """Try to load ikpy and create the kinematic chain."""
        try:
            import ikpy.chain
            
            # Try to load from URDF
            try:
                print(f"Loading URDF from {self.config.URDF_PATH}...")
                self.chain = ikpy.chain.Chain.from_urdf_file(
                    self.config.URDF_PATH,
                    # Only include active joints (not base frame or gripper)
                    active_links_mask=[False, True, True, True, True, True, False]
                )
                self.use_ikpy = True
                print("URDF loaded successfully!")
                return
            except FileNotFoundError:
                print(f"URDF file not found at {self.config.URDF_PATH}")
            except Exception as e:
                print(f"Error loading URDF: {e}")
            
            # Create chain manually if URDF not available
            print("Creating kinematic chain manually...")
            import ikpy.link as link
            
            self.chain = ikpy.chain.Chain(
                name="dofbot",
                links=[
                    link.OriginLink(),
                    link.URDFLink(
                        name="base",
                        origin_translation=[0, 0, self.config.BASE_HEIGHT],
                        origin_orientation=[0, 0, 0],
                        rotation=[0, 0, 1],  # Z rotation
                    ),
                    link.URDFLink(
                        name="shoulder",
                        origin_translation=[0, 0, self.config.LINK_1],
                        origin_orientation=[0, 0, 0],
                        rotation=[0, 1, 0],  # Y rotation
                    ),
                    link.URDFLink(
                        name="elbow",
                        origin_translation=[self.config.LINK_2, 0, 0],
                        origin_orientation=[0, 0, 0],
                        rotation=[0, 1, 0],  # Y rotation
                    ),
                    link.URDFLink(
                        name="wrist_pitch",
                        origin_translation=[self.config.LINK_3, 0, 0],
                        origin_orientation=[0, 0, 0],
                        rotation=[0, 1, 0],  # Y rotation
                    ),
                    link.URDFLink(
                        name="wrist_roll",
                        origin_translation=[0, 0, 0],
                        origin_orientation=[0, 0, 0],
                        rotation=[0, 0, 1],  # Z rotation
                    ),
                    link.URDFLink(
                        name="end_effector",
                        origin_translation=[self.config.LINK_5, 0, 0],
                        origin_orientation=[0, 0, 0],
                        rotation=[0, 0, 0],  # No rotation (fixed)
                    ),
                ],
                active_links_mask=[False, True, True, True, True, True, False]
            )
            self.use_ikpy = True
            print("Kinematic chain created!")
            
        except ImportError:
            print("ikpy not installed. Install with: pip install ikpy")
            self.use_ikpy = False
    
    def solve_ik(
        self, 
        x: float, 
        y: float, 
        z: float,
        initial_angles: Optional[List[float]] = None
    ) -> Optional[List[float]]:
        """
        Solve inverse kinematics for a target position.
        
        Args:
            x, y, z: Target position in meters.
            initial_angles: Starting angles for IK solver (radians).
            
        Returns:
            List of joint angles in degrees [s1, s2, s3, s4, s5],
            or None if no solution found.
        """
        target_position = [x, y, z]
        
        if self.use_ikpy and self.chain is not None:
            return self._solve_ikpy(target_position, initial_angles)
        else:
            return self._solve_analytical(x, y, z)
    
    def _solve_ikpy(
        self, 
        target_position: List[float],
        initial_angles: Optional[List[float]] = None
    ) -> Optional[List[float]]:
        """Solve IK using ikpy."""
        try:
            # Use initial angles or zeros
            if initial_angles is None:
                initial_angles = [0] * len(self.chain.links)
            
            # Solve IK
            angles_rad = self.chain.inverse_kinematics(
                target_position,
                initial_position=initial_angles,
                orientation_mode=None  # Position only
            )
            
            # Convert to degrees and extract active joints
            # Skip first (origin) and last (end effector)
            angles_deg = []
            for i, angle in enumerate(angles_rad[1:6]):
                deg = np.rad2deg(angle)
                # Convert to servo range (0-180, centered at 90)
                servo_angle = deg + 90
                # Clamp to valid range
                servo_angle = max(0, min(180, servo_angle))
                angles_deg.append(servo_angle)
            
            return angles_deg
            
        except Exception as e:
            print(f"IK solve failed: {e}")
            return None
    
    def _solve_analytical(
        self, 
        x: float, 
        y: float, 
        z: float
    ) -> Optional[List[float]]:
        """
        Solve IK using simple analytical geometry.
        This is a simplified 2D solution for the arm plane.
        """
        try:
            # Link lengths (convert to mm for easier math)
            L1 = self.config.LINK_2 * 1000  # Upper arm
            L2 = self.config.LINK_3 * 1000  # Forearm
            L3 = self.config.LINK_5 * 1000  # Wrist to gripper
            
            # Convert target to mm
            x_mm = x * 1000
            y_mm = y * 1000
            z_mm = z * 1000 - self.config.BASE_HEIGHT * 1000
            
            # Base rotation (joint 1)
            theta1 = np.degrees(np.arctan2(y_mm, x_mm))
            
            # Distance in horizontal plane
            r = np.sqrt(x_mm**2 + y_mm**2)
            
            # Reach to wrist (ignoring end effector for now)
            r_wrist = r - L3  # Approximate
            
            # Distance from shoulder to wrist
            d = np.sqrt(r_wrist**2 + z_mm**2)
            
            # Check reachability
            if d > L1 + L2:
                print(f"Target ({x:.3f}, {y:.3f}, {z:.3f}) out of reach!")
                return None
            
            # Elbow angle (joint 3) using law of cosines
            cos_theta3 = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
            cos_theta3 = max(-1, min(1, cos_theta3))
            theta3 = np.degrees(np.arccos(cos_theta3))
            
            # Shoulder angle (joint 2)
            alpha = np.degrees(np.arctan2(z_mm, r_wrist))
            cos_beta = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
            cos_beta = max(-1, min(1, cos_beta))
            beta = np.degrees(np.arccos(cos_beta))
            theta2 = alpha + beta
            
            # Wrist pitch (joint 4) - keep gripper level
            theta4 = 180 - theta2 - theta3
            
            # Wrist roll (joint 5) - keep at center
            theta5 = 90
            
            # Convert to servo angles (centered at 90)
            s1 = 90 + theta1
            s2 = 90 + (90 - theta2)  # Adjust for servo orientation
            s3 = 90 + (90 - theta3)
            s4 = 90 + theta4
            s5 = theta5
            
            # Clamp to valid servo range
            angles = [s1, s2, s3, s4, s5]
            angles = [max(0, min(180, a)) for a in angles]
            
            return angles
            
        except Exception as e:
            print(f"Analytical IK failed: {e}")
            return None
    
    def solve_for_cell(self, row: int, col: int, height_offset: float = 0) -> Optional[List[float]]:
        """
        Solve IK for a TicTacToe cell position.
        
        Args:
            row: Row index (0-2).
            col: Column index (0-2).
            height_offset: Additional height above the board (meters).
            
        Returns:
            Joint angles in degrees, or None if no solution.
        """
        x, y, z = self.config.cell_to_xyz(row, col)
        z += height_offset
        
        return self.solve_ik(x, y, z)
    
    def verify_angles(self, angles: List[float]) -> bool:
        """
        Verify that angles are within servo limits.
        
        Args:
            angles: List of 5 joint angles.
            
        Returns:
            True if all angles are valid.
        """
        if len(angles) < 5:
            return False
        
        for i, angle in enumerate(angles):
            servo_id = i + 1
            min_angle, max_angle = self.config.SERVO_LIMITS[servo_id]
            if not (min_angle <= angle <= max_angle):
                print(f"Servo {servo_id} angle {angle:.1f} out of range [{min_angle}, {max_angle}]")
                return False
        
        return True


# Quick test
if __name__ == "__main__":
    print("Testing DofbotIK...")
    
    ik = DofbotIK()
    
    # Test solving for center cell
    print("\nSolving for center cell (1, 1):")
    angles = ik.solve_for_cell(1, 1, height_offset=0.05)
    if angles:
        print(f"Angles: {[f'{a:.1f}' for a in angles]}")
        print(f"Valid: {ik.verify_angles(angles)}")
    
    # Test all cells
    print("\nSolving for all cells:")
    for row in range(3):
        for col in range(3):
            angles = ik.solve_for_cell(row, col, height_offset=0.05)
            if angles:
                print(f"Cell ({row},{col}): {[f'{a:.1f}' for a in angles]}")
            else:
                print(f"Cell ({row},{col}): NO SOLUTION")
    
    print("\nDofbotIK test done!")
