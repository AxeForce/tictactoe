"""
Gripper control for DOFBOT.
Handles opening and closing the gripper.
"""

import time
from typing import Optional
from .config import ArmConfig


class Gripper:
    """
    Controls the DOFBOT gripper (servo 6).
    
    The gripper is a simple open/close mechanism.
    """
    
    def __init__(self, config: Optional[ArmConfig] = None, simulate: bool = False):
        """
        Initialize the gripper.
        
        Args:
            config: Arm configuration.
            simulate: If True, don't actually control hardware.
        """
        self.config = config or ArmConfig()
        self.simulate = simulate
        self.arm = None
        self.is_open = True
        
        if not simulate:
            self._init_arm()
    
    def _init_arm(self):
        """Initialize the Arm_Lib connection."""
        try:
            from Arm_Lib import Arm_Device
            self.arm = Arm_Device()
            print("Gripper initialized!")
        except ImportError:
            print("WARNING: Arm_Lib not available. Running in simulation mode.")
            self.simulate = True
        except Exception as e:
            print(f"ERROR initializing gripper: {e}")
            self.simulate = True
    
    def open(self) -> bool:
        """
        Open the gripper.
        
        Returns:
            True if successful.
        """
        print("Opening gripper...")
        
        if self.simulate:
            time.sleep(self.config.GRIPPER_SPEED_MS / 1000)
            self.is_open = True
            return True
        
        try:
            self.arm.Arm_serial_servo_write(
                6,  # Gripper servo
                self.config.GRIPPER_OPEN_ANGLE,
                self.config.GRIPPER_SPEED_MS
            )
            time.sleep(self.config.GRIPPER_DELAY)
            self.is_open = True
            return True
        except Exception as e:
            print(f"ERROR opening gripper: {e}")
            return False
    
    def close(self) -> bool:
        """
        Close the gripper.
        
        Returns:
            True if successful.
        """
        print("Closing gripper...")
        
        if self.simulate:
            time.sleep(self.config.GRIPPER_SPEED_MS / 1000)
            self.is_open = False
            return True
        
        try:
            self.arm.Arm_serial_servo_write(
                6,  # Gripper servo
                self.config.GRIPPER_CLOSED_ANGLE,
                self.config.GRIPPER_SPEED_MS
            )
            time.sleep(self.config.GRIPPER_DELAY)
            self.is_open = False
            return True
        except Exception as e:
            print(f"ERROR closing gripper: {e}")
            return False
    
    def set_angle(self, angle: int) -> bool:
        """
        Set gripper to a specific angle.
        
        Args:
            angle: Servo angle (0-180).
            
        Returns:
            True if successful.
        """
        if self.simulate:
            print(f"[SIM] Gripper angle: {angle}")
            return True
        
        try:
            angle = max(0, min(180, angle))
            self.arm.Arm_serial_servo_write(
                6,
                angle,
                self.config.GRIPPER_SPEED_MS
            )
            time.sleep(self.config.GRIPPER_DELAY)
            return True
        except Exception as e:
            print(f"ERROR setting gripper angle: {e}")
            return False


# Quick test
if __name__ == "__main__":
    print("Testing Gripper (simulation mode)...")
    
    gripper = Gripper(simulate=True)
    
    print(f"Initial state: is_open={gripper.is_open}")
    
    gripper.close()
    print(f"After close: is_open={gripper.is_open}")
    
    gripper.open()
    print(f"After open: is_open={gripper.is_open}")
    
    print("\nGripper test done!")
