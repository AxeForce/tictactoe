"""
Arm calibration helper for TicTacToe robot.

This script helps you calibrate the arm positions for your setup.
Run this on the Raspberry Pi with the DOFBOT connected.

Usage:
    python calibrate_arm.py

Controls:
    - Number keys (1-6): Select servo to control
    - Arrow Up/Down: Increase/decrease selected servo angle
    - 'h': Go to home position
    - 's': Save current position
    - 'p': Print current angles
    - 'c': Calibrate a cell (enter row, col)
    - 'q': Quit
"""

import time
from typing import Optional, List, Dict, Tuple


class ArmCalibrator:
    """
    Interactive arm calibration tool.
    
    Allows you to manually position the arm and record positions
    for TicTacToe cells and staging areas.
    """
    
    def __init__(self):
        """Initialize the calibrator."""
        self.arm = None
        self.simulate = False
        self.current_angles: List[int] = [90, 90, 90, 90, 90, 90]  # 6 servos
        self.selected_servo = 1  # Currently selected servo (1-6)
        self.step_size = 5  # Degrees per step
        
        self.calibrated_positions: Dict[str, List[int]] = {}
        
        self._init_arm()
    
    def _init_arm(self):
        """Initialize the arm."""
        try:
            # Try to import Arm_Lib - this is hardware-specific for DOFBOT
            # Note: Arm_Lib is pre-installed on DOFBOT Raspberry Pi image
            # It's not available via pip and will cause ImportError on other systems
            from Arm_Lib import Arm_Device
            self.arm = Arm_Device()
            print("Arm hardware connected!")
            
            # Read current positions from hardware
            try:
                for i in range(6):
                    angle = self.arm.Arm_serial_servo_read(i + 1)
                    if angle is not None and isinstance(angle, (int, float)):
                        self.current_angles[i] = int(angle)
                
                print(f"Current angles from hardware: {self.current_angles}")
            except Exception as read_error:
                print(f"Warning: Could not read current angles: {read_error}")
                print("Using default angles: [90, 90, 90, 90, 90, 90]")
            
        except ImportError:
            print("Arm_Lib not available. Running in simulation mode.")
            print("Note: Arm_Lib is pre-installed on DOFBOT Raspberry Pi image.")
            print("To use hardware mode, run this on the DOFBOT Raspberry Pi.")
            self.simulate = True
        except Exception as e:
            print(f"Error connecting to arm hardware: {e}")
            print("Falling back to simulation mode.")
            self.simulate = True
    
    def move_servo(self, servo_id: int, angle: int, speed_ms: int = 500) -> bool:
        """
        Move a single servo.
        
        Args:
            servo_id: Servo number (1-6)
            angle: Target angle in degrees (0-180)
            speed_ms: Movement speed in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        if not 1 <= servo_id <= 6:
            print(f"Error: Invalid servo_id {servo_id}. Must be 1-6.")
            return False
            
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        self.current_angles[servo_id - 1] = angle
        
        if self.simulate:
            print(f"[SIM] Servo {servo_id} -> {angle}°")
            return True
        
        try:
            self.arm.Arm_serial_servo_write(servo_id, angle, speed_ms)
            time.sleep(speed_ms / 1000 + 0.1)
            return True
        except Exception as e:
            print(f"Error moving servo {servo_id}: {e}")
            return False
    
    def move_all(self, angles: List[int], speed_ms: int = 800) -> bool:
        """
        Move all servos.
        
        Args:
            angles: List of 6 angles for servos 1-6
            speed_ms: Movement speed in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        if len(angles) != 6:
            print(f"Error: Need 6 angles, got {len(angles)}")
            return False
            
        # Clamp all angles to valid range
        for i, angle in enumerate(angles):
            self.current_angles[i] = max(0, min(180, int(angle)))
        
        if self.simulate:
            print(f"[SIM] All servos -> {self.current_angles}")
            return True
        
        try:
            self.arm.Arm_serial_servo_write6(*self.current_angles, speed_ms)
            time.sleep(speed_ms / 1000 + 0.2)
            return True
        except Exception as e:
            print(f"Error moving all servos: {e}")
            return False
    
    def go_home(self) -> bool:
        """
        Move to home position.
        
        Returns:
            True if successful, False otherwise
        """
        print("Going to home position...")
        return self.move_all([90, 90, 90, 90, 90, 90])
    
    def print_angles(self) -> None:
        """Print current servo angles."""
        print("\nCurrent Angles:")
        labels = ["Base", "Shoulder", "Elbow", "Wrist Pitch", "Wrist Roll", "Gripper"]
        for i, (label, angle) in enumerate(zip(labels, self.current_angles)):
            marker = " <--" if i + 1 == self.selected_servo else ""
            print(f"  Servo {i+1} ({label}): {angle}°{marker}")
    
    def save_position(self, name: str) -> None:
        """
        Save current position with a name.
        
        Args:
            name: Name/identifier for the position
        """
        if not name.strip():
            print("Error: Position name cannot be empty")
            return
            
        self.calibrated_positions[name] = list(self.current_angles[:5])  # Exclude gripper
        print(f"Saved position '{name}': {self.current_angles[:5]}")
    
    def calibrate_cell(self, row: int, col: int) -> None:
        """
        Interactively calibrate a cell position.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
        """
        if not (0 <= row <= 2 and 0 <= col <= 2):
            print("Error: Row and col must be between 0 and 2")
            return
            
        print(f"\n=== Calibrating Cell ({row}, {col}) ===")
        print("Move the gripper to the center of the cell.")
        print("Press Enter when positioned correctly.")
        
        try:
            input("Press Enter to save position...")
        except KeyboardInterrupt:
            print("\nCalibration cancelled.")
            return
        
        name = f"cell_{row}_{col}"
        self.save_position(name)
        
        print(f"Cell ({row}, {col}) calibrated!")
    
    def run(self):
        """Run the interactive calibration."""
        print("\n" + "="*60)
        print("   DOFBOT Arm Calibration Tool")
        print("="*60)
        print("\nControls:")
        print("  1-6    : Select servo")
        print("  w/s    : Increase/decrease angle")
        print("  W/S    : Large increase/decrease")
        print("  h      : Go home")
        print("  p      : Print angles")
        print("  c      : Calibrate a cell")
        print("  g      : Toggle gripper")
        print("  o      : Generate output code")
        print("  q      : Quit")
        print("="*60 + "\n")
        
        self.print_angles()
        
        while True:
            try:
                cmd = input("\nCommand: ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'h':
                    self.go_home()
                    self.print_angles()
                elif cmd == 'p':
                    self.print_angles()
                elif cmd in ['1', '2', '3', '4', '5', '6']:
                    self.selected_servo = int(cmd)
                    print(f"Selected servo {self.selected_servo}")
                    self.print_angles()
                elif cmd == 'w':
                    new_angle = self.current_angles[self.selected_servo - 1] + self.step_size
                    self.move_servo(self.selected_servo, new_angle)
                    print(f"Servo {self.selected_servo} = {self.current_angles[self.selected_servo - 1]}°")
                elif cmd == 's':
                    new_angle = self.current_angles[self.selected_servo - 1] - self.step_size
                    self.move_servo(self.selected_servo, new_angle)
                    print(f"Servo {self.selected_servo} = {self.current_angles[self.selected_servo - 1]}°")
                elif cmd == 'W':  # Large step
                    new_angle = self.current_angles[self.selected_servo - 1] + 20
                    self.move_servo(self.selected_servo, new_angle)
                    print(f"Servo {self.selected_servo} = {self.current_angles[self.selected_servo - 1]}°")
                elif cmd == 'S':  # Large step
                    new_angle = self.current_angles[self.selected_servo - 1] - 20
                    self.move_servo(self.selected_servo, new_angle)
                    print(f"Servo {self.selected_servo} = {self.current_angles[self.selected_servo - 1]}°")
                elif cmd == 'c':
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter col (0-2): "))
                    self.calibrate_cell(row, col)
                elif cmd == 'g':
                    # Toggle gripper
                    if self.current_angles[5] < 90:
                        self.move_servo(6, 120)  # Close
                    else:
                        self.move_servo(6, 30)   # Open
                    print(f"Gripper = {self.current_angles[5]}°")
                elif cmd == 'save':
                    name = input("Enter position name: ")
                    self.save_position(name)
                elif cmd == 'o':
                    self._generate_output()
                elif cmd == '':
                    pass  # Just pressed enter
                else:
                    print(f"Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nCalibration complete!")
        self._generate_output()
    
    def _generate_output(self) -> None:
        """Generate Python code for calibrated positions."""
        print("\n" + "="*60)
        print("   Generated Configuration Code")
        print("="*60)
        print("\n# Copy this to arm_control/config.py\n")
        print("CALIBRATED_POSITIONS = {")
        
        # Sort cell positions by row, then col
        cell_positions = {}
        for name, angles in self.calibrated_positions.items():
            if name.startswith("cell_"):
                try:
                    parts = name.split("_")
                    row, col = int(parts[1]), int(parts[2])
                    cell_positions[(row, col)] = tuple(int(a) for a in angles)
                except (ValueError, IndexError):
                    print(f"    # Warning: Invalid cell name format: {name}")
        
        # Print cell positions in order
        for (row, col), angles in sorted(cell_positions.items()):
            print(f"    ({row}, {col}): {angles},")
        
        print("}")
        
        # Print staging positions if calibrated
        staging = {k: v for k, v in self.calibrated_positions.items() if "staging" in k}
        if staging:
            print("\n# Staging positions")
            for name, angles in staging.items():
                print(f"# {name}: {tuple(int(a) for a in angles)}")
        
        # Print other positions
        other = {k: v for k, v in self.calibrated_positions.items()
                if not k.startswith("cell_") and "staging" not in k}
        if other:
            print("\n# Other positions")
            for name, angles in other.items():
                print(f"# {name}: {tuple(int(a) for a in angles)}")
        
        print("\n" + "="*60)


def main():
    """Main entry point."""
    calibrator = ArmCalibrator()
    calibrator.run()


if __name__ == "__main__":
    main()
