"""
Test script: Vision-guided arm movement.
Detects a piece using the vision system and moves the arm above it.

Usage:
    python test_vision_arm.py              # Run with hardware
    python test_vision_arm.py --simulate   # Simulate arm (no hardware)
"""

import cv2
import numpy as np
import time
import argparse
from typing import Optional

# Vision imports
from vision import VisionConfig, Camera, BoardDetector, PieceDetector

# Arm control imports
from arm_control import ArmConfig, ArmController, DofbotIK


class VisionArmTest:
    """
    Test class for vision-guided arm movement.
    """
    
    def __init__(self, simulate: bool = False):
        """
        Initialize vision and arm components.
        
        Args:
            simulate: If True, simulate arm movement (no hardware).
        """
        self.simulate = simulate
        
        print("=" * 60)
        print("Vision-Guided Arm Movement Test")
        print("=" * 60)
        
        # Initialize vision components
        print("\n[1/4] Initializing vision config...")
        self.vision_config = VisionConfig()
        
        print("[2/4] Initializing camera...")
        self.camera = Camera(self.vision_config)
        
        print("[3/4] Initializing board detector...")
        self.board_detector = BoardDetector(self.vision_config)
        
        print("[4/4] Initializing piece detector...")
        self.piece_detector = PieceDetector(self.vision_config)
        
        # Initialize arm components
        print("\n[5/5] Initializing arm controller...")
        self.arm_config = ArmConfig()
        self.arm_controller = ArmController(self.arm_config, simulate=simulate)
        
        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60)
    
    def detect_pieces(self, timeout: float = 10.0) -> Optional[list]:
        """
        Detect pieces using the vision system.
        
        Args:
            timeout: Maximum time to wait for detection (seconds).
            
        Returns:
            List of detections with world coordinates, or None.
        """
        print(f"\nWaiting for piece detection (timeout: {timeout}s)...")
        
        if not self.camera.open():
            print("ERROR: Could not open camera!")
            return None
        
        start_time = time.time()
        detections_with_coords = []
        
        try:
            while time.time() - start_time < timeout:
                frame = self.camera.read()
                if frame is None:
                    continue
                
                # Detect board
                board_detection = self.board_detector.detect(frame)
                
                if not board_detection.success:
                    # Show frame without board
                    cv2.putText(frame, "Looking for board...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Detect pieces on raw frame
                raw_detections = self.piece_detector.detect_raw(frame)
                
                if not raw_detections:
                    cv2.putText(frame, "Board found, looking for pieces...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow("Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Map detections with world coordinates
                board_state = self.piece_detector.map_to_board(
                    raw_detections,
                    board_detection.transform_matrix,
                    board_detection.pixel_to_world_matrix
                )
                
                # Filter detections with valid world coordinates
                for det in board_state.detections:
                    if det.world_coords:
                        detections_with_coords.append(det)
                
                if detections_with_coords:
                    # Draw detections on frame
                    for det in raw_detections:
                        x1, y1, x2, y2 = det.bbox
                        color = (0, 255, 0) if det.color == "white" else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, det.piece_name, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.putText(frame, f"Found {len(detections_with_coords)} pieces!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Vision", frame)
                    cv2.waitKey(500)  # Show for a moment
                    break
                
                cv2.imshow("Vision", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cv2.destroyAllWindows()
        
        return detections_with_coords if detections_with_coords else None
    
    def move_above_piece(self, detection) -> bool:
        """
        Move the arm above a detected piece.
        
        Args:
            detection: PieceDetection with world_coords.
            
        Returns:
            True if successful.
        """
        if not detection.world_coords:
            print("ERROR: Detection has no world coordinates!")
            return False
        
        x, y, z = detection.world_coords
        
        # Add hover height above the piece
        hover_z = z + self.arm_config.HOVER_HEIGHT
        
        print(f"\nMoving arm above {detection.piece_type}:")
        print(f"  World position: ({x:.3f}, {y:.3f}, {z:.3f}) m")
        print(f"  Hover height: {hover_z:.3f} m")
        
        # Move to position
        success = self.arm_controller.move_to_xyz(x, y, hover_z)
        
        if success:
            print("  [OK] Arm moved successfully!")
        else:
            print("  [FAIL] Could not move arm to position!")
        
        return success
    
    def run_interactive(self):
        """
        Run interactive test - detect pieces and move arm on keypress.
        """
        print("\n" + "=" * 60)
        print("Interactive Mode")
        print("=" * 60)
        print("Controls:")
        print("  SPACE - Detect pieces and show options")
        print("  1-9   - Move arm above piece #N")
        print("  H     - Move arm to home position")
        print("  Q     - Quit")
        print("=" * 60)
        
        if not self.camera.open():
            print("ERROR: Could not open camera!")
            return
        
        current_detections = []
        
        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue
                
                # Detect board
                board_detection = self.board_detector.detect(frame)
                
                # Draw frame
                debug_frame = frame.copy()
                
                if board_detection.success:
                    # Draw board outline
                    corners = board_detection.corners.astype(int)
                    for i in range(4):
                        pt1 = tuple(corners[i])
                        pt2 = tuple(corners[(i + 1) % 4])
                        cv2.line(debug_frame, pt1, pt2, (0, 255, 0), 2)
                    
                    # Detect pieces
                    raw_detections = self.piece_detector.detect_raw(frame)
                    
                    # Map with world coords
                    board_state = self.piece_detector.map_to_board(
                        raw_detections,
                        board_detection.transform_matrix,
                        board_detection.pixel_to_world_matrix
                    )
                    
                    # Update current detections
                    current_detections = [d for d in board_state.detections if d.world_coords]
                    
                    # Draw detections with numbers
                    for i, det in enumerate(raw_detections):
                        x1, y1, x2, y2 = det.bbox
                        color = (0, 255, 0) if det.color == "white" else (255, 0, 0)
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Find corresponding mapped detection
                        for j, mapped in enumerate(current_detections):
                            if mapped.piece_type == det.piece_type and mapped.confidence == det.confidence:
                                cv2.putText(debug_frame, f"[{j+1}] {det.piece_name}", (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                break
                        else:
                            cv2.putText(debug_frame, det.piece_name, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.putText(debug_frame, f"Pieces: {len(current_detections)} | Press 1-{len(current_detections)} to move",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(debug_frame, "No board detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    current_detections = []
                
                cv2.imshow("Vision-Arm Test", debug_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                
                elif key == ord('h'):
                    print("\nMoving to home position...")
                    self.arm_controller.move_to_angles(list(self.arm_config.HOME_POSITION[:5]))
                
                elif ord('1') <= key <= ord('9'):
                    piece_num = key - ord('0')
                    if 1 <= piece_num <= len(current_detections):
                        det = current_detections[piece_num - 1]
                        print(f"\nSelected piece {piece_num}: {det.piece_type}")
                        self.move_above_piece(det)
                    else:
                        print(f"\nNo piece #{piece_num} available")
        
        finally:
            self.camera.close()
            cv2.destroyAllWindows()
    
    def run_auto(self, piece_color: str = None):
        """
        Automatic mode - detect first piece and move arm above it.
        
        Args:
            piece_color: Optional color filter ("white" or "black").
        """
        print("\n" + "=" * 60)
        print("Automatic Mode")
        if piece_color:
            print(f"Looking for: {piece_color} pieces")
        print("=" * 60)
        
        # Detect pieces
        detections = self.detect_pieces(timeout=10.0)
        
        if not detections:
            print("\nNo pieces detected!")
            return
        
        # Filter by color if specified
        if piece_color:
            detections = [d for d in detections if d.color == piece_color]
            if not detections:
                print(f"\nNo {piece_color} pieces found!")
                return
        
        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        # Print all detections
        print(f"\nFound {len(detections)} pieces:")
        for i, det in enumerate(detections):
            x, y, z = det.world_coords
            print(f"  {i+1}. {det.piece_type} ({det.confidence:.2f}) at ({x:.3f}, {y:.3f}, {z:.3f})m")
        
        # Move to first piece
        target = detections[0]
        print(f"\nMoving to highest confidence piece: {target.piece_type}")
        
        # First move to safe height
        print("\nStep 1: Moving to safe height...")
        self.arm_controller.move_to_angles(list(self.arm_config.HOME_POSITION[:5]))
        time.sleep(0.5)
        
        # Then move above piece
        print("Step 2: Moving above piece...")
        success = self.move_above_piece(target)
        
        if success:
            print("\n" + "=" * 60)
            print("SUCCESS! Arm is now above the piece.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("FAILED to move arm above piece.")
            print("=" * 60)
    
    def cleanup(self):
        """Clean up resources."""
        self.camera.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Vision-guided arm movement test")
    parser.add_argument("--simulate", action="store_true", 
                        help="Simulate arm movement (no hardware)")
    parser.add_argument("--auto", action="store_true",
                        help="Automatic mode - detect and move to first piece")
    parser.add_argument("--color", choices=["white", "black"],
                        help="Filter pieces by color (auto mode only)")
    
    args = parser.parse_args()
    
    test = VisionArmTest(simulate=args.simulate)
    
    try:
        if args.auto:
            test.run_auto(piece_color=args.color)
        else:
            test.run_interactive()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        test.cleanup()


if __name__ == "__main__":
    main()
