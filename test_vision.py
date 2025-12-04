"""
Test script for the vision module.
Tests camera, board detection, and piece detection components.

Usage:
    python test_vision.py              # Run all tests
    python test_vision.py --camera     # Test camera only
    python test_vision.py --board      # Test board detection only
    python test_vision.py --piece      # Test piece detection only
    python test_vision.py --live       # Live camera test with all components
    python test_vision.py --image PATH # Test with a static image
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from vision import VisionConfig, Camera, BoardDetector, PieceDetector
from vision.board_detector import BoardDetection
from vision.piece_detector import BoardState


def create_tictactoe_grid(board_state: BoardState = None, size: int = 400) -> np.ndarray:
    """
    Create a visual TicTacToe grid with X and O markers.
    
    Args:
        board_state: Current board state with grid of pieces.
        size: Size of the output image (square).
        
    Returns:
        BGR image of the TicTacToe grid.
    """
    # Create white background
    grid_img = np.ones((size, size, 3), dtype=np.uint8) * 255
    
    cell_size = size // 3
    line_thickness = 3
    
    # Draw grid lines (black)
    for i in range(1, 3):
        # Vertical lines
        x = i * cell_size
        cv2.line(grid_img, (x, 0), (x, size), (0, 0, 0), line_thickness)
        # Horizontal lines
        y = i * cell_size
        cv2.line(grid_img, (0, y), (size, y), (0, 0, 0), line_thickness)
    
    # Draw border
    cv2.rectangle(grid_img, (0, 0), (size-1, size-1), (0, 0, 0), line_thickness)
    
    # Draw X and O markers based on board state
    if board_state and board_state.grid:
        for row in range(3):
            for col in range(3):
                piece = board_state.grid[row][col]
                if piece is None:
                    continue
                
                # Calculate cell center
                cx = col * cell_size + cell_size // 2
                cy = row * cell_size + cell_size // 2
                
                # Determine if X (white pieces) or O (black pieces)
                is_white = "white" in piece.lower()
                
                # Draw marker
                margin = cell_size // 5
                marker_size = cell_size // 2 - margin
                
                if is_white:
                    # Draw X for white pieces (blue color)
                    color = (255, 0, 0)  # Blue
                    thickness = 8
                    cv2.line(grid_img, 
                             (cx - marker_size, cy - marker_size),
                             (cx + marker_size, cy + marker_size),
                             color, thickness)
                    cv2.line(grid_img,
                             (cx + marker_size, cy - marker_size),
                             (cx - marker_size, cy + marker_size),
                             color, thickness)
                    # Label
                    cv2.putText(grid_img, "X", (cx - 15, cy + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # Draw O for black pieces (red color)
                    color = (0, 0, 255)  # Red
                    thickness = 8
                    cv2.circle(grid_img, (cx, cy), marker_size, color, thickness)
                    # Label
                    cv2.putText(grid_img, "O", (cx - 15, cy + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add row/col labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(3):
        # Column labels (top)
        cv2.putText(grid_img, str(i), (i * cell_size + cell_size//2 - 10, 25),
                    font, 0.7, (100, 100, 100), 2)
        # Row labels (left)
        cv2.putText(grid_img, str(i), (8, i * cell_size + cell_size//2 + 10),
                    font, 0.7, (100, 100, 100), 2)
    
    return grid_img


class VisionTester:
    """Test harness for the vision module."""
    
    def __init__(self):
        self.config = VisionConfig()
        self.passed = 0
        self.failed = 0
        
    def log(self, msg: str, level: str = "INFO"):
        """Print a log message."""
        prefix = {"INFO": "[INFO]", "PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}
        print(f"{prefix.get(level, '[INFO]')} {msg}")
        
    def test_passed(self, name: str):
        """Mark a test as passed."""
        self.passed += 1
        self.log(f"{name}", "PASS")
        
    def test_failed(self, name: str, reason: str):
        """Mark a test as failed."""
        self.failed += 1
        self.log(f"{name}: {reason}", "FAIL")
        
    # ==================== CONFIG TESTS ====================
    
    def test_config(self) -> bool:
        """Test VisionConfig initialization and values."""
        print("\n" + "="*60)
        print("Testing VisionConfig")
        print("="*60)
        
        try:
            config = VisionConfig()
            
            # Check required attributes exist
            required_attrs = [
                'CAMERA_INDEX', 'CAMERA_WIDTH', 'CAMERA_HEIGHT',
                'BOARD_SIZE', 'CELL_SIZE_MM', 'ARUCO_DICT',
                'TFLITE_MODEL_PATH', 'TFLITE_CONFIDENCE',
                'PIECE_CLASS_NAMES', 'PIECE_CLASSES'
            ]
            
            for attr in required_attrs:
                if not hasattr(config, attr):
                    self.test_failed(f"Config.{attr}", "Attribute missing")
                    return False
                    
            # Validate values
            if config.BOARD_SIZE != 3:
                self.test_failed("Config.BOARD_SIZE", f"Expected 3, got {config.BOARD_SIZE}")
                return False
                
            if config.BOARD_OUTPUT_SIZE <= 0:
                self.test_failed("Config.BOARD_OUTPUT_SIZE", "Must be positive")
                return False
                
            if len(config.PIECE_CLASS_NAMES) == 0:
                self.test_failed("Config.PIECE_CLASS_NAMES", "Empty class list")
                return False
                
            self.test_passed("VisionConfig initialization")
            
            # Print config summary
            print(f"  Camera: index={config.CAMERA_INDEX}, {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
            print(f"  Board: {config.BOARD_SIZE}x{config.BOARD_SIZE}, cell={config.CELL_SIZE_MM}mm")
            print(f"  Model: {config.TFLITE_MODEL_PATH}")
            print(f"  Classes: {len(config.PIECE_CLASS_NAMES)} piece types")
            
            return True
            
        except Exception as e:
            self.test_failed("VisionConfig", str(e))
            return False
    
    # ==================== CAMERA TESTS ====================
    
    def test_camera(self) -> bool:
        """Test Camera initialization and frame capture."""
        print("\n" + "="*60)
        print("Testing Camera")
        print("="*60)
        
        try:
            camera = Camera(self.config)
            
            # Test initialization
            if camera.config is None:
                self.test_failed("Camera init", "Config not set")
                return False
            self.test_passed("Camera initialization")
            
            # Test opening camera
            if not camera.open():
                self.test_failed("Camera open", "Could not open camera")
                self.log("Try changing CAMERA_INDEX in VisionConfig", "WARN")
                return False
            self.test_passed("Camera open")
            
            # Test frame capture
            frame = camera.read()
            if frame is None:
                self.test_failed("Camera read", "No frame captured")
                camera.close()
                return False
                
            if len(frame.shape) != 3:
                self.test_failed("Camera read", f"Invalid frame shape: {frame.shape}")
                camera.close()
                return False
                
            h, w, c = frame.shape
            self.test_passed(f"Camera read ({w}x{h}x{c})")
            
            # Test multiple frames
            frame_times = []
            for i in range(10):
                start = time.time()
                frame = camera.read()
                frame_times.append(time.time() - start)
                
            avg_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            self.test_passed(f"Camera FPS: {fps:.1f}")
            
            # Test close
            camera.close()
            if camera.is_opened:
                self.test_failed("Camera close", "Still marked as open")
                return False
            self.test_passed("Camera close")
            
            # Test context manager
            with Camera(self.config) as cam:
                if not cam.is_opened:
                    self.test_failed("Camera context manager", "Not opened")
                    return False
                frame = cam.read()
                if frame is None:
                    self.test_failed("Camera context manager", "No frame")
                    return False
            self.test_passed("Camera context manager")
            
            return True
            
        except Exception as e:
            self.test_failed("Camera", str(e))
            return False
    
    # ==================== BOARD DETECTOR TESTS ====================
    
    def test_board_detector(self) -> bool:
        """Test BoardDetector initialization and detection."""
        print("\n" + "="*60)
        print("Testing BoardDetector")
        print("="*60)
        
        try:
            detector = BoardDetector(self.config)
            
            # Test initialization
            if detector.aruco_detector is None:
                self.test_failed("BoardDetector init", "ArUco detector not created")
                return False
            self.test_passed("BoardDetector initialization")
            
            # Test with synthetic image (no markers)
            blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
            detection = detector.detect(blank_image)
            
            if detection.success:
                self.test_failed("BoardDetector blank", "Should not detect on blank image")
                return False
            self.test_passed("BoardDetector rejects blank image")
            
            # Test detection result structure
            if not isinstance(detection, BoardDetection):
                self.test_failed("BoardDetector result", "Wrong return type")
                return False
                
            if not hasattr(detection, 'warped') or not hasattr(detection, 'success'):
                self.test_failed("BoardDetector result", "Missing attributes")
                return False
            self.test_passed("BoardDetector result structure")
            
            # Test cell region extraction
            fake_warped = np.zeros((self.config.BOARD_OUTPUT_SIZE, 
                                    self.config.BOARD_OUTPUT_SIZE, 3), dtype=np.uint8)
            cell = detector.get_cell_region(fake_warped, 0, 0)
            expected_size = self.config.CELL_OUTPUT_SIZE
            
            if cell.shape[0] != expected_size or cell.shape[1] != expected_size:
                self.test_failed("get_cell_region", f"Wrong size: {cell.shape}")
                return False
            self.test_passed("get_cell_region")
            
            # Test cell center calculation
            x_mm, y_mm = detector.get_cell_center_mm(1, 1)  # Center cell
            expected_center = self.config.CELL_SIZE_MM * 1.5
            if abs(x_mm - expected_center) > 0.1 or abs(y_mm - expected_center) > 0.1:
                self.test_failed("get_cell_center_mm", f"Wrong center: ({x_mm}, {y_mm})")
                return False
            self.test_passed("get_cell_center_mm")
            
            # Test debug drawing
            debug_frame = detector.draw_debug(blank_image, detection)
            if debug_frame is None or debug_frame.shape != blank_image.shape:
                self.test_failed("draw_debug", "Invalid debug frame")
                return False
            self.test_passed("draw_debug")
            
            return True
            
        except Exception as e:
            self.test_failed("BoardDetector", str(e))
            return False
    
    # ==================== PIECE DETECTOR TESTS ====================
    
    def test_piece_detector(self) -> bool:
        """Test PieceDetector initialization and detection."""
        print("\n" + "="*60)
        print("Testing PieceDetector")
        print("="*60)
        
        try:
            detector = PieceDetector(self.config)
            
            # Test initialization
            self.test_passed("PieceDetector initialization")
            
            # Check if model loaded
            if detector.interpreter is None:
                self.log("TFLite model not loaded - detection will return empty results", "WARN")
                self.log(f"Expected model at: {self.config.TFLITE_MODEL_PATH}", "WARN")
            else:
                self.test_passed("TFLite model loaded")
                print(f"  Quantized: {detector.is_quantized}")
                
            # Test with blank warped board
            blank_board = np.zeros((self.config.BOARD_OUTPUT_SIZE,
                                    self.config.BOARD_OUTPUT_SIZE, 3), dtype=np.uint8)
            board_state = detector.detect(blank_board)
            
            # Check result structure
            if not isinstance(board_state, BoardState):
                self.test_failed("PieceDetector result", "Wrong return type")
                return False
                
            if not hasattr(board_state, 'grid') or not hasattr(board_state, 'detections'):
                self.test_failed("PieceDetector result", "Missing attributes")
                return False
            self.test_passed("PieceDetector result structure")
            
            # Check grid dimensions
            if len(board_state.grid) != 3:
                self.test_failed("PieceDetector grid", f"Wrong rows: {len(board_state.grid)}")
                return False
            for row in board_state.grid:
                if len(row) != 3:
                    self.test_failed("PieceDetector grid", f"Wrong cols: {len(row)}")
                    return False
            self.test_passed("PieceDetector grid dimensions (3x3)")
            
            # Test grid display
            display = detector.get_grid_display(board_state)
            if not isinstance(display, str) or len(display) == 0:
                self.test_failed("get_grid_display", "Invalid display string")
                return False
            self.test_passed("get_grid_display")
            print("  Grid display:")
            for line in display.split('\n'):
                print(f"    {line}")
            
            # Test debug drawing
            debug_img = detector.draw_debug(blank_board, board_state)
            if debug_img is None or debug_img.shape != blank_board.shape:
                self.test_failed("draw_debug", "Invalid debug image")
                return False
            self.test_passed("draw_debug")
            
            # Test point to cell mapping
            cell_size = self.config.CELL_OUTPUT_SIZE
            test_cases = [
                (cell_size // 2, cell_size // 2, 0, 0),  # Top-left cell
                (cell_size * 2 + cell_size // 2, cell_size // 2, 0, 2),  # Top-right cell
                (cell_size + cell_size // 2, cell_size + cell_size // 2, 1, 1),  # Center cell
            ]
            for x, y, expected_row, expected_col in test_cases:
                row, col = detector._point_to_cell(x, y)
                if row != expected_row or col != expected_col:
                    self.test_failed("_point_to_cell", 
                                     f"({x},{y}) -> ({row},{col}), expected ({expected_row},{expected_col})")
                    return False
            self.test_passed("_point_to_cell mapping")
            
            return True
            
        except Exception as e:
            self.test_failed("PieceDetector", str(e))
            return False
    
    # ==================== INTEGRATION TESTS ====================
    
    def test_live(self, duration: int = 30):
        """Run live camera test with all components."""
        print("\n" + "="*60)
        print(f"Live Camera Test (press 'q' to quit, runs for {duration}s)")
        print("="*60)
        
        try:
            board_detector = BoardDetector(self.config)
            piece_detector = PieceDetector(self.config)
            
            with Camera(self.config) as camera:
                if not camera.is_opened:
                    self.test_failed("Live test", "Could not open camera")
                    return False
                    
                start_time = time.time()
                frame_count = 0
                detection_count = 0
                piece_detection_count = 0
                
                while True:
                    elapsed = time.time() - start_time
                    if elapsed > duration:
                        break
                        
                    frame = camera.read()
                    if frame is None:
                        continue
                        
                    frame_count += 1
                    
                    # Detect board
                    board_detection = board_detector.detect(frame)
                    
                    # Draw debug on camera frame
                    debug_frame = board_detector.draw_debug(frame, board_detection)
                    
                    # Add FPS counter
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(debug_frame, f"Time: {int(duration - elapsed)}s", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow("Camera", debug_frame)
                    
                    # Detect pieces on raw frame (always, even without board)
                    raw_detections = piece_detector.detect_raw(frame)
                    
                    # Draw detections on camera frame
                    for det in raw_detections:
                        x1, y1, x2, y2 = det.bbox
                        color = (0, 255, 0) if det.color == "white" else (255, 0, 0)
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(debug_frame, f"{det.piece_name} {det.confidence:.2f}",
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.putText(debug_frame, f"Raw detections: {len(raw_detections)}", 
                                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow("Camera", debug_frame)
                    
                    if board_detection.success:
                        detection_count += 1
                        
                        # Debug: print transform info once
                        if detection_count == 1:
                            print(f"\nTransform matrix shape: {board_detection.transform_matrix.shape}")
                            print(f"Board corners: {board_detection.corners}")
                            print(f"Pixel-to-world matrix available: {board_detection.pixel_to_world_matrix is not None}")
                        
                        # Map detections to board grid using transform
                        # Also pass pixel_to_world_matrix for robot coordinates
                        board_state = piece_detector.map_to_board(
                            raw_detections, 
                            board_detection.transform_matrix,
                            board_detection.pixel_to_world_matrix
                        )
                        
                        # Debug: show mapping results
                        if frame_count % 60 == 0 and raw_detections:
                            print(f"\nMapping {len(raw_detections)} detections:")
                            for det in raw_detections:
                                center = np.array([[det.center]], dtype=np.float32)
                                transformed = cv2.perspectiveTransform(center, board_detection.transform_matrix)
                                tx, ty = transformed[0][0]
                                print(f"  {det.piece_name} at {det.center} -> board ({tx:.0f}, {ty:.0f})")
                        
                        # Draw debug on warped board
                        debug_board = piece_detector.draw_debug(board_detection.warped, board_state)
                        
                        # Draw mapped detection centers on warped board
                        for det in board_state.detections:
                            cx, cy = det.center
                            cv2.circle(debug_board, (cx, cy), 10, (0, 255, 255), -1)
                            cv2.putText(debug_board, f"({cx},{cy})", (cx+15, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Add detection count to debug board
                        cv2.putText(debug_board, f"Mapped: {len(board_state.detections)}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.imshow("Board", debug_board)
                        
                        # Create and show TicTacToe grid UI
                        tictactoe_grid = create_tictactoe_grid(board_state, size=400)
                        cv2.imshow("TicTacToe", tictactoe_grid)
                        
                        # Track piece detections
                        if board_state.detections:
                            piece_detection_count += 1
                        
                        # Print grid and detections periodically
                        if frame_count % 30 == 0:
                            print("\n" + piece_detector.get_grid_display(board_state))
                            if board_state.detections:
                                for det in board_state.detections:
                                    if det.world_coords:
                                        wx, wy, wz = det.world_coords
                                        print(f"  {det.piece_type}: {det.confidence:.2f} at board {det.center} -> world ({wx:.3f}, {wy:.3f}, {wz:.3f})m")
                                    else:
                                        print(f"  {det.piece_type}: {det.confidence:.2f} at {det.center}")
                    else:
                        # Show empty TicTacToe grid when no board detected
                        empty_grid = create_tictactoe_grid(None, size=400)
                        cv2.putText(empty_grid, "No board", (130, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
                        cv2.imshow("TicTacToe", empty_grid)
                        
                        # Print raw detections when no board
                        if frame_count % 30 == 0 and raw_detections:
                            print(f"\nRaw detections (no board): {len(raw_detections)}")
                            for det in raw_detections:
                                print(f"  {det.piece_type}: {det.confidence:.2f} at {det.center}")
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        filename = f"vision_test_{int(time.time())}.png"
                        cv2.imwrite(filename, frame)
                        print(f"Saved: {filename}")
                
                cv2.destroyAllWindows()
                
                # Report results
                detection_rate = detection_count / frame_count * 100 if frame_count > 0 else 0
                piece_rate = piece_detection_count / detection_count * 100 if detection_count > 0 else 0
                print(f"\n  Frames captured: {frame_count}")
                print(f"  Board detections: {detection_count} ({detection_rate:.1f}%)")
                print(f"  Piece detections: {piece_detection_count} ({piece_rate:.1f}% of board frames)")
                print(f"  Average FPS: {frame_count / elapsed:.1f}")
                
                self.test_passed("Live camera test completed")
                return True
                
        except Exception as e:
            self.test_failed("Live test", str(e))
            cv2.destroyAllWindows()
            return False
    
    def test_with_image(self, image_path: str):
        """Test with a static image file."""
        print("\n" + "="*60)
        print(f"Testing with image: {image_path}")
        print("="*60)
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.test_failed("Load image", f"Could not load: {image_path}")
                return False
            self.test_passed(f"Image loaded ({image.shape[1]}x{image.shape[0]})")
            
            # Detect board
            board_detector = BoardDetector(self.config)
            board_detection = board_detector.detect(image)
            
            if not board_detection.success:
                self.log("Board not detected in image", "WARN")
                debug_frame = board_detector.draw_debug(image, board_detection)
                cv2.imshow("Image", debug_frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return True
                
            self.test_passed("Board detected")
            
            # Detect pieces
            piece_detector = PieceDetector(self.config)
            board_state = piece_detector.detect(board_detection.warped)
            
            print(f"  Detections: {len(board_state.detections)}")
            for det in board_state.detections:
                print(f"    - {det.piece_type}: {det.confidence:.2f}")
            
            print("\n  Grid:")
            print(piece_detector.get_grid_display(board_state))
            
            # Show results
            debug_frame = board_detector.draw_debug(image, board_detection)
            debug_board = piece_detector.draw_debug(board_detection.warped, board_state)
            
            cv2.imshow("Image", debug_frame)
            cv2.imshow("Board", debug_board)
            
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            self.test_passed("Image test completed")
            return True
            
        except Exception as e:
            self.test_failed("Image test", str(e))
            cv2.destroyAllWindows()
            return False
    
    # ==================== RUN ALL TESTS ====================
    
    def run_all(self) -> bool:
        """Run all unit tests."""
        print("\n" + "="*60)
        print("   VISION MODULE TEST SUITE")
        print("="*60)
        
        self.test_config()
        self.test_board_detector()
        self.test_piece_detector()
        
        # Camera test is optional (may not have camera)
        print("\n" + "-"*60)
        print("Camera test (requires connected camera)")
        print("-"*60)
        try:
            self.test_camera()
        except Exception as e:
            self.log(f"Camera test skipped: {e}", "WARN")
        
        # Summary
        print("\n" + "="*60)
        print("   TEST SUMMARY")
        print("="*60)
        total = self.passed + self.failed
        print(f"  Passed: {self.passed}/{total}")
        print(f"  Failed: {self.failed}/{total}")
        
        if self.failed == 0:
            print("\n  All tests passed!")
            return True
        else:
            print(f"\n  {self.failed} test(s) failed.")
            return False


def main():
    parser = argparse.ArgumentParser(description="Test the vision module")
    parser.add_argument("--camera", action="store_true", help="Test camera only")
    parser.add_argument("--board", action="store_true", help="Test board detector only")
    parser.add_argument("--piece", action="store_true", help="Test piece detector only")
    parser.add_argument("--live", action="store_true", help="Run live camera test")
    parser.add_argument("--image", type=str, help="Test with a static image")
    parser.add_argument("--duration", type=int, default=30, help="Duration for live test (seconds)")
    
    args = parser.parse_args()
    
    tester = VisionTester()
    
    # Run specific test or all tests
    if args.camera:
        tester.test_camera()
    elif args.board:
        tester.test_board_detector()
    elif args.piece:
        tester.test_piece_detector()
    elif args.live:
        tester.test_live(args.duration)
    elif args.image:
        tester.test_with_image(args.image)
    else:
        tester.run_all()
    
    # Return exit code
    sys.exit(0 if tester.failed == 0 else 1)


if __name__ == "__main__":
    main()
