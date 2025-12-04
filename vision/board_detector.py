"""
Board detector for TicTacToe robot.
Uses ArUco markers to detect the 3x3 board and warp it to a top-down view.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .config import VisionConfig


@dataclass
class BoardDetection:
    """
    Result of board detection.
    Contains the warped board image and transformation info.
    """
    warped: np.ndarray          # Top-down view of the board
    corners: np.ndarray         # Original corner positions in camera frame
    transform_matrix: np.ndarray  # Perspective transform matrix (pixel -> warped board)
    pixel_to_world_matrix: np.ndarray  # Perspective transform (pixel -> robot world XY)
    success: bool               # Whether detection was successful
    
    def pixel_to_xyz(self, px: float, py: float, z: float = None, config: 'VisionConfig' = None) -> tuple:
        """
        Convert pixel coordinates to robot world coordinates.
        
        Args:
            px: Pixel X coordinate in camera frame.
            py: Pixel Y coordinate in camera frame.
            z: Z coordinate (height). If None, uses board surface height from config.
            config: VisionConfig to get default Z value.
            
        Returns:
            (x, y, z) in meters, or None if transform not available.
        """
        if not self.success or self.pixel_to_world_matrix is None:
            return None
        
        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.pixel_to_world_matrix)
        x, y = world[0][0]
        
        # Default Z from config if not provided
        if z is None:
            if config is not None:
                z = config.BOARD_ORIGIN_Z
            else:
                z = 0.02  # Fallback default board height
        
        return float(x), float(y), float(z)


class BoardDetector:
    """
    Detects the TicTacToe board using ArUco markers.
    
    The board should have 4 ArUco markers at the corners:
    - Marker 0: Top-left
    - Marker 1: Top-right
    - Marker 2: Bottom-right
    - Marker 3: Bottom-left
    """
    
    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize the board detector.
        
        Args:
            config: Vision configuration.
        """
        self.config = config or VisionConfig()
        
        # Setup ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.config.ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, self.aruco_params)
        
        # Output size for warped image
        self.output_size = self.config.BOARD_OUTPUT_SIZE
        
        print("BoardDetector initialized!")
    
    def detect(self, frame: np.ndarray) -> BoardDetection:
        """
        Detect the board in a frame.
        
        Args:
            frame: Input image from camera.
            
        Returns:
            BoardDetection with warped board if successful.
        """
        # Detect ArUco markers
        corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        
        # Check if we found all 4 markers
        if ids is None or len(ids) < 4:
            return BoardDetection(
                warped=np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8),
                corners=np.array([]),
                transform_matrix=np.eye(3),
                pixel_to_world_matrix=None,
                success=False
            )
        
        # Find the corners we need (markers 0, 1, 2, 3)
        board_corners = self._get_board_corners(corners, ids)
        
        if board_corners is None:
            return BoardDetection(
                warped=np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8),
                corners=np.array([]),
                transform_matrix=np.eye(3),
                pixel_to_world_matrix=None,
                success=False
            )
        
        # Warp the board to top-down view
        warped, matrix = self._warp_board(frame, board_corners)
        
        # Compute pixel-to-world transform for direct coordinate mapping
        pixel_to_world = self._compute_pixel_to_world(board_corners)
        
        return BoardDetection(
            warped=warped,
            corners=board_corners,
            transform_matrix=matrix,
            pixel_to_world_matrix=pixel_to_world,
            success=True
        )
    
    def _get_board_corners(
        self, 
        marker_corners: List, 
        marker_ids: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract the board corners from detected markers.
        
        Each marker has 4 corners. We take specific corners from each marker
        to form the board boundary.
        
        Args:
            marker_corners: List of marker corner arrays from ArUco detection.
            marker_ids: Array of detected marker IDs.
            
        Returns:
            4 board corners as numpy array, or None if not all markers found.
        """
        # Create a dict to quickly look up markers by ID
        id_to_corners = {}
        for i, marker_id in enumerate(marker_ids.flatten()):
            id_to_corners[marker_id] = marker_corners[i][0]
        
        # Check we have all required markers
        required_ids = self.config.ARUCO_MARKER_IDS
        for req_id in required_ids:
            if req_id not in id_to_corners:
                if self.config.DEBUG_MODE:
                    print(f"Missing marker ID: {req_id}")
                return None
        
        # Extract the inner corners of each marker to form the board
        # Marker 0 (top-left): use bottom-right corner (index 2)
        # Marker 1 (top-right): use bottom-left corner (index 3)
        # Marker 2 (bottom-right): use top-left corner (index 0)
        # Marker 3 (bottom-left): use top-right corner (index 1)
        board_corners = np.array([
            id_to_corners[0][2],  # Top-left of board
            id_to_corners[1][3],  # Top-right of board
            id_to_corners[2][0],  # Bottom-right of board
            id_to_corners[3][1],  # Bottom-left of board
        ], dtype=np.float32)
        
        return board_corners
    
    def _warp_board(
        self, 
        frame: np.ndarray, 
        corners: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp the board region to a square top-down view.
        
        Args:
            frame: Input image.
            corners: 4 corner points of the board.
            
        Returns:
            Tuple of (warped image, transformation matrix).
        """
        # Define destination points (square output)
        dst_points = np.array([
            [0, 0],
            [self.output_size - 1, 0],
            [self.output_size - 1, self.output_size - 1],
            [0, self.output_size - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Warp the image
        warped = cv2.warpPerspective(
            frame, 
            matrix, 
            (self.output_size, self.output_size)
        )
        
        return warped, matrix
    
    def _compute_pixel_to_world(self, corners: np.ndarray) -> np.ndarray:
        """
        Compute homography from pixel coordinates to robot world coordinates.
        
        This allows direct conversion of any pixel in the camera frame to
        physical (X, Y) coordinates in the robot's frame of reference.
        
        Args:
            corners: 4 board corners in pixel coordinates.
                     Order: [top-left, top-right, bottom-right, bottom-left]
                     
        Returns:
            3x3 homography matrix for cv2.perspectiveTransform
        """
        # Source: pixel coordinates of board corners
        src_points = corners.astype(np.float32)
        
        # Destination: physical coordinates in robot frame (meters)
        # Board corners in robot coordinate system:
        #   - top-left = (BOARD_ORIGIN_X, BOARD_ORIGIN_Y)
        #   - top-right = (BOARD_ORIGIN_X, BOARD_ORIGIN_Y + board_size)
        #   - bottom-right = (BOARD_ORIGIN_X + board_size, BOARD_ORIGIN_Y + board_size)
        #   - bottom-left = (BOARD_ORIGIN_X + board_size, BOARD_ORIGIN_Y)
        board_size = 3 * self.config.CELL_SIZE_M
        
        ox = self.config.BOARD_ORIGIN_X
        oy = self.config.BOARD_ORIGIN_Y
        
        dst_points = np.array([
            [ox, oy],                           # top-left
            [ox, oy + board_size],              # top-right
            [ox + board_size, oy + board_size], # bottom-right
            [ox + board_size, oy],              # bottom-left
        ], dtype=np.float32)
        
        # Compute homography: pixel -> world
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return matrix
    
    def get_cell_region(
        self, 
        warped_board: np.ndarray, 
        row: int, 
        col: int
    ) -> np.ndarray:
        """
        Extract a single cell from the warped board.
        
        Args:
            warped_board: The warped top-down board image.
            row: Row index (0-2, top to bottom).
            col: Column index (0-2, left to right).
            
        Returns:
            Image of the cell.
        """
        cell_size = self.config.CELL_OUTPUT_SIZE
        
        x1 = col * cell_size
        y1 = row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        
        return warped_board[y1:y2, x1:x2].copy()
    
    def get_cell_center_mm(self, row: int, col: int) -> Tuple[float, float]:
        """
        Get the center of a cell in millimeters (board coordinates).
        
        Args:
            row: Row index (0-2).
            col: Column index (0-2).
            
        Returns:
            (x_mm, y_mm) center position.
        """
        cell_size = self.config.CELL_SIZE_MM
        
        # Center of cell
        x_mm = (col + 0.5) * cell_size
        y_mm = (row + 0.5) * cell_size
        
        return x_mm, y_mm
    
    def draw_debug(
        self, 
        frame: np.ndarray, 
        detection: BoardDetection
    ) -> np.ndarray:
        """
        Draw debug visualization on the frame.
        
        Args:
            frame: Input frame to draw on.
            detection: The board detection result.
            
        Returns:
            Frame with debug visualization.
        """
        debug_frame = frame.copy()
        
        if detection.success:
            # Draw board outline
            corners = detection.corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(debug_frame, pt1, pt2, (0, 255, 0), 2)
            
            # Draw corner points
            for i, corner in enumerate(corners):
                cv2.circle(debug_frame, tuple(corner), 8, (0, 0, 255), -1)
                cv2.putText(
                    debug_frame, 
                    str(i), 
                    (corner[0] + 10, corner[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 255), 
                    2
                )
            
            # Draw "Board Detected" text
            cv2.putText(
                debug_frame,
                "Board Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                debug_frame,
                "Looking for board markers...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
        
        return debug_frame


# Quick test
if __name__ == "__main__":
    from .camera import Camera
    
    print("Testing board detector...")
    
    detector = BoardDetector()
    
    with Camera() as cam:
        if cam.is_opened:
            while True:
                frame = cam.read()
                if frame is None:
                    continue
                
                detection = detector.detect(frame)
                debug_frame = detector.draw_debug(frame, detection)
                
                cv2.imshow("Board Detection", debug_frame)
                
                if detection.success:
                    cv2.imshow("Warped Board", detection.warped)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cv2.destroyAllWindows()
    print("Board detector test done!")
