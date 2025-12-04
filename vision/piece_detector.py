"""
Piece detector for TicTacToe robot.
Uses TFLite YOLO model to detect chess pieces on the board.
Optimized for Raspberry Pi 4.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from .config import VisionConfig


@dataclass
class PieceDetection:
    """
    A single piece detection result.
    """
    piece_type: str       # e.g., "white_knight", "black_bishop"
    color: str            # "white" or "black"
    piece_name: str       # e.g., "knight", "bishop"
    confidence: float     # Detection confidence (0-1)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in camera frame
    center: Tuple[int, int]  # Center/base point of detection in pixels
    world_coords: Optional[Tuple[float, float, float]] = None  # (x, y, z) in robot frame (meters)


@dataclass
class BoardState:
    """
    The state of the TicTacToe board.
    A 3x3 grid where each cell contains a piece type or None.
    """
    grid: List[List[Optional[str]]]  # 3x3 grid of piece types or None
    detections: List[PieceDetection]  # All piece detections


class PieceDetector:
    """
    Detects chess pieces on the TicTacToe board using TFLite YOLO model.
    
    Maps piece detections to the 3x3 grid cells.
    """
    
    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize the piece detector.
        
        Args:
            config: Vision configuration.
        """
        self.config = config or VisionConfig()
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_scale = 1.0
        self.input_zero_point = 0
        self.output_scale = 1.0
        self.output_zero_point = 0
        self.is_quantized = False
        self._load_model()
        
    def _load_model(self):
        """Load the TFLite model optimized for Raspberry Pi."""
        try:
            # Prefer tflite_runtime - much lighter than full TensorFlow
            import tflite_runtime.interpreter as tflite
            print("Using tflite_runtime (recommended for RPi)")
        except ImportError:
            try:
                import tensorflow.lite as tflite
                print("Using tensorflow.lite (consider installing tflite-runtime for better RPi performance)")
            except ImportError:
                print("ERROR: Neither tflite_runtime nor tensorflow found!")
                print("For Raspberry Pi, install with: pip install tflite-runtime")
                return
        
        try:
            print(f"Loading TFLite model from {self.config.TFLITE_MODEL_PATH}...")
            # Use all 4 CPU cores on RPi 4 for parallel inference
            self.interpreter = tflite.Interpreter(
                model_path=self.config.TFLITE_MODEL_PATH,
                num_threads=4
            )
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape and quantization params
            input_shape = self.input_details[0]['shape']
            input_dtype = self.input_details[0]['dtype']
            
            # Check if model is quantized (int8)
            if input_dtype == np.int8 or input_dtype == np.uint8:
                self.is_quantized = True
                # Get input quantization parameters
                input_quant = self.input_details[0].get('quantization_parameters', {})
                if input_quant:
                    self.input_scale = input_quant.get('scales', [1.0])[0]
                    self.input_zero_point = input_quant.get('zero_points', [0])[0]
                else:
                    # Fallback for older TFLite format
                    quant = self.input_details[0].get('quantization', (1.0, 0))
                    self.input_scale = quant[0]
                    self.input_zero_point = quant[1]
                
                # Get output quantization parameters
                output_quant = self.output_details[0].get('quantization_parameters', {})
                if output_quant:
                    self.output_scale = output_quant.get('scales', [1.0])[0]
                    self.output_zero_point = output_quant.get('zero_points', [0])[0]
                else:
                    quant = self.output_details[0].get('quantization', (1.0, 0))
                    self.output_scale = quant[0]
                    self.output_zero_point = quant[1]
                
                print(f"INT8 quantized model detected!")
                print(f"  Input: scale={self.input_scale}, zero_point={self.input_zero_point}")
                print(f"  Output: scale={self.output_scale}, zero_point={self.output_zero_point}")
            else:
                self.is_quantized = False
            
            print(f"TFLite model loaded! Input shape: {input_shape}, dtype: {input_dtype}")
            
        except Exception as e:
            print(f"ERROR: Could not load TFLite model: {e}")
            print("Piece detection will not work without a trained model.")
            self.interpreter = None
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for TFLite model.
        
        Args:
            image: Input image (BGR).
            
        Returns:
            Preprocessed image tensor.
        """
        input_size = self.config.TFLITE_INPUT_SIZE
        
        # Resize to model input size (use INTER_LINEAR for speed on RPi)
        resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Handle int8 quantized model (optimized for RPi)
        if self.is_quantized:
            # Direct int8 quantization - avoid intermediate float32 for memory efficiency
            # Formula: int8_val = (uint8_val / 255.0 / scale) + zero_point
            # Simplified: int8_val = uint8_val * (1.0 / (255.0 * scale)) + zero_point
            scale_factor = 1.0 / (255.0 * self.input_scale)
            quantized = (rgb.astype(np.float32) * scale_factor + self.input_zero_point).astype(np.int8)
            return np.expand_dims(quantized, axis=0)
        else:
            # Float model: normalize to [0, 1]
            normalized = rgb.astype(np.float32) * (1.0 / 255.0)
            return np.expand_dims(normalized, axis=0)
    
    def _postprocess(
        self, 
        output: np.ndarray, 
        orig_width: int, 
        orig_height: int
    ) -> List[Tuple[np.ndarray, float, int]]:
        """
        Postprocess TFLite YOLO output.
        
        Args:
            output: Raw model output.
            orig_width: Original image width.
            orig_height: Original image height.
            
        Returns:
            List of (bbox, confidence, class_id) tuples.
        """
        input_size = self.config.TFLITE_INPUT_SIZE
        conf_threshold = self.config.TFLITE_CONFIDENCE
        iou_threshold = self.config.TFLITE_IOU_THRESHOLD
        
        # YOLO output format: [batch, num_detections, 5 + num_classes]
        # or [batch, 5 + num_classes, num_detections] depending on export
        
        # Handle different output shapes
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Check if we need to transpose (YOLO exports can vary)
        if output.shape[0] < output.shape[1]:
            output = output.T  # Transpose to [num_detections, features]
        
        detections = []
        
        for detection in output:
            # YOLO format: [x_center, y_center, width, height, conf, class_scores...]
            if len(detection) < 6:
                continue
                
            x_center, y_center, width, height = detection[:4]
            
            # Get class scores (after first 4 values)
            # Some models have objectness at index 4, some don't
            if len(detection) > 5 + len(self.config.PIECE_CLASS_NAMES):
                # Has objectness score
                objectness = detection[4]
                class_scores = detection[5:]
            else:
                # No objectness, class scores start at index 4
                objectness = 1.0
                class_scores = detection[4:]
            
            # Get best class
            if len(class_scores) == 0:
                continue
                
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Combined confidence
            confidence = float(objectness * class_conf)
            
            if confidence < conf_threshold:
                continue
            
            # Debug: print raw values for first few detections
            if len(detections) < 3 and self.config.DEBUG_MODE:
                print(f"  Raw YOLO: x={x_center:.1f}, y={y_center:.1f}, w={width:.1f}, h={height:.1f}, conf={confidence:.2f}")
            
            # Check if coordinates are normalized (0-1) or in pixels (0-416)
            # If max values are <= 1, they're normalized
            if x_center <= 1.0 and y_center <= 1.0 and width <= 1.0 and height <= 1.0:
                # Normalized coordinates - scale to input size first
                x_center *= input_size
                y_center *= input_size
                width *= input_size
                height *= input_size
            
            # Convert to original image coordinates
            x_center *= orig_width / input_size
            y_center *= orig_height / input_size
            width *= orig_width / input_size
            height *= orig_height / input_size
            
            # Convert to x1, y1, x2, y2
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Clamp to image bounds
            x1 = max(0, min(orig_width - 1, x1))
            y1 = max(0, min(orig_height - 1, y1))
            x2 = max(0, min(orig_width - 1, x2))
            y2 = max(0, min(orig_height - 1, y2))
            
            detections.append((np.array([x1, y1, x2, y2]), confidence, int(class_id)))
        
        # Apply NMS
        if len(detections) > 0:
            boxes = np.array([d[0] for d in detections])
            scores = np.array([d[1] for d in detections])
            
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                conf_threshold,
                iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                detections = [detections[i] for i in indices]
            else:
                detections = []
        
        return detections
    
    def detect_raw(self, frame: np.ndarray) -> List[PieceDetection]:
        """
        Detect pieces on the raw camera frame.
        
        Args:
            frame: Raw camera image (BGR).
            
        Returns:
            List of PieceDetection objects with coordinates in frame space.
        """
        detections = []
        
        if self.interpreter is None:
            print("WARNING: No TFLite model loaded, returning empty detections.")
            return detections
        
        # Get image dimensions
        orig_height, orig_width = frame.shape[:2]
        
        # Preprocess (resizes to 416x416)
        input_tensor = self._preprocess(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize output if model is quantized
        if self.is_quantized:
            output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        # Postprocess - coordinates are scaled back to original frame size
        raw_detections = self._postprocess(output, orig_width, orig_height)
        
        # Process detections
        for bbox, conf, cls_id in raw_detections:
            if cls_id >= len(self.config.PIECE_CLASS_NAMES):
                continue
                
            class_name = self.config.PIECE_CLASS_NAMES[cls_id]
            
            # Skip pawns - we don't use them in TicTacToe
            if "pawn" in class_name.lower():
                continue
            
            if class_name not in self.config.PIECE_CLASSES:
                continue
            
            piece_type = self.config.PIECE_CLASSES[class_name]
            color = "white" if "white" in piece_type else "black"
            piece_name = piece_type.replace("white_", "").replace("black_", "")
            
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = y2  # Use base of the piece to reduce parallax error
            
            detection = PieceDetection(
                piece_type=piece_type,
                color=color,
                piece_name=piece_name,
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                center=(int(center_x), int(center_y))
            )
            detections.append(detection)
        
        return detections
    
    def map_to_board(
        self, 
        detections: List[PieceDetection], 
        transform_matrix: np.ndarray,
        pixel_to_world_matrix: Optional[np.ndarray] = None
    ) -> BoardState:
        """
        Map piece detections from camera frame to board grid.
        
        Args:
            detections: List of detections with coordinates in camera frame.
            transform_matrix: Perspective transform from board_detector (pixel -> warped board).
            pixel_to_world_matrix: Optional transform from pixel -> robot world coordinates.
            
        Returns:
            BoardState with pieces mapped to 3x3 grid.
        """
        grid = [[None for _ in range(3)] for _ in range(3)]
        mapped_detections = []
        
        for det in detections:
            # Transform center point to board coordinates (for grid mapping)
            center = np.array([[det.center]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(center, transform_matrix)
            board_x, board_y = transformed[0][0]
            
            # Compute world coordinates if transform available
            world_coords = None
            if pixel_to_world_matrix is not None:
                world = cv2.perspectiveTransform(center, pixel_to_world_matrix)
                wx, wy = world[0][0]
                wz = self.config.BOARD_ORIGIN_Z
                world_coords = (float(wx), float(wy), float(wz))
            
            # Check if point is within board bounds
            if 0 <= board_x < self.config.BOARD_OUTPUT_SIZE and \
               0 <= board_y < self.config.BOARD_OUTPUT_SIZE:
                
                # Map to grid cell
                row, col = self._point_to_cell(int(board_x), int(board_y))
                
                if 0 <= row < 3 and 0 <= col < 3:
                    # Create new detection with board coordinates and world coords
                    mapped_det = PieceDetection(
                        piece_type=det.piece_type,
                        color=det.color,
                        piece_name=det.piece_name,
                        confidence=det.confidence,
                        bbox=det.bbox,  # Keep original bbox for drawing on frame
                        center=(int(board_x), int(board_y)),  # Board coordinates (pixels)
                        world_coords=world_coords  # Robot frame (meters)
                    )
                    mapped_detections.append(mapped_det)
                    
                    # Place in grid (higher confidence wins)
                    if grid[row][col] is None:
                        grid[row][col] = det.piece_type
            
            elif world_coords is not None:
                # Piece is outside board but we still have world coordinates
                # Include it in detections but not in grid
                mapped_det = PieceDetection(
                    piece_type=det.piece_type,
                    color=det.color,
                    piece_name=det.piece_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    center=det.center,  # Keep original pixel coords
                    world_coords=world_coords
                )
                mapped_detections.append(mapped_det)
        
        return BoardState(grid=grid, detections=mapped_detections)
    
    def detect(
        self, 
        frame: np.ndarray, 
        transform_matrix: Optional[np.ndarray] = None
    ) -> BoardState:
        """
        Detect pieces on raw frame and map to board grid.
        
        Args:
            frame: Raw camera image (BGR).
            transform_matrix: Perspective transform from board detection.
                              If None, assumes frame is already warped board.
            
        Returns:
            BoardState with grid of pieces and detection details.
        """
        # Detect pieces on raw frame
        raw_detections = self.detect_raw(frame)
        
        if transform_matrix is None:
            # Legacy mode: frame is warped board, map directly
            grid = [[None for _ in range(3)] for _ in range(3)]
            for det in raw_detections:
                row, col = self._point_to_cell(det.center[0], det.center[1])
                if 0 <= row < 3 and 0 <= col < 3 and grid[row][col] is None:
                    grid[row][col] = det.piece_type
            return BoardState(grid=grid, detections=raw_detections)
        
        # Map detections to board grid using transform
        return self.map_to_board(raw_detections, transform_matrix)
    
    def _point_to_cell(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert a point in the warped board to a grid cell.
        
        Args:
            x: X coordinate in warped board.
            y: Y coordinate in warped board.
            
        Returns:
            (row, col) cell indices.
        """
        cell_size = self.config.CELL_OUTPUT_SIZE
        
        col = x // cell_size
        row = y // cell_size
        
        # Clamp to valid range
        row = max(0, min(2, row))
        col = max(0, min(2, col))
        
        return row, col
    
    def draw_debug(
        self, 
        warped_board: np.ndarray, 
        board_state: BoardState
    ) -> np.ndarray:
        """
        Draw debug visualization on the warped board.
        
        Args:
            warped_board: The warped board image.
            board_state: Detection results.
            
        Returns:
            Image with debug visualization.
        """
        debug_img = warped_board.copy()
        cell_size = self.config.CELL_OUTPUT_SIZE
        
        # Draw grid lines
        for i in range(1, 3):
            # Vertical lines
            cv2.line(
                debug_img,
                (i * cell_size, 0),
                (i * cell_size, self.config.BOARD_OUTPUT_SIZE),
                (128, 128, 128),
                2
            )
            # Horizontal lines
            cv2.line(
                debug_img,
                (0, i * cell_size),
                (self.config.BOARD_OUTPUT_SIZE, i * cell_size),
                (128, 128, 128),
                2
            )
        
        # Draw detections
        for detection in board_state.detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Color based on piece color
            color = (0, 255, 0) if detection.color == "white" else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.piece_name} {detection.confidence:.2f}"
            cv2.putText(
                debug_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        # Draw grid cell contents
        for row in range(3):
            for col in range(3):
                piece = board_state.grid[row][col]
                if piece is not None:
                    # Draw piece name in cell center
                    cx = col * cell_size + cell_size // 2
                    cy = row * cell_size + cell_size // 2
                    
                    short_name = piece.split("_")[1][0].upper()  # e.g., "K" for knight
                    color = (0, 255, 0) if "white" in piece else (255, 0, 0)
                    
                    cv2.putText(
                        debug_img,
                        short_name,
                        (cx - 10, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        3
                    )
        
        return debug_img
    
    def get_grid_display(self, board_state: BoardState) -> str:
        """
        Get a text representation of the board grid.
        
        Args:
            board_state: The board state.
            
        Returns:
            String representation of the grid.
        """
        lines = []
        lines.append("┌───┬───┬───┐")
        
        for row in range(3):
            row_str = "│"
            for col in range(3):
                piece = board_state.grid[row][col]
                if piece is None:
                    row_str += "   │"
                else:
                    # Short name: W/B + first letter
                    color = "W" if "white" in piece else "B"
                    name = piece.split("_")[1][0].upper()
                    row_str += f"{color}{name} │"
            lines.append(row_str)
            
            if row < 2:
                lines.append("├───┼───┼───┤")
        
        lines.append("└───┴───┴───┘")
        return "\n".join(lines)


# Quick test
if __name__ == "__main__":
    from .camera import Camera
    from .board_detector import BoardDetector
    
    print("Testing piece detector...")
    
    board_detector = BoardDetector()
    piece_detector = PieceDetector()
    
    with Camera() as cam:
        if cam.is_opened:
            while True:
                frame = cam.read()
                if frame is None:
                    continue
                
                # Detect board
                board_detection = board_detector.detect(frame)
                
                if board_detection.success:
                    # Detect pieces
                    board_state = piece_detector.detect(board_detection.warped)
                    
                    # Draw debug
                    debug_img = piece_detector.draw_debug(
                        board_detection.warped, 
                        board_state
                    )
                    cv2.imshow("Piece Detection", debug_img)
                    
                    # Print grid
                    print("\033[H\033[J")  # Clear screen
                    print(piece_detector.get_grid_display(board_state))
                
                # Show original frame
                debug_frame = board_detector.draw_debug(frame, board_detection)
                cv2.imshow("Camera", debug_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cv2.destroyAllWindows()
    print("Piece detector test done!")
