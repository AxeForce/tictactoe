"""
Camera module for TicTacToe robot.
Handles camera initialization and frame capture.
"""

import cv2
import numpy as np
from typing import Optional
from .config import VisionConfig


class Camera:
    """
    Simple camera wrapper class.
    Handles opening the camera and grabbing frames.
    """
    
    def __init__(self, config: Optional[VisionConfig] = None):
        """
        Initialize the camera.
        
        Args:
            config: Vision configuration. Uses defaults if not provided.
        """
        self.config = config or VisionConfig()
        self.cap = None
        self.is_opened = False
        
    def open(self) -> bool:
        """
        Open the camera.
        
        Returns:
            True if camera opened successfully, False otherwise.
        """
        print(f"Opening camera at index {self.config.CAMERA_INDEX}...")
        
        # Try to open the camera
        self.cap = cv2.VideoCapture(
            self.config.CAMERA_INDEX,
            self.config.CAMERA_BACKEND
        )
        
        if not self.cap.isOpened():
            print("ERROR: Could not open camera!")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        # Verify settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera opened! Resolution: {int(actual_width)}x{int(actual_height)}")
        
        self.is_opened = True
        return True
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera.
        
        Returns:
            The captured frame as numpy array, or None if failed.
        """
        if not self.is_opened or self.cap is None:
            print("ERROR: Camera is not opened!")
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("ERROR: Could not read frame from camera!")
            return None
        
        return frame
    
    def close(self):
        """Close the camera and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        print("Camera closed.")
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Quick test
if __name__ == "__main__":
    print("Testing camera...")
    
    with Camera() as cam:
        if cam.is_opened:
            for i in range(10):
                frame = cam.read()
                if frame is not None:
                    cv2.imshow("Camera Test", frame)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break
    
    cv2.destroyAllWindows()
    print("Camera test done!")
