"""
TicTacToe Robot UI
A graphical interface for the TicTacToe robot using Tkinter.

Shows:
- Camera view with piece detection bounding boxes
- Live board state (O for white, X for black)
- Game status and next move
- Difficulty level selection
"""

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum

# Vision imports
from vision.config import VisionConfig
from vision.camera import Camera
from vision.board_detector import BoardDetector
from vision.piece_detector import PieceDetector, BoardState

# Logic imports
from logic.game_state import GameState, Player, PIECE_ORDER
from logic.move_validator import MoveValidator
from logic.win_checker import WinChecker
from logic.ai_player import AIPlayer

# Arm control imports
from arm_control.config import ArmConfig
from arm_control.arm_controller import ArmController


class Difficulty(Enum):
    """AI difficulty levels."""
    EASY = 1      # Random moves
    MEDIUM = 2    # Some strategy
    HARD = 3      # Full minimax


@dataclass
class RobotMove:
    """Represents a pending robot move."""
    row: int
    col: int
    piece: str
    status: str = "pending"  # pending, executing, done


class TicTacToeUI:
    """
    Main UI class for the TicTacToe robot.
    """
    
    def __init__(self, simulate_arm: bool = True):
        """Initialize the UI."""
        self.simulate_arm = simulate_arm
        self.is_running = False
        self.difficulty = Difficulty.HARD
        
        # Initialize components (will be done in start())
        self.camera: Optional[Camera] = None
        self.board_detector: Optional[BoardDetector] = None
        self.piece_detector: Optional[PieceDetector] = None
        self.game_state: Optional[GameState] = None
        self.ai: Optional[AIPlayer] = None
        self.arm: Optional[ArmController] = None
        self.win_checker: Optional[WinChecker] = None
        
        # State
        self.last_board_state: Optional[BoardState] = None
        self.pending_robot_move: Optional[RobotMove] = None
        self.human_player = Player.WHITE
        self.robot_player = Player.BLACK
        
        # Create UI
        self._create_ui()
        
    def _create_ui(self):
        """Create the Tkinter UI."""
        self.root = tk.Tk()
        self.root.title("TicTacToe Robot")
        self.root.configure(bg='#1a1a2e')
        
        # Make window resizable
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='white', font=('Segoe UI', 11))
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#00d4ff')
        style.configure('Status.TLabel', font=('Segoe UI', 12), foreground='#ffd700')
        style.configure('Move.TLabel', font=('Segoe UI', 11), foreground='#00ff88')
        style.configure('TButton', font=('Segoe UI', 10, 'bold'))
        
        # Left panel - Camera view
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ttk.Label(left_frame, text="📷 Camera View", style='Title.TLabel').pack(pady=(0, 5))
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(left_frame, bg='#0f0f1a', highlightthickness=2, 
                                        highlightbackground='#00d4ff')
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right panel
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Board display section
        ttk.Label(right_frame, text="🎮 Game Board", style='Title.TLabel').pack(pady=(0, 10))
        
        # Board canvas (for the 3x3 grid)
        self.board_frame = ttk.Frame(right_frame)
        self.board_frame.pack(pady=10)
        
        self.board_cells = []
        for row in range(3):
            row_cells = []
            for col in range(3):
                cell = tk.Label(
                    self.board_frame,
                    text="",
                    font=('Segoe UI', 24, 'bold'),
                    width=4,
                    height=2,
                    bg='#16213e',
                    fg='white',
                    relief='ridge',
                    borderwidth=2
                )
                cell.grid(row=row, column=col, padx=2, pady=2)
                row_cells.append(cell)
            self.board_cells.append(row_cells)
        
        # Legend
        legend_frame = ttk.Frame(right_frame)
        legend_frame.pack(pady=5)
        ttk.Label(legend_frame, text="O = White (Human)  ", foreground='#00ff88').pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="X = Black (Robot)", foreground='#ff6b6b').pack(side=tk.LEFT)
        
        # Game status section
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        ttk.Label(right_frame, text="📊 Game Status", style='Title.TLabel').pack()
        
        self.status_label = ttk.Label(right_frame, text="Initializing...", style='Status.TLabel')
        self.status_label.pack(pady=5)
        
        self.turn_label = ttk.Label(right_frame, text="Turn: -")
        self.turn_label.pack()
        
        self.next_piece_label = ttk.Label(right_frame, text="Next piece: -")
        self.next_piece_label.pack()
        
        # Robot move section
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        ttk.Label(right_frame, text="🤖 Robot Move", style='Title.TLabel').pack()
        
        self.robot_move_label = ttk.Label(right_frame, text="Waiting for human...", style='Move.TLabel')
        self.robot_move_label.pack(pady=5)
        
        # Difficulty section
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        ttk.Label(right_frame, text="⚙️ Difficulty", style='Title.TLabel').pack()
        
        diff_frame = ttk.Frame(right_frame)
        diff_frame.pack(pady=10)
        
        self.diff_var = tk.StringVar(value="HARD")
        
        diff_buttons = [
            ("Easy", "EASY", "#4ade80"),
            ("Medium", "MEDIUM", "#fbbf24"),
            ("Hard", "HARD", "#f87171")
        ]
        
        for text, value, color in diff_buttons:
            btn = tk.Button(
                diff_frame,
                text=text,
                font=('Segoe UI', 10, 'bold'),
                width=8,
                bg=color if self.diff_var.get() == value else '#2d3748',
                fg='black' if self.diff_var.get() == value else 'white',
                activebackground=color,
                command=lambda v=value, c=color: self._set_difficulty(v, c)
            )
            btn.pack(side=tk.LEFT, padx=5)
            setattr(self, f'btn_{value.lower()}', btn)
        
        # Control buttons
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, pady=15)
        
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶ Start Game",
            font=('Segoe UI', 11, 'bold'),
            bg='#10b981',
            fg='white',
            width=12,
            command=self._start_game
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(
            control_frame,
            text="🔄 Reset",
            font=('Segoe UI', 11, 'bold'),
            bg='#6366f1',
            fg='white',
            width=12,
            command=self._reset_game
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Quit button
        tk.Button(
            right_frame,
            text="✕ Quit",
            font=('Segoe UI', 10),
            bg='#ef4444',
            fg='white',
            width=26,
            command=self._quit
        ).pack(pady=10)
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        
    def _set_difficulty(self, value: str, color: str):
        """Set the AI difficulty level."""
        self.diff_var.set(value)
        self.difficulty = Difficulty[value]
        
        # Update button colors
        for diff_name in ["easy", "medium", "hard"]:
            btn = getattr(self, f'btn_{diff_name}')
            if diff_name.upper() == value:
                btn.configure(bg=color, fg='black')
            else:
                btn.configure(bg='#2d3748', fg='white')
        
        print(f"Difficulty set to: {value}")
        
    def _start_game(self):
        """Start or resume the game."""
        if self.is_running:
            return
            
        self.start_btn.configure(state='disabled')
        self.status_label.configure(text="Starting...")
        
        # Initialize in background thread
        threading.Thread(target=self._initialize_components, daemon=True).start()
        
    def _initialize_components(self):
        """Initialize all components (runs in background thread)."""
        try:
            # Vision
            self.root.after(0, lambda: self.status_label.configure(text="Loading vision..."))
            self.vision_config = VisionConfig()
            self.camera = Camera(self.vision_config)
            self.board_detector = BoardDetector(self.vision_config)
            self.piece_detector = PieceDetector(self.vision_config)
            
            # Game logic
            self.root.after(0, lambda: self.status_label.configure(text="Loading game logic..."))
            self.game_state = GameState()
            self.win_checker = WinChecker()
            self.ai = AIPlayer(self.robot_player)
            
            # Arm control
            self.root.after(0, lambda: self.status_label.configure(text="Loading arm control..."))
            self.arm_config = ArmConfig()
            self.arm = ArmController(self.arm_config, simulate=self.simulate_arm)
            
            # Open camera
            self.root.after(0, lambda: self.status_label.configure(text="Opening camera..."))
            if not self.camera.open():
                self.root.after(0, lambda: self.status_label.configure(text="ERROR: Camera failed!"))
                self.root.after(0, lambda: self.start_btn.configure(state='normal'))
                return
            
            # Success - start game loop
            self.is_running = True
            self.root.after(0, lambda: self.status_label.configure(text="Game running!"))
            self.root.after(0, self._update_loop)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.configure(text=f"ERROR: {str(e)[:30]}"))
            self.root.after(0, lambda: self.start_btn.configure(state='normal'))
            print(f"Initialization error: {e}")
            
    def _update_loop(self):
        """Main update loop (runs on UI thread)."""
        if not self.is_running:
            return
            
        try:
            # Capture frame
            frame = self.camera.read()
            
            if frame is not None:
                # Detect board
                board_detection = self.board_detector.detect(frame)
                
                if board_detection.success:
                    # Detect pieces
                    board_state = self.piece_detector.detect(board_detection.warped)
                    
                    # Check for human move
                    if self.game_state.current_player == self.human_player:
                        self._check_human_move(board_state)
                    
                    # Draw debug on frame
                    frame = self._draw_detections(frame, board_detection, board_state)
                    
                    # Update board display
                    self._update_board_display(board_state)
                    
                    self.last_board_state = board_state
                else:
                    # Draw "no board" message
                    cv2.putText(frame, "Board not detected", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update camera canvas
                self._update_camera_canvas(frame)
                
            # Update game info
            self._update_game_info()
            
        except Exception as e:
            print(f"Update error: {e}")
            
        # Schedule next update (~30 FPS)
        if self.is_running:
            self.root.after(33, self._update_loop)
            
    def _draw_detections(self, frame, board_detection, board_state):
        """Draw detection boxes on the frame."""
        # Draw board corners
        frame = self.board_detector.draw_debug(frame, board_detection)
        
        # Draw piece detections info on frame
        y = 30
        for det in board_state.detections:
            color = (0, 255, 0) if det.color == "white" else (0, 0, 255)
            text = f"{det.piece_name}: {det.confidence:.2f}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
            
        return frame
        
    def _update_camera_canvas(self, frame):
        """Update the camera canvas with the current frame."""
        # Get canvas size
        canvas_width = self.camera_canvas.winfo_width()
        canvas_height = self.camera_canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            return
            
        # Resize frame to fit canvas while maintaining aspect ratio
        frame_height, frame_width = frame.shape[:2]
        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        
        # Update canvas
        self.camera_canvas.delete("all")
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.camera_canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self.camera_canvas.image = photo  # Keep reference
        
    def _update_board_display(self, board_state: BoardState):
        """Update the board grid display."""
        for row in range(3):
            for col in range(3):
                piece = board_state.grid[row][col]
                cell = self.board_cells[row][col]
                
                if piece is None:
                    cell.configure(text="", bg='#16213e')
                else:
                    # Determine symbol and color
                    if "white" in piece:
                        symbol = "O"
                        bg_color = '#065f46'  # Green for white/human
                        fg_color = '#10b981'
                    else:
                        symbol = "X"
                        bg_color = '#7f1d1d'  # Red for black/robot
                        fg_color = '#f87171'
                    
                    # Get piece initial (K=Knight, B=Bishop, R=Rook, Q=Queen, K=King)
                    piece_name = piece.split("_")[1]
                    piece_initial = piece_name[0].upper()
                    if piece_name == "knight":
                        piece_initial = "N"  # Use N for knight to avoid confusion with King
                    
                    cell.configure(
                        text=f"{symbol}\n{piece_initial}",
                        bg=bg_color,
                        fg=fg_color
                    )
                    
    def _update_game_info(self):
        """Update game status labels."""
        if self.game_state is None:
            return
            
        # Turn info
        if self.game_state.is_game_over:
            if self.game_state.winner:
                winner_name = "Human" if self.game_state.winner == self.human_player else "Robot"
                self.status_label.configure(text=f"🏆 {winner_name} WINS!")
            else:
                self.status_label.configure(text="🤝 It's a DRAW!")
            self.turn_label.configure(text="Game Over")
        else:
            current = "Human (O)" if self.game_state.current_player == self.human_player else "Robot (X)"
            self.turn_label.configure(text=f"Turn: {current}")
            self.status_label.configure(text="Game in progress")
        
        # Next piece
        next_piece = self.game_state.get_current_piece()
        if next_piece:
            piece_name = next_piece.split("_")[1].capitalize()
            self.next_piece_label.configure(text=f"Next piece: {piece_name}")
        else:
            self.next_piece_label.configure(text="Next piece: -")
            
        # Robot move
        if self.pending_robot_move:
            piece_name = self.pending_robot_move.piece.split("_")[1].capitalize()
            pos = f"({self.pending_robot_move.row}, {self.pending_robot_move.col})"
            self.robot_move_label.configure(
                text=f"→ {piece_name} to {pos} [{self.pending_robot_move.status}]"
            )
        elif self.game_state.current_player == self.robot_player and not self.game_state.is_game_over:
            self.robot_move_label.configure(text="Calculating...")
        else:
            self.robot_move_label.configure(text="Waiting for human...")
            
    def _check_human_move(self, current_state: BoardState):
        """Check if human made a move."""
        if self.last_board_state is None:
            return
            
        for row in range(3):
            for col in range(3):
                old_piece = self.last_board_state.grid[row][col]
                new_piece = current_state.grid[row][col]
                
                if old_piece is None and new_piece is not None:
                    if self.human_player.value in new_piece:
                        self._process_human_move(row, col, new_piece)
                        return
                        
    def _process_human_move(self, row: int, col: int, piece_type: str):
        """Process a detected human move."""
        print(f"Human placed {piece_type} at ({row}, {col})")
        
        expected_piece = self.game_state.get_current_piece()
        if piece_type != expected_piece:
            print(f"WARNING: Expected {expected_piece}, got {piece_type}")
            return
            
        if self.game_state.make_move(row, col):
            self.win_checker.update_game_state(self.game_state)
            
            if not self.game_state.is_game_over:
                # Robot's turn - calculate in background
                threading.Thread(target=self._robot_move, daemon=True).start()
                
    def _robot_move(self):
        """Execute robot's move (runs in background thread)."""
        # Get best move based on difficulty
        if self.difficulty == Difficulty.EASY:
            move = self._get_easy_move()
        elif self.difficulty == Difficulty.MEDIUM:
            move = self._get_medium_move()
        else:
            move = self.ai.get_best_move(self.game_state)
            
        if move is None:
            return
            
        target_row, target_col = move
        piece_index = self.game_state.black_pieces_placed
        piece_type = f"{self.robot_player.value}_{PIECE_ORDER[piece_index]}"
        
        # Update pending move
        self.pending_robot_move = RobotMove(
            row=target_row,
            col=target_col,
            piece=piece_type,
            status="executing"
        )
        
        print(f"Robot will place {piece_type} at ({target_row}, {target_col})")
        
        # Execute physical move
        success = self.arm.execute_robot_move(
            self.robot_player.value,
            piece_index,
            target_row,
            target_col
        )
        
        if success:
            self.game_state.make_move(target_row, target_col)
            self.win_checker.update_game_state(self.game_state)
            self.pending_robot_move.status = "done"
        else:
            self.pending_robot_move.status = "failed"
            
        # Clear pending move after a delay
        time.sleep(1)
        self.pending_robot_move = None
        
    def _get_easy_move(self):
        """Get a random valid move (easy difficulty)."""
        import random
        empty_cells = self.game_state.get_empty_cells()
        return random.choice(empty_cells) if empty_cells else None
        
    def _get_medium_move(self):
        """Get a somewhat strategic move (medium difficulty)."""
        import random
        
        # 50% chance to make optimal move, 50% random
        if random.random() < 0.5:
            return self.ai.get_best_move(self.game_state)
        else:
            return self._get_easy_move()
            
    def _reset_game(self):
        """Reset the game."""
        print("Resetting game...")
        self.game_state = GameState()
        self.last_board_state = None
        self.pending_robot_move = None
        
        # Clear board display
        for row in range(3):
            for col in range(3):
                self.board_cells[row][col].configure(text="", bg='#16213e')
                
        self.status_label.configure(text="Game reset! Remove pieces.")
        
        if self.arm:
            self.arm.go_home()
            
    def _quit(self):
        """Quit the application."""
        print("Quitting...")
        self.is_running = False
        
        if self.camera:
            self.camera.close()
            
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Run the UI main loop."""
        self.root.mainloop()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TicTacToe Robot UI")
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=True,
        help="Run in simulation mode (no arm control)"
    )
    parser.add_argument(
        "--real-arm",
        action="store_true",
        help="Use real arm hardware"
    )
    
    args = parser.parse_args()
    
    simulate = not args.real_arm
    
    print("\n" + "="*60)
    print("   TicTacToe Robot UI")
    print("="*60)
    print(f"   Mode: {'Simulation' if simulate else 'Real Hardware'}")
    print("="*60 + "\n")
    
    ui = TicTacToeUI(simulate_arm=simulate)
    ui.run()


if __name__ == "__main__":
    main()
