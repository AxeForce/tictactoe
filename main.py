"""
Main orchestration script for TicTacToe robot.

This script ties together:
- Vision (camera, board detection, piece detection)
- Logic (game state, move validation, AI)
- Arm control (IK, gripper, pick and place)

Run this script to play TicTacToe against the robot!
"""

import cv2
import time
from typing import Optional, Tuple

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


class TicTacToeRobot:
    """
    Main controller for the TicTacToe robot.
    
    Game flow:
    1. Human (WHITE) places a piece on the board
    2. Robot detects the move via vision
    3. Robot (BLACK) calculates best response
    4. Robot picks a piece and places it on the board
    5. Repeat until someone wins or it's a draw
    """
    
    def __init__(
        self,
        human_player: Player = Player.WHITE,
        simulate_arm: bool = False
    ):
        """
        Initialize the TicTacToe robot.
        
        Args:
            human_player: Which player the human controls.
            simulate_arm: If True, don't control real hardware.
        """
        print("\n" + "="*60)
        print("   TicTacToe Robot - Initializing...")
        print("="*60 + "\n")
        
        self.human_player = human_player
        self.robot_player = human_player.opposite()
        
        # Initialize vision
        print("Initializing vision system...")
        self.vision_config = VisionConfig()
        self.camera = Camera(self.vision_config)
        self.board_detector = BoardDetector(self.vision_config)
        self.piece_detector = PieceDetector(self.vision_config)
        
        # Initialize game logic
        print("Initializing game logic...")
        self.game_state = GameState()
        self.validator = MoveValidator()
        self.win_checker = WinChecker()
        self.ai = AIPlayer(self.robot_player)
        
        # Initialize arm control
        print("Initializing arm control...")
        self.arm_config = ArmConfig()
        self.arm = ArmController(self.arm_config, simulate=simulate_arm)
        
        # State tracking
        self.last_board_state: Optional[BoardState] = None
        self.is_running = False
        
        print("\n" + "="*60)
        print("   TicTacToe Robot - Ready!")
        print(f"   Human plays: {human_player.value.upper()}")
        print(f"   Robot plays: {self.robot_player.value.upper()}")
        print("="*60 + "\n")
    
    def start(self):
        """Start the game."""
        print("\nStarting TicTacToe game...")
        print("Press 'q' to quit, 's' to save screenshot\n")
        
        # Open camera
        if not self.camera.open():
            print("ERROR: Could not open camera!")
            return
        
        # Move arm to home position
        self.arm.go_home()
        
        self.is_running = True
        self._game_loop()
        
        # Cleanup
        self.camera.close()
        cv2.destroyAllWindows()
    
    def _game_loop(self):
        """Main game loop."""
        while self.is_running and not self.game_state.is_game_over:
            # Capture and process frame
            frame = self.camera.read()
            if frame is None:
                continue
            
            # Detect board
            board_detection = self.board_detector.detect(frame)
            
            if board_detection.success:
                # Detect pieces
                board_state = self.piece_detector.detect(board_detection.warped)
                
                # Check for changes (human move)
                if self.game_state.current_player == self.human_player:
                    self._check_human_move(board_state)
                
                # Draw debug visualization
                debug_board = self.piece_detector.draw_debug(
                    board_detection.warped, 
                    board_state
                )
                cv2.imshow("Board", debug_board)
                
                self.last_board_state = board_state
            
            # Draw camera view
            debug_frame = self.board_detector.draw_debug(frame, board_detection)
            self._draw_game_info(debug_frame)
            cv2.imshow("TicTacToe Robot", debug_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nGame quit by user.")
                self.is_running = False
            elif key == ord('s'):
                filename = f"tictactoe_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            elif key == ord('r'):
                self._reset_game()
        
        # Game over
        self._show_game_result()
    
    def _check_human_move(self, current_state: BoardState):
        """
        Check if the human made a move by comparing board states.
        
        Args:
            current_state: Current board state from vision.
        """
        if self.last_board_state is None:
            return
        
        # Compare grids to find new piece
        for row in range(3):
            for col in range(3):
                old_piece = self.last_board_state.grid[row][col]
                new_piece = current_state.grid[row][col]
                
                # Found a new piece!
                if old_piece is None and new_piece is not None:
                    # Check if it's the human's color
                    if self.human_player.value in new_piece:
                        self._process_human_move(row, col, new_piece)
                        return
    
    def _process_human_move(self, row: int, col: int, piece_type: str):
        """
        Process a detected human move.
        
        Args:
            row: Row where piece was placed.
            col: Column where piece was placed.
            piece_type: The piece that was placed.
        """
        print(f"\n>>> Human placed {piece_type} at ({row}, {col})")
        
        # Validate the move
        expected_piece = self.game_state.get_current_piece()
        
        if piece_type != expected_piece:
            print(f"WARNING: Expected {expected_piece}, got {piece_type}")
            print("Ignoring this move. Please use the correct piece!")
            return
        
        # Make the move in game state
        if self.game_state.make_move(row, col):
            # Check for winner
            self.win_checker.update_game_state(self.game_state)
            
            self.game_state.print_board()
            
            if not self.game_state.is_game_over:
                # Robot's turn!
                self._robot_move()
    
    def _robot_move(self):
        """Execute the robot's move."""
        print("\n>>> Robot is thinking...")
        
        # Get best move from AI
        move = self.ai.get_best_move(self.game_state)
        
        if move is None:
            print("ERROR: AI could not find a move!")
            return
        
        target_row, target_col = move
        piece_index = self.game_state.black_pieces_placed
        
        print(f">>> Robot will place {PIECE_ORDER[piece_index]} at ({target_row}, {target_col})")
        
        # Execute the physical move
        success = self.arm.execute_robot_move(
            self.robot_player.value,
            piece_index,
            target_row,
            target_col
        )
        
        if success:
            # Update game state
            self.game_state.make_move(target_row, target_col)
            
            # Check for winner
            self.win_checker.update_game_state(self.game_state)
            
            self.game_state.print_board()
        else:
            print("ERROR: Robot failed to execute move!")
    
    def _draw_game_info(self, frame):
        """Draw game information on the frame."""
        y = 80
        
        # Current turn
        turn_text = f"Turn: {self.game_state.current_player.value.upper()}"
        if self.game_state.current_player == self.human_player:
            turn_text += " (Human)"
        else:
            turn_text += " (Robot)"
        
        cv2.putText(
            frame, turn_text, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        y += 30
        
        # Next piece
        next_piece = self.game_state.get_current_piece()
        if next_piece:
            cv2.putText(
                frame, f"Next piece: {next_piece}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
            )
        y += 30
        
        # Score
        cv2.putText(
            frame, f"White: {self.game_state.white_pieces_placed}/5  Black: {self.game_state.black_pieces_placed}/5",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
        )
        
        # Game over status
        if self.game_state.is_game_over:
            if self.game_state.winner:
                result = f"{self.game_state.winner.value.upper()} WINS!"
            else:
                result = "DRAW!"
            
            cv2.putText(
                frame, result, (frame.shape[1]//2 - 100, frame.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4
            )
    
    def _show_game_result(self):
        """Show the final game result."""
        print("\n" + "="*60)
        print("   GAME OVER!")
        print("="*60)
        
        self.game_state.print_board()
        
        if self.game_state.winner:
            winner = self.game_state.winner
            if winner == self.human_player:
                print("\n🎉 Congratulations! You won!")
            else:
                print("\n🤖 Robot wins! Better luck next time!")
        else:
            print("\n🤝 It's a draw! Good game!")
        
        print("\n" + "="*60)
    
    def _reset_game(self):
        """Reset the game for a new round."""
        print("\nResetting game...")
        self.game_state = GameState()
        self.last_board_state = None
        self.arm.go_home()
        print("Game reset! Remove all pieces from the board.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TicTacToe Robot")
    parser.add_argument(
        "--simulate", 
        action="store_true",
        help="Run in simulation mode (no arm control)"
    )
    parser.add_argument(
        "--robot-first",
        action="store_true",
        help="Let the robot play first (as WHITE)"
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run without UI (console mode)"
    )
    
    args = parser.parse_args()
    
    # Launch UI by default
    if not args.no_ui:
        from ui import TicTacToeUI
        print("\n" + "="*60)
        print("   TicTacToe Robot UI")
        print("="*60 + "\n")
        ui = TicTacToeUI(simulate_arm=args.simulate)
        ui.run()
        return
    
    # Console mode (--no-ui)
    # Determine players
    if args.robot_first:
        human_player = Player.BLACK
    else:
        human_player = Player.WHITE
    
    # Create and start robot
    robot = TicTacToeRobot(
        human_player=human_player,
        simulate_arm=args.simulate
    )
    
    try:
        robot.start()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    finally:
        print("Goodbye!")


if __name__ == "__main__":
    main()
