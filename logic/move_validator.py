"""
Move validator for TicTacToe robot.
Validates that moves follow the rules.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
from .game_state import GameState, Player, PIECE_ORDER


@dataclass
class ValidationResult:
    """Result of move validation."""
    is_valid: bool
    error_message: Optional[str] = None


class MoveValidator:
    """
    Validates TicTacToe moves.
    
    Rules:
    1. Can only place on empty cells
    2. Must use pieces in order: Knight -> Bishop -> Rook -> Queen -> King
    3. Game must not be over
    """
    
    def validate_move(
        self, 
        game_state: GameState, 
        row: int, 
        col: int
    ) -> ValidationResult:
        """
        Validate a move.
        
        Args:
            game_state: Current game state.
            row: Row to place piece (0-2).
            col: Column to place piece (0-2).
            
        Returns:
            ValidationResult with is_valid and error_message.
        """
        # Check if game is over
        if game_state.is_game_over:
            return ValidationResult(
                is_valid=False,
                error_message="Game is already over!"
            )
        
        # Check if row/col are in valid range
        if not (0 <= row <= 2 and 0 <= col <= 2):
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid position ({row}, {col}). Must be 0-2."
            )
        
        # Check if cell is empty
        if game_state.board[row][col] is not None:
            return ValidationResult(
                is_valid=False,
                error_message=f"Cell ({row}, {col}) is already occupied by {game_state.board[row][col]}"
            )
        
        # Check if player has pieces left
        next_piece = game_state.get_current_piece()
        if next_piece is None:
            return ValidationResult(
                is_valid=False,
                error_message=f"{game_state.current_player.value} has no more pieces!"
            )
        
        # All checks passed!
        return ValidationResult(is_valid=True)
    
    def validate_piece_placement(
        self,
        game_state: GameState,
        row: int,
        col: int,
        piece_type: str
    ) -> ValidationResult:
        """
        Validate that a specific piece is being placed correctly.
        This is used when the robot detects a piece was placed.
        
        Args:
            game_state: Current game state.
            row: Row where piece was placed.
            col: Column where piece was placed.
            piece_type: The piece that was placed.
            
        Returns:
            ValidationResult.
        """
        # First validate the basic move
        basic_validation = self.validate_move(game_state, row, col)
        if not basic_validation.is_valid:
            return basic_validation
        
        # Check if the piece matches what should be placed
        expected_piece = game_state.get_current_piece()
        
        if piece_type != expected_piece:
            return ValidationResult(
                is_valid=False,
                error_message=f"Wrong piece! Expected {expected_piece}, got {piece_type}"
            )
        
        return ValidationResult(is_valid=True)
    
    def get_valid_moves(self, game_state: GameState) -> List[Tuple[int, int]]:
        """
        Get all valid moves for the current player.
        
        Args:
            game_state: Current game state.
            
        Returns:
            List of (row, col) valid move positions.
        """
        valid_moves = []
        
        if game_state.is_game_over:
            return valid_moves
        
        for row in range(3):
            for col in range(3):
                if game_state.board[row][col] is None:
                    valid_moves.append((row, col))
        
        return valid_moves


# Quick test
if __name__ == "__main__":
    print("Testing MoveValidator...")
    
    game = GameState()
    validator = MoveValidator()
    
    # Test valid move
    result = validator.validate_move(game, 1, 1)
    print(f"Move (1,1): valid={result.is_valid}, error={result.error_message}")
    
    # Make the move
    game.make_move(1, 1)
    
    # Test invalid move (same cell)
    result = validator.validate_move(game, 1, 1)
    print(f"Move (1,1) again: valid={result.is_valid}, error={result.error_message}")
    
    # Test out of range
    result = validator.validate_move(game, 5, 5)
    print(f"Move (5,5): valid={result.is_valid}, error={result.error_message}")
    
    # Get valid moves
    valid = validator.get_valid_moves(game)
    print(f"Valid moves: {valid}")
    
    print("\nMoveValidator test done!")
