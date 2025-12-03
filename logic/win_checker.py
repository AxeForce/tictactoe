"""
Win checker for TicTacToe robot.
Checks if a player has won or if the game is a draw.
"""

from typing import Optional, List, Tuple
from .game_state import GameState, Player


class WinChecker:
    """
    Checks for win conditions in TicTacToe.
    
    Win condition: 3 pieces of the same color in a row
    (horizontally, vertically, or diagonally)
    """
    
    # All possible winning lines (as list of (row, col) tuples)
    WINNING_LINES = [
        # Rows
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        # Columns
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        # Diagonals
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]
    
    def check_winner(self, game_state: GameState) -> Optional[Player]:
        """
        Check if there's a winner.
        
        Args:
            game_state: The current game state.
            
        Returns:
            The winning Player, or None if no winner yet.
        """
        for line in self.WINNING_LINES:
            winner = self._check_line(game_state.board, line)
            if winner is not None:
                return winner
        
        return None
    
    def _check_line(
        self, 
        board: List[List[Optional[str]]], 
        line: List[Tuple[int, int]]
    ) -> Optional[Player]:
        """
        Check if a single line has a winner.
        
        Args:
            board: The game board.
            line: List of (row, col) positions to check.
            
        Returns:
            The winning Player if all 3 are same color, None otherwise.
        """
        pieces = []
        for row, col in line:
            piece = board[row][col]
            if piece is None:
                return None  # Empty cell, no winner on this line
            pieces.append(piece)
        
        # Check if all pieces are same color
        colors = [self._get_color(p) for p in pieces]
        
        if colors[0] == colors[1] == colors[2]:
            return Player.WHITE if colors[0] == "white" else Player.BLACK
        
        return None
    
    def _get_color(self, piece_type: str) -> str:
        """Extract color from piece type string."""
        return "white" if "white" in piece_type else "black"
    
    def check_draw(self, game_state: GameState) -> bool:
        """
        Check if the game is a draw.
        
        A draw occurs when:
        - All cells are filled AND no winner
        - Both players have used all their pieces AND no winner
        
        Args:
            game_state: The current game state.
            
        Returns:
            True if the game is a draw.
        """
        # First check if there's a winner - if so, not a draw
        if self.check_winner(game_state) is not None:
            return False
        
        # Check if board is full
        empty_cells = game_state.get_empty_cells()
        if len(empty_cells) == 0:
            return True
        
        # Check if both players have used all pieces
        if (game_state.white_pieces_placed >= 5 and 
            game_state.black_pieces_placed >= 5):
            return True
        
        return False
    
    def update_game_state(self, game_state: GameState) -> GameState:
        """
        Update the game state with winner/draw information.
        
        Args:
            game_state: The game state to update.
            
        Returns:
            Updated game state.
        """
        winner = self.check_winner(game_state)
        
        if winner is not None:
            game_state.winner = winner
            game_state.is_game_over = True
        elif self.check_draw(game_state):
            game_state.is_draw = True
            game_state.is_game_over = True
        
        return game_state
    
    def get_winning_line(self, game_state: GameState) -> Optional[List[Tuple[int, int]]]:
        """
        Get the winning line if there is one.
        
        Args:
            game_state: The game state.
            
        Returns:
            The winning line as list of (row, col), or None.
        """
        for line in self.WINNING_LINES:
            if self._check_line(game_state.board, line) is not None:
                return line
        return None


# Quick test
if __name__ == "__main__":
    print("Testing WinChecker...")
    
    checker = WinChecker()
    
    # Test 1: Horizontal win
    game1 = GameState()
    game1.board = [
        ["white_knight", "white_bishop", "white_rook"],
        [None, "black_knight", None],
        ["black_bishop", None, None]
    ]
    
    winner = checker.check_winner(game1)
    print(f"Test 1 (horizontal): winner = {winner}")
    assert winner == Player.WHITE
    
    # Test 2: Vertical win
    game2 = GameState()
    game2.board = [
        ["black_knight", "white_knight", None],
        ["black_bishop", "white_bishop", None],
        ["black_rook", None, None]
    ]
    
    winner = checker.check_winner(game2)
    print(f"Test 2 (vertical): winner = {winner}")
    assert winner == Player.BLACK
    
    # Test 3: Diagonal win
    game3 = GameState()
    game3.board = [
        ["white_knight", "black_knight", None],
        [None, "white_bishop", "black_bishop"],
        [None, None, "white_rook"]
    ]
    
    winner = checker.check_winner(game3)
    print(f"Test 3 (diagonal): winner = {winner}")
    assert winner == Player.WHITE
    
    # Test 4: No winner
    game4 = GameState()
    game4.board = [
        ["white_knight", "black_knight", None],
        [None, "black_bishop", None],
        [None, None, None]
    ]
    
    winner = checker.check_winner(game4)
    print(f"Test 4 (no winner): winner = {winner}")
    assert winner is None
    
    # Test 5: Draw (full board, no winner)
    game5 = GameState()
    game5.board = [
        ["white_knight", "black_knight", "white_bishop"],
        ["black_bishop", "white_rook", "black_rook"],
        ["black_queen", "white_queen", "white_king"]
    ]
    
    is_draw = checker.check_draw(game5)
    print(f"Test 5 (draw): is_draw = {is_draw}")
    
    print("\nWinChecker test done!")
