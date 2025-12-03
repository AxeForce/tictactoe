"""
Game state management for TicTacToe robot.
Tracks the board, current player, and piece order.
"""

from enum import Enum
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


class Player(Enum):
    """The two players in the game."""
    WHITE = "white"
    BLACK = "black"
    
    def opposite(self) -> "Player":
        """Get the opposite player."""
        return Player.BLACK if self == Player.WHITE else Player.WHITE


# The order of pieces each player must use
# Knight first, then Bishop, Rook, Queen, King
PIECE_ORDER = ["knight", "bishop", "rook", "queen", "king"]


@dataclass
class Move:
    """
    A move in the game.
    """
    player: Player          # Who made the move
    row: int                # Row (0-2)
    col: int                # Column (0-2)
    piece_type: str         # Full piece type (e.g., "white_knight")
    move_number: int        # Which move this is (0-4)


@dataclass
class GameState:
    """
    The complete state of the TicTacToe game.
    
    Tracks:
    - The 3x3 board (which pieces are where)
    - Current player
    - How many pieces each player has placed (determines next piece)
    - Move history
    - Game status (ongoing, won, draw)
    """
    
    # The 3x3 board - None means empty, otherwise piece type string
    board: List[List[Optional[str]]] = field(
        default_factory=lambda: [[None for _ in range(3)] for _ in range(3)]
    )
    
    # Current player's turn
    current_player: Player = Player.WHITE
    
    # How many pieces each player has placed (0-5)
    white_pieces_placed: int = 0
    black_pieces_placed: int = 0
    
    # Move history
    moves: List[Move] = field(default_factory=list)
    
    # Game result
    winner: Optional[Player] = None
    is_draw: bool = False
    is_game_over: bool = False
    
    def get_next_piece(self, player: Player) -> Optional[str]:
        """
        Get the next piece type that a player should use.
        
        Args:
            player: The player.
            
        Returns:
            Piece type string (e.g., "white_knight"), or None if no more pieces.
        """
        pieces_placed = (
            self.white_pieces_placed if player == Player.WHITE 
            else self.black_pieces_placed
        )
        
        if pieces_placed >= len(PIECE_ORDER):
            return None  # Player has used all pieces
        
        piece_name = PIECE_ORDER[pieces_placed]
        return f"{player.value}_{piece_name}"
    
    def get_current_piece(self) -> Optional[str]:
        """Get the piece type for the current player's next move."""
        return self.get_next_piece(self.current_player)
    
    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move at the given position.
        
        Args:
            row: Row index (0-2).
            col: Column index (0-2).
            
        Returns:
            True if move was successful, False otherwise.
        """
        # Check if game is over
        if self.is_game_over:
            print("Game is already over!")
            return False
        
        # Check if cell is empty
        if self.board[row][col] is not None:
            print(f"Cell ({row}, {col}) is already occupied!")
            return False
        
        # Get the piece to place
        piece_type = self.get_current_piece()
        if piece_type is None:
            print("No more pieces to place!")
            return False
        
        # Place the piece
        self.board[row][col] = piece_type
        
        # Update piece count
        if self.current_player == Player.WHITE:
            move_num = self.white_pieces_placed
            self.white_pieces_placed += 1
        else:
            move_num = self.black_pieces_placed
            self.black_pieces_placed += 1
        
        # Record the move
        move = Move(
            player=self.current_player,
            row=row,
            col=col,
            piece_type=piece_type,
            move_number=move_num
        )
        self.moves.append(move)
        
        # Check for winner (done by external WinChecker)
        # Just switch turns here
        self.current_player = self.current_player.opposite()
        
        return True
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """
        Get all empty cells on the board.
        
        Returns:
            List of (row, col) tuples.
        """
        empty = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] is None:
                    empty.append((row, col))
        return empty
    
    def copy(self) -> "GameState":
        """Create a deep copy of the game state."""
        new_state = GameState(
            board=[[cell for cell in row] for row in self.board],
            current_player=self.current_player,
            white_pieces_placed=self.white_pieces_placed,
            black_pieces_placed=self.black_pieces_placed,
            moves=list(self.moves),
            winner=self.winner,
            is_draw=self.is_draw,
            is_game_over=self.is_game_over
        )
        return new_state
    
    def print_board(self):
        """Print the board to console."""
        print("\n  0   1   2")
        print("┌───┬───┬───┐")
        
        for row in range(3):
            row_str = "│"
            for col in range(3):
                piece = self.board[row][col]
                if piece is None:
                    row_str += "   │"
                else:
                    # Short name: W/B + first letter
                    color = "W" if "white" in piece else "B"
                    name = piece.split("_")[1][0].upper()
                    row_str += f"{color}{name} │"
            print(f"{row} {row_str}")
            
            if row < 2:
                print("├───┼───┼───┤")
        
        print("└───┴───┴───┘")
        
        # Print game info
        if self.is_game_over:
            if self.winner:
                print(f"\n🏆 {self.winner.value.upper()} WINS!")
            else:
                print("\n🤝 It's a DRAW!")
        else:
            print(f"\nCurrent turn: {self.current_player.value}")
            next_piece = self.get_current_piece()
            if next_piece:
                print(f"Next piece: {next_piece}")


# Quick test
if __name__ == "__main__":
    print("Testing GameState...")
    
    game = GameState()
    
    # Simulate a game
    moves = [
        (1, 1),  # White knight center
        (0, 0),  # Black knight top-left
        (0, 2),  # White bishop top-right
        (2, 0),  # Black bishop bottom-left
        (2, 2),  # White rook bottom-right - this should be a win!
    ]
    
    for row, col in moves:
        print(f"\n{game.current_player.value} moves to ({row}, {col})")
        game.make_move(row, col)
        game.print_board()
    
    print("\nGame state test done!")
