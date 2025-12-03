"""
AI player for TicTacToe robot.
Uses the Minimax algorithm to choose the best move.
"""

from typing import Optional, Tuple, List
from .game_state import GameState, Player
from .win_checker import WinChecker


class AIPlayer:
    """
    An AI that plays TicTacToe using the Minimax algorithm.
    
    The AI will always play optimally - it will win if possible,
    block the opponent if needed, and never lose (at worst, draw).
    """
    
    def __init__(self, player: Player = Player.BLACK):
        """
        Initialize the AI player.
        
        Args:
            player: Which player the AI controls (default: BLACK)
        """
        self.player = player
        self.win_checker = WinChecker()
        
        # Keep track of how many moves we've evaluated (for debugging)
        self.moves_evaluated = 0
    
    def get_best_move(self, game_state: GameState) -> Optional[Tuple[int, int]]:
        """
        Get the best move for the current position.
        
        Args:
            game_state: Current game state.
            
        Returns:
            (row, col) of best move, or None if no moves available.
        """
        self.moves_evaluated = 0
        
        # Check if it's our turn
        if game_state.current_player != self.player:
            print(f"Warning: It's not {self.player.value}'s turn!")
            return None
        
        # Get all valid moves
        valid_moves = game_state.get_empty_cells()
        
        if not valid_moves:
            return None
        
        # Special case: if only one move, just take it
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Special case: if center is available, often a good first move
        if len(game_state.moves) == 0 and (1, 1) in valid_moves:
            return (1, 1)
        
        # Use minimax to find best move
        best_score = float('-inf')
        best_move = valid_moves[0]
        
        for row, col in valid_moves:
            # Try this move
            new_state = game_state.copy()
            new_state.make_move(row, col)
            
            # Evaluate with minimax
            score = self._minimax(new_state, depth=9, is_maximizing=False)
            
            if score > best_score:
                best_score = score
                best_move = (row, col)
        
        print(f"AI evaluated {self.moves_evaluated} positions. Best move: {best_move} (score: {best_score})")
        
        return best_move
    
    def _minimax(
        self, 
        game_state: GameState, 
        depth: int, 
        is_maximizing: bool,
        alpha: float = float('-inf'),
        beta: float = float('inf')
    ) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            game_state: Current state to evaluate.
            depth: How deep to search.
            is_maximizing: True if maximizing player's turn.
            alpha: Alpha value for pruning.
            beta: Beta value for pruning.
            
        Returns:
            The score of the position.
        """
        self.moves_evaluated += 1
        
        # Check terminal states
        winner = self.win_checker.check_winner(game_state)
        
        if winner == self.player:
            return 10 + depth  # Win (prefer faster wins)
        elif winner == self.player.opposite():
            return -10 - depth  # Loss (prefer slower losses)
        elif self.win_checker.check_draw(game_state):
            return 0  # Draw
        
        if depth == 0:
            return 0  # No more depth, evaluate as neutral
        
        valid_moves = game_state.get_empty_cells()
        
        if not valid_moves:
            return 0  # No moves available
        
        if is_maximizing:
            max_score = float('-inf')
            for row, col in valid_moves:
                new_state = game_state.copy()
                new_state.make_move(row, col)
                score = self._minimax(new_state, depth - 1, False, alpha, beta)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Prune
            return max_score
        else:
            min_score = float('inf')
            for row, col in valid_moves:
                new_state = game_state.copy()
                new_state.make_move(row, col)
                score = self._minimax(new_state, depth - 1, True, alpha, beta)
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Prune
            return min_score
    
    def get_move_suggestion(self, game_state: GameState) -> str:
        """
        Get a human-readable move suggestion.
        
        Args:
            game_state: Current game state.
            
        Returns:
            A string describing the suggested move.
        """
        move = self.get_best_move(game_state)
        
        if move is None:
            return "No moves available!"
        
        row, col = move
        piece = game_state.get_current_piece()
        
        return f"Place {piece} at position ({row}, {col})"


# Quick test
if __name__ == "__main__":
    print("Testing AIPlayer...")
    
    ai = AIPlayer(Player.BLACK)
    
    # Test 1: AI should block a winning move
    game = GameState()
    game.board = [
        ["white_knight", "white_bishop", None],
        [None, "black_knight", None],
        [None, None, None]
    ]
    game.white_pieces_placed = 2
    game.black_pieces_placed = 1
    game.current_player = Player.BLACK
    
    game.print_board()
    print("\nAI is BLACK. White is about to win with (0,2)!")
    
    move = ai.get_best_move(game)
    print(f"AI's move: {move}")
    
    # AI should block at (0, 2)
    assert move == (0, 2), f"Expected (0, 2), got {move}"
    print("✓ AI correctly blocks the win!")
    
    # Test 2: AI should take a winning move
    game2 = GameState()
    game2.board = [
        ["black_knight", "black_bishop", None],
        [None, "white_knight", None],
        [None, None, None]
    ]
    game2.white_pieces_placed = 1
    game2.black_pieces_placed = 2
    game2.current_player = Player.BLACK
    
    game2.print_board()
    print("\nAI is BLACK. Can win with (0,2)!")
    
    move = ai.get_best_move(game2)
    print(f"AI's move: {move}")
    
    assert move == (0, 2), f"Expected (0, 2), got {move}"
    print("✓ AI correctly takes the win!")
    
    print("\nAIPlayer test done!")
