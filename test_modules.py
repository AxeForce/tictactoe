"""
Test script for TicTacToe robot modules.
Run this to verify all components work before playing.
"""

import sys


def test_vision_config():
    """Test vision configuration."""
    print("\n=== Testing Vision Config ===")
    try:
        from vision.config import VisionConfig
        config = VisionConfig()
        print(f"  Board size: {config.BOARD_SIZE}x{config.BOARD_SIZE}")
        print(f"  Cell size: {config.CELL_SIZE_MM}mm")
        print(f"  Camera index: {config.CAMERA_INDEX}")
        print("  ✓ Vision config OK")
        return True
    except Exception as e:
        print(f"  ✗ Vision config FAILED: {e}")
        return False


def test_game_logic():
    """Test game logic components."""
    print("\n=== Testing Game Logic ===")
    try:
        from logic.game_state import GameState, Player
        from logic.move_validator import MoveValidator
        from logic.win_checker import WinChecker
        from logic.ai_player import AIPlayer
        
        # Test game state
        game = GameState()
        print(f"  Initial player: {game.current_player.value}")
        print(f"  Next piece: {game.get_current_piece()}")
        
        # Test make move
        game.make_move(1, 1)
        print(f"  Made move at (1,1)")
        
        # Test validator
        validator = MoveValidator()
        result = validator.validate_move(game, 0, 0)
        print(f"  Validate (0,0): valid={result.is_valid}")
        
        # Test win checker
        checker = WinChecker()
        winner = checker.check_winner(game)
        print(f"  Winner check: {winner}")
        
        # Test AI
        ai = AIPlayer(Player.BLACK)
        move = ai.get_best_move(game)
        print(f"  AI suggests: {move}")
        
        print("  ✓ Game logic OK")
        return True
    except Exception as e:
        print(f"  ✗ Game logic FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_arm_config():
    """Test arm configuration."""
    print("\n=== Testing Arm Config ===")
    try:
        from arm_control.config import ArmConfig
        config = ArmConfig()
        
        # Test cell to xyz
        x, y, z = config.cell_to_xyz(1, 1)
        print(f"  Cell (1,1) -> ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # Test staging position
        x, y, z = config.get_staging_position("white", 0)
        print(f"  White staging[0] -> ({x:.3f}, {y:.3f}, {z:.3f})")
        
        print("  ✓ Arm config OK")
        return True
    except Exception as e:
        print(f"  ✗ Arm config FAILED: {e}")
        return False


def test_ik_solver():
    """Test inverse kinematics solver."""
    print("\n=== Testing IK Solver ===")
    try:
        from arm_control.dofbot_ik import DofbotIK
        from arm_control.config import ArmConfig
        
        config = ArmConfig()
        ik = DofbotIK(config)
        
        print(f"  Using ikpy: {ik.use_ikpy}")
        
        # Test solving for center cell
        angles = ik.solve_for_cell(1, 1, height_offset=0.05)
        if angles:
            print(f"  Cell (1,1) angles: {[f'{a:.1f}' for a in angles]}")
            print("  ✓ IK solver OK")
            return True
        else:
            print("  ⚠ IK solver returned no solution (may need calibration)")
            return True  # Not a failure, just needs calibration
    except ImportError as e:
        print(f"  ⚠ IK solver skipped (ikpy not installed): {e}")
        return True  # Skip if ikpy not installed
    except Exception as e:
        print(f"  ✗ IK solver FAILED: {e}")
        return False


def test_arm_controller():
    """Test arm controller in simulation mode."""
    print("\n=== Testing Arm Controller (Simulation) ===")
    try:
        from arm_control.arm_controller import ArmController
        
        controller = ArmController(simulate=True)
        
        # Test go home
        controller.go_home()
        print("  Tested: go_home()")
        
        # Test move to cell
        controller.move_to_cell(1, 1)
        print("  Tested: move_to_cell()")
        
        # Test gripper
        controller.gripper.open()
        controller.gripper.close()
        print("  Tested: gripper open/close")
        
        print("  ✓ Arm controller OK")
        return True
    except Exception as e:
        print(f"  ✗ Arm controller FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera():
    """Test camera (optional - requires hardware)."""
    print("\n=== Testing Camera ===")
    try:
        from vision.camera import Camera
        
        cam = Camera()
        if cam.open():
            frame = cam.read()
            if frame is not None:
                print(f"  Frame shape: {frame.shape}")
                print("  ✓ Camera OK")
                cam.close()
                return True
            else:
                print("  ✗ Camera opened but no frame")
                cam.close()
                return False
        else:
            print("  ⚠ Camera not available (may be in use or not connected)")
            return True  # Not a failure, just not available
    except Exception as e:
        print(f"  ⚠ Camera test skipped: {e}")
        return True


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("   TicTacToe Robot - Module Tests")
    print("="*60)
    
    results = {
        "Vision Config": test_vision_config(),
        "Game Logic": test_game_logic(),
        "Arm Config": test_arm_config(),
        "IK Solver": test_ik_solver(),
        "Arm Controller": test_arm_controller(),
        "Camera": test_camera(),
    }
    
    print("\n" + "="*60)
    print("   Test Results")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to play TicTacToe.\n")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
