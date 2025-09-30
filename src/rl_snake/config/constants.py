"""Game constants and configuration for the Snake game."""

from pygame.constants import K_DOWN, K_LEFT, K_RIGHT, K_UP

# Global constants
FONT_SIZE = 36

DIS_X = 800
DIS_Y = 800
DISPLAY_RES = (DIS_X, DIS_Y)

X = 100
Y = 80
WIDTH = 600
HEIGHT = 600
THICKNESS = 6

GAME_SIZE = (X, Y, WIDTH, HEIGHT)
BORDER_SIZE = (
    X - THICKNESS,
    Y - THICKNESS,
    WIDTH + 2 * THICKNESS,
    HEIGHT + 2 * THICKNESS,
)

SNAKE_SIZE = 20
SNAKE_SPEED = 5
INIT_SNAKE_COO = (DIS_X / 2, DIS_Y / 2, SNAKE_SIZE, SNAKE_SIZE)
RADIUS = SNAKE_SIZE // 2

SCORE_COO = (DIS_X // 2 - X, Y // 2)
N = WIDTH // SNAKE_SIZE

# Colors
BLUE = (89, 152, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
WHITE = (245, 245, 245)

# Movement
STEP = SNAKE_SIZE
DIRECTION = {
    K_UP: (0, -STEP),
    K_DOWN: (0, STEP),
    K_RIGHT: (STEP, 0),
    K_LEFT: (-STEP, 0),
}


class GameConstants:
    """Dynamic game constants that adapt to different grid sizes."""

    def __init__(self, N: int = 30):
        """
        Initialize game constants for an N×N grid.

        Args:
            N: Grid size (number of cells per side)
        """
        for key, value in self._get_game_constants(N).items():
            setattr(self, key, value)

    def _get_game_constants(self, N: int) -> dict:
        """
        Calculate all game constants for an N×N grid.

        Args:
            N: Grid size

        Returns:
            Dictionary containing all game constants
        """
        # Configurable parameters
        SNAKE_SIZE = 20
        X_MARGIN = 3 * N + 3 * SNAKE_SIZE
        Y_MARGIN = 2 * N + 2 * SNAKE_SIZE
        THICKNESS = 6
        FONT_SIZE = 36
        SNAKE_SPEED = 5

        # Calculated dimensions
        WIDTH = N * SNAKE_SIZE
        HEIGHT = N * SNAKE_SIZE
        DIS_X = WIDTH + 2 * X_MARGIN
        DIS_Y = HEIGHT + 2 * Y_MARGIN
        DISPLAY_RES = (DIS_X, DIS_Y)
        SCORE_COO = (DIS_X // 2 - X_MARGIN, Y_MARGIN // 2)

        # Drawing areas
        GAME_SIZE = (X_MARGIN, Y_MARGIN, WIDTH, HEIGHT)
        BORDER_SIZE = (
            X_MARGIN - THICKNESS,
            Y_MARGIN - THICKNESS,
            WIDTH + 2 * THICKNESS,
            HEIGHT + 2 * THICKNESS,
        )
        INIT_SNAKE_COO = (DIS_X / 2, DIS_Y / 2, SNAKE_SIZE, SNAKE_SIZE)
        RADIUS = SNAKE_SIZE // 2

        # Movement and directions
        STEP = SNAKE_SIZE
        DIRECTION = {
            K_UP: (0, -STEP),
            K_LEFT: (-STEP, 0),
            K_DOWN: (0, STEP),
            K_RIGHT: (STEP, 0),
        }

        return {
            "FONT_SIZE": FONT_SIZE,
            "DIS_X": DIS_X,
            "DIS_Y": DIS_Y,
            "DISPLAY_RES": DISPLAY_RES,
            "X": X_MARGIN,
            "Y": Y_MARGIN,
            "WIDTH": WIDTH,
            "HEIGHT": HEIGHT,
            "THICKNESS": THICKNESS,
            "GAME_SIZE": GAME_SIZE,
            "BORDER_SIZE": BORDER_SIZE,
            "SNAKE_SIZE": SNAKE_SIZE,
            "SNAKE_SPEED": SNAKE_SPEED,
            "SCORE_COO": SCORE_COO,
            "INIT_SNAKE_COO": INIT_SNAKE_COO,
            "RADIUS": RADIUS,
            "N": N,
            "STEP": STEP,
            "DIRECTION": DIRECTION,
            "BLUE": BLUE,
            "BLACK": BLACK,
            "GRAY": GRAY,
            "WHITE": WHITE,
        }
