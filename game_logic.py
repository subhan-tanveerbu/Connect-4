"""
=============================================================================
game_logic.py  –  Connect-4 Core Game Logic
=============================================================================
This module contains ALL rules of Connect-4:
  • Board creation and manipulation
  • Drop-piece mechanics
  • Win / draw detection  (horizontal, vertical, both diagonals)
  • Valid-move checking

It is intentionally kept free of any GUI or AI code so that the AI and GUI
modules can import it without circular dependencies.
=============================================================================
"""

import numpy as np

# ─── Board dimensions ────────────────────────────────────────────────────────
ROWS    = 6   # Standard Connect-4 has 6 rows …
COLS    = 7   # … and 7 columns

# ─── Player / cell constants ─────────────────────────────────────────────────
EMPTY   = 0   # Empty cell
HUMAN   = 1   # Human player
AI      = 2   # AI player

# ─── Win length ──────────────────────────────────────────────────────────────
WIN_LEN = 4   # Four in a row wins the game


# =============================================================================
# Board creation
# =============================================================================
def create_board() -> np.ndarray:
    """
    Return a fresh 6×7 board filled with zeros (EMPTY).

    We use NumPy for fast array operations (the AI evaluates thousands
    of board positions per move, so speed matters).

    Returns
    -------
    np.ndarray  shape (ROWS, COLS), dtype int
    """
    return np.zeros((ROWS, COLS), dtype=int)


# =============================================================================
# Drop a piece
# =============================================================================
def drop_piece(board: np.ndarray, row: int, col: int, piece: int) -> None:
    """
    Place *piece* at (row, col).

    The caller is responsible for choosing the correct landing row via
    get_next_open_row().  This function just writes the value – no
    validation – to keep it fast inside the AI search tree.

    Parameters
    ----------
    board  : game board (modified in-place)
    row    : row index  (0 = top, ROWS-1 = bottom)
    col    : column index
    piece  : HUMAN (1) or AI (2)
    """
    board[row][col] = piece


# =============================================================================
# Valid column check
# =============================================================================
def is_valid_location(board: np.ndarray, col: int) -> bool:
    """
    A column is valid if its top cell (row 0) is still EMPTY.

    Connect-4 pieces fall to the lowest empty row, so the column is
    full once row 0 is occupied.

    Parameters
    ----------
    board : game board
    col   : column to test

    Returns
    -------
    bool  True  → column has at least one empty cell
          False → column is full
    """
    return board[0][col] == EMPTY


def get_valid_locations(board: np.ndarray) -> list:
    """
    Return a list of all column indices that are not yet full.

    Used by the AI to enumerate legal moves.
    """
    return [c for c in range(COLS) if is_valid_location(board, c)]


# =============================================================================
# Next open row in a column
# =============================================================================
def get_next_open_row(board: np.ndarray, col: int) -> int:
    """
    Scan from the bottom of *col* upward and return the first empty row.

    Pieces in Connect-4 fall under gravity, so the lowest empty row is
    where a newly dropped piece lands.

    Parameters
    ----------
    board : game board
    col   : column to scan

    Returns
    -------
    int   row index of the lowest empty cell in *col*
    """
    for r in range(ROWS - 1, -1, -1):   # bottom → top
        if board[r][col] == EMPTY:
            return r
    return -1   # Should never happen if is_valid_location() was checked first


# =============================================================================
# Win detection
# =============================================================================
def winning_move(board: np.ndarray, piece: int) -> bool:
    """
    Return True if *piece* has four consecutive discs anywhere on the board.

    Checks all four directions:
      1. Horizontal  – left/right within a row
      2. Vertical    – up/down within a column
      3. Diagonal /  – bottom-left to top-right
      4. Diagonal \\  – top-left to bottom-right

    Parameters
    ----------
    board : game board
    piece : HUMAN or AI

    Returns
    -------
    bool
    """
    # ── 1. Horizontal ─────────────────────────────────────────────────────
    for r in range(ROWS):
        for c in range(COLS - 3):          # need 4 cells → stop at COLS-4
            if all(board[r][c + i] == piece for i in range(WIN_LEN)):
                return True

    # ── 2. Vertical ───────────────────────────────────────────────────────
    for r in range(ROWS - 3):             # need 4 rows → stop at ROWS-4
        for c in range(COLS):
            if all(board[r + i][c] == piece for i in range(WIN_LEN)):
                return True

    # ── 3. Diagonal  / (positive slope) ───────────────────────────────────
    for r in range(3, ROWS):              # start at row 3 (need 3 rows above)
        for c in range(COLS - 3):
            if all(board[r - i][c + i] == piece for i in range(WIN_LEN)):
                return True

    # ── 4. Diagonal  \ (negative slope) ───────────────────────────────────
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == piece for i in range(WIN_LEN)):
                return True

    return False


# =============================================================================
# Draw detection
# =============================================================================
def is_draw(board: np.ndarray) -> bool:
    """
    Return True when the board is completely full and nobody has won.

    Because winning_move() is always checked before is_draw(), this
    function only needs to confirm there are no remaining valid columns.

    Parameters
    ----------
    board : game board

    Returns
    -------
    bool
    """
    return len(get_valid_locations(board)) == 0


# =============================================================================
# Terminal-state check (convenience for the AI)
# =============================================================================
def is_terminal_node(board: np.ndarray) -> bool:
    """
    Return True if the game is over (someone won, or the board is full).

    The AI's minimax algorithm uses this to know when to stop recursing.
    """
    return (
        winning_move(board, HUMAN) or
        winning_move(board, AI)    or
        is_draw(board)
    )
