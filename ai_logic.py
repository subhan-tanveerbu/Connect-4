"""
=============================================================================
ai_logic.py  –  Connect-4 AI using Minimax + Alpha-Beta Pruning
=============================================================================
ALGORITHM OVERVIEW
──────────────────
Minimax is a recursive adversarial search algorithm used in two-player
zero-sum games.  The idea is simple:

  • The MAXIMISING player (AI) tries to pick the move that leads to the
    HIGHEST heuristic score.
  • The MINIMISING player (Human) tries to pick the move that leads to the
    LOWEST heuristic score.

At each recursive call we alternate between these two roles, looking
*depth* moves into the future.  When we reach a terminal state (win/draw)
or our depth limit, we return a numeric evaluation of the board.

ALPHA-BETA PRUNING
──────────────────
Alpha-Beta is an optimisation that makes Minimax skip ("prune") branches
that cannot possibly affect the final decision:

  alpha = best score the MAXIMISER can already guarantee
  beta  = best score the MINIMISER can already guarantee

  • In a MAX node: if we find a score ≥ beta  → prune (the minimiser would
    never let us reach this state).
  • In a MIN node: if we find a score ≤ alpha → prune (the maximiser can
    already do better elsewhere).

Alpha-Beta reduces the average branching factor from O(b^d) to O(b^(d/2)),
effectively doubling the usable search depth.

SEARCH DEPTH
────────────
DEPTH = 5 gives the AI a look-ahead of 5 half-moves (plies).  Increasing
it makes the AI stronger but slower.  At depth 5 each human move takes
< 0.5 s on a modern CPU thanks to alpha-beta pruning.

=============================================================================
"""

import math
import random
import numpy as np

from game_logic import (
    ROWS, COLS, EMPTY, HUMAN, AI,
    drop_piece, get_next_open_row,
    get_valid_locations, winning_move, is_terminal_node
)

# ─── Search depth ─────────────────────────────────────────────────────────────
DEPTH = 5          # How many plies ahead the AI looks (increase for stronger AI)

# ─── Heuristic score constants ────────────────────────────────────────────────
SCORE_WIN    =  100_000   # AI wins  (very large positive value)
SCORE_LOSE   = -100_000   # AI loses (very large negative value)
SCORE_3      =      50    # AI has 3 discs in a window with 1 empty
SCORE_2      =      10    # AI has 2 discs in a window with 2 empty
SCORE_OPP_3  =     -80    # Opponent has 3 in a window  (must block!)
SCORE_CENTER =       6    # Bonus per AI disc in the centre column


# =============================================================================
# Window evaluation
# =============================================================================
def evaluate_window(window: list, piece: int) -> int:
    """
    Score a single 4-cell *window* from the perspective of *piece*.

    A "window" is any four consecutive cells taken from a row, column,
    or diagonal.  We count how many discs each player has in that window
    to determine its value.

    Scoring logic
    ─────────────
    • 4 AI discs           → SCORE_WIN   (terminal win)
    • 3 AI + 1 empty       → SCORE_3     (one away from winning)
    • 2 AI + 2 empty       → SCORE_2     (building threat)
    • 3 opponent + 1 empty → SCORE_OPP_3 (must block – heavily penalised)

    Parameters
    ----------
    window : list of 4 cell values (EMPTY / HUMAN / AI)
    piece  : the player we are evaluating FOR (always AI in our calls)

    Returns
    -------
    int   partial score contribution of this window
    """
    opponent = HUMAN if piece == AI else AI
    score    = 0

    ai_count   = window.count(piece)
    empty_count = window.count(EMPTY)
    opp_count  = window.count(opponent)

    if ai_count == 4:
        score += SCORE_WIN
    elif ai_count == 3 and empty_count == 1:
        score += SCORE_3
    elif ai_count == 2 and empty_count == 2:
        score += SCORE_2

    # Penalise opponent's threats
    if opp_count == 3 and empty_count == 1:
        score += SCORE_OPP_3

    return score


# =============================================================================
# Full board heuristic
# =============================================================================
def score_board(board: np.ndarray, piece: int) -> int:
    """
    Evaluate the ENTIRE board and return a numeric score from *piece*'s
    perspective.

    The function sweeps over every possible 4-cell window in all four
    directions and sums up evaluate_window() scores.  A centre-column
    bonus is added to encourage the AI to occupy the middle (which opens
    up more winning directions).

    Parameters
    ----------
    board : current board state
    piece : AI (we always call this with piece=AI)

    Returns
    -------
    int   total heuristic score (higher = better for AI)
    """
    total_score = 0

    # ── Centre column preference ───────────────────────────────────────────
    # The centre column (col 3) is involved in more potential four-in-a-rows
    # than any other column, so we reward the AI for occupying it.
    center_col   = [int(board[r][COLS // 2]) for r in range(ROWS)]
    center_count = center_col.count(piece)
    total_score  += center_count * SCORE_CENTER

    # ── Horizontal windows ─────────────────────────────────────────────────
    for r in range(ROWS):
        row_array = [int(board[r][c]) for c in range(COLS)]
        for c in range(COLS - 3):
            window = row_array[c : c + 4]
            total_score += evaluate_window(window, piece)

    # ── Vertical windows ───────────────────────────────────────────────────
    for c in range(COLS):
        col_array = [int(board[r][c]) for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col_array[r : r + 4]
            total_score += evaluate_window(window, piece)

    # ── Diagonal / (positive slope) windows ───────────────────────────────
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r - i][c + i] for i in range(4)]
            total_score += evaluate_window(window, piece)

    # ── Diagonal \ (negative slope) windows ───────────────────────────────
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(4)]
            total_score += evaluate_window(window, piece)

    return total_score


# =============================================================================
# Minimax with Alpha-Beta Pruning
# =============================================================================
def minimax(
    board:        np.ndarray,
    depth:        int,
    alpha:        float,
    beta:         float,
    maximising:   bool
) -> tuple:
    """
    Minimax search with alpha-beta pruning.

    How it works (step by step)
    ───────────────────────────
    1. BASE CASES
       • Terminal board (someone won or board full) → return exact score.
       • Depth = 0 → return heuristic board score (no more look-ahead).

    2. MAXIMISING NODE  (AI's turn)
       For every valid column:
         a. Simulate dropping an AI piece there (copy the board).
         b. Recurse with depth-1, NOT maximising (human's turn).
         c. If the child's score is higher than our best so far, update.
         d. Update alpha = max(alpha, best_score).
         e. If alpha ≥ beta → PRUNE (the minimiser wouldn't choose this path).
       Return (best_col, best_score).

    3. MINIMISING NODE  (Human's turn, simulated by the AI)
       Mirror of the above but we look for the minimum score and update beta.

    Parameters
    ----------
    board       : current board (numpy array, WILL BE MODIFIED then restored)
    depth       : remaining search depth (0 = leaf node)
    alpha       : best score the maximiser can already guarantee
    beta        : best score the minimiser can already guarantee
    maximising  : True when it's the AI's (maximiser's) turn

    Returns
    -------
    (best_col : int or None, best_score : int)
        best_col  – column index of the best move at this node
                    (None at leaf nodes where no move is made)
        best_score – heuristic value of the best move
    """
    valid_cols  = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    # ── Base case 1: terminal board ────────────────────────────────────────
    if is_terminal:
        if winning_move(board, AI):
            return (None, SCORE_WIN * 10)   # x10 so immediate wins rank higher
        elif winning_move(board, HUMAN):
            return (None, SCORE_LOSE * 10)
        else:
            return (None, 0)                # Draw

    # ── Base case 2: depth limit reached ──────────────────────────────────
    if depth == 0:
        return (None, score_board(board, AI))

    # ── Recursive case: MAXIMISING (AI) ───────────────────────────────────
    if maximising:
        best_score = -math.inf
        best_col   = random.choice(valid_cols)   # default: random valid col

        # Try centre column first – small but effective move-ordering trick
        ordered_cols = sorted(valid_cols, key=lambda c: abs(c - COLS // 2))

        for col in ordered_cols:
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI)               # make move

            _, score = minimax(board, depth - 1, alpha, beta, False)

            board[row][col] = EMPTY                        # undo move

            if score > best_score:
                best_score = score
                best_col   = col

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break   # ← Beta cut-off (prune remaining siblings)

        return (best_col, best_score)

    # ── Recursive case: MINIMISING (Human) ────────────────────────────────
    else:
        best_score = math.inf
        best_col   = random.choice(valid_cols)

        ordered_cols = sorted(valid_cols, key=lambda c: abs(c - COLS // 2))

        for col in ordered_cols:
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, HUMAN)             # make move

            _, score = minimax(board, depth - 1, alpha, beta, True)

            board[row][col] = EMPTY                        # undo move

            if score < best_score:
                best_score = score
                best_col   = col

            beta = min(beta, best_score)
            if alpha >= beta:
                break   # ← Alpha cut-off (prune remaining siblings)

        return (best_col, best_score)


# =============================================================================
# Public entry point
# =============================================================================
def get_best_move(board: np.ndarray) -> int:
    """
    Return the column index of the AI's best move.

    This is the ONLY function the GUI needs to call.  It wraps minimax
    and returns just the column number.

    Parameters
    ----------
    board : current board state

    Returns
    -------
    int   best column for the AI to play in
    """
    col, _ = minimax(board, DEPTH, -math.inf, math.inf, True)
    return col
