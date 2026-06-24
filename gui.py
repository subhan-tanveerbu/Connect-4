"""
=============================================================================
gui.py  –  Pygame GUI for Connect-4
=============================================================================
This module owns everything visual:
  • Window setup and colour theme
  • Drawing the board, discs, and animations
  • Hover preview (disc follows the mouse before dropping)
  • Turn indicator panel
  • End-game overlay (win / draw message + restart button)
  • Main game loop (event handling, turn management, AI calls)
=============================================================================
"""

import sys
import math
import time
import threading
import numpy as np
import pygame

from game_logic import (
    ROWS, COLS, EMPTY, HUMAN, AI,
    create_board, drop_piece, get_next_open_row,
    is_valid_location, winning_move, is_draw
)
from ai_logic import get_best_move

# =============================================================================
# Layout & sizing constants
# =============================================================================
CELL_SIZE    = 100          # Pixels per cell (board grid)
RADIUS       = CELL_SIZE // 2 - 8   # Disc radius
TOP_PANEL    = CELL_SIZE    # Height of the top hover-preview + status bar
BOTTOM_PANEL = 90           # Height of the info / restart panel
BOARD_W      = COLS * CELL_SIZE
BOARD_H      = ROWS * CELL_SIZE
WIN_W        = BOARD_W
WIN_H        = TOP_PANEL + BOARD_H + BOTTOM_PANEL
FPS          = 60

# =============================================================================
# Colour palette  (RGB)
# =============================================================================
CLR_BG           = (15,  20,  40)    # Deep navy – window background
CLR_BOARD        = (20,  60, 140)    # Royal blue – board body
CLR_BOARD_SHADOW = (10,  40, 100)    # Slightly darker for depth
CLR_HOLE         = (15,  20,  40)    # Same as BG so holes look cut-through
CLR_HUMAN        = (220,  60,  60)   # Red – human disc
CLR_AI           = (255, 210,  40)   # Yellow – AI disc
CLR_HOVER_H      = (220,  60,  60, 160)   # Semi-transparent hover (human)
CLR_HOVER_AI     = (255, 210,  40, 160)   # Semi-transparent hover (AI)
CLR_PANEL        = (10,  15,  35)    # Bottom panel background
CLR_WHITE        = (240, 245, 255)
CLR_GRAY         = (120, 130, 150)
CLR_WIN_OVERLAY  = (0,   0,   0, 180)     # Semi-transparent win overlay
CLR_BTN          = (40, 180, 100)    # Restart button green
CLR_BTN_HOVER    = (60, 210, 120)


# =============================================================================
# Helper: load / generate fonts
# =============================================================================
def _load_fonts():
    """
    Load pygame fonts.  We try system fonts first; fall back to the
    default pygame font if none are available.

    Returns
    -------
    dict  font_name → pygame.font.Font object
    """
    pygame.font.init()
    candidates = ["consolas", "couriernew", "ubuntumono", "dejavusansmono"]

    def best_font(size):
        for name in candidates:
            f = pygame.font.SysFont(name, size, bold=True)
            if f:
                return f
        return pygame.font.Font(None, size)

    return {
        "title":   best_font(38),
        "status":  best_font(30),
        "big":     best_font(52),
        "medium":  best_font(34),
        "small":   best_font(24),
        "btn":     best_font(28),
    }


# =============================================================================
# Drawing functions
# =============================================================================

def draw_board_background(surface: pygame.Surface):
    """
    Draw the blue board rectangle with rounded corners and a subtle
    drop shadow to give it a 3-D feel.
    """
    shadow_rect = pygame.Rect(4, TOP_PANEL + 4, BOARD_W, BOARD_H)
    pygame.draw.rect(surface, CLR_BOARD_SHADOW, shadow_rect, border_radius=12)

    board_rect = pygame.Rect(0, TOP_PANEL, BOARD_W, BOARD_H)
    pygame.draw.rect(surface, CLR_BOARD, board_rect, border_radius=12)


def draw_discs(surface: pygame.Surface, board: np.ndarray):
    """
    Render all placed discs and the empty holes.

    For each cell we draw a filled circle:
      • EMPTY  → CLR_HOLE  (makes it look like a hole in the board)
      • HUMAN  → CLR_HUMAN (red)
      • AI     → CLR_AI    (yellow)

    We also add a small highlight arc at the top of each disc to
    simulate a glossy plastic look.
    """
    for r in range(ROWS):
        for c in range(COLS):
            cx = c * CELL_SIZE + CELL_SIZE // 2
            cy = TOP_PANEL + r * CELL_SIZE + CELL_SIZE // 2

            cell = board[r][c]
            if cell == EMPTY:
                colour = CLR_HOLE
            elif cell == HUMAN:
                colour = CLR_HUMAN
            else:
                colour = CLR_AI

            # Main disc
            pygame.draw.circle(surface, colour, (cx, cy), RADIUS)

            # Glossy highlight (only for placed discs)
            if cell != EMPTY:
                hilight_col = tuple(min(255, v + 60) for v in colour)
                pygame.draw.circle(
                    surface, hilight_col,
                    (cx - RADIUS // 4, cy - RADIUS // 4),
                    RADIUS // 4
                )

            # Thin border around every hole to separate cells
            pygame.draw.circle(surface, CLR_BOARD_SHADOW, (cx, cy), RADIUS, 2)


def draw_hover(surface: pygame.Surface, col: int, current_player: int):
    """
    Draw a semi-transparent preview disc above the column the mouse is
    hovering over, so the player can see where their piece will drop.

    Parameters
    ----------
    surface        : pygame display surface
    col            : column index under the mouse cursor
    current_player : HUMAN or AI
    """
    if col < 0 or col >= COLS:
        return

    colour = CLR_HUMAN if current_player == HUMAN else CLR_AI
    cx     = col * CELL_SIZE + CELL_SIZE // 2
    cy     = TOP_PANEL // 2

    # Use a temporary surface with per-pixel alpha for transparency
    hover_surf = pygame.Surface((RADIUS * 2 + 4, RADIUS * 2 + 4), pygame.SRCALPHA)
    r, g, b    = colour
    pygame.draw.circle(hover_surf, (r, g, b, 160),
                       (RADIUS + 2, RADIUS + 2), RADIUS)
    surface.blit(hover_surf, (cx - RADIUS - 2, cy - RADIUS - 2))


def draw_bottom_panel(surface: pygame.Surface, fonts: dict,
                      current_player: int, ai_thinking: bool):
    """
    Draw the status bar below the board showing whose turn it is.

    Parameters
    ----------
    surface        : pygame display surface
    fonts          : font dictionary
    current_player : HUMAN or AI
    ai_thinking    : True while the AI is computing its move
    """
    panel_rect = pygame.Rect(0, TOP_PANEL + BOARD_H, WIN_W, BOTTOM_PANEL)
    pygame.draw.rect(surface, CLR_PANEL, panel_rect)

    # Colour-coded turn dot
    dot_x, dot_y = 36, TOP_PANEL + BOARD_H + BOTTOM_PANEL // 2
    dot_col      = CLR_HUMAN if current_player == HUMAN else CLR_AI
    pygame.draw.circle(surface, dot_col, (dot_x, dot_y), 14)

    # Status text
    if ai_thinking:
        msg = "AI is thinking..."
        txt_col = CLR_GRAY
    elif current_player == HUMAN:
        msg     = "Your Turn  (Red)"
        txt_col = CLR_HUMAN
    else:
        msg     = "AI's Turn  (Yellow)"
        txt_col = CLR_AI

    text_surf = fonts["status"].render(msg, True, txt_col)
    surface.blit(text_surf, (70, TOP_PANEL + BOARD_H + BOTTOM_PANEL // 2
                              - text_surf.get_height() // 2))


def draw_end_overlay(surface: pygame.Surface, fonts: dict,
                     message: str, sub: str,
                     btn_rect: pygame.Rect, btn_hover: bool):
    """
    Draw a semi-transparent overlay with the game-over message and a
    Restart button.

    Parameters
    ----------
    surface   : pygame display surface
    fonts     : font dictionary
    message   : primary line (e.g. "You Win! 🎉")
    sub       : secondary line (e.g. "Play again?")
    btn_rect  : pygame.Rect for the restart button
    btn_hover : True if the mouse is over the button
    """
    # Dark overlay
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (0, 0))

    # Big centred message
    msg_surf = fonts["big"].render(message, True, CLR_WHITE)
    surface.blit(msg_surf,
                 (WIN_W // 2 - msg_surf.get_width() // 2, WIN_H // 2 - 80))

    sub_surf = fonts["medium"].render(sub, True, CLR_GRAY)
    surface.blit(sub_surf,
                 (WIN_W // 2 - sub_surf.get_width() // 2, WIN_H // 2 - 20))

    # Restart button
    btn_colour = CLR_BTN_HOVER if btn_hover else CLR_BTN
    pygame.draw.rect(surface, btn_colour, btn_rect, border_radius=10)
    btn_text = fonts["btn"].render("↺  Restart", True, CLR_WHITE)
    surface.blit(btn_text,
                 (btn_rect.centerx - btn_text.get_width() // 2,
                  btn_rect.centery - btn_text.get_height() // 2))


def draw_title(surface: pygame.Surface, fonts: dict):
    """Draw a small 'CONNECT 4' label at the very top of the window."""
    title = fonts["small"].render("C O N N E C T   4", True, CLR_GRAY)
    surface.blit(title, (WIN_W // 2 - title.get_width() // 2, 8))


# =============================================================================
# Animated disc drop
# =============================================================================
def animate_drop(surface: pygame.Surface, board: np.ndarray,
                 fonts: dict, col: int, piece: int,
                 clock: pygame.time.Clock):
    """
    Animate a disc falling from the top of the board to its landing row.

    We temporarily draw the disc at increasing y-positions without
    updating the board array, then write the final position once the
    animation finishes.

    Parameters
    ----------
    surface : pygame display surface
    board   : current board (read-only during animation)
    fonts   : font dictionary
    col     : column where the disc drops
    piece   : HUMAN or AI
    clock   : pygame clock (for frame-rate control)
    """
    landing_row = get_next_open_row(board, col)
    if landing_row is None or landing_row < 0:
        return

    target_y  = TOP_PANEL + landing_row * CELL_SIZE + CELL_SIZE // 2
    current_y = TOP_PANEL // 2
    colour    = CLR_HUMAN if piece == HUMAN else CLR_AI
    cx        = col * CELL_SIZE + CELL_SIZE // 2
    speed     = 12   # pixels per frame; increase for faster drop

    while current_y < target_y:
        current_y = min(current_y + speed, target_y)
        speed    += 2   # Accelerate – simulates gravity

        # Redraw background elements
        surface.fill(CLR_BG)
        draw_title(surface, fonts)
        draw_board_background(surface)
        draw_discs(surface, board)

        # Draw falling disc
        pygame.draw.circle(surface, colour, (cx, int(current_y)), RADIUS)
        hilight = tuple(min(255, v + 60) for v in colour)
        pygame.draw.circle(surface, hilight,
                           (cx - RADIUS // 4, int(current_y) - RADIUS // 4),
                           RADIUS // 4)

        pygame.display.flip()
        clock.tick(FPS)


# =============================================================================
# Main game loop
# =============================================================================
def run_game():
    """
    Initialise Pygame and run the main game loop.

    The loop handles:
      1. Event processing (mouse move, click, quit)
      2. Human turn  – waits for a mouse click on a valid column
      3. AI turn     – calls get_best_move() in a background thread so
                       the window stays responsive while the AI thinks
      4. Win / draw  – displays overlay; waits for Restart click
    """
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Connect 4  –  Human vs AI")
    clock  = pygame.time.Clock()
    fonts  = _load_fonts()

    # ── Restart button rect (centred in the overlay) ──────────────────────
    BTN_W, BTN_H = 200, 54
    btn_rect = pygame.Rect(
        WIN_W // 2 - BTN_W // 2,
        WIN_H // 2 + 50,
        BTN_W, BTN_H
    )

    # ── Per-game state ─────────────────────────────────────────────────────
    def init_state():
        return {
            "board":          create_board(),
            "current_player": HUMAN,        # Human always goes first
            "game_over":      False,
            "end_message":    "",
            "end_sub":        "",
            "hover_col":      -1,
            "ai_thinking":    False,
            "ai_col":         None,         # Result from AI thread
        }

    state = init_state()

    # ── AI threading helpers ───────────────────────────────────────────────
    def ai_worker(board_copy):
        """Run in a thread; writes result back to state["ai_col"]."""
        col = get_best_move(board_copy)
        state["ai_col"] = col

    # ══════════════════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════════════════
    while True:
        board          = state["board"]
        current_player = state["current_player"]
        game_over      = state["game_over"]

        # ── 1. EVENT HANDLING ──────────────────────────────────────────────
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Mouse movement – update hover column
            if event.type == pygame.MOUSEMOTION:
                mx, _ = event.pos
                state["hover_col"] = mx // CELL_SIZE

                # Handle restart button hover
                if game_over:
                    pass   # handled in drawing section

            # Mouse click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Restart button
                if game_over and btn_rect.collidepoint(mx, my):
                    state = init_state()
                    continue

                # Human move (only when it's the human's turn and game is live)
                if (not game_over and
                        current_player == HUMAN and
                        not state["ai_thinking"]):
                    col = mx // CELL_SIZE
                    if 0 <= col < COLS and is_valid_location(board, col):
                        # Animate the drop
                        animate_drop(screen, board, fonts, col, HUMAN, clock)
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, HUMAN)

                        if winning_move(board, HUMAN):
                            state["game_over"]   = True
                            state["end_message"] = "You Win!  🎉"
                            state["end_sub"]     = "Congratulations!"
                        elif is_draw(board):
                            state["game_over"]   = True
                            state["end_message"] = "It's a Draw!"
                            state["end_sub"]     = "Well played both sides."
                        else:
                            state["current_player"] = AI
                            # Kick off AI in background thread
                            state["ai_thinking"] = True
                            state["ai_col"]      = None
                            t = threading.Thread(
                                target=ai_worker,
                                args=(board.copy(),),
                                daemon=True
                            )
                            t.start()

        # ── 2. AI MOVE RESOLUTION ──────────────────────────────────────────
        if state["ai_thinking"] and state["ai_col"] is not None:
            col = state["ai_col"]
            state["ai_thinking"] = False

            if col is not None and is_valid_location(board, col):
                animate_drop(screen, board, fonts, col, AI, clock)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI)

                if winning_move(board, AI):
                    state["game_over"]   = True
                    state["end_message"] = "AI Wins!  🤖"
                    state["end_sub"]     = "Better luck next time."
                elif is_draw(board):
                    state["game_over"]   = True
                    state["end_message"] = "It's a Draw!"
                    state["end_sub"]     = "Well played both sides."
                else:
                    state["current_player"] = HUMAN

        # ── 3. RENDERING ───────────────────────────────────────────────────
        screen.fill(CLR_BG)
        draw_title(screen, fonts)
        draw_board_background(screen)
        draw_discs(screen, board)

        # Hover preview (only when game is live and it's human's turn)
        if (not game_over and
                current_player == HUMAN and
                not state["ai_thinking"]):
            draw_hover(screen, state["hover_col"], HUMAN)

        draw_bottom_panel(screen, fonts, current_player, state["ai_thinking"])

        # End-game overlay
        if state["game_over"]:
            mx, my      = pygame.mouse.get_pos()
            btn_hovered = btn_rect.collidepoint(mx, my)
            draw_end_overlay(
                screen, fonts,
                state["end_message"], state["end_sub"],
                btn_rect, btn_hovered
            )

        pygame.display.flip()
        clock.tick(FPS)
