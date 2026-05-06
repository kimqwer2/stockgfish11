RANKS = 10
FILES = 9
SQUARES = RANKS * FILES
# Janggi kings are restricted to the 3x3 palace after rank-flip orientation.
KING_SQUARES = 9
# Packed Janggi NNUE plane order follows Fairy-Stockfish's active HalfKAv2
# indices for janggimodern: rook, cannon, soldier, horse/knight, elephant,
# advisor/guard, king.
PIECE_TYPES = 7
PIECES = 2 * PIECE_TYPES
USE_POCKETS = False
POCKETS = 2 * FILES if USE_POCKETS else 0

PIECE_VALUES = {
    0 : 1276,  # rook
    1 : 800,   # cannon
    2 : 200,   # soldier
    3 : 520,   # horse / knight
    4 : 340,   # elephant
    5 : 400,   # advisor / guard
}
