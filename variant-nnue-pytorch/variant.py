RANKS = 10
FILES = 9
SQUARES = RANKS * FILES
# Janggi kings are restricted to the 3x3 palace after rank-flip orientation.
KING_SQUARES = 9
# Packed Janggi piece order follows Fairy-Stockfish Variant::pieceIndex:
# horse/knight, rook, advisor, cannon, soldier, elephant, king.
PIECE_TYPES = 7
PIECES = 2 * PIECE_TYPES
USE_POCKETS = False
POCKETS = 2 * FILES if USE_POCKETS else 0

PIECE_VALUES = {
    0 : 520,   # horse / knight
    1 : 1276,  # rook
    2 : 400,   # advisor / guard
    3 : 800,   # cannon
    4 : 200,   # soldier
    5 : 340,   # elephant
}
