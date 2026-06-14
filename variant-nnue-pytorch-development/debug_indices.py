#!/usr/bin/env python3
"""Print HalfKAv2^ active feature indices for a Fairy-Stockfish Janggi FEN.

Use this together with the engine-side `debug-indices` UCI command. Example:

    python3 debug_indices.py "rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1"

The script mirrors Fairy-Stockfish's HalfKAv2Variants feature layout for a
9x10 Janggi board: variant square = rank * 9 + file with the first
FEN row as rank 0, black perspective flips rank only, and king squares are
bucketed through the 3x3 palace map.
"""

import argparse
from dataclasses import dataclass

import chess

import halfka_v2
import variant

# Fairy-Stockfish's Janggi HalfKAv2 active indices use this 0-based NNUE
# piece-plane order. The king is merged into the final single king plane.
PIECE_TO_TYPE = {
    "r": 0,  # ROOK
    "c": 1,  # JANGGI_CANNON
    "p": 2,  # SOLDIER
    "n": 3,  # KNIGHT / horse
    "b": 4,  # JANGGI_ELEPHANT
    "a": 5,  # WAZIR / advisor/guard
    "k": 6,  # KING
}


@dataclass(frozen=True)
class Piece:
    square: int
    piece_type: int
    color: bool
    symbol: str


def parse_janggi_fen(fen: str):
    board = fen.split()[0]
    rows = board.split("/")
    if len(rows) != variant.RANKS:
        raise ValueError(f"expected {variant.RANKS} board rows, got {len(rows)}")

    pieces = []
    kings = {chess.WHITE: None, chess.BLACK: None}

    for row_idx, row in enumerate(rows):
        # Fairy-Stockfish's variant-square compression maps the first Janggi FEN
        # row to variant rank 0, not to the highest rank as python-chess does.
        rank = row_idx
        file_idx = 0
        i = 0
        while i < len(row):
            ch = row[i]
            if ch.isdigit():
                # Handle multi-digit empties defensively, although Janggi uses 1..9.
                j = i
                while j < len(row) and row[j].isdigit():
                    j += 1
                file_idx += int(row[i:j])
                i = j
                continue

            key = ch.lower()
            if key not in PIECE_TO_TYPE:
                raise ValueError(f"unsupported Janggi piece '{ch}' in FEN")
            if file_idx >= variant.FILES:
                raise ValueError(f"too many files in row '{row}'")

            sq = rank * variant.FILES + file_idx
            # Match the engine's debug-indices output for janggimodern: lowercase
            # startpos pieces are Color::WHITE/own for the first perspective,
            # uppercase pieces are Color::BLACK/opponent.
            color = chess.WHITE if ch.islower() else chess.BLACK
            piece_type = PIECE_TO_TYPE[key]
            pieces.append(Piece(sq, piece_type, color, ch))
            if key == "k":
                kings[color] = sq

            file_idx += 1
            i += 1

        if file_idx != variant.FILES:
            raise ValueError(f"row '{row}' has {file_idx} files, expected {variant.FILES}")

    if kings[chess.WHITE] is None or kings[chess.BLACK] is None:
        raise ValueError("FEN must contain one white king and one black king")

    return pieces, kings


def active_indices_for_perspective(pieces, kings, perspective: bool):
    oriented_king_sq = halfka_v2.orient(perspective, kings[perspective])
    king_bucket = halfka_v2.map_king(oriented_king_sq)
    indices = [
        halfka_v2.halfka_idx(
            perspective,
            king_bucket,
            piece.square,
            piece.piece_type,
            piece.color,
        )
        for piece in pieces
    ]
    return indices


def main():
    parser = argparse.ArgumentParser(description="Print Janggi HalfKAv2^ active indices for a FEN.")
    parser.add_argument("fen", help="Fairy-Stockfish Janggi FEN, or 'startpos'")
    parser.add_argument(
        "--startpos-fen",
        default="rnba1abnr/4k4/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBA1ABNR w - - 0 1",
        help="FEN used when the positional FEN argument is 'startpos'",
    )
    args = parser.parse_args()

    fen = args.startpos_fen if args.fen == "startpos" else args.fen
    pieces, kings = parse_janggi_fen(fen)

    print(f"debug-indices fen {fen}")
    print(f"debug-indices feature-dimensions {halfka_v2.NUM_INPUTS}")
    print(f"debug-indices constants files {variant.FILES} ranks {variant.RANKS} squares {variant.SQUARES} king_squares {variant.KING_SQUARES}")

    for perspective, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        indices = active_indices_for_perspective(pieces, kings, perspective)
        print(f"debug-indices {name} count {len(indices)}")
        print(f"debug-indices {name} indices", *indices)
        print(f"debug-indices {name} sorted", *sorted(indices))


if __name__ == "__main__":
    main()
