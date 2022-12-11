import csv
import gzip
import math

import numpy
import torch
from stockfish import Stockfish, StockfishException
from torch.utils.data import Dataset
from tqdm import tqdm

from board import BitBoard, Piece, Player, PIECE_TYPES
from config import LICHESS_DATASET_PATH, STOCKFISH_EXECUTABLE_PATH


BOARD_VECTOR_SIZE = 8*8*PIECE_TYPES+7


def _bitboard_to_ndarray(board: BitBoard):
    """Note that the bitboard does not include a bit-flag for the current player, castling status, etc."""
    output = numpy.zeros((14, 8, 8), dtype=float)
    for y in range(0, 8):
        for x in range(0, 8):
            ptype = board.get_piece(x, y)
            if ptype is None:
                output[0, y, x] = 1.0
            else:
                output[int(ptype), y, x] = 1.0
    return output


def _ndarray_to_board(arr):
    board = BitBoard()
    for piece_type in range(0, PIECE_TYPES):
        ptype = Piece(piece_type)
        for y in range(0, 8):
            for x in range(0, 8):
                if arr[piece_type, y, x] > 0.5:
                    board.set_piece(x, y, ptype)
    return board


def bitboard_to_tensor(board: BitBoard) -> torch.Tensor:
    arr = _bitboard_to_ndarray(board)
    flat_array = numpy.reshape(arr, (PIECE_TYPES*8*8,), order='C')
    # We need to add 7 extra spaces for next to play, castling status (4 bits), half-move count, and full-move count.
    # En-passant is encoded in the bitboard separately.
    game_info = numpy.zeros((7,), dtype=float)
    game_info[0] = 0.0 if board.next_to_move == Player.WHITE else 1.0
    # KQkq
    game_info[1] = (Piece.KING_W in board.castling_status) * 1.0
    game_info[2] = (Piece.QUEEN_W in board.castling_status) * 1.0
    game_info[3] = (Piece.KING_B in board.castling_status) * 1.0
    game_info[4] = (Piece.QUEEN_B in board.castling_status) * 1.0

    game_info[5] = board.halfstep_count
    game_info[6] = board.fullstep_count / 50.0
    return torch.tensor(numpy.hstack([flat_array, game_info]))


def tensor_to_bitboard(arr: torch.Tensor) -> BitBoard:
    array = arr.detach().numpy()
    # Cut off the last eight.
    board = _ndarray_to_board(array[:PIECE_TYPES*8*8].resize(PIECE_TYPES, 8, 8))
    if array[-7] > 0.5:
        board.next_to_move = Player.BLACK
    else:
        board.next_to_move = Player.WHITE
    if array[-6] > 0.5:
        board.castling_status.add(Piece.KING_W)
    if array[-5] > 0.5:
        board.castling_status.add(Piece.QUEEN_W)
    if array[-4] > 0.5:
        board.castling_status.add(Piece.KING_B)
    if array[-3] > 0.5:
        board.castling_status.add(Piece.QUEEN_B)
    board.halfstep_count = int(array[-2])
    board.fullstep_count = int(array[-1]*50)
    return board


class LichessPuzzleDataset(Dataset):
    def __init__(self, cap_data: int = None):
        # See https://database.lichess.org/#puzzles
        fin = gzip.open(LICHESS_DATASET_PATH, 'rt')
        cin = csv.DictReader(fin)
        cin.fieldnames = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningFamily", "OpeningVariation"]
        little_fish = Stockfish(path=STOCKFISH_EXECUTABLE_PATH, depth=3, parameters={"Hash": 64, "Threads": 1})
        #big_fish = Stockfish(path=STOCKFISH_EXECUTABLE_PATH, depth=18, parameters={"Hash": 2048, "Threads": 4, "Minimum Thinking Time": 30})
        self.fen_start = list()
        self.puzzle_start = list()
        self.moves = list()
        self.rating = list()
        self.rating_stddev = list()
        self.popularity = list()
        self.number_of_plays = list()
        self.themes = list()
        self.evaluations = list()  # stockfish.get_evaluation() -> {"type":"cp", "value":12} or {"type":"mate", "value":-3}

        for row in tqdm(cin):
            puzzle_id = row['PuzzleId']
            fen = row['FEN']
            moves = row['Moves']
            rating = row['Rating']
            rating_stddev = row['RatingDeviation']
            popularity = row['Popularity']
            number_of_plays = row['NbPlays']
            themes = row['Themes']
            url = row['GameUrl']
            opening = row['OpeningFamily']
            opening_var = row['OpeningVariation']

            self.fen_start.append(fen)  # The fen starts with the enemy turn.  We apply the first move.
            try:
                little_fish.set_fen_position(fen)
                little_fish.make_moves_from_current_position([moves.split(" ")[0]])  # sf.make_moves_from_current_position(["g4d7", "a8b8", "f1d1"])
                self.puzzle_start.append(little_fish.get_fen_position())
                self.moves.append(moves)  # Moves is a list
                self.rating.append(int(rating))
                self.rating_stddev.append(int(rating_stddev))
                self.popularity.append(float(popularity) / 100.0)
                self.number_of_plays.append(int(number_of_plays))
                self.themes.append(themes)
                stockfish_score = 0.0
                #big_fish.set_fen_position(little_fish.get_fen_position())
                #evaluation = big_fish.get_evaluation()
                #if "type" in evaluation:
                #    # This is unbounded on both sides.  It's the value in 'centipawns', so we have to pass through a sigmoid.
                #    stockfish_score = math.tanh(evaluation["value"])
                #elif "mate" in evaluation:
                #    stockfish_score = math.copysign(1.0, evaluation["value"])
                self.evaluations.append(stockfish_score)

                # DEBUG: Chop off dataset after 100 examples so we can get a test flow.
                if cap_data and len(self.puzzle_start) > cap_data:
                    break
            except StockfishException:
                print(f"Stockfish crashed after trying {fen}.  Reinitializing and skipping.")
                little_fish = Stockfish(path=STOCKFISH_EXECUTABLE_PATH, depth=3, parameters={"Hash": 64, "Threads": 1})
                #big_fish = Stockfish(path=STOCKFISH_EXECUTABLE_PATH, depth=18, parameters={"Hash": 2048, "Threads": 4, "Minimum Thinking Time": 30})

    def __len__(self):
        return len(self.puzzle_start)

    def __getitem__(self, idx):
        """Produce pairs of three outputs: a board vector of size 8*8*PIECE_TYPES+8, popularity (+1 to -1), and a stockfish evaluation from +1 to -1."""
        return bitboard_to_tensor(BitBoard.from_fen(self.puzzle_start[idx])), self.popularity[idx], self.evaluations[idx]