from enum import IntEnum


PIECE_TYPES = 14


_COLUMN_TO_INT = {
    c:num for c, num in zip("abcdefgh", range(0, 8))
}


class Player(IntEnum):
    WHITE = 1
    BLACK = 2


class Piece(IntEnum):
    PAWN_W = 1
    PAWN_B = 2
    ROOK_W = 3
    ROOK_B = 4
    KNIGHT_W = 5
    KNIGHT_B = 6
    BISHOP_W = 7
    BISHOP_B = 8
    QUEEN_W = 9
    QUEEN_B = 10
    KING_W = 11
    KING_B = 12
    EN_PASSANT_SPACE = 13

    @classmethod
    def from_fen(cls, letter: chr, default = None):
        return {
            'p': Piece.PAWN_B,
            'P': Piece.PAWN_W,
            'r': Piece.ROOK_B,
            'R': Piece.ROOK_W,
            'n': Piece.KNIGHT_B,
            'N': Piece.KNIGHT_W,
            'b': Piece.BISHOP_B,
            'B': Piece.BISHOP_W,
            'q': Piece.QUEEN_B,
            'Q': Piece.QUEEN_W,
            'k': Piece.KING_B,
            'K': Piece.KING_W,
        }.get(letter, default)


class BitBoard:
    def __init__(self):
        self.boards = [0]*PIECE_TYPES
        self.next_to_move = Player.WHITE
        self.castling_status = set()  # Just add pieces for QUEEN_B, etc.
        self.halfstep_count = 0
        self.fullstep_count = 0

    def make_move(self, mv: str):
        # NOTE: This does not handle castling yet and probably screws up clearing en-passant.
        # Convert the pair into from/to.
        from_x = _COLUMN_TO_INT[mv[0].lower()]
        from_y = int(mv[1])-1
        to_x = _COLUMN_TO_INT[mv[2].lower()]
        to_y = int(mv[3])-1
        promote_to = None
        if len(mv) > 3:
            promote_to = Piece.from_fen(mv[4])

        # Clear piece at moving spot.
        moving_piece = self.get_piece(from_x, from_y)
        self.clear_piece(from_x, from_y)
        if promote_to is None:
            self.set_piece(to_x, to_y, moving_piece)
        else:
            self.set_piece(to_x, to_y, promote_to)

    @classmethod
    def from_fen(cls, fen_string: str):
        # rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
        # 4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35
        new_board = cls()
        board_layout, current_player, castle_status, en_passant, halfmove_clock, fullmove_clock = fen_string.split(" ")
        board_rows = board_layout.split("/")
        for y, row in enumerate(reversed(board_rows)):
            x = 0
            while row:
                if row[0].isdigit():
                    x += int(row[0])
                else:
                    piece = Piece.from_fen(row[0])
                    new_board.set_piece(x, y, piece)
                    x += 1
                row = row[1:]
        
        if current_player == 'w':
            new_board.next_to_move = Player.WHITE
        elif current_player == 'b':
            new_board.next_to_move = Player.BLACK
        else:
            raise Exception(f"Bad parse.  Starting player unrecognized: {current_player}")

        # Parse castling here.
        for character in castle_status:
            new_board.castling_status.add(Piece.from_fen(character))

        # Parse en_passant here

        new_board.halfstep_count = int(halfmove_clock)
        new_board.fullmove_count = int(fullmove_clock)
        
        return new_board

    def get_piece(self, x, y):
        assert(0 <= x < 8 and 0 <= y < 8)
        # X and Y should be in the range 0-7 inclusive.
        # a1 is bottom-left, bit zero.
        # h8 is the top-right, bit 63.
        idx = (1 << x+y*8)
        for i in range(1, PIECE_TYPES):
            if self.boards[i] & idx:
                return Piece(i)
        return None
    
    def set_piece(self, x, y, piece: Piece, clear_previous=True):
        assert(x >= 0 and x < 8 and y >= 0 and y < 8)
        idx = (1 << x+y*8)
        if clear_previous:
            self.clear_piece(x, y)
        self.boards[int(piece)] |= idx

    def clear_piece(self, x, y):
        assert (x >= 0 and x < 8 and y >= 0 and y < 8)
        idx = (1 << x + y * 8)
        clear_mask = 0xFFFF_FFFF_FFFF_FFFF & ~idx
        for i in range(0, PIECE_TYPES):
            self.boards[i] &= clear_mask
