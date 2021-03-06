"""Defines a chessboard class"""
from __future__ import annotations

from copy import deepcopy
from typing import Tuple


class IllegalMoveError(Exception):
    pass


class Board:
    """
    Pieces are represented by characters in the following way:
    king: k
    queen: q
    rook: r
    knight: n
    bishop: b
    pawn: p
    With a preceding 'b' or 'w' to indicate the colour of the piece.
    A blank square is indicated by an empty string ('')

    self.board is a list of 8 lists, each of which corresponds to a row of the board. self.board[0] corresponds to
    the first row of a physical chess board (i.e. where the white major and minor pieces start the game), self.board[1]
    to the second row, and so on.
    """
    WHITE_PIECES = ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr', 'wp']
    BLACK_PIECES = ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br', 'bp']

    def __init__(self, to_move: str = 'w'):
        self.board = [
            ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr'],
            ['wp'] * 8,
            [''] * 8,
            [''] * 8,
            [''] * 8,
            [''] * 8,
            ['bp'] * 8,
            ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br']
        ]
        self.to_move = to_move
        self.en_passsant_square = ''
        self.moved_pieces = {  # this data is important for castling rules
            'w_king_rook': False,
            'w_king': False,
            'w_queen_rook': False,
            'b_king_rook': False,
            'b_king': False,
            'b_queen_rook': False
        }

    def __iter__(self):
        return BoardIterator(self)

    @staticmethod
    def convert_coordinate_to_index(c: str) -> Tuple[int, int]:
        """
        Converts a coordinate from algebraic form to index form (eg a4 -> 03)

        :param c: must be a two character algebraic chess notation
        :raises AssertionError: if start is the wrong length, refers to nonexistent squares, or is formatted incorrectly
        """
        cols = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

        assert len(c) == 2, f'{c} is not two characters long.'
        assert c[0] in cols, f'{c} refers to a column which does not exist'
        assert c[1].isdigit(), f'{c} must refer to a row using a number'
        assert 1 <= int(c[1]) <= 8, f'{c} refers to a nonexistent row'

        # the column gets converted from a letter to a number, the row simply decreases by 1
        return cols[c[0]], int(c[1]) - 1

    @staticmethod
    def convert_coordinate_to_algebraic(col: int, row: int) -> str:
        """
        Converts a coordinate from index form to algebraic form (eg (0, 3) -> a4)
        """
        columns = 'abcdefgh'
        return columns[col] + str(row + 1)

    def __setitem__(self, coordinate: str or Tuple[int, int], piece: str):
        """
        Modifies self.board such that the square referred to by coordinate contains piece

        :param coordinate: - a square on the chess board in algebraic form (e.g. a4, h7, g3, etc.), or
                           - a pair of integer coordinates in index form

        :param piece: a chess piece (e.g. wp, bk, wn, etc.)
        """
        if type(coordinate) == str:
            # convert_coordinate handles error checking in terms of whether the square referred to by coordinate
            # exists, but checking for move legality etc. is done elsewhere
            col, row = self.convert_coordinate_to_index(coordinate)
        elif type(coordinate) == tuple:
            row, col = coordinate
        else:
            raise ValueError(f'expected str or Tuple[int, int], got {type(coordinate)}.')
        self.board[row][col] = piece

    def __getitem__(self, coordinate: str or Tuple[int, int]):
        """
        :param coordinate: a coordinate in algebraic form, or as a tuple in index form (row, col)
        :return: the piece at coordinate ('' if the square is empty)
        """
        if type(coordinate) == str:
            col, row = self.convert_coordinate_to_index(coordinate)
        elif type(coordinate) == Tuple:
            row, col = coordinate
        else:
            raise ValueError(f'expected str or Tuple[int, int], got {type(coordinate)}.')
        return self.board[row][col]

    def print_ascii(self):
        """Prints an ASCII representation of the board. Row 1 (self.board[0]) is printed at the bottom"""
        for row in reversed(self.board):
            for piece in row:
                if piece == '':
                    print(' .', end=' ')
                else:
                    print(piece, end=' ')
            print('\n')

    def is_piece(self, p: str, colour: str = None):
        """
        :param colour: 'w' or 'b'
        :param p: a piece
        :return: True if p is a piece, or if colour is given, True if p is a piece of that colour. False otherwise
        """

        if colour is None:
            return p in self.WHITE_PIECES + self.BLACK_PIECES
        else:
            assert colour == 'b' or colour == 'w', f'Colour must be either \'w\' or \'b\', but it was {colour}'

            if colour == 'b':
                return p in self.BLACK_PIECES
            elif colour == 'w':
                return p in self.WHITE_PIECES

        return False

    def check_diagonal(self, row: int, col: int, left_right: str, up_down: str, length: int = 8) -> str:
        """
        Checks along the given diagonal from the square given by row, col, returning the fist piece it finds, or
        returning '' if it doesn't find a piece. Does not check the given square.

        :param length: the number of squares to be checked
        :param row: the starting row, in index form
        :param col: the starting column, in index form
        :param left_right: 'l' or 'r', depending on the direction desired
        :param up_down: 'u' or 'd', depending on the direction desired
        :return: the piece found
        """
        assert left_right == 'l' or left_right == 'r'
        assert up_down == 'u' or up_down == 'd'

        col_diff = -1 if left_right == 'l' else 1
        row_diff = -1 if up_down == 'd' else 1
        res = ''

        for i in range(length):
            # move to next square
            row += row_diff
            col += col_diff

            # check if square is on board
            if (not 0 <= row <= 7) or (not 0 <= col <= 7):
                break

            # check for piece
            res = self[(row, col)]
            if res:  # if res is a piece
                break

        return res

    def is_legal_move(self, start: str, end: str) -> Board:
        """
        Checks whether a move is legal in terms of whether or not the piece being moved can move from start to end. Does
        not check for whether the player is in check, or stalemate, or anything like that. If legal, returns a Board
        object which represents self.board after the move has been made (this Board object can be used to check other
        rules, e.g. is the king in check after the move, stalemate, etc.)

        Each piece type has its own function, which checks if that piece can move in the direction described by the move
        from start to end. Each function returns True if the piece can move in such a way, and otherwise raises a
        descriptive IllegalMoveError. The main body of this function checks if the starting and ending squares are such
        that the move could be legal, and then calls the appropriate piece function.

        If the piece function returns True, then this function creates a board object, which can be used to see if the
        move has created an illegal board state.

        This function is only for single piece moves (i.e. not castling)

        :param start: the starting position of piece
        :param end: the final position of piece
        :raises IllegalMoveError: if the move is not legal
        :return: a board object representing self.board after the move
        """

        def pawn_direction(s_col: int, s_row: int, e_col: int, e_row: int) -> bool:
            # check we're moving forwards
            if self.to_move == 'w' and s_row >= e_row:  # the row number should get larger as white, not smaller
                raise IllegalMoveError(f'Pawns can only move forward ({start, end})')
            if self.to_move == 'b' and s_row <= e_row:  # the row number should get smaller as black, not larger
                raise IllegalMoveError(f'Pawns can only move forward ({start, end})')

            # moving straight ahead
            if s_col == e_col:
                if abs(s_row - e_row) > 2:  # pawns can move at most 2 spaces
                    raise IllegalMoveError(f'Pawns cannot move more than two spaces {start, end}')
                if self.board[e_row][e_col] != '':  # moving straight ahead we can only move to an empty square
                    raise IllegalMoveError(f'Pawns cannot move straight into a piece {start, end}')
                if abs(e_row - s_row) == 2:
                    # if we're moving two spaces we must start from the pawns starting location (we've already checked
                    # that the pawn is moving in the correct direction).
                    if not (s_row == 1 or s_row == 6):
                        raise IllegalMoveError(f'Pawns can only move two spaces from the starting rank {start, end}')
                    # the square between our start and end must also be empty
                    mid_row = (s_row + e_row) // 2
                    if self.board[mid_row][s_col] != '':
                        raise IllegalMoveError(f'Pawns cannot move through other pieces {start, end}')

                    # when moving two spaces, we create an en passant opportunity
                    nonlocal new_en_passant
                    new_en_passant = True

                # if we have not returned False already, then the move is legal
                return True

            # capturing diagonally
            elif abs(s_col - s_row) == 1 and abs(s_row - s_col) == 1:
                # check for en passant
                if self.en_passsant_square:  # this will be non-empty only when an en passant is available
                    enp_col, enp_row = self.convert_coordinate_to_index(self.en_passsant_square)
                    if e_col == enp_col and e_row == enp_row:  # this is the en passant case
                        return True

                # if it's not en passant, then we must capture a piece
                if self.is_piece(ending_piece, enemy_colour):
                    return True

            else:  # we're neither moving forward, nor capturing diagonally
                raise IllegalMoveError(f'Pawns can only move forward, or capture diagonally {start, end}')

        def knight_direction(s_col: int, s_row: int, e_col: int, e_row: int):
            if abs(s_col - e_col) == 2 and abs(s_row - e_row) == 1:
                return True
            elif abs(s_col - e_col) == 1 and abs(s_row - e_row) == 2:
                return True
            else:
                raise IllegalMoveError(f'Knights can only move in L-shapes ({start, end}')

        def bishop_direction(s_col: int, s_row: int, e_col: int, e_row: int):
            if abs(e_row - s_row) != abs(e_col - s_col):  # must move the same number of rows and columns
                raise IllegalMoveError(f'Bishops must move diagonally ({start, end}')

            # need to check that each square between s and e (exclusive, since e is checked in the main body of
            # is_legal_move

            row_step = 1 if e_row > s_row else -1
            col_step = 1 if e_col > s_col else -1

            for col, row in zip(range(s_col + col_step, e_col, col_step), range(s_row + row_step, e_row, row_step)):
                # we start at s_col+1, s_row+1 because the starting square is checked in the main body of the function
                # we don't need to check the final square because that is also done in the main function.
                if self.board[row][col] != '':
                    raise IllegalMoveError(
                        f'Bishops cannot move through other pieces (piece on {row, col}) ({start, end})')

            return True

        def rook_direction(s_col: int, s_row: int, e_col: int, e_row: int):
            if s_col == e_col:  # moving along a row
                step = (e_row - s_row) // abs(e_row - s_row)  # +/- 1 depending on the direction we're moving in
                for row in range(s_row + 1, e_row, step):
                    if self.board[row][s_col] != '':  # s_col == e_col
                        raise IllegalMoveError(
                            f'Rooks cannot move through pieces (piece on {row, s_col} ({start, end})')
                return True

            elif s_row == e_row:  # moving along a column
                step = (e_col - s_col) // abs(e_col - s_col)  # +/- 1 depending on the direction we're moving in
                for col in range(s_col + 1, e_col, step):
                    if self.board[s_row][col] != '':  # s_row == e_row
                        raise IllegalMoveError(
                            f'Rooks cannot move through pieces (piece on {s_row, col} ({start, end})')
                return True

            else:
                raise IllegalMoveError(f'Rooks can only move in straight lines ({start, end}).')

        def queen_direction(s_col: int, s_row: int, e_col: int, e_row: int):
            # queens are just rooks wearing bishop's hats
            try:
                rook_direction(s_col, s_row, e_col, e_row)
                return True
            except IllegalMoveError:
                bishop_direction(s_col, s_row, e_col, e_row)
                return True

        def king_direction(s_col: int, s_row: int, e_col: int, e_row: int):
            if abs(s_col - e_col) > 1 or abs(s_row - e_row) > 1:
                raise IllegalMoveError(f'The King can only move 1 in any direction ({start, end}')
            else:  # since the king only moves 1, and the start and end are checked in the main function
                return True

        # utility stuff
        new_en_passant = False
        starting_piece = self[start]
        ending_piece = self[end]
        piece_type = starting_piece[1]
        enemy_colour = 'w' if self.to_move == 'b' else 'b'

        # check that the player isn't trying to move an opponent's piece, or an empty square
        if starting_piece == '':
            raise IllegalMoveError(f'Cannot move from an empty square ({start})')
        if not self.is_piece(starting_piece, self.to_move):
            raise IllegalMoveError(f'Cannot move from a square occupied by an enemy piece.'
                                   f'({start} contains {starting_piece})')

        # check end isn't occupied by a piece of the same colour
        if self.is_piece(ending_piece, self.to_move):
            raise IllegalMoveError(f'Cannot move to a square occupied by your own piece ({end})')

        # check end doesn't contain a king
        if ending_piece != '' and [1] == 'k':
            raise IllegalMoveError(f'Cannot capture a king ({end}')

        # do the move
        start_col, start_row = self.convert_coordinate_to_index(start)
        end_col, end_row = self.convert_coordinate_to_index(end)

        if piece_type == 'p':
            piece_function = pawn_direction
        elif piece_type == 'n':
            piece_function = knight_direction
        elif piece_type == 'b':
            piece_function = bishop_direction
        elif piece_type == 'r':
            piece_function = rook_direction
        elif piece_type == 'k':
            piece_function = king_direction
        elif piece_type == 'q':
            piece_function = queen_direction
        else:
            raise ValueError(f'The piece {starting_piece} was not a valid piece type.')
        if piece_function(start_col, start_row, end_col, end_row):  # this will raise an error if the move is illegal
            test_board = Board(to_move=self.to_move)
            test_board.board = deepcopy(self.board)

            test_board[end] = starting_piece
            test_board[start] = ''

            if new_en_passant:
                if start[1] == '2':  # move from second rank
                    test_board.en_passsant_square = start[0] + '3'
                elif start[1] == '6':  # move from sixth rank
                    test_board.en_passsant_square = start[0] + '5'
                else:  # this should never happen
                    raise ValueError(f'Tried to assign an en passant square given the move {start, end}.')

            else:  # en passant only lasts one turn
                test_board.en_passsant_square = ''

        return test_board

    def check_square_attacked(self, row: int, col: int) -> bool:
        """checks if an enemy piece is attacking sqr"""
        pass

    def is_legal_castle(self, side: str) -> bool:
        """

        :param side: 'king' or 'queen'
        :return:
        """
        if self.to_move == 'w':
            row = 0
        else:  # colour is black
            row = 7
        col = 3  # king's starting position

        # check if the king has moved
        if self.moved_pieces[f'{self.to_move}_king']:
            raise IllegalMoveError('Cannot castle after the king has moved')
        # check if the rook has moved
        if self.moved_pieces[f'{self.to_move}_{side}_rook']:
            raise IllegalMoveError(f'Cannot castle {side}side after the {side} rook has moved')

        # check to the left for kingside castling, to the right for queenside castling
        if side == 'king':
            col_diff = -1
        else:  # side == 'queen'
            col_diff = 1

        # check if king is attacked
        if self.check_square_attacked(row, col):
            raise IllegalMoveError('Cannot castle from check.')
        # check adjacent square
        if self.check_square_attacked(row, col + col_diff):
            raise IllegalMoveError('Cannot castle through check.')
        # check next square over
        if self.check_square_attacked(row, col + col_diff + col_diff):
            raise IllegalMoveError('Cannot castle into check.')

        return True

    def move(self, start: str, end: str):
        """
        Moves the piece at position start to position end, checking for legality and promotion.

        :param start: a square in algebraic form, or an instruction to castle (O-O or O-O-O).
        :param end: a square in algebraic form, or nothing if castling
        :raises IllegalMoveError: if the move is illegal
        """

        # this will throw an error if a piece is being asked to move in an illegal way
        test_board = self.is_legal_move(start, end)

        # todo check for en passant

        # todo update self.to_move


class BoardIterator:
    """Implements an iterator for the board class. Iterates from top to bottom, left to right."""

    def __init__(self, board: Board, row=0, col=0):
        self.board = board.board
        self.row = row
        self.col = col

    def __iter__(self):
        return self

    def __next__(self):
        if self.row > 7:
            raise StopIteration
        else:
            res = self.board[self.col][self.row]
            if self.row < 7:
                self.row += 1
            elif self.col < 7:
                self.row = 0
                self.col += 1
            else:  # row == 7 and col == 7
                self.row += 1
            return res
