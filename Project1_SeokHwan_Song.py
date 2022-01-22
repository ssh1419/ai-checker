import numpy as np
import copy
from collections import namedtuple

# SeokHwan Song

Game_State = namedtuple('Games_state', ['to_move', 'result', 'board', 'moves'])


def human(game, state):
    game.show(state)
    piece = game.seek_piece(state)

    if len(piece) is not 0:
        for a, b in enumerate(piece):
            print("piece {}: location: {}".format(a, b))
        while True:
            '''
            when the input is wrong
            '''
            input_move = input('Type the number of the piece that you want to move (enter an integer):  ')

            if len(piece) > int(input_move) >= 0:
                legal_moves = game.actions(piece[int(input_move)], state)
                break

            print("That is unavailable")
            print("Type again!")
            for i, d in enumerate(piece):
                print("piece {}) location: {}".format(i, d))

        print("available moves:")
        for i, d in enumerate(legal_moves):
            print("Move {}) location: {}".format(i, d[1]))

        if len(legal_moves) is not 0:

            while True:
                '''
                when the input is wrong
                '''
                idx = input('Your move? (enter a index):  ')

                if len(legal_moves) > int(idx) >= 0:
                    move = legal_moves[int(idx)]
                    break
                print("That is unavailable")
                print("Type again!")
                print("available moves:")
                for i, d in enumerate(legal_moves):
                    print("Move {}) location: {}".format(i, d[1]))

        else:
            print('You cannot move. Computer turn.')
        return move
    else:
        print('You cannot move. Computer turn.')
        return None


def computer(game, state):
    """
    Computer part
    """
    move = alphabeta_search(state, game)
    if move is None:
        print("Computer cannot move. Your turn.")
        return None
    else:
        return move


"""
Alpha-Beta Search
"""

infi = float('inf')

def alphabeta_search(state, search, d=4, test=None, fn=None):
    piece = state.to_move
    board = state.board

    def max_value(loc, state, alpha, beta, depth):
        if test(state, depth):
            return fn(state)
        v = -infi
        for a in search.actions(loc, state):
            v = max(v, min_value(search.final_move(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(loc, state, alpha, beta, depth):
        if test(state, depth):
            return fn(state)

        v = infi
        for a in search.actions(loc, state):
            v = min(v, max_value(search.final_move(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    test = test or (lambda state, depth: depth > d)
    fn = fn or (lambda state: search.result(state, piece))

    alpha_score = -infi
    beta = infi
    best_move = None

    for i in range(1, 9):
        for j in range(1, 9):
            if board[i][j] == 'X' or board[i][j] == 'B':
                for a in search.actions((i, j), state):
                    v = min_value((i, j), search.final_move(state, a), alpha_score, beta, 1)
                    if v > alpha_score:
                        alpha_score = v
                        best_move = a

    return best_move


"""
Checkers
"""


class Checkers:
    def __init__(self):
        """
        Board!
        """

        board = np.zeros((9, 9), dtype=str)
        board[1][2] = board[1][4] = board[1][6] = board[1][8] = board[2][1] = board[2][3] = board[2][5] = board[2][7] = \
            board[3][2] = board[3][4] = board[3][6] = board[3][8] = 'O'

        board[6][1] = board[6][3] = board[6][5] = board[6][7] = board[7][2] = board[7][4] = board[7][6] = board[7][8] = \
            board[8][1] = board[8][3] = board[8][5] = board[8][7] = 'X'  # X: Black

        board[1][1] = board[1][3] = board[1][5] = board[1][7] = ' '  # "  ": Empty place
        board[2][2] = board[2][4] = board[2][6] = board[2][8] = ' '
        board[3][1] = board[3][3] = board[3][5] = board[3][7] = ' '
        board[4][2] = board[4][4] = board[4][6] = board[4][8] = ' '
        board[5][1] = board[5][3] = board[5][5] = board[5][7] = ' '
        board[4][1] = board[4][3] = board[4][5] = board[4][7] = ' '
        board[5][2] = board[5][4] = board[5][6] = board[5][8] = ' '
        board[6][2] = board[6][4] = board[6][6] = board[6][8] = ' '
        board[7][1] = board[7][3] = board[7][5] = board[7][7] = ' '
        board[8][2] = board[8][4] = board[8][6] = board[8][8] = ' '

        """
        To show the locations
        """

        for j in range(9):
            board[0][j] = str(j)

        for i in range(9):
            board[i][0] = str(i)

        # Initial State
        self.initial = Game_State(to_move='O', result=0, board=board, moves=self.seek_moves(board, 'O'))

    def seek_piece(self, state):
        """
        Seek all movable pieces
        """
        # from the given template
        piece = state.to_move
        board = state.board

        horse = []
        for i in range(1, 9):
            for j in range(1, 9):
                if board[i][j] == 'O' or board[i][j] == 'R':
                    if len(self.actions((i, j), state)) is not 0:
                        horse.append((i, j))

        return horse

    def seek_moves(self, board, piece):
        """
        Seek all available moves
        """
        # check if pieces are inside the ranges
        moves = []
        if piece == 'O':
            for i in range(1, 9):
                for j in range(1, 9):
                    if board[i][j] == 'O':
                        if (i + 1 <= 8) and (j + 1 <= 8):
                            moves.append([(i, j), (i + 1, j + 1)])
                        if (i + 1 <= 8) and (j - 1 >= 1):
                            moves.append([(i, j), (i + 1, j - 1)])

                    if board[i][j] == 'R':  # Red King
                        if (i + 1 <= 8) and (j - 1 >= 1):
                            moves.append([(i, j), (i + 1, j - 1)])
                        if (i - 1 >= 1) and (j + 1 <= 8):
                            moves.append([(i, j), (i - 1, j + 1)])
                        if (i - 1 >= 1) and (j - 1 >= 1):
                            moves.append([(i, j), (i - 1, j - 1)])
                        if (i + 1 <= 8) and (j + 1 <= 8):
                            moves.append([(i, j), (i + 1, j + 1)])

            return moves

        elif piece == 'X':
            for i in range(1, 9):
                for j in range(1, 9):
                    if board[i][j] == 'X':
                        if (i - 1 >= 1) and (j - 1 >= 1):
                            moves.append([(i, j), (i - 1, j - 1)])
                        if (i - 1 >= 1) and (j + 1 <= 8):
                            moves.append([(i, j), (i - 1, j + 1)])

                    if board[i][j] == 'B':  # Black King
                        if (i - 1 >= 1) and (j + 1 <= 8):
                            moves.append([(i, j), (i - 1, j + 1)])
                        if (i + 1 <= 8) and (j - 1 >= 1):
                            moves.append([(i, j), (i + 1, j - 1)])
                        if (i - 1 >= 1) and (j - 1 >= 1):
                            moves.append([(i, j), (i - 1, j - 1)])
                        if (i + 1 <= 8) and (j + 1 <= 8):
                            moves.append([(i, j), (i + 1, j + 1)])

            return moves

    def actions(self, loc, state):
        """
        actions for moving (legal moves)
        """
        legal_moves = []
        piece = state.to_move
        moves = state.moves
        board = state.board

        # check from every moves
        for move in moves:
            (ini_row, ini_col) = move[0]
            (after_row, after_col) = move[1]
            jump_row = (after_row + (after_row - ini_row))
            jump_col = (after_col + (after_col - ini_col))

            # Selected piece
            if (ini_row, ini_col) == loc:
                # Slides
                if board[(after_row, after_col)] == ' ':
                    legal_moves.append([(ini_row, ini_col), (after_row, after_col)])
                # Jumps and Further moves after Jumps
                # if the next move has a piece there
                elif (board[(after_row, after_col)] != piece) and (board[(after_row, after_col)] != 'B') and (
                        board[(after_row, after_col)] != 'R') and (board[(after_row, after_col)] != ' '):
                    # Check if it is available to jump
                    # if it jumps check if it over the board
                    if 1 <= jump_row < 9 and 1 <= jump_col < 9:
                        if board[(jump_row, jump_col)] == ' ':
                            legal_moves.append([(ini_row, ini_col),
                                                (after_row + (after_row - ini_row), after_col + (after_col - ini_col))])

        return legal_moves

    def final_move(self, state, move):
        """
        Final moves
        """
        # from the given template
        piece = state.to_move
        board = state.board

        ini = board[move[0]]
        board[move[0]] = ' '
        board[move[1]] = ini

        (ini_row, ini_col) = move[0]
        (after_row, after_col) = move[1]

        # Sliding
        diff_row = after_row - ini_row
        diff_col = after_col - ini_col

        # Jumping
        if abs(diff_row) > 1:
            board[(int(ini_row + diff_row / 2), int(ini_col + diff_col / 2))] = ' '

        # KING
        if (piece == 'O' and after_row == 8) and (board[move[1]] == 'O'):
            board[move[1]] = 'R'  # Red King

        if (piece == 'X' and after_row == 1) and (board[move[1]] == 'X'):
            board[move[1]] = 'B'  # Black King

        return Game_State(to_move=('X' if state.to_move == 'O' else 'O'),
                          result=self.get_result(board, move[1], piece),
                          board=board, moves=self.seek_moves(board, ('X' if state.to_move == 'O' else 'O')))

    def boolean_test(self, state):
        return state.result != 0 or len(state.moves) == 0

    def get_result(self, board, move, piece):

        if np.where(board == 'X') == None and np.where(board == 'B') == None:
            return +1 if piece == 'O' else -1
        else:
            return 0

    def show(self, state):
        print(state.board)

    def stages(self):

        state = self.initial
        stage = 1
        while True:
            print("------------------")
            print("CHECKERS: Stage {}".format(stage))
            print("------------------")

            # Human's turn
            turn = human(self, state)
            if turn is not None:
                state = self.final_move(state, turn)
            if self.boolean_test(state):
                self.show(state)

            board_re = state.board.copy()

            # Computer's turn
            turn = computer(self, state)
            state = state._replace(board=board_re)
            if turn is not None:
                state = self.final_move(state, turn)
            if self.boolean_test(state):
                self.show(state)

            stage += 1

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)


def main():
    checkers = Checkers()
    initial_state = checkers.initial
    checkers.stages()


if __name__ == "__main__":
    main()
