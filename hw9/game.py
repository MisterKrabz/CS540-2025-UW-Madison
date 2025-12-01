import copy
import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    pieces = ['b', 'r']
    max_depth = 3

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.board = [[' ' for j in range(5)] for i in range(5)]
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ 
        TODO: Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        # copy state ONCE to prevent modification of the actual game board
        search_state = copy.deepcopy(state)
        
        # get all legal next moves
        moves = self.succ(search_state, self.my_piece)
        
        # if no moves are found
        if not moves:
            return []
        
        piece_count = sum(row.count('b') + row.count('r') for row in state)
        if piece_count < 8:
            # Filter out any moves that are NOT drop moves (length 1)
            valid_moves = [m for m in moves if len(m) == 1]
            if valid_moves:
                moves = valid_moves

        best_move = moves[0]
        best_val = float('-inf')
        
        # Initialize Alpha-Beta parameters
        alpha = float('-inf')
        beta = float('inf')
        
        # iterate moves to find best via minimax
        for move in moves:
            self.sim_place(search_state, move, self.my_piece)
            
            # call shadow helper to use pruning
            val = self.min_val_ab(search_state, 0, alpha, beta)
            
            # Reverse the move to restore state (Unmake)
            self.undo_move(search_state, move)
            
            if val > best_val:
                best_val = val
                best_move = move
                
            # Update Alpha
            alpha = max(alpha, best_val)

        return best_move
    
    def undo_move(self, state, move):
        """ Reverses the action of place_piece on the state list. """
        if len(move) > 1:
            # The piece is currently at move[0], move it back to move[1]
            piece = state[move[0][0]][move[0][1]]
            state[move[1][0]][move[1][1]] = piece
            # Clear the spot it moved to
            state[move[0][0]][move[0][1]] = ' ' 
        else:
            # It was a drop, so just remove the piece at move[0]
            state[move[0][0]][move[0][1]] = ' '

    def succ(self, state, my_piece): 
        """
        TODO: Generate a list of valid successors for the current game state 
        on placing your piece. (defined by self.my_piece)
        """
        moves = []
        piece_count = sum(row.count('b') + row.count('r') for row in state)
        drop_phase = piece_count < 8

        if drop_phase:
            # add all empty spots
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        moves.append([(r, c)])
        else:
            # check adjacent spots for existing pieces
            for r in range(5):
                for c in range(5):
                    if state[r][c] == my_piece:
                        # iterate 8 directions
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0: continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == ' ':
                                    moves.append([(nr, nc), (r, c)])
        return moves
    
    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    
    def heuristic_game_value(self, state):
        """ 
        TODO: Define the heuristic game value of the current board state taking into account players
        and opponents

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            float heuristic_val (heuristic computed for the game state)
        """
        # if terminal, return exact value
        t_val = self.game_value(state)
        if t_val != 0: return t_val

        score = 0.0
        # check all linear segments and boxes
        # horizontal
        for r in range(5):
            for c in range(2):
                score += self.eval_window([state[r][c+i] for i in range(4)])
        # vertical
        for c in range(5):
            for r in range(2):
                score += self.eval_window([state[r+i][c] for i in range(4)])
        # diag
        for r in range(2):
            for c in range(2):
                score += self.eval_window([state[r+i][c+i] for i in range(4)])
        # anti-diag
        for r in range(2):
            for c in range(3, 5):
                score += self.eval_window([state[r+i][c-i] for i in range(4)])
        # 2x2 box
        for r in range(4):
            for c in range(4):
                score += self.eval_window([state[r][c], state[r][c+1], state[r+1][c], state[r+1][c+1]])

        # normalize between -0.9 and 0.9
        return max(min(score / 50.0, 0.9), -0.9)
    
    def eval_window(self, w):
        """ Helper to score a window of 4 cells """
        my_c = w.count(self.my_piece)
        opp_c = w.count(self.opp)
        
        # Only my pieces
        if my_c > 0 and opp_c == 0:
            if my_c == 3: return 0.8 
            if my_c == 2: return 0.2 
            return 0.05 
            
        # Only opponents pieces
        if opp_c > 0 and my_c == 0:
            if opp_c == 3: return -0.9
            if opp_c == 2: return -0.2
            return -0.05
        
        return 0

    def game_value(self, state):
        """ 
        TODO: Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # horizontal
        for r in range(5):
            for c in range(2):
                if state[r][c] != ' ' and all(state[r][c] == state[r][c+i] for i in range(4)):
                    return 1 if state[r][c] == self.my_piece else -1
        # vertical
        for c in range(5):
            for r in range(2):
                if state[r][c] != ' ' and all(state[r][c] == state[r+i][c] for i in range(4)):
                    return 1 if state[r][c] == self.my_piece else -1
        # diag
        for r in range(2):
            for c in range(2):
                if state[r][c] != ' ' and all(state[r][c] == state[r+i][c+i] for i in range(4)):
                    return 1 if state[r][c] == self.my_piece else -1
        # anti-diag
        for r in range(2):
            for c in range(3, 5):
                if state[r][c] != ' ' and all(state[r][c] == state[r+i][c-i] for i in range(4)):
                    return 1 if state[r][c] == self.my_piece else -1
        # 2x2 box
        for r in range(4):
            for c in range(4):
                if state[r][c] != ' ' and state[r][c] == state[r][c+1] == state[r+1][c] == state[r+1][c+1]:
                    return 1 if state[r][c] == self.my_piece else -1

        return 0 # no winner yet
    
    def max_value(self, state, depth):
        """
        TODO: Complete the helper function to implement min-max as described in the writeup
        """
        return self.max_val_ab(state, depth, float('-inf'), float('inf'))
    
    def max_val_ab(self, state, depth, alpha, beta):
        if self.game_value(state) != 0: return self.game_value(state)
        if depth >= self.max_depth: return self.heuristic_game_value(state)

        val = float('-inf')
        for move in self.succ(state, self.my_piece):
            self.sim_place(state, move, self.my_piece) 
            val = max(val, self.min_val_ab(state, depth + 1, alpha, beta))
            self.undo_move(state, move)
            
            if val >= beta: return val
            alpha = max(alpha, val)
        return val

    def min_value(self, state, depth):
        return self.min_val_ab(state, depth, float('-inf'), float('inf'))
    
    def min_val_ab(self, state, depth, alpha, beta):
        if self.game_value(state) != 0: return self.game_value(state)
        if depth >= self.max_depth: return self.heuristic_game_value(state)

        val = float('inf')
        for move in self.succ(state, self.opp):
            self.sim_place(state, move, self.opp) 
            val = min(val, self.max_val_ab(state, depth + 1, alpha, beta))
            self.undo_move(state, move)
            
            if val <= alpha: return val
            beta = min(beta, val)
        return val

    def sim_move(self, state, move, piece):
        """ Helper to apply a move on a temp state """
        if len(move) > 1: state[move[1][0]][move[1][1]] = ' '
        state[move[0][0]][move[0][1]] = piece

    def sim_place(self, state, move, piece):
        """ Helper to apply a move on a temp state (simulation only). """
        if len(move) > 1:
            state[move[1][0]][move[1][1]] = ' '
        state[move[0][0]][move[0][1]] = piece

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while len(player_move) != 2 or player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while len(move_from) != 2 or move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while len(move_to) != 2 or move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()