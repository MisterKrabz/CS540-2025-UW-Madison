import time
import random
import game

class RandomPlayer(game.TeekoPlayer):
    def make_move(self, state):
        moves = self.succ(state, self.my_piece)
        return random.choice(moves)

def run_match(ai_depth):
    ai = game.TeekoPlayer()
    ai.max_depth = ai_depth
    opponent = RandomPlayer()
    
    # Randomize starting color
    if random.choice([True, False]):
        ai.my_piece = 'b'
        opponent.my_piece = 'r'
    else:
        ai.my_piece = 'r'
        opponent.my_piece = 'b'
    
    opponent.opp = ai.my_piece
    
    piece_count = 0
    turn = 0
    game_over = False
    
    ai_move_times = []
    
    # Mapping turn index to player
    players = [None, None]
    if ai.my_piece == 'b':
        players[0] = ai
        players[1] = opponent
    else:
        players[0] = opponent
        players[1] = ai

    while not game_over and piece_count < 8: # Drop Phase
        current_player = players[turn]
        
        start = time.time()
        move = current_player.make_move(current_player.board)
        end = time.time()
        
        if current_player == ai:
            ai_move_times.append(end - start)
            
        current_player.place_piece(move, current_player.my_piece)
        # Sync boards
        if current_player == ai:
            opponent.board = [row[:] for row in ai.board]
        else:
            ai.board = [row[:] for row in opponent.board]
            
        if ai.game_value(ai.board) != 0:
            game_over = True
            
        piece_count += 1
        turn = (turn + 1) % 2

    turn_limit = 0
    while not game_over and turn_limit < 200: # Move Phase
        current_player = players[turn]
        
        start = time.time()
        move = current_player.make_move(current_player.board)
        end = time.time()
        
        if current_player == ai:
            ai_move_times.append(end - start)

        current_player.place_piece(move, current_player.my_piece)
        
        # Sync boards
        if current_player == ai:
            opponent.board = [row[:] for row in ai.board]
        else:
            ai.board = [row[:] for row in opponent.board]

        if ai.game_value(ai.board) != 0:
            game_over = True
            
        turn = (turn + 1) % 2
        turn_limit += 1

    winner_val = ai.game_value(ai.board)
    max_time = max(ai_move_times) if ai_move_times else 0
    avg_time = sum(ai_move_times) / len(ai_move_times) if ai_move_times else 0
    
    return winner_val, max_time, avg_time

if __name__ == "__main__":
    DEPTH_TO_TEST = 4  # <--- CHANGE THIS PARAMETER
    TOTAL_GAMES = 5
    
    print(f"Running {TOTAL_GAMES} games at Depth {DEPTH_TO_TEST}...")
    print("-" * 40)
    
    wins = 0
    losses = 0
    ties = 0
    worst_time_overall = 0
    
    for i in range(TOTAL_GAMES):
        result, max_t, avg_t = run_match(DEPTH_TO_TEST)
        worst_time_overall = max(worst_time_overall, max_t)
        
        if result == 1:
            wins += 1
            res_str = "WIN"
        elif result == -1:
            losses += 1
            res_str = "LOSS"
        else:
            ties += 1
            res_str = "TIE"
            
        print(f"Game {i+1}: {res_str} | Max Time: {max_t:.4f}s | Avg Time: {avg_t:.4f}s")

    print("-" * 40)
    print(f"Final Stats for Depth {DEPTH_TO_TEST}:")
    print(f"Wins: {wins}/{TOTAL_GAMES} ({wins/TOTAL_GAMES*100:.1f}%)")
    print(f"Worst Move Time: {worst_time_overall:.4f}s (Must be < 5.0s)")