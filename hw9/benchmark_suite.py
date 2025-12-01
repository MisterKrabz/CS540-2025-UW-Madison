import time
import random
import copy
import types
import game

# ---------------------------------------------------------
# 1. DEFINE THE OPPONENT (RANDOM PLAYER)
# ---------------------------------------------------------
class RandomPlayer(game.TeekoPlayer):
    def make_move(self, state):
        moves = self.succ(state, self.my_piece)
        return random.choice(moves)

# ---------------------------------------------------------
# 2. DEFINE THE STRATEGIES (HEURISTICS)
# ---------------------------------------------------------

# HEURISTIC A: Your original "Linear" Logic
# Treats 3-in-a-row (0.75) only slightly better than 2-in-a-row (0.5)
def eval_linear(self, w):
    my_c = w.count(self.my_piece)
    opp_c = w.count(self.opp)
    if my_c > 0 and opp_c == 0: return my_c / 4.0
    if opp_c > 0 and my_c == 0: return -(opp_c / 4.0)
    return 0

# HEURISTIC B: "Aggressive" Logic (Non-Linear)
# Treats 3-in-a-row as CRITICAL (0.8/0.9). Forces blocks and wins.
def eval_aggressive(self, w):
    my_c = w.count(self.my_piece)
    opp_c = w.count(self.opp)
    
    # Case 1: My pieces (Offense)
    if my_c > 0 and opp_c == 0:
        if my_c == 3: return 0.8  # One move from winning!
        if my_c == 2: return 0.2  # Good setup
        return 0.05               # Weak
        
    # Case 2: Opponent pieces (Defense)
    if opp_c > 0 and my_c == 0:
        if opp_c == 3: return -0.9 # CRITICAL THREAT: Block immediately!
        if opp_c == 2: return -0.2 # Be careful
        return -0.05               # Minor threat
        
    return 0

# ---------------------------------------------------------
# 3. THE TEST ENGINE
# ---------------------------------------------------------
def run_match(ai_depth, heuristic_func, label):
    ai = game.TeekoPlayer()
    
    # --- DYNAMIC CONFIGURATION ---
    ai.max_depth = ai_depth
    # This "monkey patches" the method onto the instance dynamically
    ai.eval_window = types.MethodType(heuristic_func, ai) 
    # -----------------------------

    opponent = RandomPlayer()
    
    # Randomize colors
    if random.choice([True, False]):
        ai.my_piece = 'b'; opponent.my_piece = 'r'
    else:
        ai.my_piece = 'r'; opponent.my_piece = 'b'
    opponent.opp = ai.my_piece

    # Gameplay Loop
    players = [ai, opponent] if ai.my_piece == 'b' else [opponent, ai]
    turn = 0
    piece_count = 0
    game_over = False
    ai_times = []

    # Drop Phase
    while not game_over and piece_count < 8:
        curr = players[turn]
        if curr == ai:
            t0 = time.time()
            move = curr.make_move(curr.board)
            ai_times.append(time.time() - t0)
        else:
            move = curr.make_move(curr.board)
        
        curr.place_piece(move, curr.my_piece)
        # Sync boards
        other = players[(turn+1)%2]
        other.board = [r[:] for r in curr.board]
        
        if ai.game_value(ai.board) != 0: game_over = True
        piece_count += 1
        turn = (turn + 1) % 2

    # Move Phase (Max 150 turns to prevent infinite loops)
    limit = 0
    while not game_over and limit < 150:
        curr = players[turn]
        if curr == ai:
            t0 = time.time()
            move = curr.make_move(curr.board)
            ai_times.append(time.time() - t0)
        else:
            move = curr.make_move(curr.board)

        curr.place_piece(move, curr.my_piece)
        # Sync boards
        other = players[(turn+1)%2]
        other.board = [r[:] for r in curr.board]

        if ai.game_value(ai.board) != 0: game_over = True
        turn = (turn + 1) % 2
        limit += 1

    winner = ai.game_value(ai.board)
    max_t = max(ai_times) if ai_times else 0
    return winner, max_t

def run_suite():
    # CONFIGURATIONS TO TEST
    configs = [
        {"name": "CONFIG 1: Depth 2 + Linear (Original)", "depth": 2, "func": eval_linear},
        {"name": "CONFIG 2: Depth 3 + Linear (Original)", "depth": 3, "func": eval_linear},
        {"name": "CONFIG 3: Depth 2 + Aggressive (New)", "depth": 2, "func": eval_aggressive},
    ]

    GAMES_PER_CONFIG = 5 # 5 games is enough to get a signal

    print(f"benchmark_suite.py initializing...")
    print(f"Goal: Find a config with Max Time < 5.0s AND Win Rate >= 66%")
    print("="*60)

    for conf in configs:
        print(f"Testing {conf['name']}...")
        wins = 0
        worst_time = 0
        
        for i in range(GAMES_PER_CONFIG):
            w, t = run_match(conf['depth'], conf['func'], conf['name'])
            if w == 1: wins += 1
            if t > worst_time: worst_time = t
            print(f"  Game {i+1}: {'WIN ' if w==1 else 'LOSS'} | Max Time: {t:.4f}s")
        
        win_rate = (wins / GAMES_PER_CONFIG) * 100
        pass_time = "PASS" if worst_time < 4.8 else "FAIL" # 4.8 safety buffer
        pass_win = "PASS" if win_rate >= 60.0 else "FAIL" # 60% approx 2/3
        
        print(f"-> RESULT: Win Rate {win_rate}% ({pass_win}) | Worst Time {worst_time:.4f}s ({pass_time})")
        print("-" * 60)

if __name__ == "__main__":
    run_suite()