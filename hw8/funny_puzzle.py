import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0;

    # Map tile values to their coordinates in the goal state for O(1) lookup
    goal_positions = {};
    for i, tile in enumerate(to_state):
        if(tile != 0):
            goal_positions[tile] = (i //3, i%3)

    # Calculate distance for each tile in from_state
    for i, tile in enumerate(from_state):
        if tile != 0:
            current_row, current_col = i // 3, i % 3
            goal_row, goal_col = goal_positions[tile]
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)

    return distance
    
def get_count_heuristic(from_state, to_state=(1,2,3,4,5,6,7,0,0)):
    """
    TODO: Implement this function. This function will not directly be tested by the grader.
    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that returns the count of the number of incorrectly placed elements in the from_state variable
    """

    num_incorrect_count = 0
    
    for i in range(len(from_state)):
        tile = from_state[i]
        # Check if the tile is not empty and not in the correct position matches the goal
        if from_state[i] != to_state[i]:
            num_incorrect_count += 1

    return num_incorrect_count

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))

def print_count_succ(state):
    """
    TODO: This is based on get_count_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle along with count based heuristic 
    """
    
    succ_states = get_succ(state)
    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_count_heuristic(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """

    succ_states = []
    
    # Find the indices of the empty spots (0s)
    zero_indices = [i for i, x in enumerate(state) if x == 0]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for zero_idx in zero_indices:
        r, c = zero_idx // 3, zero_idx % 3
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if 0 <= nr < 3 and 0 <= nc < 3:
                neighbor_idx = nr * 3 + nc
                
                # Ensure we are not swapping an empty spot with another empty spot
                if state[neighbor_idx] != 0:
                    new_state = list(state)
                    # Swap the empty spot with the neighbor
                    new_state[zero_idx], new_state[neighbor_idx] = new_state[neighbor_idx], new_state[zero_idx]
                    succ_states.append(new_state)
   
    return sorted(succ_states)
    

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    # Stores in this format: (cost, current_state, (g, h, parent_state_tuple))
    pq = []
    
    # Stores visited states and their path info for reconstruction
    # Key: tuple(state), Value: (g, h, parent_state_tuple)
    closed_set = {}
    
    start_g = 0
    start_h = get_manhattan_distance(state, goal_state)
    start_cost = start_g + start_h
    
    # Push initial state. Parent is -1 to denote start.
    heapq.heappush(pq, (start_cost, state, (start_g, start_h, -1)))
    
    max_length = 0
    final_state_tuple = None
    
    while pq:
        # Update max queue length
        max_length = max(max_length, len(pq))
        
        # Pop the state with the lowest cost
        current_entry = heapq.heappop(pq)
        current_cost = current_entry[0]
        current_state = current_entry[1]
        info = current_entry[2]
        g = info[0]
        h = info[1]
        parent = info[2]
        
        current_state_tuple = tuple(current_state)
        
        # If strictly visited with lower cost, skip 
        if current_state_tuple in closed_set:
            continue
            
        # Add to closed set
        closed_set[current_state_tuple] = (g, h, parent)
        
        # Goal check
        if current_state == goal_state:
            final_state_tuple = current_state_tuple
            break
            
        # Generate successors
        succs = get_succ(current_state)
        for succ in succs:
            succ_tuple = tuple(succ)
            if succ_tuple in closed_set:
                continue
            
            new_g = g + 1
            new_h = get_manhattan_distance(succ, goal_state)
            new_cost = new_g + new_h
            
            # Push to PQ. 
            heapq.heappush(pq, (new_cost, succ, (new_g, new_h, current_state_tuple)))

    # Reconstruct path
    state_info_list = []
    if final_state_tuple is not None:
        curr = final_state_tuple
        while curr != -1:
            g_val, h_val, parent_val = closed_set[curr]
            state_info_list.append((list(curr), h_val, g_val))
            curr = parent_val
        state_info_list.reverse()



    # This is a format helper，which is only designed for format purpose.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.
    for state_info in state_info_list:
        current_state = state_info[0]
        h = state_info[1]
        move = state_info[2]
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()
    print_count_succ([2,5,1,4,0,6,7,0,3])
    print()
    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()
    solve([2,5,1,4,0,6,7,0,3])
    print()


