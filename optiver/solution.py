# simulation with arbitrary boundary condition as an argument
# generate all new possible states every time

import numpy as np
from collections import defaultdict
start = (0,0)
initial_state = {start: 1}
moves = dict(zip([(1,0), (-1,0), (0,1), (0,-1)], [0.25]*4))
n_paths = 200

square_condition = lambda x,y: np.linalg.norm((x,y), np.inf) == 2
ellipse_condition = lambda x, y: ((x - 2.5) / 30 )**2 + ( (y - 2.5) / 40 )**2 >= 1

def simulate_one_step(current_states):
    newStates = defaultdict(float)
    for cur_pos, prob_of_being_here in current_states.items():
        for movement_dir,prob_of_moving_this_way in moves.items():
            newStates[tuple(np.array(cur_pos)+np.array(movement_dir))] += prob_of_being_here*prob_of_moving_this_way
    return newStates

def simulate_paths(initial_state, n_paths, boundary_condition):
    states = initial_state
    average_num_moves = 0
    for step in range(1,n_paths):
        states = simulate_one_step(states)
        boundary_chances = 0
        for pos, prob in set(states.items()):    
            if boundary_condition(*pos):
                boundary_chances += prob
                del states[pos]
        average_num_moves += step * boundary_chances
    return round(average_num_moves)
#print(simulate_one_step(initial_state))
print(simulate_paths(initial_state,n_paths, square_condition))
print(simulate_paths(initial_state,80*n_paths, ellipse_condition)) #at 20*n_paths gives 1107 and at 40*n_paths gives 1163, seems to stabilize. Confirme with 80*n_paths which gives 1164

