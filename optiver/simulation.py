import numpy as np
from numpy import random as rand
from random import choice

start = np.array([0,0])
#food = [np.array([2,0]), np.array([-2,0]), np.array([0,2]), np.array([0,-2])]
#food = [(2,0), (-2,0), (0,2), (0,-2)]
moves = [np.array([1,0]), np.array([-1,0]), np.array([0,1]), np.array([0,-1])]
n_simulations = 80000

def random_step(position, moves):
    return position + choice(moves)


def food_walk(start, moves):
    position = start
    steps = 0
    while np.linalg.norm(position, np.inf) < 2:
        steps += 1
        position = random_step(position, moves)
    return steps

def simulation(start=start, moves=moves, n_simulations= n_simulations):
    times = []
    for s in range(n_simulations):
        time = food_walk(start, moves)
        times.append(time)
    return sum(times)/n_simulations

print(simulation())
# def food_walk(start, moves):
#     position = start
#     steps = 0
#     while tuple(position) not in food:
#         if np.linalg.norm(position, 2) > 100000:
#             return 0 
#         steps += 1
#         position = random_step(position, moves)
#     return steps

# def simulation(start=start, moves=moves, n_simulations= n_simulations):
#     times = []
#     actual_simulations = n_simulations
#     for s in range(n_simulations):
#         time = food_walk(start, moves)
#         if time:
#             times.append(time)
#         else:
#             actual_simulations -= 1
#     return sum(times)/actual_simulations

# print(simulation())
#BETTER IDEA, GENERATE ALL THE POSSIBLE NEW STATES AT EVERY TIME AND STOP WHEN ONE IS IN FOOD, THAT WAY WE PARALLELIZE THE PROCESS
