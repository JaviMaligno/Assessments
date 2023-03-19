from collections import defaultdict
from itertools import combinations

def solution(A):
    # Implement your solution here
    best_sum = -1
    sums = dict_of_sums(A)  
    for l in sums.values():
        if len(l) < 2:
            continue
        else:
            current_sum = compute_best_sum(l)
            best_sum = current_sum if current_sum > best_sum else best_sum
    return best_sum
    
def dict_of_sums(A):
    sums = defaultdict(list)    
    for a in A:
        sum_digits = sum_of_digits(a)
        sums[sum_digits].append(a)
    return sums

def sum_of_digits(a):
    return sum(map(int,list(str(a))))

def compute_best_sum(l):
    pairs = combinations(l,2)
    best_sum = max(map(sum, pairs))
    return best_sum
