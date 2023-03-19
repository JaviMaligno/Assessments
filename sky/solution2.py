def solution(S):
    # Implement your solution here
    n_a = S.count("A") // 3
    n_n = S.count("N") // 2
    n_b = S.count("B") 
    return min(n_a, n_b, n_n)