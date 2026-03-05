import numpy as np
import sys

def read_matrix(filename):
    with open(filename, 'r') as f:
        N = int(f.readline())
        data = []
        for _ in range(N):
            row = list(map(float, f.readline().split()))
            data.append(row)
    return np.array(data)

A = read_matrix(sys.argv[1])
B = read_matrix(sys.argv[2])
C_cpp = read_matrix(sys.argv[3])

C_py = np.dot(A, B)

if np.allclose(C_cpp, C_py):
    print("Все верно!")
else:
    print("Что-то пошло не так.")
