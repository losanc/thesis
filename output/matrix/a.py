import numpy as np
import meshio
import matplotlib.pyplot as plt
import math

matrix = None
mesh = None
DIM = 3
frame = 40
n_vert = None

mesh = meshio.read('mesh{}.obj'.format(frame))

points = mesh.points
n_vert = len(points)

with open('hessian{}.txt'.format(frame), 'rb') as file:
    byte = file.read()
    matrix = np.frombuffer(byte, dtype=np.float64)
    matrix = np.reshape(matrix, (n_vert * DIM, n_vert * DIM))

matrix_inv = np.linalg.inv(matrix)

dist_list = []
matrix_norm_list = []
j = 20
for i in range(20,n_vert):
    # for j in range(i + 1, n_vert):
        dist = np.linalg.norm(points[j] - points[i])
        block_matrix = matrix_inv[DIM * i:DIM * i + DIM, DIM * j:DIM * j + DIM]
        norm = np.linalg.norm(block_matrix)
        if norm<1e-14: 
            continue
        dist_list.append(dist)
        matrix_norm_list.append(math.log(norm))
        # dist_list.append(dist)
        # matrix_norm_list.append(block_matrix[0,0])

        # dist_list.append(dist)
        # matrix_norm_list.append(block_matrix[0,1])

        # dist_list.append(dist)
        # matrix_norm_list.append(block_matrix[0,2])

        # dist_list.append(dist)
        # matrix_norm_list.append(block_matrix[1,0])

        # dist_list.append(dist)
        # matrix_norm_list.append(block_matrix[1,1])

        # dist_list.append(dist)
        # matrix_norm_list.append(block_matrix[1,2])

plt.scatter(dist_list,matrix_norm_list,s=1)
plt.xlabel("vertex distance")
plt.ylabel("block matrix F-norm")
plt.show()

