#!/usr/bin/env python
import numpy as np

def generate_bayer_matrix(size):
    if size < 2:
        raise ValueError("Bayer matrix size must be at least 2x2.")
    
    bayer_matrix = np.zeros((size, size), dtype=np.uint8)
    n = 1
    while n < size:
        for i in range(n):
            for j in range(n):
                bayer_matrix[i][j] *= 4
                bayer_matrix[i + n][j] = bayer_matrix[i][j] + 2
                bayer_matrix[i][j + n] = bayer_matrix[i][j] + 3
                bayer_matrix[i + n][j + n] = bayer_matrix[i][j] + 1
        n *= 2
    
    return bayer_matrix

# Example: Generate a 4x4 Bayer matrix
bayer_matrix = generate_bayer_matrix(4)
print(bayer_matrix)
