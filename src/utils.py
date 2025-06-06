import numpy as np

def normalizeMatrix(matrix, etendue):
    matrix_norm = (matrix-np.min(matrix))/np.ptp(matrix)
    matrix_norm_noise = matrix_norm*(1-etendue)+etendue/2
    return matrix_norm_noise