# Circle packing in unit square using ADMM
# minimize \sum_{i,j} f_{ij}(z_i, z_j) + \sum_i g_i(z_i)
# f_{ij}(z_i, z_j) = 0, if ||z_i - z_j|| >= 2R
#                  = infinity, if ||z_i - z_j|| < 2R
# g_i(z_i) = 0, if R <= z_i <= 1 - R
#          = infinity, otherwise

import numpy as np
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp

def get_ind(n):
    holder = itertools.combinations(range(n), 2)
    ind_map = {}
    i = 0
    for p in holder:
        ind_map[i] = p
        i += 1
    return ind_map

# x_overlap = proximal operator of f
# x_box = proximal operator of g = projection of n_box on to a box
def prox_op(x_overlap, x_box, n_overlap, n_box, ind_map, R, n, N):
    # Compute x_overlap:
    for a, value in ind_map.items():
        i = value[0]
        j = value[1]
        if np.linalg.norm(n_overlap[a][i] - n_overlap[a][j]) >= 2 * R:
            x_overlap[a][i] = n_overlap[a][i]
            x_overlap[a][j] = n_overlap[a][j]
        else:
            # Pull them apart equally:
            diff = n_overlap[a][i] - n_overlap[a][j]
            d = np.linalg.norm(diff)
            diff_unit = diff / d
            x_overlap[a][i] = n_overlap[a][i] + (R - d / 2) * diff_unit
            x_overlap[a][j] = n_overlap[a][j] - (R - d / 2) * diff_unit
    # Compute x_box:
    for i in range(n):
        for k in range(2):
            if n_box[i][k] < R:
                x_box[i][k] = R
            elif n_box[i][k] > 1 - R:
                x_box[i][k] = 1 - R
            else:
                x_box[i][k] = n_box[i][k]
    return x_overlap, x_box, n_overlap, n_box

def admm_rest_steps(x_overlap, x_box, m_overlap, m_box, z, u_overlap, u_box, n_overlap, n_box, n, ind_map, N, alpha):
    # Update m:
    m_overlap = x_overlap + u_overlap
    m_box = x_box + u_box
    # Update z:
    m_accum = np.zeros((n, 2))
    # Get m_overlap:
    for key, value in ind_map.items():
        i = value[0]
        j = value[1]
        m_accum[i] += m_overlap[key][i]
        m_accum[j] += m_overlap[key][j]
    # Get m_box:
    m_accum += m_box
    # Average:
    z = m_accum / n
    # Update u and n:
    u_overlap += (alpha * (x_overlap - z))
    n_overlap = z - u_overlap
    u_box += (alpha * (x_box - z))
    n_box = z - u_box
    return x_overlap, x_box, m_overlap, m_box, z, u_overlap, u_box, n_overlap, n_box
    
def circle_packing_admm(R, n, max_iter, alpha):
    ind_map = get_ind(n)
    N = len(ind_map)
    x_overlap = np.zeros((N, n, 2))
    x_box = np.zeros((n, 2))
    m_overlap = np.zeros((N, n, 2))
    m_box = np.zeros((n, 2))
    z = np.zeros((n, 2))
    u_overlap = np.zeros((N, n, 2))
    u_box = np.zeros((n, 2))
    n_overlap = np.random.rand(N, n, 2)
    n_box = np.zeros((n, 2))
    for i in range(max_iter):
        x_overlap, x_box, n_overlap, n_box = prox_op(x_overlap, x_box, n_overlap, n_box, ind_map, R, n, N)
        x_overlap, x_box, m_overlap, m_box, z, u_overlap, u_box, n_overlap, n_box = admm_rest_steps(x_overlap, x_box, m_overlap, m_box, z, u_overlap, u_box, n_overlap, n_box, n, ind_map, N, alpha)
    return z

if __name__ == '__main__':
    R = 0.125 # Radius
    n = 16 # Number of circles
    max_iter = 2000
    alpha = 0.005
    z = circle_packing_admm(R, n, max_iter, alpha)
    print(z)
    # Draw circles:
    for p in z:
        circle= plt.Circle((p[0], p[1]), radius=R)
        ax = plt.gca()
        ax.add_patch(circle)
    # Draw unit box:
    box = plt.Rectangle((0, 0), 1, 1, fill=False)
    plt.gca().add_patch(box)
    plt.axis('scaled') # Make sure picutre is not deformed       
    plt.show()

