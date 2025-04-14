
import numpy as np
import matplotlib.pyplot as plt

def ConvPixel(FVec, xp, yp, A, B, Base, FIG=0):
    M = np.ones((A, B)) * Base
    for j in range(len(FVec)):
        M[xp[j] - 1, yp[j] - 1] = FVec[j]

    coords = np.vstack((xp, yp)).T
    _, idx, inv = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    for i in range(len(idx)):
        duplicates = np.where(inv == i)[0]
        if len(duplicates) > 1:
            avg_val = np.mean([FVec[d] for d in duplicates])
            M[xp[duplicates[0]] - 1, yp[duplicates[0]] - 1] = avg_val

    if FIG == 1:
        plt.imshow(M.T, aspect='auto')
        plt.show()

    return M
