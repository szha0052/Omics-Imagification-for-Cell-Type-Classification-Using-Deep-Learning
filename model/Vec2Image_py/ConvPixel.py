
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def ConvPixel(FVec, xp, yp, A, B, Base, FIG=0):
    M = np.ones((A, B)) * Base
    num_genes = len(FVec)

    for j in range(len(xp)):
        if j >= num_genes:
            continue
        xj, yj = xp[j] - 1, yp[j] - 1
        if 0 <= xj < A and 0 <= yj < B:
            M[xj, yj] = FVec[j]

    coords = np.vstack((xp, yp)).T
    _, idx, inv = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    for i in range(len(idx)):
        duplicates = np.where(inv == i)[0]
        valid_dupes = [d for d in duplicates if d < len(FVec)]
        if len(valid_dupes) > 1:
            avg_val = np.mean([FVec[d] for d in valid_dupes])
            xj, yj = xp[valid_dupes[0]] - 1, yp[valid_dupes[0]] - 1
            if 0 <= xj < A and 0 <= yj < B:
                M[xj, yj] = avg_val

    if FIG == 1:
        plt.imshow(M.T, aspect='auto')
        plt.show()

    return M
