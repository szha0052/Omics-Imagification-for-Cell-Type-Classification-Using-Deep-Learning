
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def minboundrect(x, y):
    # Dummy implementation: returns a square around the data
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    return np.array([xmin, xmax, xmax, xmin, xmin]), np.array([ymin, ymin, ymax, ymax, ymin])

def Cart2Pixel(Q, A=None, B=None):
    if 'data' not in Q:
        raise ValueError('no data provided')
    if 'Method' not in Q:
        Q['Method'] = 'tSNE'
    if 'Max_Px_Size' not in Q:
        Q['Max_Px_Size'] = 30
    if 'Dist' not in Q:
        Q['Dist'] = 'cosine'

    data = Q['data'].T  # transpose to samples x features
    if Q['Method'].lower() == 'kpca':
        raise NotImplementedError('KPCA is not implemented')
    elif Q['Method'].lower() == 'umap':
        raise NotImplementedError('UMAP is not implemented')
    else:
        if data.shape[0] < 5000:
            print('tSNE with exact algorithm is used')
            Y = TSNE(n_components=2, algorithm='exact', metric=Q['Dist']).fit_transform(data)
        else:
            print('tSNE with barnes-hut algorithm is used')
            Y = TSNE(n_components=2, algorithm='barnes_hut', metric=Q['Dist']).fit_transform(data)

    x, y = Y[:, 0], Y[:, 1]
    n, no_samples = Q['data'].shape
    xrect, yrect = minboundrect(x, y)

    grad = (yrect[1] - yrect[0]) / (xrect[1] - xrect[0])
    theta = np.arctan(grad)
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    z = R @ np.vstack((x, y))
    zrect = R @ np.vstack((xrect, yrect))

    min_dist = np.inf
    min_p1, min_p2 = 0, 0
    for p1 in range(n):
        for p2 in range(p1 + 1, n):
            d = np.sum((z[:, p1] - z[:, p2]) ** 2)
            if 0 < d < min_dist:
                min_dist = d
                min_p1, min_p2 = p1, p2

    dmin = np.linalg.norm(z[:, min_p1] - z[:, min_p2])
    rec_x_axis = np.abs(zrect[0, 0] - zrect[0, 1])
    rec_y_axis = np.abs(zrect[1, 1] - zrect[1, 2])

    if A is None or B is None:
        precision_old = np.sqrt(2)
        A = int(np.ceil(rec_x_axis * precision_old / dmin))
        B = int(np.ceil(rec_y_axis * precision_old / dmin))
        if max(A, B) > Q['Max_Px_Size']:
            precision = precision_old * Q['Max_Px_Size'] / max(A, B)
            A = int(np.ceil(rec_x_axis * precision / dmin))
            B = int(np.ceil(rec_y_axis * precision / dmin))

    xp = np.round(1 + A * (z[0] - np.min(z[0])) / (np.max(z[0]) - np.min(z[0]))).astype(int)
    yp = np.round(1 - B * (z[1] - np.max(z[1])) / (np.max(z[1]) - np.min(z[1]))).astype(int)
    A = np.max(xp)
    B = np.max(yp)

    Base = 1
    M = [ConvPixel(Q['data'][:, j], xp, yp, A, B, Base, 0) for j in range(no_samples)]
    return M, xp, yp, A, B, Base
