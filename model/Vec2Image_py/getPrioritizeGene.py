import numpy as np
from sklearn.neighbors import NearestNeighbors
from .ConvPixel import ConvPixel
from .getNETtest import NetTester
import torch

def getPrioritizeGene(dset, Out, k):
    """
    Python translation of the MATLAB function getPrioritizeGene(dset,Out,k).

    :param dset: An object with
        - Xtest: np.ndarray of shape (num_genes, num_samples)
        - test_labels: list/array of length num_samples
    :param Out: An object with
        - Norm: integer (1 or 2)
        - xp, yp: coordinates arrays of shape (num_genes,)
        - model.net: trained classifier with .predict(...) or similar
        - A, B, Base: additional parameters for ConvPixel (if needed)
    :param k: number of neighbors to retrieve in the KNN search
    :return: GeneRank, a NumPy array of shape (num_genes,) containing accuracy for each gene
    """

    # ----------- STEP 1: Normalization --------------
    if Out['Norm'] == 1:
        # Equivalent to "NORM-1" in MATLAB code
        print("\nNORM-1\n")
        out_max = np.max(dset['Xtest'], axis=1, keepdims=True)
        out_min = np.min(dset['Xtest'], axis=1, keepdims=True)
        # Avoid division by zero
        denom = (out_max - out_min)
        denom[denom == 0] = 1.0
        dset['Xtest'] = (dset['Xtest'] - out_min) / denom
        # Replace any NaNs (from 0/0) with 0
        dset['Xtest'] = np.nan_to_num(dset['Xtest'])

    elif Out['Norm'] == 2:
        # Equivalent to "NORM-2" in MATLAB code
        print("\nNORM-2\n")
        # Shift by abs(min) + 1 to ensure positivity
        out_min = np.min(dset['Xtest'], axis=1, keepdims=True)
        shift_val = np.abs(out_min) + 1
        dset['Xtest'] = np.log(dset['Xtest'] + shift_val)
        # Scale by global max
        out_max = np.max(dset['Xtest'])
        if out_max != 0:
            dset['Xtest'] = dset['Xtest'] / out_max

    # ----------- STEP 2: Prepare for KNN search ------
    # We stack xp, yp as shape (num_genes, 2)
    coords = np.column_stack((Out['xp'], Out['yp']))
    nbrs = NearestNeighbors(n_neighbors=k, metric='minkowski', p=5)
    nbrs.fit(coords)

    num_genes, _ = dset['Xtest'].shape
    errors = np.zeros(num_genes)  # Will store accuracy for each gene

    # ----------- STEP 3: Loop over genes ------------
    for i in range(num_genes):
        # Copy Xtest to shuffledata
        shuffledata = np.copy(dset['Xtest'])

        # 3.1 Find the k nearest neighbors for gene i
        query_point = coords[i, :].reshape(1, -1)
        # distances, indices
        _, indices = nbrs.kneighbors(query_point)
        mIdx = indices[0]  # mIdx is a 1D array of size k

        # 3.2 Set those neighbors' rows in shuffledata to 1
        shuffledata[mIdx, :] = 1

        # 3.3 Build the 4D array M for each test sample
        #     M.shape = (somethingA, somethingB, 1, num_test_samples)
        #     This depends on the shape that ConvPixel returns. We'll assume
        #     ConvPixel returns a 2D array (height, width). We'll stack them.
        num_test_samples = len(dset['test_labels'])
        # Suppose ConvPixel returns e.g. (height, width). We'll define them from the first call:
        # We do a single call to guess shape. In practice, adapt as needed:
        sample_conv = ConvPixel(shuffledata[:, 0], Out['xp'], Out['yp'], Out['A'], Out['B'], Out['Base'], 0)
        height, width = sample_conv.shape
        M = np.zeros((height, width, 1, num_test_samples), dtype=np.float32)

        for j in range(num_test_samples):
            M[:, :, 0, j] = ConvPixel(shuffledata[:, j], Out['xp'], Out['yp'],
                                    Out['A'], Out['B'], Out['Base'], 0)

        # 3.4 Classify using Out.model.net
        #     This step depends heavily on your model library (TensorFlow, PyTorch, etc.)
        #     We'll assume a pseudo-code approach:
        # YPred_proba = Out.model.net.predict(M)  # shape could be (num_test_samples, num_classes)
        # YPredicted = np.argmax(YPred_proba, axis=1)

        # Convert to torch.Tensor
        XTest_tensor = torch.from_numpy(M).float().to('cpu')
        YTest_tensor = torch.from_numpy(dset['test_labels']).long().to('cpu')


        # 3) Model inference
        model = Out['model']['net'].to('cpu')
        model.eval()
        with torch.no_grad():
            logits = model(XTest_tensor)  # (n_samples, num_classes)
            Y_pred = torch.argmax(logits, dim=1)  # (n_samples,)



        # 3.5 Compare with ground truth
        #     If dset.test_labels are numeric and align with argmax indexing, we can do:
        YTrue = np.array(dset['test_labels'])
        valError = np.mean(Y_pred == YTrue)  # fraction correct
        errors[i] = valError

        print(f"The running genenumber is {i+1}, accuracy = {valError:.4f}")

    # ----------- STEP 4: Return the final ranking ----
    # In MATLAB code, "GeneRank" is just the array of accuracies
    GeneRank = errors
    return GeneRank

