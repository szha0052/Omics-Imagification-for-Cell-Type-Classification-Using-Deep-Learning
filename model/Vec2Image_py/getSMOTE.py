import numpy as np
from sklearn.neighbors import NearestNeighbors

class SmoteGenerator:
    """
    A class to implement the SMOTE algorithm for oversampling minority classes in a dataset.

    Attributes
    ----------
    data : np.ndarray
        Original data, shape (n_samples, n_features).
    labels : np.ndarray
        Original labels, length n_samples.
    k : int
        Number of nearest neighbors to consider.
    
    Methods
    -------
    fit_resample()
        Perform SMOTE oversampling and return the augmented data and labels.
    """

    # def __init__(self, data, labels, k=5):
    #     """
    #     Parameters
    #     ----------
    #     data : array-like of shape (n_samples, n_features)
    #         Input data.
    #     labels : array-like of shape (n_samples,)
    #         Input labels.
    #     k : int, optional
    #         Number of nearest neighbors to consider, default is 5.
    #     """
    #     self.data = np.asarray(data, dtype=float)
    #     self.labels = np.asarray(labels)
    #     self.k = k

    def __init__(self, data, labels, k=5, seed=None):
        self.data = np.asarray(data, dtype=float)
        self.labels = np.asarray(labels)
        self.k = k
        self.seed = seed

    def fit_resample(self):
        """
        Perform SMOTE oversampling and return the augmented data and labels.

        Returns
        -------
        X_res : np.ndarray
            Augmented data, shape (m, n_features), where m >= n_samples.
        y_res : np.ndarray
            Augmented labels, length m.
        """
        unique_classes, counts = np.unique(self.labels, return_counts=True)
        max_samples = np.max(counts)

        # Calculate the number of samples needed for each class
        sample_diffs = max_samples - counts
        N_list = []
        for diff, cnt in zip(sample_diffs, counts):
            if cnt == 0:
                # If a class has no samples, theoretically this case should be handled beforehand
                N_list.append(0)
            else:
                # Generate round(diff / cnt) new samples for each sample
                N_list.append(int(round(diff / cnt)))

        X_resampled = []
        y_resampled = []

        for class_idx, cls_label in enumerate(unique_classes):
            # All samples of the current class
            mask = (self.labels == cls_label)
            X_class = self.data[mask]
            T = X_class.shape[0]

            # Add original samples to the list
            X_tmp = [x for x in X_class]
            y_tmp = [cls_label] * T

            # Perform oversampling if needed
            if N_list[class_idx] > 0 and T > 0:
                nbrs = NearestNeighbors(n_neighbors=self.k).fit(X_class)

                for i in range(T):
                    x_i = X_class[i].reshape(1, -1)
                    # Find k nearest neighbors
                    _, nn_index = nbrs.kneighbors(x_i)
                    nn_index = nn_index[0]  # Shape (k,)

                    # Randomly select N_list[class_idx] neighbors from the k neighbors
                    # To ensure no replacement, set replace=False (if N > k, replace=True is required).
                    chosen_indices = np.random.choice(
                        nn_index, 
                        size=N_list[class_idx],
                        replace=(N_list[class_idx] > self.k)
                    )

                    for neighbor_idx in chosen_indices:
                        neighbor = X_class[neighbor_idx]
                        alpha = np.random.rand()  # [0,1)
                        synthetic_sample = x_i + alpha * (neighbor - x_i)
                        X_tmp.append(synthetic_sample.flatten())
                        y_tmp.append(cls_label)

            # Append to the overall list
            X_resampled.extend(X_tmp)
            y_resampled.extend(y_tmp)

        X_res = np.array(X_resampled)
        y_res = np.array(y_resampled)

        return X_res, y_res
