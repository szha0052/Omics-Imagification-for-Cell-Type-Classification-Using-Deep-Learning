import numpy as np
import pandas as pd

class FeaturePixelMapper:
    """
    Given a fitted FftTransformer and original data,
    maps each feature index to its pixel location and corresponding FFT value.
    """
    
    def __init__(self, transformer, X):
        """
        Parameters:
            transformer (FftTransformer): A fitted instance of FftTransformer
            X (np.ndarray): Original training data (samples x features)
        """
        self.transformer = transformer
        self.X = X
        self.mapping_df = None

    def build_mapping(self):
        """
        Build the full mapping from feature index to (row, col) and real/imag FFT values.
        
        Returns:
            pd.DataFrame: A DataFrame with columns: Feature, Row, Col, RealPart, ImagPart
        """
        if not hasattr(self.transformer, '_coords'):
            raise ValueError("FftTransformer has not been fitted. Cannot extract coordinates.")
        
        coords = self.transformer._coords  # shape: (n_features, 2)
        z = self.transformer.fft_fit(self.X)  # shape: (n_features, 2)
        feature_indices = np.arange(coords.shape[0])

        self.mapping_df = pd.DataFrame({
            "Feature": feature_indices,
            "Row": coords[:, 0],
            "Col": coords[:, 1],
            "RealPart": z[:, 0],
            "ImagPart": z[:, 1]
        })
        return self.mapping_df

    def get_mapping(self):
        """
        Returns the mapping DataFrame. If not yet built, it will be built automatically.
        
        Returns:
            pd.DataFrame: Feature-to-pixel mapping table.
        """
        if self.mapping_df is None:
            return self.build_mapping()
        return self.mapping_df

    def save_mapping_to_csv(self, filename="feature_pixel_mapping.csv"):
        """
        Saves the feature-pixel-value mapping to a CSV file.

        Parameters:
            filename (str): Path to save the CSV file
        """
        df = self.get_mapping()
        df.to_csv(filename, index=False)
        print(f"Mapping saved to {filename}")


fft = FftTransformer(pixels=128)
fft.fit(Xtrain)

mapper = FeaturePixelMapper(fft, Xtrain)
df = mapper.get_mapping()

print(df.head())
mapper.save_mapping_to_csv("mapping.csv")
