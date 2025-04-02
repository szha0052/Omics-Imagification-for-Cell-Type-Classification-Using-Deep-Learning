class CoordInspector:
    def __init__(self, transformer, colors):
        """
        Initialize the inspector with a fitted FftTransformer instance.

        Parameters:
            transformer (FftTransformer): A fitted transformer that contains pixel coordinates.
            colors: A DataFrame that records all feature classifications
        """
        self.transformer = transformer
        self.colors = colors

    def get_coords(self):
        """
        Retrieve pixel coordinates of features from the transformer.

        Returns:
            np.ndarray: An array of (row, col) coordinates for each feature.
        """
        coords_exist = hasattr(self.transformer, '_coords')
        if not coords_exist:
            raise ValueError("FftTransformer not fitted")
        
        coords = self.transformer._coords.copy()
        return coords

    def show_coords(self):
        """
        Visualize the (row, col) coordinates of features on a 2D scatter plot.
        """
        coords = self.get_coords()
        x_coords = coords[:, 1]  # columns
        y_coords = coords[:, 0]  # rows

        plt.figure(figsize=(6, 6))
        plt.scatter(x_coords, y_coords, s=10, c='blue')
        plt.title("Feature Pixel Coordinates")
        plt.gca().invert_yaxis()  # To match image layout
        plt.grid(True)
        plt.xlabel("Column (X-axis)")
        plt.ylabel("Row (Y-axis)")
        plt.axis('equal')
        plt.show()

    def get_coords_with_labels(self):
        """
        A Dataframe Like Feature1, Feature2, --- (location)
                    cell1 [1,2], [11,5],....   
                    cell2 ....
        A Dataframe Like Feature1, Feature2, ---  (colors)
                    cell1 3, 5,....   
                    cell2 ....

        Combine the coordinate and color DataFrames into one,
        where each element is a (row, col, label) tuple.

        Returns:
            pd.DataFrame: DataFrame of the same shape, with each cell being a (row, col, label) tuple.
        """

        coords_df = self.get_coords()  # assumed to be DataFrame with [row, col] entries
        colors_df = self.colors

        if coords_df.shape != colors_df.shape:
            raise ValueError("Shape mismatch between coordinates and color data.")

        combined_df = coords_df.copy()
        for row in combined_df.index:
            for col in combined_df.columns:
                coord = coords_df.at[row, col]
                label = colors_df.at[row, col]
                combined_df.at[row, col] = (coord[0], coord[1], label)

        return combined_df
    
    def get_label_colormap(self, seed=42):
        """
        Generate a color map for each unique label in self.colors.

        Parameters:
            seed (int): Random seed for reproducibility.

        Returns:
            dict: A mapping from label → (R, G, B) color tuple.
        """
        label_matrix = self.colors.values
        unique_labels = np.unique(label_matrix)
        np.random.seed(seed)

        color_map = {
            label: tuple(np.random.randint(0, 256, size=3)) for label in unique_labels
        }

        return color_map
    
    def draw_rgb(self, combined_df_row, color_map, shape=(15, 15), seed=42):
        """
        Draw an RGB image from combined_df with (row, col, label) info.

        Parameters:
            combined_df (pd.DataFrame): Must contain 'row', 'col', 'label' columns.
            shape (tuple): The (height, width) of the image.
            seed (int): Random seed for consistent color generation.

        Returns:
            np.ndarray: A (3, H, W) RGB image.
        """
        H, W = shape
        rgb_image = np.zeros((3, H, W), dtype=np.uint8)


        for _, row in combined_df_row.iterrows():
            r = int(row['row'])
            c = int(row['col'])
            label = row['label']
            if 0 <= r < H and 0 <= c < W:
                color = color_map.get(label, (255, 255, 255))  # Default White
                for i in range(3):
                    rgb_image[i, r, c] = color[i]

        return rgb_image
    

    def draw_rgb_fit(self, shape=(15, 15), seed=42):
        """
        Draw RGB images for each cell using the feature coordinate and label info.

        Returns:
            dict: A dictionary of {cell_N: rgb_image (3 x H x W)}
        """
        combined_df = self.get_coords_with_labels()  # expected to be dict-like {cell: DataFrame}
        color_map = self.get_label_colormap(seed=seed)

        rgb_images = {}

        for idx, (cell_name, combined_df_row) in enumerate(combined_df.items(), start=1):
            rgb = self.draw_rgb(combined_df_row, color_map=color_map, shape=shape, seed=seed)
            rgb_images[f"cell_{idx}"] = rgb

        return rgb_images


    def save_rgb_images(self, save_dir='rgb_outputs', shape=(15, 15), seed=42):
        """
        Save RGB images for each cell as PNG files.

        Parameters:
            save_dir (str): Directory to save images.
            shape (tuple): Shape of each RGB image.
            seed (int): Seed for consistent colors.
        """
        os.makedirs(save_dir, exist_ok=True)
        rgb_images = self.draw_rgb_fit(shape=shape, seed=seed)

        for cell_name, rgb_array in rgb_images.items():
            # Convert from (3, H, W) to (H, W, 3)
            rgb_img = np.transpose(rgb_array, (1, 2, 0))
            img = Image.fromarray(rgb_img)
            img.save(os.path.join(save_dir, f"{cell_name}.png"))


            
'''# Assuming Xtrain exists and FftTransformer has been fit
fft = FftTransformer(pixels=128)
fft.fit(Xtrain)

inspector = CoordInspector(fft)

# Get and print coordinates
coords = inspector.get_coords()
print("Coordinates shape:", coords.shape)
print("First 5 coordinates (row, col):\n", coords[:5])

# Plot them
inspector.show_coords()
'''
