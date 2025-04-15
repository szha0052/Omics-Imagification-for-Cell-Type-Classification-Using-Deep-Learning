import torch
import numpy as np
from ConvPixel import ConvPixel
from sklearn.preprocessing import LabelBinarizer

class NetTester:
    """
    A class for inference testing using PyTorch, corresponding to MATLAB's getNETtest.
    """
    def __init__(self, dset, Out, device='cpu'):
        """
        Parameters
        ----------
        dset : dict
            Must include:
              - 'test_labels': Test set labels, shape (n_samples,)
              - 'Xtest': Test set features, shape (n_features, n_samples) (MATLAB style)
        Out : dict
            Must include:
              - 'Norm': int, 1 or 2, to select normalization method
              - 'Min', 'Max': Arrays or vectors matching the rows of Xtest
              - 'xp', 'yp', 'A', 'B', 'Base': Parameters used in ConvPixel
              - 'model': Dictionary or object, where 'net' is a PyTorch model
        device : str
            Device to run inference on, default is 'cpu'. Use 'cuda' for GPU (if supported).
        """
        self.dset = dset
        self.Out = Out
        self.device = device
        
        # Extract commonly used fields
        self.xp = Out['xp']
        self.yp = Out['yp']
        self.A = Out['A']
        self.B = Out['B']
        self.Base = Out['Base']

        # Load the original test set
        self.Xtest = np.array(dset['Xtest'], dtype=float)  # (n_features, n_samples)
        lb = dset.get('label_encoder', None)

        if lb:
            self.Ytest = lb.transform(dset['test_labels'])     # 得到 one-hot 编码
            self.Ytest = np.argmax(self.Ytest, axis=1)         # 取最大索引变成整数 label
        else:
         raise ValueError("Missing LabelBinarizer in dset['label_encoder']")

    def run_test(self):
        """
        Perform inference testing and return accuracy, processed test set, and predicted labels.

        Returns
        -------
        accuracy : float
            Accuracy in the range [0, 1].
        XTest_tensor : torch.Tensor
            Shape (n_samples, 1, xp, yp), preprocessed and converted to image-like test data.
        Y_pred : torch.Tensor
            Predicted labels as integers, shape (n_samples,).
        """
        # 1) Normalization
        # Norm=1 or 2
        if self.Out['Norm'] == 1:
            # Norm-1: (X - Min)/(Max - Min), then clip to [0,1]
            print("\nUsing Norm-1 ...")
            # broadcast: (n_features,1)
            denom = (self.Out['Max'] - self.Out['Min'])
            denom[denom == 0] = 1e-12  # Avoid division by zero
            self.Xtest = (self.Xtest - self.Out['Min'][:, None]) / denom[:, None]
            # Replace NaN -> 0
            self.Xtest = np.nan_to_num(self.Xtest, nan=0.0)
            self.Xtest = np.clip(self.Xtest, 0, 1)
        else:
            # Norm-2: 1) Clamp values below Min to Min; 2) Take log(x + abs(Min)+1);
            #         3) Divide by Max; 4) Clip to [0,1]
            print("\nUsing Norm-2 ...")
            for j in range(self.Xtest.shape[0]):
                row_min = self.Out['Min'][j]
                row_max = self.Out['Max'][j] if self.Out['Max'][j] != 0 else 1e-12
                mask = self.Xtest[j, :] < row_min
                # Clamp
                self.Xtest[j, mask] = row_min
                # Log
                offset = abs(row_min) + 1
                self.Xtest[j, :] = np.log(self.Xtest[j, :] + offset)
                # Scale
                self.Xtest[j, :] /= row_max
            self.Xtest = np.clip(self.Xtest, 0, 1)

        # 2) Convert each feature column to an image of shape (xp, yp),
        #    then reshape to (1, xp, yp), and finally stack into (n_samples, 1, xp, yp)
        n_samples = self.Xtest.shape[1]
        XTest_np = np.zeros((n_samples, 1, self.A, self.B), dtype=np.float32)
        for j in range(n_samples):
            # MATLAB: ConvPixel(dset.Xtest(:, j), ...)
            FVec = self.Xtest[:, j].flatten() 
            image_2d = ConvPixel(FVec, self.xp, self.yp, self.A, self.B, self.Base)
            XTest_np[j, 0, :, :] = image_2d

        # Convert to torch.Tensor
        XTest_tensor = torch.from_numpy(XTest_np).float().to(self.device)
        YTest_tensor = torch.from_numpy(self.Ytest).long().to(self.device)

        # 3) Model inference
        model = self.Out['model']['net'].to(self.device)
        model.eval()
        with torch.no_grad():
            logits = model(XTest_tensor)  # (n_samples, num_classes)
            Y_pred = torch.argmax(logits, dim=1)  # (n_samples,)

        # 4) Compute accuracy
        correct = (Y_pred == YTest_tensor).sum().item()
        accuracy = correct / n_samples

        return accuracy, XTest_tensor, Y_pred
   
