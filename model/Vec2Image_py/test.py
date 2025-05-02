import numpy as np
import pandas as pd
import random
import torch
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed(42)

from get_matrix import get_matrix
from trainer import get_net_trainer
from getNETtest import NetTester
from getSMOTE import SmoteGenerator
from getPrioritizeGene import getPrioritizeGene
from Cart2Pixel import Cart2Pixel
from ConvPixel import ConvPixel
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder



print("\nLoading full gene expression dataset...")
df = pd.read_csv("deng-reads-RawCount-modefied.csv")
X_full = df.iloc[:, 1:].to_numpy(dtype=np.float32)
labels_full = [col.split('.')[0] for col in df.columns[1:]]
label_map = {label: idx for idx, label in enumerate(set(labels_full))}
y_full = np.array([label_map[l] for l in labels_full])


dset = {
    'Xtrain': X_full,
    'train_labels': y_full.astype(int),
}

Parm = {
    'Method': 'tSNE',
    'Max_Px_Size': 30,
    'MPS_Fix': 1,
    'ValidRatio': 0.2,
    'Seed': 42,
    'NORM': 1
}


print("\nRunning get_matrix...")
Out = get_matrix(dset, Parm)

print("\nApplying SMOTE...")
smote = SmoteGenerator(dset['Xtrain'].T, dset['train_labels'], seed=42)
X_aug, y_aug = smote.fit_resample()
X_aug = X_aug.T

print("\nTesting ConvPixel...")
sample_vec = dset['Xtrain'][:, 0]
image = ConvPixel(sample_vec, dset['xp'], dset['yp'], dset['A'], dset['B'], dset['Base'])
print("Image shape:", image.shape)


# lb = LabelBinarizer()
# y_train_oh = lb.fit_transform(y_aug)
# y_val_oh = lb.transform(dset['Validation_labels'])
# dset['label_encoder'] = lb

le = LabelEncoder()
y_aug_int = le.fit_transform(y_aug)
y_val_int = le.transform(dset['Validation_labels'])

# One-hot 编码
y_train_oh = np.eye(len(le.classes_))[y_aug_int]
y_val_oh = np.eye(len(le.classes_))[y_val_int]
dset['label_encoder'] = le

print("\nConverting augmented data to images...")
X_train_imgs = np.zeros((X_aug.shape[1], 1, dset['A'], dset['B']), dtype=np.float32)
for i in range(X_aug.shape[1]):
    fvec = X_aug[:, i][Out['feature_order']] 
    X_train_imgs[i, 0, :, :] = ConvPixel(fvec, dset['xp'], dset['yp'], dset['A'], dset['B'], dset['Base'], 0)

X_val_imgs = dset['XValidation'].transpose(3, 2, 0, 1)

print("\nTraining model...")
model, train_loader, val_loader, criterion, optimizer, device = get_net_trainer(
    X_train_imgs, y_train_oh,
    X_val_imgs, y_val_oh
)

Out.update({
    'model': {'net': model}
})

# n_val = dset['XValidation'].shape[3]
# dset['Xtest'] = Out['ValidationRawdata'][:, :n_val]
# dset['test_labels'] = dset['Validation_labels'][:n_val]  

X_val_raw = Out['ValidationRawdata']  
X_val_raw = np.nan_to_num(X_val_raw) 

X_val_raw_T = X_val_raw.T  
X_val_pca = Out['pca'].transform(X_val_raw_T)  
dset['Xtest'] = X_val_pca.T 

n_val = dset['Xtest'].shape[1]
dset['test_labels'] = Out['ValidationLabelsOrdered'][:n_val]

tester = NetTester(dset, Out, device='cpu')
acc, XTest_tensor, Y_pred = tester.run_test()
print(f"\nTest accuracy: {acc:.2%}")


print("XP hash:", hash(tuple(dset['xp'])))
print("YP hash:", hash(tuple(dset['yp'])))
print("Test Out XP hash:", hash(tuple(Out['xp'])))

#gene_rank = getPrioritizeGene(dset, Out, k=5)
#print("Gene ranking shape:", gene_rank.shape)
print("First few test labels:", dset['test_labels'][:5])
print("Xtest shape:", dset['Xtest'].shape)
