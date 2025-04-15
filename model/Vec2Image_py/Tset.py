import numpy as np
import pandas as pd
import random
import torch
from get_matrix import get_matrix
from trainer import get_net_trainer
from getNETtest import NetTester
from getSMOTE import SmoteGenerator
from getPrioritizeGene import getPrioritizeGene
from Cart2Pixel import Cart2Pixel
from ConvPixel import ConvPixel
from sklearn.preprocessing import LabelBinarizer



print("\nLoading full gene expression dataset...")
df = pd.read_csv("/Users/wong/Downloads/deng-reads-RawCount-modefied.csv")
X_full = df.iloc[:, 1:].to_numpy(dtype=np.float32)
labels_full = [col.split('.')[0] for col in df.columns[1:]]
label_map = {label: idx for idx, label in enumerate(set(labels_full))}
y_full = np.array([label_map[l] for l in labels_full])

# 构造完整数据集
dset = {
    'Xtrain': X_full,
    'train_labels': y_full.astype(int),
}

# 设置参数
Parm = {
    'Method': 'tSNE',
    'Max_Px_Size': 30,
    'MPS_Fix': 1,
    'ValidRatio': 0.2,
    'Seed': 42
}



# 正式映射坐标 + 划分验证集
print("\nRunning get_matrix...")
get_matrix(dset, Parm)

# SMOTE 增强
print("\nApplying SMOTE...")
smote = SmoteGenerator(dset['Xtrain'].T, dset['train_labels'])
X_aug, y_aug = smote.fit_resample()
X_aug = X_aug.T

# ConvPixel 测试
print("\nTesting ConvPixel...")
sample_vec = dset['Xtrain'][:, 0]
image = ConvPixel(sample_vec, dset['xp'], dset['yp'], dset['A'], dset['B'], dset['Base'])
print("Image shape:", image.shape)

# 标签 one-hot 编码

lb = LabelBinarizer()
y_train_oh = lb.fit_transform(y_aug)
y_val_oh = lb.transform(dset['Validation_labels'])
dset['label_encoder'] = lb

print("\nConverting augmented data to images...")
X_train_imgs = np.zeros((X_aug.shape[1], 1, dset['A'], dset['B']), dtype=np.float32)
for i in range(X_aug.shape[1]):
    X_train_imgs[i, 0, :, :] = ConvPixel(X_aug[:, i], dset['xp'], dset['yp'], dset['A'], dset['B'], dset['Base'], 0)

X_val_imgs = dset['XValidation'].transpose(3, 2, 0, 1)

# 模型训练
print("\nTraining model...")
model, train_loader, val_loader, criterion, optimizer, device = get_net_trainer(
    X_train_imgs, y_train_oh,
    X_val_imgs, y_val_oh
)
# 构造 Out 字典供推理
Out = {
    'model': {'net': model},
    'Norm': 1,
    'xp': dset['xp'],
    'yp': dset['yp'],
    'A': dset['A'],
    'B': dset['B'],
    'Base': dset['Base'],
    'Min': np.min(dset['XValidation'], axis=1, keepdims=True),
    'Max': np.max(dset['XValidation'], axis=1, keepdims=True),
}

# 模型测试
n_val = dset['XValidation'].shape[1]
dset['Xtest'] = dset['XValidation']
dset['test_labels'] = dset['Validation_labels'][:n_val]  
tester = NetTester(dset, Out, device='cpu')
acc, XTest_tensor, Y_pred = tester.run_test()
print(f"\nTest accuracy: {acc:.2%}")

# 基因优先级评估
#gene_rank = getPrioritizeGene(dset, Out, k=5)
#print("Gene ranking shape:", gene_rank.shape)
