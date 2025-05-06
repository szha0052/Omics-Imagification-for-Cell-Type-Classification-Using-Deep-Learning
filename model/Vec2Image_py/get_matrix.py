
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from collections import defaultdict
from .Cart2Pixel import Cart2Pixel
from .ConvPixel import ConvPixel
from sklearn.decomposition import PCA


def get_matrix(dset, Parm=None):
    if Parm is None:
        Parm = {
            'Method': 'tSNE',
            'Max_Px_Size': 120,
            'MPS_Fix': 1,
            'ValidRatio': 0.1,
            'Seed': 108,
            'NORM': 1 
        }

    YTrain = np.array(dset['train_labels'])
    TrueLabel = YTrain.copy()
    classes = np.unique(YTrain)

    np.random.seed(Parm['Seed'])
    idx = []
    for cls in classes:
        indices = np.where(YTrain == cls)[0]
        selected = np.random.choice(indices, int(len(indices) * Parm['ValidRatio']), replace=False)
        idx.extend(selected)
    
    idx = np.array(idx)
    dset['XValidation'] = dset['Xtrain'][:, idx]
    dset['Xtrain'] = np.delete(dset['Xtrain'], idx, axis=1)
    dset['val_idx'] = idx
    YValidation = YTrain[idx]
    YTrain = np.delete(YTrain, idx)
    dset['train_labels'] = YTrain
    Out = {}

    if Parm.get('NORM') == 1:
        print("\nNORM-1")
        Out['Norm'] = 1
        Out['Max'] = dset['Xtrain'].max(axis=1, keepdims=True)
        Out['Min'] = dset['Xtrain'].min(axis=1, keepdims=True)
        dset['Xtrain'] = (dset['Xtrain'] - Out['Min']) / (Out['Max'] - Out['Min'])
        dset['XValidation'] = (dset['XValidation'] - Out['Min']) / (Out['Max'] - Out['Min'])
        dset['Xtrain'] = np.nan_to_num(dset['Xtrain'])
        dset['XValidation'] = np.clip(np.nan_to_num(dset['XValidation']), 0, 1)

        Out['ValidationRawdata'] = dset['XValidation'].copy()
        Out['ValidationLabelsOrdered'] = YValidation.copy()
        print("Applying PCA dimensionality reduction...")

        X_train_T = dset['Xtrain'].T  
        pca = PCA(n_components=100) 
        X_train_reduced = pca.fit_transform(X_train_T) 
        dset['Xtrain'] = X_train_reduced.T  

        X_val_T = dset['XValidation'].T
        X_val_reduced = pca.transform(X_val_T)
        dset['XValidation'] = X_val_reduced.T

        Out['pca'] = pca 

    elif Parm.get('NORM') == 2:
        print("\nNORM-2")
        Out['Norm'] = 2
        Out['Min'] = dset['Xtrain'].min(axis=1, keepdims=True)
        dset['Xtrain'] = np.log(dset['Xtrain'] + np.abs(Out['Min']) + 1)
        indV = dset['XValidation'] < Out['Min']
        for j in range(dset['XValidation'].shape[0]):
            dset['XValidation'][j, indV[j, :]] = Out['Min'][j]
        dset['XValidation'] = np.log(dset['XValidation'] + np.abs(Out['Min']) + 1)
        Out['Max'] = dset['Xtrain'].max()
        dset['Xtrain'] /= Out['Max']
        dset['XValidation'] = np.clip(dset['XValidation'] / Out['Max'], 0, 1)

    Q = {
        'data': dset['Xtrain'],
        'Method': Parm['Method'],
        'Max_Px_Size': Parm['Max_Px_Size']
    }

    # Out['ValidationRawdata'] = dset['XValidation']

    if Parm['MPS_Fix'] == 1:
        Out['M'], Out['xp'], Out['yp'], Out['A'], Out['B'], Out['Base'] = Cart2Pixel(Q, Parm['Max_Px_Size'], Parm['Max_Px_Size'])
    else:
        Out['M'], Out['xp'], Out['yp'], Out['A'], Out['B'], Out['Base'] = Cart2Pixel(Q)

    print(f"\n Pixels: {Out['A']} x {Out['B']}")

    XValidation = np.zeros((Out['A'], Out['B'], 1, len(YValidation)))
    for j in range(len(YValidation)):
        XValidation[:, :, 0, j] = ConvPixel(dset['XValidation'][:, j], Out['xp'], Out['yp'], Out['A'], Out['B'], Out['Base'], 0)

    XTrain = np.zeros((Out['A'], Out['B'], 1, len(YTrain)))
    for j in range(len(YTrain)):
        XTrain[:, :, 0, j] = Out['M'][j]

    Out['Xtrain'] = XTrain
    Out['XValidation'] = XValidation
    Out['train_labels'] = YTrain
    Out['Validation_labels'] = YValidation
    Out['ValidationLabelsOrdered'] = YValidation.copy() 
    Out['feature_order'] = np.arange(dset['Xtrain'].shape[0]) 

    dset['xp'] = Out['xp']
    dset['yp'] = Out['yp']
    dset['A'] = Out['A']
    dset['B'] = Out['B']
    dset['Base'] = Out['Base']
    dset['XValidation'] = Out['XValidation']
    dset['Validation_labels'] = Out['Validation_labels']
    dset['Min'] = Out['Min']
    dset['Max'] = Out['Max']

    return Out
