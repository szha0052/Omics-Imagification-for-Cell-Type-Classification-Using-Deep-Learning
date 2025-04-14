
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from collections import defaultdict
from Cart2Pixel import Cart2Pixel
from ConvPixel import ConvPixel

def get_matrix(dset, Parm=None):
    if Parm is None:
        Parm = {
            'Method': 'tSNE',
            'Max_Px_Size': 120,
            'MPS_Fix': 1,
            'ValidRatio': 0.1,
            'Seed': 108
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
    YValidation = YTrain[idx]
    YTrain = np.delete(YTrain, idx)

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

    Out['ValidationRawdata'] = dset['XValidation']

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

    return Out
