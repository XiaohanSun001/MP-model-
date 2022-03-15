
"""
Created on Fri Mar  5 00:34:25 2021

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import sys, os
import warnings
warnings.filterwarnings('ignore')


def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y

def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data


def prediction(data):
    model1 = joblib.load("model1.pkl")
    y_pred_m1 = model1.predict(data)
    index1 = []
    index2 = []
    a = 0
    for i in y_pred_m1:
        # if aneuploid
        if i != 0:
            index1.append(a)
            a = a + 1
        # if euploid
        else:
            index2.append(a)
            a = a + 1

    y_pred = ['F'] * len(y_pred_m1)
    # print(y_pred)
    for i in index2:
        y_pred[i] = 0

    if len(index1) != 0:
        selet1 = test_data.loc[index1]
        data1, label1 = dataSplit(selet1)
        model2 = joblib.load("model2.pkl")
        y_pred_m2 = model2.predict(data1)
        y_pred_m2 = y_pred_m2 + 1
        n = 0
        for i in index1:
            y_pred[i] = y_pred_m2[n]
            n = n + 1
    pred_label = y_pred
    return pred_label

if __name__ == "__main__":

    file_name='' +sys.argv[1]+ ''
    outdir = "../Results/"
    outname=file_name[:-4]

    test_data = pd.read_csv(file_name)
    prid_data, prid_label = dataSplit(test_data)
    pred_label = prediction(prid_data)
    accuracy = accuracy_score(prid_label, pred_label)
    print('MP1_accuracy')
    print(accuracy)

    prid_label=np.array(prid_label).reshape(-1,1)
    pred_label=np.array(pred_label).reshape(-1,1)  
    result = pd.DataFrame(np.hstack((prid_label,pred_label)))
    result.columns = ['Prid_label','Pred_label']
    result.to_csv(outdir+outname+'_MP1_result.csv',index=False)
    