
"""
Created on Fri Mar  5 00:34:25 2021

@author: Dell
"""
# There are four models,including MP2,RF,SVM, and XGB.

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import f1_score
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')



def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y
def performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    fl_micro = f1_score(y_true, y_pred, average='micro')
    fl_weighted = f1_score(y_true, y_pred, average='weighted')

    return accuracy, \
           precision_micro, precision_weighted, \
           recall_micro, recall_weighted, \
           fl_micro, fl_weighted

def MP2_prediction(prid_data,prid_label):
    model = joblib.load("MP2.pkl")
    pred_label = model.predict(prid_data)
    accuracy = accuracy_score(prid_label, pred_label)
    prid_label=np.array(prid_label).reshape(-1,1)
    pred_label=np.array(pred_label).reshape(-1,1)  
    result = pd.DataFrame(np.hstack((prid_label,pred_label)))
    result.columns = ['Prid_label','Pred_label']
    result.to_csv(outdir+outname+'_MP2_result.csv',index=False)
    return accuracy
    #accuracy, precision_micro, precision_weighted, recall_micro, recall_weighted,  fl_micro, fl_weighted = performance(prid_label, pred_label)
    #print('The evaluation of test set:')
    #print(accuracy, precision_micro, precision_weighted, recall_micro, recall_weighted, fl_micro, fl_weighted)

def RF_prediction(prid_data,prid_label):
    model = joblib.load("RF.pkl")
    pred_label = model.predict(prid_data)
    accuracy = accuracy_score(prid_label, pred_label)
    prid_label=np.array(prid_label).reshape(-1,1)
    pred_label=np.array(pred_label).reshape(-1,1)  
    result = pd.DataFrame(np.hstack((prid_label,pred_label)))
    result.columns = ['Prid_label','Pred_label']
    result.to_csv(outdir+outname+'_RF_result.csv',index=False)
    return accuracy

def SVM_prediction(prid_data,prid_label):
    model = joblib.load("SVM.pkl")
    pred_label = model.predict(prid_data)
    accuracy = accuracy_score(prid_label, pred_label)
    prid_label=np.array(prid_label).reshape(-1,1)
    pred_label=np.array(pred_label).reshape(-1,1)  
    result = pd.DataFrame(np.hstack((prid_label,pred_label)))
    result.columns = ['Prid_label','Pred_label']
    result.to_csv(outdir+outname+'_SVM_result.csv',index=False)
    return accuracy
    
def XGB_prediction(prid_data,prid_label):
    model = joblib.load("XGB.pkl")
    pred_label = model.predict(prid_data)
    accuracy = accuracy_score(prid_label, pred_label)
    prid_label=np.array(prid_label).reshape(-1,1)
    pred_label=np.array(pred_label).reshape(-1,1)  
    result = pd.DataFrame(np.hstack((prid_label,pred_label)))
    result.columns = ['Prid_label','Pred_label']
    result.to_csv(outdir+outname+'_XGB_result.csv',index=False)
    return accuracy

if __name__ == "__main__":
	  
    file_name='' +sys.argv[1]+ ''
    outdir = "../Results/"
    outname=file_name[:-4]
    test_data = pd.read_csv(file_name)
    prid_data,prid_label=dataSplit(test_data)
    MP2_accuracy= MP2_prediction(prid_data,prid_label)
    RF_accuracy= RF_prediction(prid_data,prid_label)
    SVM_accuracy= SVM_prediction(prid_data,prid_label)
    XGB_accuracy= XGB_prediction(prid_data,prid_label)
    print('MP2_accuracy       RF_accuracy        SVM_accuracy       XGB_accuracy')
    print(MP2_accuracy,RF_accuracy,SVM_accuracy,XGB_accuracy)