import numpy as np
import pandas as pd

def classes_counts(y):
    classes, counts = np.unique(y, return_counts=True)
    for i in range(len(classes)):
        print(classes[i], "=", counts[i])
    
    return classes, counts

def metrics(y, y_pred):

    y = y.astype('bool')
    y_pred = y_pred.astype('bool')

    TP = sum(y & y_pred) 
    TN = sum(~y & ~y_pred)
    FP = sum(~y & y_pred)
    FN = sum(y & ~y_pred)

    precision = TP / (TP + FP)
    accuracy = (TP + TN)/(TP + FP + TN + FN)
    recall = TP /(TP+FN)
    f1 = 2*(precision * recall)/(precision + recall)
    cm = np.array([[ TP, TN ], [ FP, FN ]])

    if np.isnan(precision):
        precision = 0
    if np.isnan(accuracy):
        accuracy = 0
    if np.isnan(recall):
        recall = 0
    if np.isnan(f1):
        f1 = 0

    return precision, accuracy, recall, f1, cm

def confusion_matrix(cm):
    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()
