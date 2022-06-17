import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


data = pd.read_excel('ama2022-pistachio\Pistachio_16_Features_Dataset\Pistachio_16_Features_Dataset.xls')

X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

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

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

precision, accuracy, recall, f1, cm = metrics(y, y_pred)