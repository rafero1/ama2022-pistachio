import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, plot_confusion_matrix, roc_curve, auc, roc_auc_score

data = pd.read_excel('ama2022-pistachio\Pistachio_16_Features_Dataset\Pistachio_16_Features_Dataset.xls')

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print(metrics.classification_report(y, log_reg.predict(X)))

pred_prob1 = log_reg.predict_proba(X_test)
fpr1, tpr1, thresh1 = metrics.roc_curve(y_test, pred_prob1[:,1], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = metrics.roc_curve(y_test, random_probs, pos_label=1)
auc_score1 = metrics.roc_auc_score(y_test, pred_prob1[:,1])

print(auc_score1)

#metrics.ConfusionMatrixDisplay.from_estimator(log_reg, X_test, y_test)