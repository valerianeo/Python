import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
print(df.head())

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
print(df.head())

actual = df.actual_label.values
model_RF = df.model_RF.values
model_LR = df.model_LR.values
predicted_RF = df.predicted_RF.values
predicted_LR = df.predicted_LR.values
score = accuracy_score(actual, predicted_RF)

conf_matr = confusion_matrix(df.actual_label.values, df.predicted_RF.values)
print("confusion_matrix:\n", conf_matr)


def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))


def find_conf_matrix_values(y_true, y_pred):
    # calculate TP, FN, FP, TN
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


# Confusion
def Makovska_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


print("Makovska_confusion_matrix:\n",
      Makovska_confusion_matrix(actual, predicted_RF))

assert np.array_equal(Makovska_confusion_matrix(
    df.actual_label.values, df.predicted_RF.values),
    confusion_matrix(df.actual_label.values, df.predicted_RF.values)), \
    'my_confusion_matrix() is not correct for RF'

assert np.array_equal(Makovska_confusion_matrix(
    df.actual_label.values, df.predicted_LR.values),
    confusion_matrix(df.actual_label.values, df.predicted_LR.values) ), \
    'my_confusion_matrix() is not correct for LR'

# Accuracy
print("Accuracy score on RF:", score)


def Makovska_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)


assert Makovska_accuracy_score(actual, predicted_RF) == accuracy_score(actual, predicted_RF), \
    'my accuracy_score failed RF'

assert Makovska_accuracy_score(actual, predicted_LR) == accuracy_score(actual, predicted_LR), \
    'my accuracy_score failed LR'

print("My accuracy score on RF:", Makovska_accuracy_score(actual, predicted_RF))
print("My accuracy score on LR:", Makovska_accuracy_score(actual, predicted_LR))

# Recall
print('Recall score on RF:', recall_score(actual, predicted_RF))


def Makovska_recal_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


assert Makovska_recal_score(actual, predicted_RF) == recall_score(actual, predicted_RF), \
    'my recal_score fails on RF'

assert Makovska_recal_score(actual, predicted_LR) == recall_score(actual, predicted_LR), \
    'my recal_score fails on LR'

print("My recall score on RF:", Makovska_recal_score(actual, predicted_RF))
print("My recall score on LR:", Makovska_recal_score(actual, predicted_LR))

# Precision
print("Precision score on RF:", precision_score(actual, predicted_RF))


def Makovska_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


assert Makovska_precision_score(actual, predicted_RF) == precision_score(actual, predicted_RF), \
    'my precision_score fails on RF'

assert Makovska_precision_score(actual, predicted_LR) == precision_score(actual, predicted_LR), \
    'my precision_score fails on LR'

print("My precision score on RF:", Makovska_precision_score(actual, predicted_RF))
print("My precision score on LR:", Makovska_precision_score(actual, predicted_LR))

# F1 score
print("F1 score on RF", f1_score(actual, predicted_RF))


def Makovska_f1_score(y_true, y_pred):
    precision = Makovska_precision_score(y_true, y_pred)
    recall = Makovska_recal_score(y_true, y_pred)
    return (2 * (precision * recall)) / (precision + recall)


assert Makovska_f1_score(actual, predicted_RF) == f1_score(actual, predicted_RF), \
    'my f1_score fails on RF'

assert Makovska_f1_score(actual, predicted_LR) == f1_score(actual, predicted_LR), \
    'my f1_score fails on LR'

print("My F1 score score on RF:", Makovska_f1_score(actual, predicted_RF))
print("My F1 score score on LR:", Makovska_f1_score(actual, predicted_LR))
print()


def test_thresholds(threshold: float = .5):
    print(f"Scores with threshold = {threshold}")
    predicted = (df.model_RF >= threshold).astype('int')

    print("Accuracy RF:", Makovska_accuracy_score(actual, predicted))
    print("Precision RF:", Makovska_precision_score(actual, predicted))
    print("Recall RF:", Makovska_recal_score(actual, predicted))
    print("F1 RF:", Makovska_f1_score(actual, predicted))
    print()


test_thresholds()
test_thresholds(.25)
test_thresholds(.6)
test_thresholds(.20)

# ROC
# Curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(actual, model_RF)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(actual, model_LR)

# AUC
auc_RF = roc_auc_score(actual, model_RF)
auc_LR = roc_auc_score(actual, model_LR)

print("AUC RF:", auc_RF)
print("AUC LR:", auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label=f'AUC RF: {auc_RF}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'AUC LR: {auc_LR}')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')

plt.legend()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()