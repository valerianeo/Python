import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score


input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 100

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints or len(X) >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line[:-1].split(', ')
        if data[-1] == "<=50K" and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == ">50K" and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

classifier = OneVsOneClassifier(SVC(kernel='sigmoid'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
precision = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=3)
recall = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=3)
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")
print("Precision: " + str(round(100*precision.mean(), 2)) + "%")
print("Recall: " + str(round(100*recall.mean(), 2)) + "%")
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
def predict(input_data):
    input_data_encoded = [-1] * len(input_data)
    count = 0
    for index, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[index] = int(input_data[index])
        else:
            input_data_encoded[index] = label_encoder[count].transform([input_data[index]])[0]
            count += 1

    input_data_encoded = np.array(input_data_encoded)
    predicted_class = classifier.predict([input_data_encoded])
    print("Input data:", input_data)
    print("Predicted class:", label_encoder[-1].inverse_transform(predicted_class)[0])


input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Married-civ-spouse', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
'0', '0', '40', 'United-States']
predict(input_data)
