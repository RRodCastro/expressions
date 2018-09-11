# SHAPE:file?,  re, le, lw, lh, im, sx, truth
from numpy import array, shape
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from time import sleep

data = open("transformH.csv", "r")

frames_data = {}
frames_data['features_names'] = ['re', 'le', 'lw', 'lh', 'im', 'sx', 'truth']
frames_data['target_names'] = ['Lie', 'Truth']


rows = len(frames_data['features_names'])

frames_data['data'] = []

frames_data['target'] = []

for i in data:
    line = i.split(",")
    frames_data['data'].append(
        list(map(lambda value: round(float(value), 2), line[:rows-1])))
    frames_data['target'].append(int(line[rows-1][0]))
frames_data["data"] = array(frames_data["data"])

# get train and test
X_train, X_test, y_train, y_test = train_test_split(
    frames_data["data"], frames_data["target"], test_size=0.3, random_state=109)
# svm classifier with # Linear Kernel
clf = svm.SVC(kernel='linear')

# Train the model
print("training....")
clf.fit(X_train, y_train)

# Predict the response
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
