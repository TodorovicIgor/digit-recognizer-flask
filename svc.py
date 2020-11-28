import tensorflow as tf
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import os
from joblib import load, dump


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

if os.path.exists('svc.joblib'):
    clf = load('svc.joblib')
else:
    clf = SVC()
    clf.fit(x_train, y_train)
    dump(clf, 'svc.joblib')

print('evaluating')
print(confusion_matrix(y_test, clf.predict(x_test)))
print('Accuracy:', accuracy_score(y_test, clf.predict(x_test)))
print('F1:', f1_score(y_test, clf.predict(x_test), average='micro'))
