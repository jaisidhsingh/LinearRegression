import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# TRAINING AND SAVING MOST ACCURATE MODEL
# best = 0
# for n in range(40):
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#     linear.fit(X_train, y_train)
#     accuracy = linear.score(X_test, y_test)
#     # print(accuracy)
#
#     if accuracy > best:
#         best = accuracy
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)
#

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Slope : ", linear.coef_)
print("Intercept : ", linear.intercept_)
print(linear.score(X_test, y_test))
predictions = linear.predict(X_test)
for x in range(len(predictions)):
    print(round(predictions[x]), X_test[x], y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()
