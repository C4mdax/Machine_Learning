'''
Luis Angel Moreno Delgado.
August, 2024.
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Loading dataset
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

#Training
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train, y_train)

#Predicting and evaluating
y_predicted = knn_model.predict(x_test)
print("Precision: ", accuracy_score(y_test, y_predicted))
