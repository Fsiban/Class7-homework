# Using KNeighbors
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics


# Loading data from sklearn lib datasets
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
columns_names = cancer.feature_names

#assign viriables to data and target
X, y = cancer.data, cancer.target

# Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

#train the model using fit() method
knn = neighbors.KNeighborsClassifier(n_neighbors =3)
knn.fit(X_train,y_train)

predicted = knn.predict(X_test)
expected = y_test

matches = (predicted == expected)

print()
print(metrics.classification_report(expected, predicted))
print()
print(metrics.confusion_matrix(expected, predicted))

