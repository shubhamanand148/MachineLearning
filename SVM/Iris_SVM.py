import sklearn.svm as svm, sklearn.datasets as datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score

iris_data = datasets.load_iris()
x = iris_data.data
y = iris_data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

classifier = svm.SVC()

#Parameter grid for Hyperparameter tuning.
param_grid = {'C': [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 70, 100, 150, 200],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(classifier, param_grid, refit=True)
trained_model = grid.fit(x_train, y_train)
best_param = grid.best_estimator_
predictions = grid.predict(x_test)

print(best_param)
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))